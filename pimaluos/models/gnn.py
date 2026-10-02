"""
Multi-relational graph attention encoder for parcels.

Architecture (one node type, R relation types):

    h1 = SemanticAttention_r( GAT_r(x) )           # relation-specific GAT, edge weights as edge_attr
    h1 = ELU(LayerNorm(h1 + Lin(x)))
    h2 = SemanticAttention_r( GAT_r(h1) )
    h2 = ELU(LayerNorm(h2 + h1))
    z  = LayerNorm(Lin(h2))                         # embedding (embed_dim)

Semantic attention follows HAN (Wang et al., 2019): a learned query scores the
mean-pooled, tanh-projected output of each relation; softmax weights combine
relations.

Pre-training is self-supervised masked feature reconstruction (GraphMAE-style,
Hou et al., 2022): a random subset of *training* nodes has its features
replaced by a learned mask token and the model reconstructs them. A held-out
set of validation nodes is never used as a reconstruction target during
training; the reported validation loss reconstructs masked validation nodes.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv

NODE = "parcel"


class SemanticAttention(nn.Module):
    def __init__(self, dim: int, att_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(dim, att_dim)
        self.q = nn.Parameter(torch.randn(att_dim) * 0.1)
        self.last_weights: Optional[torch.Tensor] = None

    def forward(self, zs: torch.Tensor) -> torch.Tensor:  # zs: [R, N, D]
        scores = torch.tanh(self.proj(zs)).mean(dim=1) @ self.q  # [R]
        w = torch.softmax(scores, dim=0)
        self.last_weights = w.detach()
        return (w[:, None, None] * zs).sum(0)


class RelationalGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, relations: List[Tuple], heads: int, dropout: float):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.relations = list(relations)
        self.convs = nn.ModuleDict({
            self.key(r): GATConv(in_dim, out_dim // heads, heads=heads, concat=True, dropout=dropout,
                                 edge_dim=1, add_self_loops=True, fill_value="mean")
            for r in self.relations
        })
        self.sem = SemanticAttention(out_dim)

    @staticmethod
    def key(rel: Tuple) -> str:
        return "__".join(rel)

    def forward(self, x: torch.Tensor, data: HeteroData) -> torch.Tensor:
        outs = []
        for r in self.relations:
            store = data[r]
            outs.append(self.convs[self.key(r)](x, store.edge_index, edge_attr=store.edge_attr))
        return self.sem(torch.stack(outs, 0))


class ParcelGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        relations: List[Tuple],
        hidden_channels: int = 256,
        embed_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.relations = list(relations)
        self.mask_token = nn.Parameter(torch.zeros(in_channels))
        self.dropout = dropout
        if self.relations:
            self.l1 = RelationalGATLayer(in_channels, hidden_channels, self.relations, heads, dropout)
            self.l2 = RelationalGATLayer(hidden_channels, hidden_channels, self.relations, heads, dropout)
        self.skip = nn.Linear(in_channels, hidden_channels)
        self.mlp2 = nn.Linear(hidden_channels, hidden_channels)  # used when there are no relations
        self.n1 = nn.LayerNorm(hidden_channels)
        self.n2 = nn.LayerNorm(hidden_channels)
        self.out = nn.Linear(hidden_channels, embed_dim)
        self.n3 = nn.LayerNorm(embed_dim)
        self.decoder = nn.Sequential(nn.Linear(embed_dim, hidden_channels), nn.ELU(),
                                     nn.Linear(hidden_channels, in_channels))

    def encode(self, data: HeteroData, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = data[NODE].x if x is None else x
        if self.relations:
            h = self.l1(x, data) + self.skip(x)
        else:
            h = self.skip(x)
        h = F.dropout(F.elu(self.n1(h)), self.dropout, self.training)
        h2 = self.l2(h, data) if self.relations else self.mlp2(h)
        h = F.dropout(F.elu(self.n2(h2 + h)), self.dropout, self.training)
        return self.n3(self.out(h))

    @torch.no_grad()
    def get_embeddings(self, data: HeteroData) -> torch.Tensor:
        self.eval()
        return self.encode(data)

    def relation_weights(self) -> Dict[str, List[float]]:
        out = {}
        for name in ["l1", "l2"]:
            layer = getattr(self, name, None)
            if layer is not None and layer.sem.last_weights is not None:
                out[name] = dict(zip([r[1] for r in self.relations], layer.sem.last_weights.tolist()))
        return out


def masked_reconstruction_loss(model: ParcelGNN, data: HeteroData, target_idx: torch.Tensor) -> torch.Tensor:
    x = data[NODE].x
    x_in = x.clone()
    x_in[target_idx] = model.mask_token
    z = model.encode(data, x_in)
    rec = model.decoder(z[target_idx])
    return F.mse_loss(rec, x[target_idx])


def pretrain_gnn(
    model: ParcelGNN,
    data: HeteroData,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    mask_rate: float = 0.15,
    val_frac: float = 0.10,
    seed: int = 0,
    patience: int = 50,
    log_every: int = 25,
    logger=None,
) -> Dict:
    """Masked-feature pre-training with a held-out validation node set.

    Returns a history dict with per-epoch train and validation losses; the model
    is left at the parameters with the lowest validation loss.
    """
    g = torch.Generator().manual_seed(seed)
    n = data[NODE].num_nodes
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(round(val_frac * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    n_mask = max(1, int(round(mask_rate * len(train_idx))))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    hist = {"train_loss": [], "val_loss": [], "best_epoch": 0, "n_train_nodes": int(len(train_idx)),
            "n_val_nodes": int(n_val)}
    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for ep in range(epochs):
        model.train()
        tgt = train_idx[torch.randperm(len(train_idx), generator=g)[:n_mask]]
        loss = masked_reconstruction_loss(model, data, tgt)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = masked_reconstruction_loss(model, data, val_idx).item()
        hist["train_loss"].append(float(loss.item()))
        hist["val_loss"].append(float(vl))
        if vl < best - 1e-6:
            best, best_state, bad, hist["best_epoch"] = vl, copy.deepcopy(model.state_dict()), 0, ep
        else:
            bad += 1
        if logger and (ep % log_every == 0 or ep == epochs - 1):
            logger.info("GNN epoch %d train %.4f val %.4f", ep, loss.item(), vl)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    hist["best_val_loss"] = float(best)
    # Reference: predicting the training-set feature mean for masked val nodes.
    x = data[NODE].x
    mean = x[train_idx].mean(0, keepdim=True)
    hist["mean_predictor_val_loss"] = float(F.mse_loss(mean.expand(n_val, -1), x[val_idx]).item())
    return hist

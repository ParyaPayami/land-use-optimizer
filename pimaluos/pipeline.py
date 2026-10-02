"""
High-level API tying the layers together (Sense -> Reason -> Verify).

>>> from pimaluos.core import SyntheticCityLoader
>>> from pimaluos.pipeline import UrbanOptSystem
>>> sysm = UrbanOptSystem(SyntheticCityLoader().load())
>>> _ = sysm.build_graph()
>>> _ = sysm.pretrain_gnn(epochs=5, seed=0)
>>> plan = sysm.optimise(iterations=2, seed=0)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from pimaluos.core.data_loader import ParcelDataset
from pimaluos.core.graph_builder import ALL_EDGE_TYPES, ParcelGraphBuilder
from pimaluos.models.agents import AGENT_TYPES, MARLTrainer, MultiAgentEnvironment, PPOConfig, UtilityWeights
from pimaluos.models.gnn import ParcelGNN, pretrain_gnn
from pimaluos.physics.capacity import CapacityModel, CapacityParams
from pimaluos.physics.verification import verify_and_repair

logger = logging.getLogger(__name__)


def standardise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    sd = a.std(0)
    sd[sd < 1e-8] = 1.0
    return (a - a.mean(0)) / sd


class UrbanOptSystem:
    def __init__(self, dataset: ParcelDataset, capacity_params: Optional[CapacityParams] = None,
                 graph_kwargs: Optional[Dict] = None, gnn_kwargs: Optional[Dict] = None):
        self.ds = dataset
        self.graph_kwargs = graph_kwargs or {}
        self.gnn_kwargs = gnn_kwargs or {}
        self.capacity = CapacityModel(dataset.gdf, capacity_params)
        self.graph = None
        self.graph_summary: Optional[Dict] = None
        self.gnn: Optional[ParcelGNN] = None

    # ----------------------------------------------------------------- Sense
    def build_graph(self, edge_types: Optional[List[str]] = None):
        b = ParcelGraphBuilder(self.ds.gdf, self.ds.features, edge_types=edge_types or ALL_EDGE_TYPES,
                               **self.graph_kwargs)
        graph = b.build()
        summary = b.summary()
        if edge_types is None:
            self.graph, self.graph_summary = graph, summary
        return graph, summary

    # ----------------------------------------------------------------- Reason
    def pretrain_gnn(self, epochs: int = 300, seed: int = 0, graph=None, relations=None, **kw) -> Dict:
        graph = graph if graph is not None else self.graph
        torch.manual_seed(seed)
        np.random.seed(seed)
        rel = list(graph.edge_types) if relations is None else relations
        model = ParcelGNN(graph["parcel"].x.shape[1], rel, **self.gnn_kwargs)
        hist = pretrain_gnn(model, graph, epochs=epochs, seed=seed, logger=logger, **kw)
        if graph is self.graph and relations is None:
            self.gnn = model
        hist["relation_weights"] = model.relation_weights()
        return {"model": model, "history": hist}

    def static_state(self, use_gnn: bool = True) -> np.ndarray:
        if use_gnn:
            if self.gnn is None:
                raise ValueError("Pre-train the GNN first (or use use_gnn=False).")
            return standardise(self.gnn.get_embeddings(self.graph).numpy())
        return standardise(self.ds.features.values)

    def make_env(self, use_gnn: bool = True, physics_weight: float = 1.0,
                 agent_types: Optional[List[str]] = None, horizon: int = 10, delta_far: float = 0.5,
                 voting_weights: Optional[Dict[str, float]] = None,
                 utility_weights: Optional[UtilityWeights] = None) -> MultiAgentEnvironment:
        return MultiAgentEnvironment(self.capacity, self.static_state(use_gnn), agent_types or AGENT_TYPES,
                                     delta_far=delta_far, horizon=horizon, physics_weight=physics_weight,
                                     voting_weights=voting_weights, utility_weights=utility_weights)

    def train_marl(self, env: MultiAgentEnvironment, iterations: int = 100, seed: int = 0,
                   ppo: Optional[PPOConfig] = None) -> MARLTrainer:
        tr = MARLTrainer(env, ppo, seed=seed)
        tr.train(iterations, logger=logger)
        return tr

    def optimise(self, iterations: int = 100, seed: int = 0, use_gnn: bool = True, **env_kw) -> np.ndarray:
        env = self.make_env(use_gnn=use_gnn, **env_kw)
        tr = self.train_marl(env, iterations, seed)
        return tr.final_plan()[0]

    # ----------------------------------------------------------------- Verify
    def verify(self, far: np.ndarray, **kw):
        return verify_and_repair(self.capacity, far, **kw)

    def evaluate(self, far: np.ndarray) -> Dict:
        return self.capacity.evaluate(far)["summary"]

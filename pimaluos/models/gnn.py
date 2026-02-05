"""
PIMALUOS GNN Models Module

Contains Graph Neural Network implementations for parcel-level predictions:
- MultiRelationalGNN: Multi-relational message passing layer
- HeterogeneousGAT: 3-layer hetero graph attention network
- ParcelGNN: Complete model with multi-task learning heads
- GraphSAGEParcelModel: Inductive learning alternative
- AttentionVisualizer: Attention weight extraction for interpretability
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv, 
    GATConv, 
    SAGEConv,
    Linear, 
    global_mean_pool, 
    global_max_pool
)
from torch_geometric.data import HeteroData


class MultiRelationalGNN(nn.Module):
    """
    Multi-relational message passing layer with separate GAT per edge type.
    
    Aggregates messages from different relation types using learned attention.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension  
        edge_types: List of edge type tuples (src, rel, dst)
        heads: Number of attention heads
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        edge_types: List[Tuple[str, str, str]], 
        heads: int = 4
    ):
        super().__init__()
        
        self.convs = nn.ModuleDict()
        self.edge_types = edge_types
        
        for edge_type in edge_types:
            src, rel, dst = edge_type
            key = f'{src}__{rel}__{dst}'
            self.convs[key] = GATConv(
                in_channels, 
                out_channels // heads,
                heads=heads,
                concat=True,
                dropout=0.2,
                add_self_loops=False
            )
        
        # Aggregation across relations
        self.relation_attn = nn.Linear(out_channels, 1)
        
        # Store attention weights for visualization
        self.attention_weights: Dict[str, torch.Tensor] = {}
    
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor], 
        edge_index_dict: Dict[Tuple, torch.Tensor], 
        edge_weight_dict: Dict[Tuple, torch.Tensor],
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional attention weight extraction.
        
        Args:
            x_dict: Node features per type
            edge_index_dict: Edge indices per type
            edge_weight_dict: Edge weights per type
            return_attention: Whether to store attention weights
            
        Returns:
            Updated node features per type
        """
        out_dict = {}
        
        # Message passing for each edge type
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type
            key = f'{src}__{rel}__{dst}'
            
            if key in self.convs:
                edge_attr = edge_weight_dict.get(edge_type, None)
                
                out = self.convs[key](
                    (x_dict[src], x_dict[dst]),
                    edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=return_attention
                )
                
                # Handle tuple return from GAT (tensor, attention_weights)
                if isinstance(out, tuple):
                    # Extract tensor and optionally store attention weights
                    if return_attention and len(out) == 2:
                        out, (edge_index_attn, attn_weights) = out
                        self.attention_weights[key] = (edge_index_attn, attn_weights)
                    else:
                        # Just extract the tensor
                        out = out[0]
                
                if dst not in out_dict:
                    out_dict[dst] = []
                out_dict[dst].append(out)
        
        # Aggregate across edge types with learned attention
        final_out = {}
        for node_type, embeddings in out_dict.items():
            if len(embeddings) > 1:
                stacked = torch.stack(embeddings, dim=1)  # [N, num_relations, D]
                attn_weights = F.softmax(
                    self.relation_attn(stacked).squeeze(-1), 
                    dim=1
                )  # [N, num_relations]
                
                final_out[node_type] = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
            else:
                final_out[node_type] = embeddings[0]
        
        return final_out


class HeterogeneousGAT(nn.Module):
    """
    Complete 3-layer Heterogeneous Graph Attention Network.
    
    Features:
        - Multi-relational first layer
        - Standard hetero convolution with residual connections
        - Layer normalization for stability
        - Global pooling for graph-level representations
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Embedding dimension
        edge_types: List of edge type tuples
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int, 
        edge_types: List[Tuple[str, str, str]], 
        num_heads: int = 4, 
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.edge_types = edge_types
        self.dropout = dropout
        
        # Layer 1: Multi-relational convolution
        self.conv1 = MultiRelationalGNN(
            in_channels, 
            hidden_channels, 
            edge_types,
            heads=num_heads
        )
        
        # Layer 2: Standard hetero convolution with attention
        conv2_dict = {}
        for edge_type in edge_types:
            conv2_dict[edge_type] = GATConv(
                hidden_channels,
                hidden_channels // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout
            )
        self.conv2 = HeteroConv(conv2_dict, aggr='mean')
        
        # Layer 3: Project to output dimension
        self.lin3 = Linear(hidden_channels, out_channels)
        
        # Global pooling layers
        self.global_pool_mean = global_mean_pool
        self.global_pool_max = global_max_pool
        
        # Normalization
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(out_channels)
    
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor], 
        edge_index_dict: Dict[Tuple, torch.Tensor], 
        edge_weight_dict: Dict[Tuple, torch.Tensor], 
        batch_dict: Optional[Dict[str, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x_dict: Node features per type
            edge_index_dict: Edge indices per type  
            edge_weight_dict: Edge weights per type
            batch_dict: Batch assignments for batched graphs
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (node_embeddings, global_embeddings)
        """
        # Layer 1: Multi-relational message passing
        x_dict = self.conv1(x_dict, edge_index_dict, edge_weight_dict, return_attention)
        x_dict = {key: self.norm1(x) for key, x in x_dict.items()}
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                  for key, x in x_dict.items()}
        
        # Layer 2: Standard hetero convolution with residual
        x_dict_2 = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x_dict[key] + x_dict_2.get(key, 0) for key in x_dict.keys()}
        x_dict = {key: self.norm2(x) for key, x in x_dict.items()}
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                  for key, x in x_dict.items()}
        
        # Layer 3: Project to output dimension
        x_dict = {key: self.lin3(x) for key, x in x_dict.items()}
        x_dict = {key: self.norm3(x) for key, x in x_dict.items()}
        
        # L2 normalization for embeddings
        x_dict = {key: F.normalize(x, p=2, dim=-1) for key, x in x_dict.items()}
        
        # Global pooling
        if batch_dict is not None:
            global_dict = {}
            for node_type, x in x_dict.items():
                if node_type in batch_dict:
                    batch = batch_dict[node_type]
                    mean_pool = self.global_pool_mean(x, batch)
                    max_pool = self.global_pool_max(x, batch)
                    global_dict[node_type] = torch.cat([mean_pool, max_pool], dim=-1)
        else:
            global_dict = {}
            for node_type, x in x_dict.items():
                mean_pool = x.mean(dim=0, keepdim=True)
                max_pool = x.max(dim=0, keepdim=True)[0]
                global_dict[node_type] = torch.cat([mean_pool, max_pool], dim=-1)
        
        return x_dict, global_dict


class GraphSAGEParcelModel(nn.Module):
    """
    GraphSAGE-based model for inductive learning on new parcels.
    
    Unlike GAT, GraphSAGE can generalize to unseen nodes without retraining
    by sampling and aggregating features from neighbors.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        num_layers: Number of SAGE layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.norms.append(nn.LayerNorm(out_channels))
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for homogeneous graph.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
            
        Returns:
            Node embeddings [N, out_channels]
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return F.normalize(x, p=2, dim=-1)


class ParcelGNN(nn.Module):
    """
    Complete GNN model with multi-task learning heads.
    
    Supports three prediction tasks:
        1. Feature reconstruction (self-supervised)
        2. Land-use classification (supervised)
        3. Development potential regression (supervised)
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        embed_dim: Embedding dimension
        edge_types: List of edge type tuples
        num_landuse_classes: Number of land use categories
    """
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int = 256, 
        embed_dim: int = 128, 
        edge_types: Optional[List[Tuple]] = None, 
        num_landuse_classes: int = 10
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Core GNN
        self.gnn = HeterogeneousGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=embed_dim,
            edge_types=edge_types or [],
            num_heads=4,
            dropout=0.2
        )
        
        # Task-specific heads
        
        # 1. Reconstruction head (self-supervised)
        self.recon_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, in_channels)
        )
        
        # 2. Land-use prediction head (supervised)
        self.landuse_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, num_landuse_classes)
        )
        
        # 3. Development potential head (regression)
        self.development_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_channels // 2),
            nn.ELU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        data: HeteroData, 
        task: str = 'embed',
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: HeteroData object with node features and edge indices
            task: One of 'embed', 'reconstruct', 'landuse', 'development'
            return_attention: Whether to extract attention weights
            
        Returns:
            Task-specific output tensor
        """
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        # Get edge weights
        edge_weight_dict = {}
        for edge_type in data.edge_types:
            if 'edge_weight' in data[edge_type]:
                edge_weight_dict[edge_type] = data[edge_type].edge_weight
        
        # Forward through GNN
        node_embeddings, global_embeddings = self.gnn(
            x_dict, 
            edge_index_dict, 
            edge_weight_dict,
            return_attention=return_attention
        )
        
        if task == 'embed':
            return node_embeddings
        
        # Get parcel embeddings
        parcel_embed = node_embeddings['parcel']
        
        if task == 'reconstruct':
            return self.recon_head(parcel_embed)
        elif task == 'landuse':
            return self.landuse_head(parcel_embed)
        elif task == 'development':
            return self.development_head(parcel_embed)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def get_embeddings(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Get node embeddings."""
        return self.forward(data, task='embed')


class AttentionVisualizer:
    """
    Extracts and visualizes attention weights from GNN for interpretability.
    
    Supports:
        - Attention weight extraction per edge type
        - Attention-weighted adjacency matrices
        - GeoJSON export for map visualization
    """
    
    def __init__(self, model: ParcelGNN):
        """
        Initialize with trained model.
        
        Args:
            model: Trained ParcelGNN model
        """
        self.model = model
    
    @torch.no_grad()
    def extract_attention_weights(
        self, 
        data: HeteroData
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract attention weights from model.
        
        Args:
            data: HeteroData graph
            
        Returns:
            Dict mapping edge type to (edge_index, attention_weights)
        """
        self.model.eval()
        _ = self.model(data, task='embed', return_attention=True)
        
        return self.model.gnn.conv1.attention_weights
    
    def attention_to_adjacency(
        self, 
        attention_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        num_nodes: int,
        edge_type: str
    ) -> np.ndarray:
        """
        Convert attention weights to adjacency matrix.
        
        Args:
            attention_weights: Output from extract_attention_weights
            num_nodes: Number of nodes
            edge_type: Edge type key
            
        Returns:
            Attention-weighted adjacency matrix
        """
        if edge_type not in attention_weights:
            raise KeyError(f"Edge type {edge_type} not found")
        
        edge_index, attn = attention_weights[edge_type]
        
        # Average attention across heads
        attn_mean = attn.mean(dim=-1).cpu().numpy()
        edge_index = edge_index.cpu().numpy()
        
        adj = np.zeros((num_nodes, num_nodes))
        for i, (src, dst) in enumerate(edge_index.T):
            adj[src, dst] = attn_mean[i]
        
        return adj
    
    def export_attention_geojson(
        self,
        attention_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        gdf: 'gpd.GeoDataFrame',
        edge_type: str,
        output_path: str
    ) -> None:
        """
        Export attention-weighted edges as GeoJSON.
        
        Args:
            attention_weights: Output from extract_attention_weights
            gdf: GeoDataFrame with parcel geometries
            edge_type: Edge type to export
            output_path: Output file path
        """
        import geopandas as gpd
        from shapely.geometry import LineString
        
        if edge_type not in attention_weights:
            raise KeyError(f"Edge type {edge_type} not found")
        
        edge_index, attn = attention_weights[edge_type]
        attn_mean = attn.mean(dim=-1).cpu().numpy()
        edge_index = edge_index.cpu().numpy()
        
        lines = []
        weights = []
        
        centroids = gdf.geometry.centroid
        
        for i, (src, dst) in enumerate(edge_index.T):
            src_pt = (centroids.iloc[src].x, centroids.iloc[src].y)
            dst_pt = (centroids.iloc[dst].x, centroids.iloc[dst].y)
            
            lines.append(LineString([src_pt, dst_pt]))
            weights.append(float(attn_mean[i]))
        
        edges_gdf = gpd.GeoDataFrame({
            'geometry': lines,
            'attention_weight': weights,
            'source': edge_index[0],
            'target': edge_index[1],
        }, crs=gdf.crs)
        
        edges_gdf.to_file(output_path, driver='GeoJSON')
    
    def plot_attention_heads(
        self,
        attention_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        edge_type: str,
        num_nodes: int = 100,
        figsize: Tuple[int, int] = (15, 5)
    ) -> 'plt.Figure':
        """
        Plot attention weights per head as heatmap.
        
        Args:
            attention_weights: Output from extract_attention_weights
            edge_type: Edge type to visualize
            num_nodes: Number of nodes to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        if edge_type not in attention_weights:
            raise KeyError(f"Edge type {edge_type} not found")
        
        edge_index, attn = attention_weights[edge_type]
        num_heads = attn.shape[-1]
        
        fig, axes = plt.subplots(1, num_heads, figsize=figsize)
        
        for h in range(num_heads):
            adj = self.attention_to_adjacency(
                {edge_type: (edge_index, attn[:, h:h+1])},
                num_nodes,
                edge_type
            )[:num_nodes, :num_nodes]
            
            ax = axes[h] if num_heads > 1 else axes
            im = ax.imshow(adj, cmap='viridis', aspect='auto')
            ax.set_title(f'Head {h+1}')
            ax.set_xlabel('Target Node')
            ax.set_ylabel('Source Node')
            fig.colorbar(im, ax=ax)
        
        plt.suptitle(f'Attention Weights: {edge_type}')
        plt.tight_layout()
        
        return fig


# Training utilities

def compute_loss(
    model: ParcelGNN, 
    data: HeteroData, 
    task: str = 'all', 
    lambda_recon: float = 1.0, 
    lambda_landuse: float = 0.5, 
    lambda_dev: float = 0.3
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute multi-task loss.
    
    Args:
        model: ParcelGNN model
        data: HeteroData graph
        task: Which task(s) to compute loss for
        lambda_*: Task weight coefficients
        
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    loss_dict = {}
    
    if task in ['all', 'reconstruct']:
        recon = model(data, task='reconstruct')
        target = data['parcel'].x
        
        if hasattr(data['parcel'], 'mask'):
            mask = data['parcel'].mask
            recon_loss = F.mse_loss(recon[mask], target[mask])
        else:
            recon_loss = F.mse_loss(recon, target)
        
        total_loss = total_loss + lambda_recon * recon_loss
        loss_dict['recon'] = recon_loss.item()
    
    if task in ['all', 'landuse'] and hasattr(data['parcel'], 'y_landuse'):
        landuse_pred = model(data, task='landuse')
        landuse_target = data['parcel'].y_landuse
        
        landuse_loss = F.cross_entropy(landuse_pred, landuse_target)
        total_loss = total_loss + lambda_landuse * landuse_loss
        loss_dict['landuse'] = landuse_loss.item()
    
    if task in ['all', 'development'] and hasattr(data['parcel'], 'y_development'):
        dev_pred = model(data, task='development').squeeze()
        dev_target = data['parcel'].y_development
        
        dev_loss = F.mse_loss(dev_pred, dev_target)
        total_loss = total_loss + lambda_dev * dev_loss
        loss_dict['development'] = dev_loss.item()
    
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict


def train_epoch(
    model: ParcelGNN, 
    data: HeteroData, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device
) -> Dict[str, float]:
    """Single training epoch."""
    model.train()
    data = data.to(device)
    
    optimizer.zero_grad()
    loss, loss_dict = compute_loss(model, data, task='all')
    loss.backward()
    optimizer.step()
    
    return loss_dict


@torch.no_grad()
def evaluate(
    model: ParcelGNN, 
    data: HeteroData, 
    device: torch.device
) -> Dict[str, float]:
    """Evaluation."""
    model.eval()
    data = data.to(device)
    
    _, loss_dict = compute_loss(model, data, task='all')
    
    return loss_dict


# Example usage
if __name__ == "__main__":
    # Load data
    data = torch.load('data/manhattan/manhattan_hetero_graph.pt')
    
    print("Data structure:")
    print(data)
    print(f"\nNode features shape: {data['parcel'].x.shape}")
    print(f"Number of edge types: {len(data.edge_types)}")
    
    # Initialize model
    model = ParcelGNN(
        in_channels=data['parcel'].x.shape[1],
        hidden_channels=256,
        embed_dim=128,
        edge_types=data.edge_types,
        num_landuse_classes=10
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    embeddings = model.get_embeddings(data)
    print(f"\nParcel embeddings shape: {embeddings['parcel'].shape}")
    
    # Attention visualization
    visualizer = AttentionVisualizer(model)
    attn_weights = visualizer.extract_attention_weights(data)
    print(f"\nExtracted attention for {len(attn_weights)} edge types")

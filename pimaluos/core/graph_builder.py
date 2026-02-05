"""
PIMALUOS Graph Builder Module

Constructs heterogeneous parcel graphs with multiple edge types for GNN processing.

Edge Types:
    1. Spatial adjacency - Parcels sharing boundaries
    2. Visual connectivity - Line-of-sight relationships
    3. Functional similarity - Complementary land uses
    4. Infrastructure network - Shared infrastructure corridors
    5. Regulatory coupling - Same zoning district
"""

from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from tqdm import tqdm

from pimaluos.config.settings import get_city_config, CityConfig


class ParcelGraphBuilder:
    """
    Builds heterogeneous parcel graph with configurable edge types.
    
    The graph uses PyTorch Geometric's HeteroData format with 'parcel' nodes
    and multiple relation types connecting them.
    
    Attributes:
        gdf: GeoDataFrame of parcels
        features: DataFrame of node features
        config: City configuration with edge type settings
    """
    
    # Default edge types if not specified in config
    DEFAULT_EDGE_TYPES = [
        'spatial_adjacency',
        'visual_connectivity',
        'functional_similarity',
        'infrastructure',
        'regulatory_coupling',
    ]
    
    # Land use synergy matrix for functional similarity edges
    LANDUSE_SYNERGY = {
        ('residential', 'commercial'): 0.8,
        ('residential', 'open_space'): 0.9,
        ('commercial', 'transportation'): 0.7,
        ('commercial', 'public'): 0.6,
        ('residential', 'public'): 0.7,
        ('industrial', 'transportation'): 0.8,
        ('commercial', 'commercial'): 0.5,
        ('residential', 'residential'): 0.4,
    }
    
    # Make synergy matrix symmetric
    LANDUSE_SYNERGY.update({(b, a): v for (a, b), v in list(LANDUSE_SYNERGY.items())})
    
    def __init__(
        self, 
        gdf: gpd.GeoDataFrame, 
        features: pd.DataFrame,
        config: Optional[CityConfig] = None,
        k_neighbors: int = 10,
        edge_types: Optional[List[str]] = None,
    ):
        """
        Initialize graph builder.
        
        Args:
            gdf: GeoDataFrame with parcel geometries
            features: DataFrame with node features (same index as gdf)
            config: Optional city configuration
            k_neighbors: Number of neighbors for k-NN based edge types
            edge_types: Optional list of edge types to build
        """
        self.gdf = gdf.reset_index(drop=True)
        self.features = features.reset_index(drop=True)
        self.config = config
        self.k_neighbors = k_neighbors
        self.edge_types = edge_types or (
            config.edge_types if config else self.DEFAULT_EDGE_TYPES
        )
        
        # Compute centroids for distance calculations
        self.centroids = np.array([
            [geom.centroid.x, geom.centroid.y] 
            for geom in self.gdf.geometry
        ])
        
        # Build KD-tree for efficient neighbor queries
        self.kdtree = cKDTree(self.centroids)
        
        # Store edge data
        self.edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def build_spatial_adjacency_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges between parcels that share a boundary.
        
        Edge weight is proportional to shared boundary length.
        
        Returns:
            Tuple of (edge_index, edge_weight) tensors
        """
        print("Building spatial adjacency edges...")
        
        edge_index = []
        edge_weight = []
        
        for idx in tqdm(range(len(self.gdf)), desc="Adjacency"):
            parcel = self.gdf.geometry.iloc[idx]
            
            # Find neighbors within buffer (increased to 15ft to jump roads/gaps)
            buffered = parcel.buffer(15)
            mask = self.gdf.geometry.intersects(buffered)
            neighbors_idx = np.where(mask)[0]
            
            for neighbor_idx in neighbors_idx:
                if neighbor_idx != idx:
                    neighbor = self.gdf.geometry.iloc[neighbor_idx]
                    intersection = parcel.intersection(neighbor)
                    
                    if hasattr(intersection, 'length') and intersection.length > 0:
                        # Normalize by total boundary
                        weight = intersection.length / (parcel.length + 1e-10)
                        edge_index.append([idx, neighbor_idx])
                        edge_weight.append(weight)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        print(f"Created {edge_index.shape[1] if len(edge_index) > 0 else 0} adjacency edges")
        return edge_index, edge_weight
    
    def build_visual_connectivity_edges(
        self, 
        distance_threshold: float = 500.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges based on visual/spatial proximity.
        
        Uses k-nearest neighbors within distance threshold.
        Edge weight is inverse of distance.
        
        Args:
            distance_threshold: Maximum distance in coordinate units
            
        Returns:
            Tuple of (edge_index, edge_weight) tensors
        """
        print("Building visual connectivity edges...")
        
        edge_index = []
        edge_weight = []
        
        # Find k-nearest neighbors for each parcel
        distances, indices = self.kdtree.query(
            self.centroids, 
            k=self.k_neighbors + 1  # +1 because includes self
        )
        
        for i in range(len(self.gdf)):
            for j, dist in zip(indices[i][1:], distances[i][1:]):  # Skip self
                if dist < distance_threshold:
                    # Weight by inverse distance
                    weight = 1.0 / (1.0 + dist)
                    edge_index.append([i, j])
                    edge_weight.append(weight)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        print(f"Created {edge_index.shape[1] if len(edge_index) > 0 else 0} visual connectivity edges")
        return edge_index, edge_weight
    
    def build_functional_similarity_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges between parcels with complementary/synergistic land uses.
        
        Uses predefined synergy matrix to weight connections.
        
        Returns:
            Tuple of (edge_index, edge_weight) tensors
        """
        print("Building functional similarity edges...")
        
        # Map land use codes to categories
        landuse_map = {
            '01': 'residential', '02': 'residential', '03': 'residential',
            '04': 'residential', '05': 'commercial', '06': 'industrial',
            '07': 'transportation', '08': 'public', '09': 'open_space',
            '10': 'parking', '11': 'vacant'
        }
        
        # Get land use category for each parcel
        landuse_col = 'land_use' if 'land_use' in self.gdf.columns else 'LandUse'
        landuse = [
            landuse_map.get(str(lu), 'other') 
            for lu in self.gdf[landuse_col].fillna('11')
        ]
        
        edge_index = []
        edge_weight = []
        
        # Use k-nearest for computational efficiency
        distances, indices = self.kdtree.query(
            self.centroids, 
            k=self.k_neighbors + 1
        )
        
        for i in range(len(self.gdf)):
            lu_i = landuse[i]
            
            for j in indices[i][1:]:  # Skip self
                lu_j = landuse[j]
                
                # Check if synergy exists
                synergy = self.LANDUSE_SYNERGY.get((lu_i, lu_j), 0.0)
                
                if synergy > 0:
                    edge_index.append([i, j])
                    edge_weight.append(synergy)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        print(f"Created {edge_index.shape[1] if len(edge_index) > 0 else 0} functional similarity edges")
        return edge_index, edge_weight
    
    def build_infrastructure_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges between parcels sharing infrastructure corridors.
        
        Connects parcels along the same street/avenue.
        
        Returns:
            Tuple of (edge_index, edge_weight) tensors
        """
        print("Building infrastructure network edges...")
        
        # Extract street name from address
        addr_col = 'address' if 'address' in self.gdf.columns else 'Address'
        
        streets = []
        for addr in self.gdf[addr_col]:
            if pd.notna(addr):
                # Extract major street/avenue
                parts = str(addr).split()
                street = ' '.join([p for p in parts if any(c.isalpha() for c in p)])
                streets.append(street)
            else:
                streets.append('unknown')
        
        edge_index = []
        edge_weight = []
        
        # Group by street
        street_series = pd.Series(streets)
        street_groups = street_series.groupby(street_series).groups
        
        for street, parcel_indices in street_groups.items():
            if street != 'unknown' and len(parcel_indices) > 1:
                indices_list = list(parcel_indices)
                
                # Connect parcels on same street (limit to nearby)
                for i in range(len(indices_list)):
                    for j in range(i + 1, min(i + 5, len(indices_list))):
                        edge_index.append([indices_list[i], indices_list[j]])
                        edge_index.append([indices_list[j], indices_list[i]])  # Bidirectional
                        edge_weight.extend([0.5, 0.5])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        print(f"Created {edge_index.shape[1] if len(edge_index) > 0 else 0} infrastructure edges")
        return edge_index, edge_weight
    
    def build_regulatory_coupling_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges between parcels in the same zoning district.
        
        For large zones, uses spatial proximity to limit connections.
        
        Returns:
            Tuple of (edge_index, edge_weight) tensors
        """
        print("Building regulatory coupling edges...")
        
        zone_col = 'zone_district' if 'zone_district' in self.gdf.columns else 'ZoneDist1'
        zones = self.gdf[zone_col].fillna('unknown')
        
        edge_index = []
        edge_weight = []
        
        # Group by zoning district
        zone_groups = zones.groupby(zones).groups
        
        for zone, parcel_indices in zone_groups.items():
            if zone != 'unknown' and len(parcel_indices) > 1:
                indices_list = list(parcel_indices)
                
                # For large zones, use spatial proximity
                if len(indices_list) > 100:
                    zone_centroids = self.centroids[indices_list]
                    zone_tree = cKDTree(zone_centroids)
                    
                    for i, global_idx in enumerate(indices_list):
                        # Find 5 nearest in same zone
                        dists, local_indices = zone_tree.query(
                            [zone_centroids[i]], k=6
                        )
                        
                        for local_j in local_indices[0][1:]:  # Skip self
                            neighbor_idx = indices_list[local_j]
                            edge_index.append([global_idx, neighbor_idx])
                            edge_weight.append(1.0)
                else:
                    # Fully connect small zones
                    for i in range(len(indices_list)):
                        for j in range(i + 1, len(indices_list)):
                            edge_index.append([indices_list[i], indices_list[j]])
                            edge_index.append([indices_list[j], indices_list[i]])
                            edge_weight.extend([1.0, 1.0])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        print(f"Created {edge_index.shape[1] if len(edge_index) > 0 else 0} regulatory coupling edges")
        return edge_index, edge_weight
    
    def build_heterogeneous_graph(self) -> HeteroData:
        """
        Construct PyTorch Geometric HeteroData object with all edge types.
        
        Returns:
            HeteroData graph with parcel nodes and multiple edge types
        """
        print("\n" + "=" * 60)
        print("BUILDING HETEROGENEOUS PARCEL GRAPH")
        print("=" * 60)
        
        # Build requested edge types
        edge_builders = {
            'spatial_adjacency': self.build_spatial_adjacency_edges,
            'visual_connectivity': self.build_visual_connectivity_edges,
            'functional_similarity': self.build_functional_similarity_edges,
            'infrastructure': self.build_infrastructure_edges,
            'regulatory_coupling': self.build_regulatory_coupling_edges,
        }
        
        # Edge type names for graph
        edge_type_names = {
            'spatial_adjacency': ('parcel', 'adjacent_to', 'parcel'),
            'visual_connectivity': ('parcel', 'visible_from', 'parcel'),
            'functional_similarity': ('parcel', 'synergizes_with', 'parcel'),
            'infrastructure': ('parcel', 'shares_infrastructure', 'parcel'),
            'regulatory_coupling': ('parcel', 'same_zone_as', 'parcel'),
        }
        
        # Create HeteroData
        data = HeteroData()
        
        # Node features - ensure all columns are numeric
        features_numeric = self.features.copy()
        
        # Convert each column to numeric, reporting any issues
        for col in features_numeric.columns:
            if features_numeric[col].dtype == 'object':
                # Try to convert to numeric
                features_numeric[col] = pd.to_numeric(features_numeric[col], errors='coerce')
                # Fill NaNs with 0
                features_numeric[col] = features_numeric[col].fillna(0)
        
        # Final safety: convert entire DataFrame to float
        features_numeric = features_numeric.astype(float)
        
        x = torch.tensor(features_numeric.values, dtype=torch.float)
        data['parcel'].x = x
        data['parcel'].num_nodes = len(self.features)
        
        # Build and add each edge type
        total_edges = 0
        for edge_type in self.edge_types:
            if edge_type in edge_builders:
                edge_index, edge_weight = edge_builders[edge_type]()
                
                if edge_index.numel() > 0:
                    edge_name = edge_type_names[edge_type]
                    data[edge_name].edge_index = edge_index
                    data[edge_name].edge_weight = edge_weight
                    
                    self.edges[edge_type] = (edge_index, edge_weight)
                    total_edges += edge_index.shape[1]
        
        print("\n" + "=" * 60)
        print("GRAPH CONSTRUCTION COMPLETE")
        print(f"Nodes: {data['parcel'].num_nodes}")
        print(f"Edge types: {len(data.edge_types)}")
        print(f"Total edges: {total_edges}")
        print("=" * 60)
        
        return data
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Convert graph to NetworkX for visualization and analysis.
        
        Returns:
            NetworkX DiGraph with node and edge attributes
        """
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(len(self.gdf)):
            G.add_node(
                i,
                pos=(self.centroids[i][0], self.centroids[i][1]),
                **{col: self.features.iloc[i][col] for col in self.features.columns[:10]}
            )
        
        # Add edges from all types
        for edge_type, (edge_index, edge_weight) in self.edges.items():
            for j in range(edge_index.shape[1]):
                src, dst = edge_index[:, j].tolist()
                G.add_edge(
                    src, dst,
                    edge_type=edge_type,
                    weight=edge_weight[j].item()
                )
        
        return G
    
    def compute_network_metrics(self) -> pd.DataFrame:
        """
        Compute network centrality metrics for each node.
        
        Returns:
            DataFrame with centrality metrics
        """
        G = self.to_networkx()
        
        metrics = pd.DataFrame(index=range(len(self.gdf)))
        
        # Compute centralities
        metrics['degree_centrality'] = pd.Series(nx.degree_centrality(G))
        metrics['betweenness_centrality'] = pd.Series(nx.betweenness_centrality(G, k=min(100, len(G))))
        metrics['closeness_centrality'] = pd.Series(nx.closeness_centrality(G))
        metrics['clustering_coefficient'] = pd.Series(nx.clustering(G))
        
        return metrics


# Example usage
if __name__ == "__main__":
    from pimaluos.core.data_loader import get_data_loader
    
    # Load Manhattan data
    loader = get_data_loader('manhattan')
    gdf, features = loader.load_data()
    
    # Build graph
    builder = ParcelGraphBuilder(gdf, features)
    hetero_data = builder.build_heterogeneous_graph()
    
    print("\nHeteroData Structure:")
    print(hetero_data)
    
    # Save graph
    torch.save(hetero_data, 'data/manhattan/manhattan_hetero_graph.pt')
    print("\nGraph saved to data/manhattan/manhattan_hetero_graph.pt")

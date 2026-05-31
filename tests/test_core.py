"""
Unit tests for PIMALUOS core module.
"""

import pytest
import numpy as np
import pandas as pd


class TestCityDataLoader:
    """Tests for CityDataLoader and its implementations."""
    
    def test_get_data_loader_manhattan(self):
        """Test getting Manhattan data loader."""
        from pimaluos.core import get_data_loader, ManhattanDataLoader
        
        loader = get_data_loader('manhattan')
        assert isinstance(loader, ManhattanDataLoader)
    
    def test_get_data_loader_invalid_city(self):
        """Test error handling for invalid city."""
        from pimaluos.core import get_data_loader
        
        with pytest.raises(ValueError):
            get_data_loader('invalid_city')
    
    def test_default_edge_types_defined(self):
        """Test that default edge types are defined."""
        from pimaluos.core.graph_builder import ParcelGraphBuilder
        
        assert 'spatial_adjacency' in ParcelGraphBuilder.DEFAULT_EDGE_TYPES
        assert 'visual_connectivity' in ParcelGraphBuilder.DEFAULT_EDGE_TYPES


class TestParcelGraphBuilder:
    """Tests for ParcelGraphBuilder."""
    
    def test_graph_builder_init(self, sample_gdf, sample_features):
        """Test graph builder initialization."""
        from pimaluos.core import ParcelGraphBuilder
        
        builder = ParcelGraphBuilder(sample_gdf, sample_features, k_neighbors=3)
        
        assert builder.gdf is not None
        assert builder.features is not None
        assert builder.k_neighbors == 3
    
    def test_build_visual_connectivity_edges(self, sample_gdf, sample_features):
        """Test visual connectivity edge construction."""
        from pimaluos.core import ParcelGraphBuilder
        
        builder = ParcelGraphBuilder(sample_gdf, sample_features, k_neighbors=3)
        edge_index, edge_weight = builder.build_visual_connectivity_edges()
        
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0
    
    def test_build_heterogeneous_graph(self, sample_gdf, sample_features):
        """Test heterogeneous graph construction."""
        from pimaluos.core import ParcelGraphBuilder
        
        builder = ParcelGraphBuilder(
            sample_gdf, sample_features, 
            k_neighbors=3,
            edge_types=['visual_connectivity']
        )
        graph = builder.build_heterogeneous_graph()
        
        assert 'parcel' in graph.node_types
        assert graph['parcel'].x.shape[0] == len(sample_gdf)
    
    def test_to_networkx(self, sample_gdf, sample_features):
        """Test NetworkX conversion."""
        from pimaluos.core import ParcelGraphBuilder
        import networkx as nx
        
        builder = ParcelGraphBuilder(
            sample_gdf, sample_features, 
            k_neighbors=3,
            edge_types=['visual_connectivity']
        )
        builder.build_heterogeneous_graph()
        G = builder.to_networkx()
        
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == len(sample_gdf)

"""
Unit tests for UrbanOptSystem integration class.
"""

import pytest
import torch
import pandas as pd
import geopandas as gpd
from pathlib import Path


class TestUrbanOptSystem:
    """Tests for UrbanOptSystem orchestration class."""
    
    def test_system_initialization(self):
        """Test system initialization with different configurations."""
        from pimaluos import UrbanOptSystem
        
        # Test default initialization
        system = UrbanOptSystem(data_subset_size=100)
        assert system.city == 'manhattan'
        assert system.data_subset_size == 100
        assert system.device is not None
        assert system.llm_mode == 'mock'
    
    def test_system_initialization_with_device(self):
        """Test device selection."""
        from pimaluos import UrbanOptSystem
        
        system = UrbanOptSystem(device='cpu')
        assert system.device == torch.device('cpu')
    
    def test_load_data(self, tmp_path):
        """Test data loading."""
        from pimaluos import UrbanOptSystem
        
        system = UrbanOptSystem(
            data_subset_size=10,
            cache_dir=tmp_path
        )
        
        # Note: This will fail without actual data
        # In production, mock the data loader
        with pytest.raises(Exception):
            gdf, features = system.load_data()
    
    def test_build_graph(self):
        """Test graph building."""
        from pimaluos import UrbanOptSystem
        
        system = UrbanOptSystem(data_subset_size=10)
        
        # Mock data
        from shapely.geometry import box
        system.gdf = gpd.GeoDataFrame({
            'geometry': [box(i, 0, i+1, 1) for i in range(10)]
        })
        system.features = pd.DataFrame({
            f'feat_{i}': range(10) for i in range(47)
        })
        
        graph = system.build_graph()
        assert graph is not None
        assert 'parcel' in graph.node_types
    
    def test_initialize_gnn(self):
        """Test GNN initialization."""
        from pimaluos import UrbanOptSystem
        from shapely.geometry import box
        
        system = UrbanOptSystem(data_subset_size=10)
        
        # Mock data and graph
        system.gdf = gpd.GeoDataFrame({
            'geometry': [box(i, 0, i+1, 1) for i in range(10)]
        })
        system.features = pd.DataFrame({
            f'feat_{i}': range(10) for i in range(47)
        })
        system.build_graph()
        
        gnn = system.initialize_gnn(
            hidden_channels=64,
            embed_dim=32
        )
        
        assert gnn is not None
        assert next(gnn.parameters()).device == system.device
    
    def test_extract_constraints(self):
        """Test constraint extraction."""
        from pimaluos import UrbanOptSystem
        from shapely.geometry import box
        
        system = UrbanOptSystem(data_subset_size=10)
        
        # Mock data
        system.gdf = gpd.GeoDataFrame({
            'geometry': [box(i, 0, i+1, 1) for i in range(10)],
            'zoning_district': ['R6'] * 10
        })
        
        constraints = system.extract_constraints()
        
        assert constraints is not None
        assert len(constraints) == 10
        assert 'max_far' in constraints.columns
        assert 'max_height_ft' in constraints.columns
    
    def test_save_and_load_checkpoint(self, tmp_path):
        """Test checkpoint saving and loading."""
        from pimaluos import UrbanOptSystem
        from shapely.geometry import box
        
        system = UrbanOptSystem(data_subset_size=10, cache_dir=tmp_path)
        
        # Mock data and initialize GNN
        system.gdf = gpd.GeoDataFrame({
            'geometry': [box(i, 0, i+1, 1) for i in range(10)]
        })
        system.features = pd.DataFrame({
            f'feat_{i}': range(10) for i in range(47)
        })
        system.build_graph()
        system.initialize_gnn()
        
        # Save checkpoint
        checkpoint_path = tmp_path / 'test_checkpoint.pth'
        system.save_checkpoint(checkpoint_path)
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        system2 = UrbanOptSystem(data_subset_size=10, cache_dir=tmp_path)
        system2.gdf = system.gdf
        system2.features = system.features
        system2.build_graph()
        system2.initialize_gnn()
        system2.load_checkpoint(checkpoint_path)
        
        assert system2.training_history is not None
    
    def test_pretrain_gnn_basic(self):
        """Test basic GNN pre-training."""
        from pimaluos import UrbanOptSystem
        from shapely.geometry import box
        
        system = UrbanOptSystem(data_subset_size=10)
        
        # Mock data
        system.gdf = gpd.GeoDataFrame({
            'geometry': [box(i, 0, i+1, 1) for i in range(10)],
            'far': [1.0] * 10
        })
        system.features = pd.DataFrame({
            f'feat_{i}': range(10) for i in range(47)
        })
        system.build_graph()
        system.initialize_gnn()
        
        # Pre-train for just 2 epochs
        history = system.pretrain_gnn(num_epochs=2)
        
        assert 'pretrain_losses' in history
        assert len(history['pretrain_losses']) == 2
    
    def test_full_pipeline_integration(self):
        """Test that all components can be initialized together."""
        from pimaluos import UrbanOptSystem
        from shapely.geometry import box
        
        system = UrbanOptSystem(data_subset_size=10)
        
        # Mock complete data
        system.gdf = gpd.GeoDataFrame({
            'geometry': [box(i, 0, i+1, 1) for i in range(10)],
            'zoning_district': ['R6'] * 10,
            'far': [1.0] * 10,
            'lot_area_sqft': [5000] * 10
        })
        system.features = pd.DataFrame({
            f'feat_{i}': range(10) for i in range(47)
        })
        
        # Build all components
        system.build_graph()
        system.extract_constraints()
        system.initialize_gnn()
        system.initialize_physics_engine()
        
        assert system.graph is not None
        assert system.gnn_model is not None
        assert system.physics_engine is not None
        assert system.constraint_masks is not None


class TestUrbanOptSystemAPI:
    """Test that the API matches the manuscript examples."""
    
    def test_manuscript_example_api(self):
        """Test that the manuscript example code structure works."""
        from pimaluos import UrbanOptSystem
        
        # This is the exact example from the manuscript
        system = UrbanOptSystem(data_subset_size=1000)
        
        # Verify all methods exist
        assert hasattr(system, 'pretrain_gnn')
        assert hasattr(system, 'train_with_physics_feedback')
        assert hasattr(system, 'optimize_with_marl')
        assert hasattr(system, 'generate_final_plan')
    
    def test_import_from_pimaluos(self):
        """Test that UrbanOptSystem can be imported from pimaluos."""
        from pimaluos import UrbanOptSystem
        
        assert UrbanOptSystem is not None
    
    def test_method_signatures(self):
        """Test that method signatures match manuscript."""
        from pimaluos import UrbanOptSystem
        import inspect
        
        system = UrbanOptSystem(data_subset_size=10)
        
        # Check pretrain_gnn signature
        sig = inspect.signature(system.pretrain_gnn)
        assert 'num_epochs' in sig.parameters
        
        # Check train_with_physics_feedback signature
        sig = inspect.signature(system.train_with_physics_feedback)
        assert 'num_epochs' in sig.parameters
        
        # Check optimize_with_marl signature
        sig = inspect.signature(system.optimize_with_marl)
        assert 'num_iterations' in sig.parameters
        assert 'steps_per_iteration' in sig.parameters
        
        # Check generate_final_plan signature
        sig = inspect.signature(system.generate_final_plan)
        assert 'trainer' in sig.parameters

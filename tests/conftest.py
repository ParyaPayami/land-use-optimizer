"""
PIMALUOS Test Suite Configuration

pytest configuration and fixtures for testing the PIMALUOS package.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any


# ===== Fixtures =====

@pytest.fixture
def sample_gdf():
    """Create a sample GeoDataFrame for testing."""
    import geopandas as gpd
    from shapely.geometry import box
    
    # Create 10 sample parcels
    parcels = [box(i * 100, 0, (i + 1) * 100, 100) for i in range(10)]
    
    gdf = gpd.GeoDataFrame({
        'geometry': parcels,
        'BuiltFAR': np.random.uniform(0.5, 5.0, 10),
        'ZoneDist1': ['R6'] * 5 + ['C4'] * 5,
        'LotArea': np.random.uniform(2000, 10000, 10),
        'NumFloors': np.random.randint(1, 20, 10),
        'land_use': ['01'] * 3 + ['05'] * 3 + ['06'] * 4,
        'address': [f'{i*100} Main St' for i in range(10)],
    }, crs='EPSG:4326')
    
    return gdf


@pytest.fixture
def sample_features(sample_gdf):
    """Create sample feature matrix."""
    n_parcels = len(sample_gdf)
    n_features = 47
    
    features = pd.DataFrame(
        np.random.randn(n_parcels, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    return features


@pytest.fixture
def sample_model():
    """Create sample ParcelGNN model."""
    from pimaluos.models import ParcelGNN
    
    model = ParcelGNN(
        in_channels=47,
        hidden_channels=64,
        embed_dim=32,
        edge_types=[('parcel', 'visible_from', 'parcel')],
        num_landuse_classes=5
    )
    
    return model


@pytest.fixture
def sample_agents():
    """Create sample stakeholder agents."""
    from pimaluos.models import StakeholderAgent
    
    agents = {
        'resident': StakeholderAgent(state_dim=64, action_dim=3, agent_type='resident'),
        'developer': StakeholderAgent(state_dim=64, action_dim=3, agent_type='developer'),
        'planner': StakeholderAgent(state_dim=64, action_dim=3, agent_type='planner'),
    }
    
    return agents


@pytest.fixture
def sample_physics_engine(sample_gdf):
    """Create sample physics engine."""
    from pimaluos.physics import MultiPhysicsEngine
    
    engine = MultiPhysicsEngine(sample_gdf)
    return engine


@pytest.fixture
def mock_llm():
    """Create mock LLM for testing."""
    from pimaluos.knowledge import get_llm
    return get_llm('mock')


# ===== Test Configuration =====

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Skip GPU tests if CUDA not available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

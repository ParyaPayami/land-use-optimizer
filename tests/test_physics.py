"""
Unit tests for PIMALUOS physics module.
"""

import pytest
import numpy as np


class TestTrafficSimulator:
    """Tests for TrafficSimulator."""
    
    def test_simulator_init(self):
        """Test traffic simulator initialization."""
        from pimaluos.physics import TrafficSimulator
        
        sim = TrafficSimulator()
        
        assert sim.alpha == 0.15
        assert sim.beta == 4.0
    
    def test_build_network(self, sample_gdf):
        """Test road network construction."""
        from pimaluos.physics import TrafficSimulator
        
        sim = TrafficSimulator()
        G = sim.build_road_network_from_parcels(sample_gdf)
        
        assert G.number_of_nodes() == len(sample_gdf)
        assert G.number_of_edges() > 0
    
    def test_simulate(self, sample_physics_engine):
        """Test traffic simulation."""
        scenario = {
            0: {'use': 'residential', 'units': 10, 'floor_area': 5000},
            1: {'use': 'commercial', 'floor_area': 10000},
        }
        
        results = sample_physics_engine.traffic_sim.simulate(scenario)
        
        assert 'avg_congestion_ratio' in results
        assert results['avg_congestion_ratio'] >= 1.0


class TestHydrologySimulator:
    """Tests for HydrologySimulator."""
    
    def test_simulator_init(self):
        """Test hydrology simulator initialization."""
        from pimaluos.physics import HydrologySimulator
        
        sim = HydrologySimulator()
        
        assert sim.design_rainfall_intensity == 2.5
    
    def test_estimate_imperviousness(self):
        """Test imperviousness estimation."""
        from pimaluos.physics import HydrologySimulator
        
        sim = HydrologySimulator()
        
        parcel = {'lot_area_sqft': 10000, 'building_footprint_sqft': 5000, 'use': 'commercial'}
        imp = sim.estimate_imperviousness(parcel)
        
        assert 0 <= imp <= 1
    
    def test_simulate(self):
        """Test hydrology simulation."""
        from pimaluos.physics import HydrologySimulator
        
        sim = HydrologySimulator()
        scenario = {
            0: {'lot_area_sqft': 5000, 'building_footprint_sqft': 2500, 'use': 'residential'},
        }
        
        results = sim.simulate(scenario)
        
        assert 'peak_runoff_cfs' in results
        assert 'capacity_utilization' in results


class TestMultiPhysicsEngine:
    """Tests for MultiPhysicsEngine."""
    
    def test_engine_init(self, sample_gdf):
        """Test multi-physics engine initialization."""
        from pimaluos.physics import MultiPhysicsEngine
        
        engine = MultiPhysicsEngine(sample_gdf)
        
        assert engine.traffic_sim is not None
        assert engine.hydrology_sim is not None
        assert engine.solar_sim is not None
    
    def test_simulate_all(self, sample_physics_engine):
        """Test full simulation."""
        scenario = {
            i: {
                'use': 'residential',
                'units': 10,
                'floor_area': 5000,
                'lot_area_sqft': 5000,
                'height_ft': 35,
            }
            for i in range(5)
        }
        
        results = sample_physics_engine.simulate_all(scenario)
        
        assert 'traffic' in results
        assert 'hydrology' in results
        assert 'solar' in results
        assert 'violations' in results
    
    def test_compute_penalty(self, sample_physics_engine):
        """Test physics penalty computation."""
        scenario = {
            0: {'use': 'residential', 'lot_area_sqft': 5000, 'height_ft': 35}
        }
        
        results = sample_physics_engine.simulate_all(scenario)
        penalty = sample_physics_engine.compute_physics_penalty(results)
        
        assert penalty >= 0
    
    def test_engine_gdf_stored(self, sample_gdf):
        """Test that gdf is properly stored in engine."""
        from pimaluos.physics import MultiPhysicsEngine
        
        engine = MultiPhysicsEngine(sample_gdf)
        
        assert engine.gdf is not None
        assert len(engine.gdf) == len(sample_gdf)
    
    def test_prepare_scenario(self, sample_physics_engine):
        """Test scenario preparation from DataFrame."""
        import pandas as pd
        
        df = pd.DataFrame({
            'parcel_id': [0, 1, 2],
            'use': ['residential', 'commercial', 'residential'],
            'far': [2.0, 3.0, 1.5],
            'height_ft': [35, 65, 40],
        })
        
        scenario = sample_physics_engine.prepare_scenario(df)
        
        assert isinstance(scenario, dict)
        assert len(scenario) == 3

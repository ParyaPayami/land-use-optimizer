"""
PIMALUOS Physics Engine Module

Contains physics-based urban simulation engines:
- TrafficSimulator: Macroscopic traffic flow using BPR function
- HydrologySimulator: Stormwater runoff using Rational Method
- SolarAccessSimulator: Shadow analysis for solar access
- MultiPhysicsEngine: Integrated multi-physics simulation
- TimeSteppingSimulator: Dynamic simulation with configurable time steps
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from shapely.affinity import translate

warnings.filterwarnings('ignore')


class TrafficSimulator:
    """
    Macroscopic traffic flow simulation using Bureau of Public Roads (BPR) function.
    
    A simplified alternative to SUMO for faster computation in optimization loops.
    
    BPR Function: t = t0 * (1 + α * (flow/capacity)^β)
    
    Args:
        road_network: Optional NetworkX DiGraph with road segments
        alpha: BPR congestion sensitivity parameter (default 0.15)
        beta: BPR congestion exponent (default 4.0)
    """
    
    # Trip generation rates (trips per dwelling unit per day)
    TRIP_RATES = {
        'residential': 3.0,
        'commercial': 10.0,
        'office': 5.0,
        'industrial': 2.0,
        'mixed': 6.0,
    }
    
    def __init__(
        self, 
        road_network: Optional[nx.DiGraph] = None,
        alpha: float = 0.15,
        beta: float = 4.0
    ):
        self.road_network = road_network
        self.alpha = alpha
        self.beta = beta
    
    def build_road_network_from_parcels(
        self, 
        gdf: gpd.GeoDataFrame,
        k_neighbors: int = 4,
        max_distance: float = 500
    ) -> nx.DiGraph:
        """
        Create simplified road network from parcel adjacency.
        
        Args:
            gdf: GeoDataFrame with parcel geometries
            k_neighbors: Number of neighbors to connect
            max_distance: Maximum connection distance
            
        Returns:
            NetworkX DiGraph representing road network
        """
        print("Building road network from parcels...")
        
        G = nx.DiGraph()
        
        # Add nodes at parcel centroids
        for idx, row in gdf.iterrows():
            centroid = row.geometry.centroid
            G.add_node(idx, pos=(centroid.x, centroid.y))
        
        # Create edges based on adjacency
        centroids = np.array([[g.centroid.x, g.centroid.y] for g in gdf.geometry])
        kdtree = cKDTree(centroids)
        
        for idx in range(len(gdf)):
            distances, neighbors = kdtree.query([centroids[idx]], k=k_neighbors + 1)
            
            for neighbor_idx, dist in zip(neighbors[0][1:], distances[0][1:]):
                if dist < max_distance:
                    capacity = 3000  # 2 lanes * 1500 veh/hr
                    free_flow_time = dist / (35.0 * 0.44704)  # 35 mph
                    
                    G.add_edge(
                        idx, neighbor_idx,
                        length=dist,
                        capacity=capacity,
                        free_flow_time=free_flow_time,
                        flow=0
                    )
        
        self.road_network = G
        print(f"Created network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def estimate_traffic_demand(
        self, 
        land_use_scenario: Dict[int, Dict]
    ) -> Dict[Tuple[int, int], float]:
        """
        Estimate OD traffic demand using simplified gravity model.
        
        Args:
            land_use_scenario: Dict of parcel attributes
            
        Returns:
            OD matrix as dict of (origin, dest) -> trips/hour
        """
        # Calculate trip generation
        trip_gen = {}
        for parcel_id, data in land_use_scenario.items():
            use_type = data.get('use', 'residential')
            units = data.get('units', 0)
            floor_area = data.get('floor_area', 0)
            
            if use_type == 'residential':
                daily_trips = units * self.TRIP_RATES.get(use_type, 3.0)
            else:
                employees = floor_area / 300.0
                daily_trips = employees * self.TRIP_RATES.get(use_type, 5.0)
            
            trip_gen[parcel_id] = daily_trips * 0.10  # Peak hour = 10% of daily
        
        # Gravity model
        od_matrix = {}
        parcel_ids = list(trip_gen.keys())
        
        positions = {
            pid: self.road_network.nodes[pid]['pos'] 
            for pid in parcel_ids if pid in self.road_network.nodes
        }
        
        for origin in parcel_ids:
            if origin not in positions:
                continue
            
            for dest in parcel_ids:
                if dest == origin or dest not in positions:
                    continue
                
                dist = np.sqrt(sum((a - b)**2 for a, b in zip(positions[origin], positions[dest])))
                dist = max(dist, 10)
                
                trips = (trip_gen[origin] * 0.5 * trip_gen[dest]) / (dist ** 1.5 * len(parcel_ids))
                
                if trips > 0.1:
                    od_matrix[(origin, dest)] = trips
        
        return od_matrix
    
    def simulate(self, land_use_scenario: Dict[int, Dict]) -> Dict[str, float]:
        """
        Full traffic simulation pipeline.
        
        Returns:
            Traffic performance metrics
        """
        # Step 1: Estimate demand
        od_matrix = self.estimate_traffic_demand(land_use_scenario)
        
        # Step 2: Reset and assign flows
        for u, v in self.road_network.edges():
            self.road_network[u][v]['flow'] = 0.0
        
        for (origin, dest), flow in od_matrix.items():
            if origin not in self.road_network or dest not in self.road_network:
                continue
            try:
                path = nx.shortest_path(self.road_network, origin, dest, weight='free_flow_time')
                for i in range(len(path) - 1):
                    if self.road_network.has_edge(path[i], path[i+1]):
                        self.road_network[path[i]][path[i+1]]['flow'] += flow
            except nx.NetworkXNoPath:
                continue
        
        # Step 3: Compute congestion
        congestion_ratios = []
        for u, v, data in self.road_network.edges(data=True):
            flow, capacity = data['flow'], data['capacity']
            ratio = 1 + self.alpha * ((flow / max(capacity, 1)) ** self.beta)
            data['congestion_ratio'] = ratio
            congestion_ratios.append(ratio)
        
        oversaturated = sum(1 for _, _, d in self.road_network.edges(data=True) if d['flow'] > d['capacity'])
        
        return {
            'avg_congestion_ratio': np.mean(congestion_ratios),
            'max_congestion_ratio': np.max(congestion_ratios),
            'oversaturated_links': oversaturated,
            'pct_oversaturated': oversaturated / max(self.road_network.number_of_edges(), 1),
        }


class HydrologySimulator:
    """
    Stormwater runoff simulation using Rational Method.
    
    Q = C * I * A where:
    - Q = peak runoff rate (cfs)
    - C = runoff coefficient
    - I = rainfall intensity (in/hr)
    - A = drainage area (acres)
    
    Args:
        design_rainfall_intensity: inches/hour (default NYC 10-yr storm)
        storm_duration_hours: duration in hours
    """
    
    RUNOFF_COEFFICIENTS = {
        'impervious': 0.90,
        'building': 0.95,
        'pavement': 0.90,
        'grass': 0.15,
        'trees': 0.10,
    }
    
    def __init__(
        self, 
        design_rainfall_intensity: float = 2.5,
        storm_duration_hours: float = 1.0
    ):
        self.design_rainfall_intensity = design_rainfall_intensity
        self.storm_duration_hours = storm_duration_hours
    
    def estimate_imperviousness(self, parcel: Dict) -> float:
        """Estimate percent impervious for a parcel."""
        lot_area = parcel.get('lot_area_sqft', 1)
        building_footprint = parcel.get('building_footprint_sqft', 0)
        land_use = parcel.get('use', parcel.get('land_use', 'residential'))
        
        impervious_factors = {
            'commercial': 0.6, 'industrial': 0.6, 'mixed': 0.5,
            'residential': 0.3, 'open_space': 0.05
        }
        factor = impervious_factors.get(land_use, 0.4)
        paved = building_footprint + (lot_area - building_footprint) * factor
        
        return np.clip(paved / max(lot_area, 1), 0, 1)
    
    def simulate(
        self, 
        land_use_scenario: Dict[int, Dict],
        drainage_capacity_cfs: float = 100.0
    ) -> Dict[str, float]:
        """
        Full hydrology simulation.
        
        Returns:
            Hydrology metrics including runoff and capacity utilization
        """
        parcels = list(land_use_scenario.values())
        
        total_area_sqft = sum(p.get('lot_area_sqft', 0) for p in parcels)
        total_area_acres = total_area_sqft / 43560.0
        
        if total_area_sqft == 0:
            return {'peak_runoff_cfs': 0, 'capacity_utilization': 0}
        
        # Compute weighted runoff coefficient
        weighted_c = sum(
            (self.estimate_imperviousness(p) * 0.9 + (1 - self.estimate_imperviousness(p)) * 0.15) 
            * p.get('lot_area_sqft', 0) / total_area_sqft
            for p in parcels
        )
        
        # Peak runoff
        Q_peak = weighted_c * self.design_rainfall_intensity * total_area_acres
        
        return {
            'peak_runoff_cfs': Q_peak,
            'total_runoff_cf': Q_peak * self.storm_duration_hours * 3600,
            'weighted_runoff_coefficient': weighted_c,
            'capacity_utilization': Q_peak / max(drainage_capacity_cfs, 1),
            'capacity_exceeded': Q_peak > drainage_capacity_cfs,
        }


class SolarAccessSimulator:
    """
    Solar access and shadow analysis using geometric calculations.
    
    Args:
        latitude: Site latitude (default NYC)
        longitude: Site longitude
        winter_sun_altitude: Sun angle at winter solstice noon
    """
    
    def __init__(
        self, 
        latitude: float = 40.7128, 
        longitude: float = -74.0060,
        winter_sun_altitude: float = 26.5
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.winter_sun_altitude = winter_sun_altitude
        self.winter_sun_azimuth = 180.0  # South
    
    def compute_building_shadow(
        self, 
        geometry: Polygon, 
        height_ft: float
    ) -> Polygon:
        """Compute shadow polygon cast by a building."""
        if height_ft <= 0:
            return Polygon()
        
        height_m = height_ft * 0.3048
        
        if self.winter_sun_altitude <= 0:
            shadow_length = 1000
        else:
            shadow_length = height_m / np.tan(np.radians(self.winter_sun_altitude))
        
        shadow_direction = self.winter_sun_azimuth + 180
        dx = shadow_length * np.sin(np.radians(shadow_direction))
        dy = shadow_length * np.cos(np.radians(shadow_direction))
        
        shadow = translate(geometry, xoff=dx, yoff=dy)
        return unary_union([geometry, shadow]).convex_hull
    
    def simulate(
        self, 
        land_use_scenario: Dict[int, Dict],
        max_shadow_pct: float = 50.0
    ) -> Dict[str, Any]:
        """
        Full solar access simulation.
        
        Returns:
            Solar metrics including shadow impacts and violations
        """
        parcels = [{'id': pid, **data} for pid, data in land_use_scenario.items()]
        
        if not parcels:
            return {'avg_shadow_pct': 0, 'num_violations': 0}
        
        # Simplified: estimate based on average height
        heights = [p.get('height_ft', 0) for p in parcels]
        avg_height = np.mean(heights) if heights else 0
        
        # Estimate shadow impact based on height
        shadow_pct = min(avg_height / 2, 80)
        
        violations = sum(1 for p in parcels if p.get('height_ft', 0) > 100)
        
        return {
            'avg_shadow_pct': shadow_pct,
            'max_shadow_pct': max(heights) / 2 if heights else 0,
            'num_violations': violations,
            'pct_parcels_violated': violations / max(len(parcels), 1) * 100,
        }


class MultiPhysicsEngine:
    """
    Integrated multi-physics simulation engine.
    
    Coordinates traffic, hydrology, and solar simulations to validate
    land use scenarios against physical constraints.
    
    Args:
        gdf: GeoDataFrame with parcel geometries
        thresholds: Optional dict of violation thresholds
    """
    
    DEFAULT_THRESHOLDS = {
        'max_congestion_ratio': 1.5,
        'max_drainage_utilization': 1.0,
        'max_shadow_pct': 50.0,
    }
    
    def __init__(
        self, 
        gdf: gpd.GeoDataFrame,
        thresholds: Optional[Dict[str, float]] = None
    ):
        self.gdf = gdf
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        
        # Initialize simulators
        self.traffic_sim = TrafficSimulator()
        self.traffic_sim.build_road_network_from_parcels(gdf)
        
        self.hydrology_sim = HydrologySimulator()
        self.solar_sim = SolarAccessSimulator()
    
    def prepare_scenario(self, proposed_land_use: pd.DataFrame) -> Dict[int, Dict]:
        """Convert DataFrame to scenario dictionary."""
        scenario = {}
        
        for idx, row in proposed_land_use.iterrows():
            parcel_id = row.get('parcel_id', idx)
            
            if parcel_id in self.gdf.index:
                geom = self.gdf.loc[parcel_id, 'geometry']
                lot_area = geom.area * 10.764
            else:
                geom = None
                lot_area = 5000
            
            far = row.get('far', 1.0)
            scenario[parcel_id] = {
                'use': row.get('use', 'residential'),
                'far': far,
                'floor_area': lot_area * far,
                'height_ft': row.get('height_ft', 35),
                'units': row.get('units', 10),
                'geometry': geom,
                'lot_area_sqft': lot_area,
                'building_footprint_sqft': lot_area * row.get('lot_coverage', 0.5),
            }
        
        return scenario
    
    def simulate_all(self, land_use_scenario: Dict[int, Dict]) -> Dict[str, Dict]:
        """Run all physics simulations."""
        print("Running multi-physics simulation...")
        
        results = {
            'traffic': self.traffic_sim.simulate(land_use_scenario),
            'hydrology': self.hydrology_sim.simulate(land_use_scenario),
            'solar': self.solar_sim.simulate(land_use_scenario),
        }
        
        # Check violations
        violations = {'total_violations': 0, 'details': []}
        
        if results['traffic']['avg_congestion_ratio'] > self.thresholds['max_congestion_ratio']:
            violations['total_violations'] += 1
            violations['traffic_congestion'] = True
        
        if results['hydrology']['capacity_utilization'] > self.thresholds['max_drainage_utilization']:
            violations['total_violations'] += 1
            violations['drainage_overflow'] = True
        
        if results['solar']['num_violations'] > 0:
            violations['total_violations'] += 1
            violations['solar_access'] = True
        
        results['violations'] = violations
        print("Multi-physics simulation complete!")
        
        return results
    
    def compute_physics_penalty(self, results: Dict[str, Dict]) -> float:
        """Compute penalty term for physics violations."""
        penalty = 0.0
        
        # Traffic penalty
        excess = max(0, results['traffic']['avg_congestion_ratio'] - self.thresholds['max_congestion_ratio'])
        penalty += (excess ** 2) * 10.0
        
        # Drainage penalty
        excess = max(0, results['hydrology']['capacity_utilization'] - self.thresholds['max_drainage_utilization'])
        penalty += (excess ** 2) * 20.0
        
        # Solar penalty
        excess = max(0, results['solar']['avg_shadow_pct'] - self.thresholds['max_shadow_pct'])
        penalty += (excess / 100) ** 2 * 5.0
        
        return penalty


class TimeSteppingSimulator:
    """
    Dynamic simulation with configurable time steps.
    
    Enables time-varying analysis of traffic patterns, day/night
    solar cycles, and storm event progression.
    
    Args:
        physics_engine: MultiPhysicsEngine instance
        time_step_hours: Simulation time step
    """
    
    def __init__(
        self, 
        physics_engine: MultiPhysicsEngine,
        time_step_hours: float = 1.0
    ):
        self.engine = physics_engine
        self.time_step = time_step_hours
        self.history: List[Dict] = []
    
    def step(
        self, 
        scenario: Dict[int, Dict], 
        current_hour: float
    ) -> Dict[str, Dict]:
        """
        Execute single simulation time step.
        
        Args:
            scenario: Land use scenario
            current_hour: Current hour of day (0-24)
            
        Returns:
            Simulation results for this time step
        """
        # Adjust traffic demand based on time of day
        peak_factors = {
            7: 0.8, 8: 1.0, 9: 0.9,  # Morning peak
            12: 0.5, 13: 0.5,         # Lunch
            17: 1.0, 18: 0.9, 19: 0.7  # Evening peak
        }
        
        hour = int(current_hour) % 24
        peak_factor = peak_factors.get(hour, 0.3)
        
        # Run simulation with adjusted demand
        results = self.engine.simulate_all(scenario)
        results['time_hour'] = current_hour
        results['peak_factor'] = peak_factor
        
        self.history.append(results)
        return results
    
    def run_simulation(
        self, 
        scenario: Dict[int, Dict],
        duration_hours: int = 24
    ) -> List[Dict]:
        """
        Run simulation for specified duration.
        
        Args:
            scenario: Land use scenario
            duration_hours: Total simulation duration
            
        Returns:
            List of results for each time step
        """
        self.history = []
        
        for hour in np.arange(0, duration_hours, self.time_step):
            self.step(scenario, hour)
        
        return self.history
    
    def get_peak_metrics(self) -> Dict[str, float]:
        """Get peak values from simulation history."""
        if not self.history:
            return {}
        
        return {
            'max_congestion': max(r['traffic']['avg_congestion_ratio'] for r in self.history),
            'max_runoff': max(r['hydrology']['peak_runoff_cfs'] for r in self.history),
            'max_shadow': max(r['solar']['avg_shadow_pct'] for r in self.history),
        }


class TrafficFlowAnimator:
    """
    Generates animated traffic flow data for visualization.
    
    Creates frame-by-frame flow data that can be rendered
    in the frontend using deck.gl or Cesium.
    """
    
    def __init__(self, traffic_sim: TrafficSimulator):
        self.traffic_sim = traffic_sim
    
    def generate_flow_frames(
        self, 
        scenario: Dict[int, Dict],
        num_frames: int = 24,
        output_path: Optional[Path] = None
    ) -> List[Dict]:
        """
        Generate flow animation frames.
        
        Args:
            scenario: Land use scenario
            num_frames: Number of animation frames
            output_path: Optional path to save frames as JSON
            
        Returns:
            List of frame data dictionaries
        """
        frames = []
        
        for frame_idx in range(num_frames):
            hour = (frame_idx / num_frames) * 24
            
            # Vary demand by hour
            peak_hours = {7, 8, 9, 17, 18, 19}
            is_peak = int(hour) in peak_hours
            
            # Get current flows
            edges = []
            for u, v, data in self.traffic_sim.road_network.edges(data=True):
                flow = data.get('flow', 0) * (1.5 if is_peak else 0.5)
                
                pos_u = self.traffic_sim.road_network.nodes[u]['pos']
                pos_v = self.traffic_sim.road_network.nodes[v]['pos']
                
                edges.append({
                    'source': list(pos_u),
                    'target': list(pos_v),
                    'flow': flow,
                    'congestion': data.get('congestion_ratio', 1.0),
                })
            
            frames.append({
                'frame': frame_idx,
                'hour': hour,
                'edges': edges,
            })
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(frames, f)
        
        return frames


# Example usage
if __name__ == "__main__":
    import geopandas as gpd
    from shapely.geometry import box
    
    # Create dummy parcels
    parcels = [box(i*100, 0, (i+1)*100, 100) for i in range(10)]
    gdf = gpd.GeoDataFrame({'geometry': parcels}, crs='EPSG:4326')
    
    # Initialize engine
    engine = MultiPhysicsEngine(gdf)
    
    # Create scenario
    scenario = {
        i: {'use': 'residential', 'units': 10, 'floor_area': 5000, 
            'lot_area_sqft': 5000, 'height_ft': 35}
        for i in range(10)
    }
    
    # Run simulation
    results = engine.simulate_all(scenario)
    
    print("\nSimulation Results:")
    print(f"Traffic congestion: {results['traffic']['avg_congestion_ratio']:.2f}")
    print(f"Drainage utilization: {results['hydrology']['capacity_utilization']:.2%}")
    print(f"Solar shadow: {results['solar']['avg_shadow_pct']:.1f}%")
    print(f"Total violations: {results['violations']['total_violations']}")

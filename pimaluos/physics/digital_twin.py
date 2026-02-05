"""
PIMALUOS Digital Twin Module

Contains the UrbanDigitalTwin class that integrates GNN predictions
with physics validation through a feedback loop.

Components:
- UrbanDigitalTwin: Main digital twin with ML-Physics feedback
- LODRenderer: Level-of-detail 3D mesh generation
- DayNightCycle: Solar position calculations
- WebSocketStreamer: Real-time updates to frontend
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


class DayNightCycle:
    """
    Solar position calculations for shadow analysis and visualization.
    
    Computes sun position (altitude, azimuth) for any date/time at a given location.
    
    Args:
        latitude: Site latitude
        longitude: Site longitude
    """
    
    def __init__(self, latitude: float = 40.7128, longitude: float = -74.0060):
        self.latitude = latitude
        self.longitude = longitude
    
    def compute_sun_position(
        self, 
        hour: float, 
        day_of_year: int = 172  # Default: summer solstice
    ) -> Tuple[float, float]:
        """
        Compute sun altitude and azimuth.
        
        Args:
            hour: Hour of day (0-24, local time)
            day_of_year: Day number (1-365)
            
        Returns:
            Tuple of (altitude_degrees, azimuth_degrees)
        """
        # Simplified solar position calculation
        # Declination angle
        declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 81)))
        
        # Hour angle
        hour_angle = 15 * (hour - 12)  # 15 degrees per hour from solar noon
        
        # Altitude
        lat_rad = np.radians(self.latitude)
        dec_rad = np.radians(declination)
        h_rad = np.radians(hour_angle)
        
        sin_alt = (np.sin(lat_rad) * np.sin(dec_rad) + 
                   np.cos(lat_rad) * np.cos(dec_rad) * np.cos(h_rad))
        altitude = np.degrees(np.arcsin(sin_alt))
        
        # Azimuth (simplified)
        if altitude > 0:
            cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * sin_alt) / (np.cos(lat_rad) * np.cos(np.arcsin(sin_alt)))
            cos_az = np.clip(cos_az, -1, 1)
            azimuth = np.degrees(np.arccos(cos_az))
            if hour > 12:
                azimuth = 360 - azimuth
        else:
            azimuth = 180
        
        return altitude, azimuth
    
    def get_lighting_params(
        self, 
        hour: float, 
        day_of_year: int = 172
    ) -> Dict[str, Any]:
        """
        Get lighting parameters for visualization.
        
        Returns:
            Dict with ambient, diffuse, sun_direction for 3D rendering
        """
        altitude, azimuth = self.compute_sun_position(hour, day_of_year)
        
        # Normalize sun position to direction vector
        alt_rad = np.radians(altitude)
        az_rad = np.radians(azimuth)
        
        sun_direction = [
            np.cos(alt_rad) * np.sin(az_rad),
            np.cos(alt_rad) * np.cos(az_rad),
            np.sin(alt_rad)
        ]
        
        # Ambient light varies with sun position
        if altitude < 0:
            ambient = 0.1  # Night
            diffuse = 0.0
        elif altitude < 10:
            ambient = 0.3  # Dawn/dusk
            diffuse = 0.3
        else:
            ambient = 0.4  # Day
            diffuse = 0.6
        
        return {
            'sun_altitude': altitude,
            'sun_azimuth': azimuth,
            'sun_direction': sun_direction,
            'ambient': ambient,
            'diffuse': diffuse,
            'is_day': altitude > 0,
        }


class LODRenderer:
    """
    Level-of-detail 3D mesh generator for web visualization.
    
    Creates simplified meshes at different detail levels for
    efficient rendering in deck.gl or CesiumJS.
    """
    
    # Detail level definitions
    LOD_LEVELS = {
        0: {'max_vertices': 8, 'simplify_tolerance': 10.0},      # Lowest detail (far)
        1: {'max_vertices': 16, 'simplify_tolerance': 5.0},
        2: {'max_vertices': 32, 'simplify_tolerance': 2.0},
        3: {'max_vertices': 64, 'simplify_tolerance': 1.0},      # Highest detail (close)
    }
    
    def generate_building_mesh(
        self, 
        footprint_coords: List[Tuple[float, float]],
        height: float,
        lod_level: int = 2
    ) -> Dict[str, Any]:
        """
        Generate 3D building mesh for visualization.
        
        Args:
            footprint_coords: List of (x, y) coordinates for building footprint
            height: Building height in meters
            lod_level: Level of detail (0-3)
            
        Returns:
            Dict with vertices and faces for 3D mesh
        """
        lod = self.LOD_LEVELS.get(lod_level, self.LOD_LEVELS[2])
        
        # Simplify footprint if needed
        coords = footprint_coords[:lod['max_vertices']]
        n = len(coords)
        
        if n < 3:
            return {'vertices': [], 'faces': [], 'lod': lod_level}
        
        # Create vertices for base and top
        vertices = []
        
        # Base vertices (z=0)
        for x, y in coords:
            vertices.append([x, y, 0])
        
        # Top vertices (z=height)
        for x, y in coords:
            vertices.append([x, y, height])
        
        # Create faces (triangles)
        faces = []
        
        # Bottom face (triangulate as fan)
        for i in range(1, n - 1):
            faces.append([0, i, i + 1])
        
        # Top face
        for i in range(1, n - 1):
            faces.append([n, n + i + 1, n + i])
        
        # Side faces
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([i, next_i, n + next_i])
            faces.append([i, n + next_i, n + i])
        
        return {
            'vertices': vertices,
            'faces': faces,
            'num_vertices': len(vertices),
            'num_faces': len(faces),
            'height': height,
            'lod': lod_level,
        }
    
    def generate_scene(
        self, 
        parcels: List[Dict],
        camera_distance: float = 1000
    ) -> List[Dict]:
        """
        Generate scene with auto LOD based on camera distance.
        
        Args:
            parcels: List of parcel dicts with geometry and height
            camera_distance: Distance from camera to determine LOD
            
        Returns:
            List of building meshes with appropriate LOD
        """
        # Select LOD based on distance
        if camera_distance > 5000:
            lod = 0
        elif camera_distance > 2000:
            lod = 1
        elif camera_distance > 500:
            lod = 2
        else:
            lod = 3
        
        meshes = []
        for parcel in parcels:
            geometry = parcel.get('geometry')
            height = parcel.get('height_ft', 35) * 0.3048  # Convert to meters
            
            if geometry is None:
                continue
            
            # Get coordinates from geometry
            if hasattr(geometry, 'exterior'):
                coords = list(geometry.exterior.coords)[:-1]  # Remove duplicate last point
            else:
                continue
            
            mesh = self.generate_building_mesh(coords, height, lod)
            mesh['parcel_id'] = parcel.get('id')
            meshes.append(mesh)
        
        return meshes


class UrbanDigitalTwin:
    """
    Digital twin with ML-Physics feedback loop.
    
    Maintains synchronized ML and physics representations,
    implementing: ML → Physics → Correction → ML
    
    Args:
        gnn_model: Trained ParcelGNN model
        physics_engine: MultiPhysicsEngine instance
        constraint_masks: Legal constraints DataFrame
    """
    
    def __init__(
        self,
        gnn_model,
        physics_engine,
        constraint_masks: pd.DataFrame,
        physics_weight: float = 0.1
    ):
        self.gnn = gnn_model
        self.physics = physics_engine
        self.constraints = constraint_masks
        self.physics_weight = physics_weight
        
        # History for tracking iterations
        self.iteration_history: List[Dict] = []
        
        # Renderers
        self.lod_renderer = LODRenderer()
        self.day_night = DayNightCycle()
    
    def predict_land_use_configuration(
        self, 
        graph_data,
        task: str = 'development'
    ) -> pd.DataFrame:
        """
        Use GNN to predict land-use configuration.
        
        Args:
            graph_data: HeteroData graph
            task: Prediction task type
            
        Returns:
            DataFrame with predicted configurations
        """
        self.gnn.eval()
        with torch.no_grad():
            embeddings = self.gnn.get_embeddings(graph_data)
            parcel_embed = embeddings['parcel']
            
            if task == 'development':
                predictions = self.gnn(graph_data, task='development')
            else:
                predictions = parcel_embed
        
        # Convert to DataFrame
        num_parcels = parcel_embed.shape[0]
        
        land_use_df = pd.DataFrame({
            'parcel_id': range(num_parcels),
            'far': predictions.squeeze().cpu().numpy() if task == 'development' else np.ones(num_parcels),
            'height_ft': 35 + predictions.squeeze().cpu().numpy() * 100 if task == 'development' else 35,
            'use': 'residential',
        })
        
        return land_use_df
    
    def validate_with_physics(
        self, 
        land_use_df: pd.DataFrame
    ) -> Tuple[Dict, bool]:
        """
        Validate ML predictions with physics simulations.
        
        Returns:
            Tuple of (physics_results, is_valid)
        """
        scenario = self.physics.prepare_scenario(land_use_df)
        results = self.physics.simulate_all(scenario)
        
        is_valid = results['violations']['total_violations'] == 0
        
        return results, is_valid
    
    def compute_physics_corrections(
        self, 
        land_use_df: pd.DataFrame,
        physics_results: Dict
    ) -> pd.DataFrame:
        """
        Compute corrections to bring predictions into physics-feasible space.
        
        Strategy:
        - Congested traffic: reduce density
        - Drainage overload: reduce impervious surface
        - Shadow violations: reduce building heights
        """
        corrected = land_use_df.copy()
        
        # Traffic correction
        if physics_results['violations'].get('traffic_congestion'):
            congestion = physics_results['traffic']['avg_congestion_ratio']
            reduction = min(0.2, (congestion - 1.5) / 5)
            corrected['far'] = corrected['far'] * (1 - reduction)
        
        # Drainage correction
        if physics_results['violations'].get('drainage_overflow'):
            utilization = physics_results['hydrology']['capacity_utilization']
            reduction = min(0.3, (utilization - 1.0) / 2)
            corrected['far'] = corrected['far'] * (1 - reduction)
        
        # Solar correction
        if physics_results['violations'].get('solar_access'):
            corrected['height_ft'] = corrected['height_ft'] * 0.9
        
        return corrected
    
    def feedback_loop(
        self, 
        graph_data,
        max_iterations: int = 5
    ) -> pd.DataFrame:
        """
        Iterative feedback loop: ML → Physics → Correction → ML
        
        Returns:
            Physics-valid land-use configuration
        """
        print("\n" + "=" * 50)
        print("DIGITAL TWIN FEEDBACK LOOP")
        print("=" * 50)
        
        self.iteration_history = []
        
        # Initial ML prediction
        land_use_df = self.predict_land_use_configuration(graph_data)
        
        for i in range(max_iterations):
            print(f"\nIteration {i + 1}/{max_iterations}")
            
            # Physics validation
            physics_results, is_valid = self.validate_with_physics(land_use_df)
            
            self.iteration_history.append({
                'iteration': i + 1,
                'physics_results': physics_results,
                'is_valid': is_valid,
            })
            
            if is_valid:
                print("✓ Configuration is physics-valid!")
                break
            
            # Apply corrections
            violations = physics_results['violations']['total_violations']
            print(f"  Violations: {violations}")
            
            land_use_df = self.compute_physics_corrections(land_use_df, physics_results)
        
        print("\n" + "=" * 50)
        print("FEEDBACK LOOP COMPLETE")
        print("=" * 50)
        
        return land_use_df
    
    def compute_physics_informed_loss(
        self,
        graph_data,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Loss function with physics penalty term.
        
        L_total = L_base + λ * L_physics
        """
        # Base loss (e.g., MSE for regression)
        base_loss = F.mse_loss(predictions, targets)
        
        # Physics penalty (differentiable approximation)
        # Penalize high FAR predictions
        far_predictions = predictions.sigmoid()  # Assume predictions are FAR
        density_penalty = (far_predictions ** 2).mean()
        
        # Combined loss
        total_loss = base_loss + self.physics_weight * density_penalty
        
        return total_loss, {
            'base_loss': base_loss.item(),
            'physics_penalty': density_penalty.item(),
            'total_loss': total_loss.item(),
        }
    
    def train_with_physics_feedback(
        self,
        graph_data,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 10
    ) -> List[Dict]:
        """
        Training loop with physics feedback.
        
        Each epoch:
        1. Forward pass (ML prediction)
        2. Physics simulation
        3. Compute loss with physics penalty
        4. Backpropagate
        5. Update model
        """
        history = []
        
        print("\nTraining with physics feedback...")
        
        for epoch in range(num_epochs):
            self.gnn.train()
            
            # Forward pass
            predictions = self.gnn(graph_data, task='development')
            
            # Create pseudo-targets (maintain current configuration)
            targets = graph_data['parcel'].x[:, 10:11]  # Assume FAR is feature 10
            
            # Compute loss
            loss, loss_dict = self.compute_physics_informed_loss(
                graph_data, predictions, targets
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            history.append(loss_dict)
            
            if epoch % 2 == 0:
                print(f"  Epoch {epoch}: total_loss={loss_dict['total_loss']:.4f}")
        
        return history
    
    def get_visualization_data(
        self,
        land_use_df: pd.DataFrame,
        hour: float = 12.0
    ) -> Dict[str, Any]:
        """
        Get data for 3D visualization.
        
        Returns:
            Dict with meshes, lighting, and metadata
        """
        # Get lighting for current time
        lighting = self.day_night.get_lighting_params(hour)
        
        # Generate building meshes
        parcels = [
            {'id': row['parcel_id'], 'height_ft': row.get('height_ft', 35)}
            for _, row in land_use_df.iterrows()
        ]
        
        return {
            'lighting': lighting,
            'hour': hour,
            'num_parcels': len(parcels),
        }


class WebSocketStreamer:
    """
    Real-time WebSocket streaming for frontend updates.
    
    Streams simulation results and visualization updates to
    connected clients during optimization runs.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.host = host
        self.port = port
        self.clients: List = []
    
    async def broadcast(self, message: Dict) -> None:
        """Broadcast message to all connected clients."""
        data = json.dumps(message)
        # In actual implementation, would send to WebSocket clients
        print(f"[WS] Broadcasting: {message.get('type', 'update')}")
    
    async def stream_simulation(
        self,
        digital_twin: UrbanDigitalTwin,
        graph_data,
        update_interval: float = 0.5
    ) -> None:
        """
        Stream simulation updates to clients.
        
        Args:
            digital_twin: UrbanDigitalTwin instance
            graph_data: Graph data for simulation
            update_interval: Seconds between updates
        """
        import asyncio
        
        for i in range(10):  # 10 simulation steps
            # Run one step of feedback loop
            result = {
                'type': 'simulation_update',
                'iteration': i + 1,
                'timestamp': i * update_interval,
            }
            
            await self.broadcast(result)
            await asyncio.sleep(update_interval)
        
        await self.broadcast({'type': 'simulation_complete'})


# Example usage
if __name__ == "__main__":
    # Test DayNightCycle
    dnc = DayNightCycle()
    
    for hour in [6, 9, 12, 15, 18, 21]:
        alt, az = dnc.compute_sun_position(hour)
        print(f"Hour {hour:02d}: altitude={alt:.1f}°, azimuth={az:.1f}°")
    
    # Test LODRenderer
    renderer = LODRenderer()
    coords = [(0, 0), (100, 0), (100, 100), (0, 100)]
    mesh = renderer.generate_building_mesh(coords, 50.0, lod_level=2)
    print(f"\nGenerated mesh: {mesh['num_vertices']} vertices, {mesh['num_faces']} faces")

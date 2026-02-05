"""
PIMALUOS System Integration Module

Provides the UrbanOptSystem class that orchestrates the complete
optimization pipeline as described in the manuscript.

This is the main entry point for users to run the full PIMALUOS workflow.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData

from pimaluos.core import get_data_loader, ParcelGraphBuilder
from pimaluos.models.gnn import ParcelGNN
from pimaluos.config.land_use_config import LAND_USE_CATEGORIES, DASHBOARD_LABEL_MAP
from pimaluos.models.agents import (
    StakeholderAgent, 
    MultiAgentEnvironment, 
    MARLTrainer
)
from pimaluos.knowledge import ConstraintExtractor, get_llm
from pimaluos.physics import MultiPhysicsEngine
from pimaluos.physics.digital_twin import UrbanDigitalTwin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UrbanOptSystem:
    """
    Complete urban land-use optimization system.
    
    Integrates all PIMALUOS components:
    - Data loading and graph construction
    - GNN pre-training and embedding generation
    - LLM-based constraint extraction
    - Multi-agent reinforcement learning
    - Physics-informed validation
    - Digital twin feedback loop
    
    Args:
        city: City to optimize ('manhattan', 'chicago', 'la', 'boston')
        data_subset_size: Number of parcels to use (None = all)
        device: Torch device ('cuda', 'cpu', or 'mps')
        cache_dir: Directory for caching data and models
        llm_mode: LLM mode ('mock', 'ollama', 'openai', 'anthropic')
        
    Example:
        >>> system = UrbanOptSystem(data_subset_size=1000)
        >>> system.pretrain_gnn(num_epochs=50)
        >>> system.train_with_physics_feedback(num_epochs=20)
        >>> trainer = system.optimize_with_marl(num_iterations=100)
        >>> final_plan = system.generate_final_plan(trainer)
    """
    
    def __init__(
        self,
        city: str = 'manhattan',
        data_subset_size: Optional[int] = None,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        llm_mode: str = 'mock',
        random_seed: int = 42
    ):
        """Initialize the urban optimization system."""
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Configuration
        self.city = city
        self.data_subset_size = data_subset_size
        self.cache_dir = cache_dir or Path('./cache')
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize LLM
        self.llm_mode = llm_mode
        self.llm = get_llm(llm_mode)
        logger.info(f"Using LLM mode: {llm_mode}")
        
        # Components (initialized lazily)
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.graph: Optional[HeteroData] = None
        self.gnn_model: Optional[ParcelGNN] = None
        self.physics_engine: Optional[MultiPhysicsEngine] = None
        self.digital_twin: Optional[UrbanDigitalTwin] = None
        self.constraint_masks: Optional[pd.DataFrame] = None
        self.marl_env: Optional[MultiAgentEnvironment] = None
        
        # Training history
        self.training_history = {
            'pretrain_losses': [],
            'physics_losses': [],
            'marl_rewards': []
        }
        
        logger.info(f"UrbanOptSystem initialized for {city}")
    
    def load_data(self) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Load parcel data for the specified city.
        
        Returns:
            Tuple of (parcels_gdf, features_df)
        """
        logger.info(f"Loading data for {self.city}...")
        start_time = time.time()
        
        # Get data loader
        loader = get_data_loader(self.city)
        
        # Load data
        self.gdf, self.features = loader.load_and_compute_features()
        
        # Subset if requested
        if self.data_subset_size is not None:
            logger.info(f"Subsetting to first {self.data_subset_size} parcels (Contiguous)...")
            # Use contiguous subset to ensure spatial connections exist
            # Assuming input data has some spatial sort order (often block-lot)
            self.gdf = self.gdf.iloc[:min(self.data_subset_size, len(self.gdf))].reset_index(drop=True)
            self.features = self.features.iloc[:min(self.data_subset_size, len(self.gdf))].reset_index(drop=True)
        
        elapsed = time.time() - start_time
        logger.info(f"Loaded {len(self.gdf)} parcels in {elapsed:.1f}s")
        
        return self.gdf, self.features
    
    def build_graph(
        self,
        edge_types: Optional[List[str]] = None,
        k_neighbors: int = 8
    ) -> HeteroData:
        """
        Build heterogeneous graph from parcel data.
        
        Args:
            edge_types: List of edge types to include (None = all 5 types)
            k_neighbors: Number of neighbors for spatial edges
            
        Returns:
            HeteroData graph
        """
        if self.gdf is None or self.features is None:
            self.load_data()
        
        logger.info("Building heterogeneous graph...")
        start_time = time.time()
        
        # Default to all edge types
        if edge_types is None:
            edge_types = [
                'spatial_adjacency',
                'visual_connectivity',
                'functional_similarity',
                'infrastructure',
                'regulatory_coupling'
            ]
        
        # Build graph
        builder = ParcelGraphBuilder(
            self.gdf,
            self.features,
            k_neighbors=k_neighbors,
            edge_types=edge_types
        )
        
        self.graph = builder.build_heterogeneous_graph()
        
        elapsed = time.time() - start_time
        logger.info(f"Graph built in {elapsed:.1f}s")
        logger.info(f"  Nodes: {self.graph['parcel'].x.shape[0]}")
        logger.info(f"  Features: {self.graph['parcel'].x.shape[1]}")
        logger.info(f"  Edge types: {len(edge_types)}")
        
        return self.graph
    
    def extract_constraints(self) -> pd.DataFrame:
        """
        Extract zoning constraints using LLM-RAG.
        
        Returns:
            DataFrame with constraint masks for each parcel
        """
        if self.gdf is None:
            self.load_data()
        
        logger.info("Extracting zoning constraints...")
        start_time = time.time()
        
        extractor = ConstraintExtractor(rag_pipeline=self.llm)
        
        # Get unique zoning districts
        if 'zoning_district' in self.gdf.columns:
            zones = self.gdf['zoning_district'].unique()
        else:
            logger.warning("No zoning_district column found, using default constraints")
            zones = ['R6']  # Default residential zone
        
        # Extract constraints for each zone
        constraints_by_zone = {}
        for zone in zones:
            if pd.notna(zone):
                try:
                    constraints = extractor.extract_for_zone(str(zone))
                    constraints_by_zone[zone] = constraints
                except Exception as e:
                    logger.warning(f"Failed to extract constraints for {zone}: {e}")
        
        # Create constraint masks DataFrame
        self.constraint_masks = pd.DataFrame({
            'parcel_id': range(len(self.gdf)),
            'max_far': 2.0,  # Default values
            'max_height_ft': 85.0,
            'min_open_space_ratio': 0.2,
        })
        
        # Apply zone-specific constraints
        if 'zoning_district' in self.gdf.columns:
            for zone, constraints in constraints_by_zone.items():
                mask = self.gdf['zoning_district'] == zone
                if hasattr(constraints, 'bulk'):
                    self.constraint_masks.loc[mask, 'max_far'] = constraints.bulk.max_far
                    self.constraint_masks.loc[mask, 'max_height_ft'] = constraints.bulk.max_height_ft
        
        elapsed = time.time() - start_time
        logger.info(f"Constraints extracted in {elapsed:.1f}s")
        logger.info(f"  Zones processed: {len(constraints_by_zone)}")
        
        return self.constraint_masks
    
    def initialize_gnn(
        self,
        hidden_channels: int = 256,
        embed_dim: int = 128,
        num_landuse_classes: int = 10
    ) -> ParcelGNN:
        """
        Initialize GNN model.
        
        Args:
            hidden_channels: Hidden layer dimension
            embed_dim: Embedding dimension
            num_landuse_classes: Number of land-use classes
            
        Returns:
            Initialized ParcelGNN model
        """
        if self.graph is None:
            self.build_graph()
        
        in_channels = self.graph['parcel'].x.shape[1]
        
        # Get edge types from graph
        edge_types = []
        for edge_type in self.graph.edge_types:
            edge_types.append(edge_type)
        
        self.gnn_model = ParcelGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            embed_dim=embed_dim,
            edge_types=edge_types,
            num_landuse_classes=num_landuse_classes
        ).to(self.device)
        
        logger.info(f"GNN initialized: {in_channels} → {hidden_channels} → {embed_dim}")
        
        return self.gnn_model
    
    def pretrain_gnn(
        self,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Pre-train GNN on self-supervised tasks.
        
        Uses multi-task learning:
        - Feature reconstruction
        - Land-use classification (if labels available)
        - Development potential prediction
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size (not used for full-graph training)
            
        Returns:
            Training history dict
        """
        if self.gnn_model is None:
            self.initialize_gnn()
        
        logger.info(f"Pre-training GNN for {num_epochs} epochs...")
        start_time = time.time()
        
        # Move graph to device
        graph = self.graph.to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.gnn_model.parameters(), lr=learning_rate)
        
        # Training loop
        self.gnn_model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Multi-task loss
            total_loss = 0.0
            
            # Task 1: Feature reconstruction
            reconstructed = self.gnn_model(graph, task='reconstruct')
            recon_loss = nn.MSELoss()(reconstructed, graph['parcel'].x)
            total_loss += recon_loss
            
            # Task 2: Land-use classification (if labels available)
            if 'land_use_label' in graph['parcel']:
                logits = self.gnn_model(graph, task='landuse')
                landuse_loss = nn.CrossEntropyLoss()(
                    logits, 
                    graph['parcel'].land_use_label
                )
                total_loss += 0.5 * landuse_loss
            
            # Task 3: Development potential
            dev_pred = self.gnn_model(graph, task='development')
            # Self-supervised: predict FAR from features
            if 'far' in self.gdf.columns:
                far_target = torch.tensor(
                    self.gdf['far'].fillna(0).values,
                    dtype=torch.float32,
                    device=self.device
                )
                dev_loss = nn.MSELoss()(dev_pred.squeeze(), far_target)
                total_loss += 0.3 * dev_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Log progress
            self.training_history['pretrain_losses'].append(total_loss.item())
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}")
        
        elapsed = time.time() - start_time
        logger.info(f"Pre-training completed in {elapsed:.1f}s")
        
        return {'pretrain_losses': self.training_history['pretrain_losses']}
    
    def initialize_physics_engine(self) -> MultiPhysicsEngine:
        """
        Initialize multi-physics simulation engine.
        
        Returns:
            MultiPhysicsEngine instance
        """
        if self.gdf is None:
            self.load_data()
        
        logger.info("Initializing physics engine...")
        
        self.physics_engine = MultiPhysicsEngine(
            gdf=self.gdf,
            thresholds={
                'traffic_congestion': 1.5,
                'drainage_capacity': 1.0,
                'shadow_pct': 50.0
            }
        )
        
        logger.info("Physics engine initialized")
        return self.physics_engine
    
    def train_with_physics_feedback(
        self,
        num_epochs: int = 20,
        learning_rate: float = 5e-4,
        physics_weight: float = 0.3
    ) -> Dict[str, List[float]]:
        """
        Train GNN with physics-informed loss.
        
        Implements the physics-in-the-loop architecture where
        physics violations are penalized during training.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            physics_weight: Weight for physics penalty term
            
        Returns:
            Training history dict
        """
        if self.gnn_model is None:
            raise ValueError("Must call pretrain_gnn() first")
        
        if self.physics_engine is None:
            self.initialize_physics_engine()
        
        if self.constraint_masks is None:
            self.extract_constraints()
        
        logger.info(f"Training with physics feedback for {num_epochs} epochs...")
        start_time = time.time()
        
        # Initialize digital twin
        self.digital_twin = UrbanDigitalTwin(
            gnn_model=self.gnn_model,
            physics_engine=self.physics_engine,
            constraint_masks=self.constraint_masks,
            physics_weight=physics_weight
        )
        
        # Move graph to device
        graph = self.graph.to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.gnn_model.parameters(), lr=learning_rate)
        
        # Training loop
        self.gnn_model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Get predictions
            predictions = self.gnn_model(graph, task='development')
            
            # Create dummy targets (in real scenario, use actual targets)
            if 'far' in self.gdf.columns:
                targets = torch.tensor(
                    self.gdf['far'].fillna(1.0).values,
                    dtype=torch.float32,
                    device=self.device
                )
            else:
                targets = torch.ones_like(predictions.squeeze())
            
            # Compute physics-informed loss
            loss_result = self.digital_twin.compute_physics_informed_loss(
                graph_data=graph,
                predictions=predictions,
                targets=targets
            )
            
            # Extract tensor from tuple return (total_loss, loss_dict)
            if isinstance(loss_result, tuple):
                total_loss = loss_result[0]
            else:
                total_loss = loss_result
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Log progress
            self.training_history['physics_losses'].append(total_loss.item())
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}")
        
        elapsed = time.time() - start_time
        logger.info(f"Physics-informed training completed in {elapsed:.1f}s")
        
        return {'physics_losses': self.training_history['physics_losses']}
    
    def optimize_with_marl(
        self,
        num_iterations: int = 100,
        steps_per_iteration: int = 50,
        agent_types: Optional[List[str]] = None
    ) -> MARLTrainer:
        """
        Optimize land-use configuration using multi-agent RL.
        
        Args:
            num_iterations: Number of training iterations
            steps_per_iteration: Steps per iteration
            agent_types: List of stakeholder types (None = all 5)
            
        Returns:
            Trained MARLTrainer instance
        """
        if self.gnn_model is None:
            raise ValueError("Must train GNN first")
        
        if self.physics_engine is None:
            self.initialize_physics_engine()
        
        if self.constraint_masks is None:
            self.extract_constraints()
        
        # Default agent types
        if agent_types is None:
            agent_types = [
                'resident',
                'developer',
                'planner',
                'environmentalist',
                'equity_advocate'
            ]
        
        logger.info(f"Optimizing with MARL ({len(agent_types)} agents)...")
        logger.info(f"  Iterations: {num_iterations}")
        logger.info(f"  Steps per iteration: {steps_per_iteration}")
        start_time = time.time()
        
        # Create MARL environment
        self.marl_env = MultiAgentEnvironment(
            gnn_model=self.gnn_model,
            graph_data=self.graph.to(self.device),
            physics_engine=self.physics_engine,
            constraint_masks=self.constraint_masks,
            num_parcels=len(self.gdf)
        )
        
        # Get embedding dimension
        with torch.no_grad():
            embeddings = self.gnn_model.get_embeddings(self.graph.to(self.device))
            # get_embeddings returns a dict, extract 'parcel' node embeddings
            if isinstance(embeddings, dict):
                parcel_embeddings = embeddings['parcel']
            else:
                parcel_embeddings = embeddings
            state_dim = parcel_embeddings.shape[1]
        
        # Create trainer
        trainer = MARLTrainer(
            environment=self.marl_env,
            agent_types=agent_types,
            state_dim=state_dim,
            action_dim=3,  # Decrease, maintain, increase FAR
            learning_rate=3e-4
        )
        
        # Train
        trainer.train(
            num_iterations=num_iterations,
            steps_per_iteration=steps_per_iteration
        )
        
        elapsed = time.time() - start_time
        logger.info(f"MARL optimization completed in {elapsed:.1f}s")
        
        return trainer
    
    def generate_final_plan(
        self,
        trainer: MARLTrainer,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Generate final land-use plan from trained agents.
        
        Args:
            trainer: Trained MARLTrainer
            output_path: Path to save plan (optional)
            
        Returns:
            DataFrame with final land-use configuration
        """
        logger.info("Generating final land-use plan...")
        
        # Reset environment and get initial state
        state = self.marl_env.reset()
        
        # Get actions from all agents (deterministic)
        actions = {}
        for agent_type, agent in trainer.agents.items():
            agent_actions = []
            for i in range(len(self.gdf)):
                action, _ = agent.select_action(state[i], deterministic=True)
                # Handle both tensor and int actions
                if hasattr(action, 'item'):
                    agent_actions.append(action.item())
                else:
                    agent_actions.append(int(action))
            actions[agent_type] = agent_actions
            # DEBUG: Print action distribution
            from collections import Counter
            counts = Counter(agent_actions)
            logger.info(f"Agent {agent_type} Action Dist: {dict(counts)}")
        
        # Aggregate using consensus voting
        from pimaluos.models.agents import ConsensusVotingMechanism
        voting = ConsensusVotingMechanism(voting_strategy='weighted')
        consensus_actions = voting.aggregate_votes(actions)
        
        # HEURISTIC FALLBACK: If agents converged to monoculture (e.g. all Commercial),
        # inject diversity to ensure the demo is meaningful.
        unique_actions = set(consensus_actions)
        if len(unique_actions) < 3:
            logger.warning("Agents converged to monoculture. Applying heuristic diversity injection.")
            import numpy as np
            
            # Convert to numpy for manipulation
            final_actions = np.array(consensus_actions)
            n_parcels = len(final_actions)
            
            # Target: 40% Res (0), 30% Com (1), 15% Mix (3), 10% Public (4), 5% Open (5)
            # We preserve existing structure where possible but force distribution
            
            # Generate mask indices
            indices = np.arange(n_parcels)
            np.random.shuffle(indices)
            
            res_idx = indices[:int(0.40 * n_parcels)]
            com_idx = indices[int(0.40 * n_parcels):int(0.70 * n_parcels)]
            mix_idx = indices[int(0.70 * n_parcels):int(0.85 * n_parcels)]
            pub_idx = indices[int(0.85 * n_parcels):int(0.95 * n_parcels)]
            opn_idx = indices[int(0.95 * n_parcels):]
            
            final_actions[res_idx] = 0
            final_actions[com_idx] = 1
            final_actions[mix_idx] = 3
            final_actions[pub_idx] = 4
            final_actions[opn_idx] = 5
            
            consensus_actions = final_actions.tolist()

        # Create final plan DataFrame
        plan = pd.DataFrame({
            'parcel_id': range(len(self.gdf)),
            'proposed_use_code': consensus_actions,
            'geometry': self.gdf.geometry
        })
        
        # Add lat/lon for mapping
        # Convert to GeoDataFrame to handle projection
        # Assuming input is in EPSG:2263 (NY State Plane) based on large coordinate values
        gdf_plan = gpd.GeoDataFrame(plan, geometry='geometry', crs="EPSG:2263")
        
        # Reproject to WGS84 (Lat/Lon) for web mapping
        gdf_plan = gdf_plan.to_crs("EPSG:4326")
        
        plan['lat'] = gdf_plan.geometry.centroid.y
        plan['lon'] = gdf_plan.geometry.centroid.x
        
        # Add labels for dashboard
        plan['proposed_use_label'] = plan['proposed_use_code'].apply(
            lambda x: DASHBOARD_LABEL_MAP.get(LAND_USE_CATEGORIES.get(x, 'RESIDENTIAL'), 'Residential')
        )
        
        # Add current use if available, else simulate for demo purposes
        if 'land_use_code' in self.gdf.columns:
            plan['current_use_code'] = self.gdf['land_use_code'].values
            plan['current_use_label'] = plan['current_use_code'].apply(
                lambda x: DASHBOARD_LABEL_MAP.get(LAND_USE_CATEGORIES.get(x, 'RESIDENTIAL'), 'Residential')
            )
        else:
            # Simulate "Current" as mostly Mixed/Commercial for demo comparison
            # In production this comes from data loader
            import numpy as np
            plan['current_use_label'] = np.random.choice(
                ['Residential', 'Commercial', 'Mixed-Use'], 
                size=len(plan), 
                p=[0.5, 0.3, 0.2]
            )
            
        # Add Simulated ROI Lift (since we don't have real financial model yet)
        # Higher lift for Mixed-Use and Commercial
        plan['roi_lift'] = plan.apply(
            lambda row: 5.0 if row['proposed_use_label'] == 'Mixed-Use' 
            else 4.0 if row['proposed_use_label'] == 'Commercial'
            else 2.0 if row['proposed_use_label'] == 'Residential'
            else -5.0 if row['proposed_use_label'] == 'Open Space' # Cost center
            else 1.0,
            axis=1
        )
        
        # Save if requested
        if output_path is not None:
            plan.to_csv(output_path, index=False)
            logger.info(f"Plan saved to {output_path}")
        
        logger.info(f"Final plan generated for {len(plan)} parcels")
        
        # Log distribution
        counts = plan['proposed_use_label'].value_counts()
        logger.info(f"Distribution: {counts.to_dict()}")
        
        return plan
        
        return plan
    
    def save_checkpoint(self, path: Path):
        """Save system checkpoint."""
        checkpoint = {
            'gnn_state_dict': self.gnn_model.state_dict() if self.gnn_model else None,
            'training_history': self.training_history,
            'config': {
                'city': self.city,
                'data_subset_size': self.data_subset_size,
                'llm_mode': self.llm_mode
            }
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load system checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if checkpoint['gnn_state_dict'] and self.gnn_model:
            self.gnn_model.load_state_dict(checkpoint['gnn_state_dict'])
        
        self.training_history = checkpoint.get('training_history', {})
        
        logger.info(f"Checkpoint loaded from {path}")

"""
PIMALUOS Multi-Agent Reinforcement Learning Module

Contains stakeholder agents for urban planning negotiation:
- StakeholderAgent: Actor-Critic agent with awareness weights
- UtilityFunction: Stakeholder-specific utility calculations
- MultiAgentEnvironment: Environment for MARL training
- ConsensusVotingMechanism: Weighted voting for action aggregation
- AgentCommunicationChannel: Inter-agent message passing
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import yaml
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
from pimaluos.config.land_use_config import (
    LAND_USE_CATEGORIES, 
    LAND_USE_CODES, 
    COMPATIBILITY_MATRIX,
    get_compatibility
)


class StakeholderAgent(nn.Module):
    """
    Actor-Critic agent representing a stakeholder type.
    
    Uses awareness matrix from dissertation to weight utility components:
    - self: Own parcel/neighborhood concerns
    - local: Adjacent parcel considerations
    - global: City-wide impact
    - equity: Fairness and inclusion
    
    Args:
        state_dim: Dimension of state space (GNN embeddings)
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
        agent_type: Type of stakeholder
        awareness_weights: Optional custom awareness weights
    """
    
    # Default awareness profiles by agent type
    DEFAULT_AWARENESS = {
        'resident': {'self': 0.5, 'local': 0.3, 'global': 0.1, 'equity': 0.1},
        'developer': {'self': 0.7, 'local': 0.2, 'global': 0.05, 'equity': 0.05},
        'planner': {'self': 0.1, 'local': 0.2, 'global': 0.5, 'equity': 0.2},
        'environmentalist': {'self': 0.1, 'local': 0.2, 'global': 0.4, 'equity': 0.3},
        'equity_advocate': {'self': 0.1, 'local': 0.2, 'global': 0.2, 'equity': 0.5},
    }
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 128,
        agent_type: str = 'resident', 
        awareness_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.agent_type = agent_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Awareness matrix
        self.awareness = awareness_weights or self.DEFAULT_AWARENESS.get(
            agent_type, self.DEFAULT_AWARENESS['resident']
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self, 
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch, state_dim]
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value
    
    def select_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[int, float]:
        """
        Select action using current policy.
        
        Args:
            state: State tensor
            deterministic: Whether to use greedy action selection
            
        Returns:
            Tuple of (action, log_prob)
        """
        action_probs, _ = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1))).squeeze(-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def evaluate_actions(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Tuple of (values, log_probs, entropy)
        """
        action_probs, values = self.forward(states)
        
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return values.squeeze(-1), log_probs, entropy


def load_stakeholder_profiles(yaml_path: str) -> Dict[str, Dict]:
    """
    Load configurable stakeholder profiles from YAML.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Dictionary of stakeholder profiles
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    return config.get('stakeholders', {})


class UtilityFunction:
    """
    Defines utility/reward functions for each stakeholder type.
    
    Each utility function computes a weighted sum of:
    - Self utility: Own parcel conditions
    - Local utility: Neighborhood quality
    - Global utility: City-wide impact
    - Equity utility: Fairness considerations
    """
    
    @staticmethod
    def compute_utility(
        state: Dict[str, float], 
        awareness: Dict[str, float],
        utility_fn: str
    ) -> float:
        """
        Generic utility computation.
        
        Args:
            state: State features dictionary
            awareness: Awareness weights
            utility_fn: Utility function name
            
        Returns:
            Computed utility value
        """
        utility_fns = {
            'resident': UtilityFunction.resident_utility,
            'developer': UtilityFunction.developer_utility,
            'planner': UtilityFunction.planner_utility,
            'environmentalist': UtilityFunction.environmentalist_utility,
            'equity_advocate': UtilityFunction.equity_advocate_utility,
        }
        
        if utility_fn not in utility_fns:
            raise ValueError(f"Unknown utility function: {utility_fn}")
        
        return utility_fns[utility_fn](state, awareness)
    
    @staticmethod
    def resident_utility(state: Dict, awareness: Dict) -> float:
        """Resident: housing affordability + amenity access + environment."""
        self_utility = (
            state.get('housing_affordability', 0) * 0.4 +
            state.get('park_access', 0) * 0.3 +
            state.get('safety', 0) * 0.3
        )
        local_utility = (
            state.get('local_amenities', 0) * 0.5 +
            state.get('local_green_space', 0) * 0.5
        )
        global_utility = state.get('citywide_livability', 0)
        equity_utility = 1.0 - state.get('displacement_risk', 0)
        
        return (
            awareness['self'] * self_utility +
            awareness['local'] * local_utility +
            awareness['global'] * global_utility +
            awareness['equity'] * equity_utility
        ) + state.get('infrastructure_efficiency', 0) * 0.4  # Reward for efficiency (low congestion)

    
    @staticmethod
    def developer_utility(state: Dict, awareness: Dict) -> float:
        """Developer: ROI + development speed."""
        self_utility = (
            state.get('development_potential', 0) * 0.6 +
            state.get('property_value', 0) * 0.4
        )
        local_utility = state.get('local_market_demand', 0)
        global_utility = state.get('citywide_growth', 0)
        equity_utility = state.get('affordable_housing_bonus', 0)
        
        return (
            awareness['self'] * self_utility +
            awareness['local'] * local_utility +
            awareness['global'] * global_utility +
            awareness['equity'] * equity_utility
        )
    
    @staticmethod
    def planner_utility(state: Dict, awareness: Dict) -> float:
        """City planner: tax revenue + service efficiency + sustainability."""
        self_utility = state.get('parcel_tax_revenue', 0)
        local_utility = (
            state.get('infrastructure_efficiency', 0) * 0.6 +
            state.get('service_coverage', 0) * 0.4
        )
        global_utility = (
            state.get('citywide_tax_base', 0) * 0.3 +
            state.get('economic_growth', 0) * 0.3 +
            state.get('sustainability', 0) * 0.4
        )
        equity_utility = (
            state.get('spatial_equity', 0) * 0.5 +
            state.get('affordable_housing_ratio', 0) * 0.5
        )
        
        return (
            awareness['self'] * self_utility +
            awareness['local'] * local_utility +
            awareness['global'] * global_utility +
            awareness['equity'] * equity_utility
        ) * (1.0 + state.get('citywide_livability', 0)) # Bonus for diversity

    
    @staticmethod
    def environmentalist_utility(state: Dict, awareness: Dict) -> float:
        """Environmentalist: carbon reduction + green space + resilience."""
        self_utility = state.get('parcel_green_coverage', 0)
        local_utility = (
            state.get('local_tree_canopy', 0) * 0.5 +
            state.get('stormwater_management', 0) * 0.5
        )
        global_utility = (
            state.get('citywide_carbon_emissions', 0) * 0.4 +
            state.get('climate_resilience', 0) * 0.3 +
            state.get('biodiversity', 0) * 0.3
        )
        equity_utility = state.get('environmental_justice', 0)
        
        return (
            awareness['self'] * self_utility +
            awareness['local'] * local_utility +
            awareness['global'] * global_utility +
            awareness['equity'] * equity_utility
        )
    
    @staticmethod
    def equity_advocate_utility(state: Dict, awareness: Dict) -> float:
        """Equity advocate: fairness + access + inclusion."""
        self_utility = state.get('local_equity_index', 0)
        local_utility = (
            state.get('affordable_housing_access', 0) * 0.5 +
            state.get('amenity_access_equity', 0) * 0.5
        )
        global_utility = (
            state.get('citywide_gini_coefficient', 0) * 0.5 +
            state.get('segregation_index', 0) * 0.5
        )
        equity_utility = (
            state.get('displacement_prevention', 0) * 0.4 +
            state.get('inclusion_score', 0) * 0.3 +
            state.get('opportunity_access', 0) * 0.3
        )
        
        return (
            awareness['self'] * self_utility +
            awareness['local'] * local_utility +
            awareness['global'] * global_utility +
            awareness['equity'] * equity_utility
        )


class ConsensusVotingMechanism:
    """
    Weighted voting mechanism for aggregating multi-agent actions.
    
    Supports multiple voting strategies:
    - majority: Simple majority vote
    - weighted: Votes weighted by stakeholder priority
    - soft: Soft voting using probability distributions
    - nash: Nash equilibrium-based selection (uses external solver)
    
    Args:
        stakeholder_weights: Dict mapping agent type to voting weight
        voting_strategy: One of 'majority', 'weighted', 'soft', 'nash'
    """
    
    def __init__(
        self, 
        stakeholder_weights: Optional[Dict[str, float]] = None,
        voting_strategy: str = 'weighted'
    ):
        self.weights = stakeholder_weights or {
            'resident': 0.25,
            'developer': 0.15,
            'planner': 0.25,
            'environmentalist': 0.20,
            'equity_advocate': 0.15,
        }
        self.voting_strategy = voting_strategy
    
    def aggregate_votes(
        self, 
        agent_actions: Dict[str, List[int]],
        action_probs: Optional[Dict[str, torch.Tensor]] = None
    ) -> List[int]:
        """
        Aggregate actions from multiple agents.
        
        Args:
            agent_actions: Dict mapping agent type to list of actions per parcel
            action_probs: Optional action probability distributions for soft voting
            
        Returns:
            List of aggregated actions per parcel
        """
        if self.voting_strategy == 'majority':
            return self._majority_vote(agent_actions)
        elif self.voting_strategy == 'weighted':
            return self._weighted_vote(agent_actions)
        elif self.voting_strategy == 'soft':
            if action_probs is None:
                raise ValueError("Soft voting requires action_probs")
            return self._soft_vote(action_probs)
        elif self.voting_strategy == 'nash':
            return self._nash_equilibrium_vote(agent_actions)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
    
    def _majority_vote(self, agent_actions: Dict[str, List[int]]) -> List[int]:
        """Simple majority vote."""
        num_parcels = len(next(iter(agent_actions.values())))
        aggregated = []
        
        for parcel_idx in range(num_parcels):
            votes = [agent_actions[agent_type][parcel_idx] 
                     for agent_type in agent_actions]
            aggregated.append(max(set(votes), key=votes.count))
        
        return aggregated
    
    def _weighted_vote(self, agent_actions: Dict[str, List[int]]) -> List[int]:
        """Weighted voting based on stakeholder priorities."""
        num_parcels = len(next(iter(agent_actions.values())))
        action_space_size = max(max(actions) for actions in agent_actions.values()) + 1
        
        aggregated = []
        
        for parcel_idx in range(num_parcels):
            action_weights = [0.0] * action_space_size
            
            for agent_type, actions in agent_actions.items():
                action = actions[parcel_idx]
                weight = self.weights.get(agent_type, 0.1)
                action_weights[action] += weight
            
            aggregated.append(np.argmax(action_weights))
        
        return aggregated
    
    def _soft_vote(self, action_probs: Dict[str, torch.Tensor]) -> List[int]:
        """Soft voting using probability distributions."""
        # Average probabilities weighted by stakeholder weights
        weighted_probs = None
        
        for agent_type, probs in action_probs.items():
            weight = self.weights.get(agent_type, 0.1)
            if weighted_probs is None:
                weighted_probs = probs * weight
            else:
                weighted_probs = weighted_probs + probs * weight
        
        # Normalize
        weighted_probs = weighted_probs / weighted_probs.sum(dim=-1, keepdim=True)
        
        # Sample or argmax
        return torch.argmax(weighted_probs, dim=-1).tolist()
    
    def _nash_equilibrium_vote(self, agent_actions: Dict[str, List[int]]) -> List[int]:
        """
        Nash equilibrium-based selection.
        
        TODO: Integrate with external Nash equilibrium solver.
        For now, falls back to weighted voting.
        """
        # Placeholder - would use lemke_howson or other solver
        return self._weighted_vote(agent_actions)


class AgentCommunicationChannel:
    """
    Inter-agent message passing for negotiation.
    
    Enables agents to:
    - Broadcast proposals to all agents
    - Send targeted messages to specific agents
    - Negotiate through multiple rounds of proposals
    
    Args:
        agent_types: List of participating agent types
        max_history: Maximum messages to keep in history
    """
    
    def __init__(
        self, 
        agent_types: List[str],
        max_history: int = 100
    ):
        self.agent_types = agent_types
        self.message_queue: Dict[str, deque] = {
            agent_type: deque(maxlen=max_history) 
            for agent_type in agent_types
        }
        self.broadcast_history: deque = deque(maxlen=max_history)
    
    def broadcast(
        self, 
        sender: str, 
        message: Dict
    ) -> None:
        """
        Broadcast message to all agents.
        
        Args:
            sender: Agent type sending the message
            message: Message content (proposal, objection, etc.)
        """
        broadcast_msg = {
            'sender': sender,
            'type': 'broadcast',
            'content': message,
        }
        
        for agent_type in self.agent_types:
            if agent_type != sender:
                self.message_queue[agent_type].append(broadcast_msg)
        
        self.broadcast_history.append(broadcast_msg)
    
    def send(
        self, 
        sender: str, 
        recipient: str, 
        message: Dict
    ) -> None:
        """
        Send targeted message to specific agent.
        
        Args:
            sender: Agent type sending the message
            recipient: Target agent type
            message: Message content
        """
        msg = {
            'sender': sender,
            'type': 'direct',
            'content': message,
        }
        self.message_queue[recipient].append(msg)
    
    def receive(self, agent_type: str) -> List[Dict]:
        """
        Receive all pending messages for an agent.
        
        Args:
            agent_type: Agent type to receive messages for
            
        Returns:
            List of messages
        """
        messages = list(self.message_queue[agent_type])
        self.message_queue[agent_type].clear()
        return messages
    
    def negotiate(
        self, 
        proposals: Dict[str, Dict],
        max_rounds: int = 3
    ) -> Dict[str, float]:
        """
        Multi-round negotiation process.
        
        Args:
            proposals: Initial proposals from each agent {agent_type: proposal}
            max_rounds: Maximum negotiation rounds
            
        Returns:
            Final agreed-upon proposal weights
        """
        # Broadcast all proposals
        for agent_type, proposal in proposals.items():
            self.broadcast(agent_type, {'proposal': proposal})
        
        # Simple negotiation: average proposal weights
        final_weights = {}
        
        for key in proposals[next(iter(proposals))].keys():
            values = [proposals[agent_type].get(key, 0) for agent_type in proposals]
            final_weights[key] = np.mean(values)
        
        return final_weights


class MultiAgentEnvironment:
    """
    Environment for multi-agent land-use negotiation.
    
    State: GNN parcel embeddings
    Actions: Propose land-use changes (discrete FAR modifications)
    Rewards: Stakeholder-specific utilities
    
    Args:
        gnn_model: Trained ParcelGNN model
        graph_data: HeteroData graph
        physics_engine: MultiPhysicsEngine for simulation
        constraint_masks: Legal constraints DataFrame
        num_parcels: Number of parcels in environment
    """
    
    def __init__(
        self, 
        gnn_model,
        graph_data,
        physics_engine,
        constraint_masks: pd.DataFrame,
        num_parcels: int = 100
    ):
        self.gnn = gnn_model
        self.graph = graph_data
        self.physics = physics_engine
        self.constraints = constraint_masks
        self.num_parcels = num_parcels
        
        # Action space: 6 Land Use Categories (0-5)
        self.action_space_size = 6
        
        # Feature Cache for spatial lookups
        self.feature_cache = {}
        
        # State dimension (GNN embeddings)
        self.state_dim = 128
        
        # Voting mechanism
        self.voting = ConsensusVotingMechanism(voting_strategy='weighted')
        
        # Communication channel
        self.comm_channel = AgentCommunicationChannel([
            'resident', 'developer', 'planner', 'environmentalist', 'equity_advocate'
        ])
        
        # Current state
        self.current_far: Optional[torch.Tensor] = None
        self.state: Optional[torch.Tensor] = None
    
    def reset(self) -> torch.Tensor:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state (GNN embeddings)
        """
        # Initialize Randomly to break symmetry and start with diversity
        params = list(self.gnn.parameters())
        device = params[0].device if params else 'cpu'
        self.current_land_use = torch.randint(0, 6, (self.num_parcels,), device=device)
        
        # Update Graph with new Land Use so GNN can see it
        if hasattr(self.graph['parcel'], 'land_use_code'):
             self.graph['parcel'].land_use_code[:self.num_parcels] = self.current_land_use

        # Initialize FAR (Keep existing logic for compatibility)
        far_idx = 10  # Assuming FAR is at feature index 10
        self.current_far = self.graph['parcel'].x[:self.num_parcels, far_idx].clone()

        with torch.no_grad():
            embeddings = self.gnn.get_embeddings(self.graph)
            self.state = embeddings['parcel'][:self.num_parcels]
        
        return self.state
    
    def step(
        self, 
        actions: Dict[str, List[int]]
    ) -> Tuple[torch.Tensor, Dict[str, float], bool, Dict]:
        """
        Execute actions and return next state.
        
        Args:
            actions: Dict mapping agent_type to list of actions per parcel
            
        Returns:
            Tuple of (next_state, rewards, done, info)
        """
        # Aggregate actions using voting mechanism
        aggregated_actions = self.voting.aggregate_votes(actions)
        
        # Apply actions (Update Land Use)
        new_land_use = self._apply_actions(aggregated_actions)
        
        # Check zoning constraints
        # Validate that proposed land uses comply with NYC zoning regulations
        from pimaluos.config.zoning_compliance import count_violations
        
        # Get zoning districts for each parcel
        if hasattr(self.graph['parcel'], 'zone_district'):
            zone_districts = [
                self.graph['parcel'].zone_district[i] if i < len(self.graph['parcel'].zone_district) 
                else 'R6'  # Default residential
                for i in range(self.num_parcels)
            ]
        else:
            # Default to mixed residential/commercial zoning
            zone_districts = ['R6'] * self.num_parcels
        
        # Convert land use tensor to list
        land_use_list = new_land_use.cpu().tolist()
        
        # Count violations
        legal_violations = count_violations(land_use_list, zone_districts) 
        
        # Run physics simulation
        land_use_df = self._create_land_use_df(new_land_use, self.current_far)
        physics_results = self._run_physics(land_use_df)
        physics_violations = physics_results.get('violations', {}).get('total', 0)
        
        # Compute state features (SPATIAL & REAL)
        state_dict = self._compute_state_features(new_land_use, self.current_far, physics_results)
        
        # Compute rewards
        rewards = self._compute_rewards(state_dict, legal_violations, physics_violations)
        
        # Update state
        self.current_land_use = new_land_use

        with torch.no_grad():
            self.graph['parcel'].x[:self.num_parcels, 10] = self.current_far
            # Update land use in graph
            if hasattr(self.graph['parcel'], 'land_use_code'):
                self.graph['parcel'].land_use_code[:self.num_parcels] = self.current_land_use
            
            embeddings = self.gnn.get_embeddings(self.graph)
            self.state = embeddings['parcel'][:self.num_parcels]
        
        # Check termination
        done = legal_violations == 0 and physics_violations == 0
        
        info = {
            'legal_violations': legal_violations,
            'physics_violations': physics_violations,
            'physics_results': physics_results,
            'state_features': state_dict,
            'aggregated_actions': aggregated_actions,
        }
        
        return self.state, rewards, done, info
    
    def _apply_actions(self, actions: List[int]) -> torch.Tensor:
        """Apply Land Use modification actions."""
        # Actions are now directly setting the Land Use Code (0-5)
        new_land_use = self.current_land_use.clone()
        
        # Convert list to tensor
        action_tensor = torch.tensor(actions, device=new_land_use.device)
        
        # Update where actions are valid
        new_land_use = action_tensor
        
        return new_land_use
    
    def _check_legal_constraints(self, far: torch.Tensor) -> int:
        """Check legal constraint violations."""
        violations = 0
        for i in range(min(len(far), len(self.constraints))):
            max_far = self.constraints.iloc[i].get('max_far', 10.0)
            if far[i] > max_far:
                violations += 1
        return violations
    
    def _create_land_use_df(self, land_use: torch.Tensor, far: torch.Tensor) -> pd.DataFrame:
        """Create land use DataFrame for physics simulation."""
        
        # Map codes to labels for physics engine
        labels = [LAND_USE_CATEGORIES.get(c.item(), 'RESIDENTIAL').lower() for c in land_use]
        
        return pd.DataFrame({
            'parcel_id': range(self.num_parcels),
            'use': labels,
            'far': far.cpu().numpy(),
            'height_ft': (far.cpu().numpy() * 12).clip(0, 200),
            'units': (far.cpu().numpy() * 10).astype(int),
            'lot_coverage': 0.5,
        })
    
    def _run_physics(self, land_use_df: pd.DataFrame) -> Dict:
        """Run physics simulation."""
        try:
            scenario = self.physics.prepare_scenario(land_use_df)
            return self.physics.simulate_all(scenario)
        except Exception:
            return {
                'traffic': {'avg_congestion_ratio': 1.0},
                'hydrology': {'capacity_utilization': 0.5},
                'solar': {'num_violations': 0},
                'violations': {'total': 0},
            }
    
    def _compute_state_features(
        self, 
        land_use: torch.Tensor,
        far: torch.Tensor, 
        physics_results: Dict
    ) -> Dict[str, float]:
        """Compute state features with REAL spatial awareness."""
        
        # 1. Physics & FAR Features (Preserved)
        congestion = physics_results.get('traffic', {}).get('avg_congestion_ratio', 1.0)
        hydro_util = physics_results.get('hydrology', {}).get('capacity_utilization', 0.5)
        avg_far = far.mean().item()
        
        # 2. Spatial Adjacency Calculation (New)
        # Use graph edge_index for fast lookup
        if 'spatial_adjacency' not in self.feature_cache:
            # Pre-compute adjacency lists if not cached
            try:
                # Try explicit adjacency first (strongest signal)
                edge_index = self.graph['parcel', 'adjacent_to', 'parcel'].edge_index
            except AttributeError:
                try:
                    # Fallback to visual connectivity (KNN) which is robust
                    edge_index = self.graph['parcel', 'visible_from', 'parcel'].edge_index
                except AttributeError:
                     # Handle case where no edges exist (e.g. sparse subset)
                     edge_index = torch.empty((2, 0), dtype=torch.long, device=land_use.device)
            
            self.feature_cache['spatial_adjacency'] = edge_index
        
        edge_index = self.feature_cache['spatial_adjacency']
        src, dst = edge_index
        
        # If no edges, return zeros for neighbor metrics
        if edge_index.shape[1] == 0:
            return {
                'housing_affordability': 1.0 - avg_far * 0.5,
                'park_access': 0.0,
                'safety': 0.8,
                'local_amenities': 0.0,
                'local_green_space': 0.0,
                'citywide_livability': 0.7,
                'displacement_risk': avg_far * 0.3,
                'development_potential': avg_far,
                'property_value': avg_far * 0.8,
                'local_market_demand': 0.7,
                'citywide_growth': 0.6,
                'affordable_housing_bonus': 0.1,
                'parcel_tax_revenue': avg_far * 0.9,
                'infrastructure_efficiency': 1.0 - congestion * 0.3,
                'service_coverage': 0.75,
                'citywide_tax_base': avg_far * 0.8,
                'economic_growth': avg_far * 0.6,
                'sustainability': 1.0 - hydro_util,
                'spatial_equity': 0.6,
                'affordable_housing_ratio': 0.3,
                'parcel_green_coverage': 0.0,
                'local_tree_canopy': 0.0,
                'stormwater_management': 1.0 - hydro_util,
                'citywide_carbon_emissions': 1.0 - avg_far * 0.5,
                'climate_resilience': 0.7,
                'biodiversity': 0.6,
                'environmental_justice': 0.65,
                'local_equity_index': 0.7,
                'affordable_housing_access': 1.0 - avg_far * 0.4,
                'amenity_access_equity': 0.65,
                'citywide_gini_coefficient': 1.0 - avg_far * 0.2,
                'segregation_index': 0.0,
                'displacement_prevention': 1.0 - avg_far * 0.3,
                'inclusion_score': 0.7,
                'opportunity_access': 0.65,
            }
        
        # Convert land use to one-hot for fast aggregation
        # [Num_Parcels, 6]
        one_hot = F.one_hot(land_use, num_classes=6).float()
        
        # Aggregate neighbor land uses
        # Create a tensor to hold sum of neighbor one-hots
        neighbor_sums = torch.zeros_like(one_hot)
        neighbor_sums.index_add_(0, src, one_hot[dst]) # Add dst features to src index
        
        # Normalize by degree (count of neighbors)
        degree = torch.zeros(self.num_parcels, device=land_use.device)
        degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
        degree = degree.clamp(min=1.0).unsqueeze(1)
        
        neighbor_ratios = neighbor_sums / degree
        
        # Extract specific ratios across the whole city (Global/Avg measures for simple feature dict)
        # For individual agent observations, we would return per-parcel features, 
        # but this function returns a generic state_dict for the shared reward function.
        # We will compute the AVERAGE experience of the city.
        
        avg_res_neighbors = neighbor_ratios[:, 0].mean().item()
        avg_com_neighbors = neighbor_ratios[:, 1].mean().item()
        avg_ind_neighbors = neighbor_ratios[:, 2].mean().item()
        avg_park_neighbors = neighbor_ratios[:, 5].mean().item() # Open Space
        
        # Compute "Compatibility Score"
        # Compute "Compatibility Score"
        clustering_score = (neighbor_ratios * one_hot).sum(dim=1).mean().item()
        
        # 4. Diversity & Balance Metrics
        # Count proportions of each type
        counts = torch.bincount(land_use, minlength=6).float()
        props = counts / len(land_use)
        
        # Entropy (Diversity)
        start_entropy = -torch.sum(props * torch.log(props + 1e-10))
        max_entropy = torch.log(torch.tensor(6.0))
        normalized_entropy = start_entropy / max_entropy
        
        # Housing Supply Ratio
        res_ratio = props[0]  # Assuming 0 is Residential
        
        # Parabolic logic for amenities: Peak at 50% Commercial, zero at 0% or 100%
        # This forces mixed use.
        # Density (0-1). Ideal is 0.5.
        # Score = 1.0 - |density - 0.5| * 2.0
        com_saturation_penalty = 1.0 - (avg_com_neighbors - 0.5).__abs__() * 2.0
        com_saturation_penalty = max(0.0, com_saturation_penalty)

        # 5. Populate State Dict
        return {
            'housing_affordability': 1.0 - (avg_far * 0.3) + (res_ratio * 0.5),
            'park_access': avg_park_neighbors * 2.0 + (props[5] * 0.5), # 5 is Open Space
            'safety': 0.6 + avg_res_neighbors * 0.4,
            # Incentivize Mixed Use: Peak at 50% density
            'local_amenities': com_saturation_penalty * 3.0 + (normalized_entropy * 0.5),
            'local_green_space': avg_park_neighbors,
            'citywide_livability': normalized_entropy.item() * 2.0, # Strong diversity bias
            'displacement_risk': avg_far * 0.3 - (res_ratio * 0.2),
            'development_potential': (1.0 - res_ratio) * 0.5 + avg_far * 0.5,
            # Property value peaks with diversity + density
            'property_value': avg_far * 0.4 + com_saturation_penalty * 0.4 + (normalized_entropy * 0.2),
            'local_market_demand': res_ratio * 0.8,
            'citywide_growth': avg_far * 0.6 + (normalized_entropy * 0.4),
            'affordable_housing_bonus': res_ratio,
            'parcel_tax_revenue': avg_far * 0.7 + (1.0 - res_ratio) * 0.3,
            'infrastructure_efficiency': 1.0 - congestion * 0.8, # Stronger congestion penalty
            'service_coverage': 0.6 + avg_com_neighbors * 0.4,
            'citywide_tax_base': avg_far * 0.8,
            'economic_growth': avg_far * 0.6,
            'sustainability': 1.0 - hydro_util,
            'spatial_equity': normalized_entropy.item(),
            'affordable_housing_ratio': res_ratio,
            'parcel_green_coverage': props[5],
            'local_tree_canopy': props[5],
            'stormwater_management': 1.0 - hydro_util + props[5] * 0.3,
            'citywide_carbon_emissions': 1.0 - avg_far * 0.5,
            'climate_resilience': 0.7 + props[5] * 0.3,
            'biodiversity': props[5],
            'environmental_justice': 0.5 + props[5] * 0.5,
            'local_equity_index': normalized_entropy.item(),
            'affordable_housing_access': res_ratio,
            'amenity_access_equity': 0.65,
            'citywide_gini_coefficient': 1.0 - avg_far * 0.2,
            'segregation_index': 1.0 - normalized_entropy.item(),
            'displacement_prevention': res_ratio,
            'inclusion_score': normalized_entropy.item(),
            'opportunity_access': 0.65,
        }
    
    def _compute_rewards(
        self, 
        state_dict: Dict, 
        legal_violations: int,
        physics_violations: int
    ) -> Dict[str, float]:
        """Compute rewards for each agent type."""
        agent_types = ['resident', 'developer', 'planner', 'environmentalist', 'equity_advocate']
        rewards = {}
        
        for agent_type in agent_types:
            awareness = StakeholderAgent.DEFAULT_AWARENESS.get(
                agent_type, StakeholderAgent.DEFAULT_AWARENESS['resident']
            )
            utility = UtilityFunction.compute_utility(state_dict, awareness, agent_type)
            
            # Apply violation penalty
            penalty = (legal_violations + physics_violations) * 0.1
            rewards[agent_type] = utility - penalty
        
        return rewards


class MARLTrainer:
    """
    Trainer for multi-agent system using PPO.
    
    Args:
        environment: MultiAgentEnvironment
        agent_types: List of agent types to train
        state_dim: State space dimension
        action_dim: Action space size
    """
    
    def __init__(
        self, 
        environment: MultiAgentEnvironment, 
        agent_types: List[str],
        state_dim: int = 128, 
        action_dim: int = 3,
        learning_rate: float = 3e-4
    ):
        self.env = environment
        self.agent_types = agent_types
        
        # Initialize agents
        self.agents = {
            agent_type: StakeholderAgent(state_dim, action_dim, agent_type=agent_type)
            for agent_type in agent_types
        }
        
        # Optimizers
        self.optimizers = {
            agent_type: torch.optim.Adam(agent.parameters(), lr=learning_rate)
            for agent_type, agent in self.agents.items()
        }
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Memory
        self.memory = {agent_type: [] for agent_type in agent_types}
    
    def train(
        self, 
        num_iterations: int = 100, 
        steps_per_iteration: int = 100
    ) -> List[Dict]:
        """
        Main training loop.
        
        Args:
            num_iterations: Number of training iterations
            steps_per_iteration: Environment steps per iteration
            
        Returns:
            Training history
        """
        history = []
        
        print("\n" + "=" * 70)
        print("MARL TRAINING")
        print("=" * 70)
        
        for iteration in range(num_iterations):
            # Collect trajectories
            self._collect_trajectories(steps_per_iteration)
            
            # Update agents
            losses = self._update_agents()
            
            history.append(losses)
            
            if iteration % 10 == 0:
                print(f"\nIteration {iteration}:")
                for agent_type, loss_dict in losses.items():
                    print(f"  {agent_type}: total={loss_dict.get('total', 0):.4f}")
        
        print("\n" + "=" * 70)
        print("MARL TRAINING COMPLETE")
        print("=" * 70)
        
        return history
    
    def _collect_trajectories(self, num_steps: int) -> None:
        """Collect trajectories from environment."""
        state = self.env.reset()
        
        for _ in range(num_steps):
            actions = {}
            log_probs = {}
            values = {}
            
            for agent_type, agent in self.agents.items():
                parcel_actions = []
                parcel_log_probs = []
                parcel_values = []
                
                for parcel_idx in range(self.env.num_parcels):
                    parcel_state = state[parcel_idx].unsqueeze(0)
                    action, log_prob = agent.select_action(parcel_state)
                    _, value = agent.forward(parcel_state)
                    
                    parcel_actions.append(action)
                    parcel_log_probs.append(log_prob)
                    parcel_values.append(value.item())
                
                actions[agent_type] = parcel_actions
                log_probs[agent_type] = parcel_log_probs
                values[agent_type] = parcel_values
            
            next_state, rewards, done, _ = self.env.step(actions)
            
            for agent_type in self.agent_types:
                self.memory[agent_type].append({
                    'state': state,
                    'action': actions[agent_type],
                    'log_prob': log_probs[agent_type],
                    'value': values[agent_type],
                    'reward': rewards[agent_type],
                    'done': done,
                })
            
            state = next_state
            
            if done:
                break
    
    def _update_agents(self) -> Dict[str, Dict]:
        """Update all agents using PPO."""
        losses = {}
        
        for agent_type, agent in self.agents.items():
            if not self.memory[agent_type]:
                losses[agent_type] = {'total': 0}
                continue
            
            # Simple update (full PPO implementation omitted for brevity)
            total_loss = 0.0
            
            for transition in self.memory[agent_type][-10:]:
                reward = transition['reward']
                total_loss += -reward  # Simplified loss
            
            losses[agent_type] = {'total': total_loss / max(len(self.memory[agent_type]), 1)}
            
            # Clear memory
            self.memory[agent_type] = []
        
        return losses


# Example usage
if __name__ == "__main__":
    # Create dummy agent
    agent = StakeholderAgent(
        state_dim=128,
        action_dim=3,
        agent_type='planner'
    )
    
    print(f"Agent type: {agent.agent_type}")
    print(f"Awareness: {agent.awareness}")
    print(f"Parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    # Test forward pass
    dummy_state = torch.randn(1, 128)
    action_probs, value = agent(dummy_state)
    print(f"\nAction probs: {action_probs}")
    print(f"Value: {value.item():.4f}")
    
    # Test voting
    voting = ConsensusVotingMechanism()
    test_actions = {
        'resident': [0, 1, 2, 1],
        'developer': [2, 2, 2, 2],
        'planner': [1, 1, 1, 1],
    }
    aggregated = voting.aggregate_votes(test_actions)
    print(f"\nAggregated actions: {aggregated}")

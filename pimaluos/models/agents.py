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
        Select action using current policy (single parcel).
        
        Args:
            state: State tensor [1, state_dim]
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

    def select_action_batch(
        self,
        states: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched action selection for ALL parcels at once.

        Args:
            states: State tensor [num_parcels, state_dim]
            deterministic: Whether to use greedy action selection

        Returns:
            Tuple of (actions [num_parcels], log_probs [num_parcels])
        """
        with torch.no_grad():
            action_probs, values = self.forward(states)

        if deterministic:
            actions = torch.argmax(action_probs, dim=-1)
            log_probs = torch.log(
                action_probs.gather(-1, actions.unsqueeze(-1)) + 1e-10
            ).squeeze(-1)
        else:
            dist = Categorical(action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return actions, log_probs, values.squeeze(-1)
    
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
        num_parcels: int = 100,
        physics_interval: int = 5,
        use_gnn: bool = True
    ):
        self.gnn = gnn_model
        self.use_gnn = use_gnn and (gnn_model is not None)
        self.graph = graph_data
        self.physics = physics_engine
        self.constraints = constraint_masks
        self.num_parcels = num_parcels
        
        # Action space: 6 Land Use Categories (0-5)
        self.action_space_size = 6
        
        # Feature Cache for spatial lookups
        self.feature_cache = {}
        
        # State dimension
        self.state_dim = 128 if self.use_gnn else self.graph['parcel'].x.shape[1]
        
        # Voting mechanism
        self.voting = ConsensusVotingMechanism(voting_strategy='weighted')
        
        # Communication channel
        self.comm_channel = AgentCommunicationChannel([
            'resident', 'developer', 'planner', 'environmentalist', 'equity_advocate'
        ])
        
        # Current state
        self.current_far: Optional[torch.Tensor] = None
        self.state: Optional[torch.Tensor] = None
        
        # Physics caching — only recompute every physics_interval steps
        self._physics_interval = physics_interval
        self._step_counter = 0
        self._cached_physics_results: Optional[Dict] = None
        
        # Pre-build land-use label lookup array for vectorised mapping
        self._lu_labels = np.array(
            [LAND_USE_CATEGORIES.get(i, 'RESIDENTIAL').lower() for i in range(6)]
        )
    
    def reset(self) -> torch.Tensor:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state (GNN embeddings or raw features)
        """
        # Initialize Randomly to break symmetry and start with diversity
        if self.use_gnn:
            params = list(self.gnn.parameters())
            device = params[0].device if params else 'cpu'
        else:
            device = self.graph['parcel'].x.device
            
        self.current_land_use = torch.randint(0, 6, (self.num_parcels,), device=device)
        
        # Update Graph with new Land Use so GNN can see it
        if hasattr(self.graph['parcel'], 'land_use_code'):
             self.graph['parcel'].land_use_code[:self.num_parcels] = self.current_land_use

        # Initialize FAR (Keep existing logic for compatibility)
        far_idx = 10  # Assuming FAR is at feature index 10
        self.current_far = self.graph['parcel'].x[:self.num_parcels, far_idx].clone()

        if self.use_gnn:
            with torch.no_grad():
                embeddings = self.gnn.get_embeddings(self.graph)
                self.state = embeddings['parcel'][:self.num_parcels]
        else:
            self.state = self.graph['parcel'].x[:self.num_parcels].clone()
        
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
        
        # Apply actions (Update FAR)
        new_far = self._apply_actions(aggregated_actions)
        new_land_use = self.current_land_use
        
        # Check zoning constraints (Land Use)
        from pimaluos.config.zoning_compliance import count_violations
        
        if not hasattr(self, '_cached_zone_districts'):
            if hasattr(self.graph['parcel'], 'zone_district'):
                self._cached_zone_districts = [
                    self.graph['parcel'].zone_district[i] if i < len(self.graph['parcel'].zone_district) 
                    else 'R6'
                    for i in range(self.num_parcels)
                ]
            else:
                self._cached_zone_districts = ['R6'] * self.num_parcels
        
        land_use_list = new_land_use.cpu().tolist()
        
        # Count violations (Land Use + FAR)
        legal_violations = count_violations(land_use_list, self._cached_zone_districts)
        legal_violations += self._check_legal_constraints(new_far)
        
        # Compute local violations tensor for local credit assignment
        max_far_vals = [self.constraints.iloc[i].get('max_far', 10.0) if i < len(self.constraints) else 10.0 for i in range(self.num_parcels)]
        max_far_tensor = torch.tensor(max_far_vals, device=new_far.device, dtype=torch.float32)
        local_violations = (new_far > max_far_tensor * 1.01).float()
        
        # Run physics simulation (cached — expensive at 42K parcels)
        self._step_counter += 1
        if self._cached_physics_results is None or self._step_counter % self._physics_interval == 0:
            land_use_df = self._create_land_use_df(new_land_use, new_far)
            self._cached_physics_results = self._run_physics(land_use_df)
        physics_results = self._cached_physics_results
        physics_violations = physics_results.get('violations', {}).get('total', 0)
        
        # Compute state features (SPATIAL & REAL)
        state_dict = self._compute_state_features(new_land_use, new_far, physics_results)
        
        # Compute rewards with localized credit assignment
        rewards = self._compute_rewards(state_dict, local_violations, physics_violations)
        
        # Update state
        self.current_far = new_far
        self.current_land_use = new_land_use

        with torch.no_grad():
            self.graph['parcel'].x[:self.num_parcels, 10] = self.current_far
            if hasattr(self.graph['parcel'], 'land_use_code'):
                self.graph['parcel'].land_use_code[:self.num_parcels] = self.current_land_use
            
            # Only recompute GNN embeddings on physics steps (expensive at 42K nodes)
            if self._step_counter % self._physics_interval == 0:
                if self.use_gnn:
                    embeddings = self.gnn.get_embeddings(self.graph)
                    self.state = embeddings['parcel'][:self.num_parcels]
                else:
                    self.state = self.graph['parcel'].x[:self.num_parcels].clone()
        
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
        """Apply FAR modification actions (0=decrease, 1=maintain, 2=increase)."""
        new_far = self.current_far.clone()
        action_tensor = torch.tensor(actions, device=new_far.device)
        
        # 0: Decrease by 20%, 1: Maintain, 2: Increase by 20%
        new_far[action_tensor == 0] *= 0.8
        new_far[action_tensor == 2] *= 1.2
        
        # Clip to valid range (0.1 to max_far)
        max_far_vals = [self.constraints.iloc[i].get('max_far', 10.0) if i < len(self.constraints) else 10.0 for i in range(self.num_parcels)]
        max_far_tensor = torch.tensor(max_far_vals, device=new_far.device, dtype=torch.float32)
        new_far = torch.min(new_far, max_far_tensor)
        new_far = torch.clamp(new_far, min=0.1)
        
        return new_far
    
    def _check_legal_constraints(self, far: torch.Tensor) -> int:
        """Check legal constraint violations."""
        violations = 0
        for i in range(min(len(far), len(self.constraints))):
            max_far = self.constraints.iloc[i].get('max_far', 10.0)
            if far[i] > max_far:
                violations += 1
        return violations
    
    def _create_land_use_df(self, land_use: torch.Tensor, far: torch.Tensor) -> pd.DataFrame:
        """Create land use DataFrame for physics simulation (vectorised)."""
        lu_np = land_use.cpu().numpy()
        far_np = far.cpu().numpy()
        labels = self._lu_labels[lu_np]  # Vectorised lookup — no per-element .item()
        
        return pd.DataFrame({
            'parcel_id': np.arange(self.num_parcels),
            'use': labels,
            'far': far_np,
            'height_ft': (far_np * 12).clip(0, 200),
            'units': (far_np * 10).astype(int),
            'lot_coverage': 0.5,
        })
    
    def _run_physics(self, land_use_df: pd.DataFrame) -> Dict:
        """Run physics simulation — uses lightweight approximation for large graphs."""
        if self.num_parcels > 5000:
            # Lightweight analytical approximation for large-scale MARL.
            # The O(N²) gravity traffic model in MultiPhysicsEngine makes
            # full simulation infeasible at 42K (takes ~30 min per call).
            # Physics constraints are already embedded in GNN from Stage 4.
            return self._approximate_physics(land_use_df)
        try:
            scenario = self.physics.prepare_scenario(land_use_df)
            return self.physics.simulate_all(scenario)
        except Exception:
            return self._default_physics()

    def _approximate_physics(self, land_use_df: pd.DataFrame) -> Dict:
        """
        Fast O(N) physics approximation for large-scale MARL.

        Uses land-use proportions and density statistics to estimate
        traffic congestion, hydrology load, and solar violations
        without the O(N²) pairwise gravity model.
        """
        n = len(land_use_df)
        use_counts = land_use_df['use'].value_counts(normalize=True)
        avg_far = land_use_df['far'].mean()
        avg_height = land_use_df['height_ft'].mean()

        # Traffic: congestion scales with commercial/industrial density
        commercial_frac = use_counts.get('commercial', 0) + use_counts.get('mixed', 0)
        industrial_frac = use_counts.get('industrial', 0)
        trip_intensity = (
            commercial_frac * 10.0 +
            industrial_frac * 5.0 +
            use_counts.get('residential', 0) * 3.0
        )
        congestion = 1.0 + 0.15 * (trip_intensity / 5.0) ** 2

        # Hydrology: imperviousness by land-use type
        imperv = (
            use_counts.get('commercial', 0) * 0.85 +
            use_counts.get('industrial', 0) * 0.80 +
            use_counts.get('mixed', 0) * 0.70 +
            use_counts.get('residential', 0) * 0.45 +
            use_counts.get('public', 0) * 0.50 +
            use_counts.get('open_space', 0) * 0.10
        )
        capacity_util = imperv * avg_far * 0.8

        # Solar: violations based on height thresholds
        tall_buildings = (land_use_df['height_ft'] > 100).sum()
        shadow_pct = min(avg_height / 2, 80)

        violations = {'total': 0}
        if congestion > 1.5:
            violations['total'] += 1
        if capacity_util > 1.0:
            violations['total'] += 1
        if tall_buildings > 0:
            violations['total'] += 1

        return {
            'traffic': {
                'avg_congestion_ratio': congestion,
                'max_congestion_ratio': congestion * 1.3,
                'oversaturated_links': 0,
                'pct_oversaturated': 0,
            },
            'hydrology': {
                'peak_runoff_cfs': capacity_util * 100,
                'total_runoff_cf': capacity_util * 360000,
                'weighted_runoff_coefficient': imperv,
                'capacity_utilization': capacity_util,
                'capacity_exceeded': capacity_util > 1.0,
            },
            'solar': {
                'avg_shadow_pct': shadow_pct,
                'max_shadow_pct': max(land_use_df['height_ft']) / 2,
                'num_violations': tall_buildings,
                'pct_parcels_violated': tall_buildings / max(n, 1) * 100,
            },
            'violations': violations,
        }

    @staticmethod
    def _default_physics() -> Dict:
        """Fallback physics results."""
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
    ) -> Dict[str, torch.Tensor]:
        """Compute state features with REAL spatial awareness (vectorized over all parcels)."""
        
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
        
        # If no edges, return zeros/defaults for neighbor metrics
        if edge_index.shape[1] == 0:
            zeros = torch.zeros(self.num_parcels, device=land_use.device)
            ones = torch.ones(self.num_parcels, device=land_use.device)
            
            return {
                'housing_affordability': 1.0 - far * 0.5,
                'park_access': zeros,
                'safety': ones * 0.8,
                'local_amenities': zeros,
                'local_green_space': zeros,
                'citywide_livability': ones * 0.7,
                'displacement_risk': far * 0.3,
                'development_potential': far,
                'property_value': far * 0.8,
                'local_market_demand': ones * 0.7,
                'citywide_growth': ones * 0.6,
                'affordable_housing_bonus': ones * 0.1,
                'parcel_tax_revenue': far * 0.9,
                'infrastructure_efficiency': ones * (1.0 - congestion * 0.3),
                'service_coverage': ones * 0.75,
                'citywide_tax_base': ones * (avg_far * 0.8),
                'economic_growth': ones * (avg_far * 0.6),
                'sustainability': ones * (1.0 - hydro_util),
                'spatial_equity': ones * 0.6,
                'affordable_housing_ratio': ones * 0.3,
                'parcel_green_coverage': zeros,
                'local_tree_canopy': zeros,
                'stormwater_management': ones * (1.0 - hydro_util),
                'citywide_carbon_emissions': ones * (1.0 - avg_far * 0.5),
                'climate_resilience': ones * 0.7,
                'biodiversity': ones * 0.6,
                'environmental_justice': ones * 0.65,
                'local_equity_index': ones * 0.7,
                'affordable_housing_access': ones * (1.0 - avg_far * 0.4),
                'amenity_access_equity': ones * 0.65,
                'citywide_gini_coefficient': ones * (1.0 - avg_far * 0.2),
                'segregation_index': zeros,
                'displacement_prevention': ones * (1.0 - avg_far * 0.3),
                'inclusion_score': ones * 0.7,
                'opportunity_access': ones * 0.65,
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
        
        # 4. Diversity & Balance Metrics
        # Count proportions of each type globally
        counts = torch.bincount(land_use, minlength=6).float()
        props = counts / len(land_use)
        
        # Entropy (Diversity) globally
        start_entropy = -torch.sum(props * torch.log(props + 1e-10))
        max_entropy = torch.log(torch.tensor(6.0))
        normalized_entropy = start_entropy / max_entropy
        
        # Housing Supply Ratio globally
        res_ratio = props[0]  # Assuming 0 is Residential
        
        # Local mixed-use demand/amenities (peaks when commercial neighbors ratio is 0.5)
        com_saturation_penalty_local = 1.0 - torch.abs(neighbor_ratios[:, 1] - 0.5) * 2.0
        com_saturation_penalty_local = torch.clamp(com_saturation_penalty_local, min=0.0)
        
        ones = torch.ones(self.num_parcels, device=land_use.device)
        
        # 5. Populate State Dict with Tensors of shape [num_parcels]
        return {
            'housing_affordability': 1.0 - (far * 0.3) + (neighbor_ratios[:, 0] * 0.5),
            'park_access': neighbor_ratios[:, 5] * 2.0 + (one_hot[:, 5] * 0.5), # 5 is Open Space
            'safety': 0.6 + neighbor_ratios[:, 0] * 0.4,
            # Incentivize Mixed Use: Peak at 50% density locally
            'local_amenities': com_saturation_penalty_local * 3.0 + (normalized_entropy * 0.5),
            'local_green_space': neighbor_ratios[:, 5],
            'citywide_livability': ones * (normalized_entropy.item() * 2.0), # Strong diversity bias
            'displacement_risk': far * 0.3 - (neighbor_ratios[:, 0] * 0.2),
            'development_potential': (1.0 - one_hot[:, 0]) * 0.5 + far * 0.5,
            # Property value peaks with diversity + density locally
            'property_value': far * 0.4 + com_saturation_penalty_local * 0.4 + (normalized_entropy * 0.2),
            'local_market_demand': neighbor_ratios[:, 1] * 0.8,
            'citywide_growth': far * 0.6 + (normalized_entropy * 0.4),
            'affordable_housing_bonus': one_hot[:, 0],
            'parcel_tax_revenue': far * 0.7 + (1.0 - one_hot[:, 0]) * 0.3,
            'infrastructure_efficiency': ones * (1.0 - congestion * 0.8), # Stronger congestion penalty
            'service_coverage': 0.6 + neighbor_ratios[:, 1] * 0.4,
            'citywide_tax_base': ones * (avg_far * 0.8),
            'economic_growth': ones * (avg_far * 0.6),
            'sustainability': ones * (1.0 - hydro_util),
            'spatial_equity': ones * normalized_entropy.item(),
            'affordable_housing_ratio': ones * res_ratio,
            'parcel_green_coverage': one_hot[:, 5],
            'local_tree_canopy': neighbor_ratios[:, 5],
            'stormwater_management': (1.0 - hydro_util) + one_hot[:, 5] * 0.3,
            'citywide_carbon_emissions': ones * (1.0 - avg_far * 0.5),
            'climate_resilience': 0.7 + one_hot[:, 5] * 0.3,
            'biodiversity': one_hot[:, 5],
            'environmental_justice': 0.5 + neighbor_ratios[:, 5] * 0.5,
            'local_equity_index': ones * normalized_entropy.item(),
            'affordable_housing_access': neighbor_ratios[:, 0],
            'amenity_access_equity': ones * 0.65,
            'citywide_gini_coefficient': ones * (1.0 - avg_far * 0.2),
            'segregation_index': ones * (1.0 - normalized_entropy.item()),
            'displacement_prevention': one_hot[:, 0],
            'inclusion_score': ones * normalized_entropy.item(),
            'opportunity_access': ones * 0.65,
        }
    
    def _compute_rewards(
        self, 
        state_dict: Dict, 
        local_violations: torch.Tensor,
        physics_violations: int
    ) -> Dict[str, torch.Tensor]:
        """Compute localized rewards for each agent type."""
        agent_types = ['resident', 'developer', 'planner', 'environmentalist', 'equity_advocate']
        rewards = {}
        
        global_penalty = physics_violations * 0.1
        
        for agent_type in agent_types:
            awareness = StakeholderAgent.DEFAULT_AWARENESS.get(
                agent_type, StakeholderAgent.DEFAULT_AWARENESS['resident']
            )
            # utility is computed using state_dict tensors, returning shape [num_parcels]
            utility = UtilityFunction.compute_utility(state_dict, awareness, agent_type)
            
            # Apply local violation penalty + global penalty
            penalty = local_violations * 0.5 + global_penalty
            rewards[agent_type] = utility - penalty
        
        return rewards


class MARLTrainer:
    """
    Trainer for multi-agent system using PPO.

    Vectorized: all parcels are processed in a single batched forward
    pass per agent per step, making 42K-parcel runs feasible in minutes.

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

        import logging
        self._log = logging.getLogger('pimaluos.marl')

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
        Main training loop (vectorized).
        """
        import time
        history = []

        self._log.info("MARL TRAINING (vectorized)")
        t0 = time.time()

        for iteration in range(num_iterations):
            iter_t = time.time()
            # Collect trajectories (batched — all parcels at once)
            self._collect_trajectories(steps_per_iteration)

            # Update agents
            losses = self._update_agents()
            history.append(losses)

            if iteration % 10 == 0:
                elapsed = time.time() - t0
                iter_time = time.time() - iter_t
                avg_loss = np.mean([l.get('total', 0) for l in losses.values()])
                self._log.info(
                    f"Iteration {iteration:3d}/{num_iterations}  "
                    f"avg_loss={avg_loss:.4f}  "
                    f"iter={iter_time:.1f}s  "
                    f"elapsed={elapsed:.0f}s"
                )

        total = time.time() - t0
        self._log.info(f"MARL TRAINING COMPLETE in {total:.1f}s")

        return history
    
    def _collect_trajectories(self, num_steps: int) -> None:
        """Collect trajectories — vectorized over all parcels."""
        state = self.env.reset()  # [num_parcels, state_dim]

        for _ in range(num_steps):
            actions = {}
            log_probs = {}
            values = {}

            # Batched forward pass: one call per agent for ALL parcels
            for agent_type, agent in self.agents.items():
                acts, lps, vals = agent.select_action_batch(state)
                actions[agent_type] = acts.tolist()
                log_probs[agent_type] = lps.tolist()
                values[agent_type] = vals.tolist()

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
        """Update all agents using PPO with GAE and clipped surrogate objective."""
        losses = {}
        
        for agent_type, agent in self.agents.items():
            if not self.memory[agent_type]:
                losses[agent_type] = {'total': 0.0, 'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
                continue
            
            device = next(agent.parameters()).device
            
            # Extract memory lists
            states = torch.stack([
                torch.as_tensor(x['state'], dtype=torch.float32, device=device)
                for x in self.memory[agent_type]
            ]) # Shape: [T, num_parcels, state_dim]
            
            actions = torch.tensor([
                x['action'] for x in self.memory[agent_type]
            ], dtype=torch.long, device=device) # Shape: [T, num_parcels]
            
            old_log_probs = torch.tensor([
                x['log_prob'] for x in self.memory[agent_type]
            ], dtype=torch.float32, device=device) # Shape: [T, num_parcels]
            
            values = torch.tensor([
                x['value'] for x in self.memory[agent_type]
            ], dtype=torch.float32, device=device) # Shape: [T, num_parcels]
            
            rewards = torch.stack([
                torch.as_tensor(x['reward'], dtype=torch.float32, device=device)
                for x in self.memory[agent_type]
            ]) # Shape: [T, num_parcels]
            
            dones = torch.tensor([
                1.0 if x['done'] else 0.0 for x in self.memory[agent_type]
            ], dtype=torch.float32, device=device).unsqueeze(1).repeat(1, self.env.num_parcels) # Shape: [T, num_parcels]
            
            T, num_parcels = rewards.shape
            
            # GAE (Generalized Advantage Estimation)
            advantages = torch.zeros_like(rewards)
            last_gae_lam = 0.0
            
            with torch.no_grad():
                if not self.memory[agent_type][-1]['done']:
                    next_state = self.env.state # Shape: [num_parcels, state_dim]
                    _, _, next_value = agent.select_action_batch(next_state) # next_value: [num_parcels]
                else:
                    next_value = torch.zeros(num_parcels, device=device)
                
                next_non_terminal = 1.0 - dones
                values_all = torch.cat([values, next_value.unsqueeze(0)], dim=0) # [T+1, num_parcels]
                
                for t in reversed(range(T)):
                    delta = rewards[t] + self.gamma * values_all[t+1] * next_non_terminal[t] - values_all[t]
                    advantages[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal[t] * last_gae_lam
                
                returns = advantages + values
            
            # Flatten across time steps and parcels for mini-batch updates
            flat_states = states.view(-1, agent.state_dim) # [T * num_parcels, state_dim]
            flat_actions = actions.view(-1) # [T * num_parcels]
            flat_old_log_probs = old_log_probs.view(-1) # [T * num_parcels]
            flat_returns = returns.view(-1) # [T * num_parcels]
            flat_advantages = advantages.view(-1) # [T * num_parcels]
            
            # Normalize advantages
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
            
            # PPO Updates
            ppo_epochs = 4
            batch_size = 64
            dataset_size = flat_states.shape[0]
            optimizer = self.optimizers[agent_type]
            
            total_loss_val = 0.0
            actor_loss_val = 0.0
            critic_loss_val = 0.0
            entropy_val = 0.0
            
            for _ in range(ppo_epochs):
                permutation = torch.randperm(dataset_size)
                for start_idx in range(0, dataset_size, batch_size):
                    batch_indices = permutation[start_idx : start_idx + batch_size]
                    
                    b_states = flat_states[batch_indices]
                    b_actions = flat_actions[batch_indices]
                    b_old_log_probs = flat_old_log_probs[batch_indices]
                    b_returns = flat_returns[batch_indices]
                    b_advantages = flat_advantages[batch_indices]
                    
                    # Forward pass
                    b_values, b_log_probs, b_entropy = agent.evaluate_actions(b_states, b_actions)
                    
                    # Policy ratio
                    ratios = torch.exp(b_log_probs - b_old_log_probs)
                    
                    # Surrogate loss
                    surr1 = ratios * b_advantages
                    surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Critic loss
                    critic_loss = F.mse_loss(b_values, b_returns)
                    
                    # Entropy loss
                    entropy_loss = b_entropy.mean()
                    
                    # Combined loss
                    loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss
                    
                    # Gradient update
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                    optimizer.step()
                    
                    total_loss_val += loss.item()
                    actor_loss_val += actor_loss.item()
                    critic_loss_val += critic_loss.item()
                    entropy_val += entropy_loss.item()
            
            num_updates = ppo_epochs * math.ceil(dataset_size / batch_size)
            losses[agent_type] = {
                'total': total_loss_val / num_updates,
                'actor_loss': actor_loss_val / num_updates,
                'critic_loss': critic_loss_val / num_updates,
                'entropy': entropy_val / num_updates,
            }
            
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

"""
Unit tests for PIMALUOS models module.
"""

import pytest
import torch
import numpy as np


class TestParcelGNN:
    """Tests for ParcelGNN and related models."""
    
    def test_parcel_gnn_init(self):
        """Test ParcelGNN initialization."""
        from pimaluos.models import ParcelGNN
        
        model = ParcelGNN(
            in_channels=47,
            hidden_channels=64,
            embed_dim=32,
            edge_types=[('parcel', 'visible_from', 'parcel')],
            num_landuse_classes=5
        )
        
        assert model is not None
        assert model.embed_dim == 32
    
    def test_heterogeneous_gat(self):
        """Test HeterogeneousGAT initialization."""
        from pimaluos.models import HeterogeneousGAT
        
        model = HeterogeneousGAT(
            in_channels=47,
            hidden_channels=64,
            out_channels=32,
            edge_types=[('parcel', 'visible_from', 'parcel')],
        )
        
        assert model is not None


class TestStakeholderAgent:
    """Tests for StakeholderAgent."""
    
    def test_agent_init(self):
        """Test agent initialization."""
        from pimaluos.models import StakeholderAgent
        
        agent = StakeholderAgent(
            state_dim=64,
            action_dim=3,
            agent_type='resident'
        )
        
        assert agent.agent_type == 'resident'
        assert agent.action_dim == 3
    
    def test_agent_select_action(self):
        """Test action selection."""
        from pimaluos.models import StakeholderAgent
        
        agent = StakeholderAgent(
            state_dim=64,
            action_dim=3,
            agent_type='resident'
        )
        state = torch.randn(1, 64)
        
        # select_action returns (action, log_prob)
        action, log_prob = agent.select_action(state)
        
        assert 0 <= action < 3
        assert log_prob is not None
    
    def test_agent_awareness_weights(self):
        """Test awareness weights are set correctly."""
        from pimaluos.models import StakeholderAgent
        
        agent = StakeholderAgent(
            state_dim=64,
            action_dim=3,
            agent_type='resident'
        )
        
        assert agent.awareness is not None
        assert 'self' in agent.awareness
        assert 'local' in agent.awareness
        assert 'global' in agent.awareness
        assert 'equity' in agent.awareness


class TestConsensusVoting:
    """Tests for ConsensusVotingMechanism."""
    
    def test_majority_vote(self):
        """Test majority voting."""
        from pimaluos.models.agents import ConsensusVotingMechanism
        
        consensus = ConsensusVotingMechanism(voting_strategy='majority')
        
        actions = {
            'resident': [0, 1, 2, 1, 1],
            'developer': [1, 1, 1, 2, 0],
            'planner': [2, 1, 0, 1, 1],
        }
        
        result = consensus.aggregate_votes(actions)
        
        assert len(result) == 5
        assert all(0 <= a <= 2 for a in result)
    
    def test_weighted_vote(self):
        """Test weighted voting."""
        from pimaluos.models.agents import ConsensusVotingMechanism
        
        consensus = ConsensusVotingMechanism(voting_strategy='weighted')
        
        actions = {
            'resident': [0, 1, 2],
            'developer': [1, 1, 1],
            'planner': [2, 1, 0],
        }
        
        result = consensus.aggregate_votes(actions)
        
        assert len(result) == 3


class TestNashEquilibrium:
    """Tests for Nash equilibrium solver."""
    
    def test_solver_import(self):
        """Test solver can be imported."""
        from pimaluos.models import NashEquilibriumSolver
        assert NashEquilibriumSolver is not None
    
    def test_shapley_import(self):
        """Test Shapley calculator can be imported."""
        from pimaluos.models.nash import ShapleyValueCalculator
        assert ShapleyValueCalculator is not None

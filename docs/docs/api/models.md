# Models Module API Reference

::: pimaluos.models

---

## Graph Neural Networks

### ParcelGNN

Multi-task GNN for parcel embedding and land-use prediction.

::: pimaluos.models.gnn.ParcelGNN
    options:
      show_source: false
      members:
        - forward
        - get_embeddings

### HeterogeneousGAT

Heterogeneous Graph Attention Network for multi-relational parcel graphs.

::: pimaluos.models.gnn.HeterogeneousGAT
    options:
      show_source: false

### GraphSAGEParcelModel

GraphSAGE variant for inductive learning.

::: pimaluos.models.gnn.GraphSAGEParcelModel
    options:
      show_source: false

---

## Multi-Agent Reinforcement Learning

### StakeholderAgent

PPO-based agent representing urban planning stakeholders.

::: pimaluos.models.agents.StakeholderAgent
    options:
      show_source: false
      members:
        - select_action
        - evaluate_actions
        - update

### UtilityFunction

Utility functions for each stakeholder type.

::: pimaluos.models.agents.UtilityFunction
    options:
      show_source: false

### ConsensusVotingMechanism

Multi-agent voting mechanism for action aggregation.

::: pimaluos.models.agents.ConsensusVotingMechanism
    options:
      show_source: false
      members:
        - vote
        - majority_vote
        - weighted_vote
        - soft_vote

### MultiAgentEnvironment

Gym-like environment for multi-agent land-use optimization.

::: pimaluos.models.agents.MultiAgentEnvironment
    options:
      show_source: false
      members:
        - reset
        - step

---

## Nash Equilibrium

### NashEquilibriumSolver

Game-theoretic solver for finding stable land-use configurations.

::: pimaluos.models.nash.NashEquilibriumSolver
    options:
      show_source: false
      members:
        - find_equilibrium_iterative
        - compute_payoff_matrix
        - find_pure_nash_equilibrium

### ShapleyValueCalculator

Compute fair benefit distribution using Shapley values.

::: pimaluos.models.nash.ShapleyValueCalculator
    options:
      show_source: false
      members:
        - compute_shapley_values
        - characteristic_function

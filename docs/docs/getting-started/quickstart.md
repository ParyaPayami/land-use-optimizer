# Quick Start

This 5-minute tutorial walks through a complete PIMALUOS workflow.

## 1. Load City Data

```python
from pimaluos.core import get_data_loader

# Get Manhattan data loader
loader = get_data_loader("manhattan")

# Download and process parcel data
gdf, features = loader.load_and_compute_features()

print(f"Loaded {len(gdf)} parcels with {features.shape[1]} features")
```

## 2. Build Parcel Graph

```python
from pimaluos.core import ParcelGraphBuilder

# Build heterogeneous graph
builder = ParcelGraphBuilder(gdf, features, k_neighbors=10)
graph = builder.build_hetero_data()

print(f"Graph has {graph.num_nodes} nodes and {len(graph.edge_types)} edge types")
```

## 3. Train GNN

```python
import torch
from pimaluos.models import ParcelGNN, train_epoch

# Initialize model
model = ParcelGNN(
    input_dim=features.shape[1],
    hidden_dim=128,
    edge_types=graph.edge_types
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    loss = train_epoch(model, graph, optimizer)
    print(f"Epoch {epoch}: loss={loss:.4f}")
```

## 4. Run Multi-Agent Optimization

```python
from pimaluos.models import StakeholderAgent, MultiAgentEnvironment
from pimaluos.physics import MultiPhysicsEngine

# Setup physics engine
physics = MultiPhysicsEngine(gdf)

# Create environment
env = MultiAgentEnvironment(
    gdf=gdf,
    gnn_model=model,
    physics_engine=physics
)

# Create agents
agents = {
    'resident': StakeholderAgent(state_dim=128, action_dim=3, agent_type='resident'),
    'developer': StakeholderAgent(state_dim=128, action_dim=3, agent_type='developer'),
    'planner': StakeholderAgent(state_dim=128, action_dim=3, agent_type='planner'),
}

# Run one step
state = env.reset()
actions = {name: agent.select_action(state) for name, agent in agents.items()}
next_state, rewards, done, info = env.step(actions)

print(f"Rewards: {rewards}")
```

## 5. Validate with Physics

```python
# Check physics constraints
results = physics.simulate_all(env.current_scenario)

print(f"Traffic congestion: {results['traffic']['avg_congestion_ratio']:.2f}")
print(f"Drainage utilization: {results['hydrology']['capacity_utilization']:.1%}")
print(f"Violations: {results['violations']['total_violations']}")
```

## 6. Start Dashboard

```bash
# Backend
python -m pimaluos.api.server

# Frontend (in another terminal)
cd dashboard
npm install
npm run dev

# Open http://localhost:3000
```

## Next Steps

- [Data Loading Guide](../guide/data-loading.md)
- [GNN Training](../guide/gnn-training.md)
- [Multi-Agent Optimization](../guide/marl.md)

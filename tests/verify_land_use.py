import torch
import pandas as pd
from pimaluos.models.agents import MultiAgentEnvironment
from pimaluos.config.land_use_config import LAND_USE_CATEGORIES


class MockGNN:
    def get_embeddings(self, graph):
        # Return random embeddings
        return {'parcel': torch.randn(10, 128)}
    def parameters(self):
        return [torch.randn(1)]

class MockPhysics:
    def prepare_scenario(self, df): return {}
    def simulate_all(self, scenario):
        return {
            'traffic': {'avg_congestion_ratio': 1.0},
            'hydrology': {'capacity_utilization': 0.5},
            'violations': {'total': 0}
        }

def test_land_use_logic():
    print("Testing Land Use Logic...")
    
    # Mock Graph
    from torch_geometric.data import HeteroData
    graph = HeteroData()
    graph['parcel'].x = torch.randn(10, 128) # 10 parcels
    
    # Create simple line graph: 0-1-2...-9
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)
    graph['parcel', 'spatial_adjacency', 'parcel'].edge_index = edge_index
    
    env = MultiAgentEnvironment(
        gnn_model=MockGNN(),
        graph_data=graph,
        physics_engine=MockPhysics(),
        constraint_masks=pd.DataFrame({'max_far': [10.0]*10}),
        num_parcels=10
    )
    
    state = env.reset()
    print("Env Reset. Current Land Use:", env.current_land_use)
    # Should be all zeros (Residential) initially
    assert (env.current_land_use == 0).all()
    
    # Step 1: Force all agents to choose '5' (Open Space)
    actions = {
        'resident': [5]*10,
        'developer': [5]*10
    }
    
    # Voting should result in 5
    next_state, rewards, done, info = env.step(actions)
    
    print("New Land Use after Step:", env.current_land_use)
    assert (env.current_land_use == 5).all()
    
    # Check features
    # Since all neighbors are use-5 (Open Space), 'park_access' and 'local_green_space' should be high
    features = info['state_features']
    print("State Features:", features)
    
    assert features['local_green_space'] > 0.5
    assert features['park_access'] > 1.0 # Boosted metric
    
    print("Verification Successful!")

if __name__ == "__main__":
    test_land_use_logic()

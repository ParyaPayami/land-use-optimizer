
import logging
import torch
import pandas as pd
from pimaluos.system import UrbanOptSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_edges():
    # Initialize system
    system = UrbanOptSystem(
        city='manhattan',
        data_subset_size=1000,
        device='cpu'
    )
    
    # Load data
    system.load_data()
    
    # Build graph
    system.build_graph()
    
    # Check edges
    edge_index = system.graph['parcel', 'spatial_adjacency', 'parcel'].edge_index
    num_edges = edge_index.shape[1]
    
    print(f"\nTotal Parcels: {system.num_parcels}")
    print(f"Total Spatial Edges: {num_edges}")
    
    if num_edges == 0:
        print("CRITICAL WARNING: No spatial edges found! Spatial rewards will account for nothing.")
    else:
        print(f"Average Degree: {num_edges / system.num_parcels:.2f}")

if __name__ == "__main__":
    verify_edges()

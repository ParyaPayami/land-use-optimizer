"""
Edge Type Ablation Study

Tests the contribution of different edge types to PIMALUOS performance.

Configurations tested:
1. All edges (baseline)
2. No functional similarity
3. No regulatory coupling  
4. Spatial only (adjacency + visual)
5. Each edge type individually

Metrics:
- GNN embedding quality
- Final plan quality
- Training convergence
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from pimaluos import UrbanOptSystem


def run_edge_ablation(
    data_subset: int = 100,
    gnn_epochs: int = 5,
    physics_epochs: int = 3,
    random_seed: int = 42,
    output_dir: Path = Path('results/ablation')
):
    """
    Run edge type ablation experiments.
    
    Tests different combinations of edge types to understand their contribution.
    """
    print("="*70)
    print("EDGE TYPE ABLATION STUDY")
    print("="*70)
    print(f"Data subset: {data_subset} parcels")
    print(f"GNN epochs: {gnn_epochs}, Physics epochs: {physics_epochs}")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define edge type configurations to test
    configurations = {
        'all_edges': ['spatial_adjacency', 'visual_connectivity', 'functional_similarity', 
                      'infrastructure', 'regulatory_coupling'],
        'no_functional': ['spatial_adjacency', 'visual_connectivity', 
                         'infrastructure', 'regulatory_coupling'],
        'no_regulatory': ['spatial_adjacency', 'visual_connectivity', 'functional_similarity', 
                         'infrastructure'],
        'spatial_only': ['spatial_adjacency', 'visual_connectivity'],
        'functional_only': ['functional_similarity'],
        'regulatory_only': ['regulatory_coupling'],
    }
    
    results = {}
    
    for config_name, edge_types in configurations.items():
        print("-"*70)
        print(f"Testing: {config_name}")
        print(f"Edge types: {edge_types}")
        print("-"*70)
        
        try:
            # Initialize system
            system = UrbanOptSystem(
                city='manhattan',
                data_subset_size=data_subset,
                llm_mode='mock',
                device='cpu',
                random_seed=random_seed
            )
            
            # Load data
            start_time = time.time()
            gdf, features = system.load_data()
            
            # Build graph with specific edge types
            graph = system.build_graph(edge_types=edge_types)
            
            # Extract constraints
            constraints = system.extract_constraints()
            
            # Pre-train GNN
            history_pretrain = system.pretrain_gnn(num_epochs=gnn_epochs)
            
            # Physics-informed training
            history_physics = system.train_with_physics_feedback(num_epochs=physics_epochs)
            
            elapsed = time.time() - start_time
            
            # Get final embeddings
            with torch.no_grad():
                embeddings = system.gnn_model.get_embeddings(graph.to(system.device))
                if isinstance(embeddings, dict):
                    parcel_embeddings = embeddings['parcel']
                else:
                    parcel_embeddings = embeddings
            
            # Compute embedding quality (using simple variance as proxy)
            embedding_variance = float(torch.var(parcel_embeddings).item())
            embedding_mean_norm = float(torch.norm(parcel_embeddings.mean(0)).item())
            
            # Store results
            results[config_name] = {
                'edge_types': edge_types,
                'num_edge_types': len(edge_types),
                'num_edges': int(sum([graph[et].edge_index.shape[1] for et in graph.edge_types])),
                'final_pretrain_loss': float(history_pretrain['pretrain_losses'][-1]),
                'final_physics_loss': float(history_physics['physics_losses'][-1]),
                'embedding_variance': embedding_variance,
                'embedding_mean_norm': embedding_mean_norm,
                'total_time': elapsed,
            }
            
            print(f"✓ Complete: {config_name}")
            print(f"  Edges: {results[config_name]['num_edges']}")
            print(f"  Final physics loss: {results[config_name]['final_physics_loss']:.4f}")
            print(f"  Embedding variance: {embedding_variance:.4f}")
            print(f"  Time: {elapsed:.1f}s")
            
        except Exception as e:
            print(f"✗ Failed: {config_name}")
            print(f"  Error: {e}")
            results[config_name] = {
                'edge_types': edge_types,
                'error': str(e)
            }
        
        print()
    
    # Generate comparison table
    print("="*70)
    print("ABLATION RESULTS SUMMARY")
    print("="*70)
    
    comparison_data = []
    for config_name, data in results.items():
        if 'error' not in data:
            comparison_data.append({
                'Configuration': config_name.replace('_', ' ').title(),
                'Num Edge Types': data['num_edge_types'],
                'Total Edges': data['num_edges'],
                'Physics Loss↓': f"{data['final_physics_loss']:.4f}",
                'Embedding Var': f"{data['embedding_variance']:.4f}",
                'Time(s)': f"{data['total_time']:.1f}",
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Save results
    results_file = output_dir / 'edge_types_impact.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")
    
    table_file = output_dir / 'edge_types_table.csv'
    df_comparison.to_csv(table_file, index=False)
    print(f"✓ Table saved to {table_file}")
    
    print("\n" + "="*70)
    print("EDGE TYPE ABLATION COMPLETE")
    print("="*70)
    
    return results, df_comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run edge type ablation study')
    parser.add_argument('--data_subset', type=int, default=100, help='Number of parcels')
    parser.add_argument('--gnn_epochs', type=int, default=5, help='GNN pre-training epochs')
    parser.add_argument('--physics_epochs', type=int, default=3, help='Physics training epochs')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/ablation', help='Output directory')
    
    args = parser.parse_args()
    
    run_edge_ablation(
        data_subset=args.data_subset,
        gnn_epochs=args.gnn_epochs,
        physics_epochs=args.physics_epochs,
        random_seed=args.random_seed,
        output_dir=Path(args.output)
    )

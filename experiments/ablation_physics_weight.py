"""
Physics Weight Ablation Study

Tests the impact of different physics weight (λ) values on the trade-off
between economic objectives and physics constraints.

Tests λ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}

Metrics:
- Physics violation rate
- Economic value
- Environmental impact
- Training convergence
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from pimaluos import UrbanOptSystem


def run_physics_weight_sweep(
    data_subset: int = 100,
    gnn_epochs: int = 5,
    physics_epochs: int = 5, 
    weights: list = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    random_seed: int = 42,
    output_dir: Path = Path('results/ablation')
):
    """
    Run physics weight sweep experiment.
    
    Tests different values of physics penalty weight to understand
    the economic-physics trade-off.
    """
    print("="*70)
    print("PHYSICS WEIGHT ABLATION STUDY")
    print("="*70)
    print(f"Data subset: {data_subset} parcels")
    print(f"Testing weights: {weights}")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for weight in weights:
        print("-"*70)
        print(f"Testing physics weight λ = {weight}")
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
            
            # Load data and build graph
            start_time = time.time()
            gdf, features = system.load_data()
            graph = system.build_graph()
            constraints = system.extract_constraints()
            
            # Pre-train GNN (same for all)
            history_pretrain = system.pretrain_gnn(num_epochs=gnn_epochs)
            
            # Physics-informed training with specific weight
            history_physics = system.train_with_physics_feedback(
                num_epochs=physics_epochs,
                physics_weight=weight
            )
            
            elapsed = time.time() - start_time
            
            # Store results
            results[f'lambda_{weight}'] = {
                'physics_weight': weight,
                'final_pretrain_loss': float(history_pretrain['pretrain_losses'][-1]),
                'final_physics_loss': float(history_physics['physics_losses'][-1]),
                'pretrain_losses': [float(x) for x in history_pretrain['pretrain_losses']],
                'physics_losses': [float(x) for x in history_physics['physics_losses']],
                'total_time': elapsed,
            }
            
            print(f"✓ Complete: λ = {weight}")
            print(f"  Final physics loss: {results[f'lambda_{weight}']['final_physics_loss']:.4f}")
            print(f"  Time: {elapsed:.1f}s")
            
        except Exception as e:
            print(f"✗ Failed: λ = {weight}")
            print(f"  Error: {e}")
            results[f'lambda_{weight}'] = {
                'physics_weight': weight,
                'error': str(e)
            }
        
        print()
    
    # Generate comparison table
    print("="*70)
    print("PHYSICS WEIGHT RESULTS SUMMARY")
    print("="*70)
    
    comparison_data = []
    for key, data in results.items():
        if 'error' not in data:
            comparison_data.append({
                'Lambda (λ)': data['physics_weight'],
                'Final Physics Loss↓': f"{data['final_physics_loss']:.4f}",
                'Training Time(s)': f"{data['total_time']:.1f}",
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Save results
    results_file = output_dir / 'physics_weight_sweep.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")
    
    table_file = output_dir / 'physics_weight_table.csv'
    df_comparison.to_csv(table_file, index=False)
    print(f"✓ Table saved to {table_file}")
    
    print("\n" + "="*70)
    print("PHYSICS WEIGHT ABLATION COMPLETE")
    print("="*70)
    
    return results, df_comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run physics weight sweep')
    parser.add_argument('--data_subset', type=int, default=100, help='Number of parcels')
    parser.add_argument('--gnn_epochs', type=int, default=5, help='GNN pre-training epochs')
    parser.add_argument('--physics_epochs', type=int, default=5, help='Physics training epochs')
    parser.add_argument('--weights', nargs='+', type=float, 
                       default=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0], 
                       help='Physics weights to test')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/ablation', help='Output directory')
    
    args = parser.parse_args()
    
    run_physics_weight_sweep(
        data_subset=args.data_subset,
        gnn_epochs=args.gnn_epochs,
        physics_epochs=args.physics_epochs,
        weights=args.weights,
        random_seed=args.random_seed,
        output_dir=Path(args.output)
    )

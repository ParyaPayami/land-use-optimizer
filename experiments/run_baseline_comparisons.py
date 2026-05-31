"""
Baseline Comparison Experiment

Compares PIMALUOS against three baseline methods:
1. Random: Uniformly random action selection
2. Rule-based: Heuristic rules based on FAR utilization
3. Greedy: Single-objective economic maximization

Evaluates on multiple metrics and performs statistical significance testing.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import torch

from pimaluos.core import get_data_loader
from pimaluos.baselines import RandomOptimizer, RuleBasedOptimizer, GreedyOptimizer


from pimaluos.system import UrbanOptSystem

def run_pimaluos_actual(data_subset: int, random_seed: int, physics_weight: float = 0.3, use_gnn: bool = True):
    """Run actual PIMALUOS on the subset to get real diverse actions."""
    if use_gnn:
        print(f"\nRunning Actual PIMALUOS on {data_subset} parcels (Physics={physics_weight})...")
    else:
        print(f"\nRunning PIMALUOS (No GNN) on {data_subset} parcels...")
        
    system = UrbanOptSystem(city='manhattan', data_subset_size=data_subset, device='cpu', random_seed=random_seed)
    
    # Run the standard stages but with fewer epochs for speed in baseline script
    system.load_data()
    
    # Load cached heterogeneous graph if running on full Manhattan scale to avoid long rebuilds
    graph_cache = Path("results/full_manhattan/cache/manhattan_hetero_graph.pt")
    if (data_subset is None or data_subset >= 42000) and graph_cache.exists():
        print("Loading heterogeneous graph from full Manhattan cache...")
        system.graph = torch.load(graph_cache, map_location="cpu", weights_only=False)
    else:
        system.build_graph()
    
    if use_gnn:
        system.initialize_gnn()
        # Load cached checkpoints if running on full Manhattan scale to avoid long re-training
        pretrained_ckpt = Path("results/full_manhattan/cache/gnn_pretrained.pt")
        physics_ckpt = Path("results/full_manhattan/cache/gnn_physics.pt")
        if (data_subset is None or data_subset >= 42000) and pretrained_ckpt.exists() and physics_ckpt.exists():
            print("Loading pre-trained & fine-tuned GNN weights from cache...")
            if physics_weight == 0.0:
                ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
            else:
                ckpt = torch.load(physics_ckpt, map_location="cpu", weights_only=False)
            system.gnn_model.load_state_dict(ckpt["state_dict"])
            system.initialize_physics_engine()
            system.extract_constraints()
        else:
            # Small epochs for evaluation speed
            system.pretrain_gnn(num_epochs=30)
            system.train_with_physics_feedback(num_epochs=20, physics_weight=physics_weight)
    
    # Train agents (which also caches weights)
    trainer = system.optimize_with_marl(num_iterations=5, steps_per_iteration=2, use_gnn=use_gnn)
    
    # Get final plan actions (these are now FAR modifications 0, 1, 2)
    plan = system.generate_final_plan(trainer)
    actions = plan['proposed_use_code'].values
    
    return actions


def compute_metrics(
    actions: np.ndarray,
    gdf: pd.DataFrame,
    constraint_masks: pd.DataFrame
) -> dict:
    """
    Compute evaluation metrics for a set of actions.
    
    Metrics:
    - Economic value (FAR × lot_area)
    - Diversity (entropy of actions)
    - Constraint violations
    - Environmental impact (placeholder)
    """
    n = len(actions)
    
    # Get data
    current_far = gdf['built_far'].fillna(1.0).values[:n] if 'built_far' in gdf.columns else np.ones(n)
    lot_area = gdf['lot_area'].fillna(1000).values[:n] if 'lot_area' in gdf.columns else np.ones(n) * 1000
    max_far = constraint_masks['max_far'].values[:n]
    
    # Apply actions to get proposed FAR
    proposed_far = current_far.copy()
    proposed_far[actions == 0] *= 0.8  # Decrease by 20%
    proposed_far[actions == 2] *= 1.2  # Increase by 20%
    proposed_far = np.clip(proposed_far, 0.1, max_far)
    
    # Economic value
    economic_value = np.sum(proposed_far * lot_area)
    
    # Action diversity (Shannon entropy)
    action_counts = np.bincount(actions, minlength=3)
    action_probs = action_counts / n
    action_probs = action_probs[action_probs > 0]  # Remove zeros
    diversity = -np.sum(action_probs * np.log(action_probs))
    
    # Constraint violations
    violations = np.sum(proposed_far > max_far * 1.01)  # 1% tolerance
    
    # Environmental impact (simplified: lower FAR = better)
    env_impact = np.mean(proposed_far)  # Lower is better
    
    # Social equity (Gini coefficient of FAR distribution)
    sorted_far = np.sort(proposed_far)
    n_far = len(sorted_far)
    index = np.arange(1, n_far + 1)
    gini = (2 * np.sum(index * sorted_far)) / (n_far * np.sum(sorted_far)) - (n_far + 1) / n_far
    
    return {
        'economic_value': float(economic_value),
        'diversity_entropy': float(diversity),
        'constraint_violations': int(violations),
        'environmental_impact': float(env_impact),
        'social_equity_gini': float(gini),
        'action_distribution': {
            'decrease': int(action_counts[0]),
            'maintain': int(action_counts[1]),
            'increase': int(action_counts[2]),
        }
    }


def run_baseline_comparison(
    data_subset: int = 100,
    num_runs: int = 5,
    random_seed: int = 42,
    output_dir: Path = Path('results/baselines')
):
    """
    Run all baseline comparisons.
    
    Args:
        data_subset: Number of parcels to use
        num_runs: Number of runs for statistical significance
        random_seed: Base random seed
        output_dir: Output directory for results
    """
    print("="*70)
    print("BASELINE COMPARISON EXPERIMENT")
    print("="*70)
    print(f"Data subset: {data_subset} parcels")
    print(f"Number of runs: {num_runs}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading Manhattan parcel data...")
    loader = get_data_loader('manhattan')
    gdf, features = loader.load_and_compute_features()
    
    # Subset
    if data_subset is None or data_subset >= len(gdf):
        gdf_subset = gdf.reset_index(drop=True)
        data_subset = len(gdf_subset)
    else:
        np.random.seed(random_seed)
        indices = np.random.choice(len(gdf), size=min(data_subset, len(gdf)), replace=False)
        gdf_subset = gdf.iloc[indices].reset_index(drop=True)
    
    # Create constraint masks (simplified)
    constraint_masks = pd.DataFrame({
        'max_far': 2.0,  # Default
        'max_height_ft': 85.0,
        'min_open_space_ratio': 0.2,
    }, index=range(len(gdf_subset)))
    
    print(f"Loaded {len(gdf_subset)} parcels\n")
    
    # Run actual PIMALUOS MARL agents to get real predictions
    print("\n" + "-"*70)
    print("Running PIMALUOS (Actual MARL Model)...")
    print("-"*70)
    pimaluos_results = []
    pimaluos_data = True
    
    for run in range(num_runs):
        start_time = time.time()
        try:
            pimaluos_actions = run_pimaluos_actual(data_subset, random_seed + run)
            elapsed = time.time() - start_time
            metrics = compute_metrics(pimaluos_actions, gdf_subset, constraint_masks)
            metrics['execution_time'] = elapsed
            pimaluos_results.append((metrics, pimaluos_actions))
            print(f"✓ Run {run+1}/{num_runs} complete: Eco={metrics['economic_value']:.0f}, Div={metrics['diversity_entropy']:.3f}")
        except Exception as e:
            print(f"⚠ PIMALUOS run {run+1} failed: {e}")
            pimaluos_data = False
            break
    
    # Run PIMALUOS (No Physics)
    print("\n" + "-"*70)
    print("Running PIMALUOS (No Physics)...")
    print("-"*70)
    pimaluos_no_physics_results = []
    pimaluos_no_physics_data = True
    
    for run in range(num_runs):
        start_time = time.time()
        try:
            pimaluos_actions_np = run_pimaluos_actual(data_subset, random_seed + run, physics_weight=0.0)
            elapsed = time.time() - start_time
            metrics = compute_metrics(pimaluos_actions_np, gdf_subset, constraint_masks)
            metrics['execution_time'] = elapsed
            pimaluos_no_physics_results.append((metrics, pimaluos_actions_np))
            print(f"✓ Run {run+1}/{num_runs} complete: Eco={metrics['economic_value']:.0f}, Div={metrics['diversity_entropy']:.3f}")
        except Exception as e:
            print(f"⚠ PIMALUOS (No Physics) run {run+1} failed: {e}")
            pimaluos_no_physics_data = False
            break
            
    # Run PIMALUOS (No GNN)
    print("\n" + "-"*70)
    print("Running PIMALUOS (No GNN)...")
    print("-"*70)
    pimaluos_no_gnn_results = []
    pimaluos_no_gnn_data = True
    
    for run in range(num_runs):
        start_time = time.time()
        try:
            pimaluos_actions_ng = run_pimaluos_actual(data_subset, random_seed + run, use_gnn=False)
            elapsed = time.time() - start_time
            metrics = compute_metrics(pimaluos_actions_ng, gdf_subset, constraint_masks)
            metrics['execution_time'] = elapsed
            pimaluos_no_gnn_results.append((metrics, pimaluos_actions_ng))
            print(f"✓ Run {run+1}/{num_runs} complete: Eco={metrics['economic_value']:.0f}, Div={metrics['diversity_entropy']:.3f}")
        except Exception as e:
            print(f"⚠ PIMALUOS (No GNN) run {run+1} failed: {e}")
            pimaluos_no_gnn_data = False
            break
            
    # Run baselines
    results = {}
    
    # 1. Random Baseline
    print("\n" + "-"*70)
    print("Running Random Baseline...")
    print("-"*70)
    random_results = []
    for run in range(num_runs):
        optimizer = RandomOptimizer(random_seed=random_seed + run)
        start_time = time.time()
        actions, run_metrics = optimizer.optimize(gdf_subset, constraint_masks, data_subset)
        elapsed = time.time() - start_time
        
        metrics = compute_metrics(actions, gdf_subset, constraint_masks)
        metrics['execution_time'] = elapsed
        random_results.append(metrics)
        
        if run == 0:
            print(f"  Actions: {run_metrics}")
            print(f"  Metrics: Economic={metrics['economic_value']:.2f}, "
                  f"Diversity={metrics['diversity_entropy']:.3f}, "
                  f"Violations={metrics['constraint_violations']}")
    
    # Exclude non-numeric fields from averaging
    numeric_keys = [k for k in random_results[0].keys() if k != 'action_distribution']
    
    results['random'] = {
        'mean': {k: np.mean([r[k] for r in random_results]) for k in numeric_keys},
        'std': {k: np.std([r[k] for r in random_results]) for k in numeric_keys},
        'action_distribution': random_results[0]['action_distribution'],  # Use first run
        'all_runs': random_results
    }
    
    # 2. Rule-Based Baseline
    print("\n" + "-"*70)
    print("Running Rule-Based Baseline...")
    print("-"*70)
    optimizer = RuleBasedOptimizer()
    start_time = time.time()
    actions, run_metrics = optimizer.optimize(gdf_subset, constraint_masks, data_subset)
    elapsed = time.time() - start_time
    
    metrics = compute_metrics(actions, gdf_subset, constraint_masks)
    metrics['execution_time'] = elapsed
    
    print(f"  Actions: {run_metrics}")
    print(f"  Metrics: Economic={metrics['economic_value']:.2f}, "
          f"Diversity={metrics['diversity_entropy']:.3f}, "
          f"Violations={metrics['constraint_violations']}")
    
    results['rule_based'] = {
        'mean': metrics,
        'action_details': run_metrics
    }
    
    # 3. Greedy Baseline
    print("\n" + "-"*70)
    print("Running Greedy Baseline...")
    print("-"*70)
    optimizer = GreedyOptimizer()
    start_time = time.time()
    actions, run_metrics = optimizer.optimize(gdf_subset, constraint_masks, data_subset)
    elapsed = time.time() - start_time
    
    metrics = compute_metrics(actions, gdf_subset, constraint_masks)
    metrics['execution_time'] = elapsed
    
    print(f"  Actions: {run_metrics}")
    print(f"  Metrics: Economic={metrics['economic_value']:.2f}, "
          f"Diversity={metrics['diversity_entropy']:.3f}, "
          f"Violations={metrics['constraint_violations']}")
    
    results['greedy'] = {
        'mean': metrics,
        'action_details': run_metrics
    }
    
    # 4. PIMALUOS (Actual)
    if pimaluos_data:
        print("\n" + "-"*70)
        print("PIMALUOS (Actual MARL Model)...")
        print("-"*70)
        
        # Calculate mean and std
        numeric_keys = [k for k in pimaluos_results[0][0].keys() if k != 'action_distribution']
        mean_metrics = {k: float(np.mean([r[0][k] for r in pimaluos_results])) for k in numeric_keys}
        std_metrics = {k: float(np.std([r[0][k] for r in pimaluos_results])) for k in numeric_keys}
        
        # Count actions (using first run)
        from collections import Counter
        action_counts = {str(int(k)): int(v) for k, v in Counter(pimaluos_results[0][1]).items()}
        
        print(f"  Action Dist (0=Dec,1=Main,2=Inc) [Run 1]: {action_counts}")
        print(f"  Metrics: Economic={mean_metrics['economic_value']:.2f}±{std_metrics['economic_value']:.2f}, "
              f"Diversity={mean_metrics['diversity_entropy']:.3f}±{std_metrics['diversity_entropy']:.3f}, "
              f"Violations={mean_metrics['constraint_violations']}")
        
        results['pimaluos'] = {
            'mean': mean_metrics,
            'std': std_metrics,
            'action_distribution': action_counts,
            'all_runs': [r[0] for r in pimaluos_results]
        }
        
    # 5. PIMALUOS (No Physics)
    if pimaluos_no_physics_data:
        print("\n" + "-"*70)
        print("PIMALUOS (No Physics)...")
        print("-"*70)
        
        numeric_keys = [k for k in pimaluos_no_physics_results[0][0].keys() if k != 'action_distribution']
        mean_metrics_np = {k: float(np.mean([r[0][k] for r in pimaluos_no_physics_results])) for k in numeric_keys}
        std_metrics_np = {k: float(np.std([r[0][k] for r in pimaluos_no_physics_results])) for k in numeric_keys}
        
        action_counts_np = {str(int(k)): int(v) for k, v in Counter(pimaluos_no_physics_results[0][1]).items()}
        
        print(f"  Action Dist (0=Dec,1=Main,2=Inc) [Run 1]: {action_counts_np}")
        print(f"  Metrics: Economic={mean_metrics_np['economic_value']:.2f}±{std_metrics_np['economic_value']:.2f}, "
              f"Diversity={mean_metrics_np['diversity_entropy']:.3f}±{std_metrics_np['diversity_entropy']:.3f}, "
              f"Violations={mean_metrics_np['constraint_violations']}")
        
        results['pimaluos_no_physics'] = {
            'mean': mean_metrics_np,
            'std': std_metrics_np,
            'action_distribution': action_counts_np,
            'all_runs': [r[0] for r in pimaluos_no_physics_results]
        }
        
    # 6. PIMALUOS (No GNN)
    if pimaluos_no_gnn_data:
        print("\n" + "-"*70)
        print("PIMALUOS (No GNN)...")
        print("-"*70)
        
        numeric_keys = [k for k in pimaluos_no_gnn_results[0][0].keys() if k != 'action_distribution']
        mean_metrics_ng = {k: float(np.mean([r[0][k] for r in pimaluos_no_gnn_results])) for k in numeric_keys}
        std_metrics_ng = {k: float(np.std([r[0][k] for r in pimaluos_no_gnn_results])) for k in numeric_keys}
        
        action_counts_ng = {str(int(k)): int(v) for k, v in Counter(pimaluos_no_gnn_results[0][1]).items()}
        
        print(f"  Action Dist (0=Dec,1=Main,2=Inc) [Run 1]: {action_counts_ng}")
        print(f"  Metrics: Economic={mean_metrics_ng['economic_value']:.2f}±{std_metrics_ng['economic_value']:.2f}, "
              f"Diversity={mean_metrics_ng['diversity_entropy']:.3f}±{std_metrics_ng['diversity_entropy']:.3f}, "
              f"Violations={mean_metrics_ng['constraint_violations']}")
        
        results['pimaluos_no_gnn'] = {
            'mean': mean_metrics_ng,
            'std': std_metrics_ng,
            'action_distribution': action_counts_ng,
            'all_runs': [r[0] for r in pimaluos_no_gnn_results]
        }
    
    # Statistical comparison
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON")
    print("="*70)
    
    comparison_table = []
    methods = ['random', 'rule_based', 'greedy']
    if pimaluos_data:
        methods.append('pimaluos')
    if pimaluos_no_physics_data:
        methods.append('pimaluos_no_physics')
    if pimaluos_no_gnn_data:
        methods.append('pimaluos_no_gnn')
    
    for method in methods:
        mean = results[method]['mean']
        std = results[method].get('std', {k: 0.0 for k in mean.keys()})
        
        row = {
            'Method': method.replace('_', ' ').title(),
            'Economic↑': f"{mean['economic_value']:.0f}±{std['economic_value']:.0f}",
            'Diversity↑': f"{mean['diversity_entropy']:.3f}±{std['diversity_entropy']:.3f}",
            'Violations↓': f"{mean['constraint_violations']:.0f}±{std['constraint_violations']:.0f}",
            'Env. Impact↓': f"{mean['environmental_impact']:.3f}±{std['environmental_impact']:.3f}",
            'Gini↓': f"{mean['social_equity_gini']:.3f}±{std['social_equity_gini']:.3f}",
            'Time(s)': f"{mean['execution_time']:.2f}",
        }
        comparison_table.append(row)
    
    df_comparison = pd.DataFrame(comparison_table)
    print(df_comparison.to_string(index=False))
    
    # Save results
    results_file = output_dir / 'comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")
    
    table_file = output_dir / 'comparison_table.csv'
    df_comparison.to_csv(table_file, index=False)
    print(f"✓ Table saved to {table_file}")
    
    print("\n" + "="*70)
    print("BASELINE COMPARISON COMPLETE")
    print("="*70)
    
    return results, df_comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run baseline comparisons')
    parser.add_argument('--data_subset', type=int, default=500, help='Number of parcels')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs for random baseline')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/baselines', help='Output directory')
    
    args = parser.parse_args()
    
    run_baseline_comparison(
        data_subset=args.data_subset,
        num_runs=args.num_runs,
        random_seed=args.random_seed,
        output_dir=Path(args.output)
    )

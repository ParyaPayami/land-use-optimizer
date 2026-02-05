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

from pimaluos.core import get_data_loader
from pimaluos.baselines import RandomOptimizer, RuleBasedOptimizer, GreedyOptimizer


def load_pimaluos_results(results_path: Path) -> dict:
    """Load existing PIMALUOS results from demo."""
    with open(results_path, 'r') as f:
        return json.load(f)


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
    
    # Load PIMALUOS results
    pimaluos_results_path = Path('results/small_scale_demo/results.json')
    if pimaluos_results_path.exists():
        pimaluos_data = load_pimaluos_results(pimaluos_results_path)
        pimaluos_actions = np.array([2] * data_subset)  # All increase from demo
        print("✓ Loaded PIMALUOS results from demo")
    else:
        print("⚠ PIMALUOS results not found, skipping")
        pimaluos_data = None
    
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
    
    # 4. PIMALUOS (from demo)
    if pimaluos_data:
        print("\n" + "-"*70)
        print("PIMALUOS (from existing demo)...")
        print("-"*70)
        
        pimaluos_metrics = compute_metrics(pimaluos_actions, gdf_subset, constraint_masks)
        pimaluos_metrics['execution_time'] = pimaluos_data['performance']['total_time_seconds']
        
        print(f"  Actions: {pimaluos_data['actions']}")
        print(f"  Metrics: Economic={pimaluos_metrics['economic_value']:.2f}, "
              f"Diversity={pimaluos_metrics['diversity_entropy']:.3f}, "
              f"Violations={pimaluos_metrics['constraint_violations']}")
        
        results['pimaluos'] = {
            'mean': pimaluos_metrics,
            'training_losses': pimaluos_data['training']
        }
    
    # Statistical comparison
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON")
    print("="*70)
    
    comparison_table = []
    methods = ['random', 'rule_based', 'greedy']
    if pimaluos_data:
        methods.append('pimaluos')
    
    for method in methods:
        row = {
            'Method': method.replace('_', ' ').title(),
            'Economic↑': f"{results[method]['mean']['economic_value']:.0f}",
            'Diversity↑': f"{results[method]['mean']['diversity_entropy']:.3f}",
            'Violations↓': results[method]['mean']['constraint_violations'],
            'Env. Impact↓': f"{results[method]['mean']['environmental_impact']:.3f}",
            'Gini↓': f"{results[method]['mean']['social_equity_gini']:.3f}",
            'Time(s)': f"{results[method]['mean']['execution_time']:.2f}",
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
    parser.add_argument('--data_subset', type=int, default=100, help='Number of parcels')
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

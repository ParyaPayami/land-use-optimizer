
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Setup plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
colors = sns.color_palette("viridis")

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_training_curves():
    """Generate training curves from checkpoint."""
    checkpoint_path = Path("results/small_scale_demo/checkpoint.pth")
    if not checkpoint_path.exists():
        print("Checkpoint not found, skipping training curves.")
        return

    print("Generating training curves...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract losses
    pretrain_losses = checkpoint.get('pretrain_losses', [])
    physics_losses = checkpoint.get('physics_losses', [])
    
    # 1. Pre-training Loss
    if pretrain_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(pretrain_losses)+1), pretrain_losses, marker='o', linewidth=2)
        plt.title("GNN Pre-training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "gnn_pretrain_loss.png", dpi=300)
        plt.close()
        
    # 2. Physics-Informed Loss
    if physics_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(physics_losses)+1), physics_losses, marker='s', color='orange', linewidth=2)
        plt.title("Physics-Informed Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "physics_training_loss.png", dpi=300)
        plt.close()
    
    print("✓ Training curves saved.")

def plot_baseline_comparison():
    """Generate bar chart for baseline comparison."""
    results_path = Path("results/baselines/comparison_results.json")
    if not results_path.exists():
        print("Baseline results not found, skipping.")
        return

    print("Generating baseline comparison chart...")
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    # Extract metrics
    methods = []
    economic_values = []
    
    # Map raw keys to display names
    name_map = {
        'pimaluos': 'PIMALUOS',
        'random': 'Random',
        'rule_based': 'Rule-Based',
        'greedy': 'Greedy'
    }
    
    for key, val in data.items():
        methods.append(name_map.get(key, key.title()))
        if 'mean' in val:
            economic_values.append(val['mean']['economic_value'])
        else:
            # Handle old structure if needed
            economic_values.append(val.get('economic_value', 0))
            
    # Create DataFrame
    df = pd.DataFrame({
        'Method': methods,
        'Economic Value': economic_values
    })
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Method', y='Economic Value', data=df, palette='viridis')
    
    # Add value labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{int(p.get_height()):,}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.title("Economic Value Comparison: PIMALUOS vs Baselines", fontsize=14)
    plt.xlabel("")
    plt.ylabel("Total Economic Value")
    plt.ylim(min(economic_values)*0.98, max(economic_values)*1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_comparison.png", dpi=300)
    plt.close()
    print("✓ Baseline chart saved.")

def plot_edge_ablation():
    """Generate chart for edge type ablation."""
    results_path = Path("results/ablation/edge_types_impact.json")
    if not results_path.exists():
        print("Ablation results not found, skipping.")
        return

    print("Generating edge ablation chart...")
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    configs = []
    losses = []
    
    for key, val in data.items():
        if 'error' in val:
            continue
        configs.append(key.replace('_', ' ').title())
        losses.append(val['final_physics_loss'])
        
    df = pd.DataFrame({
        'Configuration': configs,
        'Physics Loss': losses
    })
    
    # Sort by loss
    df = df.sort_values('Physics Loss')
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Configuration', y='Physics Loss', data=df, palette='magma')
    
    # Add value labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.4f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=11)
        
    plt.title("Impact of Edge Types on Physics-Informed Loss (Lower is Better)", fontsize=14)
    plt.xlabel("")
    plt.ylabel("Final Physics Loss")
    plt.ylim(min(losses)*0.95, max(losses)*1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "edge_ablation.png", dpi=300)
    plt.close()
    print("✓ Edge ablation chart saved.")

def plot_physics_sweep():
    """Generate chart for physics weight sweep."""
    results_path = Path("results/ablation/physics_weight_sweep.json")
    if not results_path.exists():
        print("Physics sweep results not found, skipping.")
        return

    print("Generating physics sweep chart...")
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    lambdas = []
    losses = []
    
    for key, val in data.items():
        if 'error' in val:
            continue
        lambdas.append(val['physics_weight'])
        losses.append(val['final_physics_loss'])
        
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, losses, marker='D', linestyle='--', linewidth=2, markersize=8)
    
    plt.title("Effect of Physics Weight ($\lambda$) on Final Loss", fontsize=14)
    plt.xlabel("Physics Penalty Weight ($\lambda$)")
    plt.ylabel("Final Physics Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "physics_weight_sweep.png", dpi=300)
    plt.close()
    print("✓ Physics sweep chart saved.")

if __name__ == "__main__":
    print(f"Generating figures to {OUTPUT_DIR}...")
    plot_training_curves()
    plot_baseline_comparison()
    plot_edge_ablation()
    plot_physics_sweep()
    print("Done!")

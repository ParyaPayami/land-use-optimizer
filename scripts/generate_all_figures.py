"""
generate_all_figures.py
Generates all paper figures in the PIMALUOS design language:
  - Background: #F3F4F6 (light grey dot-grid, matches architecture.html)
  - Card face:  #FFFFFF with subtle shadow/border
  - Palette:    Pastel, muted, modern (sky-blue, sage-green, lavender, peach, rose)
  - Typography: Clean, sans-serif, dark-grey text
  - Style:      Soft, minimal, no sharp/saturated colors
"""

import json
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Design System (mirrors architecture.html CSS variables) ─────────────────
BG          = '#F3F4F6'   # --bg-main
CARD        = '#FFFFFF'   # --box-bg (opaque for print)
BORDER      = '#E5E7EB'   # box border
TEXT        = '#1F2937'   # --text-main
MUTED       = '#6B7280'   # --text-muted
MONO        = '#475569'   # --text-mono

# Pastel accent palette pulled from architecture.html gradients
C_BLUE      = '#93C5FD'   # light-blue (Orchestration / Resident)
C_BLUE_D    = '#3B82F6'   # blue deep accent
C_GREEN     = '#86EFAC'   # light-green (Spatial / Environment)
C_GREEN_D   = '#10B981'
C_PURPLE    = '#C4B5FD'   # light-purple (Physics / City Planner)
C_PURPLE_D  = '#8B5CF6'
C_ORANGE    = '#FCD34D'   # light-orange (Developer / Zoning)
C_ORANGE_D  = '#F59E0B'
C_PINK      = '#F9A8D4'   # light-pink (Equity / MARL)
C_PINK_D    = '#EC4899'
C_TEAL      = '#5EEAD4'   # teal (RAG)
C_SLATE     = '#CBD5E1'   # slate (neutral baselines)

OUTPUT_DIR = Path('results/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Shared matplotlib style ──────────────────────────────────────────────────
def setup_style():
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':          11,
        'axes.titlesize':     13,
        'axes.titleweight':   'bold',
        'axes.titlecolor':    TEXT,
        'axes.labelcolor':    MUTED,
        'axes.labelsize':     10,
        'axes.edgecolor':     BORDER,
        'axes.linewidth':     0.8,
        'axes.facecolor':     CARD,
        'axes.grid':          True,
        'grid.color':         BORDER,
        'grid.linewidth':     0.6,
        'grid.linestyle':     '--',
        'xtick.color':        MUTED,
        'ytick.color':        MUTED,
        'xtick.labelsize':    9,
        'ytick.labelsize':    9,
        'figure.facecolor':   BG,
        'text.color':         TEXT,
        'legend.frameon':     True,
        'legend.facecolor':   CARD,
        'legend.edgecolor':   BORDER,
        'legend.fontsize':    9,
        'savefig.facecolor':  BG,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
    })

def add_dot_grid(fig):
    """Add subtle dot-grid pattern matching architecture.html body background."""
    ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
    ax_bg.set_xlim(0, 1); ax_bg.set_ylim(0, 1)
    ax_bg.axis('off')
    ax_bg.set_facecolor(BG)
    xs = np.arange(0.02, 1.0, 0.04)
    ys = np.arange(0.02, 1.0, 0.04)
    xx, yy = np.meshgrid(xs, ys)
    ax_bg.scatter(xx.ravel(), yy.ravel(), s=1.5, color='#D1D5DB', zorder=-9, linewidths=0)

def finalize(fig, path):
    """Save with consistent settings."""
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  ✓ Saved {path}')

# ─── Figure 1 & 2: GNN Pre-training + Physics Loss ───────────────────────────
def plot_training_curves():
    checkpoint_path = Path('results/small_scale_demo/checkpoint.pth')
    if not checkpoint_path.exists():
        print('  Checkpoint not found, skipping training curves.')
        return
    print('Generating training curves...')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    history     = checkpoint.get('training_history', {})
    pre_losses  = history.get('pretrain_losses', [])
    phy_losses  = history.get('physics_losses', [])

    # GNN pretrain
    if pre_losses:
        setup_style()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        add_dot_grid(fig)
        xs = range(1, len(pre_losses) + 1)
        ax.fill_between(xs, pre_losses, alpha=0.18, color=C_PURPLE_D)
        ax.plot(xs, pre_losses, color=C_PURPLE_D, linewidth=2.2, marker='o',
                markersize=5, markerfacecolor=CARD, markeredgecolor=C_PURPLE_D,
                markeredgewidth=1.5)
        ax.set_title('GNN Pre-training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.spines[['top','right']].set_visible(False)
        finalize(fig, OUTPUT_DIR / 'gnn_pretrain_loss.png')

    # Physics loss
    if phy_losses:
        setup_style()
        fig, ax = plt.subplots(figsize=(7, 4.2))
        add_dot_grid(fig)
        xs = range(1, len(phy_losses) + 1)
        ax.fill_between(xs, phy_losses, alpha=0.18, color=C_ORANGE_D)
        ax.plot(xs, phy_losses, color=C_ORANGE_D, linewidth=2.2, marker='s',
                markersize=5, markerfacecolor=CARD, markeredgecolor=C_ORANGE_D,
                markeredgewidth=1.5)
        ax.set_title('Physics-Informed Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.spines[['top','right']].set_visible(False)
        finalize(fig, OUTPUT_DIR / 'physics_training_loss.png')

# ─── Figure 3: Baseline Comparison ───────────────────────────────────────────
def plot_baseline_comparison():
    results_path = Path('results/baselines/comparison_results.json')
    if not results_path.exists():
        print('  Baseline results not found, skipping.')
        return
    print('Generating baseline comparison chart...')

    with open(results_path) as f:
        data = json.load(f)

    name_map = {
        'pimaluos':             'PIMALUOS',
        'pimaluos_no_physics':  'No Physics',
        'pimaluos_no_gnn':      'No GNN',
        'greedy':               'Greedy',
        'random':               'Random',
        'rule_based':           'Rule-Based',
    }
    # Assign one pastel color per method; PIMALUOS gets the signature blue
    color_map = {
        'PIMALUOS':    C_BLUE_D,
        'No Physics':  C_ORANGE_D,
        'No GNN':      C_PINK_D,
        'Greedy':      C_PURPLE_D,
        'Random':      C_SLATE,
        'Rule-Based':  C_GREEN_D,
    }
    pastel_map = {
        'PIMALUOS':    C_BLUE,
        'No Physics':  C_ORANGE,
        'No GNN':      C_PINK,
        'Greedy':      C_PURPLE,
        'Random':      '#E2E8F0',
        'Rule-Based':  C_GREEN,
    }

    methods, values = [], []
    for key, val in data.items():
        label = name_map.get(key, key.replace('_', ' ').title())
        methods.append(label)
        ev = val['mean']['economic_value'] if 'mean' in val else val.get('economic_value', 0)
        values.append(ev)

    df = pd.DataFrame({'Method': methods, 'Economic Value': values})
    # Sort: PIMALUOS first, then descending
    order = ['PIMALUOS', 'No Physics', 'No GNN', 'Greedy', 'Random', 'Rule-Based']
    df['sort'] = df['Method'].apply(lambda x: order.index(x) if x in order else 99)
    df = df.sort_values('sort')

    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    add_dot_grid(fig)

    bar_colors  = [pastel_map.get(m, C_SLATE) for m in df['Method']]
    edge_colors = [color_map.get(m, MONO)    for m in df['Method']]

    bars = ax.bar(df['Method'], df['Economic Value'],
                  color=bar_colors, edgecolor=edge_colors,
                  linewidth=1.2, width=0.55, zorder=3)

    # Value labels above bars
    for bar, val in zip(bars, df['Economic Value']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(df['Economic Value']) * 0.004),
                f'{int(val):,}',
                ha='center', va='bottom', fontsize=8.5,
                color=TEXT, fontweight='semibold')

    # Highlight PIMALUOS bar
    ymin = min(values) * 0.998
    ymax = max(values) * 1.022
    ax.set_ylim(ymin, ymax)
    ax.set_title('Economic Value Comparison — PIMALUOS vs Baselines')
    ax.set_ylabel('Total Economic Value (USD)')
    ax.set_xlabel('')
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x', labelrotation=15)
    fig.tight_layout()
    finalize(fig, OUTPUT_DIR / 'baseline_comparison.png')

# ─── Figure 4: Edge Ablation ──────────────────────────────────────────────────
def plot_edge_ablation():
    results_path = Path('results/ablation/edge_types_impact.json')
    if not results_path.exists():
        print('  Ablation results not found, skipping.')
        return
    print('Generating edge ablation chart...')

    with open(results_path) as f:
        data = json.load(f)

    configs, losses = [], []
    for key, val in data.items():
        if 'error' in val:
            continue
        configs.append(key.replace('_', ' ').title())
        losses.append(val['final_physics_loss'])

    df = pd.DataFrame({'Config': configs, 'Loss': losses}).sort_values('Loss')

    # Cycle through pastel palette
    palette = [C_GREEN, C_BLUE, C_PURPLE, C_PINK, C_ORANGE, C_TEAL, C_SLATE]
    palette_d = [C_GREEN_D, C_BLUE_D, C_PURPLE_D, C_PINK_D, C_ORANGE_D, '#14B8A6', MONO]
    colors_p = [palette[i % len(palette)]   for i in range(len(df))]
    colors_d = [palette_d[i % len(palette_d)] for i in range(len(df))]

    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    add_dot_grid(fig)

    bars = ax.bar(df['Config'], df['Loss'],
                  color=colors_p, edgecolor=colors_d,
                  linewidth=1.2, width=0.5, zorder=3)

    for bar, val in zip(bars, df['Loss']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(df['Loss']) * 0.008,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=8.5,
                color=TEXT, fontweight='semibold')

    ax.set_ylim(0, max(df['Loss']) * 1.18)
    ax.set_title('Impact of Edge Types on Physics-Informed Loss  (lower is better)')
    ax.set_ylabel('Final Physics Loss')
    ax.set_xlabel('')
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x', labelrotation=12)
    fig.tight_layout()
    finalize(fig, OUTPUT_DIR / 'edge_ablation.png')

# ─── Figure 5: Physics Weight Sweep ──────────────────────────────────────────
def plot_physics_sweep():
    results_path = Path('results/ablation/physics_weight_sweep.json')
    if not results_path.exists():
        print('  Physics sweep results not found, skipping.')
        return
    print('Generating physics weight sweep...')

    with open(results_path) as f:
        data = json.load(f)

    lambdas, losses = [], []
    for val in data.values():
        if 'error' in val:
            continue
        lambdas.append(val['physics_weight'])
        losses.append(val['final_physics_loss'])

    idx = sorted(range(len(lambdas)), key=lambda i: lambdas[i])
    lambdas = [lambdas[i] for i in idx]
    losses  = [losses[i]  for i in idx]

    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    add_dot_grid(fig)

    ax.fill_between(lambdas, losses, alpha=0.15, color=C_BLUE_D)
    ax.plot(lambdas, losses, color=C_BLUE_D, linewidth=2.2,
            marker='D', markersize=6,
            markerfacecolor=CARD, markeredgecolor=C_BLUE_D, markeredgewidth=1.5)

    ax.set_title('Effect of Physics Weight (λ) on Final Loss')
    ax.set_xlabel('Physics Penalty Weight  λ')
    ax.set_ylabel('Final Physics Loss')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    finalize(fig, OUTPUT_DIR / 'physics_weight_sweep.png')

# ─── Figure 6 (already done): Pareto frontier ─────────────────────────────────
def restyle_pareto():
    """Re-render the Pareto frontier to match the design language."""
    results_path = Path('results/small_scale_demo/checkpoint.pth')
    if not results_path.exists():
        print('  Checkpoint not found, skipping Pareto re-style.')
        return
    print('Re-styling Pareto frontier...')

    checkpoint = torch.load(results_path, map_location='cpu')
    pareto = checkpoint.get('pareto_solutions', None)
    if pareto is None:
        print('  No pareto_solutions in checkpoint, skipping.')
        return

    econ = [p[0] for p in pareto]
    envs = [p[1] for p in pareto]
    soc  = [p[2] if len(p) > 2 else 0.6 for p in pareto]

    # Find knee (closest to utopia)
    e_n = (np.array(econ) - min(econ)) / (max(econ) - min(econ) + 1e-9)
    v_n = (np.array(envs) - min(envs)) / (max(envs) - min(envs) + 1e-9)
    dists = np.sqrt((1 - e_n)**2 + (1 - v_n)**2)
    knee  = int(np.argmin(dists))

    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    add_dot_grid(fig)

    sc = ax.scatter(econ, envs, c=soc, cmap='cool', s=70, alpha=0.80,
                    edgecolors=BORDER, linewidths=0.8, zorder=3)

    ax.scatter(econ[knee], envs[knee], marker='*', s=280,
               color='#F43F5E', edgecolors=CARD,
               linewidths=1.2, zorder=5, label='Knee Solution')

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label('Social Score  (Amenity Access / Housing)', color=MUTED, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=8)

    ax.set_title('Multi-Objective Pareto Frontier — PIMALUOS')
    ax.set_xlabel('Economic Score  (FAR & Development Utilization) →')
    ax.set_ylabel('Environmental Score  (Green Space & Runoff Mitigation) →')
    ax.legend(loc='lower left', fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    finalize(fig, OUTPUT_DIR / 'pareto_frontier.png')


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Generating PIMALUOS-styled figures → {OUTPUT_DIR}')
    plot_training_curves()
    plot_baseline_comparison()
    plot_edge_ablation()
    plot_physics_sweep()
    restyle_pareto()
    print('Done.')

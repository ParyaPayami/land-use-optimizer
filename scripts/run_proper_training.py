#!/usr/bin/env python3
"""
Re-run PIMALUOS training with proper epoch counts and regenerate figures.

GNN pretrain:  50 epochs  (was 10)
Physics:       30 epochs  (was 5)
MARL:          50 iters   (was 20)
"""

import sys
import time
import json
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from pimaluos import UrbanOptSystem

# ─── Config ──────────────────────────────────────────────────────────────────
OUT_DIR     = Path('results/proper_training')
FIG_DIR     = Path('results/figures')
CHECKPOINT  = OUT_DIR / 'checkpoint.pth'
N_PARCELS   = 100
GNN_EPOCHS  = 500
PHY_EPOCHS  = 150   # converges ~ep50, stop at 3× to confirm plateau
MARL_ITERS  = 200
MARL_STEPS  = 20

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

# ─── Design tokens (match architecture.html) ─────────────────────────────────
BG    = '#F3F4F6'
CARD  = '#FFFFFF'
BORDER= '#E5E7EB'
TEXT  = '#1F2937'
MUTED = '#6B7280'

def setup_style():
    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.sans-serif':  ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size':        11,
        'axes.titlesize':   13,
        'axes.titleweight': 'bold',
        'axes.titlecolor':  TEXT,
        'axes.labelcolor':  MUTED,
        'axes.edgecolor':   BORDER,
        'axes.linewidth':   0.8,
        'axes.facecolor':   CARD,
        'axes.grid':        True,
        'grid.color':       BORDER,
        'grid.linewidth':   0.6,
        'grid.linestyle':   '--',
        'xtick.color':      MUTED,
        'ytick.color':      MUTED,
        'figure.facecolor': BG,
        'text.color':       TEXT,
        'legend.facecolor': CARD,
        'legend.edgecolor': BORDER,
        'savefig.facecolor':BG,
        'savefig.dpi':      300,
    })

def add_dot_grid(fig):
    ax = fig.add_axes([0, 0, 1, 1], zorder=-10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    xs = np.arange(0.02, 1.0, 0.04)
    ys = np.arange(0.02, 1.0, 0.04)
    xx, yy = np.meshgrid(xs, ys)
    ax.scatter(xx.ravel(), yy.ravel(), s=1.5, color='#D1D5DB', zorder=-9, linewidths=0)

# ─── Step 1: Run training ─────────────────────────────────────────────────────
log.info("=" * 60)
log.info(f"  PIMALUOS Proper Training")
log.info(f"  GNN={GNN_EPOCHS}ep  Physics={PHY_EPOCHS}ep  MARL={MARL_ITERS}it")
log.info("=" * 60)

t0 = time.time()

system = UrbanOptSystem(
    city='manhattan',
    data_subset_size=N_PARCELS,
    llm_mode='mock',
    device='cpu',
    random_seed=42,
)

log.info("Loading data...")
system.load_data()
system.build_graph()
system.extract_constraints()

log.info(f"Pre-training GNN for {GNN_EPOCHS} epochs...")
system.pretrain_gnn(
    num_epochs=GNN_EPOCHS,
    learning_rate=5e-4,   # higher than the old 1e-4
    grad_clip=1.0,
)

log.info(f"Physics-informed training for {PHY_EPOCHS} epochs...")
system.train_with_physics_feedback(
    num_epochs=PHY_EPOCHS,
    learning_rate=5e-4,
    physics_weight=0.3,
)

log.info(f"MARL optimisation for {MARL_ITERS} iterations...")
trainer = system.optimize_with_marl(
    num_iterations=MARL_ITERS,
    steps_per_iteration=MARL_STEPS,
)

log.info("Saving checkpoint...")
system.save_checkpoint(CHECKPOINT)

elapsed = time.time() - t0
log.info(f"Total training time: {elapsed:.1f}s")

# ─── Step 2: Regenerate GNN pretrain loss figure ──────────────────────────────
pre = system.training_history['pretrain_losses']
phy = system.training_history['physics_losses']

log.info(f"GNN pretrain: {len(pre)} epochs  {pre[0]:.4f} → {pre[-1]:.4f}  "
         f"({100*(pre[0]-pre[-1])/pre[0]:.1f}% drop)")
log.info(f"Physics:      {len(phy)} epochs  {phy[0]:.4f} → {phy[-1]:.4f}  "
         f"({100*(phy[0]-phy[-1])/phy[0]:.1f}% drop)")

setup_style()

# ── GNN pretrain ──
fig, ax = plt.subplots(figsize=(7, 4.2))
add_dot_grid(fig)
xs = range(1, len(pre) + 1)
ax.fill_between(xs, pre, alpha=0.14, color='#8B5CF6')
ax.plot(xs, pre, color='#8B5CF6', linewidth=2.2,
        marker='o', markersize=4, markerfacecolor=CARD,
        markeredgecolor='#8B5CF6', markeredgewidth=1.4)
# zoom y-axis to actual range
ylo = min(pre) * 0.998; yhi = max(pre) * 1.002
ax.set_ylim(ylo, yhi)
ax.set_title('GNN Pre-training Loss  (Feature Reconstruction, Huber)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Loss')
ax.spines[['top', 'right']].set_visible(False)

delta = pre[0] - pre[-1]
# annotation at midpoint of curve
mid = len(pre) // 2
ax.annotate(
    f'−{delta:.4f}  ({100*delta/pre[0]:.1f}% reduction)',
    xy=(len(pre), pre[-1]),
    xytext=(max(3, len(pre) - len(pre)//3), pre[0] - (pre[0]-pre[-1])*0.25),
    fontsize=9, color='#8B5CF6',
    arrowprops=dict(arrowstyle='->', color='#8B5CF6', lw=1.2),
)
fig.tight_layout()
fig.savefig(FIG_DIR / 'gnn_pretrain_loss.png', dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
log.info("Saved gnn_pretrain_loss.png")

# ── Physics loss ──
setup_style()
fig, ax = plt.subplots(figsize=(7, 4.2))
add_dot_grid(fig)
xp = range(1, len(phy) + 1)
ax.fill_between(xp, phy, alpha=0.14, color='#F59E0B')
ax.plot(xp, phy, color='#F59E0B', linewidth=2.2,
        marker='s', markersize=5, markerfacecolor=CARD,
        markeredgecolor='#F59E0B', markeredgewidth=1.4)

ax.set_title(f'Physics-Informed Training Loss  ({len(phy)} Epochs)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Physics Penalty Loss')
ax.spines[['top', 'right']].set_visible(False)

delta_p = phy[0] - phy[-1]
ax.annotate(
    f'−{delta_p:.4f}  ({100*delta_p/phy[0]:.1f}% reduction)',
    xy=(len(phy), phy[-1]),
    xytext=(max(2, len(phy) // 2 - 2), phy[0] - (phy[0]-phy[-1])*0.3),
    fontsize=9, color='#F59E0B',
    arrowprops=dict(arrowstyle='->', color='#F59E0B', lw=1.2),
)
fig.tight_layout()
fig.savefig(FIG_DIR / 'physics_training_loss.png', dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
log.info("Saved physics_training_loss.png")

# ─── Step 3: Save summary JSON ────────────────────────────────────────────────
summary = {
    'gnn_epochs': len(pre),
    'physics_epochs': len(phy),
    'gnn_loss_start': pre[0],
    'gnn_loss_end': pre[-1],
    'gnn_pct_drop': round(100*(pre[0]-pre[-1])/pre[0], 2),
    'physics_loss_start': phy[0],
    'physics_loss_end': phy[-1],
    'physics_pct_drop': round(100*(phy[0]-phy[-1])/phy[0], 2),
    'total_time_s': round(elapsed, 1),
}
with open(OUT_DIR / 'training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

log.info("=" * 60)
log.info("  DONE — figures saved to results/figures/")
log.info(f"  GNN drop:     {summary['gnn_pct_drop']}%")
log.info(f"  Physics drop: {summary['physics_pct_drop']}%")
log.info("=" * 60)

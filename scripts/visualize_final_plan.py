"""
visualize_final_plan.py
Side-by-side Current vs Optimised land-use map for Manhattan 42,075 parcels.
Dark background — the user confirmed this gives the best contrast for the map.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ─── Dark-theme config for the map only ───────────────────────────────────────
MAP_BG   = '#0D1117'   # near-black
CARD_BG  = '#161B22'   # slightly lighter card
GRID_CLR = '#21262D'   # very subtle grid
TEXT_CLR = '#E6EDF3'   # off-white
MUTED    = '#8B949E'

# Pastel / neon that pops on dark — consistent with architecture.html hue families
COLOR_MAP = {
    'Residential': '#60A5FA',   # blue  (Orchestration layer)
    'Commercial':  '#F472B6',   # pink  (Equity / MARL)
    'Industrial':  '#94A3B8',   # slate (neutral)
    'Mixed-Use':   '#A78BFA',   # purple (Physics layer)
    'Public':      '#FBBF24',   # amber (Developer / Zoning)
    'Open Space':  '#34D399',   # green (Spatial layer)
}


def main():
    csv_path = Path('results/full_manhattan/manhattan_landuse_plan_42k.csv')
    if not csv_path.exists():
        print(f'Error: {csv_path} not found.')
        return

    print('Loading Manhattan plan...')
    df = pd.read_csv(csv_path).dropna(subset=['lat', 'lon'])
    print(f'  {len(df):,} parcels loaded.')

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(22, 13), facecolor=MAP_BG)

    for ax, col, title in zip(
        axes,
        ['current_use_label', 'proposed_use_label'],
        ['Current Land Use', 'PIMALUOS — Optimised Land Use']
    ):
        ax.set_facecolor(CARD_BG)
        ax.grid(True, color=GRID_CLR, linestyle='--', alpha=0.5, linewidth=0.5)

        for label, color in COLOR_MAP.items():
            sub = df[df[col] == label]
            # Two-pass glow: translucent halo + sharp core
            ax.scatter(sub['lon'], sub['lat'], c=color, s=3.5,
                       alpha=0.12, edgecolors='none', zorder=2)
            ax.scatter(sub['lon'], sub['lat'], c=color, s=0.6,
                       alpha=0.92, edgecolors='none', label=label, zorder=3)

        ax.set_title(title, color=TEXT_CLR, fontsize=18, fontweight='bold', pad=16)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel('Longitude', color=MUTED, fontsize=10, labelpad=6)
        ax.set_ylabel('Latitude',  color=MUTED, fontsize=10, labelpad=6)
        for spine in ax.spines.values():
            spine.set_color(GRID_CLR)
            spine.set_linewidth(0.8)
        ax.set_aspect('equal', adjustable='box')

    # ── Unified legend ────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor=color, markersize=10, markeredgecolor='none')
        for label, color in COLOR_MAP.items()
    ]
    legend = fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=6,
        fontsize=13,
        facecolor='#1C2128',
        edgecolor=GRID_CLR,
        bbox_to_anchor=(0.5, 0.03),
        framealpha=0.95,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_CLR)
        text.set_fontweight('bold')

    fig.suptitle(
        'PIMALUOS — Manhattan Spatial Land Use Optimisation  (42,075 Parcels)',
        color=TEXT_CLR, fontsize=23, fontweight='bold', y=0.96
    )
    plt.subplots_adjust(bottom=0.13, top=0.90, wspace=0.10)

    out = Path('results/figures/final_land_use_map.png')
    print(f'Saving → {out}')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=MAP_BG)
    plt.close()
    print('✓ Done.')


if __name__ == '__main__':
    main()

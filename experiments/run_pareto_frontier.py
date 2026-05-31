"""
PIMALUOS Pareto Frontier Optimization Run

Runs NSGA-II to discover Pareto-optimal land-use configurations across 
conflicting stakeholder objectives and exports the Pareto front plot.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pimaluos.system import UrbanOptSystem
from pimaluos.models.pareto import ParetoOptimizer


def run_pareto():
    print("=" * 70)
    print("RUNNING MULTI-OBJECTIVE PARETO FRONTIER OPTIMIZATION (NSGA-II)")
    print("=" * 70)

    # Initialize system
    system = UrbanOptSystem(
        city="manhattan", data_subset_size=100, device="cpu", random_seed=42
    )
    system.load_data()
    system.build_graph()
    system.extract_constraints()
    system.initialize_physics_engine()

    # Initialize Pareto Optimizer
    optimizer = ParetoOptimizer(
        num_parcels=100,
        constraint_masks=system.constraint_masks,
        physics_engine=system.physics_engine,
        gdf=system.gdf,
    )

    # Run optimization (small population and generations for speed)
    solutions = optimizer.optimize(population_size=30, num_generations=10, seed=42)

    # Find knee solution
    knee = optimizer.find_knee_solution(solutions)

    # Create directories
    output_dir = Path("results/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Export CSV
    csv_path = output_dir / "pareto_front.csv"
    optimizer.export_pareto_front(solutions, str(csv_path))

    # Plot Pareto Front (Economic vs Environmental score)
    objectives = np.array([sol.objectives for sol in solutions])
    economic = objectives[:, 0]
    environmental = objectives[:, 1]
    social = objectives[:, 2]
    equity = objectives[:, 3]

    # Clean styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Scatter plot with Social score as color mapping
    sc = ax.scatter(
        economic,
        environmental,
        c=social,
        cmap="viridis",
        s=80,
        alpha=0.85,
        edgecolors="none",
        label="Pareto Solutions",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Social Score (Amenity Access / Housing)", fontsize=10)

    # Highlight knee solution (compromise point)
    knee_obj = knee.objectives
    ax.scatter(
        knee_obj[0],
        knee_obj[1],
        color="#e63946",
        marker="*",
        s=250,
        edgecolors="black",
        linewidths=1.5,
        label="Knee Solution (Best Compromise)",
        zorder=5,
    )

    # Label details
    ax.set_title(
        "PIMALUOS Multi-Objective Pareto Frontier (100 Manhattan Parcels)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Economic Score (FAR & Development Utilization) →", fontsize=10)
    ax.set_ylabel("Environmental Score (Green Space & Runoff Mitigation) →", fontsize=10)
    
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none")
    
    plt.tight_layout()

    plot_path = fig_dir / "pareto_frontier.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    print(f"\n✓ Pareto front CSV saved to {csv_path}")
    print(f"✓ Pareto frontier plot saved to {plot_path}")
    print("=" * 70)
    print("PARETO FRONTIER GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_pareto()

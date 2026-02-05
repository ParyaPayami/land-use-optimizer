# PIMALUOS Jupyter Notebooks

This directory contains Jupyter notebooks for visualizing and analyzing PIMALUOS results.

## Notebooks

### 1. Training Visualization (`01_training_visualization.ipynb`)

Visualizes the GNN training process:
- Pre-training loss curves
- Physics-informed training loss curves
- Action distribution analysis
- Performance metrics

**Prerequisites:**
- Run `demo_small_scale.py` first to generate results

**Usage:**
```bash
jupyter notebook 01_training_visualization.ipynb
```

### 2. Pareto Optimization (`02_pareto_optimization.ipynb`)

Demonstrates multi-objective optimization:
- Runs NSGA-II Pareto optimization
- 2D projections of 4D Pareto front
- 3D visualization colored by 4th objective
- Trade-off analysis and correlations

**Prerequisites:**
- Install pymoo: `pip install pymoo`
- Manhattan data downloaded

**Usage:**
```bash
jupyter notebook 02_pareto_optimization.ipynb
```

## Installation

Install Jupyter and required packages:

```bash
pip install jupyter matplotlib seaborn plotly pymoo
```

## Running All Notebooks

To run all notebooks and generate figures:

```bash
jupyter nbconvert --to notebook --execute 01_training_visualization.ipynb
jupyter nbconvert --to notebook --execute 02_pareto_optimization.ipynb
```

## Output

Notebooks save figures to:
- `../results/small_scale_demo/` - Training visualizations
- `../results/` - Pareto front visualizations

All figures are saved at 300 DPI for publication quality.

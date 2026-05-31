#!/bin/bash

# reproduce_all.sh
# Automates the entire PIMALUOS empirical pipeline from data downloading to figure generation.

set -e

echo "========================================================"
echo "    PIMALUOS Reproducibility & Validation Pipeline      "
echo "========================================================"

# 1. Setup Environment
echo -e "\n[1/5] Setting up directories..."
mkdir -p results/baselines results/marl_validation results/figures
export PYTHONPATH=".:$PYTHONPATH"

# 2. Pre-train GNN & Run Baseline Comparisons (No GNN Fix included)
# We use a 500 parcel subset for reasonably fast execution on standard hardware
echo -e "\n[2/5] Running Baseline Comparisons & GNN Pre-training (500 parcels)..."
python experiments/run_baseline_comparisons.py --data_subset 500 --num_runs 1

# 3. MARL Validation (Reward Convergence & Agent Ablations)
echo -e "\n[3/5] Running Deep MARL Validation..."
python experiments/run_marl_validation.py

# 4. Generate Core Figures
echo -e "\n[4/5] Generating Core Result Figures..."
python scripts/generate_all_figures.py

# 5. Generate Difference Maps (High Res Vectors)
echo -e "\n[5/5] Generating High-Resolution Difference Maps..."
python scripts/generate_difference_maps.py

echo -e "\n========================================================"
echo " Pipeline Complete. All results and figures are located in 'results/' "
echo "========================================================"

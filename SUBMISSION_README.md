# PIMALUOS Submission Package

**Date:** January 7, 2026

This directory contains the artifacts for the "Computers, Environment and Urban Systems" (CEUS) submission.

## 📦 Contents

1.  **Manuscript Source**:
    *   `manuscript tex/paper.tex` (Original)
    *   `MANUSCRIPT_CHANGES.tex` (**NEW**: Copy content from here into paper.tex)
    *   `results/figures/*.png` (Hi-res figures)

2.  **Code Archive**:
    *   `pimaluos/` (Core library)
    *   `scripts/` (Reproduction scripts)
    *   `demo_small_scale.py` (Main demo)
    *   `requirements.txt`

3.  **Data**:
    *   `results/small_scale_demo/` (Raw results)
    *   `results/baselines/` (Baseline data)
    *   `results/ablation/` (Ablation data)

## 🚀 Reproduction Instructions

To reproduce the results presented in the paper:

1.  **Install**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Demo**:
    ```bash
    python demo_small_scale.py
    ```

3.  **Run Baselines**:
    ```bash
    python experiments/run_baseline_comparisons.py --data_subset 100
    ```

4.  **Run Ablation**:
    ```bash
    python experiments/ablation_edge_types.py
    python experiments/ablation_physics_weight.py
    ```

5.  **Generate Figures**:
    ```bash
    python scripts/generate_all_figures.py
    ```

## 📝 Manuscript Update Guide

1.  Open `manuscript tex/paper.tex`.
2.  Replace the placeholder **Results** section (Section 5.2) with the content from `MANUSCRIPT_CHANGES.tex`.
3.  Add the **Baseline Methods** text to Section 5.1.
4.  Ensure `results/figures/` is in the LaTeX path or move images to the root image directory.

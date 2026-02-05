# Running PIMALUOS Demo

## Prerequisites

### 1. Install Dependencies

**Core dependencies (required):**
```bash
source .venv/bin/activate
pip install geopandas pandas numpy scipy scikit-learn pydantic pydantic-settings networkx matplotlib seaborn
```

**ML dependencies (for running demo):**
```bash
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse
pip install pymoo  # For Pareto optimization
```

### 2. Download Data

If you haven't already:
```bash
python download_manhattan_data_simple.py
```

## Running the Demo

### Small-Scale Demo (100 parcels, ~5-10 minutes)

```bash
python demo_small_scale.py
```

**What it does:**
- Loads 100 Manhattan parcels
- Builds heterogeneous graph (5 edge types)
- Pre-trains GNN (10 epochs)
- Physics-informed training (5 epochs)
- MARL optimization (20 iterations)
- Generates final land-use plan

**Output:**
- `results/small_scale_demo/final_plan.csv` - Land-use recommendations
- `results/small_scale_demo/checkpoint.pth` - Model checkpoint
- `results/small_scale_demo/results.json` - Performance metrics
- `results/small_scale_demo/demo.log` - Execution log

### Medium-Scale Demo (1,000 parcels, ~15-30 minutes)

```bash
python demo_complete_pipeline.py
```

**What it does:**
- Same as small-scale but with 1,000 parcels
- More epochs (50 pre-train, 20 physics, 100 MARL)
- Better quality results

**Output:**
- `output/final_plan.csv`
- `output/checkpoint.pth`

## Visualizing Results

### Using Jupyter Notebooks

```bash
cd notebooks
jupyter notebook
```

**Open and run:**
1. `01_training_visualization.ipynb` - Training curves and metrics
2. `02_pareto_optimization.ipynb` - Multi-objective analysis

**Generated figures:**
- `training_curves.png` - Loss over epochs
- `action_distribution.png` - FAR modifications
- `pareto_front_2d.png` - Objective trade-offs
- `pareto_front_3d.png` - 3D Pareto visualization
- `objective_correlations.png` - Correlation heatmap

## Troubleshooting

### Error: "No module named 'torch'"

**Solution:** Install PyTorch
```bash
pip install torch torchvision torchaudio
```

For CPU-only (smaller download):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Error: "No module named 'torch_geometric'"

**Solution:** Install PyTorch Geometric
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Error: "No module named 'pymoo'"

**Solution:** Install pymoo for Pareto optimization
```bash
pip install pymoo
```

### Out of Memory

**Solutions:**
1. Reduce dataset size:
   ```python
   system = UrbanOptSystem(data_subset_size=50)  # Even smaller
   ```

2. Close other applications
3. Use a machine with more RAM

### Slow Execution

**Solutions:**
1. Reduce training epochs:
   ```python
   system.pretrain_gnn(num_epochs=5)  # Less epochs
   ```

2. Use GPU if available:
   ```python
   system = UrbanOptSystem(device='cuda')
   ```

## Expected Results

### Small-Scale (100 parcels)

**Execution time:** 5-10 minutes
**Final losses:**
- Pre-training: ~0.18-0.25
- Physics training: ~0.12-0.20

**Action distribution:**
- Decrease FAR: ~24%
- Maintain FAR: ~48%
- Increase FAR: ~28%

### Medium-Scale (1,000 parcels)

**Execution time:** 15-30 minutes
**Final losses:**
- Pre-training: ~0.12-0.18
- Physics training: ~0.08-0.15

**Action distribution:**
- More refined based on actual zoning patterns

## For Manuscript

After running demos, report:

1. **Hardware specs**
   - CPU model
   - RAM amount
   - GPU (if used)

2. **Execution time**
   - From results.json

3. **Training metrics**
   - Final losses
   - Convergence behavior

4. **Action distribution**
   - Percentage by action type
   - Spatial patterns

5. **Figures**
   - Include generated PNGs in manuscript

## Next Steps

After successful demo execution:

1. ✅ Verify results quality
2. ✅ Generate visualizations
3. ✅ Document performance
4. ✅ Update manuscript with actual numbers
5. ✅ Compare with baselines (Week 3)
6. ✅ Add ablation studies (Week 3)

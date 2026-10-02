# PIMALUOS

**Multi-agent floor-area optimisation on parcel graphs with planning-scale capacity screens.**

PIMALUOS takes a parcel layer (NYC MapPLUTO for Manhattan out of the box) and:

| Layer | What it does | What it is *not* |
|---|---|---|
| **Sense**: `pimaluos.core` | Standardises MapPLUTO, engineers lot features (all from MapPLUTO columns or geometry), and builds a graph with five symmetric relation types: shared boundary, k-nearest proximity, identical land use among neighbours, same street, same zoning district. | Not a line-of-sight model; "proximity" is a distance relation. |
| **Knowledge**: `pimaluos.knowledge` | LLM-RAG extraction of district base FAR limits from Zoning Resolution text, evaluated against the limits DCP publishes in MapPLUTO (`pimaluos rag-benchmark`). | The optimisation uses MapPLUTO's lot-level limits, not LLM output. |
| **Reason**: `pimaluos.models` | Self-supervised multi-relational GAT embeddings; five stakeholder agents (PPO, shared policy per stakeholder) propose ±0.5 FAR per lot; weighted plurality vote (ties → status quo); NSGA-III over the same decision space; voting-game analysis. | Land use is not changed; the decision is floor area (FAR) per lot. |
| **Verify**: `pimaluos.physics` | Vectorised screens: BPR travel-time index on frontage-based cell capacity, Rational-Method combined-sewer load vs existing load + headroom, winter-solstice noon shadow screen; a repair loop removes added floor area that causes violations. | Not traffic assignment, hydraulic sewer modelling or ray-traced shadows. |

Zoning compliance is a **hard bound**: existing FAR ≤ planned FAR ≤ max(zoning max FAR, existing FAR).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA/MPS build
pip install -e ".[dev]"            # add ",rag" for the LLM-RAG benchmark
pytest -q                          # unit + end-to-end smoke tests (synthetic city)
```

`requirements-lock.txt` lists the exact versions the tests were run with.

## Data

Download **MapPLUTO** (shapefile or file geodatabase) from NYC Department of City Planning,
<https://www.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page>. Record the release
(e.g. 24v4) in `configs/paper.yaml`. The loader keeps Manhattan lots (`BoroCode == 1`) and reprojects
to EPSG:2263 (US feet).

For the RAG benchmark you also need the text of the NYC Zoning Resolution (PDF or text) for the
version matching your MapPLUTO release, and an API key or a local Ollama model.

## Reproduce the manuscript

```bash
# 1. All experiments (seeds, ablations, Nash, voting sensitivity, NSGA-III)
pimaluos run --config configs/paper.yaml --pluto /path/to/MapPLUTO.zip --out results/paper

# 2. Optional: RAG extraction benchmark
pimaluos rag-benchmark --pluto /path/to/MapPLUTO.zip --zr-dir /path/to/zoning_resolution \
    --provider openai --model gpt-4o --out results/rag

# 3. Figures, LaTeX tables and every number quoted in the text
pimaluos report --results results/paper --rag results/rag --out paper/generated

# 4. Compile
cd paper && latexmk -pdf FINAL_SUBMISSION.tex
```

Every number in the manuscript is a macro written by step 3. Before step 3 has run, the PDF
shows **[TBD]** in their place. `results/paper/manifest.json` records the data file, feature
list, configuration, package versions, hardware and wall-clock time of every stage.

`configs/smoke.yaml` runs the whole pipeline on a small **synthetic** city in seconds. It is
used by CI, and its numbers are not results.

## Outputs

`results/<run>/`: `manifest.json`, `graph_summary.json`, `gnn.json` (train/validation curves),
`edge_ablation.json`, `marl.json`, `metrics.csv` (seed × method × verified × metric),
`nash.json`, `voting_sensitivity.json`, `pareto.json`, `plans/*.npz` (FAR per lot).

## Limitations

- The capacity screens are coarse, parameterised planning screens (see `CapacityParams`). Their
  absolute thresholds are assumptions; results should be read as relative comparisons between plans.
- Equity indicators are MapPLUTO-only proxies (assessed value per unit, open space per resident).
  No census data are joined.
- Only Manhattan is configured. Other cities need a YAML column mapping and recalibrated screens.

## Citation

See `CITATION.cff`. License: MIT.

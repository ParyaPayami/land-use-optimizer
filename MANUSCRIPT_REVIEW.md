# PIMALUOS — Pre-submission Review of Manuscript, Code, Results and References

> **Status (October 2026):** this review describes the repository at `59ef445`. The issues listed
> below were addressed in the subsequent commits on branch `claude/project-review-critique-wevctd`:
> the code was rewritten (see the commit message of `360f519`), the fabricated or unsupported artifacts
> were removed, the manuscript was rewritten in `paper/FINAL_SUBMISSION.tex` with every number generated
> by `pimaluos report`, and the bibliography was rebuilt from verified entries (`paper/references.bib`).
> The full Manhattan experiments still have to be run on the author's machine with MapPLUTO; until then
> the manuscript shows **[TBD]** in place of results.

**Manuscript:** "PIMALUOS: An Open-Source Physics-Informed Multi-Agent Framework for Urban Land-Use Optimization" (`FINAL_SUBMISSION.pdf`, byte-identical to the uploaded copy). Target journal: *Computers, Environment and Urban Systems* (CEUS).
**Code reviewed:** `main` @ `59ef445`. I also checked the history, especially `667f077`, `167a341` and `78e3972`.
**Reviewer stance:** this review is written the way a rigorous CEUS referee and a software-paper reproducibility auditor would read the submission. Every finding cites the file and line, or the committed artifact, it rests on.

> **Limits of this audit.** It is a static read of all source files plus a numerical inspection of every committed result file. I could not re-run the GNN/MARL pipeline in this sandbox: PyTorch wheels were blocked by the network proxy, and MapPLUTO is not in the repo. Zenodo and doi.org were also unreachable, so the two archive DOIs are unverified. None of the conclusions below depend on re-running anything. They follow from the code paths and from the result files the repository itself ships.

---

## 0. Bottom line

**Do not submit this manuscript in its current form.** The problems are not polish issues. Most headline numbers in the paper cannot be traced to anything in the repository, and several are contradicted by the repository's own result files and code:

1. The **42,075-parcel results** (Table 6, abstract, the Wilcoxon test, PoA, the equity ablation, the weight sweep, the 127-solution Pareto front, 92.6 % coverage) exist **only in the `.tex`**. No script or result file produces them. The committed baseline results are from a **500-parcel** run, and they show the **opposite** of the paper's main qualitative claim (§1, C1).
2. The **RAG "validation benchmark" (Table 2, κ = 0.92, two annotators)** is a **programmatically generated file**: the "RAG output" is copied from the "human annotation", and every item is marked as agreeing (C2).
3. The **"optimised Manhattan land-use map"** (Fig. 3b) and the dashboard data come from a "heuristic diversity injection". It randomly shuffles parcels into fixed 40/30/15/10/5 % quotas, and the "current land use" panel is `np.random.choice` (C3).
4. The **"physics-informed" training loss contains no physics**. It is `MSE(pred, 1) + λ·mean(sigmoid(pred)²)` (C4).
5. The **"total buildable floor area"** metric assumes every lot is 1,000 sq ft (a column-name bug), with FAR capped at 2.0 for all of Manhattan (C5, C6).

If any of the numbers in (1)–(2) do come from experiments that exist outside the repository, those experiments, logs and annotations must be released and the paper rewritten around what they actually show. If they do not, the claims must be removed. Submitting as is would expose the work to a research-integrity finding, not just a rejection. The good news is that the **software idea is reasonable** and could support an honest, publishable CEUS software paper. A recovery plan is in §11.

---

## 1. Critical issues (must be resolved before any submission)

### C1. Headline results are not produced by any code in the repository, and the committed results contradict them

| Paper claim | What the repository shows |
|---|---|
| Table 6, "3 independent full-scale seed runs on all 42,075 parcels", PIMALUOS 77,665,494 ± 25,482 sq ft, diversity 0.593 | `results/baselines/comparison_results.json` / `comparison_table.csv` are from a **500-parcel** run (`reproduce_all.sh:20`, `--data_subset 500 --num_runs 1`; values ≈ 9×10⁵). There, **PIMALUOS diversity = 0.003 ± 0.006** (every parcel gets action 0, "decrease"), and **No-GNN diversity = 0.874**. That is the reverse of the paper's 0.593 vs 0.000. |
| "PIMALUOS (No GNN) diversity 0.000" | Committed No-GNN: 0.874 ± 0.106 (highest of all methods). |
| Seeds 42, 202, 404 (§5.2.1) | Script uses `random_seed + run` → 42, 43, 44 (`experiments/run_baseline_comparisons.py:195`). |
| Wilcoxon z = −18.42, p < 0.0001 | No Wilcoxon test exists anywhere in the codebase (`grep -rniw wilcoxon` → none). |
| Cohen's d = 0.082 | The script computes a *run-level* d with pooled std (`run_baseline_comparisons.py:431-436`). Applied to the paper's own Table 6 numbers, that formula gives d ≈ 8.7, not 0.082. |
| Traffic exceedances 14 ± 2, runoff overflows 8 ± 1 for "No Physics" | `compute_metrics` (`run_baseline_comparisons.py:75-134`) computes **no** traffic, runoff or solar metric at all. |
| PoA = 0.940 (4,845.2 / 5,154.5) | `NashEquilibriumSolver` / `ParetoAnalyzer.price_of_anarchy` are never called by any experiment. They also cannot run against the current environment (see M9). |
| Equity ablation: Gini 0.245 → 0.382, displacement +38.4 % | No equity-weight ablation exists. No ACS/HVS/OSM/Parks data is loaded anywhere (Table 3 data sources are unused). |
| Weight sweep w_dev 0.15→0.45 ⇒ +1.8 % floor area, −5.0 % sewer compliance | No such sweep exists. `run_marl_validation.py` docstring lists it (item 4) but never implements it. |
| NSGA-III, 100 generations, 127 Pareto solutions, seeded from PPO policies/GNN embeddings | `experiments/run_pareto_frontier.py:25,41`: **100 parcels, NSGA-II, population 30, 10 generations, no seeding**. `results/baselines/pareto_front.csv` has **30 rows, all `feasible=False`**. |
| Pareto ranges (76.22–77.94 M sq ft; sewer 92.4→68.1 %; solar 91.2→98.7 %; congestion 1.38→1.04) | None of these quantities is an objective or an output of `pareto.py`. The exported objectives are unitless scores (economic 201–716). |
| Table 7 edge ablation "on all 42,075 parcels": All 1,433,904 edges, loss 0.3413, 90 min; Spatial-only 404,360, 0.3598, 82 min | `results/ablation/edge_types_table.csv` is a **100-parcel, 5-epoch** run: 5,888 vs 660 edges, **≈7 s** per config. The paper pairs these loss values with the 42K edge counts and invented runtimes. "Spatial only" in the code is **2** edge types, not 1 (`ablation_edge_types.py:60`). The loss is the *physics-stage* loss, not reconstruction loss. |
| Figs 2c/2d: "500 epochs … on all 42,075 parcels", loss 0.544→0.166 (−69.5 %), physics 0.365→0.159 | `scripts/run_proper_training.py:28` → `N_PARCELS = 100`. `training_summary.json` reports 650 epochs in 163 s. The actual 42K run (`results/full_manhattan/stage_3_gnn_pretrain.done`) used **30 epochs, final loss 3.92**, and physics fine-tuning **20 epochs**, not 500/150. |
| "Locked at commit 167a341" (Table 8, §5.4) | `167a341` contains **no `experiments/`, no `reproduce_all.sh`, no `results/baselines` or `results/ablation`**. They were added in `78e3972`, the same commit that first introduced the 42K numbers into the `.tex` with no backing result file. |

**Required action:** either release the full-scale runs (scripts, raw outputs, logs, seeds) that produce every number in the paper, or replace the numbers with the ones the code actually produces. In the second case the paper's narrative changes. On the committed evidence PIMALUOS *does* collapse to a monoculture, which the original code itself acknowledges (see C3).

### C2. The RAG "Zoning Extraction Validation Benchmark" is synthetic

`scripts/generate_zoning_benchmark.py` writes `data/zoning_rag_benchmark.json`:

- The 50 "sections of the official NYC Zoning Resolution" are **sentences written inside the script** (lines 15-71). Section numbers such as "Section 23-149 … R6" do not correspond to the actual ZR. Several values are wrong for the real ZR: R6 FAR is not simply 2.00, C5-2's 10.0 is labelled "commercial", and C4-3 is described with a "residential FAR".
- `"rag_extracted"` is **literally the same tuple** as `"human_annotations"`. The code comment reads "*Mock structured RAG output matching the ground truth perfectly (representing the high F1 performance)*" (lines 83-90). Every item has `complete_agreement: True` (lines 107-112).
- "OCR errors" are injected by string replacement (lines 76-81).
- The district counts are 25 R / 17 C / 8 M, not the paper's "18 residential, 12 commercial, 8 manufacturing". The items are an enumeration, not a "stratified random sample".

Nothing in the repository supports Table 2 (precision/recall/F1 per parameter), the "two independent senior urban planning researchers", Cohen's κ = 0.92, the third adjudicator, or "96.8 % → 93.5 % on nested overlays". The file in fact implies 100 % agreement. No NYC Zoning Resolution corpus ("800+ documents") is shipped, and the full pipeline runs in mock mode, where `MockLLM` returns `max_far: 2.0, max_height_ft: 65` for **every** zone (`pimaluos/knowledge/llm.py:186-202`). There is also no implementation of the claimed "two-stage overlay query" in `parser.py` or `rag.py`.

**Required action:** if a human-annotated evaluation was performed, publish the annotation files, annotator protocol, raw LLM outputs, scoring script and κ computation. If it was not, delete Table 2, the κ claim and the annotator description, and either run a real evaluation (see §11) or drop the RAG-accuracy claim.

### C3. The published "optimised Manhattan plan" (Fig. 3b, dashboard) is a random quota assignment

`results/full_manhattan/manhattan_landuse_plan_42k.csv` is identical to `results/full_scale_simulation/manhattan_landuse.csv`, which the dashboard reads:

- Proposed codes: 16,830 / 12,622 / 6,311 / 4,208 / 2,104 parcels. That is **exactly 40.0 / 30.0 / 15.0 / 10.0 / 5.0 % of 42,075** with quota rounding.
- Spatially, the codes are i.i.d.: 30,023 class changes between consecutive parcels, against ≈30,084 expected for a random permutation. They are statistically independent of the "current" use.
- Provenance: the original `generate_final_plan` (`git show 667f077:pimaluos/system.py`, ≈ lines 620-645) contains "**HEURISTIC FALLBACK: If agents converged to monoculture … inject diversity to ensure the demo is meaningful**". It does `np.random.shuffle(indices)` and assigns 40 % Res, 30 % Com, 15 % Mix, 10 % Public, 5 % Open.
- The "current land use" column is `np.random.choice(['Residential','Commercial','Mixed-Use'], p=[0.5,0.3,0.2])` (`pimaluos/system.py:686-693`). The `roi_lift` values are hard-coded constants per label (`system.py:697-704`).
- `scripts/generate_difference_maps.py:57-66` falls back to `np.random.choice([0,1,2], p=[0.2,0.5,0.3])`, because the plan CSV has no `far` column.
- `scripts/generate_full_manhattan_plan.py` generates a **synthetic lat/lon grid** and rule-based "optimisation", and writes it to the same `results/full_scale_simulation/` path.

Fig. 3b therefore compares a random map with a random map. The current code removed the fallback (`system.py:653-654`), and its comment admits that the true MARL output "may converge to a monoculture".

### C4. "Physics-informed" training contains no physics simulation

- `UrbanDigitalTwin.compute_physics_informed_loss` (`pimaluos/physics/digital_twin.py:418-444`) is `MSE(predictions, targets) + λ · mean(sigmoid(predictions)²)`. It never calls the traffic, hydrology or solar engines. It is an L2 shrinkage on predicted FAR. (It also applies `sigmoid` to an output that has already passed through a `Sigmoid`, `gnn.py:398`.)
- The targets are a **dummy all-ones vector**: `'far'` is not a column after renaming, so `torch.ones_like(...)` is used. The code comment says "Create dummy targets (in real scenario, use actual targets)" (`system.py:492-500`).
- Shapes `[N,1]` vs `[N]` broadcast to an **N×N** matrix in `F.mse_loss`. At N = 42,075 that is ≈1.8 × 10⁹ elements (≈7 GB fp32, before gradients). This contradicts "peak memory < 2 GB" (§5.3), and `run_full_manhattan.py:105` itself notes MPS OOM above 18 GB.
- There is a **single** `physics_weight`. The paper's λ_t = 0.3, λ_h = 0.2, λ_s = 0.2 do not exist.
- **Fig. 2e is a tautology.** Final loss = base + λ·penalty, so loss rises linearly in λ (0.199, 0.239, 0.320, 0.401, 0.481, 0.602). The figure cannot "confirm λ = 0.3 as optimal"; by the plotted criterion λ = 0 is best.
- "No Physics" in Table 6 only skips this L2 fine-tuning. The MARL reward and evaluation are unchanged.

The term "physics-informed" (title, abstract, PIMALUOS acronym) is therefore not supported even in the weakened "planning-scale capacity models" sense defended in §3.4. At full scale, the MARL environment also replaces the engines with an "O(N) approximation" based only on city-wide land-use proportions (`agents.py:846-925`). In that approximation:
- **no physics violation can ever fire.** Labels are `'mixed_use'` while the code looks up `'mixed'`. Congestion is ≈1.05 for uniform random land use, below the 1.5 threshold. Hydrology is ≤ 0.72 because FAR is capped at 2.0. Height is `far*12` ≤ 24 ft, below the 100 ft threshold.

### C5. The "total buildable floor area" metric assumes every lot is 1,000 sq ft

`compute_metrics` uses `gdf['lot_area']`, but the loader renames `LotArea → lot_area_sqft` (`data_loader.py:330`). The fallback therefore sets **every lot to 1,000 sq ft** (`run_baseline_comparisons.py:93`; the same bug is in `greedy_baseline.py:66`).

The headline 77,665,494 "sq ft" is therefore 1,000 × Σ FAR clipped to 2.0, i.e. mean FAR ≈ 1.85. For scale, Manhattan's real built floor area is on the order of 10⁹ sq ft. The number is off by more than an order of magnitude and is not a floor area.

### C6. All zoning constraints silently default to FAR 2.0 / 85 ft; "0 zoning violations" holds by construction

- `extract_constraints` looks for column `zoning_district`, but the loader produces `zone_district`. It therefore falls back to `zones = ['R6']` and assigns **`max_far = 2.0, max_height_ft = 85` to every parcel in Manhattan** (`system.py:222-252`). The baseline script hard-codes the same (`run_baseline_comparisons.py:177-181`).
- Every method then clips proposed FAR to `max_far` before counting violations as `proposed_far > max_far*1.01` (`run_baseline_comparisons.py:100,112`; `agents.py:815,757`). Zero violations is guaranteed for any method. It is not "RAG pre-filtering of the action space".
- Consequence: 433 of 500 sampled parcels are "over-developed" (`comparison_results.json`, rule-based details). Every method, Greedy included, **cuts existing Manhattan FAR by ~58 %** (`greedy.economic_improvement = −0.579`). Midtown's C5/C6 districts (FAR 10–15) are treated as FAR 2.

---

## 2. Major methodological problems (independent of C1–C6)

**M1. Action ≠ land use: the MARL acts on FAR, but the outputs are presented as land-use plans.** The agents choose {decrease, maintain, increase} FAR (`action_dim=3`). `generate_final_plan` then writes the FAR-action index into `proposed_use_code` and maps it through `LAND_USE_CATEGORIES`, so decrease→"Residential", maintain→"Commercial", increase→"Industrial" (`system.py:657-677`). The paper's land-use maps, "mixed-use", "entropy of land-use mix" and "Green Focus (C = 0.15)" have no corresponding decision variable.

**M2. The environment's land use is random noise that actions never change.** `reset()` draws `current_land_use = torch.randint(0, 6, …)` (`agents.py:678`), and `step()` sets `new_land_use = self.current_land_use` (`agents.py:733`). Most reward terms (neighbour ratios, entropy, park access, compatibility) depend on this random vector. The reward signal is therefore dominated by noise unrelated to the agent's action. The GNN never sees land use either: `graph['parcel']` has no `land_use_code` attribute, so the `hasattr` branches are dead.

**M3. "FAR" in the environment is actually the z-scored lot-coverage feature.** `far_idx = 10` (`agents.py:685`), but feature 10 in `compute_node_features` is `lot_coverage`. Index 8 is `current_far`; index 9 is `max_far`. The value is then clamped to [0.1, 2.0]. `step()` writes it back into `graph['parcel'].x[:,10]` (`agents.py:778`), **mutating the shared graph** in place. Subsequent trainers or ablations reusing the graph object (e.g. all three trainers in `run_marl_validation.py`) start from a corrupted feature matrix.

**M4. Utility functions in the code bear no relation to Eqs. (2)–(6).** The paper defines, e.g., U_dev = FAR_used/FAR_max − γ·Violations and U_equity = −Gini − DisplacementRisk. The code instead uses awareness-weighted mixtures of ~35 hand-made proxies (`agents.py:254-366, 1049-1087`). Examples:
- `citywide_gini_coefficient = 1 − 0.2·avg_far` enters the equity utility with a **positive** sign, so a higher "Gini" raises equity utility.
- Constants such as `amenity_access_equity = 0.65` and `opportunity_access = 0.65` are fixed.
- `displacement_risk = 0.3·far − 0.2·neighbour_res`.

No census, rent-burden or vulnerability data enter (contradicting Table 3). Fig. 1 shows yet another set of formulas.

**M5. Agent architecture, actions and hyperparameters differ from Table 4.**
- Actor/critic: 128-unit ReLU MLPs (`agents.py:65,81-97`), vs the paper's 64×64 Tanh.
- Actions: ×0.8 / ×1.2 multiplicative (`agents.py:808-810`), vs ±0.5 FAR.
- Observation: 128-dim (no 5-dim capacity state), vs the paper's 133.
- No linear LR decay. GAE λ = 0.95 and four PPO epochs are unreported.
- Paper training: 20 iterations × 5 steps ≈ 100 environment steps. That is far too little for a claim of "policy-stable consensus", and the baseline script uses 5 × 2 = 10 steps (`run_baseline_comparisons.py:66`).
- No convergence evidence is shown at full scale. The committed `reward_convergence.json` is from 500 parcels.

**M6. The voting mechanism is not what the paper describes.** It is a weighted argmax with fixed weights 0.25/0.15/0.25/0.20/0.15 (`agents.py:389-395, 438-455`). Ties resolve to the **lowest index (= decrease)**, not to "status quo", and there is no plurality check. The `'nash'` strategy is a `TODO` that falls back to weighted voting (`agents.py:475-483`).

**M7. The baselines are strawmen, so the comparison is uninformative.**
- *Greedy* is described as "hill-climbing for total buildable floor area". In code it increases the top 60 % and **decreases the bottom 10 %** in a single pass (`greedy_baseline.py:84-97`). A true greedy maximiser of Σ FAR × area would simply increase every parcel below the cap, giving the trivial upper bound.
- *Random* is not uniform: `at_max` parcels are forced to "maintain".
- *Diversity* is the Shannon entropy of the **three FAR actions**. It says nothing about land-use mix. Its maximum is ln 3 = 1.099, so "Random = maximum-entropy chaos" and "0.593 is the optimal middle ground" are circular and unjustified.
- Missing baselines: an exact or LP/ILP optimum for the stated objective, a single-agent PPO with the summed reward (only on 500 parcels in `marl_validation`, where it beats MARL economically), NSGA-II alone, and a status-quo plan.

**M8. GNN: the described architecture is not the implemented one, and the training signal is weak.**
- 4 heads (`gnn.py:372`), not 8. Two message-passing layers plus a Linear layer, not a "3-layer HGAT".
- There is one node type. "Heterogeneous" refers only to relation types, and Eq. (1)'s "node i of type φ" is misleading.
- Edge weights are passed as `edge_attr` to `GATConv` without `edge_dim`, so they are **silently ignored** (`gnn.py:58-65, 102-107`).
- Pre-training is **only feature reconstruction**. `land_use_label` is never created and `'far'` is not a column, so the "multi-task: land-use classification, development potential" claim is false (`system.py:369-390`).
- There is no train/validation split, and all reported losses are training losses. The "embedding variance 0.0078" ablation metric is constant by construction (L2-normalised 128-d vectors have variance 1/128).
- All 57 features feed the GNN (`system.py:280`); none are "dropped to 47". The 57 are 46 engineered features plus the land-use one-hot, and contain no BBL, address or coordinates. Appendix Table A.10 lists 38 features, several of which do not exist in MapPLUTO or in the code (sale price, sale year, lot frontage/depth, transit/park proximity).
- About 12 features are constant placeholders that z-score to 0 (`data_loader.py:162-171, 194-196, 183`): subway/bus/park distance, four centralities, tree canopy, median income, density, pct_rental and elevation.

**M9. Game-theory module (PoA, Nash) cannot run and is conceptually misapplied.**
- Environment rewards are per-parcel tensors, but `compute_payoff_matrix` stores them into scalar slots, and `find_equilibrium_iterative` compares `tensor > float` (`nash.py:79, 206`). Both raise errors.
- Every `env.reset()` re-randomises land use, so payoffs are noise.
- Best response is O(iters × agents × N × 3) full environment steps, infeasible at 42K.
- "Social welfare" is the mean equilibrium FAR (`nash.py:255-258`), not the sum of utilities.
- PoA is reported as Nash/optimum = 0.94. The standard definition (Koutsoupias & Papadimitriou) is optimum/worst equilibrium ≥ 1, and the paper's mixed use of "PoA", "efficiency ratio" and "6 % loss" needs a formal definition.
- The utopian welfare 5,154.5 comes from random sampling (`ParetoAnalyzer.compute_pareto_frontier`), not an optimum.

**M10. The Pareto layer is disconnected from the rest and mis-specified.**
- Decision variables are continuous FAR on [0, 2.0], with no link to MARL or land use.
- The economic objective divides by `current_far + 1e-6` (`pareto.py:173`), so vacant lots dominate it; that is why scores reach 700.
- The "housing" term never matches PLUTO's numeric `LandUse` codes (`pareto.py:234`), so it is constant 0.5.
- Eq. (9) is a scalarised penalty objective, inconsistent with a Pareto search.
- All returned solutions are infeasible, and `rank=0` is hard-coded.

**M11. The graph construction contradicts §3.1.2 and §5.1.**
- *Visual connectivity* is plain k-NN of centroids within 500 ft. There is **no ray casting** (`graph_builder.py:189-228`).
- *Functional similarity* links k-NN parcels with **complementary** uses via a hand-set synergy matrix, not "identical land-use codes" (`graph_builder.py:230-282`). If PLUTO `LandUse` is read as numeric (1…11), the lookup `'01'`… fails and every parcel becomes `'other'`.
- *Infrastructure* concatenates the alphabetic tokens of the address. "123 WEST 45 STREET" and "500 WEST 110 STREET" both become "WEST STREET", merging all numbered streets. It then links each parcel to the next four in **row order**, not spatial order (`graph_builder.py:295-324`). This is not "shared utility corridors or transit access".
- *Regulatory* links only 5-NN within large zones.
- The "unique undirected relations" in §5.1 are exactly half of each directed count. That is impossible for directed k-NN relations, so the numbers were computed by division, not counted.

**M12. The physics engines are not what §3.4 describes, even as simplified models.**
- *Traffic:* the road network is k-NN of parcel centroids with a constant capacity of 3,000 veh/h, not road width/lanes. The threshold is applied to the BPR travel-time ratio, not V/C. Demand is an O(N²) gravity model, infeasible at 42K, which is why it is bypassed.
- *Hydrology:* one city-wide Q against a constant 100 cfs (`engine.py:255-290`), with no sewersheds. Lot area is multiplied by 10.764 on geometry already in feet (`engine.py:412`).
- *Solar:* `shadow_pct = avg_height/2` and "violation = height > 100 ft" (`engine.py:353-366`). `compute_building_shadow` is never used, and it mixes metres with State-Plane feet.
- The claims "> 50 % of adjacent direct sunlight", "winter solstice shadow casting" and "95 % solar-rights compliance" are unsupported. The paper's own Pareto range reports 91.2 % solar compliance, below the stated 95 % target, alongside "0 solar violations".

**M13. Data-engineering errors that change results.**
- `max_far = ResidFAR + CommFAR + FacilFAR` (`data_loader.py:370-374, 395-399`). These are **alternative** maxima, so the value should be `max(...)` or use-specific.
- `max_height_ft = num_floors × 12` is the *existing* height, labelled a regulatory constraint (`data_loader.py:219`).
- `SplitZone` is mapped to `special_district` (`data_loader.py:348`); since SplitZone is "Y/N", `notna()` makes it 1 for nearly all lots.
- `dist_to_center` subtracts lat/lon degrees from EPSG:2263 feet (`data_loader.py:156-159`).
- `YearAlter1 = 0` means "never altered", so `years_since_renovation = 2024`.
- The MapPLUTO version (23v3) and the zoning shapefile date (`nycgiszoningfeatures_202511shp`, Nov 2025) are mismatched.
- `requests.get` has no status check, and the NYC Open Data export of `64uk-42ks` (PLUTO tabular) may not return MapPLUTO geometry. Verify this.
- `zoning_compliance.get_allowed_uses` slices `zone[:2]`, so "R10" becomes "R1", which disallows commercial. It references `ZONING_ALLOWED_USES['SPECIAL']`, which does not exist (KeyError), and lists industrial as allowed in C4.

---

## 3. Paper ↔ code consistency table

| # | Paper says | Code / artifacts say |
|---|---|---|
| 1 | d_in = 47 after dropping 10 identifiers | 57 features, nothing dropped, no identifiers among them |
| 2 | 3-layer HGAT, 8 heads | 2 GAT layers + Linear, 4 heads |
| 3 | Multi-task pre-training | Reconstruction only |
| 4 | 500 / 150 epochs, ~90 / ~50 min (42K) | 42K run: 30 / 20 epochs; 500/150 was run on 100 parcels in 163 s |
| 5 | λ_t, λ_h, λ_s = 0.3 / 0.2 / 0.2 | Single λ; no physics in loss |
| 6 | Action ±0.5 FAR | ×0.8 / ×1.2 |
| 7 | PPO 64×64 Tanh, linear LR decay, obs 133 | 128×128 ReLU, constant LR, obs 128 |
| 8 | Ties → status quo | Ties → lowest index (decrease) |
| 9 | RAG: OpenAI embeddings + FAISS; Llama-2 (§3.2) / Llama-3 (§5.4) | NumPy cosine search; FAISS/Chroma unused; Ollama default `llama2`; experiments use mock |
| 10 | Two-stage overlay query | Not implemented |
| 11 | Visual edges via ray casting | k-NN |
| 12 | Functional = identical codes | Complementary synergy pairs |
| 13 | Infrastructure = utility/transit corridors | Street-name tokens, row-order neighbours |
| 14 | Spatial-only ablation = 1 type | 2 types |
| 15 | NSGA-III, 100 gen, 127 solutions, PPO-seeded | NSGA-II, 10 gen, 30 infeasible solutions, unseeded |
| 16 | Greedy = hill-climbing | One-shot top-60 %↑ / bottom-10 %↓ |
| 17 | Random = uniform | Uniform then masked |
| 18 | Physics: solstice shadow casting, sewer-shed capacity, road capacity from lanes | Avg-height heuristic, single city-wide Q, constant 3,000 veh/h |
| 19 | Peak memory < 2 GB | N×N broadcast in physics loss; code comments report >18 GB on MPS |
| 20 | Stage 5 MARL = 52 s; Table 6 opt time 132 s | Different code paths (20×5 vs 5×2 steps); 132 s includes graph/ckpt load |
| 21 | 24 tests, 92.6 % coverage, CI configured | 52 test functions (mostly import/smoke); no coverage report; **CI has failed on every recent push** (runs 14–18) |
| 22 | `pip install pimaluos` | Not on PyPI (404) |
| 23 | `pip install -r requirements.txt` then `./reproduce_all.sh` | `requirements.txt` has only streamlit/pandas/numpy/plotly/requests: no torch, PyG or geopandas |
| 24 | `reproduce_all.sh` downloads Zenodo data and reproduces the 42K experiment | No download step; runs 500 parcels × 1 seed; reproduces neither Table 6 nor 7, nor Figs 2b/2e/2f/3 |
| 25 | `download_manhattan_data.py` (Table 8) | File does not exist |
| 26 | Supplementary at `/tree/main/docs` | `docs/` was deleted (`965966e`) |
| 27 | Zenodo DOI `10.5281/zenodo.108927` (Table 8) **and** `10.5281/zenodo.11478203` (Software Availability) | Two different DOIs; the first is a 2014-era ID format. `.zenodo.json` has placeholder creators ("PIMALUOS Team"), date 2024, and an `XXXXX` related DOI |
| 28 | Software v0.1.0; "all figures synchronised" | Fig. 3a screenshot shows "v2.4.0-stable", "+18.1 %", "No constraints violated in last 94 iterations", which appear nowhere in the code |
| 29 | Two independent annotators + adjudicator | Sole-author paper; no annotator acknowledged; benchmark generated by script |
| 30 | Figure replication via `scripts/generate_all_figures.py` | Training-curve and Pareto figures read `results/small_scale_demo/checkpoint.pth`, which is not in the repo (its demo script was deleted in `816b3fd`) |

---

## 4. Figures and tables: specific problems

- **Fig. 2a** is built from the 500-parcel JSON. It shows PIMALUOS with the **lowest** value (≈897,792, first run) and Greedy the highest. That directly contradicts Table 6 and the text. The y-axis says "Total Economic Value (USD)" while the caption says "floor area".
- **Fig. 2b** shows seven configurations; "Functional Only" (0.334) and "Regulatory Only" (0.335) beat "All Edges" (0.341), and "All Edges No Physics" is best (0.223). This undermines §5.2.2's argument that the full heterogeneous graph is "essential". Table 7 shows only two rows.
- **Fig. 2c/2d** curves are 100-parcel runs (C1). The physics curve's smooth exponential plateau reflects a sigmoid head regressed to 1 plus an L2 term, not "physics constraints efficiently encoded".
- **Fig. 2e** is a tautology (C4).
- **Fig. 2f** shows about 30 points, not 127. The "knee" star sits at the extreme-economic corner, and a second star in the legend overlaps the plot. A 2-D projection of a 4-objective front also shows dominated-looking points without explanation.
- **Fig. 3a** is a dashboard screenshot with version and metrics that do not match the paper or code. "Benchmark Analysis" bars (Random 30.4, Rule 97.5, Greedy 98.1, PIMALUOS 100) have no source.
- **Fig. 3b** compares a random "current" map with a random-quota "optimised" map (C3). Both panels look nearly identical purple at publication size, and real PLUTO land use is available but unused.
- **Fig. 1** utilities (e.g. "Developer U = ROI:0.6 + Spd:0.4") differ from Eqs. 2–6 and from the code. It lists Next.js/deck.gl and an "Eval Agent" that the paper does not evaluate.
- **Table 2:** precision/recall for numeric value extraction is not well defined without stating what a false positive or false negative is, and per-parameter support counts are missing. The "OCR parsing errors on historical tables" error source contradicts the text-based pipeline.
- **Table 6:** "± 0" for deterministic baselines is fine, but the table mixes run-level std with parcel-level hypothesis tests. Time units mix optimisation time and checkpoint loading.
- **Table 9** misdescribes prior work. Zheng et al. (2023) [9] is a GNN + deep-RL planner, not a "GAN"; "DeepUrbanPlanner" is not its name. UrbanSim and CityScope are not "single-domain physics simulators". The closest prior work, Qian et al. (2023) consensus-based MARL with GNN for participatory land-use planning, is omitted.

---

## 5. Statistics and evaluation design

1. **The unit of analysis is wrong for significance.** Parcels are not independent: they share a policy and spatial dependence. A Wilcoxon over N = 42,075 parcels is pseudo-replication. The appropriate unit is the independent training run (seed), with ≥ 5–10 seeds, reporting mean ± 95 % CI and a paired test across seeds.
2. **Cohen's d** must be computed consistently with the test. The current script and the paper disagree by two orders of magnitude.
3. **The objective is trivial under the current metric.** With FAR clipped at a uniform cap and no other constraints, the optimum is "increase everywhere". Report that bound. A 0.2 % "win" over a deliberately sub-optimal greedy is not evidence.
4. **The diversity metric** should be land-use mix (e.g. entropy of floor area by use within a walkable buffer) on a real land-use decision, not entropy of FAR action labels.
5. **There is no external validity check.** Compare against actual rezonings, e.g. the Midtown East or Inwood rezonings, or DCP Housing Opportunity analyses, or have planners rate plans.
6. **Sensitivity analyses are missing:** k-NN k, buffer distance, reward weights, physics thresholds, and number of MARL steps.
7. **Training dynamics:** report reward/return curves per agent at the scale used, with seeds, and show that policies do not collapse. The committed evidence shows they do.

---

## 6. Code-quality and software-engineering issues (for a CEUS software paper)

- **CI is red on every push.** The matrix includes Python 3.9 while `requires-python >= 3.10`. The `docs` job runs `cd docs`, but `docs/` is deleted. The `dashboard` job needs `package-lock.json`, which is absent. Ruff and mypy are not clean.
- `setup.py` / `pyproject.toml` declare the console script `pimaluos.cli:main`, but **`pimaluos/cli.py` does not exist**. Author metadata is "PIMALUOS Team / pimaluos@example.com".
- The dependency story is split three ways (`requirements.txt`, `requirements-full.txt`, `setup.py`). "Locked for reproducibility" is stated, but only ranges are given; there is no lockfile and no Docker image.
- There is no data-acquisition script, no checksums, no MapPLUTO version pinning, and caches are `.gitignore`d. Graph, checkpoints and constraints cannot be obtained.
- Tests are mostly import or "not None" assertions. There are none for `MultiAgentEnvironment.step`, the training losses, the voting tie-break, `NashEquilibriumSolver` (which would have revealed M9), the metric functions (which would have caught C5) or `extract_constraints` (which would have caught C6).
- Dead or placeholder components are advertised as features: `AgentCommunicationChannel.negotiate` (averaging), `WebSocketStreamer` (prints), `TimeSteppingSimulator` (computes `peak_factor` but never applies it), the `'nash'` voting strategy (TODO), and the Chicago/LA/Boston loaders (return empty GeoDataFrames). The conclusion's transferability claims should not rest on these.
- `generate_final_plan` has a duplicated `return plan` (`system.py:717-719`). Debug logging ("DEBUG: Print action distribution") ships in the release.
- Global `warnings.filterwarnings('ignore')` in `data_loader.py:29` and `engine.py:25` hides real problems, such as the N×N broadcast warning.
- Hard-coded absolute paths in artifacts: `stage_6_plan.done` contains `/Users/ppayami/Desktop/...`.
- The repository is ~383 MB, mostly `.git` (PDFs and figures committed repeatedly). Large binaries belong in Zenodo or Git LFS.

---

## 7. Reference audit

I checked each reference against publisher and indexer records.

| Ref | Status | Problem / correction |
|---|---|---|
| [1] Batty 2013 | OK | — |
| [2] Boeing 2017 | OK | — |
| [3] Waddell 2002 | OK | Not cited in §2.1 text, where UrbanSim is discussed |
| [4] Alonso et al. 2018 | **Wrong details** | Title is "CityScope: A Data-Driven Interactive Simulation Tool for Urban Design. Use Case Volpe", *Unifying Themes in Complex Systems IX*, pp. 253–261, DOI 10.1007/978-3-319-96661-8_27 (not _30, 273–284) |
| [5] Ito et al. 2025 ZenSVI | OK | — |
| [6] Mahajan 2024 greenR | Minor | Journal/volume text duplicated |
| [7] Sevtsuk & Alhassan 2025 | OK | — |
| [8] Jin et al. TKDE | **Wrong details** | Vol. 36(10), pp. 5388–5408 (2024), not 35(10), 10567–10584 (2023) |
| [9] Zheng et al. 2023 | Authors misordered; **mischaracterised** | Authors: Zheng, Lin, Zhao, Wu, Jin, **Li**. A DRL/GNN planner, not a GAN (Table 9) |
| [10] "Wang, S. et al. 2025" arXiv 2504.02009 | **Wrong authors** | Li, Z., Xia, L., Ren, X., Tang, J., Chen, T., Xu, Y., Huang, C. |
| [11] Zhang et al. 2024 "LLM-Zoning", *Comput. Urban Sci.* 4(1) | **Not found; likely non-existent** | Replace with real work, e.g. Bartik, Gupta & Milo, "Generative regulatory measurement" (NBER/AI-zoning project); National Zoning Atlas / zoning-gpt |
| [12] Dai et al. 2024 *Cities* 151 | Exists | Title truncated: add "…: Experiences from nature-based solutions in China"; verify article number |
| [13] Lewis et al. 2020 | OK | — |
| [14] Zhang, Yang & Başar 2021 | OK | — |
| [15] Chu et al. 2020 | OK | — |
| [16] Li, M. et al. 2024 "Consensus-based MARL for participatory urban planning", *SCS* 104 | **Not found; likely non-existent** | The real work is Qian, K. et al. (2023) "AI Agent as Urban Planner: Steering Stakeholder Dynamics in Urban Planning via Consensus-based Multi-Agent Reinforcement Learning", arXiv:2310.16772. It is the closest prior work and must be discussed and compared |
| [17] AlBalkhy et al. 2024 | OK | — |
| [18] Karniadakis et al. 2021 | OK | — |
| [19] Jiang & Luo 2022 | **Wrong venue/DOI** | *Expert Systems with Applications* 207, 117921 (not IEEE T-ITS) |
| [20] Wang et al. "Deep learning for spatio-temporal data mining" | **Wrong details** | Wang, Cao & Yu, TKDE 34(8), 3681–3700 (2022), DOI 10.1109/TKDE.2020.3025580 |
| [21] Liu et al. 2023 "Urban function classification using spatial graph convolutional networks", *CEUS* 101, 101924 | **Not found; likely non-existent** | Article 101924 is a different paper (Liu et al., street-tree inventory, CEUS 100) |
| [22] Yao et al. 2017 | **Wrong title/issue/pages/DOI** | "…integrating points-of-interest and Google **Word2Vec model**", IJGIS 31(4), 825–848, DOI 10.1080/13658816.2016.1244608 |
| [23] Guo et al. 2019 "Attention-based spatial planning with deep RL", *ASOC* 85, 105820 | **Not found; likely non-existent** | Remove or replace |
| [24] Abadal et al. 2021 | Exists, **miscited** | It is a GNN-accelerator survey, cited for "HGATs". Cite Wang et al. 2019 (HAN, WWW) and Hu et al. 2020 (HGT) instead |

Four references appear not to exist and seven more have substantive errors. CEUS editors routinely check references, and fabricated citations alone are grounds for desk rejection. Re-build the bibliography from DOIs (Crossref/Zotero) and verify every entry.

**Missing literature a CEUS referee will expect:**
- Multi-objective land-use allocation: Stewart, Janssen & van Herwijnen (2004); Cao et al. (2011, 2012, NSGA-II land use, IJGIS/LUP); Ligmann-Zielinska et al.; Aerts et al.; Huang et al.; Memmah et al. (2015) review.
- Land-use change models: CLUE-S, FLUS, cellular automata.
- Automated urban planning with deep generative models: Wang, D. et al. (LUCGAN, KDD/TKDD 2020–2023).
- Qian et al. (2023), the closest prior work.
- Methods: PPO (Schulman 2017), MAPPO (Yu et al. 2022), NSGA-III (Deb & Jain 2014), GAT (Veličković 2018), HAN/HGT, PyG (Fey & Lenssen 2019), PINNs (Raissi et al. 2019).
- Engineering models: BPR (Bureau of Public Roads 1964), Rational Method / NYC DEP design storms, ITE Trip Generation.
- Price of Anarchy (Koutsoupias & Papadimitriou 1999).
- Data: NYC DCP MapPLUTO (with version), the NYC Zoning Resolution.
- Urban digital twins (Batty 2018, *EPB*).
- Algorithmic-planning equity: Rawlsian and displacement literature, e.g. Chapple & Zuk.

---

## 8. Writing, framing and ethics

- **"Physics-informed"** should be dropped from the title, acronym and abstract unless engineering models are actually coupled into training. Even then, "capacity-constrained" or "engineering-model-constrained" is more accurate. The paper's own disclaimer (§3.4) already concedes the PINN sense does not apply.
- **The abstract over-claims.** "Prevents the zero-entropy collapse of standard optimisation" is unsupported: Greedy has 0.530, and only the authors' own ablation has 0. "Satisfy physical environmental, transportation and solar capacities" is unsupported (C4, M12). "Empirically validate cooperative mechanisms" is unsupported (C1, M9).
- **Contributions list:** "novel heterogeneous graph formulation" overlaps prior multi-relational urban graphs (e.g. HUGAT, and Qian et al.); "extensive validation" is not supported.
- **Internal inconsistencies:**
  - Llama-2 vs Llama-3.
  - FAISS vs ChromaDB.
  - "four layers" vs "Sense-Reason-Verify" (three).
  - Weights "initialised w_i = 1/3" for five agents.
  - Solar target 95 % vs reported 91.2 %.
  - The negotiation runtime: 52 s, 132 s and "~30–60 min" all appear in the code comments or the paper.
- **The ethics section** relies on "the Equity Advocate agent penalises plans targeting low-income tracts". The code has no tract, income or rent data (M4), so this mitigation claim is false and should be removed or implemented. Add a concrete limitations section on the use of assessed value as a value proxy, on displacement, and on the absence of community input.
- **CRediT and acknowledgements:** the CRediT statement lists "Funding acquisition", but no funding statement is given. Add one, or "This research received no specific grant…". If annotators existed, name or acknowledge them and describe consent and compensation.
- **Generative-AI declaration:** the repo shows substantial AI-assisted generation of code, manuscript text, the benchmark generator and response-to-review commits. Elsevier policy requires the declaration to state accurately *what* the tools were used for. "Coding and formatting improvements" understates this. Make sure every number in the paper was produced by code you ran and checked.

---

## 9. What is genuinely good and worth keeping

- The idea of a modular, open-source pipeline that wires PLUTO parcels, a multi-relational parcel graph, zoning-constraint extraction, stakeholder agents and capacity checks is attractive for CEUS's software / Open Urban Data Science track.
- Real MapPLUTO ingestion at 42K parcels with STRtree adjacency is a useful, reusable component.
- Batched PPO over all parcels with shared per-stakeholder policies is a sensible scalable design.
- The authors already acknowledge several limitations (simplified engines, need for human validation of RAG). Extending that candour to the whole paper is the way forward.

---

## 10. Prioritised fix list (code)

| Priority | Fix | Location |
|---|---|---|
| P0 | Use `zone_district` in `extract_constraints`, and look up real per-district max FAR (residential/commercial/facility by use; `max`, not sum) | `system.py:222-252`, `data_loader.py:370-399` |
| P0 | Use `lot_area_sqft` in metrics and greedy | `run_baseline_comparisons.py:93`, `greedy_baseline.py:66` |
| P0 | Read FAR from the `current_far` column by name, not index 10; do not mutate the shared graph | `agents.py:685,778` |
| P0 | Make land use an actual decision (or drop land-use claims); stop random re-initialisation | `agents.py:678,733`, `system.py:657-704` |
| P0 | Delete the random "current use" and random-quota plan artifacts; regenerate maps from real PLUTO `LandUse` and real outputs | `system.py:686-704`, `results/full_*`, `generate_difference_maps.py`, `generate_full_manhattan_plan.py` |
| P0 | Either couple engines into the loss/reward (differentiable surrogate or REINFORCE-style penalty) or rename the method; fix the `[N,1]` vs `[N]` broadcast | `digital_twin.py:418-444`, `system.py:490-507` |
| P1 | Implement Eqs. 2–6 as written, with real data (ACS 5-yr tract income/rent burden, NYC Parks, MTA stations), or change the equations to match the code | `agents.py:254-366, 937-1087` |
| P1 | Fix the infrastructure-edge street parsing (keep ordinal numbers; order by position along street), implement or rename "visual" edges, and pass `edge_dim` to `GATConv` | `graph_builder.py`, `gnn.py` |
| P1 | Real greedy (and an exact optimum bound); uniform random; status-quo baseline | `pimaluos/baselines/*` |
| P1 | Fix the Nash module to operate on aggregated scalar utilities for a small game, or remove PoA claims | `nash.py` |
| P1 | Fix the Pareto economic objective (`/(current_far+1e-6)`), the housing mask, and NSGA-III use; connect it to the MARL output if claimed | `pareto.py` |
| P1 | Fix the zoning-compliance prefix parsing and the `'SPECIAL'` KeyError | `zoning_compliance.py:71-82` |
| P2 | Remove constant placeholder features, or compute them (subway distance from MTA GTFS, park distance from NYC Parks) | `data_loader.py:161-196` |
| P2 | Green CI; add `cli.py` or remove the entry point; single pinned lockfile plus Dockerfile; data-download script with checksums | packaging |
| P2 | Real tests for env step, metrics, constraints, voting and losses | `tests/` |

---

## 11. Path to a publishable paper

**Option A: honest software paper (fastest, realistic for CEUS).**
1. Fix the P0 bugs. Run the full 42K pipeline end-to-end with ≥ 5 seeds via a single script that writes every table and figure from saved outputs, with no numbers typed into the `.tex`. Generate LaTeX tables from CSV, e.g. `pandas.to_latex`.
2. Report what the system actually does, including negative results (collapse, ablation outcomes) and the honest scope of each module.
3. Replace "physics-informed" with an accurate term. Present engines as capacity checks used for screening, and show their outputs on real scenarios.
4. Run a **real** RAG evaluation:
   - sample real ZR sections (with section IDs and version date) for the districts present in Manhattan;
   - have ≥ 2 annotators extract values blind;
   - compute κ with a script;
   - run GPT/Llama extraction with logged prompts and outputs;
   - report exact-match accuracy with CIs;
   - also compare extracted max FAR with PLUTO's ResidFAR/CommFAR/FacilFAR, a free large-scale check.
5. Archive the exact release on Zenodo (one DOI) and quote it consistently. Publish to PyPI or drop the `pip install pimaluos` claim. Restore `docs/`.
6. Rebuild the bibliography from DOIs. Add Qian et al. 2023 and the land-use-allocation literature, and position against them in Table 9.

**Option B: methods/research paper (longer).** Do everything in Option A, then:
- Make land use plus FAR the action.
- Use real PLUTO land use as the initial state.
- Implement the stated utilities with census and amenity data.
- Use proper MARL training budgets with learning curves.
- Run a well-posed Nash/PoA analysis on a small sub-game.
- Seed NSGA-III from MARL and compare against cold start (the hypothesis the paper already states).
- Evaluate against real DCP rezonings or expert panels, with run-level statistics.

---

## 12. Checklist of manuscript statements to remove or substantiate

Remove or replace each of the following with code-backed numbers:
- [ ] Abstract and §5.2.1 numbers: 77,665,494 / 77,341,970 sq ft, entropy 0.593 / 0.000 / 0.530 / 0.810 / 0.498
- [ ] Table 6 (entire), "3 independent full-scale runs", seeds 42/202/404, opt times
- [ ] Wilcoxon z = −18.42; Cohen's d = 0.082
- [ ] "14 ± 2 traffic exceedances / 8 ± 1 sewer overflows"
- [ ] PoA 0.940, welfare 4,845.2 / 5,154.5
- [ ] 127 Pareto solutions, NSGA-III 100 generations, PPO seeding, all Pareto range numbers, knee 73 % / 84 % / 96.4 %
- [ ] Weight sweep (1.8 % / 5.0 %)
- [ ] Equity ablation (0.245 → 0.382; +38.4 %)
- [ ] Table 7 (42K edges, 90 / 82 min); "Spatial only = 1 type"
- [ ] Figs 2c/2d "on all 42,075 parcels", 500/150 epochs, 69.5 % / 56.6 %
- [ ] Fig 2e "λ = 0.3 optimal"
- [ ] Table 2, κ = 0.92, annotators, 96.8 % → 93.5 %, two-stage overlay strategy
- [ ] Fig 3a (version/metrics) and Fig 3b (random maps)
- [ ] λ_t / λ_h / λ_s, 8 heads, d_in = 47, 64×64 Tanh, ±0.5 FAR, obs 133
- [ ] "ray casting", "identical land-use codes", "utility/transit corridors"
- [ ] Solar "> 50 % adjacent sunlight / 95 % compliance"; hydrology "local sewer-shed capacity"; traffic "capacity from width/lanes"
- [ ] "Peak memory < 2 GB", "52 s"
- [ ] "24 tests, 92.6 % coverage", CI claims, commit 167a341, Zenodo DOIs, `download_manhattan_data.py`, `pip install pimaluos`, docs URL
- [ ] Equity-agent mitigation claim in §6.2
- [ ] References [4], [8], [9], [10], [11], [16], [19], [20], [21], [22], [23], [24]

---

*Prepared from a full read of `pimaluos/`, `experiments/`, `scripts/`, `tests/`, CI config, packaging, all committed results, git history, the manuscript PDF/TeX, and external checks of PyPI, GitHub Actions and the literature.*

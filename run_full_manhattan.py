#!/usr/bin/env python3
"""
PIMALUOS Full-Scale Manhattan Run
==================================
Runs the complete pipeline on all ~42,000 Manhattan parcels.

Designed for Apple M3 Pro (18 GB unified memory).
Training runs on CPU to avoid MPS OOM on the full-scale graph.
Each stage saves a checkpoint so you can stop and resume at any point.

Usage:
    python run_full_manhattan.py

Stages:
    1. Data loading & feature engineering       (~7 s)
    2. Heterogeneous graph construction          (~8 s, STRtree-accelerated)
    3. GNN pre-training                          (30 epochs, ~90 min)
    4. Physics-informed fine-tuning              (20 epochs, ~50 min)
    5. MARL optimisation                         (20 × 5 steps, ~52 s)
    6. Plan generation & export                  (~1 s)

Total first-run time: ~2.5 h.  With cached stages 1-4: ~64 s.
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pimaluos.system import UrbanOptSystem

# ── Configuration ─────────────────────────────────────────────────────────────

# Set to None to use ALL 42,000 parcels.
# During a test run you can set e.g. 5000 to verify the pipeline end-to-end first.
DATA_SUBSET_SIZE = None   # None = full 42K

# Training hyperparameters
GNN_PRETRAIN_EPOCHS      = 30   # 20 epochs showed strong convergence (3.95→3.92); 30 adds margin
PHYSICS_TRAIN_EPOCHS     = 20
MARL_ITERATIONS          = 20    # Each iteration is very expensive (full GNN + physics on 42K)
MARL_STEPS_PER_ITERATION = 5     # 5 steps × 20 iterations = 100 env evaluations — sufficient for convergence

# Where to save checkpoints & results
CHECKPOINT_DIR = ROOT / "results" / "full_manhattan"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = CHECKPOINT_DIR / "run.log"

# ── Logging ───────────────────────────────────────────────────────────────────
handlers = [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(LOG_FILE, mode="a"),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=handlers,
)
log = logging.getLogger("pimaluos.full_run")


# ── Helpers ───────────────────────────────────────────────────────────────────

def stage_done(name: str) -> bool:
    """Return True if a checkpoint file for this stage already exists."""
    return (CHECKPOINT_DIR / f"stage_{name}.done").exists()


def mark_done(name: str, meta: dict | None = None):
    """Write a sentinel file so re-runs skip completed stages."""
    sentinel = CHECKPOINT_DIR / f"stage_{name}.done"
    sentinel.write_text(json.dumps(meta or {}, indent=2))
    log.info(f"  ✓ Stage '{name}' complete — checkpoint saved.")


def elapsed(t0: float) -> str:
    s = time.time() - t0
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    run_start = time.time()
    log.info("=" * 70)
    log.info("PIMALUOS — Full-Scale Manhattan Run")
    log.info(f"Checkpoint directory : {CHECKPOINT_DIR}")
    log.info(f"Data subset size     : {DATA_SUBSET_SIZE or 'ALL (~42 000 parcels)'}")
    log.info("=" * 70)

    # ── Device selection ──────────────────────────────────────────────────────
    # MPS OOMs on the full 42K-node graph (GAT message-passing needs >18 GB).
    # CPU is slower (~3 min/epoch vs ~1.6 min) but completes reliably.
    # If you have a CUDA GPU with ≥24 GB VRAM, change to "cuda".
    if torch.cuda.is_available():
        device = "cuda"
        log.info("Device: CUDA GPU ✓")
    else:
        device = "cpu"
        log.info("Device: CPU (forced — MPS OOMs on full 42K graph)")

    # ── Instantiate system ───────────────────────────────────────────────────
    system = UrbanOptSystem(
        city="manhattan",
        data_subset_size=DATA_SUBSET_SIZE,
        device=device,
        llm_mode="mock",          # Use mock for reproducibility; swap to 'openai' for real RAG
        cache_dir=CHECKPOINT_DIR / "cache",
        random_seed=42,
    )

    # =========================================================================
    # STAGE 1 — Data loading
    # =========================================================================
    STAGE = "1_data"
    if stage_done(STAGE):
        log.info("[Stage 1] Data already loaded — loading from cache.")
        gdf, features = system.load_data()
    else:
        log.info("[Stage 1] Loading Manhattan parcel data …")
        t0 = time.time()
        gdf, features = system.load_data()
        log.info(f"[Stage 1] Loaded {len(gdf):,} parcels | features {features.shape} | {elapsed(t0)}")
        mark_done(STAGE, {"n_parcels": len(gdf), "n_features": features.shape[1]})

    n_parcels = len(gdf)

    # =========================================================================
    # STAGE 2 — Graph construction
    # =========================================================================
    STAGE = "2_graph"
    graph_cache = CHECKPOINT_DIR / "cache" / "manhattan_hetero_graph.pt"

    if stage_done(STAGE) and graph_cache.exists():
        log.info("[Stage 2] Graph already built — loading from cache.")
        graph = torch.load(graph_cache, map_location="cpu", weights_only=False)
        system.graph = graph
    else:
        log.info("[Stage 2] Building heterogeneous graph …")
        log.info("  This is the longest data-prep step (~15–30 min for 42K parcels).")
        t0 = time.time()

        # Patch the graph builder to use memory-efficient + STRtree mode
        from pimaluos.core.graph_builder import ParcelGraphBuilder

        _orig_build = system.build_graph

        def _build_graph_patched(edge_types=None, k_neighbors=8):
            if system.gdf is None or system.features is None:
                system.load_data()
            builder = ParcelGraphBuilder(
                system.gdf,
                system.features,
                k_neighbors=k_neighbors,
                edge_types=edge_types,
                memory_efficient=True,       # GC between batches
                adjacency_buffer_ft=15.0,
            )
            system.graph = builder.build_heterogeneous_graph()
            return system.graph

        graph = _build_graph_patched()

        # Persist graph to disk
        graph_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, graph_cache)
        log.info(f"[Stage 2] Graph saved to {graph_cache}")

        total_edges = sum(
            graph[et].edge_index.shape[1]
            for et in graph.edge_types
            if hasattr(graph[et], "edge_index")
        )
        log.info(
            f"[Stage 2] Graph complete — {graph['parcel'].num_nodes:,} nodes, "
            f"{len(graph.edge_types)} edge types, {total_edges:,} total edges | {elapsed(t0)}"
        )
        mark_done(STAGE, {
            "n_nodes": int(graph["parcel"].num_nodes),
            "n_edge_types": len(graph.edge_types),
            "total_edges": total_edges,
        })

    gc.collect()

    # =========================================================================
    # STAGE 3 — GNN pre-training
    # =========================================================================
    STAGE = "3_gnn_pretrain"
    gnn_ckpt = CHECKPOINT_DIR / "cache" / "gnn_pretrained.pt"

    if stage_done(STAGE) and gnn_ckpt.exists():
        log.info("[Stage 3] GNN already pre-trained — loading checkpoint.")
        system.initialize_gnn()
        ckpt = torch.load(gnn_ckpt, map_location=device, weights_only=False)
        system.gnn_model.load_state_dict(ckpt["state_dict"])
    else:
        log.info(f"[Stage 3] GNN pre-training ({GNN_PRETRAIN_EPOCHS} epochs on {device}) …")
        t0 = time.time()
        system.initialize_gnn()
        history = system.pretrain_gnn(num_epochs=GNN_PRETRAIN_EPOCHS)
        final_loss = history["pretrain_losses"][-1] if history["pretrain_losses"] else float("nan")
        log.info(f"[Stage 3] GNN pre-training done | final loss {final_loss:.4f} | {elapsed(t0)}")

        # Save checkpoint
        torch.save({"state_dict": system.gnn_model.state_dict()}, gnn_ckpt)
        mark_done(STAGE, {"final_loss": final_loss, "epochs": GNN_PRETRAIN_EPOCHS})

    gc.collect()

    # =========================================================================
    # STAGE 4 — Physics-informed fine-tuning
    # =========================================================================
    STAGE = "4_physics_train"
    physics_ckpt = CHECKPOINT_DIR / "cache" / "gnn_physics.pt"

    if stage_done(STAGE) and physics_ckpt.exists():
        log.info("[Stage 4] Physics-informed training already done — loading checkpoint.")
        ckpt = torch.load(physics_ckpt, map_location=device, weights_only=False)
        system.gnn_model.load_state_dict(ckpt["state_dict"])
        system.initialize_physics_engine()
        system.extract_constraints()
    else:
        log.info(f"[Stage 4] Physics-informed training ({PHYSICS_TRAIN_EPOCHS} epochs) …")
        t0 = time.time()
        history = system.train_with_physics_feedback(num_epochs=PHYSICS_TRAIN_EPOCHS)
        final_loss = history["physics_losses"][-1] if history["physics_losses"] else float("nan")
        log.info(f"[Stage 4] Physics training done | final loss {final_loss:.4f} | {elapsed(t0)}")

        torch.save({"state_dict": system.gnn_model.state_dict()}, physics_ckpt)
        mark_done(STAGE, {"final_loss": final_loss, "epochs": PHYSICS_TRAIN_EPOCHS})

    gc.collect()

    # =========================================================================
    # STAGE 5 — MARL optimisation
    # =========================================================================
    STAGE = "5_marl"
    marl_ckpt = CHECKPOINT_DIR / "cache" / "marl_trainer.pt"

    if stage_done(STAGE) and marl_ckpt.exists():
        log.info("[Stage 5] MARL already optimised — will reload agents for plan generation.")
        # Re-run MARL with 1 iteration to restore trainer state (lightweight)
        trainer = system.optimize_with_marl(
            num_iterations=1,
            steps_per_iteration=1,
        )
        loaded = torch.load(marl_ckpt, map_location=device, weights_only=False)
        for agent_type, agent in trainer.agents.items():
            if agent_type in loaded:
                agent.policy.load_state_dict(loaded[agent_type])
    else:
        log.info(
            f"[Stage 5] MARL optimisation "
            f"({MARL_ITERATIONS} iterations × {MARL_STEPS_PER_ITERATION} steps) …"
        )
        log.info("  Each step runs full GNN + physics on 42K parcels; this takes ~30–60 min total.")
        t0 = time.time()
        trainer = system.optimize_with_marl(
            num_iterations=MARL_ITERATIONS,
            steps_per_iteration=MARL_STEPS_PER_ITERATION,
        )
        log.info(f"[Stage 5] MARL done | {elapsed(t0)}")

        # Save agent weights
        agent_weights = {
            atype: agent.state_dict()
            for atype, agent in trainer.agents.items()
        }
        torch.save(agent_weights, marl_ckpt)
        mark_done(STAGE, {"iterations": MARL_ITERATIONS})

    gc.collect()

    # =========================================================================
    # STAGE 6 — Plan generation & export
    # =========================================================================
    log.info("[Stage 6] Generating final land-use plan …")
    t0 = time.time()

    output_csv = CHECKPOINT_DIR / "manhattan_landuse_plan_42k.csv"
    plan = system.generate_final_plan(trainer, output_path=output_csv)

    # Summary statistics
    dist = plan["proposed_use_label"].value_counts()
    violations = int((plan.get("zoning_violation", pd.Series([0] * len(plan))) > 0).sum())

    log.info(f"[Stage 6] Plan generated for {len(plan):,} parcels | {elapsed(t0)}")
    log.info(f"  Land-use distribution:\n{dist.to_string()}")
    log.info(f"  Zoning violations: {violations}")
    log.info(f"  Saved to: {output_csv}")

    # Also export as GeoJSON for GIS / dashboard
    try:
        import geopandas as gpd
        gdf_plan = gpd.GeoDataFrame(
            plan.drop(columns=["geometry"], errors="ignore"),
            geometry=system.gdf.geometry.values,
            crs=system.gdf.crs,
        ).to_crs("EPSG:4326")
        geojson_out = CHECKPOINT_DIR / "manhattan_landuse_plan_42k.geojson"
        gdf_plan.to_file(geojson_out, driver="GeoJSON")
        log.info(f"  GeoJSON saved to: {geojson_out}")
    except Exception as e:
        log.warning(f"  GeoJSON export failed: {e}")

    mark_done("6_plan", {
        "n_parcels": len(plan),
        "zoning_violations": violations,
        "output_csv": str(output_csv),
    })

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("FULL RUN COMPLETE")
    log.info(f"Total wall time : {elapsed(run_start)}")
    log.info(f"Parcels optimised: {n_parcels:,}")
    log.info(f"Outputs in       : {CHECKPOINT_DIR}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()

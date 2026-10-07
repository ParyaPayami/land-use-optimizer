"""
All experiments reported in the manuscript, driven by one YAML config.

    pimaluos run --config configs/paper.yaml --pluto /path/to/MapPLUTO.zip --out results/paper
    pimaluos report --results results/paper --out paper/generated

Outputs (``<out>/``):

``manifest.json``        data provenance, config, versions, hardware, timings
``graph_summary.json``   node/edge counts per relation (exact)
``gnn.json``             per-seed pre-training curves (train/validation)
``edge_ablation.json``   per-seed validation loss for each edge configuration
``marl.json``            per-seed, per-variant training histories
``metrics.csv``          long table: seed, method, verified, metric, value (all outcomes)
``nash.json``            per-seed voting-game analysis
``voting_sensitivity.json``  post-hoc aggregation with varied voting weights
``pareto.json``          NSGA-III cold vs seeded: fronts, knees, hypervolume curves
``uncertainty.json``     Monte Carlo re-evaluation of the first seed's repaired plans
``plans/*.npz``          added floor area by use for every plan (for maps)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

import pimaluos
from pimaluos import baselines as B
from pimaluos.context.build import build_context, load_context, save_context
from pimaluos.core.data_loader import SyntheticCityLoader, get_data_loader
from pimaluos.core.graph_builder import ALL_EDGE_TYPES, RELATION_NAMES
from pimaluos.models.agents import AGENT_TYPES, MARLTrainer, PPOConfig
from pimaluos.models.gnn import ParcelGNN
from pimaluos.models.nash import analyse_consensus
from pimaluos.models.pareto import normalisation, plan_objectives, run_nsga3
from pimaluos.outcomes import USES, OutcomeParams
from pimaluos.physics.capacity import CapacityParams
from pimaluos.pipeline import UrbanOptSystem
from pimaluos.uncertainty import monte_carlo

logger = logging.getLogger("pimaluos.experiments")

DEFAULT_CONFIG = {
    "city": "manhattan",
    "pluto_release": None,
    "seeds": [0, 1, 2, 3, 4],
    "graph": {"k_neighbors": 8},
    "gnn": {"epochs": 300, "lr": 1e-3, "patience": 50, "hidden_channels": 256, "embed_dim": 128, "heads": 4},
    "edge_ablation": {"enabled": True, "seeds": [0, 1, 2], "epochs": 150},
    "marl": {"iterations": 100, "horizon": 10, "delta_far": 0.5, "ppo": {}},
    "variants": ["pimaluos", "no_gnn", "no_capacity_feedback", "single_agent_planner", "no_equity_agent",
                 "self_aware"],
    "awareness": 0.5,
    "baseline_horizon": 10,
    "nash": {"enabled": True, "n_lots": 200},
    "voting_sensitivity": {"enabled": True, "developer_weights": [0.1, 0.2, 0.3, 0.4, 0.5]},
    "pareto": {"enabled": True, "seeds": [0, 1, 2], "pop_size": 120, "generations": 100, "n_partitions": 2},
    "uncertainty": {"enabled": True, "n_draws": 200},
    "capacity": {},
    "outcomes": {},
    "context": {"raw_dir": "data/raw/context", "cache_dir": "data/processed/context_{city}"},
}


def _merge(base: Dict, over: Dict) -> Dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path: Optional[str]) -> Dict:
    cfg = DEFAULT_CONFIG
    if path:
        cfg = _merge(DEFAULT_CONFIG, yaml.safe_load(Path(path).read_text()) or {})
    return cfg



def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file so an interrupted write never leaves a corrupt checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _hardware() -> Dict:
    """CPU model, core count, memory and GPU availability, for the paper's computation note."""
    cpu, mem = platform.processor() or platform.machine(), None
    try:
        info = Path("/proc/cpuinfo").read_text()
        cpu = next(ln.split(":", 1)[1].strip() for ln in info.splitlines() if ln.startswith("model name"))
        kb = next(int(ln.split()[1]) for ln in Path("/proc/meminfo").read_text().splitlines()
                  if ln.startswith("MemTotal"))
        mem = round(kb / 1024 ** 2, 1)
    except (OSError, StopIteration, ValueError):
        pass
    return {"cpu_model": cpu, "cpu_count": os.cpu_count(), "memory_gb": mem, "cuda": torch.cuda.is_available()}

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


class _Recorder:
    def __init__(self):
        self.rows: List[Dict] = []

    def add(self, seed, method, verified, summary: Dict):
        for k, v in summary.items():
            self.rows.append({"seed": seed, "method": method, "verified": bool(verified), "metric": k, "value": v})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def run_all(config: Dict, out_dir: Path, pluto_path: Optional[str] = None) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "plans").mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}
    t_start = time.time()

    # ------------------------------------------------------------ data + graph
    t0 = time.time()
    ctx = None
    if config["city"] == "synthetic":
        ds = SyntheticCityLoader(**config.get("synthetic", {})).load()
    else:
        ds = get_data_loader(config["city"], pluto_path).load()
        cc = config["context"]
        cache = Path(cc["cache_dir"].format(city=config["city"]))
        if (cache / "meta.json").exists():
            ctx = load_context(cache)
            if len(ctx.node_of_lot) != len(ds.gdf):
                ctx = None
        if ctx is None:
            ctx = build_context(ds.gdf, cc["raw_dir"])
            save_context(ctx, cache)
    timings["data_s"] = time.time() - t0
    gnn_cfg = dict(config["gnn"])
    gnn_kwargs = {k: gnn_cfg.pop(k) for k in ["hidden_channels", "embed_dim", "heads"] if k in gnn_cfg}
    sysm = UrbanOptSystem(ds, ctx, CapacityParams(**config.get("capacity", {})),
                          OutcomeParams(**config.get("outcomes", {})),
                          graph_kwargs=config.get("graph", {}), gnn_kwargs=gnn_kwargs)
    om = sysm.outcomes
    t0 = time.time()
    sysm.build_graph()
    timings["graph_s"] = time.time() - t0
    (out_dir / "graph_summary.json").write_text(json.dumps(sysm.graph_summary, indent=1))
    cap = sysm.capacity
    np.savez_compressed(out_dir / "plans" / "lots.npz", x=cap.x, y=cap.y, far0=cap.far0, ub=cap.ub)
    ds.gdf[["geometry"]].to_parquet(out_dir / "plans" / "lots_geometry.parquet")
    rec = _Recorder()
    baseline_summary = {**cap.baseline["summary"], **om.base["summary"]}
    env_ = om.env
    data_stats = {
        "existing_floor_area_sqft": float((cap.far0 * cap.A).sum()),
        "zoning_capacity_floor_area_sqft": float(((cap.far0 * cap.A) + env_.total).sum()),
        "capacity_res_market_sqft": float(env_.res_market.sum()),
        "capacity_res_uap_sqft": float((env_.res_total - env_.res_market).sum()),
        "capacity_commercial_sqft": float(env_.com.sum()), "capacity_facility_sqft": float(env_.facility.sum()),
        "capacity_total_sqft": float(env_.total.sum()), "n_mih_lots": int(env_.mih.sum()),
        "lots_with_headroom": int((env_.total > 1.0).sum()),
        "context": sysm.ctx.meta, "context_params": sysm.ctx.params,
        "n_taz": int(cap.n_taz), "n_catchments": int(cap.n_catch),
        "n_vulnerable_lots": int(cap.vulnerable.sum()), "n_flood_lots": int(cap.flood.sum()),
        "existing_taz_over_capacity": int(baseline_summary["taz_over_capacity"]),
        "existing_taz_over_capacity_frontage_only": int(cap.taz_over_frontage_only),
        "lots_above_zoning_max_existing": int(baseline_summary["lots_above_zoning_max_existing"]),
        "n_shadow_pairs": int(len(cap.pair_i)),
    }

    seeds = list(config["seeds"])
    gnn_out, marl_out, nash_out, vote_out = {}, {}, {}, {}
    mcfg = config["marl"]
    ppo = PPOConfig(**mcfg.get("ppo", {}))
    aw = config.get("awareness", 0.5)
    variant_specs = {
        "pimaluos": dict(use_gnn=True, physics_weight=1.0, agent_types=AGENT_TYPES, awareness=aw),
        "no_gnn": dict(use_gnn=False, physics_weight=1.0, agent_types=AGENT_TYPES, awareness=aw),
        "no_capacity_feedback": dict(use_gnn=True, physics_weight=0.0, agent_types=AGENT_TYPES, awareness=aw),
        "single_agent_planner": dict(use_gnn=True, physics_weight=1.0, agent_types=["planner"], awareness=aw),
        "no_equity_agent": dict(use_gnn=True, physics_weight=1.0,
                                agent_types=[a for a in AGENT_TYPES if a != "equity_advocate"], awareness=aw),
        "self_aware": dict(use_gnn=True, physics_weight=1.0, agent_types=AGENT_TYPES, awareness=0.0),
    }
    timings.update(gnn_s=0.0, marl_s=0.0, gnn_epochs=0, marl_iters=0)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    for seed in seeds:
        ck = ckpt_dir / f"seed_{seed}.json"
        if ck.exists():
            done = json.loads(ck.read_text())
            logger.info("=== seed %d: loaded from checkpoint ===", seed)
            gnn_out[seed], marl_out[seed] = done["gnn"], done["marl"]
            rec.rows.extend(done["rows"])
            if done.get("nash") is not None:
                nash_out[seed] = done["nash"]
            if done.get("vote") is not None:
                vote_out[seed] = done["vote"]
            for k in ("gnn_s", "marl_s", "gnn_epochs", "marl_iters"):
                timings[k] += done["timings"].get(k, 0)
            continue
        n_rows_before = len(rec.rows)
        # Seconds and the epochs/iterations they cover (an interrupted stage counts only
        # the work done after resuming, so the per-epoch rates stay exact).
        seed_t = {"gnn_s": 0.0, "marl_s": 0.0, "gnn_epochs": 0, "marl_iters": 0}
        logger.info("=== seed %d ===", seed)
        # Stage checkpoints inside a seed, so an interrupted run loses at most one stage.
        gnn_ck = ckpt_dir / f"seed_{seed}_gnn.pt"
        if gnn_ck.exists():
            st = torch.load(gnn_ck, weights_only=False)
            model = ParcelGNN(sysm.graph["parcel"].x.shape[1], list(sysm.graph.edge_types), **sysm.gnn_kwargs)
            model.load_state_dict(st["state_dict"])
            model.eval()
            sysm.gnn = model
            gnn_out[seed] = st["history"]
            seed_t["gnn_s"] = st["seconds"]
            seed_t["gnn_epochs"] = st.get("epochs_trained", len(st["history"]["train_loss"]))
            logger.info("seed %d: GNN loaded from stage checkpoint", seed)
        else:
            t0 = time.time()
            res = sysm.pretrain_gnn(epochs=gnn_cfg["epochs"], seed=seed, lr=gnn_cfg.get("lr", 1e-3),
                                    patience=gnn_cfg.get("patience", 50),
                                    resume_path=str(ckpt_dir / f"seed_{seed}_gnn.partial.pt"))
            seed_t["gnn_s"] = time.time() - t0
            seed_t["gnn_epochs"] = len(res["history"]["train_loss"]) - res["history"]["resumed_from_epoch"]
            gnn_out[seed] = res["history"]
            _atomic_torch_save({"state_dict": sysm.gnn.state_dict(), "history": res["history"],
                                "seconds": seed_t["gnn_s"], "epochs_trained": seed_t["gnn_epochs"]}, gnn_ck)
        timings["gnn_s"] += seed_t["gnn_s"]
        timings["gnn_epochs"] += seed_t["gnn_epochs"]

        # Baselines (deterministic except random).
        plans = {
            "status_quo": B.status_quo(om),
            "random": B.random_plan(om, config["baseline_horizon"], mcfg["delta_far"], seed),
            "rule_based": B.rule_based_plan(om, config["baseline_horizon"], mcfg["delta_far"]),
            "buildout_market": B.buildout_market(om),
            "buildout_uap": B.buildout_uap(om),
        }
        marl_out[seed] = {}
        trainers = {}
        for v in config["variants"]:
            spec = variant_specs[v]
            env = sysm.make_env(use_gnn=spec["use_gnn"], physics_weight=spec["physics_weight"],
                                agent_types=spec["agent_types"], horizon=mcfg["horizon"],
                                delta_far=mcfg["delta_far"], awareness=spec["awareness"])
            v_ck = ckpt_dir / f"seed_{seed}_{v}.pt"
            if v_ck.exists():
                st = torch.load(v_ck, weights_only=False)
                tr = MARLTrainer(env, ppo, seed=seed)
                for a, m in tr.agents.items():
                    m.load_state_dict(st["agents"][a])
                marl_out[seed][v], dt = st["history"], st["seconds"]
                n_it = st.get("iterations_trained", len(st["history"]))
                logger.info("seed %d: MARL variant %s loaded from stage checkpoint", seed, v)
            else:
                t0 = time.time()
                tr = sysm.train_marl(env, mcfg["iterations"], seed, ppo,
                                     resume_path=str(ckpt_dir / f"seed_{seed}_{v}.partial.pt"))
                dt = time.time() - t0
                n_it = mcfg["iterations"] - tr.resumed_from
                marl_out[seed][v] = [
                    {"iteration": h["iteration"],
                     "returns": {a: h[a]["episode_return"] for a in spec["agent_types"]},
                     "entropy": {a: h[a]["entropy"] for a in spec["agent_types"]},
                     "added_floor_area_sqft": h["plan_summary"]["added_floor_area_sqft"]}
                    for h in tr.history]
                # seconds covers only the iterations trained by this process.
                _atomic_torch_save({"agents": {a: m.state_dict() for a, m in tr.agents.items()},
                                    "history": marl_out[seed][v], "seconds": dt,
                                    "iterations_trained": n_it}, v_ck)
            seed_t["marl_s"] += dt
            seed_t["marl_iters"] += n_it
            timings["marl_s"] += dt
            timings["marl_iters"] += n_it
            trainers[v] = tr
            plans[v] = tr.final_plan()[0]

        verified = {}
        for name, plan in plans.items():
            rec.add(seed, name, False, om.evaluate(plan)["summary"])
            vplan, hist = sysm.verify(plan)
            verified[name] = vplan
            s = om.evaluate(vplan)["summary"]
            s["repair_iterations"] = len(hist) - 1
            if s["capacity_violations"]:
                logger.warning("seed %d %s: %d violations remain after repair", seed, name, s["capacity_violations"])
            rec.add(seed, name, True, s)
            if seed == seeds[0]:
                np.savez_compressed(out_dir / "plans" / f"{name}.npz", plan=plan, plan_verified=vplan)

        if config["nash"]["enabled"] and "pimaluos" in trainers:
            nash_out[seed] = analyse_consensus(trainers["pimaluos"].env, plans["pimaluos"],
                                               n_lots=config["nash"]["n_lots"], seed=seed)

        if config["voting_sensitivity"]["enabled"] and "pimaluos" in trainers:
            tr = trainers["pimaluos"]
            vote_out[seed] = []
            for wd in config["voting_sensitivity"]["developer_weights"]:
                rest = (1.0 - wd) / (len(AGENT_TYPES) - 1)
                w = {a: (wd if a == "developer" else rest) for a in AGENT_TYPES}
                tr.env.voting.weights = w
                vp, _ = tr.final_plan()
                vote_out[seed].append({"developer_weight": wd, **om.evaluate(vp)["summary"]})
            tr.env.voting.weights = {a: 0.2 for a in AGENT_TYPES}

        if seed == seeds[0] and "pimaluos" in trainers:
            torch.save({"gnn": sysm.gnn.state_dict(),
                        "agents": {a: m.state_dict() for a, m in trainers["pimaluos"].agents.items()}},
                       out_dir / "pimaluos_seed0.pt")
        _atomic_write_text(ck, json.dumps({"gnn": gnn_out[seed], "marl": marl_out[seed],
                                  "rows": rec.rows[n_rows_before:], "nash": nash_out.get(seed),
                                  "vote": vote_out.get(seed), "timings": seed_t}, default=_json_default))

    rec.frame().to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "gnn.json").write_text(json.dumps(gnn_out))
    (out_dir / "marl.json").write_text(json.dumps(marl_out))
    (out_dir / "nash.json").write_text(json.dumps(nash_out, default=_json_default))
    (out_dir / "voting_sensitivity.json").write_text(json.dumps(vote_out))

    # ------------------------------------------------------------ edge ablation
    if config["edge_ablation"]["enabled"]:
        configs = {"all": ALL_EDGE_TYPES, "spatial_adjacency_only": ["spatial_adjacency"]}
        for et in ALL_EDGE_TYPES:
            configs[f"without_{et}"] = [e for e in ALL_EDGE_TYPES if e != et]
        abl_ck = ckpt_dir / "edge_ablation.json"
        abl = json.loads(abl_ck.read_text()) if abl_ck.exists() else {}
        configs["no_graph"] = []
        for name, ets in configs.items():
            rels = [RELATION_NAMES[e] for e in ets]
            abl.setdefault(name, {"edge_types": ets, "val_loss": {}})
            for seed in config["edge_ablation"]["seeds"]:
                if str(seed) in abl[name]["val_loss"]:
                    continue
                ta = time.time()
                h = sysm.pretrain_gnn(epochs=config["edge_ablation"]["epochs"], seed=seed, relations=rels,
                                      patience=gnn_cfg.get("patience", 50),
                                      resume_path=str(ckpt_dir / f"ablation_{name}_{seed}.partial.pt"))["history"]
                abl[name]["val_loss"][str(seed)] = h["best_val_loss"]
                abl[name].setdefault("seconds", {})[str(seed)] = time.time() - ta
                if name == "no_graph":
                    abl.setdefault("mean_predictor", {"edge_types": [], "val_loss": {}})
                    abl["mean_predictor"]["val_loss"][str(seed)] = h["mean_predictor_val_loss"]
                _atomic_write_text(abl_ck, json.dumps(abl, indent=1))
        (out_dir / "edge_ablation.json").write_text(json.dumps(abl, indent=1))
        # Summed per run so that time spent before an interruption is counted.
        timings["edge_ablation_s"] = float(sum(sum(v.get("seconds", {}).values()) for v in abl.values()))

    # ------------------------------------------------------------ Pareto
    seed0 = {k: np.load(out_dir / "plans" / f"{k}.npz") for k in ["pimaluos", "buildout_market", "buildout_uap"]
             if (out_dir / "plans" / f"{k}.npz").exists()}
    if config["pareto"]["enabled"]:
        pc = config["pareto"]
        seeds_for_init = [np.zeros((om.n, 4))]
        if "pimaluos" in seed0:
            seeds_for_init += [seed0["pimaluos"]["plan"], seed0["pimaluos"]["plan_verified"]]
        seeds_for_init += [seed0[k]["plan_verified"] for k in ("buildout_market", "buildout_uap") if k in seed0]
        refs = [B.buildout_market(om), B.buildout_uap(om)]
        par_ck = ckpt_dir / "pareto.json"
        par = json.loads(par_ck.read_text()) if par_ck.exists() else {}
        for seed in pc["seeds"]:
            par.setdefault(str(seed), {})
            for mode, sp in [("cold", None), ("seeded", seeds_for_init)]:
                if mode in par[str(seed)]:
                    continue
                tp = time.time()
                r = run_nsga3(om, pc["pop_size"], pc["generations"], pc["n_partitions"], seed, sp, refs)
                knee = r["plans"][r["knee"]]
                par[str(seed)][mode] = {"F": r["F"].tolist(), "knee": r["knee"], "hv": r["hv_history"],
                                        "knee_summary": om.evaluate(knee)["summary"], "feasible": r["feasible"],
                                        "n_solutions": int(len(r["F"])), "n_ref_dirs": r["n_ref_dirs"],
                                        "seconds": time.time() - tp}
                _atomic_write_text(par_ck, json.dumps(par, default=_json_default))
                if seed == pc["seeds"][0]:
                    np.savez_compressed(out_dir / "plans" / f"pareto_knee_{mode}.npz", plan=knee)
        # Is each first-seed plan dominated by a front, and how many front plans lie inside the
        # hypervolume reference box (hypervolume is zero when none do)?
        ref, _ = normalisation(om, refs)
        compare = {}
        if "pimaluos" in seed0:
            compare.update(pimaluos=seed0["pimaluos"]["plan"], pimaluos_verified=seed0["pimaluos"]["plan_verified"])
        compare.update({f"{k}_verified": seed0[k]["plan_verified"] for k in ("buildout_market", "buildout_uap")
                        if k in seed0})
        f_plans = {k: plan_objectives(om, f)[0] for k, f in compare.items()}
        for runs in par.values():
            for e in runs.values():
                F = np.asarray(e["F"])
                e["n_in_reference_box"] = int((F <= ref).all(axis=1).sum())
                e["dominates_plan"] = {k: bool(((F <= f).all(axis=1) & (F < f).any(axis=1)).any())
                                       for k, f in f_plans.items()}
        (out_dir / "pareto.json").write_text(json.dumps(par, default=_json_default))
        timings["pareto_s"] = float(sum(e.get("seconds", 0.0) for v in par.values() for e in v.values()))

    # ------------------------------------------------------------ uncertainty
    if config.get("uncertainty", {}).get("enabled") and seed0:
        tu = time.time()
        names = [n for n in ["pimaluos", "buildout_market", "buildout_uap"] if n in seed0]
        others = ["rule_based", "no_capacity_feedback", "single_agent_planner"]
        mc_plans = {f"{n}_verified": seed0[n]["plan_verified"] for n in names}
        mc_plans.update({f"{n}_verified": np.load(out_dir / "plans" / f"{n}.npz")["plan_verified"] for n in others
                         if (out_dir / "plans" / f"{n}.npz").exists()})
        unc = monte_carlo(om, mc_plans, n_draws=config["uncertainty"]["n_draws"], seed=0)
        (out_dir / "uncertainty.json").write_text(json.dumps(unc, default=_json_default))
        timings["uncertainty_s"] = time.time() - tu

    timings["total_s"] = time.time() - t_start
    manifest = {
        "pimaluos_version": pimaluos.__version__,
        "config": config,
        "data": ds.meta,
        "data_stats": data_stats,
        "baseline_summary": baseline_summary,
        "capacity_params": cap.params_dict(),
        "outcome_params": om.params_dict(),
        "uses": list(USES),
        "timings": timings,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform(), "processor": platform.processor(),
                        **_hardware(),
                        "threads": torch.get_num_threads()},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    logger.info("All experiments finished in %.1f s -> %s", timings["total_s"], out_dir)
    return out_dir

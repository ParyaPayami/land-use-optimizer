"""
Turn a results directory into manuscript assets.

    pimaluos report --results results/paper --out paper/generated [--rag results/rag]

Writes ``macros.tex`` (every number quoted in the text), ``tab_*.tex`` and
``fig_*.pdf``. Every macro in :data:`MACROS` is always defined; values that are
unavailable render as ``\\textbf{[TBD]}`` so a missing experiment is visible in
the PDF instead of silently omitted. ``write_pending`` writes the all-TBD file
used before any run exists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

TBD = r"\textbf{[TBD]}"

# Plans and outcomes for which per-method macros are generated:
# \<Method><Outcome>Prop (proposed plan) and \<Method><Outcome>Ver (after repair), and
# \PimVs<Method><Outcome>Pct (percentage difference of repaired PIMALUOS vs the repaired method).
METHOD_MACRO = {"status_quo": "Sq", "random": "Rand", "rule_based": "Rule", "buildout_market": "BoMkt",
                "buildout_uap": "BoUap", "pimaluos": "Pim", "no_gnn": "NoGnn", "no_capacity_feedback": "NoCap",
                "single_agent_planner": "Single", "no_equity_agent": "NoEq", "self_aware": "SelfAware"}
OUTCOME_MACRO = {  # metric: (suffix, scale, decimals)
    "homes_added": ("HomesK", 1e-3, 1), "affordable_homes_added": ("AffHomesK", 1e-3, 1),
    "affordable_share": ("AffSharePct", 100, 1), "access_index": ("Access", 1, 3),
    "residents_full_15min_share": ("FullPct", 100, 2), "jobs_added": ("JobsK", 1e-3, 1),
    "tax_revenue_musd": ("TaxM", 1, 0), "market_value_added_busd": ("ValueB", 1, 1),
    "jobs_housing_balance": ("JH", 1, 3), "land_use_mix": ("Mix", 1, 3),
    "lifecycle_carbon_kt": ("CarbonMt", 1e-3, 2), "embodied_carbon_kt": ("EmbMt", 1e-3, 2),
    "operational_carbon_kt_per_yr": ("OpKt", 1, 0), "transit_oriented_share": ("TransitPct", 100, 1),
    "flood_zone_added_floor_area_sqft": ("FloodM", 1e-6, 2), "vulnerable_lot_added_floor_area_sqft": ("VulnM", 1e-6, 2),
    "displacement_exposure_share": ("DispPct", 100, 1), "added_floor_area_sqft": ("FAM", 1e-6, 1),
    "lots_newly_shaded": ("Shaded", 1, 0), "traffic_violations": ("Traffic", 1, 1),
    "catchments_over_capacity": ("Sewer", 1, 1), "repair_iterations": ("RepairIt", 1, 1),
}
COMPARATORS = ["random", "rule_based", "buildout_market", "buildout_uap", "no_gnn", "no_capacity_feedback",
               "single_agent_planner", "no_equity_agent", "self_aware"]
UNC_COMPARATORS = ["buildout_market", "buildout_uap", "rule_based", "no_capacity_feedback", "single_agent_planner"]
UNC_OUTCOMES = {"homes_added": "Homes", "affordable_homes_added": "AffHomes", "access_index": "Access",
                "residents_full_15min_share": "Full", "jobs_added": "Jobs", "tax_revenue_musd": "Tax",
                "market_value_added_busd": "Value", "jobs_housing_balance": "JH", "land_use_mix": "Mix",
                "embodied_carbon_kt": "Emb", "operational_carbon_kt_per_yr": "Op", "lifecycle_carbon_kt": "Carbon",
                "transit_oriented_share": "Transit", "flood_zone_added_floor_area_sqft": "Flood",
                "vulnerable_lot_added_floor_area_sqft": "Vuln"}
BASE_MACROS = {"access_index": ("BaseAccess", 1, 3), "residents_full_15min_share": ("BaseFullPct", 100, 1),
               "jobs_housing_balance": ("BaseJH", 1, 3), "land_use_mix": ("BaseMix", 1, 3)}

CONTEXT_MACROS = [
    "WalkNodes", "WalkshedNodes", "WalkMinutes", "DestFood", "DestHealth", "DestEducation", "DestParksAcres",
    "DestCivic", "DestTransit", "JobsTotalM", "PopTotalM", "WorkersTotalK", "PersonsPerUnit", "UnitSqft",
    "EmployedPerResident", "JobsKsfOffice", "JobsKsfRetail", "JobsKsfFacility", "JobsModelRsq",
    "EciRes", "EciOffice", "EciRetail", "EciFacility", "EciResN", "EciOfficeN", "EciFacilityN",
    "OpcRes", "OpcOffice", "OpcRetail", "OpcFacility", "OpcResN", "OpcOfficeN", "OpcRetailN", "OpcFacilityN",
    "FacilitySqftPerSite", "DailyNeedsSharePct", "CapResMarketM", "CapResUapM", "CapComM", "CapFacM", "CapTotalM",
    "NMihLots", "TaxRateTwo", "TaxRateFour", "MihShare", "LifecycleYears", "AffValueFactor", "NUncDraws",
]

MACROS = [
    # data / graph
    "NParcels", "NParcelsRaw", "NFeaturesEngineered", "NFeatures", "PlutoRelease",
    "NEdgesDirected", "NEdgesUndirected", "NIsolated",
    "EdgesAdjDir", "EdgesProxDir", "EdgesFunDir", "EdgesStreetDir", "EdgesRegDir",
    "ExistingFAM", "ZoningCapFAM", "LotsHeadroom", "LotsAboveMax", "NTaz", "NCatch", "ExistingTazOverRaw",
    "NVulnerable", "NFlood", "ExistingTazOver", "NSeeds", "NSeedsAbl", "NSeedsPareto",
    # GNN
    "GnnValLoss", "GnnMeanPredLoss", "GnnNoGraphLoss", "GnnEpochs", "GnnBestEpoch",
    # Nash
    "NashLots", "NashNontrivial", "NashShareSincereOpt", "NashLossSincere", "NashLossWorstNE",
    "NashMedianPoA", "NashPoaDefined", "NashMeanNE",
    # Pareto
    "ParetoPop", "ParetoGen", "ParetoRefDirs", "ParetoNCold", "ParetoNSeeded", "HVCold", "HVSeeded",
    "ParetoColdInBox", "ParetoColdFeasible", "ParetoSeededFeasible",
    "ParetoDomPim", "ParetoDomPimVer", "ParetoDomBoMktVer", "ParetoDomBoUapVer",
    "KneeHomesK", "KneeAffHomesK", "KneeAccess", "KneeJobsK", "KneeCarbonMt", "KneeShaded", "KneeVulnM", "KneeJH",
    # budgets
    "GnnPatience", "MarlIters", "MarlHorizon", "AblEpochs", "Awareness",
    # learning dynamics of the PIMALUOS variant (seed means)
    "MarlChangeLastTenPct", "MarlChangeLastFiftyPct", "MarlFAFirstM", "MarlFALastM",
] + [f"Marl{a}{w}" for a in ("Res", "Dev", "Pla", "Env", "Eq") for w in ("First", "Last")] + [
    # timings / hardware
    "TimeTotalH", "GnnSecPerEpoch", "MarlSecPerIter", "AblMinPerConfig", "ParetoMinPerRun", "TimeGraphS",
    "Hardware", "TorchVersion",
] + CONTEXT_MACROS + [m for (m, _, _) in BASE_MACROS.values()] + [
    f"{mm}{om}{st}" for mm in METHOD_MACRO.values() for (om, _, _) in OUTCOME_MACRO.values() for st in ("Prop", "Ver")
] + [
    f"PimVs{METHOD_MACRO[c]}{om}Pct" for c in COMPARATORS for (om, _, _) in OUTCOME_MACRO.values()
] + [
    f"Unc{METHOD_MACRO[c]}{o}" for c in UNC_COMPARATORS for o in UNC_OUTCOMES.values()
]


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return TBD
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", "{,}")
    return f"{x:,.{nd}f}".replace(",", "{,}")


def _rate(t: Dict, seconds: str, count: Optional[str], n: Optional[int] = None) -> float:
    n = t.get(count, 0) if count else n
    return t[seconds] / n if n and t.get(seconds) else np.nan


def _n_abl_runs(abl) -> int:
    return sum(len(v.get("val_loss", {})) for k, v in (abl or {}).items() if k != "mean_predictor")


def _n_pareto_runs(pareto) -> int:
    return sum(len(v) for v in (pareto or {}).values())


def _compute_hours(t: Dict, config: Dict, gnn: Optional[Dict] = None) -> float:
    """Total computation in hours. Stages interrupted and resumed count only the work
    done after resuming, so GNN pre-training and multi-agent training are the measured
    seconds per epoch/iteration times the epochs/iterations the run required."""
    epochs = (sum(len(h["train_loss"]) for h in gnn.values()) if gnn
              else len(config["seeds"]) * config["gnn"]["epochs"])
    gnn_s = _rate(t, "gnn_s", "gnn_epochs") * epochs
    marl_s = _rate(t, "marl_s", "marl_iters") * len(config["seeds"]) * len(config["variants"]) \
        * config["marl"]["iterations"]
    keys = ("data_s", "graph_s", "edge_ablation_s", "pareto_s")
    if not (np.isfinite(gnn_s) and np.isfinite(marl_s)) or any(k not in t for k in keys):
        return t.get("total_s", np.nan) / 3600
    return (sum(t[k] for k in keys) + gnn_s + marl_s) / 3600


def _hardware_text(env: Dict) -> str:
    if env.get("cpu_count"):
        txt = f"{env['cpu_count']} CPU cores ({env['cpu_model']})"
        if env.get("memory_gb"):
            txt += f" with {env['memory_gb']:g} GB memory"
        txt += ", using the GPU" if env.get("cuda") else ", no GPU"
    else:
        txt = env.get("processor") or env["platform"]
    return txt.replace("_", r"\_").replace("(R)", "").replace("@", "at")


def _pm(vals, nd=2, scale=1.0):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float) * scale
    if len(v) == 0:
        return TBD
    if len(v) == 1:
        return _fmt(float(v[0]), nd)
    return f"{_fmt(float(v.mean()), nd)} $\\pm$ {_fmt(float(v.std(ddof=1)), nd)}"


def write_macros(values: Dict[str, str], path: Path):
    lines = ["% Auto-generated by pimaluos.reporting -- do not edit by hand."]
    for name in MACROS:
        lines.append(f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{values.get(name, TBD)}}}")
    path.write_text("\n".join(lines) + "\n")


TABLE_COLUMNS = {"tab_graph_rows": 3, "tab_edge_ablation_rows": 3, "tab_social_economic_rows": 9,
                 "tab_environment_rows": 9, "tab_voting_rows": 6, "tab_uncertainty_rows": 6, "tab_params_rows": 3}


def _fill_missing(out: Path):
    """Every table/list file the manuscript inputs must exist; missing ones get a TBD row."""
    for name, ncol in TABLE_COLUMNS.items():
        f = out / f"{name}.tex"
        if not f.exists():
            f.write_text(f"\\multicolumn{{{ncol}}}{{c}}{{{TBD} (run \\texttt{{pimaluos report}})}} \\\\\n")
    f = out / "features_used.tex"
    if not f.exists():
        f.write_text(TBD + "\n")


def write_pending(out: Path):
    """Write the all-TBD asset folder used before any run exists (paper/pending)."""
    out.mkdir(parents=True, exist_ok=True)
    write_macros({}, out / "macros.tex")
    _fill_missing(out)


def _metric(df: pd.DataFrame, method: str, metric: str, verified: bool):
    s = df[(df.method == method) & (df.metric == metric) & (df.verified == verified)]
    return s.sort_values("seed")["value"].astype(float).tolist()


# --------------------------------------------------------------------- figures
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (fixed order)
GREY, INK, MUTED, GRID = "#8a8986", "#0b0b0b", "#52514e", "#e4e3df"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.5, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })
    return plt


def _band(ax, curves, color, label):
    L = min(len(c) for c in curves)
    a = np.array([c[:L] for c in curves])
    m, s = a.mean(0), a.std(0)
    x = np.arange(1, L + 1)
    ax.plot(x, m, color=color, label=label)
    if len(curves) > 1:
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, linewidth=0)


def fig_gnn(gnn: Dict, out: Path):
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    _band(ax, [h["train_loss"] for h in gnn.values()], SERIES[0], "Training (masked train nodes)")
    _band(ax, [h["val_loss"] for h in gnn.values()], SERIES[1], "Validation (masked held-out nodes)")
    ref = np.mean([h["mean_predictor_val_loss"] for h in gnn.values()])
    ax.axhline(ref, color=GREY, linestyle="--", linewidth=1)
    ax.text(1, ref, " feature-mean predictor", va="bottom", color=MUTED, fontsize=7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked reconstruction MSE")
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(out / "fig_gnn_loss.pdf")
    plt.close(fig)


def fig_marl(marl: Dict, out: Path):
    plt = _style()
    agents = ["resident", "developer", "planner", "environmentalist", "equity_advocate"]
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.7), sharex=True)
    for ax, a in zip(axes, agents):
        curves = [[h["returns"][a] for h in marl[s]["pimaluos"]] for s in marl if "pimaluos" in marl[s]]
        if curves:
            _band(ax, curves, SERIES[0], a)
        ax.set_title(a.replace("_", " "), fontsize=8)
        ax.set_xlabel("Iteration")
    axes[0].set_ylabel("Episode return")
    fig.tight_layout()
    fig.savefig(out / "fig_marl_returns.pdf")
    plt.close(fig)


def fig_hv(pareto: Dict, out: Path):
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for mode, c in [("cold", SERIES[2]), ("seeded", SERIES[1])]:
        _band(ax, [pareto[s][mode]["hv"] for s in pareto], c, mode.capitalize() + " start")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalised hypervolume")
    ax.legend(fontsize=7)
    fig.savefig(out / "fig_hypervolume.pdf")
    plt.close(fig)


# --------------------------------------------------------------------- tables
def table_graph(graph: Dict, out: Path):
    names = {"spatial_adjacency": "Spatial adjacency", "proximity": "Proximity (k-NN)",
             "functional_similarity": "Functional similarity", "street_frontage": "Street frontage",
             "regulatory_coupling": "Regulatory coupling"}
    lines = []
    for k, lab in names.items():
        e = graph["edge_types"].get(k)
        if e:
            lines.append(f"{lab} & {_fmt(e['directed'])} & {_fmt(e['undirected'])} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Total & {_fmt(graph['total_directed'])} & {_fmt(graph['total_undirected'])} \\\\")
    (out / "tab_graph_rows.tex").write_text("\n".join(lines) + "\n")


def table_ablation(abl: Dict, out: Path):
    order = ["all"] + [k for k in abl if k.startswith("without_")] + ["spatial_adjacency_only", "no_graph",
                                                                     "mean_predictor"]
    lab = {"all": "All five relations", "spatial_adjacency_only": "Spatial adjacency only",
           "no_graph": "No graph (MLP)", "mean_predictor": "Feature-mean predictor"}
    lines = []
    for k in order:
        if k not in abl:
            continue
        name = lab.get(k, "Without " + k.replace("without_", "").replace("_", " "))
        lines.append(f"{name} & {len(abl[k]['edge_types'])} & {_pm(list(abl[k]['val_loss'].values()), 3)} \\\\")
    (out / "tab_edge_ablation_rows.tex").write_text("\n".join(lines) + "\n")


def table_features(manifest: Dict, out: Path):
    feats = manifest["data"]["features_used"]
    txt = ", \\allowbreak ".join("\\texttt{" + f.replace("_", r"\_") + "}" for f in feats)
    (out / "features_used.tex").write_text(txt + "\n")


METHOD_ROWS = [("status_quo", "Status quo"), ("random", "Random"), ("rule_based", "Rule-based growth"),
               ("buildout_market", "Build-out, market rate"), ("buildout_uap", "Build-out with UAP"),
               ("pimaluos", "PIMALUOS"), ("no_gnn", "\\quad No GNN"),
               ("no_capacity_feedback", "\\quad No capacity feedback"),
               ("single_agent_planner", "\\quad Single agent (planner)"),
               ("no_equity_agent", "\\quad No equity agent"), ("self_aware", "\\quad Self-aware agents")]


def _rows(df: pd.DataFrame, cols, out: Path, name: str):
    lines = []
    for m, lab in METHOD_ROWS:
        if not _metric(df, m, "added_floor_area_sqft", True):
            continue
        cells = [_pm(_metric(df, m, k, v), nd, sc) for k, v, sc, nd in cols]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n")


def table_social_economic(df: pd.DataFrame, out: Path):
    """Repaired plans: homes, affordable homes, access, full 15-minute coverage, jobs, tax, J-H balance, mix."""
    cols = [("homes_added", True, 1e-3, 1), ("affordable_homes_added", True, 1e-3, 1), ("access_index", True, 1, 3),
            ("residents_full_15min_share", True, 100, 2), ("jobs_added", True, 1e-3, 1),
            ("tax_revenue_musd", True, 1, 0), ("jobs_housing_balance", True, 1, 3), ("land_use_mix", True, 1, 3)]
    _rows(df, cols, out, "tab_social_economic_rows")


def table_environment(df: pd.DataFrame, out: Path):
    """Proposed plans (screens before repair) and repaired plans (carbon, transit, flood, equity, floor area)."""
    cols = [("traffic_violations", False, 1, 1), ("catchments_over_capacity", False, 1, 1),
            ("lots_newly_shaded", False, 1, 0), ("lifecycle_carbon_kt", True, 1e-3, 2),
            ("transit_oriented_share", True, 100, 1), ("flood_zone_added_floor_area_sqft", True, 1e-6, 2),
            ("vulnerable_lot_added_floor_area_sqft", True, 1e-6, 2), ("added_floor_area_sqft", True, 1e-6, 1)]
    _rows(df, cols, out, "tab_environment_rows")


def table_voting(vote: Dict, out: Path):
    if not vote:
        return
    ws = [r["developer_weight"] for r in next(iter(vote.values()))]
    keys = [("homes_added", 1e-3, 1), ("affordable_homes_added", 1e-3, 1), ("access_index", 1, 3),
            ("jobs_added", 1e-3, 1), ("lifecycle_carbon_kt", 1e-3, 2)]
    lines = []
    for i, w in enumerate(ws):
        cells = [_pm([vote[s][i][k] for s in vote], nd, sc) for k, sc, nd in keys]
        lines.append(f"{w:.2f} & " + " & ".join(cells) + r" \\")
    (out / "tab_voting_rows.tex").write_text("\n".join(lines) + "\n")


def table_uncertainty(unc: Dict, out: Path):
    """Share of parameter draws in which repaired PIMALUOS is better than each comparator."""
    sb = unc["share_better"]
    labels = {"homes_added": "Homes", "affordable_homes_added": "Affordable homes", "access_index": "Access index",
              "jobs_added": "Jobs", "tax_revenue_musd": "Property tax", "jobs_housing_balance": "Jobs-housing balance",
              "land_use_mix": "Land-use mix", "lifecycle_carbon_kt": "Life-cycle carbon",
              "transit_oriented_share": "Transit-oriented share", "vulnerable_lot_added_floor_area_sqft":
              "Floor area on vulnerable lots"}
    comps = [c for c in UNC_COMPARATORS if f"pimaluos_verified|{c}_verified" in sb]
    lines = []
    for k, lab in labels.items():
        cells = [_fmt(100 * sb[f"pimaluos_verified|{c}_verified"][k], 0) for c in comps]
        cells += [""] * (5 - len(cells))
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    (out / "tab_uncertainty_rows.tex").write_text("\n".join(lines) + "\n")


PARAM_ROWS = [  # (label, key in outcome_params, decimals, source)
    ("Persons per new home", "persons_per_unit", 2, "2020 Census blocks / PLUTO units"),
    ("Floor area per new home (sq ft)", "unit_sqft_new", 0, "PLUTO, buildings since 2010"),
    ("Employed residents per resident", "employed_per_resident", 3, "LODES 2023 RAC / 2020 Census"),
    ("Jobs per 1,000 sq ft: office", "jobs_per_ksf_office", 2, "LODES 2023 WAC, NNLS on PLUTO"),
    ("Jobs per 1,000 sq ft: retail", "jobs_per_ksf_retail", 2, "LODES 2023 WAC, NNLS on PLUTO"),
    ("Jobs per 1,000 sq ft: community facility", "jobs_per_ksf_facility", 2, "LODES 2023 WAC, NNLS on PLUTO"),
    ("Embodied carbon, residential (kg/m$^2$)", "eci_res_kg_m2", 0, "CLF WBLCA v2, median"),
    ("Embodied carbon, office (kg/m$^2$)", "eci_office_kg_m2", 0, "CLF WBLCA v2, median"),
    ("Embodied carbon, community facility (kg/m$^2$)", "eci_facility_kg_m2", 0, "CLF WBLCA v2, median"),
    ("Operational carbon, residential (kg/ft$^2$/yr)", "opc_res_kg_ft2", 2, "LL84 2024, built 2010+"),
    ("Operational carbon, office (kg/ft$^2$/yr)", "opc_office_kg_ft2", 2, "LL84 2024, built 2010+"),
    ("Operational carbon, retail (kg/ft$^2$/yr)", "opc_retail_kg_ft2", 2, "LL84 2024, built 2010+"),
    ("Operational carbon, community facility (kg/ft$^2$/yr)", "opc_facility_kg_ft2", 2, "LL84 2024, built 2010+"),
    ("Floor area per new service site (sq ft)", "facility_sqft_per_site", 0, "PLUTO / Facilities Database"),
    ("Daily-needs share of new retail", "daily_needs_share_of_retail", 3, "food-store / retail floor area"),
    ("MIH affordable share", "mih_affordable_share", 2, "ZR 23-154, Option 1"),
    ("Tax rate class 2 (per \\$100 AV)", "tax_rate_class2", 3, "NYC DOF FY2026"),
    ("Tax rate class 4 (per \\$100 AV)", "tax_rate_class4", 3, "NYC DOF FY2026"),
    ("Assessment ratio, classes 2 and 4", "assessment_ratio", 2, "NYC DOF"),
    ("Income-restricted value factor", "affordable_value_factor", 2, "assumption (varied 0.3--0.6)"),
    ("Life-cycle period (years)", "lifecycle_years", 0, "assumption"),
]


def table_params(man: Dict, out: Path):
    op = man.get("outcome_params", {})
    lines = []
    for lab, k, nd, src in PARAM_ROWS:
        val = op.get(k)
        cell = _fmt(int(val), 0) if nd == 0 and val is not None else _fmt(val, nd)
        lines.append(f"{lab} & {cell} & {src} \\\\")
    (out / "tab_params_rows.tex").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------- figures (new)
PILLAR_AXES = [("homes_added", "Homes", 1), ("affordable_homes_added", "Affordable\nhomes", 1),
               ("access_index", "Access\nindex", 1), ("jobs_added", "Jobs", 1),
               ("jobs_housing_balance", "Jobs-housing\nbalance", 1), ("tax_revenue_musd", "Property\ntax", 1),
               ("lifecycle_carbon_kt", "Life-cycle\ncarbon (low)", -1),
               ("vulnerable_lot_added_floor_area_sqft", "Vulnerable-lot\nfloor area (low)", -1)]


def fig_pillars(df: pd.DataFrame, out: Path):
    """Repaired plans on every goal, min-max scaled across plans (1 = best plan on that goal)."""
    plt = _style()
    shown = [("rule_based", "Rule-based growth", GREY, ":"), ("buildout_market", "Build-out, market", GREY, "--"),
             ("buildout_uap", "Build-out with UAP", INK, "--"), ("pimaluos", "PIMALUOS", SERIES[0], "-"),
             ("no_capacity_feedback", "No capacity feedback", SERIES[1], "-"),
             ("single_agent_planner", "Single agent", SERIES[2], "-")]
    means = {}
    for m, *_ in shown:
        vals = [_metric(df, m, k, True) for k, _, _ in PILLAR_AXES]
        if all(vals):
            means[m] = np.array([np.mean(v) for v in vals])
    if len(means) < 2:
        return
    M = np.array(list(means.values()))
    sign = np.array([d for *_, d in PILLAR_AXES], float)
    lo, hi = M.min(0), M.max(0)
    S = np.where(hi > lo, (M - lo) / np.maximum(hi - lo, 1e-12), 0.5)
    S = np.where(sign > 0, S, 1 - S)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    x = np.arange(len(PILLAR_AXES))
    for (m, lab, c, ls) in shown:
        if m not in means:
            continue
        i = list(means).index(m)
        ax.plot(x, S[i], color=c, linestyle=ls, marker="o", markersize=3, label=lab,
                linewidth=2.0 if m == "pimaluos" else 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in PILLAR_AXES], fontsize=7)
    ax.set_ylabel("Scaled outcome (1 = best plan)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7)
    fig.savefig(out / "fig_pillars.pdf")
    plt.close(fig)


def fig_tradeoff(df: pd.DataFrame, pareto: Optional[Dict], out: Path):
    """Two projections of the eight-objective space for proposed plans and NSGA-III fronts:
    homes vs access index, and jobs vs life-cycle carbon."""
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    pairs = [(0, 2, "homes_added", "access_index", "Homes added (thousands)", "Access index", 1e-3, 1),
             (3, 5, "jobs_added", "lifecycle_carbon_kt", "Jobs added (thousands)", "Life-cycle carbon (Mt)",
              1e-3, 1e-3)]
    labelled = {"rule_based": "Rule-based", "buildout_market": "Build-out", "buildout_uap": "Build-out UAP",
                "pimaluos": "PIMALUOS", "random": "Random"}
    for ax, (i, j, kx, ky, lx, ly, sx, sy) in zip(axes, pairs):
        if pareto:
            s0 = sorted(pareto, key=lambda k: int(k))[0]
            for mode, c in [("cold", SERIES[2]), ("seeded", SERIES[1])]:
                F = np.array(pareto[s0][mode]["F"])
                sgn = [-1, -1, -1, -1, -1, 1, 1, 1]
                ax.scatter(sgn[i] * F[:, i] * sx, sgn[j] * F[:, j] * sy, s=8, color=c, alpha=0.55, linewidths=0,
                           label=f"NSGA-III ({mode})")
        for m, lab in labelled.items():
            a, b = _metric(df, m, kx, False), _metric(df, m, ky, False)
            if not a:
                continue
            p = (np.mean(a) * sx, np.mean(b) * sy)
            col = SERIES[0] if m == "pimaluos" else GREY
            ax.scatter(*p, s=26, color=col, edgecolors="white", linewidths=0.8, zorder=4)
            ax.annotate(lab, p, fontsize=6.5, color=INK, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
    axes[0].legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    fig.savefig(out / "fig_tradeoff.pdf")
    plt.close(fig)


USE_COLORS = {"Residential": SERIES[0], "Office": SERIES[1], "Retail": "#d6a62a", "Community facility": SERIES[2]}


def fig_map(results: Path, out: Path):
    """Repaired PIMALUOS plan (first seed): lots coloured by the use with most added floor area."""
    lots, plan = results / "plans" / "lots.npz", results / "plans" / "pimaluos.npz"
    geom = results / "plans" / "lots_geometry.parquet"
    if not (lots.exists() and plan.exists() and geom.exists()):
        return
    import geopandas as gpd
    from matplotlib.patches import Patch

    plt = _style()
    P = np.load(plan)["plan_verified"]
    g = gpd.read_parquet(geom)
    tot = P.sum(1)
    g["use"] = np.array(list(USE_COLORS))[P.argmax(1)]
    fig, ax = plt.subplots(figsize=(3.2, 5.4))
    g.plot(ax=ax, color=GRID, linewidth=0)
    inc = g[tot > 1.0]
    for u, c in USE_COLORS.items():
        sub = inc[inc["use"] == u]
        if len(sub):
            sub.plot(ax=ax, color=c, linewidth=0)
    ax.legend(handles=[Patch(color=c, label=u) for u, c in USE_COLORS.items()], loc="upper left", fontsize=6.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out / "fig_map.pdf")
    plt.close(fig)


MARL_AGENT_MACRO = {"resident": "Res", "developer": "Dev", "planner": "Pla", "environmentalist": "Env",
                    "equity_advocate": "Eq"}


def marl_macros(marl: Dict, variant: str = "pimaluos") -> Dict[str, str]:
    """Seed-mean returns of each stakeholder at the first and last iteration, the largest
    relative change of a stakeholder's return over the last 10 and 50 iterations, and the
    floor area proposed at the first and last iteration."""
    hist = [marl[s][variant] for s in marl if variant in marl[s]]
    if not hist:
        return {}
    n = min(len(h) for h in hist)
    out: Dict[str, str] = {}
    changes = {10: [], 50: []}
    for a, mm in MARL_AGENT_MACRO.items():
        r = np.array([[it["returns"][a] for it in h[:n]] for h in hist if a in h[0]["returns"]])
        if not len(r):
            continue
        mean = r.mean(0)
        out[f"Marl{mm}First"], out[f"Marl{mm}Last"] = _fmt(float(mean[0]), 2), _fmt(float(mean[-1]), 2)
        for k in changes:
            if n > k:
                changes[k].append(abs(mean[-1] - mean[-1 - k]) / max(abs(mean[-1 - k]), 1e-9))
    for k, name in ((10, "MarlChangeLastTenPct"), (50, "MarlChangeLastFiftyPct")):
        out[name] = _fmt(100 * max(changes[k]), 1) if changes[k] else TBD  # undefined for short runs
    fa = np.array([[it["added_floor_area_sqft"] for it in h[:n]] for h in hist]).mean(0)
    out.update(MarlFAFirstM=_fmt(float(fa[0]) / 1e6, 0), MarlFALastM=_fmt(float(fa[-1]) / 1e6, 0))
    return out


# --------------------------------------------------------------------- main
def make_report(results: Path, out: Path) -> Dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    R = lambda name: json.loads((results / name).read_text()) if (results / name).exists() else None  # noqa: E731
    man, graph = R("manifest.json"), R("graph_summary.json")
    gnn, abl, marl = R("gnn.json"), R("edge_ablation.json"), R("marl.json")
    nash, vote, pareto, unc = R("nash.json"), R("voting_sensitivity.json"), R("pareto.json"), R("uncertainty.json")
    df = pd.read_csv(results / "metrics.csv") if (results / "metrics.csv").exists() else pd.DataFrame(
        columns=["seed", "method", "verified", "metric", "value"])
    v: Dict[str, str] = {}

    if man:
        d, ds, t, cfg = man["data"], man["data_stats"], man["timings"], man["config"]
        if d.get("synthetic"):
            print("WARNING: results come from the SYNTHETIC test city; do not report them.")
        v.update(NParcels=_fmt(d["n_parcels_used"]), NParcelsRaw=_fmt(d["n_parcels_in_study_area"]),
                 NFeaturesEngineered=_fmt(d["n_features_engineered"]), NFeatures=_fmt(d["n_features_used"]),
                 PlutoRelease=(", ".join(d.get("pluto_versions") or []) or str(cfg.get("pluto_release") or TBD)),
                 ExistingFAM=_fmt(ds["existing_floor_area_sqft"] / 1e6, 1),
                 ZoningCapFAM=_fmt(ds["zoning_capacity_floor_area_sqft"] / 1e6, 1),
                 LotsHeadroom=_fmt(ds["lots_with_headroom"]), LotsAboveMax=_fmt(ds["lots_above_zoning_max_existing"]),
                 NTaz=_fmt(ds["n_taz"]), NCatch=_fmt(ds["n_catchments"]), NVulnerable=_fmt(ds["n_vulnerable_lots"]),
                 NFlood=_fmt(ds["n_flood_lots"]), ExistingTazOver=_fmt(ds["existing_taz_over_capacity"]),
                 ExistingTazOverRaw=_fmt(ds.get("existing_taz_over_capacity_frontage_only")),
                 NSeeds=_fmt(len(cfg["seeds"])), NSeedsAbl=_fmt(len(cfg["edge_ablation"]["seeds"])),
                 NSeedsPareto=_fmt(len(cfg["pareto"]["seeds"])), GnnEpochs=_fmt(cfg["gnn"]["epochs"]),
                 GnnPatience=_fmt(cfg["gnn"].get("patience", 50)), MarlIters=_fmt(cfg["marl"]["iterations"]),
                 MarlHorizon=_fmt(cfg["marl"]["horizon"]), AblEpochs=_fmt(cfg["edge_ablation"].get("epochs")),
                 Awareness=_fmt(cfg.get("awareness", 0.5), 1),
                 ParetoPop=_fmt(cfg["pareto"]["pop_size"]), ParetoGen=_fmt(cfg["pareto"]["generations"]),
                 NUncDraws=_fmt(cfg.get("uncertainty", {}).get("n_draws")),
                 TimeTotalH=_fmt(_compute_hours(t, cfg, gnn), 1),
                 GnnSecPerEpoch=_fmt(_rate(t, "gnn_s", "gnn_epochs"), 1),
                 MarlSecPerIter=_fmt(_rate(t, "marl_s", "marl_iters"), 1),
                 AblMinPerConfig=_fmt(_rate(t, "edge_ablation_s", None, n=_n_abl_runs(abl)) / 60, 1),
                 ParetoMinPerRun=_fmt(_rate(t, "pareto_s", None, n=_n_pareto_runs(pareto)) / 60, 1),
                 TimeGraphS=_fmt(t.get("graph_s", np.nan), 0),
                 Hardware=_hardware_text(man["environment"]),
                 TorchVersion=man["environment"]["torch"].replace("_", r"\_"))
        cm, cp, op = ds.get("context", {}), ds.get("context_params", {}), man.get("outcome_params", {})
        nd = cm.get("n_destinations", {})
        g = lambda dct, k, sc=1.0, n=2: _fmt(dct[k] * sc, n) if k in dct else TBD  # noqa: E731
        gi = lambda dct, k: _fmt(int(round(dct[k]))) if k in dct else TBD  # noqa: E731
        v.update(WalkNodes=gi(cm, "n_nodes"), WalkshedNodes=gi(cm, "mean_walkshed_nodes"),
                 WalkMinutes=gi(cp, "walk_minutes"), DestFood=gi(nd, "food"), DestHealth=gi(nd, "health"),
                 DestEducation=gi(nd, "education"), DestCivic=gi(nd, "civic"), DestTransit=gi(nd, "transit"),
                 DestParksAcres=gi(cm.get("supply_totals", {}), "parks"),
                 JobsTotalM=g(cm, "jobs_total", 1e-6, 2), PopTotalM=g(cm, "pop_total", 1e-6, 2),
                 WorkersTotalK=g(cm, "workers_total", 1e-3, 0), PersonsPerUnit=g(cp, "persons_per_unit"),
                 UnitSqft=gi(cp, "unit_sqft_new"), EmployedPerResident=g(cp, "employed_per_resident"),
                 JobsKsfOffice=g(cp, "jobs_per_ksf_office"), JobsKsfRetail=g(cp, "jobs_per_ksf_retail"),
                 JobsKsfFacility=g(cp, "jobs_per_ksf_facility"), JobsModelRsq=g(cp, "jobs_model_r2"),
                 EciRes=gi(cp, "eci_res_kg_m2"), EciOffice=gi(cp, "eci_office_kg_m2"),
                 EciRetail=gi(cp, "eci_retail_kg_m2"), EciFacility=gi(cp, "eci_facility_kg_m2"),
                 EciResN=gi(cp, "eci_res_n"), EciOfficeN=gi(cp, "eci_office_n"), EciFacilityN=gi(cp, "eci_facility_n"),
                 OpcRes=g(cp, "opc_res_kg_ft2"), OpcOffice=g(cp, "opc_office_kg_ft2"),
                 OpcRetail=g(cp, "opc_retail_kg_ft2"), OpcFacility=g(cp, "opc_facility_kg_ft2"),
                 OpcResN=gi(cp, "opc_res_n"), OpcOfficeN=gi(cp, "opc_office_n"), OpcRetailN=gi(cp, "opc_retail_n"),
                 OpcFacilityN=gi(cp, "opc_facility_n"),
                 FacilitySqftPerSite=gi(op, "facility_sqft_per_site"),
                 DailyNeedsSharePct=g(op, "daily_needs_share_of_retail", 100, 1),
                 CapResMarketM=g(ds, "capacity_res_market_sqft", 1e-6, 0),
                 CapResUapM=g(ds, "capacity_res_uap_sqft", 1e-6, 0),
                 CapComM=g(ds, "capacity_commercial_sqft", 1e-6, 0), CapFacM=g(ds, "capacity_facility_sqft", 1e-6, 0),
                 CapTotalM=g(ds, "capacity_total_sqft", 1e-6, 0), NMihLots=gi(ds, "n_mih_lots"),
                 TaxRateTwo=g(op, "tax_rate_class2", 1, 3), TaxRateFour=g(op, "tax_rate_class4", 1, 3),
                 MihShare=g(op, "mih_affordable_share", 100, 0), LifecycleYears=gi(op, "lifecycle_years"),
                 AffValueFactor=g(op, "affordable_value_factor", 1, 2))
        base = man.get("baseline_summary", {})
        for k, (name, sc, nd_) in BASE_MACROS.items():
            v[name] = _fmt(base[k] * sc, nd_) if k in base else TBD
        table_features(man, out)
        table_params(man, out)
    if graph:
        e = graph["edge_types"]
        v.update(NEdgesDirected=_fmt(graph["total_directed"]), NEdgesUndirected=_fmt(graph["total_undirected"]),
                 NIsolated=_fmt(graph["isolated_nodes"]),
                 EdgesAdjDir=_fmt(e.get("spatial_adjacency", {}).get("directed")),
                 EdgesProxDir=_fmt(e.get("proximity", {}).get("directed")),
                 EdgesFunDir=_fmt(e.get("functional_similarity", {}).get("directed")),
                 EdgesStreetDir=_fmt(e.get("street_frontage", {}).get("directed")),
                 EdgesRegDir=_fmt(e.get("regulatory_coupling", {}).get("directed")))
        table_graph(graph, out)
    if gnn:
        v.update(GnnValLoss=_pm([h["best_val_loss"] for h in gnn.values()], 3),
                 GnnMeanPredLoss=_pm([h["mean_predictor_val_loss"] for h in gnn.values()], 3),
                 GnnBestEpoch=_pm([h["best_epoch"] for h in gnn.values()], 0))
        fig_gnn(gnn, out)
    if abl:
        v["GnnNoGraphLoss"] = _pm(list(abl["no_graph"]["val_loss"].values()), 3)
        table_ablation(abl, out)
    if len(df):
        for m, mm in METHOD_MACRO.items():
            for k, (om, sc, nd_) in OUTCOME_MACRO.items():
                for st, ver in (("Prop", False), ("Ver", True)):
                    v[f"{mm}{om}{st}"] = _pm(_metric(df, m, k, ver), nd_, sc)
        for c in COMPARATORS:
            for k, (om, _, _) in OUTCOME_MACRO.items():
                x, y = np.array(_metric(df, "pimaluos", k, True)), np.array(_metric(df, c, k, True))
                if len(x) and len(x) == len(y) and np.all(np.abs(y) > 1e-12):
                    v[f"PimVs{METHOD_MACRO[c]}{om}Pct"] = _pm(list(100 * (x - y) / np.abs(y)), 1)
        table_social_economic(df, out)
        table_environment(df, out)
        fig_pillars(df, out)
        fig_tradeoff(df, pareto, out)
    if marl:
        fig_marl(marl, out)
        v.update(marl_macros(marl))
    if nash:
        S = [nash[s]["summary"] for s in nash]
        v.update(NashLots=_fmt(S[0]["n_lots"]), NashNontrivial=_pm([s["n_nontrivial"] for s in S], 0),
                 NashShareSincereOpt=_pm([100 * s["share_sincere_optimal"] for s in S], 1),
                 NashLossSincere=_pm([s["mean_welfare_loss_sincere"] for s in S], 3),
                 NashLossWorstNE=_pm([s["mean_welfare_loss_worst_ne"] for s in S], 3),
                 NashMedianPoA=_pm([s["median_poa"] for s in S], 2),
                 NashPoaDefined=_pm([100 * s["share_poa_defined"] for s in S], 1),
                 NashMeanNE=_pm([s["mean_ne_outcomes"] for s in S], 2))
    if vote:
        table_voting(vote, out)
    if pareto:
        P = list(pareto.values())
        ks = lambda key, sc, nd_: _pm([p["seeded"]["knee_summary"][key] for p in P], nd_, sc)  # noqa: E731
        v.update(ParetoRefDirs=_fmt(P[0]["cold"]["n_ref_dirs"]),
                 ParetoNCold=_pm([p["cold"]["n_solutions"] for p in P], 0),
                 ParetoNSeeded=_pm([p["seeded"]["n_solutions"] for p in P], 0),
                 HVCold=_pm([p["cold"]["hv"][-1] for p in P], 3), HVSeeded=_pm([p["seeded"]["hv"][-1] for p in P], 3),
                 ParetoColdFeasible=f"{sum(bool(p['cold'].get('feasible')) for p in P)} of {len(P)}",
                 ParetoSeededFeasible=f"{sum(bool(p['seeded'].get('feasible')) for p in P)} of {len(P)}",
                 KneeHomesK=ks("homes_added", 1e-3, 1), KneeAffHomesK=ks("affordable_homes_added", 1e-3, 1),
                 KneeAccess=ks("access_index", 1, 3), KneeJobsK=ks("jobs_added", 1e-3, 1),
                 KneeCarbonMt=ks("lifecycle_carbon_kt", 1e-3, 2), KneeShaded=ks("lots_newly_shaded", 1, 0),
                 KneeVulnM=ks("vulnerable_lot_added_floor_area_sqft", 1e-6, 2),
                 KneeJH=ks("jobs_housing_balance", 1, 3),
                 ParetoColdInBox=_fmt(int(sum(p["cold"].get("n_in_reference_box", 0) for p in P)))
                 if all("n_in_reference_box" in p["cold"] for p in P) else TBD)
        for key, plan in [("ParetoDomPim", "pimaluos"), ("ParetoDomPimVer", "pimaluos_verified"),
                          ("ParetoDomBoMktVer", "buildout_market_verified"),
                          ("ParetoDomBoUapVer", "buildout_uap_verified")]:
            d = [p["seeded"].get("dominates_plan", {}).get(plan) for p in P]
            v[key] = f"{sum(map(bool, d))} of {len(d)}" if all(x is not None for x in d) else TBD
        fig_hv(pareto, out)
    if unc:
        sb = unc["share_better"]
        for c in UNC_COMPARATORS:
            key = f"pimaluos_verified|{c}_verified"
            if key in sb:
                for k, o in UNC_OUTCOMES.items():
                    v[f"Unc{METHOD_MACRO[c]}{o}"] = _fmt(100 * sb[key][k], 0)
        table_uncertainty(unc, out)
    fig_map(results, out)

    write_macros(v, out / "macros.tex")
    _fill_missing(out)
    return v

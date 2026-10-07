"""
Parameter uncertainty: Monte Carlo re-evaluation of fixed plans.

Plans are held fixed and their outcomes are recomputed under ``n_draws``
random draws of the uncertain parameters:

* embodied and operational carbon intensity of each use: uniform between the
  observed interquartile range (CLF WBLCA v2; LL84 2024);
* jobs per sq ft of each use: x U(0.75, 1.25) (LODES calibration, R^2 < 0.5);
* persons per new unit: x U(0.9, 1.1);
* daily-needs share of new retail and floor area per new service site:
  x U(0.5, 2) (calibrated from existing conditions);
* value of income-restricted relative to market-rate floor area: U(0.3, 0.6).

For every pair of plans and every outcome, the share of draws in which one
plan is better than the other measures whether a comparison depends on these
assumptions.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Dict, List

import numpy as np

from pimaluos.outcomes import USES, OutcomeModel

# outcome -> +1 if larger is better, -1 if smaller is better
OUTCOME_DIRECTION = {
    "homes_added": 1, "affordable_homes_added": 1, "access_index": 1, "residents_full_15min_share": 1,
    "jobs_added": 1, "tax_revenue_musd": 1, "market_value_added_busd": 1, "jobs_housing_balance": 1,
    "land_use_mix": 1, "embodied_carbon_kt": -1, "operational_carbon_kt_per_yr": -1, "lifecycle_carbon_kt": -1,
    "transit_oriented_share": 1, "flood_zone_added_floor_area_sqft": -1,
    "vulnerable_lot_added_floor_area_sqft": -1,
}


def _draw(om: OutcomeModel, rng: np.random.Generator) -> OutcomeModel:
    ctx = copy.copy(om.ctx)
    cp = dict(ctx.params)
    for u in USES:
        cp[f"eci_{u}_kg_m2"] = rng.uniform(cp[f"eci_{u}_q25"], cp[f"eci_{u}_q75"])
        cp[f"opc_{u}_kg_ft2"] = rng.uniform(cp[f"opc_{u}_q25"], cp[f"opc_{u}_q75"])
    for u in ("office", "retail", "facility"):
        cp[f"jobs_per_ksf_{u}"] *= rng.uniform(0.75, 1.25)
    cp["persons_per_unit"] *= rng.uniform(0.9, 1.1)
    ctx.params = cp
    p = replace(om.p, daily_needs_share_of_retail=om.p.daily_needs_share_of_retail * rng.uniform(0.5, 2.0),
                facility_sqft_per_site=om.p.facility_sqft_per_site * rng.uniform(0.5, 2.0),
                affordable_value_factor=rng.uniform(0.3, 0.6))
    return OutcomeModel.__new__(OutcomeModel)._init_like(om, ctx, p)


def monte_carlo(om: OutcomeModel, plans: Dict[str, np.ndarray], n_draws: int = 200, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    keys = list(OUTCOME_DIRECTION)
    vals = {name: {k: [] for k in keys} for name in plans}
    for _ in range(n_draws):
        m = _draw(om, rng)
        for name, plan in plans.items():
            s = m.evaluate(plan)["summary"]
            for k in keys:
                vals[name][k].append(s[k])
    summary = {name: {k: {"median": float(np.median(v)), "q05": float(np.quantile(v, 0.05)),
                          "q95": float(np.quantile(v, 0.95))} for k, v in d.items()} for name, d in vals.items()}
    names: List[str] = list(plans)
    better = {}
    for a in names:
        for b in names:
            if a == b:
                continue
            better[f"{a}|{b}"] = {k: float(np.mean(OUTCOME_DIRECTION[k] * (np.array(vals[a][k]) - np.array(vals[b][k]))
                                                   > 1e-9)) for k in keys}
    return {"n_draws": n_draws, "summary": summary, "share_better": better}

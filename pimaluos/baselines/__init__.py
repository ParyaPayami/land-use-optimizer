"""
Baseline plans, expressed like PIMALUOS plans (added sq ft by use, lots x 4),
projected into the same zoning envelope and evaluated with the same outcome
model and repair loop.

* ``status_quo``          no additions.
* ``random_plan``         ``horizon`` steps of uniformly random actions.
* ``rule_based_plan``     as-of-right growth: each step, lots below 70 % of their
                          total capacity add residential floor area where
                          market-rate residential capacity remains, otherwise
                          commercial, otherwise community facility.
* ``buildout_market``     every lot filled to its total capacity, market-rate
                          residential first, then commercial, then facility
                          (single objective: floor area without affordability).
* ``buildout_uap``        as ``buildout_market`` but residential up to
                          AffResFAR, i.e. using the City of Yes Universal
                          Affordability Preference.
* repaired versions of any plan via :func:`pimaluos.physics.verification.repair_plan`;
  the repaired build-outs are capacity-aware greedy baselines.
"""

from __future__ import annotations

import numpy as np

from pimaluos.outcomes import FACILITY, OFFICE, RES, OutcomeModel


def status_quo(om: OutcomeModel) -> np.ndarray:
    return np.zeros((om.n, 4))


def random_plan(om: OutcomeModel, horizon: int = 10, delta: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    plan = np.zeros((om.n, 4))
    step = delta * om.A
    for _ in range(horizon):
        a = rng.integers(0, 5, om.n)
        add = np.zeros_like(plan)
        m = a > 0
        add[np.where(m)[0], a[m] - 1] = step[m]
        plan = om.env.clip(plan + add)
    return plan


def rule_based_plan(om: OutcomeModel, horizon: int = 10, delta: float = 0.5, threshold: float = 0.7) -> np.ndarray:
    e = om.env
    plan = np.zeros((om.n, 4))
    step = delta * om.A
    for _ in range(horizon):
        under = plan.sum(1) < threshold * e.total
        res_left = e.res_market - plan[:, RES] > 1.0
        com_left = e.com - plan[:, OFFICE] > 1.0
        add = np.zeros_like(plan)
        add[under & res_left, RES] = step[under & res_left]
        m = under & ~res_left & com_left
        add[m, OFFICE] = step[m]
        m = under & ~res_left & ~com_left
        add[m, FACILITY] = step[m]
        plan = e.clip(plan + add)
    return plan


def _fill(om: OutcomeModel, res_cap: np.ndarray) -> np.ndarray:
    e = om.env
    plan = np.zeros((om.n, 4))
    plan[:, RES] = np.minimum(res_cap, e.total)
    left = e.total - plan[:, RES]
    plan[:, OFFICE] = np.minimum(e.com, left)
    left = left - plan[:, OFFICE]
    plan[:, FACILITY] = np.minimum(e.facility, left)
    return e.clip(plan)


def buildout_market(om: OutcomeModel) -> np.ndarray:
    return _fill(om, om.env.res_market)


def buildout_uap(om: OutcomeModel) -> np.ndarray:
    return _fill(om, om.env.res_total)


__all__ = ["status_quo", "random_plan", "rule_based_plan", "buildout_market", "buildout_uap"]

"""
Many-objective search over multi-use plans with NSGA-III (pymoo).

Decision variables: for every lot and use, the fraction of the lot's capacity
for that use that is built, x[i, u] in [0, 1]; the plan is
``ZoningEnvelope.clip(x * use_caps)`` (so the total and shared commercial caps
always hold).

Objectives (all minimised), two per pillar plus sunlight and displacement:

1. - homes added                       (social)
2. - affordable homes added            (social)
3. - per-capita 15-minute access index (social)
4. - jobs added                        (economic)
5. - jobs-housing balance              (economic)
6.   life-cycle carbon, 30 years       (environmental)
7.   lots newly shaded                 (environmental)
8.   floor area added on vulnerable lots (equity)

Constraint: traffic violations + sewer catchments over capacity <= 0.

``seed_plans`` lets half of the initial population start from given plans
(e.g. the multi-agent plan and repaired build-outs) with Gaussian
perturbations; the rest is uniform random. Normalised hypervolume per
generation is recorded so cold and seeded starts can be compared.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from pimaluos.outcomes import OutcomeModel

try:
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.callback import Callback
    from pymoo.core.problem import Problem
    from pymoo.indicators.hv import HV
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    PYMOO = True
except ImportError:  # pragma: no cover
    PYMOO = False
    Problem = object
    Callback = object

OBJECTIVES = ["neg_homes", "neg_affordable_homes", "neg_access_index", "neg_jobs", "neg_jobs_housing_balance",
              "lifecycle_carbon_kt", "lots_newly_shaded", "vulnerable_lot_added_floor_area_sqft"]
SIGN = np.array([-1, -1, -1, -1, -1, 1, 1, 1], dtype=float)
KEYS = ["homes_added", "affordable_homes_added", "access_index", "jobs_added", "jobs_housing_balance",
        "lifecycle_carbon_kt", "lots_newly_shaded", "vulnerable_lot_added_floor_area_sqft"]


def plan_objectives(om: OutcomeModel, plan: np.ndarray):
    s = om.evaluate(plan)["summary"]
    f = SIGN * np.array([s[k] for k in KEYS], dtype=float)
    g = float(s["traffic_violations"] + s["catchments_over_capacity"])
    return f, g


class PlanProblem(Problem):
    def __init__(self, om: OutcomeModel):
        self.om = om
        self.caps = om.env.use_caps()
        super().__init__(n_var=om.n * 4, n_obj=len(KEYS), n_ieq_constr=1, xl=0.0, xu=1.0)

    def to_plan(self, x: np.ndarray) -> np.ndarray:
        return self.om.env.clip(np.clip(x, 0, 1).reshape(self.om.n, 4) * self.caps)

    def to_x(self, plan: np.ndarray) -> np.ndarray:
        return np.where(self.caps > 0, plan / np.maximum(self.caps, 1e-9), 0.0).clip(0, 1).ravel()

    def _evaluate(self, X, out, *args, **kwargs):
        res = [plan_objectives(self.om, self.to_plan(x)) for x in X]
        out["F"] = np.array([r[0] for r in res])
        out["G"] = np.array([[r[1]] for r in res])


class _HVRecorder(Callback):
    def __init__(self, ref_point: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.hv = HV(ref_point=np.ones(len(ref_point)))
        self.ref, self.scale = ref_point, scale
        self.values: List[float] = []

    def notify(self, algorithm):
        opt = algorithm.opt
        F = opt.get("F")
        feas = opt.get("feasible").ravel() if opt.get("feasible") is not None else np.ones(len(F), bool)
        Fn = (F[feas] - self.ref) / self.scale + 1.0
        Fn = Fn[(Fn <= 1.0).all(axis=1)]
        self.values.append(float(self.hv(Fn)) if len(Fn) else 0.0)


def normalisation(om: OutcomeModel, plans: List[np.ndarray]):
    """Reference (worst) point and scale for hypervolume, from the status quo and
    the given reference plans (e.g. the build-outs)."""
    Fs = np.array([plan_objectives(om, p)[0] for p in [np.zeros((om.n, 4))] + list(plans)])
    worst, best = Fs.max(0), Fs.min(0)
    scale = np.maximum(worst - best, 1e-9)
    return worst + 0.1 * scale, scale * 1.1


def run_nsga3(
    om: OutcomeModel,
    pop_size: int = 120,
    generations: int = 100,
    n_partitions: int = 2,
    seed: int = 0,
    seed_plans: Optional[List[np.ndarray]] = None,
    reference_plans: Optional[List[np.ndarray]] = None,
    seed_noise: float = 0.05,
) -> Dict:
    if not PYMOO:
        raise ImportError("pip install pymoo")
    prob = PlanProblem(om)
    rng = np.random.default_rng(seed)
    X0 = rng.random((pop_size, prob.n_var))
    if seed_plans:
        seeds = [prob.to_x(p) for p in seed_plans]
        m = len(seeds)
        for k in range(pop_size // 2):
            xb = seeds[k % m]
            X0[k] = xb if k < m else np.clip(xb + rng.normal(0, seed_noise, prob.n_var), 0, 1)
    ref_dirs = get_reference_directions("das-dennis", len(KEYS), n_partitions=n_partitions)
    algo = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, sampling=X0, eliminate_duplicates=True)
    ref, scale = normalisation(om, reference_plans or [])
    cb = _HVRecorder(ref, scale)
    res = minimize(prob, algo, ("n_gen", generations), seed=seed, callback=cb, verbose=False)
    feasible = res.X is not None
    if feasible:
        F, X = np.atleast_2d(res.F), np.atleast_2d(res.X)
    else:  # no feasible plan found: report the least-violating population members
        cv = res.pop.get("CV").ravel()
        keep = cv <= cv.min() + 1e-9
        F, X = res.pop.get("F")[keep], res.pop.get("X")[keep]
    plans = [prob.to_plan(x) for x in X]
    Fn = (F - F.min(0)) / np.maximum(F.max(0) - F.min(0), 1e-12)
    knee = int(np.argmin(np.linalg.norm(Fn, axis=1)))
    return {"F": F, "plans": plans, "knee": knee, "hv_history": cb.values, "objectives": OBJECTIVES,
            "n_ref_dirs": int(len(ref_dirs)), "feasible": feasible}

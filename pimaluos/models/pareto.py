"""
Many-objective search over FAR plans with NSGA-III (pymoo).

Decision variables: the floor-area increment of each lot as a fraction of its
headroom, x_i in [0, 1], FAR_i = far0_i + x_i (ub_i - far0_i).

Objectives (all minimised):

1. -added floor area (sq ft)
2. capacity exceedance = sum over cells of max(0, V/C - max(threshold, V/C_0))
   + sum over catchments of max(0, utilisation - 1)
3. lots newly shaded at winter-solstice noon
4. added floor area on displacement-vulnerable residential lots (sq ft)

``seed_plans`` lets the initial population include given plans (e.g. the MARL
consensus plan, status quo, zoning build-out) with Gaussian perturbations; the
remaining individuals are uniform random. Hypervolume per generation is
recorded so cold-start and seeded runs can be compared.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from pimaluos.physics.capacity import CapacityModel

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

OBJECTIVES = ["neg_added_floor_area", "capacity_exceedance", "lots_newly_shaded", "vulnerable_added_floor_area"]


def plan_objectives(cap: CapacityModel, far: np.ndarray) -> np.ndarray:
    r = cap.evaluate(far)
    p = cap.p
    vc0 = cap.baseline["vc_taz"]
    traffic = np.maximum(0.0, r["vc_taz"] - np.maximum(p.vc_threshold, vc0)).sum()
    sewer = np.maximum(0.0, r["sewer_util"] - 1.0).sum()
    added = np.maximum(r["delta_floor_area"], 0)
    return np.array([-added.sum(), traffic + sewer, float(r["new_shaded"].sum()),
                     float(added[cap.vulnerable].sum())])


class FARProblem(Problem):
    def __init__(self, cap: CapacityModel):
        self.cap = cap
        self.head = cap.ub - cap.far0
        super().__init__(n_var=cap.n, n_obj=4, xl=0.0, xu=1.0)

    def to_far(self, x: np.ndarray) -> np.ndarray:
        return self.cap.far0 + np.clip(x, 0, 1) * self.head

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([plan_objectives(self.cap, self.to_far(x)) for x in X])


class _HVRecorder(Callback):
    def __init__(self, ref_point: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.hv = HV(ref_point=np.ones(len(ref_point)))
        self.ref, self.scale = ref_point, scale
        self.values: List[float] = []

    def notify(self, algorithm):
        F = algorithm.opt.get("F")
        Fn = (F - self.ref) / self.scale + 1.0  # map to [0, ~1] relative to reference
        Fn = Fn[(Fn <= 1.0).all(axis=1)]
        self.values.append(float(self.hv(Fn)) if len(Fn) else 0.0)


def normalisation(cap: CapacityModel):
    """Reference (worst) and scale for hypervolume, from status quo and build-out."""
    sq = plan_objectives(cap, cap.far0)
    bo = plan_objectives(cap, cap.ub)
    worst = np.maximum(sq, bo)
    best = np.minimum(sq, bo)
    scale = np.maximum(worst - best, 1e-9)
    return worst + 0.1 * scale, scale * 1.1


def run_nsga3(
    cap: CapacityModel,
    pop_size: int = 120,
    generations: int = 100,
    n_partitions: int = 7,
    seed: int = 0,
    seed_plans: Optional[List[np.ndarray]] = None,
    seed_noise: float = 0.05,
) -> Dict:
    if not PYMOO:
        raise ImportError("pip install pymoo")
    prob = FARProblem(cap)
    rng = np.random.default_rng(seed)
    X0 = rng.random((pop_size, cap.n))
    if seed_plans:
        head = np.maximum(prob.head, 1e-12)
        seeds = [np.where(prob.head > 0, (p - cap.far0) / head, 0.0) for p in seed_plans]
        m = len(seeds)
        for k in range(pop_size // 2):
            xb = seeds[k % m]
            X0[k] = xb if k < m else np.clip(xb + rng.normal(0, seed_noise, cap.n), 0, 1)
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=n_partitions)
    algo = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, sampling=X0, eliminate_duplicates=True)
    ref, scale = normalisation(cap)
    cb = _HVRecorder(ref, scale)
    res = minimize(prob, algo, ("n_gen", generations), seed=seed, callback=cb, verbose=False)
    F = np.atleast_2d(res.F)
    X = np.atleast_2d(res.X)
    fars = np.array([prob.to_far(x) for x in X])
    Fn = (F - F.min(0)) / np.maximum(F.max(0) - F.min(0), 1e-12)
    knee = int(np.argmin(np.linalg.norm(Fn, axis=1)))
    return {"F": F, "far": fars, "knee": knee, "hv_history": cb.values,
            "objectives": OBJECTIVES, "n_ref_dirs": int(len(ref_dirs))}

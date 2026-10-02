"""
Baseline plans, all expressed as FAR vectors on the same bounds as PIMALUOS
(existing FAR <= FAR <= max(zoning max FAR, existing FAR)) and evaluated with the
same capacity screens.

* ``status_quo``        existing FAR.
* ``random_plan``       ``horizon`` uniformly random {-delta, 0, +delta} steps.
* ``rule_based_plan``   each step, +delta on lots below 70 % of their maximum FAR.
* ``zoning_buildout``   every lot at its maximum FAR (single objective: floor area).
* ``verified(plan)``    any plan passed through the verification/repair loop;
                        ``verified(zoning_buildout)`` is a capacity-aware greedy.
"""

from __future__ import annotations

import numpy as np

from pimaluos.physics.capacity import CapacityModel
from pimaluos.physics.verification import verify_and_repair


def status_quo(cap: CapacityModel) -> np.ndarray:
    return cap.far0.copy()


def random_plan(cap: CapacityModel, horizon: int = 10, delta: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    far = cap.far0.copy()
    for _ in range(horizon):
        far = np.clip(far + (rng.integers(0, 3, cap.n) - 1) * delta, cap.far0, cap.ub)
    return far


def rule_based_plan(cap: CapacityModel, horizon: int = 10, delta: float = 0.5,
                    threshold: float = 0.7) -> np.ndarray:
    far = cap.far0.copy()
    for _ in range(horizon):
        under = far < threshold * cap.ub
        far = np.where(under, np.minimum(far + delta, cap.ub), far)
    return far


def zoning_buildout(cap: CapacityModel) -> np.ndarray:
    return cap.ub.copy()


def verified(cap: CapacityModel, far: np.ndarray, **kw) -> np.ndarray:
    return verify_and_repair(cap, far, **kw)[0]


__all__ = ["status_quo", "random_plan", "rule_based_plan", "zoning_buildout", "verified"]

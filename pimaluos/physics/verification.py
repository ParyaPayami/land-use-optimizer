"""
Verification loop: evaluate a FAR plan with the capacity screens and repair it.

A plan *violates* a screen when it makes things worse than existing conditions:

* traffic: a cell is above the V/C threshold and its V/C increased;
* sewer: a catchment's load exceeds existing load plus headroom;
* solar: the plan newly shades a lot at winter-solstice noon.

Repair step: every lot whose floor-area increase contributes to a violation has
its increase multiplied by ``shrink``. Iterate until no violation remains or
``max_iter`` is reached. Decreases are never amplified, so the loop only removes
added floor area and terminates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from pimaluos.physics.capacity import CapacityModel

if TYPE_CHECKING:  # pragma: no cover
    from pimaluos.outcomes import OutcomeModel


def violation_counts(result: Dict) -> Dict[str, int]:
    s = result["summary"]
    return {
        "traffic": s["traffic_violations"],
        "sewer": s["catchments_over_capacity"],
        "solar": s["lots_newly_shaded"],
    }


def contributing_lots(model: CapacityModel, result: Dict) -> np.ndarray:
    """Lots with a positive floor-area increase that contribute to any violation."""
    inc = result["delta_floor_area"] > 0
    bad_traffic = result["taz_violation"][model.taz]
    bad_sewer = result["catch_over"][model.catch]
    bad_solar = result["shadow_imposed"] > 0
    return inc & (bad_traffic | bad_sewer | bad_solar)


def verify_and_repair(
    model: CapacityModel, far: np.ndarray, max_iter: int = 30, shrink: float = 0.5
) -> Tuple[np.ndarray, List[Dict]]:
    far = np.clip(np.asarray(far, dtype=float), 0.0, model.ub)
    history = []
    it = 0
    while True:
        res = model.evaluate(far)
        counts = violation_counts(res)
        history.append({"iteration": it, **counts,
                        "added_floor_area_sqft": res["summary"]["added_floor_area_sqft"]})
        if sum(counts.values()) == 0:
            break
        bad = contributing_lots(model, res)
        if not bad.any():
            break  # remaining violations are not caused by added floor area
        # Geometric shrinking first; after max_iter, revert offenders to existing FAR.
        factor = shrink if it < max_iter else 0.0
        far = np.where(bad, model.far0 + (far - model.far0) * factor, far)
        it += 1
    return far, history


def repair_plan(om: "OutcomeModel", plan: np.ndarray, max_iter: int = 30,
                shrink: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
    """Repair a multi-use plan: every lot whose added floor area contributes to a
    traffic, sewer or new-shade violation has all its additions multiplied by
    ``shrink``; after ``max_iter`` rounds offending lots return to existing
    conditions. Terminates because additions only shrink and the existing city
    has no violations."""
    plan = om.env.clip(np.asarray(plan, float))
    history = []
    it = 0
    while True:
        res = om.evaluate(plan)
        counts = violation_counts(res["capacity"])
        history.append({"iteration": it, **counts, "added_floor_area_sqft": res["summary"]["added_floor_area_sqft"]})
        if sum(counts.values()) == 0:
            break
        bad = contributing_lots(om.cap, res["capacity"])
        if not bad.any():
            break
        factor = shrink if it < max_iter else 0.0
        plan = np.where(bad[:, None], plan * factor, plan)
        it += 1
    return plan, history

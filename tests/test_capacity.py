import numpy as np

from pimaluos.physics import gini, verify_and_repair


def test_baseline_has_no_violations(cap):
    s = cap.baseline["summary"]
    assert s["added_floor_area_sqft"] == 0
    assert s["traffic_violations"] == 0 and s["catchments_over_capacity"] == 0 and s["lots_newly_shaded"] == 0
    assert s["zoning_violations"] == 0


def test_upper_bound_keeps_existing_bulk(cap):
    assert np.all(cap.ub >= cap.far0) and np.all(cap.ub >= cap.max_far)


def test_buildout_adds_area(cap):
    assert cap.evaluate(cap.ub)["summary"]["added_floor_area_sqft"] > 0


def test_envelope_monotone_in_far(cap):
    _, _, h1 = cap.envelope(cap.far0)
    _, _, h2 = cap.envelope(cap.ub)
    assert np.all(h2 >= h1 - 1e-9)


def test_verification_terminates_clean_and_within_bounds(cap):
    far, hist = verify_and_repair(cap, cap.ub, max_iter=10)
    last = hist[-1]
    assert last["traffic"] == 0 and last["sewer"] == 0 and last["solar"] == 0
    assert np.all(far >= cap.far0 - 1e-9) and np.all(far <= cap.ub + 1e-9)


def test_gini():
    assert gini(np.ones(5)) == 0.0
    assert np.isclose(gini(np.array([0, 0, 0, 1.0])), 0.75)
    assert np.isclose(gini(np.array([1.0, 3.0]), np.array([3.0, 1.0])), gini(np.array([1, 1, 1, 3.0])))

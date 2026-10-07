import numpy as np

from pimaluos import baselines as B
from pimaluos.outcomes import FACILITY, OFFICE, RES, RETAIL
from pimaluos.physics.verification import repair_plan
from pimaluos.uncertainty import monte_carlo


def test_status_quo_adds_nothing(om):
    s = om.evaluate(np.zeros((om.n, 4)))["summary"]
    for k in ["homes_added", "jobs_added", "lifecycle_carbon_kt", "tax_revenue_musd", "added_floor_area_sqft"]:
        assert s[k] == 0
    assert s["capacity_violations"] == 0


def test_envelope_clip_respects_caps(om):
    big = np.full((om.n, 4), 1e9)
    p = om.env.clip(big)
    e = om.env
    assert np.all(p[:, RES] <= e.res_total + 1e-6)
    assert np.all(p[:, OFFICE] + p[:, RETAIL] <= e.com + 1e-6)
    assert np.all(p[:, FACILITY] <= e.facility + 1e-6)
    assert np.all(p.sum(1) <= e.total + 1e-6)


def test_affordable_floor_area_uap_and_mih(om):
    e = om.env
    res = e.res_total.copy()
    aff = om.affordable(res)
    uap = res - e.res_market
    assert np.all(aff >= uap - 1e-6)
    m = e.mih & (e.res_market > 0)
    if m.any():
        assert np.allclose(aff[m], uap[m] + om.p.mih_affordable_share * e.res_market[m])
    assert np.allclose(om.affordable(np.minimum(res, e.res_market))[~e.mih], 0)


def test_outcomes_linear_in_floor_area(om):
    p = np.zeros((om.n, 4))
    i = int(np.argmax(om.env.com))
    p[i, OFFICE] = 1000.0
    s1 = om.evaluate(p)["summary"]
    p[i, OFFICE] = 2000.0
    s2 = om.evaluate(p)["summary"]
    assert np.isclose(s2["jobs_added"], 2 * s1["jobs_added"])
    assert np.isclose(s2["embodied_carbon_kt"], 2 * s1["embodied_carbon_kt"])


def test_residents_without_services_lower_access_and_services_raise_it(om):
    base = om.base["summary"]["access_index"]
    p = np.zeros((om.n, 4))
    p[:, RES] = om.env.res_total
    assert om.evaluate(om.env.clip(p))["summary"]["access_index"] < base
    q = np.zeros((om.n, 4))
    q[:, FACILITY] = om.env.facility
    assert om.evaluate(om.env.clip(q))["summary"]["access_index"] > base


def test_repair_clears_violations(om):
    for plan in (B.buildout_market(om), B.buildout_uap(om), B.random_plan(om, seed=1)):
        fixed, hist = repair_plan(om, plan)
        assert om.evaluate(fixed)["summary"]["capacity_violations"] == 0
        assert np.all(fixed <= plan + 1e-6)


def test_monte_carlo_shares(om):
    out = monte_carlo(om, {"a": B.rule_based_plan(om), "b": B.buildout_uap(om)}, n_draws=4, seed=0)
    for v in out["share_better"]["a|b"].values():
        assert 0.0 <= v <= 1.0
    s = out["summary"]["a"]["lifecycle_carbon_kt"]
    assert s["q05"] <= s["median"] <= s["q95"]

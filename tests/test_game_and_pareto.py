import numpy as np

from pimaluos.models.agents import AGENT_TYPES, ConsensusVotingMechanism, MultiAgentEnvironment
from pimaluos.models.nash import analyse_consensus, analyse_lot
from pimaluos.models.pareto import plan_objectives, run_nsga3


def test_unanimous_preference_is_optimal_equilibrium():
    pay = np.zeros((5, 5))
    pay[2] = 1.0
    r = analyse_lot(pay, AGENT_TYPES, ConsensusVotingMechanism())
    assert r["optimum"] == 2 and r["sincere"] == 2 and r["sincere_is_optimal"]
    assert r["welfare_loss_sincere"] == 0


def test_analyse_consensus_runs(om, ds):
    env = MultiAgentEnvironment(om, ds.features.values)
    out = analyse_consensus(env, np.zeros((om.n, 4)), n_lots=5, seed=0)
    assert out["summary"]["n_lots"] == 5


def test_objectives_status_quo(om):
    f, g = plan_objectives(om, np.zeros((om.n, 4)))
    assert np.allclose(f[[0, 1, 3, 5, 6, 7]], 0) and g == 0
    assert np.isclose(-f[2], om.base["summary"]["access_index"])


def test_nsga3_small(om):
    r = run_nsga3(om, pop_size=40, generations=3, n_partitions=2, seed=0, seed_plans=[np.zeros((om.n, 4))])
    assert r["F"].shape[1] == 8 and len(r["hv_history"]) == 3
    for p in r["plans"]:
        assert np.allclose(om.env.clip(p), p)

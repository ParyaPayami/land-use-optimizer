import numpy as np

from pimaluos.models.agents import AGENT_TYPES, ConsensusVotingMechanism, MultiAgentEnvironment
from pimaluos.models.nash import analyse_consensus, analyse_lot
from pimaluos.models.pareto import plan_objectives, run_nsga3


def test_unanimous_preference_is_optimal_equilibrium():
    pay = np.zeros((3, 5))
    pay[2] = 1.0
    r = analyse_lot(pay, AGENT_TYPES, ConsensusVotingMechanism())
    assert r["optimum"] == 2 and r["sincere"] == 2 and r["sincere_is_optimal"]
    assert r["welfare_loss_sincere"] == 0


def test_analyse_consensus_runs(cap, ds):
    env = MultiAgentEnvironment(cap, ds.features.values)
    out = analyse_consensus(env, cap.far0, n_lots=5, seed=0)
    assert out["summary"]["n_lots"] == 5


def test_objectives_status_quo_zero(cap):
    assert np.allclose(plan_objectives(cap, cap.far0), 0)


def test_nsga3_small(cap):
    r = run_nsga3(cap, pop_size=16, generations=3, n_partitions=3, seed=0, seed_plans=[cap.far0])
    assert r["F"].shape[1] == 4 and len(r["hv_history"]) == 3
    assert np.all(r["far"] >= cap.far0 - 1e-9) and np.all(r["far"] <= cap.ub + 1e-9)

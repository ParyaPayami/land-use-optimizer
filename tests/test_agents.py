import numpy as np

from pimaluos.models.agents import ConsensusVotingMechanism, MARLTrainer, MultiAgentEnvironment, PPOConfig


def test_vote_weighted_plurality_and_status_quo_ties():
    v = ConsensusVotingMechanism({"a": 0.6, "b": 0.25, "c": 0.15})
    assert v.aggregate({"a": np.array([2]), "b": np.array([0]), "c": np.array([0])})[0] == 2
    v = ConsensusVotingMechanism({"a": 0.5, "b": 0.5})
    assert v.aggregate({"a": np.array([0, 2]), "b": np.array([2, 1])}).tolist() == [1, 1]


def test_env_bounds_and_reward_is_utility_increment(cap, ds):
    env = MultiAgentEnvironment(cap, ds.features.values, horizon=3)
    s = env.reset()
    assert s.shape == (cap.n, ds.features.shape[1] + 5)
    u0 = env.utilities
    _, r, _, _ = env.step({a: np.full(cap.n, 2) for a in env.agent_types})
    assert np.all(env.far >= cap.far0 - 1e-9) and np.all(env.far <= cap.ub + 1e-9)
    for a in env.agent_types:
        assert np.allclose(r[a], env.utilities[a] - u0[a], atol=1e-5)
    for _ in range(5):
        env.step({a: np.zeros(cap.n, int) for a in env.agent_types})
    assert np.allclose(env.far, cap.far0)  # decreases never go below existing FAR


def test_physics_weight_zero_removes_capacity_penalty(cap, ds):
    u0 = MultiAgentEnvironment(cap, ds.features.values, physics_weight=0.0)._utilities(cap.evaluate(cap.ub))
    u1 = MultiAgentEnvironment(cap, ds.features.values, physics_weight=1.0)._utilities(cap.evaluate(cap.ub))
    assert np.all(u0["developer"] >= u1["developer"] - 1e-9)


def test_trainer_runs_and_plan_is_feasible(cap, ds):
    env = MultiAgentEnvironment(cap, ds.features.values, horizon=3)
    tr = MARLTrainer(env, PPOConfig(minibatch=256), seed=0)
    hist = tr.train(2)
    assert len(hist) == 2 and "episode_return" in hist[0]["planner"]
    far, props = tr.final_plan()
    assert len(props) == 3
    assert cap.evaluate(far)["summary"]["zoning_violations"] == 0

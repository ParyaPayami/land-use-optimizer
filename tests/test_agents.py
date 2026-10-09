import numpy as np

from pimaluos import baselines as B
from pimaluos.models.agents import ACTION_KEEP, ConsensusVotingMechanism, MARLTrainer, MultiAgentEnvironment, PPOConfig


def test_vote_weighted_plurality_and_status_quo_ties():
    v = ConsensusVotingMechanism({"a": 0.6, "b": 0.25, "c": 0.15})
    assert v.aggregate({"a": np.array([2]), "b": np.array([1]), "c": np.array([1])})[0] == 2
    v = ConsensusVotingMechanism({"a": 0.5, "b": 0.5})
    assert v.aggregate({"a": np.array([3, 2]), "b": np.array([4, 2])}).tolist() == [ACTION_KEEP, 2]


def test_env_respects_envelope_and_reward_is_utility_increment(om, ds):
    env = MultiAgentEnvironment(om, ds.features.values, horizon=3)
    s = env.reset()
    assert s.shape == (om.n, env.state_dim)
    u0 = env.utilities
    for a_ in range(1, 5):
        _, r, _, _ = env.step({a: np.full(om.n, a_) for a in env.agent_types})
    assert np.allclose(om.env.clip(env.plan), env.plan)
    assert np.all(env.plan >= 0)
    plan = env.plan.copy()
    u1 = env.utilities
    _, r, _, _ = env.step({a: np.full(om.n, ACTION_KEEP) for a in env.agent_types})
    assert np.allclose(env.plan, plan)
    for a in env.agent_types:
        assert np.allclose(r[a], env.utilities[a] - u1[a], atol=1e-5)
    assert any(not np.allclose(env.utilities[a], u0[a]) for a in env.agent_types)


def test_physics_weight_zero_removes_capacity_penalty(om, ds):
    r = om.evaluate(B.buildout_uap(om))
    u0 = MultiAgentEnvironment(om, ds.features.values, physics_weight=0.0, awareness=0.0)._utilities(r)
    u1 = MultiAgentEnvironment(om, ds.features.values, physics_weight=1.0, awareness=0.0)._utilities(r)
    assert np.all(u0["developer"] >= u1["developer"] - 1e-9)


def test_awareness_mixes_with_borough_mean(om, ds):
    r = om.evaluate(B.rule_based_plan(om))
    u_self = MultiAgentEnvironment(om, ds.features.values, awareness=0.0)._utilities(r)
    u_city = MultiAgentEnvironment(om, ds.features.values, awareness=1.0)._utilities(r)
    for a in u_self:
        assert np.allclose(u_city[a], u_self[a].mean())


def test_trainer_runs_and_plan_is_feasible(om, ds):
    env = MultiAgentEnvironment(om, ds.features.values, horizon=3)
    tr = MARLTrainer(env, PPOConfig(minibatch=256), seed=0)
    hist = tr.train(2)
    assert len(hist) == 2 and "episode_return" in hist[0]["planner"]
    plan, props = tr.final_plan()
    assert len(props) == 3 and plan.shape == (om.n, 4)
    assert np.allclose(om.env.clip(plan), plan)
    assert om.evaluate(plan)["capacity"]["summary"]["zoning_violations"] == 0


def test_borda_counts_lower_preferences_and_ties_keep():
    # Plurality: two first choices for action 1 beat one for action 2.
    # Borda: action 2 is everyone's first or second choice and wins on points.
    ballots = {"a": np.array([[1, 2, 0, 3, 4]]), "b": np.array([[1, 2, 0, 3, 4]]),
               "c": np.array([[2, 0, 3, 4, 1]])}
    w = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    assert ConsensusVotingMechanism(w).aggregate({k: v[:, 0] for k, v in ballots.items()})[0] == 1
    ballots["c"] = np.array([[2, 0, 3, 4, 1]])
    v = ConsensusVotingMechanism(w, rule="borda")
    # points: action 1 = 4+4+0 = 8, action 2 = 3+3+4 = 10
    assert v.aggregate(ballots)[0] == 2
    # Exactly opposed rankings tie on every action: status quo.
    tie = {"a": np.array([[1, 2, 3, 4, 0]]), "b": np.array([[0, 4, 3, 2, 1]])}
    assert ConsensusVotingMechanism({"a": 0.5, "b": 0.5}, rule="borda").aggregate(tie)[0] == ACTION_KEEP


def test_borda_ballot_ranks_taken_action_first_and_trainer_runs(om, ds):
    import torch

    env = MultiAgentEnvironment(om, ds.features.values, horizon=2, voting_rule="borda")
    tr = MARLTrainer(env, PPOConfig(minibatch=256), seed=0)
    m = tr.agents["planner"]
    st = torch.from_numpy(env.reset())
    act = m.act(st)[0]
    r = m.ranking(st, act)
    assert r.shape == (om.n, 5) and np.array_equal(r[:, 0], act.numpy())
    assert np.all(np.sort(r, axis=1) == np.arange(5))
    hist = tr.train(1)
    assert len(hist) == 1
    plan, props = tr.final_plan()
    assert props[0]["planner"].shape == (om.n, 5)
    assert np.allclose(om.env.clip(plan), plan)

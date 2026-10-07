"""
Game-theoretic analysis of the consensus vote.

For a sample of lots, each lot defines a voting game around the final
consensus plan: every stakeholder casts one of the actions {keep, residential,
office, retail, facility} for that lot, the weighted-plurality rule picks the
outcome, and agent a's payoff for outcome o is its utility for that lot when
action o is applied to that lot (all other lots fixed at the plan), evaluated
exactly with the outcome model. Utilities are relative to "keep".

Reported per lot:

* sincere outcome: each agent votes for its own best outcome;
* utilitarian optimum: the outcome maximising the sum of payoffs (welfare);
* the set of pure Nash equilibria among profiles in which no agent votes for
  its strictly worst outcome (removing weakly dominated votes, which otherwise
  makes every non-pivotal profile an equilibrium);
* welfare loss of the worst such equilibrium and of the sincere outcome
  relative to the optimum.

Because welfare can be negative, the ratio form of the price of anarchy is
reported on welfare shifted so that the worst outcome has welfare 0:
PoA = (W_opt - W_min) / (W_worstNE - W_min) >= 1 (undefined, reported as NaN,
when every outcome has equal welfare or the worst equilibrium is the worst
outcome).
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional

import numpy as np

from pimaluos.models.agents import ConsensusVotingMechanism, MultiAgentEnvironment


def outcome_payoffs(env: MultiAgentEnvironment, plan: np.ndarray, lots: np.ndarray) -> np.ndarray:
    """Return payoffs[lot, outcome, agent] (relative to 'keep')."""
    agents = env.agent_types
    n_out = env.voting.n_actions
    pay = np.zeros((len(lots), n_out, len(agents)))
    env.plan = plan.copy()
    base_u = env._utilities(env.om.evaluate(plan))
    for li, i in enumerate(lots):
        for o in range(1, n_out):
            actions = np.zeros(env.n, dtype=int)
            actions[i] = o
            env.plan = plan.copy()
            u = env._utilities(env.om.evaluate(env.apply(actions)))
            for ai, a in enumerate(agents):
                pay[li, o, ai] = u[a][i] - base_u[a][i]
    env.plan = plan.copy()
    return pay


def analyse_lot(pay: np.ndarray, agents: List[str], voting: ConsensusVotingMechanism) -> Dict:
    """pay: [n_outcomes, n_agents]."""
    n_a = len(agents)
    n_out = pay.shape[0]
    welfare = pay.sum(1)
    opt = int(np.argmax(welfare))
    sincere_votes = pay.argmax(0)
    def undominated(a):
        col = pay[:, a]
        strictly_worst = (col <= col.min() + 1e-12) & (col < col.max() - 1e-12)
        return [o for o in range(n_out) if not strictly_worst[o]]

    allowed = [undominated(a) for a in range(n_a)]

    def outcome(profile):
        return int(voting.aggregate({agents[a]: np.array([profile[a]]) for a in range(n_a)})[0])

    sincere = outcome(sincere_votes)
    ne_outcomes = set()
    for prof in product(*allowed):
        o = outcome(prof)
        stable = True
        for a in range(n_a):
            for alt in allowed[a]:
                if alt == prof[a]:
                    continue
                p2 = list(prof)
                p2[a] = alt
                if pay[outcome(p2), a] > pay[o, a] + 1e-12:
                    stable = False
                    break
            if not stable:
                break
        if stable:
            ne_outcomes.add(o)
    w_min = welfare.min()
    worst_ne = min(ne_outcomes, key=lambda o: welfare[o]) if ne_outcomes else None
    poa = np.nan
    if worst_ne is not None and welfare[worst_ne] - w_min > 1e-12:
        poa = (welfare[opt] - w_min) / (welfare[worst_ne] - w_min)
    return {
        "optimum": opt, "sincere": sincere, "n_ne_outcomes": len(ne_outcomes),
        "sincere_is_optimal": sincere == opt,
        "welfare_loss_sincere": float(welfare[opt] - welfare[sincere]),
        "welfare_loss_worst_ne": float(welfare[opt] - welfare[worst_ne]) if worst_ne is not None else np.nan,
        "poa": float(poa),
        "trivial": bool(np.allclose(welfare, welfare[0])),
    }


def analyse_consensus(env: MultiAgentEnvironment, plan: np.ndarray, n_lots: int = 200,
                      seed: int = 0, lots: Optional[np.ndarray] = None) -> Dict:
    rng = np.random.default_rng(seed)
    room = env.om.env.total - plan.sum(1)
    eligible = np.where(room > 1.0)[0]
    if lots is None:
        lots = rng.choice(eligible, size=min(n_lots, len(eligible)), replace=False)
    pay = outcome_payoffs(env, plan, lots)
    rows = [analyse_lot(pay[k], env.agent_types, env.voting) for k in range(len(lots))]
    nontriv = [r for r in rows if not r["trivial"]]
    poas = np.array([r["poa"] for r in nontriv if np.isfinite(r["poa"])])
    def mean_of(key, fn=np.mean):
        return float(fn([r[key] for r in nontriv])) if nontriv else np.nan

    agg = {
        "n_lots": int(len(rows)),
        "n_nontrivial": int(len(nontriv)),
        "share_sincere_optimal": mean_of("sincere_is_optimal"),
        "mean_welfare_loss_sincere": mean_of("welfare_loss_sincere"),
        "mean_welfare_loss_worst_ne": mean_of("welfare_loss_worst_ne", np.nanmean),
        "median_poa": float(np.median(poas)) if len(poas) else np.nan,
        "share_poa_defined": float(len(poas) / len(nontriv)) if nontriv else np.nan,
        "mean_ne_outcomes": mean_of("n_ne_outcomes"),
    }
    return {"summary": agg, "lots": lots.tolist(), "per_lot": rows}

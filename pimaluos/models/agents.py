"""
Stakeholder agents, consensus voting and the multi-agent FAR environment.

Decision
    Each lot's FAR is adjusted by one of three actions, {-delta, 0, +delta}
    (delta = 0.5 by default), starting from the existing built FAR. FAR is
    bounded below by the existing FAR (plans add floor area; "decrease"
    retracts an earlier increase, it never demolishes existing floor area) and
    above by ``ub = max(zoning max FAR, existing FAR)``, so zoning compliance
    holds by construction (a hard constraint, not a learned behaviour).

Agents
    Five stakeholder types (resident, developer, planner, environmentalist,
    equity advocate). Each type has one policy shared over all lots
    (parameter sharing). Each agent proposes an action for every lot; the
    proposals are aggregated by :class:`ConsensusVotingMechanism`.

State (per lot)
    GNN embedding (or standardised raw features in the No-GNN ablation)
    concatenated with five dynamic signals: current FAR / ub, cell V/C,
    catchment sewer utilisation, number of lots newly shaded by this lot, and
    the cumulative FAR change.

Rewards (per lot, per agent): Eqs. (2)-(6) of the manuscript, with these
MapPLUTO-derived proxies (no census data are joined):

    resident   = w1 * HousingSupply - w2 * Congestion + w3 * GreenAccess
    developer  = FAR / ub - gamma * Violations
    planner    = w1 * TaxRevenue + w2 * InfraEfficiency - w3 * PublicCost
    environment= -Impervious + SolarAccess - FloodExposure
    equity     = -DisplacementRisk - GreenGini

    HousingSupply    added residential floor area / lot area (FAR units)
    Congestion       max(0, V/C - threshold) of the lot's traffic cell
    GreenAccess      open space per resident in the 3x3 cell neighbourhood,
                     divided by its city-wide existing median
    Violations       1 if the lot contributes to any capacity violation
    TaxRevenue       added floor area x assessed value per building sq ft,
                     scaled by the city-wide median value per lot sq ft
    InfraEfficiency  1 - Congestion
    PublicCost       max(0, sewer utilisation - 1) of the lot's catchment
    Impervious       change in footprint coverage
    SolarAccess      - (lots newly shaded by this lot)
    FloodExposure    added FAR on lots in the 2007/2015 FEMA flood zones
    DisplacementRisk added FAR on vulnerable residential lots (bottom quartile
                     of assessed value per unit)
    GreenGini        city-wide Gini of green space per resident (shared)

    ``physics_weight`` scales every capacity-derived term (Congestion,
    Violations, PublicCost, SolarAccess); setting it to 0 gives the
    "no capacity feedback" ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from pimaluos.physics.capacity import CapacityModel
from pimaluos.physics.verification import contributing_lots

AGENT_TYPES = ["resident", "developer", "planner", "environmentalist", "equity_advocate"]
ACTION_DECREASE, ACTION_MAINTAIN, ACTION_INCREASE = 0, 1, 2
N_DYNAMIC = 5

DEFAULT_VOTING_WEIGHTS = {
    "resident": 0.20, "developer": 0.20, "planner": 0.20,
    "environmentalist": 0.20, "equity_advocate": 0.20,
}


@dataclass
class UtilityWeights:
    resident: Tuple[float, float, float] = (1.0, 1.0, 0.5)   # housing, congestion, green
    developer_gamma: float = 1.0
    planner: Tuple[float, float, float] = (1.0, 0.5, 1.0)    # tax, efficiency, public cost
    environment: Tuple[float, float, float] = (1.0, 0.2, 1.0)  # impervious, solar, flood
    equity: Tuple[float, float] = (1.0, 1.0)                  # displacement, gini


class ConsensusVotingMechanism:
    """Weighted plurality vote per lot; ties resolve to the status quo (maintain)."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = dict(weights or DEFAULT_VOTING_WEIGHTS)

    def aggregate(self, proposals: Dict[str, np.ndarray]) -> np.ndarray:
        agents = list(proposals)
        n = len(proposals[agents[0]])
        score = np.zeros((n, 3))
        for a in agents:
            score[np.arange(n), np.asarray(proposals[a])] += self.weights.get(a, 0.0)
        best = score.max(1, keepdims=True)
        winners = np.isclose(score, best)
        out = np.argmax(score, axis=1)
        tie = winners.sum(1) > 1
        out[tie & winners[:, ACTION_MAINTAIN]] = ACTION_MAINTAIN
        # Ties not involving "maintain" (increase vs decrease) also resolve to maintain.
        out[tie & ~winners[:, ACTION_MAINTAIN]] = ACTION_MAINTAIN
        return out

    # Backwards-compatible name.
    aggregate_votes = aggregate


class StakeholderAgent(nn.Module):
    """Actor-critic with 2 x 64 tanh MLPs (shared across lots)."""

    def __init__(self, state_dim: int, agent_type: str, hidden: int = 64, action_dim: int = 3):
        super().__init__()
        self.agent_type = agent_type
        self.state_dim = state_dim
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, action_dim))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def dist(self, s: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(s))

    @torch.no_grad()
    def act(self, s: torch.Tensor, deterministic: bool = False):
        d = self.dist(s)
        a = d.probs.argmax(-1) if deterministic else d.sample()
        return a, d.log_prob(a), self.critic(s).squeeze(-1)


class MultiAgentEnvironment:
    """Vectorised multi-lot FAR environment over the capacity screens."""

    def __init__(
        self,
        capacity: CapacityModel,
        static_state: np.ndarray,
        agent_types: Optional[List[str]] = None,
        delta_far: float = 0.5,
        horizon: int = 10,
        physics_weight: float = 1.0,
        voting_weights: Optional[Dict[str, float]] = None,
        utility_weights: Optional[UtilityWeights] = None,
    ):
        self.cap = capacity
        self.static = np.asarray(static_state, dtype=np.float32)
        self.n = capacity.n
        self.agent_types = list(agent_types or AGENT_TYPES)
        self.delta = delta_far
        self.horizon = horizon
        self.physics_weight = physics_weight
        self.voting = ConsensusVotingMechanism(voting_weights)
        self.w = utility_weights or UtilityWeights()
        self.state_dim = self.static.shape[1] + N_DYNAMIC
        base = capacity.baseline
        self.green_ref = float(np.median(base["green_per_capita"][capacity.is_res])) if capacity.is_res.any() else 1.0
        self.green_ref = max(self.green_ref, 1e-6)
        vpls = capacity.value_per_bldg_sqft
        self.value_scale = float(np.median(vpls)) if np.isfinite(vpls).any() else 1.0
        self.reset()

    # ---------------------------------------------------------------- dynamics
    def reset(self) -> np.ndarray:
        self.t = 0
        self.far = self.cap.far0.copy()
        self.result = self.cap.baseline
        self.utilities = self._utilities(self.result)
        return self._state()

    def _state(self) -> np.ndarray:
        r = self.result
        ub = np.maximum(self.cap.ub, 1e-6)
        dyn = np.column_stack([
            self.far / ub,
            r["vc_lot"],
            r["sewer_util_lot"],
            np.minimum(r["shadow_imposed"], 10) / 10.0,
            (self.far - self.cap.far0) / max(self.delta * self.horizon, 1e-6),
        ]).astype(np.float32)
        return np.concatenate([self.static, dyn], axis=1)

    def apply(self, actions: np.ndarray) -> np.ndarray:
        step = (np.asarray(actions) - 1) * self.delta
        return np.clip(self.far + step, self.cap.far0, self.cap.ub)

    def _utilities(self, r: Dict) -> Dict[str, np.ndarray]:
        c, w, pw = self.cap, self.w, self.physics_weight
        dfar = r["far"] - c.far0
        housing = dfar * c.res_share
        congestion = np.maximum(0.0, r["vc_lot"] - c.p.vc_threshold) * r["taz_violation"][c.taz]
        green = r["green_per_capita"] / self.green_ref
        violations = contributing_lots(c, r).astype(float)
        tax = dfar * c.A * c.value_per_bldg_sqft / (c.A * self.value_scale)
        public_cost = np.maximum(0.0, r["sewer_util_lot"] - 1.0)
        impervious = r["coverage"] - c.cov0
        solar = -r["shadow_imposed"]
        flood = np.maximum(dfar, 0) * c.flood
        displacement = np.maximum(dfar, 0) * c.vulnerable
        green_gini = r["summary"]["green_space_gini"]
        r1, r2, r3 = w.resident
        p1, p2, p3 = w.planner
        e1, e2, e3 = w.environment
        q1, q2 = w.equity
        return {
            "resident": r1 * housing - pw * r2 * congestion + r3 * green,
            "developer": r["far"] / np.maximum(c.ub, 1e-6) - pw * w.developer_gamma * violations,
            "planner": p1 * tax + p2 * (1.0 - pw * congestion) - pw * p3 * public_cost,
            "environmentalist": -e1 * impervious + pw * e2 * solar - e3 * flood,
            "equity_advocate": -q1 * displacement - q2 * green_gini * np.ones(c.n),
        }

    def step(self, proposals: Dict[str, np.ndarray]):
        if len(proposals) == 1:
            actions = np.asarray(next(iter(proposals.values())))
        else:
            actions = self.voting.aggregate(proposals)
        self.far = self.apply(actions)
        self.result = self.cap.evaluate(self.far)
        new_u = self._utilities(self.result)
        rewards = {a: (new_u[a] - self.utilities[a]).astype(np.float32) for a in self.agent_types}
        self.utilities = new_u
        self.t += 1
        done = self.t >= self.horizon
        return self._state(), rewards, done, {"actions": actions}


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 4096
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden: int = 64


class MARLTrainer:
    """Independent PPO per stakeholder type, trained on vectorised episodes.

    Rewards are per-step utility *increments*, so the undiscounted return of an
    episode equals the change in the agent's utility from existing conditions.
    The learning rate decays linearly to zero over training.
    """

    def __init__(self, env: MultiAgentEnvironment, cfg: Optional[PPOConfig] = None, seed: int = 0):
        self.env = env
        self.cfg = cfg or PPOConfig()
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.agents = {a: StakeholderAgent(env.state_dim, a, self.cfg.hidden) for a in env.agent_types}
        self.opts = {a: torch.optim.Adam(m.parameters(), lr=self.cfg.lr) for a, m in self.agents.items()}
        self.history: List[Dict] = []

    def _rollout(self):
        env = self.env
        s = env.reset()
        buf = {a: {"s": [], "a": [], "lp": [], "v": [], "r": []} for a in env.agent_types}
        done = False
        while not done:
            st = torch.from_numpy(s)
            props = {}
            for a, m in self.agents.items():
                act, lp, v = m.act(st)
                props[a] = act.numpy()
                buf[a]["s"].append(st)
                buf[a]["a"].append(act)
                buf[a]["lp"].append(lp)
                buf[a]["v"].append(v)
            s, rew, done, _ = env.step(props)
            for a in env.agent_types:
                buf[a]["r"].append(torch.from_numpy(rew[a]))
        return buf

    def _update(self, a: str, b: Dict, frac_remaining: float) -> Dict:
        c = self.cfg
        m, opt = self.agents[a], self.opts[a]
        for g in opt.param_groups:
            g["lr"] = c.lr * frac_remaining
        S = torch.stack(b["s"])           # [T, N, D]
        A = torch.stack(b["a"])
        LP = torch.stack(b["lp"])
        V = torch.stack(b["v"])
        R = torch.stack(b["r"])
        T = R.shape[0]
        adv = torch.zeros_like(R)
        last = torch.zeros_like(R[0])
        for t in reversed(range(T)):
            nv = V[t + 1] if t + 1 < T else torch.zeros_like(V[0])  # episode ends at T
            delta = R[t] + c.gamma * nv - V[t]
            last = delta + c.gamma * c.gae_lambda * last
            adv[t] = last
        ret = adv + V
        S, A, LP, ret, adv = (x.reshape(T * R.shape[1], *x.shape[2:]) for x in (S, A, LP, ret, adv))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = S.shape[0]
        stats = {"actor": 0.0, "critic": 0.0, "entropy": 0.0, "k": 0}
        for _ in range(c.epochs):
            perm = torch.from_numpy(self.rng.permutation(n))
            for i in range(0, n, c.minibatch):
                idx = perm[i:i + c.minibatch]
                d = m.dist(S[idx])
                ratio = torch.exp(d.log_prob(A[idx]) - LP[idx])
                s1 = ratio * adv[idx]
                s2 = torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * adv[idx]
                actor = -torch.min(s1, s2).mean()
                critic = (m.critic(S[idx]).squeeze(-1) - ret[idx]).pow(2).mean()
                ent = d.entropy().mean()
                loss = actor + c.value_coef * critic - c.entropy_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), c.max_grad_norm)
                opt.step()
                stats["actor"] += actor.item()
                stats["critic"] += critic.item()
                stats["entropy"] += ent.item()
                stats["k"] += 1
        k = max(stats.pop("k"), 1)
        out = {key: v / k for key, v in stats.items()}
        out["episode_return"] = float(R.sum(0).mean().item())
        return out

    def train(self, iterations: int, logger=None) -> List[Dict]:
        for it in range(iterations):
            buf = self._rollout()
            frac = 1.0 - it / max(iterations, 1)
            rec = {"iteration": it}
            for a in self.env.agent_types:
                rec[a] = self._update(a, buf[a], frac)
            rec["plan_summary"] = self.env.result["summary"]
            self.history.append(rec)
            if logger and (it % 5 == 0 or it == iterations - 1):
                rets = {a: round(rec[a]["episode_return"], 4) for a in self.env.agent_types}
                logger.info("MARL it %d returns %s added_fa %.0f", it, rets,
                            rec["plan_summary"]["added_floor_area_sqft"])
        return self.history

    @torch.no_grad()
    def final_plan(self, deterministic: bool = True) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
        """Roll out the trained policies greedily for one episode."""
        env = self.env
        s = env.reset()
        done = False
        proposals_log = []
        while not done:
            st = torch.from_numpy(s)
            props = {a: m.act(st, deterministic)[0].numpy() for a, m in self.agents.items()}
            proposals_log.append(props)
            s, _, done, _ = env.step(props)
        return env.far.copy(), proposals_log

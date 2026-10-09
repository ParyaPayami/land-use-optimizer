"""
Stakeholder agents, consensus voting and the multi-agent planning environment.

Decision
    At every step each lot receives one of five actions: keep, or add
    ``delta`` FAR (x lot area) of residential, office, retail or community-
    facility floor area. Additions are projected into the zoning envelope
    (:class:`~pimaluos.outcomes.ZoningEnvelope`): per-use caps from MapPLUTO,
    residential floor area above ResidFAR only as income-restricted floor area
    (City of Yes Universal Affordability Preference), and a total cap. Existing
    floor area is never removed.

Agents
    Five stakeholder types, each with one policy shared over all lots
    (parameter sharing). Every agent proposes an action for every lot; the
    proposals are aggregated by :class:`ConsensusVotingMechanism`, either by
    weighted plurality (each agent's sampled action is its vote) or by a
    weighted Borda count (each agent ranks all actions: its sampled action
    first, the others by its policy's probabilities).

State (per lot)
    GNN embedding (or standardised raw features in the No-GNN ablation) and
    ``N_DYNAMIC`` dynamic signals: used share of the lot's total, residential,
    commercial and facility capacity; cell V/C; catchment sewer utilisation;
    lots newly shaded by this lot; per-capita access, jobs-housing balance and
    land-use mix in the lot's 15-minute walkshed; transit within 15 minutes;
    flood zone; displacement vulnerability; Mandatory Inclusionary Housing area.

Utilities (per lot; rewards are per-step increments). All floor-area terms are
in FAR units (sq ft / lot area); money and carbon are scaled by borough medians
per sq ft so that one FAR of typical floor area is about one unit:

    resident    = a1 log(access) + a2 mix - pw (a3 congestion + a4 newly_shaded)
    developer   = market value - pw * violations
    planner     = tax + jobs + homes + jobs-housing balance - pw * sewer overload
    environment = - life-cycle carbon + transit-oriented floor area - flood-zone
                  floor area - pw * shade imposed on other lots
    equity      = affordable homes - floor area on vulnerable lots
                  + log(access) where vulnerable residents live

``awareness`` (beta) mixes each lot's own utility with the borough mean:
u = (1 - beta) u_lot + beta mean(u), from self-interested (0) to fully
city-aware (1) agents, following the awareness levels of Qian et al. (2023).
``physics_weight`` scales every capacity-derived term; 0 gives the
"no capacity feedback" ablation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from pimaluos.outcomes import FACILITY, OFFICE, RES, RETAIL, OutcomeModel
from pimaluos.physics.verification import contributing_lots

AGENT_TYPES = ["resident", "developer", "planner", "environmentalist", "equity_advocate"]
ACTIONS = ["keep", "residential", "office", "retail", "facility"]
ACTION_KEEP = 0
ACTION_USE = {1: RES, 2: OFFICE, 3: RETAIL, 4: FACILITY}
N_ACTIONS = len(ACTIONS)
N_DYNAMIC = 14

DEFAULT_VOTING_WEIGHTS = {a: 0.20 for a in AGENT_TYPES}


@dataclass
class UtilityWeights:
    resident: Tuple[float, float, float, float] = (5.0, 1.0, 1.0, 0.5)  # access, mix, congestion, shaded
    developer_gamma: float = 1.0
    planner: Tuple[float, float, float, float, float] = (1.0, 0.5, 0.5, 1.0, 1.0)  # tax, jobs, homes, jh, sewer
    environment: Tuple[float, float, float, float] = (1.0, 0.5, 1.0, 0.2)  # carbon, transit, flood, shade
    equity: Tuple[float, float, float] = (2.0, 1.0, 5.0)  # affordable, vulnerable, access


VOTING_RULES = ("plurality", "borda")


class ConsensusVotingMechanism:
    """Weighted vote per lot; ties resolve to the status quo (keep).

    ``plurality``: each proposal is one action per lot (``[n]``), worth the agent's weight.
    ``borda``: each proposal is a ranking of all actions per lot (``[n, n_actions]``, best
    first); the action in position r earns ``weight * (n_actions - 1 - r)`` points.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, n_actions: int = N_ACTIONS,
                 rule: str = "plurality"):
        if rule not in VOTING_RULES:
            raise ValueError(f"unknown voting rule {rule!r}; choose from {VOTING_RULES}")
        self.weights = dict(weights or DEFAULT_VOTING_WEIGHTS)
        self.n_actions = n_actions
        self.rule = rule

    def aggregate(self, proposals: Dict[str, np.ndarray]) -> np.ndarray:
        agents = list(proposals)
        n = len(proposals[agents[0]])
        score = np.zeros((n, self.n_actions))
        rows = np.arange(n)
        for a in agents:
            w = self.weights.get(a, 0.0)
            p = np.asarray(proposals[a])
            if self.rule == "borda":
                points = (self.n_actions - 1 - np.arange(p.shape[1])).astype(float)
                np.add.at(score, (rows[:, None], p), w * points[None, :])
            else:
                score[rows, p] += w
        best = score.max(1, keepdims=True)
        winners = np.isclose(score, best)
        out = np.argmax(score, axis=1)
        out[winners.sum(1) > 1] = ACTION_KEEP  # any tie keeps the status quo
        return out

    aggregate_votes = aggregate


class StakeholderAgent(nn.Module):
    """Actor-critic with 2 x 64 tanh MLPs (shared across lots)."""

    def __init__(self, state_dim: int, agent_type: str, hidden: int = 64, action_dim: int = N_ACTIONS):
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

    @torch.no_grad()
    def ranking(self, s: torch.Tensor, first: torch.Tensor) -> np.ndarray:
        """Ballot for a ranked vote: ``first`` (the action taken) at the top, the other
        actions by decreasing policy probability. Returns ``[n, n_actions]``."""
        p = self.dist(s).probs.clone()
        p[torch.arange(p.shape[0]), first] = 2.0
        return torch.argsort(p, dim=1, descending=True, stable=True).numpy()


class MultiAgentEnvironment:
    """Vectorised multi-lot, multi-use planning environment over the outcome model."""

    def __init__(
        self,
        outcomes: OutcomeModel,
        static_state: np.ndarray,
        agent_types: Optional[List[str]] = None,
        delta_far: float = 0.5,
        horizon: int = 10,
        physics_weight: float = 1.0,
        awareness: float = 0.5,
        voting_weights: Optional[Dict[str, float]] = None,
        utility_weights: Optional[UtilityWeights] = None,
        voting_rule: str = "plurality",
    ):
        self.om = outcomes
        self.cap = outcomes.cap
        self.static = np.asarray(static_state, dtype=np.float32)
        self.n = outcomes.n
        self.agent_types = list(agent_types or AGENT_TYPES)
        self.delta = delta_far
        self.horizon = horizon
        self.physics_weight = physics_weight
        self.awareness = awareness
        self.voting = ConsensusVotingMechanism(voting_weights, rule=voting_rule)
        self.w = utility_weights or UtilityWeights()
        self.state_dim = self.static.shape[1] + N_DYNAMIC
        om, A = outcomes, outcomes.A
        self.A = np.maximum(A, 1.0)
        self.step_sqft = self.delta * self.A
        caps = om.env.use_caps()
        self.caps = np.column_stack([om.env.total, caps[:, RES], caps[:, OFFICE], caps[:, FACILITY]])
        # Scales: one FAR of typical floor area ~ one utility unit.
        self.value_scale = float(np.median(om.av_res)) / om.p.assessment_ratio
        self.tax_scale = float(np.median(om.av_res)) * om.p.tax_rate_class2 / 100.0
        self.carbon_scale = float(om.eci[RES] + om.p.lifecycle_years * om.opc[RES])
        self.jobs_scale = float(om.jobs_per_sqft[OFFICE])
        self.homes_scale = 1.0 / om.unit_sqft
        lot0 = om.base["lot"]
        self.vuln = om.cap.vulnerable.astype(float)
        node_vuln = np.bincount(om.node, weights=self.vuln * om.ctx.pop0, minlength=om.n_nodes)
        self.vuln_area = ((om.R @ node_vuln) > 0)[om.node].astype(float)
        self.flags = np.column_stack([lot0["transit_ok"], om.cap.flood, om.cap.vulnerable,
                                      om.env.mih]).astype(np.float32)
        self.reset()

    # ---------------------------------------------------------------- dynamics
    def reset(self) -> np.ndarray:
        self.t = 0
        self.plan = np.zeros((self.n, 4))
        self.result = self.om.base
        self.utilities = self._utilities(self.result)
        return self._state()

    @property
    def far(self) -> np.ndarray:
        return self.om.far_of(self.plan)

    def _state(self) -> np.ndarray:
        r, c = self.result, self.result["capacity"]
        used = np.column_stack([self.plan.sum(1), self.plan[:, RES], self.plan[:, OFFICE] + self.plan[:, RETAIL],
                                self.plan[:, FACILITY]]) / np.maximum(self.caps, 1.0)
        lot = r["lot"]
        dyn = np.column_stack([
            np.minimum(used, 1.0),
            c["vc_lot"], c["sewer_util_lot"], np.minimum(c["shadow_imposed"], 10) / 10.0,
            np.log(np.maximum(lot["access"], 1e-3)), lot["jh"], lot["mix"],
            self.flags,
        ]).astype(np.float32)
        return np.concatenate([self.static, dyn], axis=1)

    def apply(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions)
        add = np.zeros_like(self.plan)
        for a, u in ACTION_USE.items():
            m = actions == a
            add[m, u] = self.step_sqft[m]
        return self.om.env.clip(self.plan + add)

    def _utilities(self, r: Dict) -> Dict[str, np.ndarray]:
        om, w, pw, A = self.om, self.w, self.physics_weight, self.A
        lot, c = r["lot"], r["capacity"]
        plan = r["plan"]
        log_acc = np.log(np.maximum(lot["access"], 1e-3))
        congestion = np.maximum(0.0, c["vc_lot"] - self.cap.p.vc_threshold) * c["taz_violation"][self.cap.taz]
        violations = contributing_lots(self.cap, c).astype(float)
        sewer = np.maximum(0.0, c["sewer_util_lot"] - 1.0)
        homes = lot["homes"] / (A * self.homes_scale)
        aff = lot["affordable_homes"] / (A * self.homes_scale)
        added = plan.sum(1) / A
        carbon = (lot["embodied"] + om.p.lifecycle_years * lot["operational"]) / (A * self.carbon_scale)
        r1, r2, r3, r4 = w.resident
        p1, p2, p3, p4, p5 = w.planner
        e1, e2, e3, e4 = w.environment
        q1, q2, q3 = w.equity
        u = {
            "resident": r1 * log_acc + r2 * lot["mix"] - pw * (r3 * congestion + r4 * c["new_shaded"]),
            "developer": lot["value"] / (A * self.value_scale) - pw * w.developer_gamma * violations,
            "planner": (p1 * lot["tax"] / (A * self.tax_scale) + p2 * lot["jobs"] / (A * self.jobs_scale)
                        + p3 * homes + p4 * lot["jh"] - pw * p5 * sewer),
            "environmentalist": (-e1 * carbon + e2 * added * lot["transit_ok"] - e3 * added * self.cap.flood
                                 - pw * e4 * c["shadow_imposed"]),
            "equity_advocate": q1 * aff - q2 * added * self.vuln + q3 * log_acc * self.vuln_area,
        }
        b = self.awareness
        return {a: (1 - b) * v + b * v.mean() for a, v in u.items()}

    def step(self, proposals: Dict[str, np.ndarray]):
        if len(proposals) == 1:
            actions = np.asarray(next(iter(proposals.values())))
            if actions.ndim == 2:  # a single ranked ballot: its first choice
                actions = actions[:, 0]
        else:
            actions = self.voting.aggregate(proposals)
        self.plan = self.apply(actions)
        self.result = self.om.evaluate(self.plan)
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
        self.agents = {a: StakeholderAgent(env.state_dim, a, self.cfg.hidden, N_ACTIONS) for a in env.agent_types}
        self.opts = {a: torch.optim.Adam(m.parameters(), lr=self.cfg.lr) for a, m in self.agents.items()}
        self.history: List[Dict] = []
        self.resumed_from = 0

    def _ballot(self, m: StakeholderAgent, st: torch.Tensor, act: torch.Tensor) -> np.ndarray:
        """The agent's vote: its action (plurality) or a full ranking led by it (Borda)."""
        return m.ranking(st, act) if self.env.voting.rule == "borda" else act.numpy()

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
                props[a] = self._ballot(m, st, act)
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

    def train(self, iterations: int, logger=None, resume_path=None, ckpt_every: int = 5) -> List[Dict]:
        """Train for ``iterations`` PPO iterations.

        If ``resume_path`` is given, the full training state (including RNG states)
        is saved there every ``ckpt_every`` iterations and training resumes from it,
        bit-for-bit, when it exists.
        """
        start = 0
        if resume_path is not None and os.path.exists(resume_path):
            ck = torch.load(resume_path, weights_only=False)
            for a in self.agents:
                self.agents[a].load_state_dict(ck["agents"][a])
                self.opts[a].load_state_dict(ck["opts"][a])
            self.history, start = ck["history"], ck["iteration"]
            self.rng.bit_generator.state = ck["np_rng"]
            torch.set_rng_state(ck["torch_rng"])
            if logger:
                logger.info("MARL resumed at iteration %d", start)
        self.resumed_from = start
        for it in range(start, iterations):
            if resume_path is not None and it > start and it % ckpt_every == 0:
                tmp = f"{resume_path}.tmp"
                torch.save({"agents": {a: m.state_dict() for a, m in self.agents.items()},
                            "opts": {a: o.state_dict() for a, o in self.opts.items()},
                            "history": self.history, "iteration": it,
                            "np_rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state()}, tmp)
                os.replace(tmp, resume_path)
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
        """Roll out the trained policies greedily for one episode; returns the plan
        (added sq ft by use, lots x 4) and the proposals at each step."""
        env = self.env
        s = env.reset()
        done = False
        proposals_log = []
        while not done:
            st = torch.from_numpy(s)
            props = {}
            for a, m in self.agents.items():
                act = m.act(st, deterministic)[0]
                props[a] = self._ballot(m, st, act)
            proposals_log.append(props)
            s, _, done, _ = env.step(props)
        return env.plan.copy(), proposals_log

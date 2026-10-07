"""
High-level API tying the layers together (Sense -> Reason -> Verify).

>>> from pimaluos.core import SyntheticCityLoader
>>> from pimaluos.pipeline import UrbanOptSystem
>>> sysm = UrbanOptSystem(SyntheticCityLoader().load())
>>> _ = sysm.build_graph()
>>> _ = sysm.pretrain_gnn(epochs=5, seed=0)
>>> plan = sysm.optimise(iterations=2, seed=0)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

import pandas as pd

from pimaluos.context.build import CATEGORIES, CityContext, synthetic_context
from pimaluos.core.data_loader import ParcelDataset
from pimaluos.core.graph_builder import ALL_EDGE_TYPES, ParcelGraphBuilder
from pimaluos.models.agents import AGENT_TYPES, MARLTrainer, MultiAgentEnvironment, PPOConfig, UtilityWeights
from pimaluos.models.gnn import ParcelGNN, pretrain_gnn
from pimaluos.outcomes import OutcomeModel, OutcomeParams
from pimaluos.physics.capacity import CapacityModel, CapacityParams
from pimaluos.physics.verification import repair_plan

logger = logging.getLogger(__name__)


def standardise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    sd = a.std(0)
    sd[sd < 1e-8] = 1.0
    return (a - a.mean(0)) / sd


CONTEXT_FEATURES = (["pop_density_log", "jobs_density_log"] + [f"walk_min_{c}" for c in CATEGORIES]
                    + ["access_index_log", "jobs_housing_balance", "land_use_mix"])


def context_features(om: OutcomeModel) -> pd.DataFrame:
    """Standardised lot-level context features appended to the GNN node features:
    population and job density, walking minutes to each everyday destination
    category (capped at 60), and per-capita access, jobs-housing balance and
    land-use mix in the lot's 15-minute walkshed."""
    ctx, lot, A = om.ctx, om.base["lot"], np.maximum(om.A, 1.0)
    cols = [np.log1p(ctx.pop0 / A * 1000), np.log1p(ctx.jobs0 / A * 1000)]
    cols += [np.minimum(ctx.minutes0[om.node, k], 60.0) for k in range(len(CATEGORIES))]
    cols += [np.log(np.maximum(lot["access"], 1e-3)), lot["jh"], lot["mix"]]
    return pd.DataFrame(standardise(np.column_stack(cols)), columns=CONTEXT_FEATURES)


class UrbanOptSystem:
    def __init__(self, dataset: ParcelDataset, context: Optional[CityContext] = None,
                 capacity_params: Optional[CapacityParams] = None, outcome_params: Optional[OutcomeParams] = None,
                 graph_kwargs: Optional[Dict] = None, gnn_kwargs: Optional[Dict] = None):
        self.ds = dataset
        self.graph_kwargs = graph_kwargs or {}
        self.gnn_kwargs = gnn_kwargs or {}
        self.capacity = CapacityModel(dataset.gdf, capacity_params)
        if context is None:
            if not dataset.meta.get("synthetic"):
                raise ValueError("A CityContext is required for real cities (pimaluos.context.build).")
            context = synthetic_context(dataset.gdf)
        self.ctx = context
        self.outcomes = OutcomeModel(dataset.gdf, self.capacity, context, outcome_params)
        cf = context_features(self.outcomes)
        if not set(CONTEXT_FEATURES) & set(self.ds.features.columns):
            self.ds.features = pd.concat([self.ds.features.reset_index(drop=True), cf], axis=1)
            self.ds.feature_names = list(self.ds.features.columns)
            self.ds.meta["features_used"] = self.ds.feature_names
            self.ds.meta["n_features_used"] = len(self.ds.feature_names)
        self.graph = None
        self.graph_summary: Optional[Dict] = None
        self.gnn: Optional[ParcelGNN] = None

    # ----------------------------------------------------------------- Sense
    def build_graph(self, edge_types: Optional[List[str]] = None):
        b = ParcelGraphBuilder(self.ds.gdf, self.ds.features, edge_types=edge_types or ALL_EDGE_TYPES,
                               **self.graph_kwargs)
        graph = b.build()
        summary = b.summary()
        if edge_types is None:
            self.graph, self.graph_summary = graph, summary
        return graph, summary

    # ----------------------------------------------------------------- Reason
    def pretrain_gnn(self, epochs: int = 300, seed: int = 0, graph=None, relations=None, **kw) -> Dict:
        graph = graph if graph is not None else self.graph
        torch.manual_seed(seed)
        np.random.seed(seed)
        rel = list(graph.edge_types) if relations is None else relations
        model = ParcelGNN(graph["parcel"].x.shape[1], rel, **self.gnn_kwargs)
        hist = pretrain_gnn(model, graph, epochs=epochs, seed=seed, logger=logger, **kw)
        if graph is self.graph and relations is None:
            self.gnn = model
        hist["relation_weights"] = model.relation_weights()
        return {"model": model, "history": hist}

    def static_state(self, use_gnn: bool = True) -> np.ndarray:
        if use_gnn:
            if self.gnn is None:
                raise ValueError("Pre-train the GNN first (or use use_gnn=False).")
            return standardise(self.gnn.get_embeddings(self.graph).numpy())
        return standardise(self.ds.features.values)

    def make_env(self, use_gnn: bool = True, physics_weight: float = 1.0,
                 agent_types: Optional[List[str]] = None, horizon: int = 10, delta_far: float = 0.5,
                 awareness: float = 0.5, voting_weights: Optional[Dict[str, float]] = None,
                 utility_weights: Optional[UtilityWeights] = None) -> MultiAgentEnvironment:
        return MultiAgentEnvironment(self.outcomes, self.static_state(use_gnn), agent_types or AGENT_TYPES,
                                     delta_far=delta_far, horizon=horizon, physics_weight=physics_weight,
                                     awareness=awareness, voting_weights=voting_weights,
                                     utility_weights=utility_weights)

    def train_marl(self, env: MultiAgentEnvironment, iterations: int = 100, seed: int = 0,
                   ppo: Optional[PPOConfig] = None, resume_path=None) -> MARLTrainer:
        tr = MARLTrainer(env, ppo, seed=seed)
        tr.train(iterations, logger=logger, resume_path=resume_path)
        return tr

    def optimise(self, iterations: int = 100, seed: int = 0, use_gnn: bool = True, **env_kw) -> np.ndarray:
        env = self.make_env(use_gnn=use_gnn, **env_kw)
        tr = self.train_marl(env, iterations, seed)
        return tr.final_plan()[0]

    # ----------------------------------------------------------------- Verify
    def verify(self, plan: np.ndarray, **kw):
        return repair_plan(self.outcomes, plan, **kw)

    def evaluate(self, plan: np.ndarray) -> Dict:
        return self.outcomes.evaluate(plan)["summary"]

"""Interrupted training resumed from a checkpoint must match an uninterrupted run exactly."""
import pytest
import torch

from pimaluos.models.agents import MARLTrainer, MultiAgentEnvironment, PPOConfig
from pimaluos.models.gnn import ParcelGNN, pretrain_gnn


class _Interrupt(Exception):
    pass


class _StopAt:
    """Logger stub that raises once training logs the given step."""

    def __init__(self, token):
        self.token = token

    def info(self, msg, *args):
        if args and args[0] == self.token and "resumed" not in msg:
            raise _Interrupt


def test_gnn_resume_is_exact(graph, tmp_path):
    data, _ = graph

    def run(**kw):
        torch.manual_seed(0)
        m = ParcelGNN(data["parcel"].x.shape[1], list(data.edge_types), hidden_channels=32, embed_dim=16)
        return pretrain_gnn(m, data, epochs=25, seed=0, patience=100, log_every=1, ckpt_every=10, **kw)

    ref = run()
    ck = str(tmp_path / "gnn.pt")
    with pytest.raises(_Interrupt):
        run(resume_path=ck, logger=_StopAt(15))
    res = run(resume_path=ck)
    assert res["val_loss"] == ref["val_loss"] and res["best_val_loss"] == ref["best_val_loss"]


def test_marl_resume_is_exact(cap, ds, tmp_path):
    def trainer():
        env = MultiAgentEnvironment(cap, ds.features.values, horizon=3)
        return MARLTrainer(env, PPOConfig(minibatch=256), seed=0)

    ref = trainer()
    ref.train(12)
    ck = str(tmp_path / "marl.pt")
    with pytest.raises(_Interrupt):
        trainer().train(12, logger=_StopAt(10), resume_path=ck)
    tr = trainer()
    tr.train(12, resume_path=ck)
    key = [(h["iteration"], h["planner"]["episode_return"]) for h in ref.history]
    assert [(h["iteration"], h["planner"]["episode_return"]) for h in tr.history] == key
    assert (tr.final_plan()[0] == ref.final_plan()[0]).all()

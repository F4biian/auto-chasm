"""Metrics regressions: val_ prefix, multi-layer torch eval, token weighting."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import Model, ProbeConfig
from auto_chasm.sweep import _BestPerLayerCallback
from auto_chasm.trainers._metrics import resolve_early_stopping_metric
from auto_chasm.trainers.trainable import _TrainableModel, evaluate_joint_model


class _TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = [nn.Linear(8, 8) for _ in range(3)]
        self.output_proj = nn.Linear(8, 16)

    def __call__(self, x: mx.array, **k: object) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = 8
    num_hidden_layers = 3


def test_met3_val_prefix_stripped_not_substring() -> None:
    """A probe name containing 'val_' resolves correctly (removeprefix, not replace)."""
    metrics = {"loss": 0.5, "retrieval_macro_f1": 0.88, "interval_acc": 0.7}
    assert resolve_early_stopping_metric("val_retrieval_macro_f1", metrics) == 0.88
    assert resolve_early_stopping_metric("val_interval_acc", metrics) == 0.7
    assert resolve_early_stopping_metric("val_loss", metrics) == 0.5
    # sweep._score has the same fix.
    cb = _BestPerLayerCallback.__new__(_BestPerLayerCallback)
    cb.score_metric = "val_retrieval_macro_f1"
    assert cb._score({"L0_retrieval_macro_f1": 0.9, "L0_loss": 0.1}, "L0") == 0.9


def test_met1_multilayer_probe_torch_eval_uses_all_layers() -> None:
    """Torch eval forwards ALL captured layers for a multi-layer concat probe (no crash)."""
    pytest.importorskip("torch")
    from auto_chasm import JointLoss
    from auto_chasm.trainers import default_binary_metrics
    from auto_chasm.trainers._metrics import evaluate_torch_model
    from tests.conftest import DummyTokenizer, _make_torch_tiny_mlp

    base = _make_torch_tiny_mlp(hidden_dim=8, vocab_size=16, num_layers=3)

    class _C:
        hidden_size = 8
        num_hidden_layers = 3

    base.config = _C()
    m = Model(base, DummyTokenizer(), backend_name="torch")
    m.attach_probe(ProbeConfig(name="p", layers=[0, 2], aggregation="concat"))  # multi-layer
    data = [{"tokens": [1, 2, 3, 4], "labels": [0, 0, 1, 0]}]
    # Before the fix this crashed: the concat head (2*hidden) was fed one layer.
    res = evaluate_torch_model(
        m,
        data,
        batch_size=1,
        max_seq_length=8,
        loss_fn=JointLoss(),
        eval_metrics_fn=default_binary_metrics,
    )
    assert "p_accuracy" in res


def test_met2_eval_metrics_are_token_weighted() -> None:
    """Eval metrics are token-weighted across batches, not a per-batch mean."""
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden"))
    tm = _TrainableModel(m.model, m._probes)
    # Two single-sample batches: one has 1 valid token, one has 9 (batch order is
    # shuffled internally, so key the score on the mask, not the call order).
    data = [
        {"tokens": [1, 2], "labels": [-100, 5]},
        {"tokens": list(range(1, 11)), "labels": [-100] + [5] * 9},
    ]

    def stub_metrics(_model, _captured, _targets, mask):  # noqa: ANN001, ANN202
        return {"acc": 1.0 if float(mask.sum()) == 1 else 0.0}  # 1-token batch scores 1.0

    from auto_chasm import JointLoss

    res = evaluate_joint_model(
        tm,
        data,
        batch_size=1,
        max_seq_length=16,
        loss_fn=JointLoss(weights={"lm_head": 0.0, "p": 1.0}),
        eval_metrics_fn=stub_metrics,
    )
    # token-weighted: (1.0*1 + 0.0*9) / (1+9) = 0.1  (a per-batch mean would give 0.5)
    assert res["acc"] == pytest.approx(0.1, abs=1e-6)

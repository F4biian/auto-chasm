"""Oracle tests for multi-class (ce) and regression (mse/mae) probe losses.

Guards two things: the trainable model must keep a multi-class head's
``[B, T, C]`` shape (it used to ``squeeze(-1)`` and break it), and the new
``ce``/``mse`` probe-loss shortcuts must match an independent recomputation.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import JointLoss, Model, ProbeConfig
from auto_chasm.trainers.loss import _canonical_loss_name
from auto_chasm.trainers.trainable import _TrainableModel


class _TinyMlp(nn.Module):
    def __init__(self, h: int = 16, v: int = 32, layers: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(v, h)
        self.layers = [nn.Linear(h, h) for _ in range(layers)]
        self.output_proj = nn.Linear(h, v)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = 16
    num_hidden_layers = 4


def _model(out_features: int) -> Model:
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(
        ProbeConfig(
            name="p",
            layers=[0],
            aggregation="last",
            module_config={"out_features": out_features},
        )
    )
    return m


def test_resolve_ce_and_mae():
    assert _canonical_loss_name("ce") == "ce"
    assert _canonical_loss_name("mae") == "mae"


def test_multiclass_head_keeps_btc_shape():
    m = _model(out_features=3)
    tm = _TrainableModel(m.model, m._probes)
    _, probes = tm(mx.array([[1, 2, 3, 4, 5]]))
    assert probes["p"].shape == (1, 5, 3)  # C=3 preserved, not squeezed to (1, 5)


def test_ce_loss_matches_independent_recompute():
    m = _model(out_features=3)
    loss = JointLoss(weights={"lm_head": 0.0}, losses={"p": "ce"})
    tm = _TrainableModel(m.model, m._probes)
    batch = mx.array([[1, 2, 3, 4, 5]])
    labels = mx.array([[0, 1, 2, 1, 0]])
    lengths = mx.array([[0, 4]])

    total, _, comp = loss(tm, batch, labels, lengths)

    # Independent recompute of the masked-mean cross-entropy.
    _, probes = tm(batch[:, :-1])
    logits = probes["p"]
    tgt = labels[:, 1:]
    steps = mx.arange(1, tgt.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:]) & (tgt != -100)
    ce = nn.losses.cross_entropy(logits, tgt.astype(mx.int32), reduction="none")
    expected = float((ce * mask).sum() / mx.maximum(mask.sum(), 1))
    assert float(comp["p"]) == pytest.approx(expected, rel=1e-5)
    assert float(total) == pytest.approx(expected, rel=1e-5)


def test_mse_loss_matches_independent_recompute():
    m = _model(out_features=1)
    loss = JointLoss(weights={"lm_head": 0.0}, losses={"p": "mse"})
    tm = _TrainableModel(m.model, m._probes)
    batch = mx.array([[1, 2, 3, 4, 5]])
    labels = mx.array([[0.1, 0.5, 0.9, 0.5, 0.1]])
    lengths = mx.array([[0, 4]])

    _, _, comp = loss(tm, batch, labels, lengths)

    _, probes = tm(batch[:, :-1])
    logits = probes["p"]  # squeezed to [B, T]
    tgt = labels[:, 1:].astype(mx.float32)
    steps = mx.arange(1, tgt.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:]) & (tgt != -100)
    mse = nn.losses.mse_loss(logits, tgt, reduction="none")
    expected = float((mse * mask).sum() / mx.maximum(mask.sum(), 1))
    assert float(comp["p"]) == pytest.approx(expected, rel=1e-4)

"""Oracle tests for backend-agnostic ops and custom joint losses.

Verifies that a custom joint loss like ``l_ce * lam * exp(l_bce)`` can be
written cleanly (``JointOutputs.probes["x"].bce(...)`` works) and that the
``ops`` helpers agree numerically across MLX and PyTorch.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import pytest
import torch

from auto_chasm import ops
from auto_chasm.config import ProbeConfig
from auto_chasm.model import Model
from auto_chasm.outputs import JointOutputs
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


@pytest.mark.parametrize(
    "fn,val,expected",
    [
        (ops.exp, 1.0, math.e),
        (ops.log, math.e, 1.0),
        (ops.sqrt, 4.0, 2.0),
        (ops.sigmoid, 0.0, 0.5),
        (ops.softplus, 0.0, math.log(2.0)),
    ],
)
def test_ops_value_and_backend_parity(fn, val, expected):
    out_mlx = float(fn(mx.array(val)))
    out_torch = float(fn(torch.tensor(val)))
    assert out_mlx == pytest.approx(expected, rel=1e-5)
    assert out_torch == pytest.approx(expected, rel=1e-5)


def test_ops_clamp_parity():
    assert float(ops.clamp(mx.array(5.0), hi=2.0)) == pytest.approx(2.0)
    assert float(ops.clamp(torch.tensor(-5.0), lo=-1.0)) == pytest.approx(-1.0)


def test_custom_joint_loss_l_ce_times_exp_l_bce():
    """Oracle: a custom joint loss equals l_ce * lam * exp(l_bce) exactly."""
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(ProbeConfig(name="digit", layers=[0], aggregation="last"))
    tm = _TrainableModel(m.model, m._probes)

    batch = mx.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    labels = mx.array([[0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0]])
    lengths = mx.array([[0, 5], [0, 5]])
    lam = 0.5

    lm_logits, probes = tm(batch[:, :-1])
    o = JointOutputs(lm_logits, probes, batch[:, 1:], lengths)
    l_ce = o.lm_ce
    l_bce = o.probes["digit"].bce(labels[:, 1:].astype(mx.float32), mask=o.mask)
    total = l_ce * lam * ops.exp(l_bce)

    expected = float(l_ce) * lam * math.exp(float(l_bce))
    assert float(total) == pytest.approx(expected, rel=1e-5)
    # And it is a live tensor, not a Python float (stays differentiable).
    assert isinstance(total, mx.array)


def test_custom_loss_with_bce_trains_under_value_and_grad():
    """Regression: o.probes[x].bce() must work INSIDE a trained loss_fn.

    The masked-mean guards used to do a Python ``if denom == 0`` on a traced
    array, which crashes under MLX value_and_grad — i.e. exactly during
    training, which eager unit tests did not exercise.
    """
    from auto_chasm import Trainer

    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(ProbeConfig(name="digit", layers=[0], aggregation="last"))

    def my_loss(model, batch, labels, lengths):
        lm_logits, probes = model(batch[:, :-1])
        o = JointOutputs(lm_logits, probes, batch[:, 1:], lengths)
        l_bce = o.probes["digit"].bce(labels[:, 1:], mask=o.mask)  # int labels: cast internally
        total = o.lm_ce + 2.0 * ops.exp(l_bce)
        return total, o.ntoks, {"bce": l_bce}

    data = [{"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 0, 0]} for _ in range(6)]
    result = Trainer(
        model=m, loss_fn=my_loss, num_iters=3, batch_size=2, logging_steps=1, verbose=False
    ).train(data)
    assert len(result["history"].train_losses) >= 1

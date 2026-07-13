"""Oracle + parity tests for class-weighted BCE (the binary counterpart of CE).

Pins: the weighted-BCE value vs an independent numpy recompute; uniform weights
``[1, 1]`` reproduce the plain masked-mean BCE; MLX==torch parity on non-uniform
weights and with ``-100`` ignores; the all-ignored finite ``0`` guard; the
length-2 guard; JointLoss integration (a binary probe with ``class_weights`` no
longer raises and actually changes the loss; ``"balanced"`` resolves; sequence
level works); and the honest compute-time raise for a non-weightable loss.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm import JointLoss, Model, ProbeConfig, Trainer
from auto_chasm.outputs import ProbeOutput
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


def _binary_model(granularity: str = "token") -> Model:
    """A tiny model with ONE binary (``out_features=1``) probe named ``p``."""
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(
        ProbeConfig(
            name="p",
            layers=[0],
            aggregation="last",
            granularity=granularity,
            module_config={"out_features": 1},
        )
    )
    return m


def _np_weighted_bce(
    logits: np.ndarray, targets: np.ndarray, mask: np.ndarray, weights: list
) -> float:
    """Independent numpy weighted masked-mean BCE (the oracle).

    BCE-with-logits ``= softplus(z) - z*t``; the per-position weight linearly
    interpolates ``w_neg -> w_pos`` over the (clamped) target; divides by the
    summed weight, guarding only the all-masked 0/0 with ``1e-8``.
    """
    valid = (targets != -100) & mask.astype(bool)
    bce = np.logaddexp(0.0, logits) - logits * targets  # raw target, matches lib BCE
    t01 = np.clip(targets, 0.0, 1.0)
    w0, w1 = float(weights[0]), float(weights[1])
    w_each = w0 + (w1 - w0) * t01
    ww = w_each * valid
    denom = ww.sum()
    return float((bce * ww).sum() / max(denom, 1e-8))


# --------------------------------------------------------------------------- #
# Unit oracle + parity on ProbeOutput.bce(weights=...)                         #
# --------------------------------------------------------------------------- #


def test_weighted_bce_value_oracle_mlx() -> None:
    """probe.bce(weights=...) equals an independent numpy weighted masked-mean BCE."""
    logits = mx.array([[2.0, -1.0, 0.5, 3.0]])
    targets = mx.array([[1.0, 0.0, 1.0, 0.0]])
    mask = mx.array([[True, True, True, True]])
    weights = [1.0, 3.0]
    out = float(ProbeOutput(logits=logits).bce(targets, mask=mask, weights=weights))
    expected = _np_weighted_bce(np.array(logits), np.array(targets), np.array(mask), weights)
    assert out == pytest.approx(expected, abs=1e-5)


def test_weighted_bce_honors_minus_100_ignore() -> None:
    """A ``-100`` position is excluded from both the numerator and denominator."""
    logits = mx.array([[2.0, -1.0, 0.5, 3.0]])
    targets = mx.array([[1.0, 0.0, -100.0, 0.0]])  # 3rd position ignored
    mask = mx.array([[True, True, True, True]])
    weights = [1.0, 4.0]
    out = float(ProbeOutput(logits=logits).bce(targets, mask=mask, weights=weights))
    expected = _np_weighted_bce(np.array(logits), np.array(targets), np.array(mask), weights)
    assert out == pytest.approx(expected, abs=1e-5)
    assert np.isfinite(out)


def test_weighted_bce_uniform_equals_unweighted() -> None:
    """weights=[1, 1] reproduces the plain masked-mean BCE."""
    logits = mx.array([[2.0, -1.0, 0.5, 3.0, -2.0]])
    targets = mx.array([[1.0, 0.0, 1.0, 0.0, 1.0]])
    mask = mx.array([[True, True, True, True, True]])
    po = ProbeOutput(logits=logits)
    plain = float(po.bce(targets, mask=mask))
    unit = float(po.bce(targets, mask=mask, weights=[1.0, 1.0]))
    assert plain == pytest.approx(unit, abs=1e-6)


def test_weighted_bce_non_uniform_changes_value() -> None:
    """A skewed weight vector actually changes the loss (not a no-op)."""
    logits = mx.array([[2.0, -1.0, 0.5, 3.0]])
    targets = mx.array([[1.0, 0.0, 1.0, 0.0]])
    mask = mx.array([[True, True, True, True]])
    po = ProbeOutput(logits=logits)
    plain = float(po.bce(targets, mask=mask))
    skew = float(po.bce(targets, mask=mask, weights=[0.2, 5.0]))
    assert abs(plain - skew) > 1e-4


def test_weighted_bce_mlx_torch_parity() -> None:
    """weighted BCE agrees across MLX and torch (non-uniform weights, with -100)."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    ln = rng.standard_normal((2, 5)).astype(np.float32)
    t = np.array([[1.0, 0.0, 1.0, -100.0, 0.0], [0.0, 1.0, -100.0, 1.0, 1.0]], dtype=np.float32)
    mask_np = np.ones((2, 5), bool)
    weights = [0.7, 2.3]

    m_out = float(
        ProbeOutput(logits=mx.array(ln)).bce(mx.array(t), mask=mx.array(mask_np), weights=weights)
    )
    t_out = float(
        ProbeOutput(logits=torch.tensor(ln)).bce(
            torch.tensor(t), mask=torch.tensor(mask_np), weights=weights
        )
    )
    assert m_out == pytest.approx(t_out, abs=1e-5)


def test_weighted_bce_all_ignored_is_finite_zero() -> None:
    """An all-(-100) window yields a finite 0 (no NaN from 0/0)."""
    po = ProbeOutput(logits=mx.array([[1.0, 2.0]]))
    out = float(
        po.bce(mx.array([[-100.0, -100.0]]), mask=mx.array([[True, True]]), weights=[1.0, 3.0])
    )
    assert out == 0.0
    assert np.isfinite(out)


def test_weighted_bce_wrong_length_raises() -> None:
    """Binary class weights must have exactly 2 entries."""
    po = ProbeOutput(logits=mx.array([[1.0, -1.0]]))
    with pytest.raises(ValueError, match="exactly 2"):
        po.bce(mx.array([[1.0, 0.0]]), mask=mx.array([[True, True]]), weights=[1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# JointLoss integration                                                        #
# --------------------------------------------------------------------------- #


def test_jointloss_bce_class_weights_applied_skewed_not_uniform() -> None:
    """class_weights flow through JointLoss to a bce probe: skewed moves the loss, uniform does not.

    (Magnitude correctness is covered by the ProbeOutput.bce unit oracle above; this
    only pins the *routing* — a skewed vector is not a no-op, a uniform one is. The
    untrained probe emits near-zero logits, so the skewed shift is small but nonzero;
    the point is that it is applied at all, and that ``[1, 1]`` is bit-identical to
    the unweighted path.)
    """
    mx.random.seed(0)
    m = _binary_model()
    tm = _TrainableModel(m.model, m._probes)
    batch = mx.array([[1, 2, 3, 4, 5]])
    labels = mx.array([[0, 1, 1, 0, 1]])
    lengths = mx.array([[0, 4]])
    base = float(JointLoss(weights={"lm_head": 0.0})(tm, batch, labels, lengths)[0])
    skewed = float(
        JointLoss(weights={"lm_head": 0.0}, class_weights=[1.0, 5.0])(tm, batch, labels, lengths)[0]
    )
    uniform = float(
        JointLoss(weights={"lm_head": 0.0}, class_weights=[1.0, 1.0])(tm, batch, labels, lengths)[0]
    )
    assert np.isfinite(skewed)
    assert skewed != base  # skewed weights are applied (a no-op would be bit-identical)
    assert uniform == pytest.approx(base, abs=1e-6)  # uniform weights are a no-op


def test_trainer_resolves_balanced_for_bce() -> None:
    """Trainer._resolve_class_weights fills a 2-vector for a binary bce probe."""
    m = _binary_model()
    m.freeze_model()
    m.unfreeze_all_probes()
    loss = JointLoss(weights={"lm_head": 0.0}, class_weights="balanced")
    trainer = Trainer(model=m, loss_fn=loss, num_iters=1, batch_size=1, save_steps=0, verbose=False)
    trainer._resolve_class_weights([{"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 1, 1]}])
    assert isinstance(loss._class_weights, list)
    assert len(loss._class_weights) == 2


def test_seq_level_bce_class_weights_works() -> None:
    """Sequence-level (response) bce + class_weights computes a finite loss (no raise)."""
    m = _binary_model(granularity="response")
    tm = _TrainableModel(m.model, m._probes)
    out = JointLoss(weights={"lm_head": 0.0}, class_weights=[1.0, 2.0])(
        tm, mx.array([[1, 2, 3, 4, 5]]), mx.array([[1, 1, 1, 0, 0]]), mx.array([[0, 4]])
    )[0]
    assert np.isfinite(float(out))


def test_class_weights_on_mse_probe_raises_at_compute() -> None:
    """class_weights on an mse probe raises at compute (would be a silent no-op)."""
    m = _binary_model()
    tm = _TrainableModel(m.model, m._probes)
    loss = JointLoss(weights={"lm_head": 0.0}, losses={"p": "mse"}, class_weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="class weights only apply"):
        loss(tm, mx.array([[1, 2, 3, 4, 5]]), mx.array([[0, 1, 1, 0, 1]]), mx.array([[0, 4]]))

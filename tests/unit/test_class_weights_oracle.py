"""Oracle + parity tests for class-weighted CE and the 'balanced' resolution.

Pins: the weighted CE value vs an independent numpy recompute; uniform weights
reproduce the unweighted path (so ``class_weights=None`` is unchanged); the
``-100``/all-ignored guards; MLX==torch parity on non-uniform weights; the
honest raises (unresolved 'balanced', sequence-level CE, custom-loss combo); the
``balanced_class_weights`` formula; and the trainer's auto-resolution.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm import JointLoss, Model, ProbeConfig, Trainer, classification_metrics
from auto_chasm.data import balanced_class_weights
from auto_chasm.trainers._loss_ce import weighted_ce
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


def _model(out_features: int, granularity: str = "token") -> Model:
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(
        ProbeConfig(
            name="p",
            layers=[0],
            aggregation="last",
            granularity=granularity,
            module_config={"out_features": out_features},
        )
    )
    return m


def _np_weighted_ce(
    logits: np.ndarray, targets: np.ndarray, mask: np.ndarray, weights: list
) -> float:
    """Independent numpy weighted masked-mean CE (the oracle)."""
    valid = (targets != -100) & mask
    safe = np.clip(targets, 0, None)
    mx_ = logits.max(-1, keepdims=True)
    logp = logits - mx_ - np.log(np.exp(logits - mx_).sum(-1, keepdims=True))
    ce = -np.take_along_axis(logp, safe[..., None], axis=-1).squeeze(-1)
    w = np.array(weights)[safe]
    ww = w * valid
    # True weighted mean: divide by the SUMMED weight, guarding only all-masked 0/0
    # (never clamp to 1.0 — that distorts the loss for summed weights < 1).
    denom = ww.sum()
    return float((ce * ww).sum() / (denom if denom > 0 else 1.0))


def test_weighted_ce_mlx_value_oracle() -> None:
    """weighted_ce equals an independent numpy weighted masked-mean CE (MLX)."""
    logits = mx.array([[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]]])
    shifted = mx.array([[0, 1, -100]])
    label_valid = shifted != -100
    probe_mask = mx.logical_and(mx.array([[True, True, True]]), label_valid)
    weights = [1.0, 3.0, 0.5]
    out = float(weighted_ce(logits, shifted, label_valid, probe_mask, weights))
    expected = _np_weighted_ce(np.array(logits), np.array(shifted), np.ones((1, 3), bool), weights)
    assert out == pytest.approx(expected, abs=1e-5)


def test_weighted_ce_fractional_weights_not_rescaled() -> None:
    """Summed weights < 1 give the TRUE weighted mean, not a loss clamped by denom=1.

    Regression for the clamp-to-1.0 bug: with small/normalized class weights the
    summed weight is < 1, so clamping the denominator to 1.0 divided by the wrong
    number and shrank the loss (scale became batch-content-dependent).
    """
    logits = mx.array([[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    shifted = mx.array([[0, 1]])
    label_valid = shifted != -100
    probe_mask = mx.array([[True, True]])
    weights = [0.05, 0.08, 0.02]  # summed weight over the two valid tokens = 0.13 (< 1)
    out = float(weighted_ce(logits, shifted, label_valid, probe_mask, weights))
    expected = _np_weighted_ce(np.array(logits), np.array(shifted), np.ones((1, 2), bool), weights)
    assert out == pytest.approx(expected, abs=1e-5)
    # Hand-computed true weighted mean = (0.05*0.2395 + 0.08*0.5514)/0.13 ≈ 0.4315.
    # The old denom-clamp-to-1.0 returned sum(w*ce)/1.0 ≈ 0.0561 (~7.7x too small).
    assert out == pytest.approx(0.4315, abs=1e-3)


def test_make_joint_loss_ce_with_class_weights_does_not_raise() -> None:
    """`make_joint_loss(probe_loss="ce", class_weights=[...])` validates against ce.

    Regression: the legacy adapter validated class_weights against the "bce" JointLoss
    default before setting the real default loss, so this valid combination wrongly
    raised "no probe uses probe_loss='ce'".
    """
    from auto_chasm.trainers.loss import _joint_loss_from_legacy
    from auto_chasm.trainers.trainable import make_joint_loss

    # The internal adapter returns the JointLoss object with the weights applied.
    jl = _joint_loss_from_legacy(lm_weight=0.0, probe_loss="ce", class_weights=[1.0, 2.0, 3.0])
    assert isinstance(jl, JointLoss)
    assert jl._class_weights == [1.0, 2.0, 3.0]

    # make_joint_loss (MLX) wraps it in a callable — the point is it does NOT raise.
    fn = make_joint_loss(lm_weight=0.0, probe_loss="ce", class_weights=[1.0, 2.0, 3.0])
    assert callable(fn)


def test_uniform_weights_equal_unweighted() -> None:
    """class_weights=[1,1,1] reproduces the unweighted ('None') CE bit-for-bit."""
    m = _model(3)
    tm = _TrainableModel(m.model, m._probes)
    batch = mx.array([[1, 2, 3, 4, 5]])
    labels = mx.array([[0, 1, 2, 1, 0]])
    lengths = mx.array([[0, 4]])
    unweighted = JointLoss(weights={"lm_head": 0.0}, losses={"p": "ce"})(
        tm, batch, labels, lengths
    )[0]
    weighted = JointLoss(
        weights={"lm_head": 0.0}, losses={"p": "ce"}, class_weights=[1.0, 1.0, 1.0]
    )(tm, batch, labels, lengths)[0]
    assert float(unweighted) == pytest.approx(float(weighted), abs=1e-5)


def test_non_uniform_weights_change_loss() -> None:
    """A non-uniform weight vector actually changes the loss (not a no-op)."""
    m = _model(3)
    tm = _TrainableModel(m.model, m._probes)
    batch = mx.array([[1, 2, 3, 4, 5]])
    labels = mx.array([[0, 0, 0, 1, 2]])
    lengths = mx.array([[0, 4]])
    base = float(
        JointLoss(weights={"lm_head": 0.0}, losses={"p": "ce"})(tm, batch, labels, lengths)[0]
    )
    skew = float(
        JointLoss(weights={"lm_head": 0.0}, losses={"p": "ce"}, class_weights=[5.0, 0.1, 0.1])(
            tm, batch, labels, lengths
        )[0]
    )
    assert abs(base - skew) > 1e-4


def test_all_ignored_is_finite_zero() -> None:
    """An all-(-100) window yields a finite 0 (no NaN)."""
    logits = mx.array([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]])
    shifted = mx.array([[-100, -100]])
    lv = shifted != -100
    out = float(weighted_ce(logits, shifted, lv, lv, [1.0, 1.0, 1.0]))
    assert out == 0.0
    assert np.isfinite(out)


def test_weighted_ce_mlx_torch_parity() -> None:
    """The unified weighted_ce agrees across MLX and torch on non-uniform weights."""
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(0)
    ln = rng.standard_normal((2, 4, 5)).astype(np.float32)
    t = np.array([[0, 1, -100, 4], [2, 2, 3, -100]])
    mask_np = np.ones((2, 4), bool)
    weights = [0.5, 2.0, 1.0, 0.3, 1.7]

    sh = mx.array(t)
    lv = sh != -100
    pm = mx.logical_and(mx.array(mask_np), lv)
    mout = float(weighted_ce(mx.array(ln), sh, lv, pm, weights))

    sh_t = torch.tensor(t)
    lv_t = sh_t != -100
    pm_t = torch.tensor(mask_np) & lv_t
    tout = float(weighted_ce(torch.tensor(ln), sh_t, lv_t, pm_t, weights))
    assert mout == pytest.approx(tout, abs=1e-5)


def test_unresolved_balanced_raises() -> None:
    """Using class_weights='balanced' without trainer resolution raises."""
    m = _model(3)
    tm = _TrainableModel(m.model, m._probes)
    loss = JointLoss(weights={"lm_head": 0.0}, losses={"p": "ce"}, class_weights="balanced")
    with pytest.raises(NotImplementedError, match="balanced"):
        loss(tm, mx.array([[1, 2, 3, 4, 5]]), mx.array([[0, 1, 2, 1, 0]]), mx.array([[0, 4]]))


def test_seq_level_class_weights_raises() -> None:
    """class_weights on a sequence-level (response) CE probe raises."""
    m = _model(3, granularity="response")
    tm = _TrainableModel(m.model, m._probes)
    loss = JointLoss(weights={"lm_head": 0.0}, losses={"p": "ce"}, class_weights=[1.0, 1.0, 1.0])
    with pytest.raises(NotImplementedError, match="sequence-level"):
        loss(tm, mx.array([[1, 2, 3, 4, 5]]), mx.array([[2, 2, 2, 2, 2]]), mx.array([[0, 4]]))


def test_class_weights_with_custom_loss_raises() -> None:
    """class_weights on a probe whose loss is a custom callable raises at compute.

    The default probe loss is the (weightable) ``bce``, so construction can't tell a
    custom-loss-only probe apart from a to-be-attached bce probe; the precise
    per-probe enforcement runs at compute time (where the resolved loss is known).
    """
    m = _model(3)
    tm = _TrainableModel(m.model, m._probes)
    loss = JointLoss(
        weights={"lm_head": 0.0},
        losses={"p": lambda logits, t, msk: logits.sum()},
        class_weights=[1.0, 1.0, 1.0],
    )
    with pytest.raises(ValueError, match="class weights only apply"):
        loss(tm, mx.array([[1, 2, 3, 4, 5]]), mx.array([[0, 1, 2, 1, 0]]), mx.array([[0, 4]]))


def test_balanced_class_weights_formula() -> None:
    """balanced_class_weights = total / (C * max(count, 1)) over non-(-100) labels."""
    data = [
        {"tokens": [1, 2, 3], "labels": [0, 0, 1]},
        {"tokens": [4, 5], "labels": [2, -100]},
    ]
    # counts [2, 1, 1], total 4, C 3
    assert balanced_class_weights(data, 3) == pytest.approx([4 / 6, 4 / 3, 4 / 3])
    # num_classes inferred from data (max label + 1 = 3)
    assert balanced_class_weights(data) == pytest.approx([4 / 6, 4 / 3, 4 / 3])


def test_trainer_resolves_balanced() -> None:
    """Trainer._resolve_class_weights replaces 'balanced' with a concrete vector."""
    m = _model(3)
    m.freeze_model()
    m.unfreeze_all_probes()
    loss = JointLoss(weights={"lm_head": 0.0}, losses={"p": "ce"}, class_weights="balanced")
    trainer = Trainer(model=m, loss_fn=loss, num_iters=1, batch_size=1, save_steps=0, verbose=False)
    trainer._resolve_class_weights([{"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 2, 2]}])
    assert isinstance(loss._class_weights, list)
    assert len(loss._class_weights) == 3


def test_exp2_pattern_train_balanced_and_metric_early_stop(tmp_path) -> None:
    """End-to-end: Trainer.train with class_weights='balanced' + metric early stop.

    Mirrors exp2_new: a single head, balanced weights resolved from the data,
    early stopping on a per-probe adjacent-accuracy metric, and test metrics out.
    """
    m = _model(3)
    m.freeze_model()
    m.unfreeze_all_probes()
    data = [
        {"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 2, 2]},
        {"tokens": [6, 7, 8, 9, 10], "labels": [1, 1, 0, 2, 1]},
    ]
    trainer = Trainer(
        model=m,
        loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"p": "ce"}, class_weights="balanced"),
        num_iters=6,
        batch_size=2,
        eval_steps=3,
        save_steps=0,
        verbose=False,
        output_dir=str(tmp_path),
        eval_metrics_fn=classification_metrics(num_classes=3, ordinal_tol=1),
        early_stopping_metric="val_p_adj",
        early_stopping_higher_is_better=True,
        early_stopping_patience=5,
    )
    res = trainer.train(data, val_data=data, test_data=data)
    assert "p_acc" in res["test_metrics"]
    assert isinstance(trainer.loss_fn._class_weights, list)  # 'balanced' was resolved


def test_class_weights_dict_typo_key_raises() -> None:
    """A dict class_weights key that names no probe raises (was silently unweighted)."""
    jl = JointLoss(
        weights={"lm_head": 0.0, "p": 1.0}, losses={"p": "ce"}, class_weights={"typo": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="Unknown class_weights key"):
        jl._validate_weight_keys(["p"])
    # A correct probe key validates cleanly.
    ok = JointLoss(
        weights={"lm_head": 0.0, "p": 1.0}, losses={"p": "ce"}, class_weights={"p": [1.0, 2.0]}
    )
    ok._validate_weight_keys(["p"])

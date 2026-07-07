"""Oracle tests for ``lm_ce``, ``bce``, ``mae``, and loss masking.

Each test asserts a CORRECT VALUE against an INDEPENDENT recomputation, not
merely "it runs" (mirrors
``test_classification_regression.py::test_ce_loss_matches_independent_recompute``).
``ce`` and ``mse`` already have oracles there and are intentionally not
duplicated here. This file covers the remaining ``JointLoss`` component values
and the masking contract from ``the masking rules`` rule 4:

- ``lm_head`` — next-token cross-entropy over the length mask.
- probe ``bce`` — masked binary-cross-entropy-with-logits (hand-checkable
  ``logits == 0 -> ln 2`` per token).
- probe ``mae`` — masked mean absolute error vs targets.
- masking — ``-100`` labels and out-of-length-range padding contribute ZERO
  (flipping a masked label leaves the loss bit-identical), and an all-``-100``
  batch yields a finite ``0`` via the zero-mask guard, not ``NaN``/``inf``.
- MLX vs torch parity for the masked ``bce`` value.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm import Model, ProbeConfig
from auto_chasm.trainers.loss import JointLoss
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


class _FixedMlxModel:
    """MLX fake model returning fixed ``(lm_logits, {"probe": logits})``.

    Lets the probe value be hand-checkable (e.g. zero logits -> BCE ln 2) by
    bypassing the random trainable head.
    """

    def __init__(self, lm: np.ndarray, pr: np.ndarray) -> None:
        self.lm = mx.array(lm)
        self.pr = {"probe": mx.array(pr)}

    def __call__(self, inputs: mx.array) -> tuple:
        return self.lm, self.pr


class _FixedTorchModel:
    """Torch counterpart of :class:`_FixedMlxModel` for cross-backend parity."""

    def __init__(self, lm: np.ndarray, pr: np.ndarray) -> None:
        import torch

        self.lm = torch.tensor(lm)
        self.pr = {"probe": torch.tensor(pr)}

    def __call__(self, inputs: object) -> tuple:
        return self.lm, self.pr


class TestLmCeOracle:
    """``lm_ce`` equals an independent masked next-token cross-entropy."""

    def test_lm_ce_matches_independent_recompute(self) -> None:
        m = _model(out_features=1)
        loss = JointLoss(weights={"p": 0.0})
        tm = _TrainableModel(m.model, m._probes)
        batch = mx.array([[1, 2, 3, 4, 5, 6]])
        labels = mx.array([[0, 1, 0, 1, 0, 1]])
        lengths = mx.array([[1, 5]])

        total, _, comp = loss(tm, batch, labels, lengths)

        # Independent recompute: masked-mean CE of LM logits vs shifted tokens
        # over the length mask ``lengths[:, 0] <= step < lengths[:, 1]``.
        lm_logits, _ = tm(batch[:, :-1])
        targets = batch[:, 1:]
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
        ce = nn.losses.cross_entropy(lm_logits, targets, reduction="none")
        expected = float((ce * mask).astype(mx.float32).sum() / mx.maximum(mask.sum(), 1))

        assert float(comp["lm_head"]) == pytest.approx(expected, rel=1e-5)
        # probe_weight=0 -> total is the LM term alone.
        assert float(total) == pytest.approx(expected, rel=1e-5)

    def test_lm_ce_respects_length_mask(self) -> None:
        # Tokens outside [lengths[0], lengths[1]) must not contribute: a tighter
        # window over the SAME logits yields a strictly different masked mean.
        m = _model(out_features=1)
        loss = JointLoss(weights={"p": 0.0})
        tm = _TrainableModel(m.model, m._probes)
        batch = mx.array([[3, 1, 4, 1, 5, 9]])
        labels = mx.array([[0, 0, 0, 0, 0, 0]])

        _, _, wide = loss(tm, batch, labels, mx.array([[0, 5]]))
        _, _, narrow = loss(tm, batch, labels, mx.array([[2, 4]]))

        lm_logits, _ = tm(batch[:, :-1])
        targets = batch[:, 1:]
        ce_each = nn.losses.cross_entropy(lm_logits, targets, reduction="none")
        steps = mx.arange(1, targets.shape[1] + 1)
        nmask = mx.logical_and(steps >= 2, steps < 4)
        expected_narrow = float(
            (ce_each * nmask).astype(mx.float32).sum() / mx.maximum(nmask.sum(), 1)
        )
        assert float(narrow["lm_head"]) == pytest.approx(expected_narrow, rel=1e-5)
        assert float(narrow["lm_head"]) != pytest.approx(float(wide["lm_head"]), rel=1e-5)


class TestBceOracle:
    """``probe_bce`` equals an independent masked BCE-with-logits."""

    def test_bce_zero_logits_is_ln2_per_token(self) -> None:
        # Hand-checkable ground truth: sigmoid(0) = 0.5, so BCE-with-logits is
        # -[y*ln(.5) + (1-y)*ln(.5)] = ln 2 for EVERY token, regardless of label.
        b, t = 1, 6
        lm = np.zeros((b, t - 1, 32), np.float32)
        pr = np.zeros((b, t - 1), np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
        labels = np.array([[1, 1, 0, 1, 0, 1]], np.int32)
        lengths = np.array([[0, 5]], np.int32)

        loss = JointLoss(weights={"lm_head": 0.0})
        _, _, comp = loss(
            _FixedMlxModel(lm, pr), mx.array(batch), mx.array(labels), mx.array(lengths)
        )
        assert float(comp["probe"]) == pytest.approx(math.log(2.0), abs=1e-6)

    def test_bce_matches_independent_recompute(self) -> None:
        # Non-trivial logits/labels with a ``-100`` and a length-mask hole.
        rng = np.random.default_rng(0)
        b, t = 1, 6
        lm = np.zeros((b, t - 1, 32), np.float32)
        pr = rng.standard_normal((b, t - 1)).astype(np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
        labels = np.array([[1, 0, 1, -100, 0, 1]], np.int32)
        lengths = np.array([[1, 5]], np.int32)

        loss = JointLoss(weights={"lm_head": 0.0})
        _, _, comp = loss(
            _FixedMlxModel(lm, pr), mx.array(batch), mx.array(labels), mx.array(lengths)
        )

        shifted = mx.array(labels)[:, 1:]
        ptgt = shifted.astype(mx.float32)
        steps = mx.arange(1, t)
        mask = mx.logical_and(steps >= lengths[0, 0], steps < lengths[0, 1])
        mask = mx.logical_and(mask, ptgt != -100)
        bce = nn.losses.binary_cross_entropy(mx.array(pr), ptgt, reduction="none", with_logits=True)
        expected = float((bce * mask).astype(mx.float32).sum() / mx.maximum(mask.sum(), 1))
        assert float(comp["probe"]) == pytest.approx(expected, rel=1e-5)


class TestMaeOracle:
    """``probe_mae`` equals an independent masked mean absolute error."""

    def test_mae_matches_independent_recompute(self) -> None:
        rng = np.random.default_rng(1)
        b, t = 1, 6
        lm = np.zeros((b, t - 1, 32), np.float32)
        pr = rng.standard_normal((b, t - 1)).astype(np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
        # A ``-100`` regression target must be skipped, not treated as -100.0.
        labels = np.array([[0.0, 0.5, 1.5, -100.0, 2.0, 3.0]], np.float32)
        lengths = np.array([[0, 5]], np.int32)

        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "mae"})
        _, _, comp = loss(
            _FixedMlxModel(lm, pr), mx.array(batch), mx.array(labels), mx.array(lengths)
        )

        shifted = mx.array(labels)[:, 1:]
        ptgt = shifted.astype(mx.float32)
        steps = mx.arange(1, t)
        mask = mx.logical_and(steps >= lengths[0, 0], steps < lengths[0, 1])
        mask = mx.logical_and(mask, ptgt != -100)
        mae = mx.abs(mx.array(pr) - ptgt) * mask
        expected = float(mae.astype(mx.float32).sum() / mx.maximum(mask.sum(), 1))
        assert float(comp["probe"]) == pytest.approx(expected, rel=1e-5)


class TestMaskingOracle:
    """Masked positions contribute ZERO and the zero-mask guard stays finite.

    This is the masking-contract check: hand-padded batches with
    ``-100`` labels and out-of-range padding, proving the mask actually
    excludes those positions rather than merely down-weighting them.
    """

    def _setup(self) -> tuple:
        rng = np.random.default_rng(7)
        b, t = 1, 6
        lm = np.zeros((b, t - 1, 32), np.float32)
        pr = rng.standard_normal((b, t - 1)).astype(np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
        return lm, pr, batch

    def test_flipping_masked_label_leaves_loss_unchanged(self) -> None:
        lm, pr, batch = self._setup()
        # Length window [1, 4): valid shifted indices 0,1,2 (token steps 1,2,3).
        # shifted index 4 (label position 5) is doubly masked: it is a ``-100``
        # AND it falls outside the length window.
        lengths = np.array([[1, 4]], np.int32)
        labels = np.array([[7, 0, 1, 0, -100, 1]], np.int32)
        loss = JointLoss(weights={"lm_head": 0.0})

        _, n1, c1 = loss(
            _FixedMlxModel(lm, pr), mx.array(batch), mx.array(labels), mx.array(lengths)
        )

        # Flip the masked label to a wildly different value. If the mask truly
        # excludes it, the BCE value must be BIT-identical.
        flipped = labels.copy()
        flipped[0, 5] = 999
        _, n2, c2 = loss(
            _FixedMlxModel(lm, pr), mx.array(batch), mx.array(flipped), mx.array(lengths)
        )

        assert float(c1["probe"]) == float(c2["probe"])
        assert float(n1) == float(n2)

    def test_padding_outside_length_window_contributes_zero(self) -> None:
        # Two length windows over the SAME data: the narrow one only differs by
        # excluding extra (valid-label) tokens. The narrow loss must equal a
        # recompute that drops exactly those tokens, proving padding is excised.
        lm, pr, batch = self._setup()
        labels = np.array([[1, 0, 1, 0, 1, 0]], np.int32)
        loss = JointLoss(weights={"lm_head": 0.0})
        lengths = np.array([[2, 4]], np.int32)

        _, ntoks, comp = loss(
            _FixedMlxModel(lm, pr), mx.array(batch), mx.array(labels), mx.array(lengths)
        )

        shifted = mx.array(labels)[:, 1:].astype(mx.float32)
        steps = mx.arange(1, 6)
        mask = mx.logical_and(steps >= 2, steps < 4)
        bce = nn.losses.binary_cross_entropy(
            mx.array(pr), shifted, reduction="none", with_logits=True
        )
        expected = float((bce * mask).astype(mx.float32).sum() / mx.maximum(mask.sum(), 1))
        assert float(comp["probe"]) == pytest.approx(expected, rel=1e-5)
        assert float(ntoks) == float(mask.sum())

    def test_all_minus_100_batch_is_finite_zero(self) -> None:
        # Zero-mask guard: every probe label is ``-100`` so the masked-mean
        # denominator would be 0. The guard must yield finite 0, not NaN/inf.
        lm, pr, batch = self._setup()
        labels = np.full((1, 6), -100, np.int32)
        lengths = np.array([[0, 5]], np.int32)
        loss = JointLoss(weights={"lm_head": 0.0})

        _, _, comp = loss(
            _FixedMlxModel(lm, pr), mx.array(batch), mx.array(labels), mx.array(lengths)
        )
        value = float(comp["probe"])
        assert math.isfinite(value)
        assert value == pytest.approx(0.0, abs=1e-7)

    def test_empty_length_window_is_finite_zero(self) -> None:
        # An empty length window (lengths[0] == lengths[1]) masks every token.
        # The zero-mask guard must still return a finite 0.
        lm, pr, batch = self._setup()
        labels = np.array([[1, 0, 1, 0, 1, 0]], np.int32)
        lengths = np.array([[3, 3]], np.int32)
        loss = JointLoss(weights={"lm_head": 0.0})

        _, ntoks, comp = loss(
            _FixedMlxModel(lm, pr), mx.array(batch), mx.array(labels), mx.array(lengths)
        )
        value = float(comp["probe"])
        assert math.isfinite(value)
        assert value == pytest.approx(0.0, abs=1e-7)
        assert float(ntoks) == 0.0


class TestMaskingBackendParity:
    """The masked ``bce`` value agrees between MLX and torch within 1e-5."""

    def test_bce_masking_mlx_torch_parity(self) -> None:
        pytest.importorskip("torch")
        import torch

        rng = np.random.default_rng(11)
        b, t = 1, 6
        lm = np.zeros((b, t - 1, 32), np.float32)
        pr = rng.standard_normal((b, t - 1)).astype(np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
        labels = np.array([[1, 0, 1, -100, 0, 1]], np.int32)
        lengths = np.array([[1, 4]], np.int32)
        loss = JointLoss(weights={"lm_head": 0.0})

        _, mn, mc = loss(
            _FixedMlxModel(lm, pr), mx.array(batch), mx.array(labels), mx.array(lengths)
        )
        _, tn, tc = loss(
            _FixedTorchModel(lm, pr),
            torch.tensor(batch),
            torch.tensor(labels),
            torch.tensor(lengths),
        )
        assert abs(float(mc["probe"]) - float(tc["probe"])) < 1e-5
        assert float(mn) == float(tn)

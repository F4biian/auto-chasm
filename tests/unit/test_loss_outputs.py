"""Tests for loss computation and outputs.

Covers ``src/auto_chasm/trainers/loss.py`` (``JointLoss``),
``src/auto_chasm/outputs.py`` (``ProbeOutput`` losses / ``JointOutputs``),
``src/auto_chasm/ops.py``.

Every numerical assertion is checked against a hand-computed / independent
``numpy`` (or independent-recompute) ground truth — never merely "it runs".
Tests named ``test_BUG_*`` are regression tests for specific past defects; the
rest are general regression coverage.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm.outputs import ProbeOutput
from auto_chasm.trainers.loss import JointLoss

# ---------------------------------------------------------------------------
# Fixed in-memory fake models (deterministic, hand-checkable)
# ---------------------------------------------------------------------------


class _FixedMlx:
    """MLX fake returning fixed ``(lm_logits, {name: logits})``."""

    def __init__(self, lm: np.ndarray, probes: dict[str, np.ndarray]) -> None:
        self.lm = mx.array(lm)
        self.pr = {k: mx.array(v) for k, v in probes.items()}

    def __call__(self, inputs: object, mask: object | None = None) -> tuple:
        return self.lm, self.pr


class _FixedTorch:
    """Torch counterpart of :class:`_FixedMlx`."""

    def __init__(self, lm: np.ndarray, probes: dict[str, np.ndarray]) -> None:
        import torch

        self.lm = torch.tensor(lm)
        self.pr = {k: torch.tensor(v) for k, v in probes.items()}

    def __call__(self, inputs: object, mask: object | None = None) -> tuple:
        return self.lm, self.pr


# ===========================================================================
# CONFIRMED BUG 1 (critical): JointLoss probe_ce silently wrong on MLX for an
# out-of-range class index, while torch raises IndexError.  outputs.py has a
# guard (`_check_class_indices`) for exactly this, but JointLoss does not call
# it, so a mislabeled multi-class dataset produces a *silently wrong number*
# on MLX and a crash on torch — a cross-backend correctness divergence and a
# research-poisoning footgun.
# ===========================================================================


class TestCeOutOfRangeIndex:
    """Cross-entropy with an out-of-range class index must error, not silently miscompute."""

    def _setup(self, C: int = 3, oor: int = 99):
        rng = np.random.default_rng(1)
        b, t = 1, 6
        lm = np.zeros((b, t - 1, 32), np.float32)
        pr = rng.standard_normal((b, t - 1, C)).astype(np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
        # A valid (in-length) position carries an out-of-range class index.
        labels = np.array([[0, 1, 2, oor, 0, 1]], np.int32)
        lengths = np.array([[0, 5]], np.int32)
        return lm, {"probe": pr}, batch, labels, lengths

    def test_torch_raises_on_out_of_range_ce_index(self) -> None:
        """Torch (correctly) refuses an out-of-range CE class index.

        After the fix both backends raise the same clear ``ValueError`` from
        ``_check_class_indices``; torch's own ``IndexError`` is also accepted in
        case the guard is ever bypassed.
        """
        pytest.importorskip("torch")
        import torch

        lm, pr, batch, labels, lengths = self._setup()
        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "ce"})
        with pytest.raises((ValueError, IndexError)):
            loss(
                _FixedTorch(lm, pr),
                torch.tensor(batch),
                torch.tensor(labels),
                torch.tensor(lengths),
            )

    def test_BUG_mlx_ce_out_of_range_index_silently_wrong(self) -> None:
        """MLX must NOT silently compute a number for an out-of-range class index.

        Desired behavior: a clear ``ValueError`` (matching ``outputs.ce``'s
        ``_check_class_indices`` guard and torch's ``IndexError``).  Currently
        MLX gathers garbage scores and returns a plausible-looking but wrong
        loss, diverging from torch and poisoning any multi-class run with a
        mislabeled target.
        """
        lm, pr, batch, labels, lengths = self._setup()
        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "ce"})
        with pytest.raises((ValueError, IndexError)):
            loss(
                _FixedMlx(lm, pr),
                mx.array(batch),
                mx.array(labels),
                mx.array(lengths),
            )

    def test_BUG_ce_out_of_range_index_cross_backend_divergence(self) -> None:
        """An identical mislabeled batch must behave the same on both backends.

        One backend crashing while the other returns a (wrong) number is a
        silent-divergence bug: the same code+data gives different scientific
        results depending on the machine.
        """
        pytest.importorskip("torch")
        import torch

        lm, pr, batch, labels, lengths = self._setup()
        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "ce"})

        torch_raised = False
        try:
            loss(
                _FixedTorch(lm, pr),
                torch.tensor(batch),
                torch.tensor(labels),
                torch.tensor(lengths),
            )
        except (ValueError, IndexError):
            torch_raised = True

        mlx_raised = False
        try:
            loss(
                _FixedMlx(lm, pr),
                mx.array(batch),
                mx.array(labels),
                mx.array(lengths),
            )
        except (ValueError, IndexError):
            mlx_raised = True

        assert torch_raised == mlx_raised, (
            "Backends diverge on an out-of-range CE class index: "
            f"torch_raised={torch_raised}, mlx_raised={mlx_raised}."
        )


# ===========================================================================
# CONFIRMED BUG 2 (high): ProbeOutput.ce diverges across backends when targets
# contain the -100 ignore sentinel and no mask is passed.  torch's
# functional.cross_entropy defaults to ignore_index=-100 (drops the element);
# MLX's cross_entropy gathers index -100 (wraparound garbage).  The
# _check_class_indices guard explicitly *permits* -100, implying .ce() is meant
# to handle it — but the two backends produce different numbers.
# ===========================================================================


class TestProbeOutputCeIgnoreIndex:
    """Probe-output CE with -100 (ignore index) targets agrees across backends."""

    def _data(self):
        rng = np.random.default_rng(3)
        lg = rng.standard_normal((1, 3, 4)).astype(np.float32)  # [B, T, C]
        tgt = np.array([[1, -100, 2]], np.int64)
        return lg, tgt

    def test_BUG_probe_ce_minus100_no_mask_cross_backend(self) -> None:
        """``.ce()`` with a -100 target (no mask) must agree across backends."""
        pytest.importorskip("torch")
        import torch

        lg, tgt = self._data()
        vm = float(ProbeOutput(logits=mx.array(lg)).ce(mx.array(tgt)))
        vt = float(ProbeOutput(logits=torch.tensor(lg)).ce(torch.tensor(tgt)))
        assert vm == pytest.approx(vt, abs=1e-4), (
            f"ProbeOutput.ce diverges on a -100 target with no mask: "
            f"MLX={vm}, torch={vt}. MLX gathers index -100 (garbage); torch "
            f"applies ignore_index=-100."
        )

    def test_BUG_probe_ce_minus100_matches_masked_recompute(self) -> None:
        """``.ce()`` with a -100 target must equal an independent masked CE.

        Ground truth: cross-entropy over ONLY the non-(-100) positions, divided
        by the count of those positions.  On MLX the -100 gather corrupts the
        value; this pins the correct number.
        """
        lg, tgt = self._data()
        # numpy reference over valid (non -100) positions only.
        valid = tgt[0] != -100
        logp = lg[0] - np.log(np.exp(lg[0]).sum(axis=-1, keepdims=True))
        ce_each = -logp[np.arange(3), np.clip(tgt[0], 0, None)]
        expected = float(ce_each[valid].mean())

        v = float(ProbeOutput(logits=mx.array(lg)).ce(mx.array(tgt)))
        assert v == pytest.approx(expected, rel=1e-5), (
            f"ProbeOutput.ce(MLX) with -100 should equal the masked-mean CE "
            f"{expected}; got {v} (the -100 position leaked in)."
        )

    def test_probe_ce_with_explicit_mask_matches_backends(self) -> None:
        """Regression: with an explicit mask excluding -100, backends agree."""
        pytest.importorskip("torch")
        import torch

        lg, tgt = self._data()
        mask_m = mx.array([[True, False, True]])
        mask_t = torch.tensor([[True, False, True]])
        vm = float(ProbeOutput(logits=mx.array(lg)).ce(mx.array(tgt), mask=mask_m))
        vt = float(ProbeOutput(logits=torch.tensor(lg)).ce(torch.tensor(tgt), mask=mask_t))
        assert vm == pytest.approx(vt, abs=1e-4)


# ===========================================================================
# Regression coverage: bce / ce / mse / mae value correctness (oracle vs numpy)
# ===========================================================================


class TestProbeOutputValueOracles:
    """Probe loss values (bce, mse, mae, ce) checked against numpy oracles."""

    def test_bce_zero_logits_is_ln2(self) -> None:
        po = ProbeOutput(logits=mx.zeros((1, 3)))
        v = float(po.bce(mx.array([[1.0, 0.0, 1.0]])))
        assert v == pytest.approx(math.log(2.0), abs=1e-6)

    def test_bce_accepts_integer_labels(self) -> None:
        # Docstring/_cast_like contract: integer labels accepted without error.
        po = ProbeOutput(logits=mx.zeros((1, 3)))
        v = float(po.bce(mx.array([[1, 0, 1]])))
        assert v == pytest.approx(math.log(2.0), abs=1e-6)

    def test_bce_matches_numpy(self) -> None:
        rng = np.random.default_rng(5)
        lg = rng.standard_normal((2, 4)).astype(np.float32)
        y = rng.integers(0, 2, (2, 4)).astype(np.float32)
        # numpy BCE-with-logits = softplus(logit) - logit*y
        sp = np.log1p(np.exp(-np.abs(lg))) + np.maximum(lg, 0)
        expected = float((sp - lg * y).mean())
        v = float(ProbeOutput(logits=mx.array(lg)).bce(mx.array(y)))
        assert v == pytest.approx(expected, rel=1e-5)

    def test_mse_matches_numpy(self) -> None:
        rng = np.random.default_rng(6)
        lg = rng.standard_normal((2, 3)).astype(np.float32)
        y = rng.standard_normal((2, 3)).astype(np.float32)
        expected = float(((lg - y) ** 2).mean())
        v = float(ProbeOutput(logits=mx.array(lg)).mse(mx.array(y)))
        assert v == pytest.approx(expected, rel=1e-5)

    def test_mae_matches_numpy(self) -> None:
        rng = np.random.default_rng(7)
        lg = rng.standard_normal((2, 3)).astype(np.float32)
        y = rng.standard_normal((2, 3)).astype(np.float32)
        expected = float(np.abs(lg - y).mean())
        v = float(ProbeOutput(logits=mx.array(lg)).mae(mx.array(y)))
        assert v == pytest.approx(expected, rel=1e-5)

    def test_ce_single_class_batch(self) -> None:
        # out_features=1 (degenerate single class): softmax is always 1, CE = 0.
        po = ProbeOutput(logits=mx.zeros((1, 2, 1)))
        v = float(po.ce(mx.array([[0, 0]])))
        assert v == pytest.approx(0.0, abs=1e-6)

    def test_ce_out_of_range_raises_in_outputs(self) -> None:
        # outputs.ce DOES guard (unlike JointLoss): both backends raise.
        po = ProbeOutput(logits=mx.zeros((1, 2, 3)))
        with pytest.raises(ValueError):
            po.ce(mx.array([[0, 5]]))

    def test_mse_masked_mean_is_over_unmasked_only(self) -> None:
        # The mean must be over UNMASKED elements only, not all elements.
        lg = mx.array([[1.0, 2.0, 3.0, 4.0]])
        y = mx.array([[1.0, 0.0, 3.0, 0.0]])
        mask = mx.array([[True, False, True, False]])  # only matches -> error 0
        v = float(ProbeOutput(logits=lg).mse(y, mask=mask))
        assert v == pytest.approx(0.0, abs=1e-7)


# ===========================================================================
# JointLoss value oracles (per-probe routing, masking, lm_weight)
# ===========================================================================


class TestJointLossValues:
    """JointLoss component values: LM weighting, per-probe routing, and masking."""

    def _fixed(self, lm, probes):
        return _FixedMlx(lm, probes)

    def test_bce_value_matches_numpy_with_mask(self) -> None:
        rng = np.random.default_rng(8)
        b, t = 1, 6
        lm = np.zeros((b, t - 1, 32), np.float32)
        pr = rng.standard_normal((b, t - 1)).astype(np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
        labels = np.array([[1, 0, 1, -100, 0, 1]], np.int32)
        lengths = np.array([[1, 5]], np.int32)

        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})
        _, _, comp = loss(
            self._fixed(lm, {"probe": pr}),
            mx.array(batch),
            mx.array(labels),
            mx.array(lengths),
        )

        # numpy ground truth over valid (length window AND not -100) positions.
        shifted = labels[0, 1:].astype(np.float32)
        steps = np.arange(1, t)
        m = (steps >= 1) & (steps < 5) & (shifted != -100)
        sp = np.log1p(np.exp(-np.abs(pr[0]))) + np.maximum(pr[0], 0)
        bce = sp - pr[0] * shifted
        expected = float((bce * m).sum() / max(m.sum(), 1))
        assert float(comp["probe"]) == pytest.approx(expected, rel=1e-5)

    def test_lm_weight_zero_drops_lm_term(self) -> None:
        # With lm_weight=0, total must equal the probe-only loss (no LM contribution).
        rng = np.random.default_rng(9)
        b, t = 1, 6
        lm = rng.standard_normal((b, t - 1, 32)).astype(np.float32)  # nonzero LM logits
        pr = np.zeros((b, t - 1), np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
        labels = np.array([[1, 1, 1, 1, 1, 1]], np.int32)
        lengths = np.array([[0, 5]], np.int32)

        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})
        total, _, comp = loss(
            self._fixed(lm, {"probe": pr}),
            mx.array(batch),
            mx.array(labels),
            mx.array(lengths),
        )
        assert "lm_head" not in comp  # pure-classifier mode omits the LM key
        assert float(total) == pytest.approx(float(comp["probe"]), rel=1e-6)
        assert float(comp["probe"]) == pytest.approx(math.log(2.0), abs=1e-6)

    def test_lm_weight_scales_lm_term(self) -> None:
        # total = lm_weight * lm_ce + probe; doubling lm_weight doubles the LM part.
        rng = np.random.default_rng(10)
        b, t = 1, 6
        lm = rng.standard_normal((b, t - 1, 32)).astype(np.float32)
        pr = np.zeros((b, t - 1), np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
        labels = np.array([[1, 1, 1, 1, 1, 1]], np.int32)
        lengths = np.array([[0, 5]], np.int32)

        l1 = JointLoss(losses={"probe": "bce"})
        l2 = JointLoss(weights={"lm_head": 2.0}, losses={"probe": "bce"})
        m = self._fixed(lm, {"probe": pr})
        t1, _, c1 = l1(m, mx.array(batch), mx.array(labels), mx.array(lengths))
        t2, _, c2 = l2(m, mx.array(batch), mx.array(labels), mx.array(lengths))
        probe = float(c1["probe"])
        lm_ce = float(c1["lm_head"])
        assert float(c2["lm_head"]) == pytest.approx(lm_ce, rel=1e-6)
        assert float(t1) == pytest.approx(lm_ce + probe, rel=1e-5)
        assert float(t2) == pytest.approx(2.0 * lm_ce + probe, rel=1e-5)

    def test_per_probe_routing_uses_own_labels(self) -> None:
        # Two heads, identical +10 logits, opposite labels -> very different losses.
        class Two:
            """Fake model with two probe heads emitting identical +10 logits."""

            def __call__(self, i, mask=None):
                bb, tt = i.shape
                return mx.zeros((bb, tt, 8)), {
                    "a": mx.full((bb, tt), 10.0),
                    "b": mx.full((bb, tt), 10.0),
                }

        loss = JointLoss(weights={"lm_head": 0.0}, losses={"a": "bce", "b": "bce"})
        labels = {"a": mx.array([[1, 1, 1, 1]]), "b": mx.array([[0, 0, 0, 0]])}
        _, _, comp = loss(Two(), mx.array([[1, 2, 3, 4]]), labels, mx.array([[0, 4]]))
        sp10 = math.log1p(math.exp(-10.0))
        assert float(comp["a"]) == pytest.approx(sp10, abs=1e-5)
        assert float(comp["b"]) == pytest.approx(10.0 + sp10, abs=1e-4)

    def test_per_probe_loss_overrides(self) -> None:
        # probe a uses bce, probe b uses mse; component keys reflect each loss.
        class Two:
            """Fake model with two probe heads emitting distinct per-head logits."""

            def __call__(self, i, mask=None):
                bb, tt = i.shape
                return mx.zeros((bb, tt, 8)), {
                    "a": mx.zeros((bb, tt)),
                    "b": mx.full((bb, tt), 3.0),
                }

        loss = JointLoss(weights={"lm_head": 0.0}, losses={"a": "bce", "b": "mse"})
        labels = {"a": mx.array([[1, 1, 1]]), "b": mx.array([[1.0, 1.0, 1.0]])}
        _, _, comp = loss(Two(), mx.array([[1, 2, 3]]), labels, mx.array([[0, 3]]))
        assert set(comp) == {"a", "b"}
        assert float(comp["a"]) == pytest.approx(math.log(2.0), abs=1e-6)
        # b: (3 - 1)^2 = 4 for every valid token.
        assert float(comp["b"]) == pytest.approx(4.0, rel=1e-6)

    def test_single_probe_component_name_has_no_suffix(self) -> None:
        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})
        lm = np.zeros((1, 3, 32), np.float32)
        pr = np.zeros((1, 3), np.float32)
        _, _, comp = loss(
            _FixedMlx(lm, {"probe": pr}),
            mx.array([[1, 2, 3, 4]]),
            mx.array([[1, 1, 1, 1]]),
            mx.array([[0, 4]]),
        )
        assert "probe" in comp and ":" not in "".join(comp.keys())

    def test_all_masked_batch_is_finite_zero(self) -> None:
        # Every probe label -100 -> finite 0, never NaN, ntoks counts length window.
        lm = np.zeros((1, 5, 32), np.float32)
        pr = np.random.default_rng(0).standard_normal((1, 5)).astype(np.float32)
        labels = np.full((1, 6), -100, np.int32)
        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})
        _, _, comp = loss(
            _FixedMlx(lm, {"probe": pr}),
            mx.array([[1, 2, 3, 4, 5, 6]]),
            mx.array(labels),
            mx.array([[0, 5]]),
        )
        v = float(comp["probe"])
        assert math.isfinite(v) and v == pytest.approx(0.0, abs=1e-7)

    def test_empty_length_window_no_divide_by_zero(self) -> None:
        lm = np.zeros((1, 5, 32), np.float32)
        pr = np.zeros((1, 5), np.float32)
        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})
        _, ntoks, comp = loss(
            _FixedMlx(lm, {"probe": pr}),
            mx.array([[1, 2, 3, 4, 5, 6]]),
            mx.array([[1, 1, 1, 1, 1, 1]]),
            mx.array([[3, 3]]),  # empty window
        )
        assert float(ntoks) == 0.0
        assert math.isfinite(float(comp["probe"]))
        assert float(comp["probe"]) == pytest.approx(0.0, abs=1e-7)


# ===========================================================================
# Cross-backend parity for JointLoss values (MLX vs torch)
# ===========================================================================


class TestJointLossParity:
    """JointLoss values match between the MLX and torch backends."""

    def test_bce_parity(self) -> None:
        pytest.importorskip("torch")
        import torch

        rng = np.random.default_rng(11)
        lm = np.zeros((1, 5, 32), np.float32)
        pr = rng.standard_normal((1, 5)).astype(np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
        labels = np.array([[1, 0, 1, -100, 0, 1]], np.int32)
        lengths = np.array([[1, 5]], np.int32)
        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})
        _, mn, mc = loss(
            _FixedMlx(lm, {"probe": pr}), mx.array(batch), mx.array(labels), mx.array(lengths)
        )
        _, tn, tc = loss(
            _FixedTorch(lm, {"probe": pr}),
            torch.tensor(batch),
            torch.tensor(labels),
            torch.tensor(lengths),
        )
        assert float(mc["probe"]) == pytest.approx(float(tc["probe"]), abs=1e-5)
        assert float(mn) == float(tn)

    def test_ce_parity_in_range(self) -> None:
        # In-range CE (incl. a -100) must match across backends.
        pytest.importorskip("torch")
        import torch

        rng = np.random.default_rng(12)
        C = 4
        lm = np.zeros((1, 5, 32), np.float32)
        pr = rng.standard_normal((1, 5, C)).astype(np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
        labels = np.array([[0, 1, 2, -100, 3, 1]], np.int32)
        lengths = np.array([[0, 5]], np.int32)
        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "ce"})
        _, _, mc = loss(
            _FixedMlx(lm, {"probe": pr}), mx.array(batch), mx.array(labels), mx.array(lengths)
        )
        _, _, tc = loss(
            _FixedTorch(lm, {"probe": pr}),
            torch.tensor(batch),
            torch.tensor(labels),
            torch.tensor(lengths),
        )
        assert float(mc["probe"]) == pytest.approx(float(tc["probe"]), abs=1e-5)

    def test_huge_logits_bce_finite_and_parity(self) -> None:
        # Overflow guard: extreme logits stay finite and match across backends.
        pytest.importorskip("torch")
        import torch

        lm = np.zeros((1, 5, 32), np.float32)
        pr = np.array([[100.0, -100.0, 100.0, -100.0, 100.0]], np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 6]], np.int32)
        labels = np.array([[0, 0, 1, 0, 1, 0]], np.int32)
        lengths = np.array([[0, 5]], np.int32)
        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})
        vm = float(
            loss(
                _FixedMlx(lm, {"probe": pr}),
                mx.array(batch),
                mx.array(labels),
                mx.array(lengths),
            )[2]["probe"]
        )
        vt = float(
            loss(
                _FixedTorch(lm, {"probe": pr}),
                torch.tensor(batch),
                torch.tensor(labels),
                torch.tensor(lengths),
            )[2]["probe"]
        )
        assert math.isfinite(vm) and math.isfinite(vt)
        assert vm == pytest.approx(vt, abs=1e-4)


# ===========================================================================
# JointOutputs.lm_ce oracle + edge cases
# ===========================================================================


class TestJointOutputs:
    """Joint LM cross-entropy and token counts over the length window."""

    def test_lm_ce_matches_independent_recompute(self) -> None:
        from auto_chasm.outputs import JointOutputs

        rng = np.random.default_rng(13)
        lm = rng.standard_normal((1, 5, 32)).astype(np.float32)
        targets = np.array([[2, 3, 4, 5, 6]], np.int32)
        lengths = np.array([[1, 5]], np.int32)
        o = JointOutputs(mx.array(lm), {}, mx.array(targets), mx.array(lengths))
        v = float(o.lm_ce)

        steps = np.arange(1, 6)
        m = (steps >= 1) & (steps < 5)
        logp = lm[0] - np.log(np.exp(lm[0]).sum(axis=-1, keepdims=True))
        ce = -logp[np.arange(5), targets[0]]
        expected = float((ce * m).sum() / max(m.sum(), 1))
        assert v == pytest.approx(expected, rel=1e-5)

    def test_ntoks_counts_length_window(self) -> None:
        from auto_chasm.outputs import JointOutputs

        targets = mx.array([[2, 3, 4, 5, 6]])
        lengths = mx.array([[1, 4]])  # steps 1,2,3 -> 3 tokens
        o = JointOutputs(mx.zeros((1, 5, 8)), {}, targets, lengths)
        assert float(o.ntoks) == 3.0

    def test_lm_ce_parity(self) -> None:
        pytest.importorskip("torch")
        import torch

        from auto_chasm.outputs import JointOutputs

        rng = np.random.default_rng(14)
        lm = rng.standard_normal((1, 5, 32)).astype(np.float32)
        targets = np.array([[2, 3, 4, 5, 6]], np.int64)
        lengths = np.array([[1, 5]], np.int32)
        vm = float(JointOutputs(mx.array(lm), {}, mx.array(targets), mx.array(lengths)).lm_ce)
        vt = float(
            JointOutputs(torch.tensor(lm), {}, torch.tensor(targets), torch.tensor(lengths)).lm_ce
        )
        assert vm == pytest.approx(vt, abs=1e-4)


# ===========================================================================
# ops.py primitives (backend-agnostic math used by custom losses)
# ===========================================================================


class TestOps:
    """Numeric ops (softplus, sigmoid) for overflow safety and cross-backend parity."""

    def test_softplus_overflow_safe(self) -> None:
        from auto_chasm import ops

        # Naive log1p(exp(x)) overflows at x>=89 (float32); the safe identity must
        # return x for large x with a finite value.
        x = mx.array([89.0, 100.0, 0.0, -100.0])
        sp = ops.softplus(x)
        arr = np.array(sp.tolist())
        assert np.all(np.isfinite(arr))
        assert arr[0] == pytest.approx(89.0, abs=1e-2)
        assert arr[1] == pytest.approx(100.0, abs=1e-2)
        assert arr[2] == pytest.approx(math.log(2.0), abs=1e-4)
        assert arr[3] == pytest.approx(0.0, abs=1e-3)

    def test_sigmoid_matches_numpy(self) -> None:
        from auto_chasm import ops

        x = np.array([0.0, 2.0, -3.0, 50.0], np.float32)
        v = np.array(ops.sigmoid(mx.array(x)).tolist())
        expected = 1.0 / (1.0 + np.exp(-x))
        assert np.allclose(v, expected, atol=1e-5)

    def test_ops_parity_mlx_torch(self) -> None:
        pytest.importorskip("torch")
        import torch

        from auto_chasm import ops

        x = np.array([0.3, 1.5, -2.0, 4.0], np.float32)
        for name in ("exp", "log", "sqrt", "sigmoid", "softplus"):
            fn = getattr(ops, name)
            xm = mx.array(np.abs(x) if name in ("log", "sqrt") else x)
            xt = torch.tensor(np.abs(x) if name in ("log", "sqrt") else x)
            vm = np.array(fn(xm).tolist())
            vt = fn(xt).detach().numpy()
            assert np.allclose(vm, vt, atol=1e-4), f"ops.{name} diverges: {vm} vs {vt}"


class TestMulticlassCeTrainingTraced:
    """Multi-class CE training must run through the real (traced) Trainer.

    Regression: the class-index bounds check calls ``.item()``, which MLX
    forbids inside ``value_and_grad`` (``[eval] Attempting to eval an array
    during function transformations``). That broke ALL valid multi-class CE
    training on MLX. The eager oracle tests above call the loss directly and so
    missed it; this trains through the real ``Trainer`` to exercise the traced
    path.
    """

    def test_multiclass_ce_trains_through_trainer_mlx(self) -> None:
        from auto_chasm import Model, ProbeConfig, Trainer

        mx.random.seed(0)

        class _Tiny(nn.Module):
            """Tiny MLX LM with a per-token hidden state for CE-probe training."""

            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(32, 16)
                self.layers = [nn.Linear(16, 16) for _ in range(2)]
                self.output_proj = nn.Linear(16, 32)

            def __call__(self, x, **kw):
                h = self.embedding(x)
                for layer in self.layers:
                    h = nn.gelu(layer(h))
                return self.output_proj(h)

        class _Cfg:
            """Config stub exposing hidden size and layer count."""

            hidden_size = 16
            num_hidden_layers = 2

        base = _Tiny()
        base.config = _Cfg()
        model = Model(base, None, "mlx")
        model.attach_probe(ProbeConfig(name="kind", layers=[-1], module_config={"out_features": 3}))
        data = [
            {"tokens": [1, 2, 3, 4], "labels": [0, 1, 2, 1]},
            {"tokens": [5, 6, 7, 8], "labels": [2, 0, 1, 0]},
        ]
        # Must NOT raise "[eval] Attempting to eval ..." from the bounds check,
        # and must produce a finite loss history.
        result = Trainer(
            model, JointLoss(losses={"kind": "ce"}), num_iters=3, batch_size=2, verbose=False
        ).train(data)
        hist = result["history"] if isinstance(result, dict) else result
        losses = [e.train_loss for e in hist.entries]
        assert losses and all(math.isfinite(x) for x in losses)

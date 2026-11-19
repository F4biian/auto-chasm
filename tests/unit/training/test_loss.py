"""Tests for loss module — JointLoss class, custom loss, MSE, pure classifier."""

from __future__ import annotations

import mlx.core as mx
import pytest

from auto_chasm.trainers.loss import JointLoss, _canonical_loss_name


class TestCanonicalLossName:
    """Tests for the built-in loss-name resolution."""

    def test_bce(self) -> None:
        assert _canonical_loss_name("bce") == "bce"

    def test_mse(self) -> None:
        assert _canonical_loss_name("mse") == "mse"

    def test_ce_and_mae(self) -> None:
        assert _canonical_loss_name("ce") == "ce"
        assert _canonical_loss_name("mae") == "mae"

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown loss"):
            _canonical_loss_name("l1")


class TestJointLoss:
    """Tests for the JointLoss class (backend-agnostic)."""

    @pytest.fixture
    def loss(self):
        # A single-tensor probe return is normalized to the name "probe";
        # default weights (all 1.0) include the LM term + the probe.
        return JointLoss(losses={"probe": "bce"})

    def test_returns_triple(self, loss):
        class FakeModel:
            """Test helper."""

            def __call__(self, inputs):
                b, t = inputs.shape
                return (mx.zeros((b, t, 32)), mx.zeros((b, t)))

        batch = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[0, 0, 1, 0, 0]])
        lengths = mx.array([[0, 5]])

        total, ntoks, components = loss(FakeModel(), batch, labels, lengths)
        assert total.ndim == 0
        assert float(ntoks) > 0
        assert "lm_head" in components
        assert "probe" in components

    def test_pure_classifier_nolm(self):
        # weights={"lm_head": 0.0} reproduces the old lm_weight=0.0 (no LM term).
        loss = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})

        class FakeModel:
            """Test helper."""

            def __call__(self, inputs):
                b, t = inputs.shape
                return (mx.zeros((b, t, 32)), mx.ones((b, t)))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0, 1, 0]])
        lengths = mx.array([[0, 3]])

        total, ntoks, components = loss(FakeModel(), batch, labels, lengths)
        assert "lm_head" not in components
        assert "probe" in components

    def test_mse_loss(self):
        loss = JointLoss(losses={"probe": "mse"})

        class FakeModel:
            """Test helper."""

            def __call__(self, inputs):
                b, t = inputs.shape
                return (mx.zeros((b, t, 32)), mx.ones((b, t)))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0, 1, 0]])
        lengths = mx.array([[0, 3]])

        total, ntoks, components = loss(FakeModel(), batch, labels, lengths)
        assert "probe" in components

    def test_custom_loss(self):
        def my_custom(logits, targets, mask):
            return mx.sum(logits) * 0.0

        loss = JointLoss(losses={"probe": my_custom})

        class FakeModel:
            """Test helper."""

            def __call__(self, inputs):
                b, t = inputs.shape
                return (mx.zeros((b, t, 32)), mx.ones((b, t)))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0, 1, 0]])
        lengths = mx.array([[0, 3]])

        total, ntoks, components = loss(FakeModel(), batch, labels, lengths)
        assert "probe" in components

    def test_zero_probe_weight(self):
        # Drop the probe term (weight 0.0) -> LM-only training.
        loss = JointLoss(weights={"probe": 0.0})

        class FakeModel:
            """Test helper."""

            def __call__(self, inputs):
                b, t = inputs.shape
                return (mx.zeros((b, t, 32)), mx.ones((b, t)))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0, 1, 0]])
        lengths = mx.array([[0, 3]])

        total, ntoks, components = loss(FakeModel(), batch, labels, lengths)
        assert "probe" not in components

    def test_null_probe_logits(self):
        loss = JointLoss()

        class FakeModel:
            """Test helper."""

            def __call__(self, inputs):
                b, t = inputs.shape
                return (mx.zeros((b, t, 32)), None)

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0, 1, 0]])
        lengths = mx.array([[0, 3]])

        total, ntoks, components = loss(FakeModel(), batch, labels, lengths)
        assert "probe" not in components

    def test_is_callable(self) -> None:
        loss = JointLoss()
        assert callable(loss)

    def test_error_on_unknown_probe_loss(self) -> None:
        with pytest.raises(ValueError, match="Unknown loss"):
            JointLoss(losses={"probe": "l1"})

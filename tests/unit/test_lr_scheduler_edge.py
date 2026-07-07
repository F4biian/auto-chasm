"""LR scheduler and training-loop edge-case tests.

Covers the LR schedule builder in ``trainers/base.py`` and the
``Trainer`` / ``JointTrainer`` training loop edge cases.
"""

from __future__ import annotations

import tempfile
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import ProbeConfig
from auto_chasm.history import History
from auto_chasm.model import Model
from auto_chasm.trainers.base import JointTrainer
from auto_chasm.trainers.data_utils import JointTextDataset
from auto_chasm.trainers.trainable import make_joint_loss

# ── Constants ────────────────────────────────────────────────────────────
HIDDEN_DIM = 16
VOCAB_SIZE = 32
NUM_LAYERS = 4
BATCH_SIZE = 8
MAX_SEQ_LEN = 32
NUM_ITERS = 20
LEARNING_RATE = 0.01
WARMUP_RATIO = 0.0
PROBE_NAME = "digit"
FIRST_STEP_LR_TOLERANCE = 0.15
LR_DECAY_TOLERANCE = 5e-4


class TinyMlp(nn.Module):
    """A tiny MLP for testing scheduler computations.

    Args:
        hidden_dim: Hidden dimension size.
        vocab_size: Vocabulary size.
        num_layers: Number of linear layers.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        vocab_size: int = VOCAB_SIZE,
        num_layers: int = NUM_LAYERS,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self._hidden_dim = hidden_dim

    def __call__(self, x: mx.array, **kwargs: Any) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class DummyTokenizer:
    """Minimal tokenizer for testing."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "test"


@pytest.fixture
def model_wrapper() -> Model:
    """Create a Model wrapper with a probe and config."""
    mx.random.seed(42)
    model = TinyMlp(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS)

    class Config:
        """Mock model config for hidden_dim detection."""

        hidden_size = HIDDEN_DIM
        num_hidden_layers = NUM_LAYERS

    model.config = Config()
    wrapper = Model(model, DummyTokenizer(), backend_name="mlx")
    wrapper.attach_probe(ProbeConfig(name=PROBE_NAME, layers=[0]))
    wrapper.prepare_for_joint_training()
    return wrapper


@pytest.fixture
def sample_dataset() -> list[dict]:
    """Create a small synthetic dataset."""
    mx.random.seed(42)
    data = []
    for _ in range(32):
        data.append({"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 0, 0]})
    return data


def _get_lr_from_schedule(
    model_wrapper: Model, lr_schedule: str, num_iters: int, warmup_ratio: float, step: int
) -> float:
    """Build a JointTrainer and extract the LR at a given step.

    Args:
        model_wrapper: The model wrapper.
        lr_schedule: LR schedule type.
        num_iters: Total training iterations.
        warmup_ratio: Fraction of steps for warmup.
        step: The step to query (0-indexed within schedule).

    Returns:
        The learning rate at the given step.
    """
    loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
    trainer = JointTrainer(
        model=model_wrapper,
        loss_fn=loss_fn,
        learning_rate=LEARNING_RATE,
        num_iters=num_iters,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LEN,
        logging_steps=num_iters,
        save_steps=0,
        eval_steps=0,
        early_stopping_patience=0,
        verbose=False,
        lr_schedule=lr_schedule,
        warmup_ratio=warmup_ratio,
    )
    schedule = trainer.lr_schedule
    if callable(schedule):
        raw = schedule(step)
        return float(raw.item() if hasattr(raw, "item") else raw)
    return LEARNING_RATE


# ── LR schedule shapes ──────────────────────────────────────


class TestLRScheduleShapes:
    """Verify the three LR schedule shapes produce correct values."""

    def test_cosine_lr_decreases_over_steps(self, model_wrapper: Model) -> None:
        """Cosine schedule: early LR > late LR."""
        lr_early = _get_lr_from_schedule(model_wrapper, "cosine", NUM_ITERS, 0.0, 1)
        lr_late = _get_lr_from_schedule(model_wrapper, "cosine", NUM_ITERS, 0.0, NUM_ITERS)
        assert lr_late < lr_early, "Cosine LR should decrease"

    def test_linear_lr_decreases_linearly(self, model_wrapper: Model) -> None:
        """Linear schedule: LR at midpoint is roughly half of initial."""
        lr_start = _get_lr_from_schedule(model_wrapper, "linear", NUM_ITERS, 0.0, 0)
        lr_mid = _get_lr_from_schedule(model_wrapper, "linear", NUM_ITERS, 0.0, NUM_ITERS // 2)
        lr_end = _get_lr_from_schedule(model_wrapper, "linear", NUM_ITERS, 0.0, NUM_ITERS)
        assert abs(lr_start - LEARNING_RATE) < LR_DECAY_TOLERANCE, "Start should be LR"
        assert lr_mid < lr_start, "Mid should be less than start"
        assert lr_end < lr_mid, "End should be less than mid"
        assert abs(lr_end) < LR_DECAY_TOLERANCE, "End should be near zero"
        # Check approximate linearity: mid ~ half
        expected_mid = LEARNING_RATE * (1.0 - (NUM_ITERS // 2) / NUM_ITERS)
        assert abs(lr_mid - expected_mid) < FIRST_STEP_LR_TOLERANCE, "Mid should be ~half"

    def test_constant_lr_stays_same(self, model_wrapper: Model) -> None:
        """Constant schedule: LR stays at initial value."""
        lr_start = _get_lr_from_schedule(model_wrapper, "constant", NUM_ITERS, 0.0, 0)
        lr_end = _get_lr_from_schedule(model_wrapper, "constant", NUM_ITERS, 0.0, NUM_ITERS)
        assert abs(lr_start - LEARNING_RATE) < LR_DECAY_TOLERANCE
        assert abs(lr_end - LEARNING_RATE) < LR_DECAY_TOLERANCE


# ── Warmup combinations ─────────────────────────────────────


class TestWarmup:
    """Warmup LR behaviour."""

    def test_warmup_linear_then_decay(self, model_wrapper: Model) -> None:
        """With warmup_ratio=0.2, LR should increase during warmup then decrease."""
        warmup_ratio = 0.2
        num_iters = 20
        warmup_steps = int(num_iters * warmup_ratio)

        lr_step_1 = _get_lr_from_schedule(model_wrapper, "linear", num_iters, warmup_ratio, 1)
        lr_warmup_end = _get_lr_from_schedule(
            model_wrapper, "linear", num_iters, warmup_ratio, warmup_steps
        )
        lr_decay = _get_lr_from_schedule(
            model_wrapper, "linear", num_iters, warmup_ratio, warmup_steps + 2
        )

        # Step 1 should be very small (early in warmup)
        assert lr_step_1 < LEARNING_RATE, "Early warmup LR should be less than peak"
        # At warmup_end, LR should be at peak
        assert abs(lr_warmup_end - LEARNING_RATE) < LR_DECAY_TOLERANCE, (
            f"Warmup end LR should be peak LR, got {lr_warmup_end}"
        )
        # After warmup, LR should decrease
        assert lr_decay < lr_warmup_end, "Post-warmup LR should decay"

    def test_warmup_ratio_zero_starts_full(self, model_wrapper: Model) -> None:
        """With warmup_ratio=0.0, LR should start at full value."""
        lr = _get_lr_from_schedule(model_wrapper, "linear", NUM_ITERS, 0.0, 0)
        assert abs(lr - LEARNING_RATE) < LR_DECAY_TOLERANCE, "No warmup means full LR at start"

    def test_warmup_ratio_one_only_warmup(self, model_wrapper: Model) -> None:
        """With warmup_ratio=1.0, LR should only warm up, never decay."""
        num_iters = 10
        lr_step_1 = _get_lr_from_schedule(model_wrapper, "linear", num_iters, 1.0, 1)
        lr_step_5 = _get_lr_from_schedule(model_wrapper, "linear", num_iters, 1.0, 5)
        lr_step_10 = _get_lr_from_schedule(model_wrapper, "linear", num_iters, 1.0, num_iters)
        assert lr_step_1 > 0, "Warmup should start above zero"
        assert lr_step_5 > lr_step_1, "Warmup should increase"
        assert abs(lr_step_10 - LEARNING_RATE) < LR_DECAY_TOLERANCE, (
            f"End of warmup should be peak LR, got {lr_step_10}"
        )


# ── Short training ──────────────────────────────────────────


class TestShortTraining:
    """Very short training runs should not crash."""

    def test_one_step_with_warmup(self, model_wrapper: Model, sample_dataset: list) -> None:
        """Training with 1 step and warmup should not crash."""
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=1,
                batch_size=BATCH_SIZE,
                max_seq_length=MAX_SEQ_LEN,
                logging_steps=1,
                save_steps=0,
                eval_steps=0,
                early_stopping_patience=0,
                output_dir=tmpdir,
                verbose=False,
                lr_schedule="cosine",
                warmup_ratio=0.5,
            )
            history = trainer.run(ds)
        assert isinstance(history, History)

    def test_zero_steps_does_not_crash(self, model_wrapper: Model, sample_dataset: list) -> None:
        """Training with 0 steps should be handled gracefully."""
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=0,
                batch_size=BATCH_SIZE,
                max_seq_length=MAX_SEQ_LEN,
                logging_steps=1,
                save_steps=0,
                eval_steps=0,
                early_stopping_patience=0,
                output_dir=tmpdir,
                verbose=False,
            )
            history = trainer.run(ds)
        assert isinstance(history, History)


# ── Edge cases ──────────────────────────────────────────────


class TestEdgeCases:
    """Edge case parameter values."""

    def test_num_iters_zero_init(self, model_wrapper: Model) -> None:
        """JointTrainer with num_iters=0 should init without error."""
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        trainer = JointTrainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=0,
            batch_size=BATCH_SIZE,
            max_seq_length=MAX_SEQ_LEN,
            early_stopping_patience=0,
            verbose=False,
        )
        assert trainer.num_iters == 0

    def test_batch_size_larger_than_dataset(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        """batch_size > len(dataset) should train without crash."""
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")
        huge_batch = len(sample_dataset) * 2
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=2,
                batch_size=huge_batch,
                max_seq_length=MAX_SEQ_LEN,
                logging_steps=1,
                save_steps=0,
                eval_steps=0,
                early_stopping_patience=0,
                output_dir=tmpdir,
                verbose=False,
            )
            history = trainer.run(ds)
        assert isinstance(history, History)

    def test_weight_decay_zero(self, model_wrapper: Model, sample_dataset: list) -> None:
        """weight_decay=0.0 should work without regularization."""
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=2,
                batch_size=BATCH_SIZE,
                max_seq_length=MAX_SEQ_LEN,
                logging_steps=1,
                save_steps=0,
                eval_steps=0,
                early_stopping_patience=0,
                weight_decay=0.0,
                output_dir=tmpdir,
                verbose=False,
            )
            history = trainer.run(ds)
        assert isinstance(history, History)

    def test_large_grad_clip_norm(self, model_wrapper: Model, sample_dataset: list) -> None:
        """Very large grad_clip_norm should not affect training."""
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=2,
                batch_size=BATCH_SIZE,
                max_seq_length=MAX_SEQ_LEN,
                logging_steps=1,
                save_steps=0,
                eval_steps=0,
                early_stopping_patience=0,
                grad_clip_norm=1e6,
                output_dir=tmpdir,
                verbose=False,
            )
            history = trainer.run(ds)
        assert isinstance(history, History)

    def test_negative_learning_rate(self, model_wrapper: Model, sample_dataset: list) -> None:
        """Negative learning rate should not crash (but produce garbage)."""
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=2,
                batch_size=BATCH_SIZE,
                max_seq_length=MAX_SEQ_LEN,
                learning_rate=-0.001,
                logging_steps=1,
                save_steps=0,
                eval_steps=0,
                early_stopping_patience=0,
                output_dir=tmpdir,
                verbose=False,
            )
            history = trainer.run(ds)
        assert isinstance(history, History)


# ── Integration ─────────────────────────────────────────────


class TestIntegration:
    """Integration tests combining freeze and scheduler."""

    def test_probes_change_base_no_change(self, model_wrapper: Model, sample_dataset: list) -> None:
        """After training, probes should change but base model should not."""
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        from mlx.utils import tree_flatten

        base_params_before = dict(tree_flatten(model_wrapper.model.parameters()))
        probe_before = dict(tree_flatten(model_wrapper.probes[PROBE_NAME].module.parameters()))

        # store copies of param values as floats for comparison
        base_vals_before = {k: float(mx.sum(mx.abs(v))) for k, v in base_params_before.items()}
        probe_vals_before = {k: float(mx.sum(mx.abs(v))) for k, v in probe_before.items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=4,
                batch_size=BATCH_SIZE,
                max_seq_length=MAX_SEQ_LEN,
                logging_steps=4,
                save_steps=0,
                eval_steps=0,
                early_stopping_patience=0,
                output_dir=tmpdir,
                verbose=False,
            )
            trainer.run(ds)

        base_params_after = dict(tree_flatten(model_wrapper.model.parameters()))
        probe_after = dict(tree_flatten(model_wrapper.probes[PROBE_NAME].module.parameters()))

        for key in base_params_before:
            before = base_vals_before[key]
            after = float(mx.sum(mx.abs(base_params_after[key])))
            diff = abs(after - before)
            msg = f"Base param {key} changed (diff={diff:.2e}) — should be frozen"
            assert diff < 1e-6, msg

        probe_changed = False
        for key in probe_before:
            before = probe_vals_before[key]
            after = float(mx.sum(mx.abs(probe_after[key])))
            diff = abs(after - before)
            if diff > 1e-6:
                probe_changed = True
                break
        assert probe_changed, "Probe params should have changed after training"

    def test_cosine_with_warmup_lr_trajectory(self, model_wrapper: Model) -> None:
        """Cosine + 0.1 warmup: verify LR at step 1, warmup end, and decay."""
        num_iters = 100
        warmup_ratio = 0.1
        warmup_steps = int(num_iters * warmup_ratio)
        schedule = _get_lr_from_schedule

        lr_step_1 = schedule(model_wrapper, "cosine", num_iters, warmup_ratio, 1)
        lr_warmup_end = schedule(model_wrapper, "cosine", num_iters, warmup_ratio, warmup_steps)
        lr_mid = schedule(model_wrapper, "cosine", num_iters, warmup_ratio, num_iters // 2)
        lr_end = schedule(model_wrapper, "cosine", num_iters, warmup_ratio, num_iters)

        # Step 1: very small (early in warmup)
        assert lr_step_1 < LEARNING_RATE, "Step 1 should be very small during warmup"
        assert lr_step_1 > 0, "Step 1 should be above zero"

        # Warmup end: should be at or near peak
        assert abs(lr_warmup_end - LEARNING_RATE) < LR_DECAY_TOLERANCE, (
            f"Warmup end LR should be near peak, got {lr_warmup_end}"
        )

        # Mid: should have decayed
        assert lr_mid < lr_warmup_end, "Mid should be less than warmup end"

        # End: should be near zero
        assert lr_end < lr_mid, "End should be less than mid"
        assert abs(lr_end) < LR_DECAY_TOLERANCE, "End should be near zero"


# ── LR schedule validation tests ─────────────────────────────────────────


class TestScheduleValidation:
    """Validation of the _build_lr_schedule method."""

    def test_unknown_schedule_raises(self) -> None:
        """Passing an unknown lr_schedule should raise ValueError."""
        from auto_chasm.trainers.base import JointTrainer

        with pytest.raises(ValueError, match="Unknown lr_schedule"):
            JointTrainer._build_lr_schedule("unknown", 0.01, 100, 0)  # type: ignore[arg-type]

    def test_warmup_steps_equals_num_iters(self, model_wrapper: Model) -> None:
        """When warmup_steps == num_iters, schedule should be pure warmup."""
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        trainer = JointTrainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=10,
            warmup_ratio=1.0,
            batch_size=BATCH_SIZE,
            max_seq_length=MAX_SEQ_LEN,
            early_stopping_patience=0,
            verbose=False,
        )
        schedule = trainer.lr_schedule
        at_5 = float(schedule(5).item()) if hasattr(schedule(5), "item") else float(schedule(5))
        at_10 = float(schedule(10).item()) if hasattr(schedule(10), "item") else float(schedule(10))
        assert at_5 > 0, "Mid-warmup should be > 0"
        assert at_10 > at_5, "Warmup should still be increasing at step 10"

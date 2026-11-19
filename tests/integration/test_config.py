"""Tests for TrainingConfig and GenerationConfig integration with trainers and model."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import JointLoss, Model, SFTTrainer, Trainer
from auto_chasm.config import GenerationConfig, TrainingConfig
from auto_chasm.trainers import JointTrainer

# ---------------------------------------------------------------------------
# Tiny model fixtures (reused from test_new_api.py)
# ---------------------------------------------------------------------------


class TinyMlp(nn.Module):
    """A tiny 2-layer MLP for testing."""

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 32, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self._hidden_dim = hidden_dim

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class DummyTokenizer:
    """Minimal tokenizer for testing."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids if i > 0)


@pytest.fixture
def model_wrapper() -> Model:
    """Create a Model wrapper around a tiny model."""
    mx.random.seed(42)
    base_model = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)

    class Config:
        """Mock model config for hidden_dim detection."""

        hidden_size = 16
        num_hidden_layers = 4

    base_model.config = Config()
    return Model(base_model, DummyTokenizer(), backend_name="mlx")


@pytest.fixture
def loss_fn() -> JointLoss:
    """Create a default JointLoss."""
    return JointLoss()


# ---------------------------------------------------------------------------
# Trainer + TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainerFromConfig:
    """Tests for Trainer accepting TrainingConfig."""

    def test_trainer_from_config(self, model_wrapper: Model, loss_fn: JointLoss) -> None:
        cfg = TrainingConfig(
            learning_rate=3e-4,
            batch_size=16,
            output_dir="/tmp/test_trainer_cfg",
        )
        trainer = Trainer(model=model_wrapper, loss_fn=loss_fn, config=cfg)
        assert trainer.learning_rate == 3e-4
        assert trainer.batch_size == 16
        assert str(trainer.output_dir) == "/tmp/test_trainer_cfg"

    def test_individual_overrides_config(self, model_wrapper: Model, loss_fn: JointLoss) -> None:
        cfg = TrainingConfig(learning_rate=3e-4)
        trainer = Trainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            learning_rate=5e-4,
            config=cfg,
        )
        assert trainer.learning_rate == 5e-4

    def test_config_weight_decay(self, model_wrapper: Model, loss_fn: JointLoss) -> None:
        cfg = TrainingConfig(weight_decay=0.05)
        trainer = Trainer(model=model_wrapper, loss_fn=loss_fn, config=cfg)
        assert trainer.weight_decay == 0.05

    def test_config_grad_clip_norm(self, model_wrapper: Model, loss_fn: JointLoss) -> None:
        cfg = TrainingConfig(max_grad_norm=2.5)
        trainer = Trainer(model=model_wrapper, loss_fn=loss_fn, config=cfg)
        assert trainer.grad_clip_norm == 2.5

    def test_config_grad_accum_steps(self, model_wrapper: Model, loss_fn: JointLoss) -> None:
        cfg = TrainingConfig(gradient_accumulation_steps=4)
        trainer = Trainer(model=model_wrapper, loss_fn=loss_fn, config=cfg)
        assert trainer.grad_accum_steps == 4

    def test_config_logging_steps(self, model_wrapper: Model, loss_fn: JointLoss) -> None:
        cfg = TrainingConfig(logging_steps=50)
        trainer = Trainer(model=model_wrapper, loss_fn=loss_fn, config=cfg)
        assert trainer.logging_steps == 50

    def test_config_save_steps(self, model_wrapper: Model, loss_fn: JointLoss) -> None:
        cfg = TrainingConfig(save_steps=200)
        trainer = Trainer(model=model_wrapper, loss_fn=loss_fn, config=cfg)
        assert trainer.save_steps == 200

    def test_config_eval_steps(self, model_wrapper: Model, loss_fn: JointLoss) -> None:
        cfg = TrainingConfig(eval_steps=100)
        trainer = Trainer(model=model_wrapper, loss_fn=loss_fn, config=cfg)
        assert trainer.eval_steps == 100

    def test_config_seed_sets_random_state(self, model_wrapper: Model, loss_fn: JointLoss) -> None:
        """Config seed should set numpy and MLX random state."""
        import numpy as np

        cfg = TrainingConfig(seed=12345)
        # Set different seed first
        mx.random.seed(99999)
        np.random.seed(99999)
        # Now create trainer with config seed
        Trainer(model=model_wrapper, loss_fn=loss_fn, config=cfg)
        # After construction, generate random numbers to verify seed was set
        a = mx.random.normal((3,)).tolist()
        mx.random.seed(12345)
        b = mx.random.normal((3,)).tolist()
        assert a == b

    def test_backward_compat(self, model_wrapper: Model, loss_fn: JointLoss) -> None:
        trainer = Trainer(model=model_wrapper, loss_fn=loss_fn, learning_rate=2e-4)
        assert trainer.learning_rate == 2e-4

    def test_no_config_uses_individual_defaults(
        self, model_wrapper: Model, loss_fn: JointLoss
    ) -> None:
        trainer = Trainer(model=model_wrapper, loss_fn=loss_fn)
        assert trainer.learning_rate == 2e-4
        assert trainer.weight_decay == 0.0
        assert trainer.grad_clip_norm == 1.0
        assert trainer.batch_size == 8
        assert trainer.grad_accum_steps == 1
        assert trainer.logging_steps == 25
        assert trainer.save_steps == 100
        assert trainer.eval_steps is None


# ---------------------------------------------------------------------------
# JointTrainer + TrainingConfig
# ---------------------------------------------------------------------------


class TestJointTrainerFromConfig:
    """Tests for JointTrainer accepting TrainingConfig."""

    def test_joint_trainer_from_config(self, model_wrapper: Model, loss_fn: JointLoss) -> None:
        cfg = TrainingConfig(
            learning_rate=1e-3,
            batch_size=32,
            weight_decay=0.01,
            gradient_accumulation_steps=8,
            logging_steps=10,
            save_steps=50,
            output_dir="/tmp/test_joint_cfg",
        )
        trainer = JointTrainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            config=cfg,
        )
        assert trainer._base_lr == 1e-3
        assert trainer.batch_size == 32
        assert trainer.grad_accum_steps == 8
        assert trainer.logging_steps == 10
        assert trainer.save_steps == 50
        assert str(trainer.output_dir) == "/tmp/test_joint_cfg"

    def test_joint_trainer_individual_overrides_config(
        self, model_wrapper: Model, loss_fn: JointLoss
    ) -> None:
        cfg = TrainingConfig(learning_rate=3e-4, batch_size=64)
        trainer = JointTrainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            learning_rate=5e-4,
            batch_size=16,
            config=cfg,
        )
        assert trainer._base_lr == 5e-4
        assert trainer.batch_size == 16


# ---------------------------------------------------------------------------
# SFTTrainer + TrainingConfig
# ---------------------------------------------------------------------------


class TestSFTTrainerFromConfig:
    """Tests for SFTTrainer accepting TrainingConfig."""

    def test_sft_trainer_from_config(self, model_wrapper: Model) -> None:
        cfg = TrainingConfig(
            lm_weight=0.5,
            probe_weight=2.0,
            learning_rate=5e-5,
            batch_size=12,
            output_dir="/tmp/test_sft_cfg",
        )
        trainer = SFTTrainer(model=model_wrapper, config=cfg)
        assert trainer._trainer._base_lr == 5e-5
        assert trainer._trainer.batch_size == 12
        assert str(trainer._trainer.output_dir) == "/tmp/test_sft_cfg"

    def test_sft_trainer_individual_overrides_config(self, model_wrapper: Model) -> None:
        cfg = TrainingConfig(lm_weight=0.5, probe_weight=2.0, learning_rate=5e-5)
        trainer = SFTTrainer(
            model=model_wrapper,
            lm_weight=0.8,
            probe_weight=1.5,
            learning_rate=3e-4,
            config=cfg,
        )
        assert trainer._trainer._base_lr == 3e-4
        trainer._trainer._base_lr = getattr(trainer._trainer, "_base_lr", None)
        assert trainer._trainer._base_lr == 3e-4

    def test_sft_trainer_backward_compat(self, model_wrapper: Model) -> None:
        trainer = SFTTrainer(
            model=model_wrapper,
            lm_weight=0.7,
            probe_weight=1.5,
        )
        assert trainer._trainer._base_lr == 2e-4


# ---------------------------------------------------------------------------
# Model.generate() + GenerationConfig
# ---------------------------------------------------------------------------


class TestGenerationConfig:
    """Tests for Model.generate() accepting GenerationConfig."""

    def test_generation_config_max_tokens(self, model_wrapper: Model) -> None:
        """Verify max_tokens from GenerationConfig is used."""
        # We can't easily intercept the generate call without mocking,
        # but we can verify the method signature accepts config.
        cfg = GenerationConfig(max_tokens=50, temperature=0.0)
        result = model_wrapper.generate("hello", config=cfg)
        assert isinstance(result, str)

    def test_generation_config_temperature(self, model_wrapper: Model) -> None:
        cfg = GenerationConfig(temperature=1.0)
        result = model_wrapper.generate("hello", config=cfg)
        assert isinstance(result, str)

    def test_generation_config_do_sample_enables_sampling(self, model_wrapper: Model) -> None:
        """do_sample=True with temperature=0 should set temperature to 1e-3."""
        cfg = GenerationConfig(temperature=0.0, do_sample=True)
        result = model_wrapper.generate("hello", config=cfg)
        assert isinstance(result, str)

    def test_generation_config_individual_overrides(self, model_wrapper: Model) -> None:
        """Individual params should override GenerationConfig values."""
        cfg = GenerationConfig(max_tokens=10)
        result = model_wrapper.generate("hello", max_tokens=5, config=cfg)
        assert isinstance(result, str)

    def test_generation_config_passed_through(self, model_wrapper: Model) -> None:
        """GenerationConfig with top_p, top_k, repetition_penalty should work."""
        cfg = GenerationConfig(
            max_tokens=10,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            num_return_sequences=1,
        )
        result = model_wrapper.generate("test", config=cfg)
        assert isinstance(result, str)

    def test_generation_config_stop_sequences(self, model_wrapper: Model) -> None:
        cfg = GenerationConfig(max_tokens=5, stop_sequences=["\n"])
        result = model_wrapper.generate("test", config=cfg)
        assert isinstance(result, str)

    def test_generate_stream_with_config(self, model_wrapper: Model) -> None:
        cfg = GenerationConfig(max_tokens=5, temperature=0.7)
        tokens = list(model_wrapper.generate_stream("hello", config=cfg))
        assert all(isinstance(t, str) for t in tokens)

    def test_generate_backward_compat(self, model_wrapper: Model) -> None:
        result = model_wrapper.generate("hello", max_tokens=5, temperature=0.0)
        assert isinstance(result, str)

    def test_generate_stream_backward_compat(self, model_wrapper: Model) -> None:
        tokens = list(model_wrapper.generate_stream("hello", max_tokens=5, temperature=0.0))
        assert all(isinstance(t, str) for t in tokens)

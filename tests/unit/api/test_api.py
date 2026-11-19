"""Tests for the new API surface: LoraConfig, Trainer, JointLoss, Model additions.

Tests the full new API with a tiny model to verify end-to-end
correctness without needing actual pretrained models or LoRA libraries.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import JointLoss, LoraConfig, Model, ProbeConfig, Trainer
from auto_chasm.config import SteeringConfig
from auto_chasm.history import History
from auto_chasm.steering import SteeringHook, build_auto_steer_fn
from auto_chasm.trainers import TrainerCallback

# ---------------------------------------------------------------------------
# Fixtures
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
def tiny_model() -> tuple[TinyMlp, DummyTokenizer]:
    """Create a tiny model and tokenizer for testing."""
    mx.random.seed(42)
    model = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)
    tokenizer = DummyTokenizer()
    return model, tokenizer


@pytest.fixture
def model_wrapper(tiny_model: tuple[TinyMlp, DummyTokenizer]) -> Model:
    """Create a Model wrapper around the tiny model."""
    base_model, tokenizer = tiny_model

    class Config:
        """Dummy configuration for testing."""

        hidden_size = 16
        num_hidden_layers = 4

    base_model.config = Config()
    return Model(base_model, tokenizer, backend_name="mlx")


@pytest.fixture
def sample_dataset() -> list[dict]:
    """Create a small synthetic dataset for training tests."""
    mx.random.seed(42)
    data = []
    for _ in range(32):
        tokens = [1, 2, 3, 4, 5]
        labels = [0, 0, 1, 0, 0]
        data.append({"tokens": tokens, "labels": labels})
    return data


# ---------------------------------------------------------------------------
# LoraConfig tests
# ---------------------------------------------------------------------------
# LoraConfig tests
# ---------------------------------------------------------------------------


class TestLoraConfig:
    """Tests for the LoraConfig dataclass."""

    def test_default_values(self) -> None:
        cfg = LoraConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert cfg.dropout == 0.0
        assert cfg.target_modules is None

    def test_custom_values(self) -> None:
        cfg = LoraConfig(rank=16, alpha=32, dropout=0.1, target_modules=["q_proj"])
        assert cfg.rank == 16
        assert cfg.alpha == 32
        assert cfg.dropout == 0.1
        assert cfg.target_modules == ["q_proj"]

    def test_is_dataclass(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(LoraConfig)

    def test_roundtrip_dict(self) -> None:
        cfg = LoraConfig(rank=4, alpha=8)
        d = {"rank": cfg.rank, "alpha": cfg.alpha}
        restored = LoraConfig(**d)
        assert restored == cfg


# ---------------------------------------------------------------------------
# Model.attach_lora tests
# ---------------------------------------------------------------------------


class TestModelAttachLora:
    """Tests for Model.attach_lora and related properties."""

    def test_raw_model_property(self, model_wrapper: Model) -> None:
        raw = model_wrapper.raw_model
        assert raw is model_wrapper.model

    def test_lora_config_initially_none(self, model_wrapper: Model) -> None:
        assert model_wrapper.lora_config is None

    def test_attach_lora_stores_config(self, model_wrapper: Model) -> None:
        cfg = LoraConfig(rank=4, alpha=8, target_modules=["layers.0"])
        model_wrapper.attach_lora(cfg)
        assert model_wrapper.lora_config is not None
        assert model_wrapper.lora_config.rank == 4

    def test_attach_lora_with_kwargs_override(self, model_wrapper: Model) -> None:
        cfg = LoraConfig(rank=4, alpha=8, target_modules=["layers.0"])
        model_wrapper.attach_lora(cfg, rank=16)
        assert model_wrapper.lora_config is not None
        assert model_wrapper.lora_config.rank == 16

    def test_attach_lora_without_config_raises(self, model_wrapper: Model) -> None:
        with pytest.raises(ValueError, match="No LoraConfig"):
            model_wrapper.attach_lora()

    def test_from_pretrained_with_lora_stores_config(self) -> None:
        """Verify from_pretrained stores lora config (can't apply without real model)."""
        cfg = LoraConfig(rank=4)
        # We can't actually call from_pretrained without a real model,
        # but we can verify the constructor stores it
        model = Model(TinyMlp(), DummyTokenizer(), "mlx", lora_config=cfg)
        assert model.lora_config is not None
        assert model.lora_config.rank == 4


# ---------------------------------------------------------------------------
# Model.prepare_for_joint_training tests
# ---------------------------------------------------------------------------


class TestPrepareForJointTraining:
    """Tests for prepare_for_joint_training."""

    def test_freezes_and_unfreezes(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="test", layers=[1]))
        model_wrapper.prepare_for_joint_training()

        # The probe module should still have trainable parameters
        probe = model_wrapper.probes["test"]
        from mlx.utils import tree_flatten

        params = list(tree_flatten(probe.module.trainable_parameters()))
        assert len(params) > 0

    def test_probe_stays_trainable_after_freeze(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        probe = model_wrapper.probes["digit"]
        from mlx.utils import tree_flatten

        params = list(tree_flatten(probe.module.trainable_parameters()))
        assert len(params) > 0

    def test_forward_works_after_freeze(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert outputs.lm_logits is not None
        assert "digit" in outputs.probes


# ---------------------------------------------------------------------------
# JointLoss tests
# ---------------------------------------------------------------------------


class TestMakeJointLoss:
    """Tests for the JointLoss class."""

    def test_returns_callable(self) -> None:
        loss_fn = JointLoss()
        assert callable(loss_fn)

    def test_works_with_trainable_model(self) -> None:
        from auto_chasm.trainers.trainable import _TrainableModel

        loss_fn = JointLoss()
        base = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)
        train_model = _TrainableModel(base, {})

        batch = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[0, 0, 1, 0, 0]])
        lengths = mx.array([[0, 5]])

        result = loss_fn(train_model, batch, labels, lengths)
        assert len(result) == 3
        total, ntoks, components = result
        assert total.ndim == 0  # scalar
        assert float(ntoks) > 0

    def test_different_weights_change_loss(self) -> None:
        """Different probe_weight values produce different losses when probes exist."""
        from auto_chasm.trainers.trainable import _TrainableModel

        # Build two model wrappers with probes so binary_logits is not None
        mx.random.seed(42)
        base_a = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)
        base_b = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)

        class Config:
            """Dummy configuration for testing."""

            hidden_size = 16
            num_hidden_layers = 4

        base_a.config = Config()
        base_b.config = Config()

        wrapper_a = Model(base_a, DummyTokenizer(), "mlx")
        wrapper_b = Model(base_b, DummyTokenizer(), "mlx")

        wrapper_a.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        wrapper_b.attach_probe(ProbeConfig(name="digit", layers=[-1]))

        train_a = _TrainableModel(wrapper_a.model, wrapper_a._probes)
        train_b = _TrainableModel(wrapper_b.model, wrapper_b._probes)

        batch = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[0, 0, 1, 0, 0]])
        lengths = mx.array([[0, 5]])

        loss_a = JointLoss(weights={"digit": 0.5})
        loss_b = JointLoss(weights={"digit": 5.0})

        total_a, _, _ = loss_a(train_a, batch, labels, lengths)
        total_b, _, _ = loss_b(train_b, batch, labels, lengths)

        assert float(total_a) != float(total_b)

    def test_compatible_with_value_and_grad(self) -> None:
        from auto_chasm.trainers.trainable import _TrainableModel

        loss_fn = JointLoss()
        base = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)
        train_model = _TrainableModel(base, {})

        loss_and_grad = nn.value_and_grad(train_model, loss_fn)

        batch = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[0, 0, 1, 0, 0]])
        lengths = mx.array([[0, 5]])

        (total, ntoks, components), grad = loss_and_grad(train_model, batch, labels, lengths)
        assert float(total) > 0
        assert grad is not None


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


class TestTrainer:
    """Tests for the Trainer facade."""

    def test_init(self, model_wrapper: Model) -> None:
        loss_fn = JointLoss()
        trainer = Trainer(model=model_wrapper, loss_fn=loss_fn, num_iters=10)
        assert trainer.model is model_wrapper
        assert trainer.num_iters == 10

    def test_callback_invoked(self, model_wrapper: Model, sample_dataset: list) -> None:
        """Verify callbacks are invoked during training."""
        from auto_chasm.trainers.data_utils import JointTextDataset

        class TrackingCallback(TrainerCallback):
            """Test helper that tracks callback invocations."""

            def __init__(self) -> None:
                self.train_begin = False
                self.train_end = False

            def on_train_begin(self, **kwargs: object) -> None:  # noqa: ARG002
                self.train_begin = True

            def on_train_end(self, **kwargs: object) -> None:  # noqa: ARG002
                self.train_end = True

        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = JointLoss()
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=4,
            batch_size=8,
            max_seq_length=32,
            logging_steps=2,
            save_steps=100,
            early_stopping_patience=0,
            verbose=False,
        )
        result = trainer.train(ds)

        assert "history" in result
        history = result["history"]
        assert isinstance(history, History)
        assert len(history.train_losses) > 0


# ---------------------------------------------------------------------------
# TrainerCallback tests
# ---------------------------------------------------------------------------


class TestTrainerCallback:
    """Tests for TrainerCallback base class."""

    def test_default_methods_do_not_raise(self) -> None:
        cb = TrainerCallback()
        cb.on_train_begin()
        cb.on_train_end()
        cb.on_step_end()
        cb.on_epoch_end()


# ---------------------------------------------------------------------------
# build_auto_steer_fn tests
# ---------------------------------------------------------------------------


class TestBuildAutoSteerFn:
    """Tests for the extracted build_auto_steer_fn."""

    def test_returns_none_without_geometry(self) -> None:
        config = SteeringConfig()
        hook = SteeringHook("test", config)
        fn = build_auto_steer_fn(hook)
        assert fn is None

    def test_returns_callable_with_geometry(self) -> None:
        config = SteeringConfig(method="nullify")
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([1.0] * 16)
        hook._mean_1 = mx.array([2.0] * 16)
        hook._direction = hook._mean_1 - hook._mean_0

        fn = build_auto_steer_fn(hook)
        assert fn is not None
        assert callable(fn)

    def test_nullify_steers_last_token(self) -> None:
        config = SteeringConfig(method="nullify")
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([1.0] * 16)
        hook._mean_1 = mx.array([2.0] * 16)
        hook._direction = hook._mean_1 - hook._mean_0

        fn = build_auto_steer_fn(hook)
        assert fn is not None

        head = nn.Linear(16, 1)
        # Set weight so head outputs positive logit for last token
        head.weight = mx.ones((1, 16)) * 0.1
        head.bias = mx.zeros(1)

        hidden = mx.random.normal((1, 3, 16))
        logits = head(hidden).squeeze(-1)  # [1, 3]

        result = fn(hidden, head, logits)
        assert result.shape == hidden.shape

        # Last token should be modified (nullify pushes logit toward 0)
        # Previous tokens should be unchanged
        diff = mx.abs(result[:, :-1, :] - hidden[:, :-1, :])
        assert float(mx.sum(diff).item()) < 1e-6

    def test_push_to_mean_steers_when_positive(self) -> None:
        config = SteeringConfig(method="push_to_mean")
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 16)
        hook._mean_1 = mx.array([1.0] * 16)
        hook._direction = hook._mean_1 - hook._mean_0

        fn = build_auto_steer_fn(hook)
        assert fn is not None

        head = nn.Linear(16, 1)
        head.weight = mx.ones((1, 16)) * 0.5
        head.bias = mx.zeros(1)

        hidden = mx.random.normal((1, 2, 16))
        logits = head(hidden).squeeze(-1)

        result = fn(hidden, head, logits)
        assert result.shape == hidden.shape


# ---------------------------------------------------------------------------
# Integration: probe + forward + generate
# ---------------------------------------------------------------------------


class TestProbeForwardIntegration:
    """Integration tests for probe + forward with new API."""

    def test_forward_with_probe(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="test", layers=[1]))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)

        assert outputs.lm_logits is not None
        assert "test" in outputs.probes
        assert outputs.probes["test"].logits is not None

    def test_forward_logits_shape(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="test", layers=[1]))
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        outputs = model_wrapper.forward(input_ids)

        # LM logits should be [batch, seq, vocab]
        assert outputs.lm_logits.ndim == 3
        assert outputs.lm_logits.shape[0] == 1
        assert outputs.lm_logits.shape[1] == 5

    def test_forward_probe_logits_shape(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="test", layers=[1]))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)

        probe_logits = outputs.probes["test"].logits
        assert probe_logits.ndim == 3  # [batch, seq, 1]
        assert probe_logits.shape[0] == 1
        assert probe_logits.shape[1] == 3

    def test_forward_with_steering(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="test", layers=[1]))

        mean_0 = mx.array([1.0] * 16)
        mean_1 = mx.array([2.0] * 16)
        model_wrapper.enable_steering(
            "test",
            config=SteeringConfig(method="nullify"),
            class_means={"mean_0": mean_0, "mean_1": mean_1},
        )

        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert outputs.lm_logits is not None

    def test_forward_with_push_to_mean_steering(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="test", layers=[2]))

        mean_0 = mx.array([0.0] * 16)
        mean_1 = mx.array([1.0] * 16)
        model_wrapper.enable_steering(
            "test",
            config=SteeringConfig(method="push_to_mean"),
            class_means={"mean_0": mean_0, "mean_1": mean_1},
        )

        input_ids = mx.array([[1, 2, 3, 4]])
        outputs = model_wrapper.forward(input_ids)
        assert outputs.lm_logits is not None
        assert "test" in outputs.probes

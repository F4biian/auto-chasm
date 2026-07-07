"""Freeze/unfreeze API tests for model.py.

Tests the freeze/unfreeze API surface for consistency, correctness,
and edge cases.
"""

from __future__ import annotations

import warnings
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import ProbeConfig
from auto_chasm.model import Model
from auto_chasm.trainers.data_utils import JointTextDataset, iterate_batches
from auto_chasm.trainers.trainable import _TrainableModel, make_joint_loss

# ── Constants ────────────────────────────────────────────────────────────
HIDDEN_DIM = 32
VOCAB_SIZE = 64
NUM_LAYERS = 4
BATCH_SIZE = 8
MAX_SEQ_LEN = 32
PROBE_NAME_A = "digit"
PROBE_NAME_B = "letter"


class TinyMlp(nn.Module):
    """A tiny MLP that behaves like a transformer for testing.

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
def tiny_model() -> tuple[TinyMlp, DummyTokenizer]:
    """Create a tiny model and tokenizer."""
    mx.random.seed(42)
    return TinyMlp(), DummyTokenizer()


@pytest.fixture
def model_wrapper(tiny_model: tuple[TinyMlp, DummyTokenizer]) -> Model:
    """Create a Model wrapper with config and a probe attached."""
    base_model, tokenizer = tiny_model

    class Config:
        """Mock model config for hidden_dim detection."""

        hidden_size = HIDDEN_DIM
        num_hidden_layers = NUM_LAYERS

    base_model.config = Config()
    wrapper = Model(base_model, tokenizer, backend_name="mlx")
    wrapper.attach_probe(ProbeConfig(name=PROBE_NAME_A, layers=[-1]))
    wrapper.attach_probe(ProbeConfig(name=PROBE_NAME_B, layers=[0]))
    return wrapper


@pytest.fixture
def sample_dataset() -> list[dict]:
    """Create a small synthetic dataset."""
    mx.random.seed(42)
    data = []
    for _ in range(32):
        data.append({"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 0, 0]})
    return data


def _count_trainable_params(module: Any) -> int:
    """Count trainable parameter groups in an MLX module.

    Args:
        module: The MLX module to inspect.

    Returns:
        Number of trainable parameter groups.
    """
    from mlx.utils import tree_flatten

    return len(list(tree_flatten(module.trainable_parameters())))


def _count_all_params(module: Any) -> int:
    """Count all parameter groups in an MLX module.

    Args:
        module: The MLX module to inspect.

    Returns:
        Number of parameter groups.
    """
    from mlx.utils import tree_flatten

    return len(list(tree_flatten(module.parameters())))


# ── Basic freeze/unfreeze ─────────────────────────────────────


class TestBasicFreezeUnfreeze:
    """Basic freeze_model / unfreeze_model lifecycle."""

    def test_prepare_then_freeze_model_stays_frozen(self, model_wrapper: Model) -> None:
        """After prepare_for_joint_training, freezing again keeps model frozen."""
        model_wrapper.prepare_for_joint_training()
        base_params_before = _count_all_params(model_wrapper.model)
        frozen_before = _count_trainable_params(model_wrapper.model)
        assert frozen_before == 0, "Model should be frozen after prepare_for_joint_training"

        model_wrapper.freeze_model()
        frozen_after = _count_trainable_params(model_wrapper.model)
        assert frozen_after == 0, "Model should stay frozen after re-freeze"
        assert _count_all_params(model_wrapper.model) == base_params_before, "Param count unchanged"

    def test_freeze_then_unfreeze_model(self, model_wrapper: Model) -> None:
        """Freeze then unfreeze — all params should be trainable again."""
        total_params = _count_all_params(model_wrapper.model)
        model_wrapper.freeze_model()
        assert _count_trainable_params(model_wrapper.model) == 0
        model_wrapper.unfreeze_model()
        assert _count_trainable_params(model_wrapper.model) == total_params


# ── Probe freeze/unfreeze with nonexistent names ──────────────


class TestProbeNameErrors:
    """Error handling for probe name lookups."""

    def test_unfreeze_nonexistent_probe_raises_key_error(self, model_wrapper: Model) -> None:
        """unfreeze_probe('nonexistent') should raise KeyError."""
        with pytest.raises(KeyError, match="Probe.*not found"):
            model_wrapper.unfreeze_probe("nonexistent")

    def test_freeze_nonexistent_probe_raises_key_error(self, model_wrapper: Model) -> None:
        """freeze_probe('nonexistent') should raise KeyError (currently silent)."""
        with pytest.raises(KeyError):
            model_wrapper.freeze_probe("nonexistent")

    def test_unfreeze_nonexistent_probe_different_name(self, model_wrapper: Model) -> None:
        """unfreeze_probe with a different nonexistent name also raises KeyError."""
        with pytest.raises(KeyError):
            model_wrapper.unfreeze_probe("no_such_probe_ever")


# ── Test 6: Double prepare_for_joint_training ────────────────────────────


class TestDoublePrepare:
    """Calling prepare_for_joint_training twice."""

    def test_prepare_for_joint_training_twice(self, model_wrapper: Model) -> None:
        """Calling prepare_for_joint_training twice should not crash."""
        model_wrapper.prepare_for_joint_training()
        model_wrapper.prepare_for_joint_training()
        assert _count_trainable_params(model_wrapper.model) == 0, "Model stays frozen"


# ── Training with selective freeze/unfreeze ───────────────────


class TestSelectiveFreezeTraining:
    """Train step with selective freeze/unfreeze patterns."""

    def test_freeze_model_unfreeze_probe_only_probe_changes(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        """When model is frozen but a probe is unfrozen, only the probe changes."""
        model_wrapper.prepare_for_joint_training()
        model_wrapper.freeze_probe(PROBE_NAME_B)
        model_wrapper.unfreeze_probe(PROBE_NAME_A)

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        train_model = _TrainableModel(model_wrapper.model, model_wrapper._probes)
        # _TrainableModel unfreezes all probes; re-freeze probe B
        train_model.get_probe(PROBE_NAME_B).freeze()

        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")
        batch = next(iterate_batches(ds, BATCH_SIZE, MAX_SEQ_LEN, loop=True))
        tokens, labels, lengths = batch

        from mlx.utils import tree_flatten

        (_, _, _), grad = nn.value_and_grad(train_model, loss_fn)(
            train_model, mx.array(tokens), mx.array(labels), mx.array(lengths)
        )

        grad_flat = tree_flatten(grad)
        grad_keys = [k for k, _ in grad_flat]
        probe_a_has_grad = any(f"probe_{PROBE_NAME_A}" in k for k in grad_keys)
        probe_b_has_grad = any(f"probe_{PROBE_NAME_B}" in k for k in grad_keys)

        assert probe_a_has_grad, "Unfrozen probe A should have gradients"
        assert not probe_b_has_grad, "Frozen probe B should have no gradients"

    def test_all_frozen_no_gradients(self, model_wrapper: Model) -> None:
        """When everything is frozen, no params should be trainable."""
        model_wrapper.freeze_model()
        model_wrapper.freeze_probe(PROBE_NAME_A)
        model_wrapper.freeze_probe(PROBE_NAME_B)

        train_model = _TrainableModel(model_wrapper.model, model_wrapper._probes)
        # _TrainableModel unfreezes all probes; re-freeze them
        train_model.get_probe(PROBE_NAME_A).freeze()
        train_model.get_probe(PROBE_NAME_B).freeze()

        n_trainable = _count_trainable_params(train_model)
        assert n_trainable == 0, (
            f"Expected 0 trainable params when everything frozen, got {n_trainable}"
        )


# ── Test 9: Deprecated method ────────────────────────────────────────────


class TestDeprecatedMethod:
    """The deprecated freeze_adapters_and_unfreeze_probes method."""

    def test_deprecated_method_still_works_and_warns(self, model_wrapper: Model) -> None:
        """freeze_adapters_and_unfreeze_probes should emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model_wrapper.freeze_adapters_and_unfreeze_probes()

        has_warning = any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert has_warning, "Should have emitted a DeprecationWarning"

        # Verify it actually froze model + unfroze probes
        assert _count_trainable_params(model_wrapper.model) == 0, "Model should be frozen"
        for pname in model_wrapper.probes:
            assert _count_trainable_params(model_wrapper.probes[pname].module) > 0, (
                f"Probe {pname} should be unfrozen"
            )


# ── Test 10: unfreeze_lora_adapters without LoRA ─────────────────────────


class TestUnfreezeLoraWithoutLora:
    """Calling unfreeze_lora_adapters when no LoRA is attached."""

    def test_unfreeze_lora_without_lora_does_not_crash(self, model_wrapper: Model) -> None:
        """unfreeze_lora_adapters without LoRA should be a no-op, not crash."""
        model_wrapper.unfreeze_lora_adapters()

    def test_unfreeze_lora_no_side_effects_on_params(self, model_wrapper: Model) -> None:
        """unfreeze_lora_adapters should not unfreeze base model when no LoRA."""
        model_wrapper.freeze_model()
        model_wrapper.unfreeze_lora_adapters()
        assert _count_trainable_params(model_wrapper.model) == 0, (
            "Model should still be frozen after unfreezing non-existent LoRA"
        )

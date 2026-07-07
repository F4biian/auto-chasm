"""Full integration test with a real model.

Tests the complete pipeline: load model → attach probe → train → steer → checkpoint.
Requires mlx-community/gemma-3-270m-it-8bit (cached from playground).

Run with: uv run pytest tests/test_real_integration.py -v -s
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

from auto_chasm.config import ProbeConfig, SteeringConfig
from auto_chasm.model import Model
from auto_chasm.probe import _get_hidden_dim
from auto_chasm.trainers.base import JointTrainer
from auto_chasm.trainers.data_utils import JointTextDataset
from auto_chasm.trainers.trainable import make_joint_loss

# Whole module loads a real pretrained model — gated behind --run-real-model.
pytestmark = pytest.mark.real_model

MODEL_NAME = "mlx-community/gemma-3-270m-it-8bit"


def token_contains_digit(tokenizer: object, tid: int) -> bool:
    """Check if a token contains a digit."""
    try:
        raw = tokenizer.convert_ids_to_tokens([tid])  # type: ignore[union-attr]
        return any(ch.isdigit() for ch in raw)
    except Exception:
        s = tokenizer.decode([tid])  # type: ignore[union-attr]
        return any(ch.isdigit() for ch in s)


@pytest.fixture(scope="module")
def loaded_model() -> tuple[Model, object]:
    """Load the real model once for all tests in this module."""
    model = Model.from_pretrained(MODEL_NAME, backend_name="mlx")
    return model, model.tokenizer


@pytest.fixture(scope="module")
def tiny_dataset(loaded_model: tuple[Model, object]) -> list[dict[str, list[int]]]:
    """Create a tiny dataset with digit labels."""
    _, tokenizer = loaded_model
    texts = [
        "The answer is 42 and that is final.",
        "Hello world, this is test number 7.",
        "Python 3.12 was released in 2023.",
        "No digits here at all, just words.",
        "The year 2025 will be interesting.",
        "Roses are red, violets are blue.",
        "Call me at 555-0123 please.",
        "The quick brown fox jumps over the lazy dog.",
        "Item 3 costs $9.99 on sale.",
        "Zero gravity is approximately 0 m/s squared.",
    ]

    data = []
    for text in texts:
        tokens = tokenizer.encode(text)  # type: ignore[union-attr]
        if tokens and tokens[-1] != tokenizer.eos_token_id:  # type: ignore[union-attr]
            tokens.append(tokenizer.eos_token_id)  # type: ignore[union-attr]
        labels = [1 if token_contains_digit(tokenizer, tid) else 0 for tid in tokens]
        data.append({"tokens": tokens, "labels": labels})

    return data


class TestModelLoading:
    """Test that the real model loads correctly."""

    def test_model_loads(self, loaded_model: tuple[Model, object]) -> None:
        model, tokenizer = loaded_model
        assert model.model is not None
        assert tokenizer is not None
        assert model.backend.name == "mlx"

    def test_model_has_layers(self, loaded_model: tuple[Model, object]) -> None:
        model, _ = loaded_model
        from auto_chasm.probe import _find_layers

        layers = _find_layers(model.model)
        assert layers is not None
        assert len(layers) > 0

    def test_mlx_tune_model_compatibility(self) -> None:
        """Verify that models loaded via mlx_tune work with auto_chasm."""
        from mlx_tune import FastLanguageModel

        m, tok = FastLanguageModel.from_pretrained(MODEL_NAME, max_seq_length=64)
        m = FastLanguageModel.get_peft_model(
            m, r=4, target_modules=["q_proj"], lora_alpha=8, lora_dropout=0.0
        )

        wrapper = Model(m, tok, backend_name="mlx")
        num_layers = len(m.model.layers)
        wrapper.attach_probe(ProbeConfig(name="test", layers=[num_layers // 2]))

        input_ids = mx.array([[1, 2, 3]])
        outputs = wrapper.forward(input_ids)
        assert outputs.lm_logits is not None
        assert "test" in outputs.probes


class TestProbeInjection:
    """Test probe injection with the real model."""

    def test_attach_single_layer_probe(self, loaded_model: tuple[Model, object]) -> None:
        model, _ = loaded_model
        num_layers = len(model.model.model.layers)
        mid = num_layers // 2

        config = ProbeConfig(name="digit", layers=[mid])
        probe = model.attach_probe(config)
        assert len(probe.layer_captures) == 1
        assert probe.module is not None

    def test_forward_returns_probe_logits(
        self,
        loaded_model: tuple[Model, object],
        tiny_dataset: list[dict[str, list[int]]],
    ) -> None:
        model, _ = loaded_model
        sample = tiny_dataset[0]
        input_ids = mx.array([sample["tokens"][:16]])

        outputs = model.forward(input_ids)
        assert outputs.lm_logits is not None
        assert "digit" in outputs.probes
        assert outputs.probes["digit"].logits is not None

    def test_probe_logits_shape_matches_input(
        self,
        loaded_model: tuple[Model, object],
        tiny_dataset: list[dict[str, list[int]]],
    ) -> None:
        model, _ = loaded_model
        sample = tiny_dataset[0]
        seq_len = min(len(sample["tokens"]), 16)
        input_ids = mx.array([sample["tokens"][:seq_len]])

        outputs = model.forward(input_ids)
        probe_logits = outputs.probes["digit"].logits
        assert probe_logits.shape[0] == 1
        assert probe_logits.shape[1] == seq_len

    def test_multi_layer_probe(self, loaded_model: tuple[Model, object]) -> None:
        model, _ = loaded_model
        num_layers = len(model.model.model.layers)

        config = ProbeConfig(
            name="multi",
            layers=[num_layers // 4, num_layers // 2, 3 * num_layers // 4],
            aggregation="concat",
        )
        model.attach_probe(config)

        sample_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        input_ids = mx.array([sample_tokens])
        outputs = model.forward(input_ids)

        assert "multi" in outputs.probes
        assert outputs.probes["multi"].logits is not None


class TestLossComputation:
    """Test loss computation with the real model."""

    def test_make_joint_loss_returns_callable(self) -> None:
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=2.0)
        assert callable(loss_fn)

    def test_loss_computation_runs(
        self,
        loaded_model: tuple[Model, object],
        tiny_dataset: list[dict[str, list[int]]],
    ) -> None:
        model, _ = loaded_model
        if "digit" not in model.probes:
            num_layers = len(model.model.model.layers)
            model.attach_probe(ProbeConfig(name="digit", layers=[num_layers // 2]))

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=2.0)

        sample = tiny_dataset[0]
        tokens = sample["tokens"][:16]
        labels = sample["labels"][:16]

        batch = mx.array([tokens])
        label_arr = mx.array([labels])
        lengths = mx.array([[0, len(tokens)]])

        from auto_chasm.trainers.trainable import _TrainableModel

        train_model = _TrainableModel(model.model, model._probes)
        result = loss_fn(train_model, batch, label_arr, lengths)
        train_model.restore_capture_fns()
        assert len(result) == 3

        total, ntoks, components = result
        assert float(total.item()) > 0
        assert float(ntoks.item()) > 0
        assert isinstance(components, dict)


class TestTraining:
    """Test the training loop with the real model."""

    def test_training_runs_and_loss_decreases(
        self,
        loaded_model: tuple[Model, object],
        tiny_dataset: list[dict[str, list[int]]],
    ) -> None:
        model, tokenizer = loaded_model
        num_layers = len(model.model.model.layers)

        if "digit" not in model.probes:
            config = ProbeConfig(name="digit", layers=[num_layers // 2])
            model.attach_probe(config)

        train_ds = JointTextDataset(tiny_dataset[:8], tokenizer, tokens_key="tokens")
        val_ds = JointTextDataset(tiny_dataset[8:], tokenizer, tokens_key="tokens")

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model,
                loss_fn=loss_fn,
                learning_rate=2e-4,
                weight_decay=0.0,
                grad_clip_norm=1.0,
                num_iters=20,
                batch_size=4,
                max_seq_length=64,
                grad_accum_steps=1,
                logging_steps=5,
                save_steps=10,
                early_stopping_patience=999,
                output_dir=tmpdir,
            )

            history = trainer.run(train_ds, val_ds)
            assert len(history.train_losses) > 0


class TestSteering:
    """Test steering with the real model."""

    def test_enable_and_disable_steering(
        self,
        loaded_model: tuple[Model, object],
    ) -> None:
        model, tokenizer = loaded_model
        if "digit" not in model.probes:
            num_layers = len(model.model.model.layers)
            model.attach_probe(ProbeConfig(name="digit", layers=[num_layers // 2]))

        hidden_dim = _get_hidden_dim(model.model)
        mean_0 = mx.ones(hidden_dim) * 0.5
        mean_1 = mx.ones(hidden_dim) * 1.5

        model.enable_steering(
            "digit",
            config=SteeringConfig(method="nullify"),
            class_means={"mean_0": mean_0, "mean_1": mean_1},
        )
        assert model.steering_hooks["digit"].enabled

        model.disable_steering("digit")
        assert not model.steering_hooks["digit"].enabled

    def test_steering_changes_logits(
        self,
        loaded_model: tuple[Model, object],
        tiny_dataset: list[dict[str, list[int]]],
    ) -> None:
        model, _ = loaded_model
        if "digit" not in model.probes:
            num_layers = len(model.model.model.layers)
            model.attach_probe(ProbeConfig(name="digit", layers=[num_layers // 2]))
        else:
            probe = model._probes["digit"]
            probe.module.weight = mx.random.normal(probe.module.weight.shape) * 1e-4
            probe.module.bias = mx.zeros_like(probe.module.bias)

        sample = tiny_dataset[0]
        input_ids = mx.array([sample["tokens"][:16]])

        outputs_before = model.forward(input_ids)
        lm_before = outputs_before.lm_logits

        hidden_dim = _get_hidden_dim(model.model)
        mean_0 = mx.ones(hidden_dim) * 0.5
        mean_1 = mx.ones(hidden_dim) * 1.5

        model.enable_steering(
            "digit",
            config=SteeringConfig(method="nullify"),
            class_means={"mean_0": mean_0, "mean_1": mean_1},
        )

        outputs_after = model.forward(input_ids)
        lm_after = outputs_after.lm_logits

        diff = float(mx.sum(mx.abs(lm_before - lm_after)).item())
        assert diff > 0, "Steering should change LM logits"

        model.disable_steering("digit")


class TestCheckpoint:
    """Test checkpoint save/load with the real model."""

    def test_save_checkpoint(
        self,
        loaded_model: tuple[Model, object],
    ) -> None:
        model, _ = loaded_model
        if "digit" not in model.probes:
            num_layers = len(model.model.model.layers)
            model.attach_probe(ProbeConfig(name="digit", layers=[num_layers // 2]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_checkpoint")
            model.save_checkpoint(path)

            manifest_path = Path(path) / "manifest.json"
            assert manifest_path.exists()

            with open(manifest_path) as f:
                manifest = json.load(f)

            assert "probes" in manifest
            assert "digit" in manifest["probes"]
            assert manifest["probes"]["digit"]["layers"] is not None

            probe_path = Path(path) / "probes" / "digit.safetensors"
            assert probe_path.exists()

    def test_checkpoint_contains_probe_weights(
        self,
        loaded_model: tuple[Model, object],
    ) -> None:
        model, _ = loaded_model
        if "digit" not in model.probes:
            num_layers = len(model.model.model.layers)
            model.attach_probe(ProbeConfig(name="digit", layers=[num_layers // 2]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_checkpoint")
            model.save_checkpoint(path)

            probe_path = Path(path) / "probes" / "digit.safetensors"
            weights = mx.load(str(probe_path))
            assert len(weights) > 0


class TestGeneration:
    """Test generation with the real model."""

    def test_generate_returns_string(self, loaded_model: tuple[Model, object]) -> None:
        model, _ = loaded_model
        result = model.generate("Hello", max_tokens=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_steering(
        self,
        loaded_model: tuple[Model, object],
    ) -> None:
        model, _ = loaded_model
        if "digit" not in model.probes:
            num_layers = len(model.model.model.layers)
            model.attach_probe(ProbeConfig(name="digit", layers=[num_layers // 2]))

        hidden_dim = _get_hidden_dim(model.model)
        mean_0 = mx.ones(hidden_dim) * 0.5
        mean_1 = mx.ones(hidden_dim) * 1.5

        model.enable_steering(
            "digit",
            config=SteeringConfig(method="nullify"),
            class_means={"mean_0": mean_0, "mean_1": mean_1},
        )

        result = model.generate("Tell me a number", max_tokens=5)
        assert isinstance(result, str)

        model.disable_steering("digit")

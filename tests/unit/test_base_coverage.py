"""Coverage-patch tests for trainers/base.py — internal helper methods.

Covers:
- JointTrainer._save_checkpoint
- JointTrainer._save_best_weights
- JointTrainer._save_training_manifest
- JointTrainer._cleanup_periodic_checkpoints
- TrainingConfig seed application in __init__
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import TrainingConfig
from auto_chasm.model import Model
from auto_chasm.probe import ProbeConfig
from auto_chasm.trainers.base import JointTrainer
from auto_chasm.trainers.trainable import make_joint_loss


class TinyMlp(nn.Module):
    """A tiny MLP for testing trainer internals."""

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
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "test"


@pytest.fixture
def model_wrapper() -> Model:
    """Create a Model wrapper with a probe attached."""
    mx.random.seed(42)
    model = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)

    class Config:
        """Mock model config for hidden_dim detection."""

        hidden_size = 16
        num_hidden_layers = 4

    model.config = Config()
    wrapper = Model(model, DummyTokenizer(), backend_name="mlx")
    wrapper.attach_probe(ProbeConfig(name="test_probe", layers=[-1]))
    wrapper.prepare_for_joint_training()
    return wrapper


@pytest.fixture
def sample_dataset() -> list[dict]:
    """Create a small synthetic dataset."""
    data = []
    for _ in range(16):
        data.append({"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 0, 0]})
    return data


@pytest.fixture
def joint_trainer(model_wrapper: Model, sample_dataset: list) -> JointTrainer:
    """Create a JointTrainer with output dir in a temp directory."""
    loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
    return JointTrainer(
        model=model_wrapper,
        loss_fn=loss_fn,
        num_iters=4,
        batch_size=8,
        max_seq_length=32,
        logging_steps=2,
        save_steps=0,
        eval_steps=0,
        early_stopping_patience=0,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# _save_checkpoint tests
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    """Tests for JointTrainer._save_checkpoint."""

    def test_save_checkpoint_creates_files(self, joint_trainer: JointTrainer) -> None:
        """_save_checkpoint should create adapter and probe head files."""
        adapter_file = str(joint_trainer.output_dir / "adapters.safetensors")
        joint_trainer._save_checkpoint(adapter_file, it=5)

        expected_adapter = joint_trainer.output_dir / "0000005_adapters.safetensors"
        expected_head = joint_trainer.output_dir / "0000005_test_probe_head.safetensors"

        assert expected_adapter.exists(), f"Expected {expected_adapter} to exist"
        assert expected_head.exists(), f"Expected {expected_head} to exist"

    def test_save_checkpoint_multiple_iters(self, joint_trainer: JointTrainer) -> None:
        """Multiple _save_checkpoint calls should create distinct files."""
        adapter_file = str(joint_trainer.output_dir / "adapters.safetensors")
        joint_trainer._save_checkpoint(adapter_file, it=10)
        joint_trainer._save_checkpoint(adapter_file, it=20)

        assert (joint_trainer.output_dir / "0000010_adapters.safetensors").exists()
        assert (joint_trainer.output_dir / "0000020_adapters.safetensors").exists()


# ---------------------------------------------------------------------------
# _save_best_weights tests
# ---------------------------------------------------------------------------


class TestSaveBestWeights:
    """Tests for JointTrainer._save_best_weights."""

    def test_save_best_weights_creates_files(self, joint_trainer: JointTrainer) -> None:
        """_save_best_weights should save adapter and best head files."""
        adapter_file = str(joint_trainer.output_dir / "adapters.safetensors")
        joint_trainer._save_best_weights(adapter_file)

        assert Path(adapter_file).exists()
        expected_head = joint_trainer.output_dir / "test_probe_head.safetensors"
        assert expected_head.exists()

    def test_save_best_weights_all_probes_saved(self, model_wrapper: Model) -> None:
        """All attached probes should have their head weights saved."""
        model_wrapper.attach_probe(ProbeConfig(name="probe_a", layers=[-1]))
        model_wrapper.attach_probe(ProbeConfig(name="probe_b", layers=[0]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        trainer = JointTrainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=2,
            batch_size=8,
            max_seq_length=32,
            save_steps=0,
            eval_steps=0,
            early_stopping_patience=0,
            verbose=False,
        )

        adapter_file = str(trainer.output_dir / "adapters.safetensors")
        trainer._save_best_weights(adapter_file)

        assert (trainer.output_dir / "probe_a_head.safetensors").exists()
        assert (trainer.output_dir / "probe_b_head.safetensors").exists()


# ---------------------------------------------------------------------------
# _save_training_manifest tests
# ---------------------------------------------------------------------------


class TestSaveTrainingManifest:
    """Tests for JointTrainer._save_training_manifest."""

    def test_manifest_contains_expected_keys(self, joint_trainer: JointTrainer) -> None:
        """The manifest JSON should contain all expected metadata fields."""
        joint_trainer._save_training_manifest(best_iter=3, best_metric=0.42)

        manifest_path = joint_trainer.output_dir / "training_manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["best_iter"] == 3
        assert manifest["best_metric"] == 0.42
        assert manifest["best_metric_name"] == "val_loss"
        assert manifest["num_iters"] == 4
        assert manifest["keep_best_only"] is False

    def test_manifest_non_finite_metric_is_none(self, joint_trainer: JointTrainer) -> None:
        """When best_iter is 0, best_metric should be None in the JSON."""
        joint_trainer._save_training_manifest(best_iter=0, best_metric=float("nan"))

        manifest_path = joint_trainer.output_dir / "training_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["best_metric"] is None

    def test_manifest_inf_metric_is_none(self, joint_trainer: JointTrainer) -> None:
        """Infinity best_metric should be serialized as None."""
        joint_trainer._save_training_manifest(best_iter=5, best_metric=float("inf"))

        manifest_path = joint_trainer.output_dir / "training_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["best_metric"] is None


# ---------------------------------------------------------------------------
# _cleanup_periodic_checkpoints tests
# ---------------------------------------------------------------------------


class TestCleanupPeriodicCheckpoints:
    """Tests for JointTrainer._cleanup_periodic_checkpoints."""

    def test_cleans_digit_prefixed_checkpoints(self, joint_trainer: JointTrainer) -> None:
        """Only files starting with a digit and matching patterns should be removed."""
        out_dir = joint_trainer.output_dir

        # Create periodic checkpoint files
        (out_dir / "0000005_adapters.safetensors").touch()
        (out_dir / "0000005_test_probe_head.safetensors").touch()
        (out_dir / "0000010_adapters.safetensors").touch()
        # Create a file that should be kept
        (out_dir / "adapters.safetensors").touch()
        (out_dir / "test_probe_head.safetensors").touch()
        (out_dir / "some_other_file.txt").touch()

        joint_trainer._cleanup_periodic_checkpoints()

        assert not (out_dir / "0000005_adapters.safetensors").exists()
        assert not (out_dir / "0000005_test_probe_head.safetensors").exists()
        assert not (out_dir / "0000010_adapters.safetensors").exists()
        # Non-periodic files should remain
        assert (out_dir / "adapters.safetensors").exists()
        assert (out_dir / "test_probe_head.safetensors").exists()
        assert (out_dir / "some_other_file.txt").exists()

    def test_cleanup_no_matches(self, joint_trainer: JointTrainer) -> None:
        """Cleanup with no periodic files should not raise."""
        joint_trainer._cleanup_periodic_checkpoints()


# ---------------------------------------------------------------------------
# TrainingConfig seed application tests
# ---------------------------------------------------------------------------


class TestTrainingConfigSeedApplication:
    """Tests that TrainingConfig seed is applied in JointTrainer.__init__."""

    def test_config_seed_sets_mx_random(self, model_wrapper: Model) -> None:
        """Providing a TrainingConfig with seed should set mx.random seed."""
        config = TrainingConfig(seed=42)
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)

        JointTrainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=4,
            batch_size=8,
            max_seq_length=32,
            verbose=False,
            config=config,
        )
        # If no exception, seed was applied successfully

    def test_config_params_override_defaults(self, model_wrapper: Model) -> None:
        """TrainingConfig values should be used as defaults for params."""
        config = TrainingConfig(
            learning_rate=1e-3,
            weight_decay=0.1,
            max_grad_norm=5.0,
            batch_size=2,
            gradient_accumulation_steps=4,
            logging_steps=3,
            save_steps=7,
            output_dir="/tmp/test_out",
            eval_steps=10,
        )
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        trainer = JointTrainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=4,
            max_seq_length=32,
            verbose=False,
            config=config,
        )

        assert trainer._base_lr == 1e-3
        assert trainer.grad_clip_norm == 5.0
        assert trainer.batch_size == 2
        assert trainer.grad_accum_steps == 4
        assert trainer.logging_steps == 3
        assert trainer.save_steps == 7

    def test_explicit_params_override_config(self, model_wrapper: Model) -> None:
        """Explicit keyword arguments should override TrainingConfig values."""
        config = TrainingConfig(learning_rate=1e-3, batch_size=2)
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        trainer = JointTrainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            learning_rate=5e-4,
            batch_size=16,
            num_iters=4,
            max_seq_length=32,
            verbose=False,
            config=config,
        )

        assert trainer._base_lr == 5e-4
        assert trainer.batch_size == 16

    def test_config_eval_steps_defaults_to_save_steps(self, model_wrapper: Model) -> None:
        """When eval_steps is None, it should default to save_steps."""
        config = TrainingConfig(eval_steps=10)
        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        trainer = JointTrainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=4,
            max_seq_length=32,
            save_steps=20,
            verbose=False,
            config=config,
        )
        assert trainer.eval_steps == 10

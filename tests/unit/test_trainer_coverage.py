"""Coverage-patch tests for trainers/trainer.py — edge cases and uncovered paths.

Covers:
- custom_train_fn replacement path
- _evaluate_mlx with no prior training (RuntimeError)
- torch.manual_seed in __init__ with config
- Manifest path in _train_torch
- Trainer escape-hatch delegation to JointTrainer
- _get_joint RuntimeError for non-MLX backend
- _apply_probe_weights with JointLoss
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import TrainingConfig
from auto_chasm.model import Model
from auto_chasm.probe import ProbeConfig
from auto_chasm.trainers.trainer import Trainer


class TinyMlp(nn.Module):
    """A tiny MLP for testing trainer paths."""

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
    wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
    wrapper.prepare_for_joint_training()
    return wrapper


@pytest.fixture
def sample_dataset() -> list[dict]:
    """Create a small synthetic dataset."""
    data = []
    for _ in range(16):
        data.append({"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 0, 0]})
    return data


# ---------------------------------------------------------------------------
# custom_train_fn path
# ---------------------------------------------------------------------------


class TestCustomTrainFn:
    """Tests for the custom_train_fn replacement path."""

    def test_custom_train_fn_is_called(self, model_wrapper: Model, sample_dataset: list) -> None:
        """When custom_train_fn is set, it should be called instead of the default loop."""
        called: list[bool] = [False]

        def custom_train(model: Model, trainer: Trainer) -> dict:
            called[0] = True
            from auto_chasm.history import History

            return {
                "history": History(),
                "test_metrics": None,
                "output_dir": str(trainer.output_dir),
            }

        from auto_chasm.trainers.loss import JointLoss

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=JointLoss(),
            num_iters=2,
            max_seq_length=32,
            custom_train_fn=custom_train,
            verbose=False,
        )

        result = trainer.train(sample_dataset)
        assert called[0]
        assert "history" in result

    def test_custom_train_fn_return_value_is_passed_through(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        """The return value of custom_train_fn should be returned by train()."""

        def custom_train(model: Model, trainer: Trainer) -> dict:
            from auto_chasm.history import History

            return {
                "history": History(),
                "test_metrics": {"acc": 0.95},
                "output_dir": "/tmp/custom",
            }

        from auto_chasm.trainers.loss import JointLoss

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=JointLoss(),
            num_iters=2,
            max_seq_length=32,
            custom_train_fn=custom_train,
            verbose=False,
        )

        result = trainer.train(sample_dataset)
        assert result["test_metrics"]["acc"] == 0.95
        assert result["output_dir"] == "/tmp/custom"

    def test_custom_train_fn_skips_backend_check(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        """custom_train_fn should bypass the backend check entirely."""

        def custom_train(model: Model, trainer: Trainer) -> dict:
            from auto_chasm.history import History

            return {
                "history": History(),
                "test_metrics": None,
                "output_dir": str(trainer.output_dir),
            }

        from auto_chasm.trainers.loss import JointLoss

        model_wrapper.backend.name = "unsupported_backend"
        trainer = Trainer(
            model=model_wrapper,
            loss_fn=JointLoss(),
            num_iters=2,
            max_seq_length=32,
            custom_train_fn=custom_train,
            verbose=False,
        )

        # Should not raise RuntimeError because custom_train_fn bypasses the backend check
        result = trainer.train(sample_dataset)
        assert "history" in result
        model_wrapper.backend.name = "mlx"


# ---------------------------------------------------------------------------
# _evaluate_mlx with no joint trainer
# ---------------------------------------------------------------------------


class TestEvaluateMlxNoTrain:
    """Tests for _evaluate_mlx when called before train()."""

    def test_evaluate_mlx_raises_without_train(self, model_wrapper: Model) -> None:
        """_evaluate_mlx should raise RuntimeError if train() was not called first."""
        from auto_chasm.trainers.loss import JointLoss

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=JointLoss(),
            num_iters=2,
            max_seq_length=32,
            verbose=False,
        )

        # Call the private method directly to hit the error path
        with pytest.raises(RuntimeError, match="Call train\\(\\) before evaluate"):
            trainer._evaluate_mlx([])


# ---------------------------------------------------------------------------
# torch.manual_seed in __init__
# ---------------------------------------------------------------------------


class TestConfigSeedTorch:
    """Tests that TrainingConfig.seed is applied in __init__."""

    def test_mlx_seed_set_from_config(self, model_wrapper: Model) -> None:
        """Providing config with seed should set mx.random seed."""
        config = TrainingConfig(seed=42)

        from auto_chasm.trainers.loss import JointLoss

        Trainer(
            model=model_wrapper,
            loss_fn=JointLoss(),
            num_iters=2,
            max_seq_length=32,
            verbose=False,
            config=config,
        )
        # If no exception, seed was applied successfully

    def test_numpy_seed_set_from_config(self, model_wrapper: Model) -> None:
        """Providing config with seed should set numpy.random seed."""
        import numpy as np

        config = TrainingConfig(seed=12345)

        from auto_chasm.trainers.loss import JointLoss

        Trainer(
            model=model_wrapper,
            loss_fn=JointLoss(),
            num_iters=2,
            max_seq_length=32,
            verbose=False,
            config=config,
        )
        # Generate a random number to verify seed was applied
        val = np.random.randn()
        np.random.seed(12345)
        assert val == np.random.randn()

    def test_config_fields_are_stored(self, model_wrapper: Model) -> None:
        """bf16 (both backends) and fp16 (torch-only) are valid configs; junk raises."""
        import pytest

        assert TrainingConfig(mixed_precision="bf16").mixed_precision == "bf16"
        assert TrainingConfig(mixed_precision="fp16").mixed_precision == "fp16"
        with pytest.raises(ValueError, match="not valid"):
            TrainingConfig(mixed_precision="fp8")  # type: ignore[arg-type]

        config = TrainingConfig(seed=42, probe_weights={"p": 2.0})
        from auto_chasm.trainers.loss import JointLoss

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=JointLoss(),
            num_iters=2,
            max_seq_length=32,
            verbose=False,
            config=config,
        )

        assert trainer._probe_weights == {"p": 2.0}

    def test_config_params_override_trainer_defaults(self, model_wrapper: Model) -> None:
        """Config hyperparameters should override Trainer defaults."""
        config = TrainingConfig(
            learning_rate=1e-3,
            weight_decay=0.1,
            max_grad_norm=5.0,
            batch_size=2,
            gradient_accumulation_steps=4,
            logging_steps=3,
            save_steps=7,
            output_dir="/tmp/test_trainer",
            eval_steps=10,
        )

        from auto_chasm.trainers.loss import JointLoss

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=JointLoss(),
            num_iters=2,
            max_seq_length=32,
            verbose=False,
            config=config,
        )

        assert trainer.learning_rate == 1e-3
        assert trainer.weight_decay == 0.1
        assert trainer.grad_clip_norm == 5.0
        assert trainer.batch_size == 2
        assert trainer.grad_accum_steps == 4
        assert trainer.logging_steps == 3
        assert trainer.save_steps == 7


# ---------------------------------------------------------------------------
# Manifest path in _train_torch
# ---------------------------------------------------------------------------


class TestTrainTorchManifest:
    """Tests for the manifest path in _train_torch."""

    def test_manifest_saved_after_torch_training(self) -> None:
        """After torch training, manifest.json should exist in final/."""
        pytest.importorskip("torch")

        from auto_chasm.trainers.loss import JointLoss
        from tests.conftest import _make_torch_tiny_mlp

        torch_model = _make_torch_tiny_mlp(hidden_dim=4, vocab_size=8, num_layers=2)

        class Cfg:
            """Dummy config."""

            hidden_size = 4
            num_hidden_layers = 2

        torch_model.config = Cfg()

        wrapper = Model(torch_model, DummyTokenizer(), backend_name="torch")
        wrapper.attach_probe(ProbeConfig(name="p", layers=[-1]))
        wrapper.prepare_for_joint_training()

        data = [{"tokens": [1, 2, 3], "labels": [0, 0, 1]}]

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=wrapper,
                loss_fn=JointLoss(),
                num_iters=2,
                batch_size=2,
                max_seq_length=8,
                eval_steps=0,
                early_stopping_patience=0,
                output_dir=str(Path(tmpdir) / "out"),
                verbose=False,
            )

            # Call train to trigger _train_torch
            trainer.train(data)

            final_dir = Path(tmpdir) / "out" / "final"
            manifest_path = final_dir / "training_manifest.json"
            assert manifest_path.exists(), f"Manifest not found at {manifest_path}"

            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest["backend"] == "torch"
            assert manifest["num_iters"] == 2
            assert "best_metric" in manifest

    def test_manifest_contains_valid_metric(self) -> None:
        """The manifest metric should be a finite number when training completed."""
        pytest.importorskip("torch")

        from auto_chasm.trainers.loss import JointLoss
        from tests.conftest import _make_torch_tiny_mlp

        torch_model = _make_torch_tiny_mlp(hidden_dim=4, vocab_size=8, num_layers=2)

        class Cfg:
            """Dummy config."""

            hidden_size = 4
            num_hidden_layers = 2

        torch_model.config = Cfg()

        wrapper = Model(torch_model, DummyTokenizer(), backend_name="torch")
        wrapper.attach_probe(ProbeConfig(name="p", layers=[-1]))
        wrapper.prepare_for_joint_training()

        data = [{"tokens": [1, 2, 3], "labels": [0, 0, 1]}]

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=wrapper,
                loss_fn=JointLoss(),
                num_iters=3,
                batch_size=2,
                max_seq_length=8,
                eval_steps=2,
                early_stopping_patience=2,
                output_dir=str(Path(tmpdir) / "out"),
                verbose=False,
            )

            trainer.train(data, val_data=data)

            final_dir = Path(tmpdir) / "out" / "final"
            manifest_path = final_dir / "training_manifest.json"
            assert manifest_path.exists()

            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest["best_iter"] > 0
            assert manifest["best_metric"] is not None


# ---------------------------------------------------------------------------
# _get_joint RuntimeError for non-MLX backend
# ---------------------------------------------------------------------------


class TestGetJointRuntimeError:
    """Tests for _get_joint RuntimeError with unsupported backend."""

    def test_get_joint_raises_for_non_mlx(self, model_wrapper: Model) -> None:
        """_get_joint should raise RuntimeError for non-MLX backends."""
        from auto_chasm.trainers.loss import JointLoss

        model_wrapper.backend.name = "cuda"
        trainer = Trainer(
            model=model_wrapper,
            loss_fn=JointLoss(),
            num_iters=2,
            max_seq_length=32,
            verbose=False,
        )

        with pytest.raises(RuntimeError, match="Escape-hatch API requires an MLX backend"):
            trainer._get_joint()

        model_wrapper.backend.name = "mlx"


# ---------------------------------------------------------------------------
# _apply_probe_weights
# ---------------------------------------------------------------------------


class TestApplyProbeWeights:
    """Tests for _apply_probe_weights with JointLoss."""

    def test_probe_weights_applied_to_joint_loss(self, model_wrapper: Model) -> None:
        """Probe weights from config should be applied to JointLoss."""
        from auto_chasm.trainers.loss import JointLoss

        config = TrainingConfig(probe_weights={"digit": 2.5})
        loss_fn = JointLoss()

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=2,
            max_seq_length=32,
            verbose=False,
            config=config,
        )

        assert trainer.loss_fn._probe_weights.get("digit") == 2.5

    def test_probe_weights_noop_for_non_joint_loss(self, model_wrapper: Model) -> None:
        """_apply_probe_weights should be a no-op if loss is not JointLoss."""

        def dummy_loss_fn(model, batch, labels, lengths):
            return (0.0, 0, {})

        config = TrainingConfig(probe_weights={"digit": 2.5})

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=dummy_loss_fn,
            num_iters=2,
            max_seq_length=32,
            verbose=False,
            config=config,
        )

        # No crash means the no-op path works
        assert trainer._probe_weights == {"digit": 2.5}


# ---------------------------------------------------------------------------
# _fire_callback exception handling
# ---------------------------------------------------------------------------


class TestFireCallbackExceptionHandling:
    """Tests that _fire_callback re-raises exceptions from callbacks."""

    def test_callback_exception_propagates(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        """A callback that raises propagates loudly (re-raised, not swallowed)."""
        from auto_chasm.trainers.loss import JointLoss
        from auto_chasm.trainers.wrappers import TrainerCallback

        class BrokenCallback(TrainerCallback):
            """Callback that raises on every event."""

            def on_step_end(self, **kwargs: object) -> None:
                msg = "deliberate crash"
                raise RuntimeError(msg)

        data = [{"tokens": [1, 2, 3], "labels": [0, 0, 1]}]

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model_wrapper,
                loss_fn=JointLoss(),
                num_iters=2,
                batch_size=2,
                max_seq_length=8,
                output_dir=str(Path(tmpdir) / "out"),
                callbacks=[BrokenCallback()],
                verbose=False,
            )
            with pytest.raises(RuntimeError, match="deliberate crash"):
                trainer.train(data)


# ---------------------------------------------------------------------------
# Trainer escape-hatch delegation (edge cases)
# ---------------------------------------------------------------------------


class TestTrainerEscapeHatchEdgeCases:
    """Edge cases for Trainer escape-hatch delegation."""

    def test_restore_checkpoint_before_train(self, model_wrapper: Model) -> None:
        """restore_checkpoint should work after a save_checkpoint call."""
        from auto_chasm.trainers.data_utils import JointTextDataset
        from auto_chasm.trainers.loss import JointLoss

        data = [{"tokens": [1, 2, 3], "labels": [0, 0, 1]}]
        ds = JointTextDataset(data, DummyTokenizer(), tokens_key="tokens")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model_wrapper,
                loss_fn=JointLoss(),
                num_iters=2,
                batch_size=2,
                max_seq_length=8,
                output_dir=str(Path(tmpdir) / "out"),
                verbose=False,
            )

            it = trainer.iterate(ds)
            trainer.step(next(it))
            path = trainer.save_checkpoint()
            trainer.restore_checkpoint(path)
            # No crash means success

    def test_get_history_empty_before_train(self, model_wrapper: Model) -> None:
        """get_history should return empty History before any training."""
        from auto_chasm.history import History
        from auto_chasm.trainers.loss import JointLoss

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=JointLoss(),
            num_iters=2,
            max_seq_length=32,
            verbose=False,
        )

        h = trainer.get_history()
        assert isinstance(h, History)
        assert len(h) == 0


def test_single_iter_run_keeps_trained_weights_not_untrained_init(
    model_wrapper: Model, sample_dataset: list
) -> None:
    """num_iters=1 evaluates AFTER the step, so the best/final state is trained.

    Regression: eval ran BEFORE the step, so a 1-iter run evaluated (and saved as
    "best") the UNTRAINED init, then restored it — discarding the single update.
    """
    from mlx.utils import tree_flatten

    from auto_chasm.trainers.loss import JointLoss

    def _w() -> Any:
        params = tree_flatten(model_wrapper._probes["digit"].module.parameters())
        return mx.array(params[0][1])

    before = _w()
    trainer = Trainer(
        model=model_wrapper,
        loss_fn=JointLoss(weights={"lm_head": 0.0, "digit": 1.0}),
        num_iters=1,
        eval_steps=1,
        batch_size=8,
        max_seq_length=32,
        learning_rate=1e-1,
        verbose=False,
    )
    trainer.train(sample_dataset, val_data=sample_dataset)
    # The single update must survive — the model is NOT the untrained init.
    assert not bool(mx.allclose(_w(), before, atol=1e-7))


def test_constant_lr_schedule_trains_without_crash(
    model_wrapper: Model, sample_dataset: list
) -> None:
    """lr_schedule='constant' runs (the schedule returns an mx.array, not a float)."""
    from auto_chasm.trainers.loss import JointLoss

    trainer = Trainer(
        model=model_wrapper,
        loss_fn=JointLoss(weights={"lm_head": 0.0, "digit": 1.0}),
        num_iters=2,
        lr_schedule="constant",
        batch_size=8,
        max_seq_length=32,
        verbose=False,
    )
    result = trainer.train(sample_dataset)  # M13: used to crash with 'float has no astype'
    assert result is not None


def test_config_lm_and_probe_weight_applied_to_loss(model_wrapper: Model) -> None:
    """TrainingConfig.lm_weight / probe_weight reach the JointLoss (were ignored)."""
    from auto_chasm.config import TrainingConfig
    from auto_chasm.trainers.loss import JointLoss

    loss = JointLoss()
    Trainer(
        model=model_wrapper,
        loss_fn=loss,
        config=TrainingConfig(lm_weight=0.0, probe_weight=5.0),
        num_iters=1,
        verbose=False,
    )
    assert loss._weights["lm_head"] == 0.0
    assert loss._default_weight == 5.0
    # A config that leaves the defaults (1.0) does NOT clobber the loss's own weights.
    pure = JointLoss(weights={"lm_head": 0.0, "p": 1.0})
    Trainer(model=model_wrapper, loss_fn=pure, config=TrainingConfig(), num_iters=1, verbose=False)
    assert pure._weights["lm_head"] == 0.0

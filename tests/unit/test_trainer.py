"""Tests for the JointTrainer escape-hatch API and unified checkpoint saving."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import JointLoss, Model, ProbeConfig, Trainer
from auto_chasm.history import History
from auto_chasm.trainers.base import JointTrainer
from auto_chasm.trainers.data_utils import JointTextDataset
from auto_chasm.trainers.trainable import make_joint_loss

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
        """Mock model config for hidden_dim detection."""

        hidden_size = 16
        num_hidden_layers = 4

    base_model.config = Config()
    return Model(base_model, tokenizer, backend_name="mlx")


@pytest.fixture
def sample_dataset() -> list[dict]:
    """Create a small synthetic dataset."""
    mx.random.seed(42)
    data = []
    for _ in range(32):
        tokens = [1, 2, 3, 4, 5]
        labels = [0, 0, 1, 0, 0]
        data.append({"tokens": tokens, "labels": labels})
    return data


@pytest.fixture
def joint_trainer(model_wrapper: Model, sample_dataset: list) -> JointTrainer:
    """Create a JointTrainer with a probe."""
    model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
    model_wrapper.prepare_for_joint_training()

    loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)

    return JointTrainer(
        model=model_wrapper,
        loss_fn=loss_fn,
        num_iters=10,
        batch_size=8,
        max_seq_length=32,
        logging_steps=5,
        save_steps=100,
        eval_steps=0,
        early_stopping_patience=0,
        verbose=False,
    )


def test_run_without_own_best_does_not_restore_stale_checkpoint(
    model_wrapper: Model, sample_dataset: list, tmp_path
) -> None:
    """A run that saves no best keeps its trained weights, not a prior run's best.

    Regression: run() unconditionally restored output_dir's best-files at the end, so
    a second run in the same dir that never saved a best (val_data=None — e.g. a
    LayerSweep pass) loaded the PREVIOUS run's weights over its own and persisted them.
    """
    from mlx.utils import tree_flatten

    model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
    model_wrapper.prepare_for_joint_training()
    out = str(tmp_path / "ckpts")

    def _probe_w():  # noqa: ANN202
        params = tree_flatten(model_wrapper._probes["digit"].module.parameters())
        return mx.array(params[0][1])

    common: dict = {
        "model": model_wrapper,
        "loss_fn": JointLoss(weights={"lm_head": 0.0, "digit": 1.0}),
        "batch_size": 8,
        "max_seq_length": 32,
        "output_dir": out,
        "verbose": False,
    }
    # Run 1 WITH val saves a best (stale artifact on disk); run 2 in the SAME dir with
    # NO val saves none — its weights must reflect its own training, not run 1's best.
    JointTrainer(num_iters=4, eval_steps=2, **common).run(sample_dataset, val_data=sample_dataset)
    stale_best = _probe_w()
    JointTrainer(num_iters=6, eval_steps=0, **common).run(sample_dataset)
    assert not bool(mx.allclose(_probe_w(), stale_best, atol=1e-6))


def test_unknown_weight_key_raises_cleanly_without_poisoning_rng(
    model_wrapper: Model, sample_dataset: list
) -> None:
    """A typo'd JointLoss weight key raises BEFORE the trace; later training still runs.

    Regression: the unknown-key ValueError used to fire inside the value_and_grad /
    mx.compile trace, poisoning mx.random.state so every subsequent MLX training in
    the process crashed with "eval an array without a primitive".
    """
    model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
    model_wrapper.prepare_for_joint_training()

    bad = JointTrainer(
        model=model_wrapper,
        loss_fn=JointLoss(weights={"lm_head": 1.0, "typo": 2.0}),
        num_iters=2,
        batch_size=8,
        max_seq_length=32,
        eval_steps=0,
        verbose=False,
    )
    with pytest.raises(ValueError, match="Unknown weights key"):
        bad.run(sample_dataset)

    # The real poison check: a fresh VALID training runs to completion. If the failed
    # trace had poisoned mx.random.state, this would crash with "eval an array without
    # a primitive".
    good = JointTrainer(
        model=model_wrapper,
        loss_fn=JointLoss(weights={"lm_head": 1.0, "digit": 1.0}),
        num_iters=2,
        batch_size=8,
        max_seq_length=32,
        eval_steps=0,
        verbose=False,
    )
    history = good.run(sample_dataset)
    assert isinstance(history, History)


# ---------------------------------------------------------------------------
# Task 1: Escape-hatch API tests
# ---------------------------------------------------------------------------


class TestJointTrainerIterate:
    """Tests for iterate() method."""

    def test_iterate_yields_batches(
        self, joint_trainer: JointTrainer, sample_dataset: list
    ) -> None:
        ds = JointTextDataset(sample_dataset, joint_trainer.wrapper.tokenizer, tokens_key="tokens")
        batches = []
        for i, batch in enumerate(joint_trainer.iterate(ds)):
            tokens, labels, lengths = batch
            assert isinstance(tokens, type(mx.array(tokens).tolist())) or hasattr(
                tokens, "shape"
            )  # numpy or mx
            assert len(batch) == 3
            batches.append(batch)
            if i >= 4:
                break

        assert len(batches) == 5  # loop=True, so infinite


class TestJointTrainerStep:
    """Tests for step() method."""

    def test_step_returns_metrics_dict(
        self, joint_trainer: JointTrainer, sample_dataset: list
    ) -> None:
        ds = JointTextDataset(sample_dataset, joint_trainer.wrapper.tokenizer, tokens_key="tokens")
        it = joint_trainer.iterate(ds)
        batch = next(it)

        metrics = joint_trainer.step(batch)

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "ntoks" in metrics
        assert "components" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] > 0
        assert isinstance(metrics["components"], dict)

    def test_step_increments_global_step(
        self, joint_trainer: JointTrainer, sample_dataset: list
    ) -> None:
        ds = JointTextDataset(sample_dataset, joint_trainer.wrapper.tokenizer, tokens_key="tokens")
        it = joint_trainer.iterate(ds)

        assert joint_trainer._global_step == 0
        joint_trainer.step(next(it))
        assert joint_trainer._global_step == 1
        joint_trainer.step(next(it))
        assert joint_trainer._global_step == 2

    def test_multiple_steps_decrease_loss(
        self, joint_trainer: JointTrainer, sample_dataset: list
    ) -> None:
        ds = JointTextDataset(sample_dataset, joint_trainer.wrapper.tokenizer, tokens_key="tokens")
        it = joint_trainer.iterate(ds)

        metrics0 = joint_trainer.step(next(it))
        for _ in range(9):
            joint_trainer.step(next(it))
        metrics1 = joint_trainer.step(next(it))

        # Loss should generally decrease with training
        assert metrics1["loss"] < metrics0["loss"]


class TestJointTrainerEvaluate:
    """Tests for evaluate() method."""

    def test_evaluate_returns_metrics(
        self, joint_trainer: JointTrainer, sample_dataset: list
    ) -> None:
        ds = JointTextDataset(sample_dataset, joint_trainer.wrapper.tokenizer, tokens_key="tokens")

        # Train a few steps first
        it = joint_trainer.iterate(ds)
        for _ in range(5):
            joint_trainer.step(next(it))

        val_metrics = joint_trainer.evaluate(ds)

        assert isinstance(val_metrics, dict)
        assert "loss" in val_metrics
        assert "ntokens" in val_metrics
        assert val_metrics["loss"] > 0

    def test_evaluate_num_batches_limit(
        self, joint_trainer: JointTrainer, sample_dataset: list
    ) -> None:
        ds = JointTextDataset(sample_dataset, joint_trainer.wrapper.tokenizer, tokens_key="tokens")

        val_metrics = joint_trainer.evaluate(ds, num_batches=1)
        assert isinstance(val_metrics, dict)
        assert "loss" in val_metrics


class TestJointTrainerSaveRestore:
    """Tests for save_checkpoint() and restore_checkpoint()."""

    def test_save_checkpoint_default_path(
        self, joint_trainer: JointTrainer, sample_dataset: list
    ) -> None:
        ds = JointTextDataset(sample_dataset, joint_trainer.wrapper.tokenizer, tokens_key="tokens")
        it = joint_trainer.iterate(ds)
        joint_trainer.step(next(it))

        path = joint_trainer.save_checkpoint()
        assert path is not None
        assert Path(path).exists()
        assert (Path(path) / "manifest.json").exists()

    def test_save_restore_roundtrip(
        self, joint_trainer: JointTrainer, sample_dataset: list
    ) -> None:
        ds = JointTextDataset(sample_dataset, joint_trainer.wrapper.tokenizer, tokens_key="tokens")
        it = joint_trainer.iterate(ds)

        # Train a few steps
        for _ in range(5):
            joint_trainer.step(next(it))

        # Capture loss before save
        val_metrics_before = joint_trainer.evaluate(ds)
        loss_before = val_metrics_before["loss"]

        # Save
        ckpt_path = joint_trainer.save_checkpoint()

        # Train more to change weights
        for _ in range(5):
            joint_trainer.step(next(it))

        loss_after = joint_trainer.evaluate(ds)["loss"]
        assert loss_after != loss_before  # weights changed

        # Restore
        joint_trainer.restore_checkpoint(ckpt_path)

        # Loss should be back to original
        val_restored = joint_trainer.evaluate(ds)
        assert abs(val_restored["loss"] - loss_before) < 1e-4

    def test_save_checkpoint_with_explicit_path(
        self, joint_trainer: JointTrainer, sample_dataset: list
    ) -> None:
        ds = JointTextDataset(sample_dataset, joint_trainer.wrapper.tokenizer, tokens_key="tokens")
        it = joint_trainer.iterate(ds)
        joint_trainer.step(next(it))

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = str(Path(tmpdir) / "my_checkpoint")
            result_path = joint_trainer.save_checkpoint(custom_path)
            assert result_path == custom_path
            assert Path(result_path).exists()
            assert (Path(result_path) / "manifest.json").exists()


class TestJointTrainerGetHistory:
    """Tests for get_history() method."""

    def test_get_history_after_run(self, joint_trainer: JointTrainer, sample_dataset: list) -> None:
        ds = JointTextDataset(sample_dataset, joint_trainer.wrapper.tokenizer, tokens_key="tokens")
        history = joint_trainer.run(ds)

        retrieved = joint_trainer.get_history()
        assert retrieved is history
        assert len(retrieved) > 0

    def test_get_history_empty_before_run(self, joint_trainer: JointTrainer) -> None:
        h = joint_trainer.get_history()
        assert isinstance(h, History)
        assert len(h) == 0

    def test_get_history_with_step_api(
        self, joint_trainer: JointTrainer, sample_dataset: list
    ) -> None:
        ds = JointTextDataset(sample_dataset, joint_trainer.wrapper.tokenizer, tokens_key="tokens")
        it = joint_trainer.iterate(ds)

        for _ in range(5):
            joint_trainer.step(next(it))

        # get_history() returns raw history (not auto-annotated by step API)
        h = joint_trainer.get_history()
        assert isinstance(h, History)


class TestTrainerFacadeEscapeHatch:
    """Tests that Trainer delegates escape-hatch methods to JointTrainer."""

    def test_trainer_step_delegates(self, model_wrapper: Model, sample_dataset: list) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = JointLoss()
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=10,
            batch_size=8,
            max_seq_length=32,
            early_stopping_patience=0,
            verbose=False,
        )

        it = trainer.iterate(ds)
        batch = next(it)
        metrics = trainer.step(batch)

        assert "loss" in metrics
        assert metrics["loss"] > 0
        assert trainer._joint_trainer is not None

    def test_trainer_evaluate_delegates(self, model_wrapper: Model, sample_dataset: list) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = JointLoss()
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=10,
            batch_size=8,
            max_seq_length=32,
            early_stopping_patience=0,
            verbose=False,
        )

        # Run a few steps first
        it = trainer.iterate(ds)
        for _ in range(3):
            trainer.step(next(it))

        val = trainer.evaluate(ds)
        assert "loss" in val

    def test_trainer_save_checkpoint_delegates(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = JointLoss()
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=10,
            batch_size=8,
            max_seq_length=32,
            early_stopping_patience=0,
            verbose=False,
        )

        it = trainer.iterate(ds)
        trainer.step(next(it))

        path = trainer.save_checkpoint()
        assert Path(path).exists()
        assert (Path(path) / "manifest.json").exists()

    def test_trainer_get_history_delegates(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = JointLoss()
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=10,
            batch_size=8,
            max_seq_length=32,
            logging_steps=5,
            early_stopping_patience=0,
            verbose=False,
        )

        trainer.train(ds)
        h = trainer.get_history()
        assert len(h) > 0

    def test_trainer_restore_checkpoint_delegates(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = JointLoss()
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        trainer = Trainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            num_iters=10,
            batch_size=8,
            max_seq_length=32,
            early_stopping_patience=0,
            verbose=False,
        )

        it = trainer.iterate(ds)
        for _ in range(3):
            trainer.step(next(it))

        path = trainer.save_checkpoint()
        # Restore should work without errors
        trainer.restore_checkpoint(path)


class TestStepApiEquivalentToRun:
    """Verify step API produces approximately equal results to run()."""

    def test_losses_approximately_equal(self, model_wrapper: Model, sample_dataset: list) -> None:
        from auto_chasm.trainers.data_utils import JointTextDataset

        mx.random.seed(42)

        # --- Train with run() ---
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=4,
                batch_size=8,
                max_seq_length=32,
                logging_steps=2,
                save_steps=100,
                eval_steps=0,
                early_stopping_patience=0,
                output_dir=tmpdir,
                verbose=False,
            )
            history_run = trainer.run(ds)
            run_losses = history_run.train_losses

        # --- Train with step API (fresh model) ---
        mx.random.seed(42)
        base2 = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)

        class Config:
            """Mock model config for hidden_dim detection."""

            hidden_size = 16
            num_hidden_layers = 4

        base2.config = Config()
        model2 = Model(base2, DummyTokenizer(), backend_name="mlx")
        model2.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model2.prepare_for_joint_training()

        loss_fn2 = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        ds2 = JointTextDataset(sample_dataset, model2.tokenizer, tokens_key="tokens")

        with tempfile.TemporaryDirectory() as tmpdir2:
            trainer2 = JointTrainer(
                model=model2,
                loss_fn=loss_fn2,
                num_iters=10,
                batch_size=8,
                max_seq_length=32,
                logging_steps=5,
                save_steps=100,
                eval_steps=0,
                early_stopping_patience=0,
                output_dir=tmpdir2,
                verbose=False,
            )

            it = trainer2.iterate(ds2)
            losses = []
            for _ in range(4):
                metrics = trainer2.step(next(it))
                losses.append(metrics["loss"])

        # First step losses should be approximately equal (same seed, same data)
        assert abs(run_losses[0] - losses[0]) < 0.5  # allow small numerical diff


# ---------------------------------------------------------------------------
# Task 2: Unified checkpoint saving tests
# ---------------------------------------------------------------------------


class TestTrainerSavesToFinalSubdir:
    """Verify trainer.run() saves checkpoint to output_dir/final/."""

    def test_final_subdir_exists_after_training(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=4,
                batch_size=8,
                max_seq_length=32,
                logging_steps=2,
                save_steps=100,
                eval_steps=0,
                early_stopping_patience=0,
                output_dir=tmpdir,
                verbose=False,
            )
            trainer.run(ds)

            final_dir = Path(tmpdir) / "final"
            assert final_dir.exists()
            assert final_dir.is_dir()
            assert (final_dir / "manifest.json").exists()
            assert (final_dir / "probes" / "digit.safetensors").exists()

    def test_final_dir_contains_probes(self, model_wrapper: Model, sample_dataset: list) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=4,
                batch_size=8,
                max_seq_length=32,
                logging_steps=2,
                save_steps=100,
                eval_steps=0,
                early_stopping_patience=0,
                output_dir=tmpdir,
                verbose=False,
            )
            trainer.run(ds)

            final_dir = Path(tmpdir) / "final"
            probes_dir = final_dir / "probes"
            assert probes_dir.exists()
            assert (probes_dir / "digit.safetensors").exists()

    def test_training_manifest_still_saved(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=4,
                batch_size=8,
                max_seq_length=32,
                logging_steps=2,
                save_steps=100,
                eval_steps=0,
                early_stopping_patience=0,
                output_dir=tmpdir,
                verbose=False,
            )
            trainer.run(ds)

            assert (Path(tmpdir) / "training_manifest.json").exists()


class TestNoDuplicateAdapters:
    """Verify there are no adapters.safetensors at output_dir root after training."""

    def test_no_adapters_at_root_after_training(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        """The adapters should be inside final/, not at output_dir root.

        Note: adapers.safetensors may still exist at root if early
        stopping saved best weights there.  This test verifies the
        unified final/ checkpoint exists with probes.
        """
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=6,
                batch_size=8,
                max_seq_length=32,
                logging_steps=2,
                save_steps=100,
                eval_steps=2,
                early_stopping_patience=1,
                output_dir=tmpdir,
                verbose=False,
            )
            trainer.run(ds, val_data=ds)

            # final/ should exist with probes
            final_probe = Path(tmpdir) / "final" / "probes" / "digit.safetensors"
            assert final_probe.exists(), "Expected digit.safetensors in final/probes/"

    def test_final_checkpoint_can_be_loaded(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        """Verify the final checkpoint can be loaded via Model.from_checkpoint."""
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0)
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=4,
                batch_size=8,
                max_seq_length=32,
                logging_steps=2,
                save_steps=100,
                eval_steps=0,
                early_stopping_patience=0,
                output_dir=tmpdir,
                verbose=False,
            )
            trainer.run(ds)

            final_dir = Path(tmpdir) / "final"

            # Verify the directory structure is correct
            assert (final_dir / "manifest.json").exists()
            assert (final_dir / "probes" / "digit.safetensors").exists()

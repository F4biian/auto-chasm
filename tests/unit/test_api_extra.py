"""Extra tests for the new API: manifest, history, and generic loss."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from auto_chasm import JointLoss, Model, ProbeConfig
from auto_chasm.checkpoint import load_training_manifest
from auto_chasm.history import History, HistoryEntry

# ---------------------------------------------------------------------------
# Tiny model helpers (same as test_new_api.py)
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


# ---------------------------------------------------------------------------
# Training manifest and checkpoint integration
# ---------------------------------------------------------------------------


class TestTrainingManifest:
    """Tests for training manifest save/load."""

    def test_manifest_written_after_training(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        from auto_chasm.trainers.base import JointTrainer
        from auto_chasm.trainers.data_utils import JointTextDataset
        from auto_chasm.trainers.trainable import make_joint_loss as make_mlx_loss

        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = make_mlx_loss(lm_weight=1.0, probe_weight=1.0)
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

            manifest_path = Path(tmpdir) / "training_manifest.json"
            assert manifest_path.exists()

            manifest = json.loads(manifest_path.read_text())
            assert "best_iter" in manifest
            assert "best_metric" in manifest
            assert "best_metric_name" in manifest
            assert manifest["num_iters"] == 4
            assert manifest["min_delta"] == 1e-4

    def test_keep_best_only_cleans_periodic(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        from auto_chasm.trainers.base import JointTrainer
        from auto_chasm.trainers.data_utils import JointTextDataset
        from auto_chasm.trainers.trainable import make_joint_loss as make_mlx_loss

        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = make_mlx_loss(lm_weight=1.0, probe_weight=1.0)
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=6,
                batch_size=8,
                max_seq_length=32,
                logging_steps=2,
                save_steps=2,
                eval_steps=2,
                early_stopping_patience=0,
                keep_best_only=True,
                output_dir=tmpdir,
                verbose=False,
            )
            trainer.run(ds, val_data=ds)

            # Periodic checkpoints should be cleaned up
            files = list(Path(tmpdir).iterdir())
            periodic = [f for f in files if f.name[0].isdigit()]
            assert len(periodic) == 0, f"Expected no periodic checkpoints, found: {periodic}"

            # Best checkpoint should still exist
            assert (Path(tmpdir) / "adapters.safetensors").exists()

    def test_eval_steps_decoupled_from_save_steps(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        from auto_chasm.trainers.base import JointTrainer
        from auto_chasm.trainers.data_utils import JointTextDataset
        from auto_chasm.trainers.trainable import make_joint_loss as make_mlx_loss

        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = make_mlx_loss(lm_weight=1.0, probe_weight=1.0)
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=6,
                batch_size=8,
                max_seq_length=32,
                logging_steps=2,
                save_steps=100,  # never save periodically
                eval_steps=2,  # but evaluate every 2 steps
                early_stopping_patience=0,
                output_dir=tmpdir,
                verbose=False,
            )
            history = trainer.run(ds, val_data=ds)

            # Should have run validation (eval_steps=2, so at iter 2, 4, 6)
            assert len(history.val_losses) > 0

    def test_patience_zero_disables_early_stopping(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        """patience=0 should disable early stopping, not trigger immediately."""
        from auto_chasm.trainers.base import JointTrainer
        from auto_chasm.trainers.data_utils import JointTextDataset
        from auto_chasm.trainers.trainable import make_joint_loss as make_mlx_loss

        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = make_mlx_loss(lm_weight=1.0, probe_weight=1.0)
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
                early_stopping_patience=0,
                output_dir=tmpdir,
                verbose=False,
            )
            history = trainer.run(ds, val_data=ds)

            # With patience=0, training should run to completion
            # (early stopping disabled), not stop immediately
            assert len(history.train_losses) >= 2

    def test_min_delta_controls_improvement(
        self, model_wrapper: Model, sample_dataset: list
    ) -> None:
        """A very large min_delta should prevent any 'improvement'."""
        from auto_chasm.trainers.base import JointTrainer
        from auto_chasm.trainers.data_utils import JointTextDataset
        from auto_chasm.trainers.trainable import make_joint_loss as make_mlx_loss

        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = make_mlx_loss(lm_weight=1.0, probe_weight=1.0)
        ds = JointTextDataset(sample_dataset, model_wrapper.tokenizer, tokens_key="tokens")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JointTrainer(
                model=model_wrapper,
                loss_fn=loss_fn,
                num_iters=10,
                batch_size=8,
                max_seq_length=32,
                logging_steps=2,
                save_steps=100,
                eval_steps=2,
                early_stopping_patience=1,
                early_stopping_metric="val_loss",
                min_delta=9999.0,
                output_dir=tmpdir,
                verbose=False,
            )
            history = trainer.run(ds, val_data=ds)

            # With min_delta=9999, nothing counts as improvement,
            # so early stopping triggers after patience=1 eval rounds
            # Training should have stopped early
            assert len(history.train_losses) < 5


class TestLoadTrainingManifest:
    """Tests for load_training_manifest utility."""

    def test_returns_none_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_training_manifest(tmpdir)
            assert result is None

    def test_finds_manifest_in_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {"best_iter": 42, "best_metric": 0.123}
            (Path(tmpdir) / "training_manifest.json").write_text(json.dumps(manifest))

            result = load_training_manifest(tmpdir)
            assert result is not None
            assert result["best_iter"] == 42

    def test_finds_manifest_in_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "adapters"
            sub.mkdir()
            manifest = {"best_iter": 99, "best_metric": 0.05}
            (sub / "training_manifest.json").write_text(json.dumps(manifest))

            result = load_training_manifest(tmpdir)
            assert result is not None
            assert result["best_iter"] == 99


# ---------------------------------------------------------------------------
# History tests
# ---------------------------------------------------------------------------


class TestHistory:
    """Tests for the History and HistoryEntry classes."""

    def test_empty_history(self) -> None:
        h = History()
        assert len(h) == 0
        assert h.train_losses == []
        assert h.val_losses == []
        assert h.last() is None
        assert h.best_val_loss() is None

    def test_append_and_access(self) -> None:
        h = History()
        h.append(HistoryEntry(step=10, train_loss=0.5))
        h.append(HistoryEntry(step=20, train_loss=0.3))
        assert len(h) == 2
        assert h[0].step == 10
        assert h[1].step == 20
        assert h.train_losses == [0.5, 0.3]

    def test_val_losses_filters_none(self) -> None:
        h = History()
        h.append(HistoryEntry(step=10, train_loss=0.5))
        h.append(HistoryEntry(step=20, train_loss=0.3, val_loss=0.4))
        h.append(HistoryEntry(step=30, train_loss=0.2))
        assert h.val_losses == [0.4]
        assert h.val_steps == [20]

    def test_component_series(self) -> None:
        h = History()
        h.append(HistoryEntry(step=10, loss_components={"lm_ce": 0.5, "probe_bce": 0.3}))
        h.append(HistoryEntry(step=20, loss_components={"lm_ce": 0.4, "probe_bce": 0.2}))
        assert h.component_series("lm_ce") == [0.5, 0.4]
        assert h.component_series("probe_bce") == [0.3, 0.2]
        assert h.component_series("nonexistent") == []

    def test_metric_series(self) -> None:
        h = History()
        h.append(HistoryEntry(step=10, val_metrics={"accuracy": 0.8, "f1": 0.7}))
        h.append(HistoryEntry(step=20, val_metrics={"accuracy": 0.9, "f1": 0.85}))
        assert h.metric_series("accuracy") == [0.8, 0.9]
        assert h.metric_series("f1") == [0.7, 0.85]

    def test_best_val_loss(self) -> None:
        h = History()
        h.append(HistoryEntry(step=10, val_loss=0.5))
        h.append(HistoryEntry(step=20, val_loss=0.3))
        h.append(HistoryEntry(step=30, val_loss=0.4))
        best = h.best_val_loss()
        assert best is not None
        assert best.step == 20
        assert best.val_loss == 0.3

    def test_best_val_metric(self) -> None:
        h = History()
        h.append(HistoryEntry(step=10, val_metrics={"f1": 0.5}))
        h.append(HistoryEntry(step=20, val_metrics={"f1": 0.8}))
        h.append(HistoryEntry(step=30, val_metrics={"f1": 0.6}))
        best = h.best_val_metric("f1", higher_is_better=True)
        assert best is not None
        assert best[0] == 20
        assert best[1] == 0.8

    def test_iteration(self) -> None:
        h = History()
        h.append(HistoryEntry(step=1))
        h.append(HistoryEntry(step=2))
        steps = [e.step for e in h]
        assert steps == [1, 2]

    def test_repr(self) -> None:
        h = History()
        h.append(HistoryEntry(step=1))
        assert "1 entries" in repr(h)

    def test_json_roundtrip(self) -> None:
        h = History()
        h.append(
            HistoryEntry(
                step=10,
                train_loss=0.5,
                loss_components={"lm_ce": 0.4, "probe_bce": 0.1},
                learning_rate=2e-4,
            )
        )
        h.append(
            HistoryEntry(
                step=20,
                train_loss=0.3,
                val_loss=0.35,
                val_metrics={"accuracy": 0.9, "f1": 0.85},
                loss_components={"lm_ce": 0.25, "probe_bce": 0.05},
            )
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        h.save_json(path)
        h2 = History.load_json(path)

        assert len(h2) == 2
        assert h2[0].step == 10
        assert h2[0].train_loss == 0.5
        assert h2[0].loss_components == {"lm_ce": 0.4, "probe_bce": 0.1}
        assert h2[1].val_loss == 0.35
        assert h2[1].val_metrics["f1"] == 0.85
        assert h2.train_losses == [0.5, 0.3]
        assert h2.val_losses == [0.35]

        Path(path).unlink()

    def test_entry_to_dict_from_dict(self) -> None:
        entry = HistoryEntry(
            step=42,
            train_loss=1.23,
            loss_components={"lm_ce": 1.0, "probe_bce": 0.23},
            wall_time=10.5,
        )
        d = entry.to_dict()
        assert d["step"] == 42
        assert d["train_loss"] == 1.23

        restored = HistoryEntry.from_dict(d)
        assert restored.step == 42
        assert restored.loss_components == {"lm_ce": 1.0, "probe_bce": 0.23}


# ---------------------------------------------------------------------------
# Generic loss component tests
# ---------------------------------------------------------------------------


class TestGenericLoss:
    """Tests for the generic (non-hardcoded) loss components."""

    def test_mse_probe_loss(self) -> None:
        """JointLoss with probe_loss='mse' should work."""
        from auto_chasm.trainers.trainable import _TrainableModel

        loss_fn = JointLoss(losses={"probe": "mse"})
        base = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)
        train_model = _TrainableModel(base, {})

        batch = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[0, 0, 1, 0, 0]])
        lengths = mx.array([[0, 5]])

        total, ntoks, components = loss_fn(train_model, batch, labels, lengths)
        assert total.ndim == 0
        assert float(ntoks) > 0
        assert "lm_head" in components

    def test_pure_classifier_mode(self) -> None:
        """lm_weight=0 should skip LM loss entirely."""
        from auto_chasm.trainers.trainable import _TrainableModel

        loss_fn = JointLoss(weights={"lm_head": 0.0})
        base = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)
        train_model = _TrainableModel(base, {})

        batch = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[0, 0, 1, 0, 0]])
        lengths = mx.array([[0, 5]])

        total, ntoks, components = loss_fn(train_model, batch, labels, lengths)
        assert "lm_head" not in components
        assert total.ndim == 0

    def test_components_dict_dynamic(self) -> None:
        """Components dict should have dynamic keys, not fixed 'ce'/'bce'."""
        from auto_chasm.trainers.trainable import _TrainableModel

        loss_fn = JointLoss()
        base = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)
        train_model = _TrainableModel(base, {})

        batch = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[0, 0, 1, 0, 0]])
        lengths = mx.array([[0, 5]])

        _, _, components = loss_fn(train_model, batch, labels, lengths)
        # Should NOT have 'ce' or 'bce' keys — those are the old hardcoded names
        assert "ce" not in components
        assert "bce" not in components
        # Should have the new generic names
        assert "lm_head" in components

    def test_history_with_generic_components(self) -> None:
        """History should track loss_components with dynamic keys."""
        from auto_chasm.trainers.base import JointTrainer
        from auto_chasm.trainers.data_utils import JointTextDataset

        model_wrapper = Model(TinyMlp(), DummyTokenizer(), "mlx")
        model_wrapper.model.config = type("C", (), {"hidden_size": 16, "num_hidden_layers": 4})()
        model_wrapper.attach_probe(ProbeConfig(name="digit", layers=[-1]))
        model_wrapper.prepare_for_joint_training()

        loss_fn = JointLoss(weights={"digit": 0.5})
        data = [{"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 0, 0]}] * 32
        ds = JointTextDataset(data, model_wrapper.tokenizer, tokens_key="tokens")

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
            history = trainer.run(ds)

            # Every entry should have loss_components
            for entry in history:
                assert "lm_head" in entry.loss_components
                # No hardcoded 'ce' or 'bce' keys
                assert "ce" not in entry.loss_components
                assert "bce" not in entry.loss_components

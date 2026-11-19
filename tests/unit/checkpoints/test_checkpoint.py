"""Tests for checkpoint — export, import, load adapters/probes, training manifest."""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.checkpoint import (
    export_checkpoint,
    import_checkpoint,
    load_checkpoint,
    load_training_manifest,
)
from auto_chasm.config import ProbeConfig, SteeringConfig
from auto_chasm.model import Model


class TinyMlp(nn.Module):
    """Tiny MLP for checkpoint testing."""

    def __init__(self, hidden_dim: int = 8, vocab_size: int = 16, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class DummyTokenizer:
    """Minimal tokenizer for testing."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 16 for c in text[:5]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids if i > 0)


class TestExportImport:
    """Tests for export_checkpoint and import_checkpoint."""

    def test_export_import_roundtrip(self) -> None:
        base = TinyMlp()
        tokenizer = DummyTokenizer()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        model = Model(base, tokenizer, "mlx")
        model._base_model_name = "test-model"

        # Save a checkpoint first
        model.attach_probe(ProbeConfig(name="test_probe", layers=[1]))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            model.save_checkpoint(str(ckpt))

            # Export
            archive = Path(tmp) / "exported.auto_chasm"
            export_checkpoint(str(ckpt), str(archive))
            assert archive.exists()
            assert archive.stat().st_size > 0

            # Import back
            out = Path(tmp) / "imported"
            result_path = import_checkpoint(str(archive), str(out))

            assert Path(result_path).exists()
            assert (Path(result_path) / "manifest.json").exists()

    def test_export_creates_tarball(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "manifest.json").write_text('{"test": true}')

            archive = Path(tmp) / "out.auto_chasm"
            export_checkpoint(str(ckpt), str(archive))
            assert tarfile.is_tarfile(str(archive))

    def test_import_extracts_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "manifest.json").write_text('{"test": true}')

            archive = Path(tmp) / "out.auto_chasm"
            export_checkpoint(str(ckpt), str(archive))

            out = Path(tmp) / "restored"
            result = import_checkpoint(str(archive), str(out))

            # Should contain the manifest
            content = list(Path(result).iterdir())
            names = [f.name for f in content]
            assert "manifest.json" in names


class TestLoadTrainingManifest:
    """Tests for load_training_manifest edge cases."""

    def test_none_for_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = load_training_manifest(tmp)
            assert result is None

    def test_none_when_no_manifest_in_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / "subdir"
            sub.mkdir()
            result = load_training_manifest(tmp)
            assert result is None

    def test_reads_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {
                "base_model": "test-model",
                "best_iter": 100,
                "best_metric": 0.5,
            }
            (Path(tmp) / "training_manifest.json").write_text(json.dumps(manifest))
            result = load_training_manifest(tmp)
            assert result is not None
            assert result["best_iter"] == 100
            assert result["best_metric"] == 0.5


class TestSaveCheckpointWithSteering:
    """Test checkpoint save includes steering data."""

    def test_steering_data_in_manifest(self) -> None:
        base = TinyMlp()
        tokenizer = DummyTokenizer()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        model = Model(base, tokenizer, "mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        model.enable_steering(
            "p",
            config=SteeringConfig(method="nullify"),
            class_means={"mean_0": mx.array([1.0] * 8), "mean_1": mx.array([2.0] * 8)},
        )

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            model.save_checkpoint(str(ckpt))

            manifest = json.loads((ckpt / "manifest.json").read_text())
            assert "steering" in manifest
            assert "p" in manifest["steering"]
            steering_file = ckpt / "steering" / "p.json"
            assert steering_file.exists()

    def test_steering_data_roundtrip(self) -> None:
        base = TinyMlp()
        tokenizer = DummyTokenizer()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        model = Model(base, tokenizer, "mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        model.enable_steering(
            "p",
            config=SteeringConfig(method="push_to_mean", scale=2.0),
            class_means={"mean_0": mx.array([1.0] * 8), "mean_1": mx.array([2.0] * 8)},
        )

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            model.save_checkpoint(str(ckpt))

            steering_data = json.loads((ckpt / "steering" / "p.json").read_text())
            assert steering_data["probe_name"] == "p"
            assert steering_data["method"] == "push_to_mean"
            assert steering_data["scale"] == 2.0


class TestSaveCheckpointEdgeCases:
    """Edge cases for checkpoint saving."""

    def test_save_without_probes(self) -> None:
        base = TinyMlp()
        tokenizer = DummyTokenizer()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        model = Model(base, tokenizer, "mlx")

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            model.save_checkpoint(str(ckpt))

            manifest = json.loads((ckpt / "manifest.json").read_text())
            assert manifest["probes"] == {}
            assert manifest["steering"] == {}

    def test_save_with_aggregation_callable(self) -> None:
        base = TinyMlp()
        tokenizer = DummyTokenizer()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        model = Model(base, tokenizer, "mlx")

        def my_agg(states: list) -> mx.array:
            return states[0]

        model.attach_probe(ProbeConfig(name="p", layers=[0, 1], aggregation=my_agg))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            model.save_checkpoint(str(ckpt))
            manifest = json.loads((ckpt / "manifest.json").read_text())
            assert manifest["probes"]["p"]["aggregation"] == "__callable__"

    def test_load_checkpoint_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path/checkpoint")

    def test_full_finetune_save_warns_base_not_persisted(self, caplog) -> None:  # noqa: ANN001
        """A full fine-tune (unfrozen base, no LoRA) warns its base weights aren't saved."""
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[-1]))
        model.prepare_for_joint_training()
        model.unfreeze_model()  # full fine-tune: base is trainable

        import logging

        with tempfile.TemporaryDirectory() as tmp, caplog.at_level(logging.WARNING, "auto_chasm"):
            model.save_checkpoint(str(Path(tmp) / "ckpt"))
        assert "does not persist base-model weights" in caplog.text

        # A pure-probe model (base frozen) does NOT warn.
        caplog.clear()
        model.freeze_model()
        with tempfile.TemporaryDirectory() as tmp, caplog.at_level(logging.WARNING, "auto_chasm"):
            model.save_checkpoint(str(Path(tmp) / "ckpt2"))
        assert "does not persist base-model weights" not in caplog.text

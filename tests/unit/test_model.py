"""Tests for Model — save/load class means, checkpoint restore, edge cases."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import ProbeConfig
from auto_chasm.model import Model


class TinyMlp(nn.Module):
    """Tiny MLP for testing."""

    def __init__(self, hidden_dim: int = 8, vocab_size: int = 16, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self._hidden_dim = hidden_dim

    def __call__(self, x: mx.array) -> tuple:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return (self.output_proj(h),)

    @staticmethod
    def sanitize(weights):
        return weights


class DummyTokenizer:
    """Test helper."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "test"


class TestModelClassMeans:
    """Tests for class means save/load."""

    @pytest.fixture
    def model(self) -> Model:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        return Model(base, DummyTokenizer(), "mlx")

    def test_save_load_class_means_roundtrip(self, model) -> None:
        means = {
            "probe_a": {"mean_0": mx.array([1.0, 2.0, 3.0]), "mean_1": mx.array([4.0, 5.0, 6.0])}
        }

        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / "means.safetensors")
            model.save_class_means(means, p)
            assert Path(p).exists()

            loaded = model.load_class_means(p)
            assert isinstance(loaded, dict)
            assert "probe_a" in loaded
            assert "mean_0" in loaded["probe_a"]
            assert "mean_1" in loaded["probe_a"]

    def test_save_load_flat_means(self, model) -> None:
        means = {"mean_0": mx.array([1.0, 2.0]), "mean_1": mx.array([3.0, 4.0])}

        with tempfile.TemporaryDirectory() as tmp:
            p = str(Path(tmp) / "flat.safetensors")
            model.save_class_means(means, p)
            loaded = model.load_class_means(p)
            assert isinstance(loaded, dict)


class TestModelRestoration:
    """Tests for restore_original_layers."""

    def test_restore_with_no_probes(self) -> None:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        model.restore_original_layers()  # should not raise

    def test_restore_clears_original_layers(self) -> None:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))

        model.restore_original_layers()
        assert len(model._original_layers) == 0


class TestModelSteeringHooks:
    """Tests for steering hook management."""

    def test_enable_steering_missing_probe(self) -> None:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        with pytest.raises(KeyError, match="not attached"):
            model.enable_steering("nonexistent")

    def test_disable_steering_unknown_probe(self) -> None:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        model.disable_steering("nonexistent")  # should not raise


class TestModelProperties:
    """Tests for model property access."""

    def test_probes_dict(self) -> None:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        assert isinstance(model.probes, dict)
        assert len(model.probes) == 0

    def test_steering_hooks_dict(self) -> None:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8
            num_hidden_layers = 2

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        assert isinstance(model.steering_hooks, dict)


class TestModelFromPretrainedError:
    """Tests for Model.from_pretrained error handling."""

    def test_from_pretrained_mlx_load_error(self) -> None:
        """Loading a nonexistent model should raise."""
        with pytest.raises(Exception):
            Model.from_pretrained("nonexistent-model-xyz-12345", backend_name="mlx")

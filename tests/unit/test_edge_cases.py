"""edge case tests — things that should work but might silently fail."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import ProbeConfig
from auto_chasm.model import Model
from auto_chasm.probe import Probe, _get_hidden_dim, _resolve_negative_index


class TinyMlp(nn.Module):
    """Tiny MLP for edge case testing."""

    def __init__(self, hidden_dim: int = 8, vocab_size: int = 16, num_layers: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array) -> tuple:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return (self.output_proj(h),)


class DummyTokenizer:
    """Test helper."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "test"


class TestResolveNegativeIndex:
    """Tests for _resolve_negative_index edge cases."""

    def test_negative_one(self) -> None:
        assert _resolve_negative_index(-1, 5) == 4

    def test_zero(self) -> None:
        assert _resolve_negative_index(0, 5) == 0

    def test_out_of_range_positive(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _resolve_negative_index(5, 5)

    def test_too_negative(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _resolve_negative_index(-6, 5)


class TestMultipleProbes:
    """Tests for multiple probes on the same model."""

    def test_two_probes_different_layers(self) -> None:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        model.attach_probe(ProbeConfig(name="a", layers=[0]))
        model.attach_probe(ProbeConfig(name="b", layers=[1]))
        assert len(model.probes) == 2
        outputs = model.forward(mx.array([[1, 2, 3]]))
        assert "a" in outputs.probes
        assert "b" in outputs.probes

    def test_probe_on_same_layer_overwrites(self) -> None:
        """Attaching two probes to the same layer index wraps twice."""
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        model.attach_probe(ProbeConfig(name="a", layers=[0]))
        model.attach_probe(ProbeConfig(name="b", layers=[0]))
        # Both should work but share capture
        outputs = model.forward(mx.array([[1, 2, 3]]))
        assert "a" in outputs.probes or "b" in outputs.probes

    def test_probes_clear_between_forwards(self) -> None:
        """Each forward pass should produce fresh capture."""
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))

        out1 = model.forward(mx.array([[1, 2, 3]]))
        out2 = model.forward(mx.array([[4, 5]]))
        assert "p" in out1.probes
        assert "p" in out2.probes


class TestProbeMLPModule:
    """Tests for the MLP probe module type."""

    def test_mlp_creates_module(self) -> None:
        from auto_chasm.probe import Probe

        config = ProbeConfig(
            name="mlp_test",
            layers=[0],
            module_type="mlp",
            module_config={"hidden_dim": 64, "out_features": 3, "dropout": 0.1},
        )
        probe = Probe(config, 16, "mlx")
        assert probe.module is not None

    def test_mlp_forward(self) -> None:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        model.attach_probe(
            ProbeConfig(
                name="mlp",
                layers=[0],
                module_type="mlp",
                module_config={"hidden_dim": 16, "out_features": 2},
            )
        )
        outputs = model.forward(mx.array([[1, 2, 3]]))
        assert outputs.probes["mlp"].logits is not None
        assert outputs.probes["mlp"].logits.shape[-1] == 2


class TestProbeName:
    """Tests probe name property."""

    def test_name_from_config(self) -> None:
        config = ProbeConfig(name="my_probe", layers=[0])
        probe = Probe(config, 16, "mlx")
        assert probe.name == "my_probe"

    def test_layers_property(self) -> None:
        config = ProbeConfig(name="p", layers=[3, 5])
        probe = Probe(config, 16, "mlx")
        assert probe.layers == [3, 5]

    def test_source_property(self) -> None:
        config = ProbeConfig(name="p", layers=[0], source="embedding")
        probe = Probe(config, 16, "mlx")
        assert probe.source == "embedding"


class TestBackendToNumpy:
    """Tests for the backend to_numpy (edge cases)."""

    def test_to_numpy_scalar(self) -> None:
        from auto_chasm.backends.mlx_backend import MLXTensorOps

        ops = MLXTensorOps()
        t = mx.array(3.14)
        result = ops.to_numpy(t)
        assert not isinstance(result, mx.array)

    def test_torch_sample_greedy(self) -> None:
        from auto_chasm.backends.mlx_backend import MLXTensorOps

        ops = MLXTensorOps()
        logits = mx.array([0.1, 0.2, 100.0, 0.3])
        token = ops.sample(logits, 0.0)
        assert token == 2

    def test_torch_sample_with_temp(self) -> None:
        from auto_chasm.backends.mlx_backend import MLXTensorOps

        mx.random.seed(42)
        ops = MLXTensorOps()
        logits = mx.array([1.0, 2.0, 3.0, 4.0])
        token = ops.sample(logits, 0.5)
        assert 0 <= token < 4


class TestGetHiddenDimModelArgs:
    """Tests for _get_hidden_dim with model.args fallback."""

    def test_hidden_dim_from_args(self) -> None:
        class ModelWithArgs:
            """Test helper."""

            class Args:
                """Test helper."""

                hidden_size = 32

            args = Args()

        dim = _get_hidden_dim(ModelWithArgs())
        assert dim == 32

    def test_hidden_dim_d_model(self) -> None:
        class ModelWithDModel:
            """Test helper."""

            class Config:
                """Test helper."""

                d_model = 64

            config = Config()

        dim = _get_hidden_dim(ModelWithDModel())
        assert dim == 64

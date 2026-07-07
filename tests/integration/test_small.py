"""Integration tests — probe injection, training, steering.

Tests the full pipeline with a tiny model to verify end-to-end
correctness on MLX.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import ProbeConfig, SteeringConfig
from auto_chasm.model import Model
from auto_chasm.probe import LayerCapture, _find_layers, _get_hidden_dim
from auto_chasm.steering import SteeringHook


class TinyMlp(nn.Module):
    """A tiny 2-layer MLP for testing (no tokenizer needed)."""

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 32, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

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


class TestFindLayers:
    """Tests for _find_layers utility."""

    def test_finds_layers(self, tiny_model: tuple[TinyMlp, DummyTokenizer]) -> None:
        model, _ = tiny_model
        layers = _find_layers(model)
        assert layers is not None
        assert len(layers) == 4

    def test_returns_none_for_no_layers(self) -> None:
        class NoLayers(nn.Module):
            """Module with no sub-layers for testing ``_find_layers``."""

            def __call__(self, x: mx.array) -> mx.array:
                return x

        assert _find_layers(NoLayers()) is None


class TestGetHiddenDim:
    """Tests for _get_hidden_dim utility."""

    def test_from_config(self, tiny_model: tuple[TinyMlp, DummyTokenizer]) -> None:
        model, _ = tiny_model

        class Config:
            """Dummy configuration for testing."""

            hidden_size = 16

        model.config = Config()
        assert _get_hidden_dim(model) == 16

    def test_raises_without_config(self) -> None:
        class NoConfig(nn.Module):
            """Module without a ``config`` attribute for testing."""

            pass

        with pytest.raises(ValueError, match="Cannot determine"):
            _get_hidden_dim(NoConfig())


class TestProbeInjection:
    """Tests for probe injection into models."""

    def test_inject_at_layer(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="test", layers=[1])
        probe = model_wrapper.attach_probe(config)
        assert len(probe.layer_captures) == 1
        assert "test" in model_wrapper.probes

    def test_inject_multiple_layers(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="multi", layers=[0, 2])
        probe = model_wrapper.attach_probe(config)
        assert len(probe.layer_captures) == 2

    def test_inject_negative_index(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="neg", layers=[-1])
        probe = model_wrapper.attach_probe(config)
        assert len(probe.layer_captures) == 1

    def test_inject_invalid_layer_raises(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="bad", layers=[100])
        with pytest.raises(ValueError, match="out of range"):
            model_wrapper.attach_probe(config)

    def test_forward_captures_hidden(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="test", layers=[1])
        model_wrapper.attach_probe(config)

        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)

        assert outputs.lm_logits is not None
        assert "test" in outputs.probes
        assert outputs.probes["test"].logits is not None

    def test_probe_logits_shape(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="test", layers=[1])
        model_wrapper.attach_probe(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        outputs = model_wrapper.forward(input_ids)

        probe_logits = outputs.probes["test"].logits
        assert probe_logits.shape[0] == 1
        assert probe_logits.shape[1] == 5

    def test_multi_layer_aggregation_concat(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="multi", layers=[0, 1], aggregation="concat")
        model_wrapper.attach_probe(config)

        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)

        probe_logits = outputs.probes["multi"].logits
        assert probe_logits is not None

    def test_multi_layer_aggregation_mean(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="multi", layers=[0, 1], aggregation="mean")
        model_wrapper.attach_probe(config)

        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)

        probe_logits = outputs.probes["multi"].logits
        assert probe_logits is not None

    def test_custom_module_type(self, model_wrapper: Model) -> None:
        def custom_module(in_dim: int, cfg: dict) -> nn.Linear:
            return nn.Linear(in_dim, cfg.get("out_features", 1))

        config = ProbeConfig(
            name="custom",
            layers=[1],
            module_type=custom_module,
            module_config={"out_features": 1},
        )
        model_wrapper.attach_probe(config)

        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert "custom" in outputs.probes


class TestLayerCapture:
    """Tests for the LayerCapture wrapper."""

    def test_captures_output(self, tiny_model: tuple[TinyMlp, DummyTokenizer]) -> None:
        model, _ = tiny_model
        captured = []

        original = model.layers[0]
        wrapped = LayerCapture(
            original,
            layer_idx=0,
            capture_fn=lambda h: captured.append(h),
        )

        x = mx.array(
            [
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                ]
            ]
        )
        wrapped(x)

        assert len(captured) == 1

    def test_steering_modifies_output(self, tiny_model: tuple[TinyMlp, DummyTokenizer]) -> None:
        model, _ = tiny_model

        def steer_fn(hidden: mx.array, head: nn.Linear, logits: mx.array) -> mx.array:
            return hidden * 0.5

        original = model.layers[0]
        head = nn.Linear(16, 1)
        wrapped = LayerCapture(
            original,
            layer_idx=0,
            capture_fn=lambda h: None,
            steer_fn=steer_fn,
            binary_head=head,
        )

        x = mx.array([[1.0] * 16])
        result = wrapped(x)
        assert result is not None

    def test_make_layer_capture_mlx(self, tiny_model: tuple[TinyMlp, DummyTokenizer]) -> None:
        """make_layer_capture returns MLX LayerCapture for backend='mlx'."""
        from auto_chasm.probe import _MLXLayerCapture, make_layer_capture

        model, _ = tiny_model
        original = model.layers[0]
        capture = make_layer_capture(original, layer_idx=0, backend_name="mlx")
        assert isinstance(capture, _MLXLayerCapture)

    def test_make_layer_capture_torch(self) -> None:
        """make_layer_capture returns Torch LayerCapture for backend='torch'."""
        import torch.nn as tnn

        from auto_chasm.probe import _TorchLayerCapture, make_layer_capture

        linear = tnn.Linear(4, 2)
        capture = make_layer_capture(linear, layer_idx=0, backend_name="torch")
        assert isinstance(capture, _TorchLayerCapture)
        assert isinstance(capture, tnn.Module)

    def test_torch_layer_capture_forward(self) -> None:
        """Torch LayerCapture should capture hidden states correctly."""
        import torch
        import torch.nn as tnn

        from auto_chasm.probe import make_layer_capture

        linear = tnn.Linear(4, 2)
        captured = []
        wrap = make_layer_capture(
            linear, layer_idx=0, capture_fn=lambda h: captured.append(h), backend_name="torch"
        )
        x = torch.randn(1, 4)
        out = wrap(x)
        assert out.shape == (1, 2)
        assert len(captured) == 1

    def test_torch_layer_capture_steering(self) -> None:
        """Torch LayerCapture should apply steering."""
        import torch
        import torch.nn as tnn

        from auto_chasm.probe import make_layer_capture

        linear = tnn.Linear(4, 2)
        head = tnn.Linear(2, 1)

        def steer_fn(hidden, _head, _logits):  # type: ignore[no-untyped-def]
            return hidden * 0.5

        wrap = make_layer_capture(
            linear,
            layer_idx=0,
            capture_fn=lambda h: None,
            steer_fn=steer_fn,
            binary_head=head,
            backend_name="torch",
        )
        x = torch.randn(1, 4)
        result = wrap(x)
        assert result is not None

    def test_torch_layer_capture_in_module_list(self) -> None:
        """Torch LayerCapture can be assigned to nn.ModuleList."""
        import torch.nn as tnn

        from auto_chasm.probe import make_layer_capture

        layers = tnn.ModuleList([tnn.Linear(4, 2), tnn.Linear(4, 2)])
        original = layers[1]
        capture = make_layer_capture(original, layer_idx=1, backend_name="torch")
        layers[1] = capture
        assert layers[1] is capture


class TestSteeringHook:
    """Tests for SteeringHook."""

    def test_default_state(self) -> None:
        config = SteeringConfig()
        hook = SteeringHook("test", config)
        assert not hook.enabled
        assert not hook.has_geometry

    def test_enable_disable(self) -> None:
        config = SteeringConfig()
        hook = SteeringHook("test", config)
        hook.enable()
        assert hook.enabled
        hook.disable()
        assert not hook.enabled

    def test_noop_without_geometry(self) -> None:
        config = SteeringConfig()
        hook = SteeringHook("test", config)
        hook.enable()

        hidden = mx.array([[1.0, 2.0, 3.0]])
        head = nn.Linear(3, 1)
        logits = mx.array([[0.5]])

        result = hook.steer(hidden, head, logits)
        assert float(mx.sum(result - hidden).item()) == 0.0

    def test_serialization(self) -> None:
        config = SteeringConfig(method="push_to_mean", scale=2.0)
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([1.0, 2.0, 3.0])
        hook._mean_1 = mx.array([4.0, 5.0, 6.0])
        hook._head_norm = 1.5

        data = hook.to_dict()
        assert data["probe_name"] == "test"
        assert data["method"] == "push_to_mean"

        restored = SteeringHook.from_dict(data)
        assert restored.probe_name == "test"
        assert restored.has_geometry


class TestSteeringIntegration:
    """Integration tests for steering with the full model."""

    def test_enable_steering(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="test", layers=[1])
        model_wrapper.attach_probe(config)

        mean_0 = mx.array([1.0] * 16)
        mean_1 = mx.array([2.0] * 16)
        steer_config = SteeringConfig(method="nullify")
        model_wrapper.enable_steering(
            "test",
            config=steer_config,
            class_means={"mean_0": mean_0, "mean_1": mean_1},
        )

        assert "test" in model_wrapper.steering_hooks
        assert model_wrapper.steering_hooks["test"].enabled

    def test_disable_steering(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="test", layers=[1])
        model_wrapper.attach_probe(config)

        mean_0 = mx.array([1.0] * 16)
        mean_1 = mx.array([2.0] * 16)
        model_wrapper.enable_steering(
            "test",
            config=SteeringConfig(),
            class_means={"mean_0": mean_0, "mean_1": mean_1},
        )
        model_wrapper.disable_steering("test")
        assert not model_wrapper.steering_hooks["test"].enabled

    def test_forward_with_steering(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="test", layers=[1])
        model_wrapper.attach_probe(config)

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


class TestCheckpoint:
    """Tests for checkpoint save/load."""

    def test_save_and_load(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="test", layers=[1])
        model_wrapper.attach_probe(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint")
            model_wrapper.save_checkpoint(path)

            assert (Path(path) / "manifest.json").exists()
            assert (Path(path) / "probes").exists()
            assert (Path(path) / "probes" / "test.safetensors").exists()

            manifest_path = Path(path) / "manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)

            assert "probes" in manifest
            assert "test" in manifest["probes"]
            assert manifest["probes"]["test"]["layers"] == [1]

    def test_save_with_steering(self, model_wrapper: Model) -> None:
        config = ProbeConfig(name="test", layers=[1])
        model_wrapper.attach_probe(config)

        mean_0 = mx.array([1.0] * 16)
        mean_1 = mx.array([2.0] * 16)
        model_wrapper.enable_steering(
            "test",
            config=SteeringConfig(),
            class_means={"mean_0": mean_0, "mean_1": mean_1},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "checkpoint")
            model_wrapper.save_checkpoint(path)

            manifest_path = Path(path) / "manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)

            assert "steering" in manifest
            assert "test" in manifest["steering"]


class TestMakeJointLoss:
    """Tests for the MLX-specific make_joint_loss."""

    def test_returns_callable(self) -> None:
        from auto_chasm.trainers.trainable import make_joint_loss

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=0.5)
        assert callable(loss_fn)

    def test_signature_compatible_with_value_and_grad(self) -> None:
        from auto_chasm.trainers.trainable import _TrainableModel, make_joint_loss

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=0.5)
        base = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)
        train_model = _TrainableModel(base, {})

        batch = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[0, 0, 1, 0, 0]])
        lengths = mx.array([[0, 5]])

        result = loss_fn(train_model, batch, labels, lengths)
        assert len(result) == 3

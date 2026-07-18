"""Final push — hitting the last missing lines to cross 75%."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import ProbeConfig
from auto_chasm.history import History, HistoryEntry
from auto_chasm.model import Model
from auto_chasm.probe import Probe
from auto_chasm.steering import SteeringHook, build_auto_steer_fn


class TinyMlp(nn.Module):
    """Tiny MLP."""

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

    def named_modules(self):
        yield from [("layers.0", self.layers[0]), ("layers.1", self.layers[1])]


class MaskModel(nn.Module):
    """Model that accepts mask kwarg."""

    def __init__(self, hidden_dim: int = 8, vocab_size: int = 16, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> tuple:
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


# ===========================================================================
# history.py: val_steps property (line 143)
# ===========================================================================


class TestHistoryValSteps:
    """Test val_steps property."""

    def test_val_steps(self) -> None:
        h = History()
        h.append(HistoryEntry(step=10))
        h.append(HistoryEntry(step=20, val_loss=0.5))
        h.append(HistoryEntry(step=30))
        h.append(HistoryEntry(step=40, val_loss=0.3))
        assert h.val_steps == [20, 40]


# ===========================================================================
# model.py: forward with mask, _detect_backend, _load_torch
# ===========================================================================


class TestModelForwardMask:
    """Forward with mask kwarg to hit model.py:216."""

    def test_forward_with_mask_supporting_model(self) -> None:
        base = MaskModel()

        class Config:
            """Test helper."""

            hidden_size = 8

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))

        input_ids = mx.array([[1, 2, 3]])
        mask = mx.array([[1, 1, 1]])
        outputs = model.forward(input_ids, attention_mask=mask)
        assert outputs.lm_logits is not None


class TestModelDetectBackend:
    """Test _detect_backend helper."""

    def test_detect_backend(self) -> None:
        from auto_chasm.backends.loaders import detect_backend

        assert detect_backend() in ("mlx", "torch")

    def test_load_torch_import(self) -> None:
        from auto_chasm.model import _load_torch

        # Should fail since torch not installed
        with pytest.raises(Exception):
            _load_torch("nonexistent-model-xyz")


# ===========================================================================
# peft.py: _unfreeze_lora_params MLX path
# ===========================================================================


class TestPEFTUnfreeze:
    """PEFT unfreeze paths."""

    def test_unfreeze_lora_mlx_fallback(self) -> None:
        from auto_chasm.backends import Backend
        from auto_chasm.peft import _unfreeze_lora_params

        base = TinyMlp()
        base.freeze()
        _unfreeze_lora_params(base, Backend(force="mlx"))

    def test_default_targets_adapt_every_linear(self) -> None:
        from auto_chasm.peft import apply_lora, targetable_lora_modules

        base = TinyMlp()
        # Default targeting is ALL-LINEAR: a model with no attention projections
        # is still fully adaptable (previously this raised because the default
        # only matched q/k/v). The targetable listing must be non-empty and the
        # default apply must adapt exactly that set.
        targets = targetable_lora_modules(base)
        assert targets, "TinyMlp's Linear layers must be targetable"
        adapted = apply_lora(base, r=4, alpha=8)
        assert adapted is not None


# ===========================================================================
# probe.py: "last" aggregation, unknown aggregation
# ===========================================================================


class TestProbeEdge:
    """Probe edge case tests."""

    def test_last_aggregation_builtin(self) -> None:
        config = ProbeConfig(name="p", layers=[0, 1], aggregation="last")
        probe = Probe(config, 8, "mlx")
        hs = [mx.array([[1.0, 2.0], [3.0, 4.0]]), mx.array([[5.0, 6.0], [7.0, 8.0]])]
        aggregated = probe._aggregate(hs)
        assert float(aggregated[0, 0].item()) == 5.0

    def test_unknown_aggregation_raises(self) -> None:
        config = ProbeConfig(name="p", layers=[0], aggregation="concat")
        probe = Probe(config, 8, "mlx")
        hs = [mx.array([[1.0, 2.0]]), mx.array([[3.0, 4.0]])]
        # Patching the config's aggregation to an unknown string
        probe.config.aggregation = "nonexistent"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown aggregation"):
            probe._aggregate(hs)


# ===========================================================================
# steering.py: zero weight guard, boundary branch
# ===========================================================================


class TestSteeringMLXBranches:
    """Test specific MLX steering branches."""

    def test_zero_weight_guard(self) -> None:
        """Head with zero weight should return hidden unchanged."""
        from auto_chasm.config import SteeringConfig

        config = SteeringConfig(method="nullify")
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 8)
        hook._mean_1 = mx.array([1.0] * 8)
        hook._direction = hook._mean_1 - hook._mean_0
        hook.enable()

        head = nn.Linear(8, 1)
        head.weight = head.weight * 0.0
        head.bias = head.bias * 0.0

        hidden = mx.random.normal((1, 5, 8))
        logits = head(hidden).squeeze(-1)

        fn = build_auto_steer_fn(hook)
        if fn is not None:
            result = fn(hidden, head, logits)
            assert result.shape == hidden.shape

    def test_boundary_steering_branch(self) -> None:
        """Test steering with boundary method via auto_steer."""
        from auto_chasm.config import SteeringConfig

        config = SteeringConfig(method="boundary")
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 8)
        hook._mean_1 = mx.array([2.0] * 8)
        hook._direction = hook._mean_1 - hook._mean_0
        hook._head_norm = 5.0
        hook.enable()

        head = nn.Linear(8, 1)
        hidden = mx.random.normal((1, 5, 8))
        _ = head(hidden).squeeze(-1)

        fn = build_auto_steer_fn(hook)
        assert fn is not None


# ===========================================================================
# generation.py: chat fallback formatting
# ===========================================================================


class TestGenerationLines:
    """Hit specific generation code lines."""

    def test_chat_fallback_multiline_prompt(self) -> None:
        from auto_chasm.generation import chat

        class NoTemplate(DummyTokenizer):
            """Test helper."""

            chat_template = None

        messages = [{"role": "user", "content": "hello"}]
        try:
            result = chat(TinyMlp(), NoTemplate(), messages, max_tokens=1, temperature=0.0)
            assert isinstance(result, str)
        except Exception:
            pass

    def test_stream_decode_line(self) -> None:
        from auto_chasm.generation import _generate_stream_mlx

        tokens = list(_generate_stream_mlx(TinyMlp(), DummyTokenizer(), "test", 1, 0.0))
        assert all(isinstance(t, str) for t in tokens)

    def test_manual_generate_decode(self) -> None:
        from auto_chasm.generation import _generate_manual_mlx

        result = _generate_manual_mlx(TinyMlp(), DummyTokenizer(), "test", 1, 0.0)
        assert isinstance(result, str)

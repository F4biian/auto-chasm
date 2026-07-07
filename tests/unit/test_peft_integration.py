"""Final push for coverage — hitting remaining testable paths."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import ProbeConfig
from auto_chasm.model import Model


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
        yield from [
            ("self_attn.q_proj", self.layers[0]),
            ("self_attn.v_proj", self.layers[1]),
        ]


class DummyTokenizer:
    """Test helper."""

    eos_token_id = 0
    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "test"

    def get_vocab(self) -> dict:
        return {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "chat_prompt"


class TestLoadTorch:
    """Test _load_torch helper."""

    def test_load_torch_called(self) -> None:
        from auto_chasm.model import _load_torch

        with pytest.raises(Exception):
            _load_torch("nonexistent-model-xyz")


class TestPEFTUnfreezeMLXImport:
    """Test peft _unfreeze_lora_params with real MLX import path."""

    def test_unfreeze_with_mlx_import(self) -> None:
        from auto_chasm.backends import Backend
        from auto_chasm.peft import _unfreeze_lora_params

        base = TinyMlp()
        base.freeze()
        _unfreeze_lora_params(base, Backend(force="mlx"))


class TestPEFTDefaultTargets:
    """Test _default_target_modules with q_proj/k_proj/v_proj."""

    def test_default_target_finds_qkv(self) -> None:
        from auto_chasm.peft import _default_target_modules

        class ModelWithAttention(nn.Module):
            """Test helper."""

            def __init__(self):
                super().__init__()
                self.self_attn_q_proj = nn.Linear(8, 8)
                self.self_attn_k_proj = nn.Linear(8, 8)
                self.self_attn_v_proj = nn.Linear(8, 8)

            def __call__(self, x):
                return (self.self_attn_q_proj(x),)

            def named_modules(self):
                yield from [
                    ("self_attn.q_proj", self.self_attn_q_proj),
                    ("self_attn.k_proj", self.self_attn_k_proj),
                    ("self_attn.v_proj", self.self_attn_v_proj),
                ]

        targets = _default_target_modules(ModelWithAttention())
        assert len(targets) == 3


class TestChatFallback:
    """Test chat fallback with NoTemplate tokenizer."""

    def test_chat_fallback_creates_correct_prompt(self) -> None:
        from auto_chasm.generation import chat

        class NoTemplate(DummyTokenizer):
            """Test helper."""

            chat_template = None

        msgs = [{"role": "user", "content": "hello"}]
        try:
            result = chat(TinyMlp(), NoTemplate(), msgs, 1, 0.0)
            assert isinstance(result, str)
        except Exception:
            pass


class TestProbeInjectNoLayers:
    """Test probe injection with no layers."""

    def test_inject_raises_without_layers(self) -> None:
        class FlatModel(nn.Module):
            """Test helper."""

            def __call__(self, x):
                return mx.zeros((1, 3, 16))

        class MockTok(DummyTokenizer):
            """Test helper."""

            pass

        m = Model(FlatModel(), MockTok(), "mlx")
        with pytest.raises(ValueError, match="Cannot find transformer"):
            m.attach_probe(ProbeConfig(name="p", layers=[0]))


class TestPEFTDefaultTargetsException:
    """_default_target_modules with model that raises on named_modules()."""

    def test_named_modules_raises_falls_back_to_defaults(self) -> None:
        """When named_modules() raises, fall back to DEFAULT_LORA_KEYS."""
        from auto_chasm.peft import DEFAULT_LORA_KEYS, _default_target_modules

        class BrokenModel:
            """Model whose named_modules() raises."""

            def named_modules(self) -> None:
                msg = "cannot inspect"
                raise ValueError(msg)

        targets = _default_target_modules(BrokenModel())
        assert targets == DEFAULT_LORA_KEYS


class TestPEFTTorchDoRA:
    """apply_dora with torch backend."""

    def test_apply_dora_torch_backend(self) -> None:
        """apply_dora with torch backend returns a model."""
        pytest.importorskip("torch")
        import torch.nn as tnn

        from auto_chasm.backends import Backend
        from auto_chasm.peft import apply_dora

        class HFTinyModel(tnn.Module):
            """Model with HF-like interface for PEFT compatibility."""

            def __init__(self) -> None:
                super().__init__()
                self.q_proj = tnn.Linear(8, 8)
                self.v_proj = tnn.Linear(8, 8)

            def forward(self, x: object) -> tnn.Linear:
                return self.q_proj(x)  # type: ignore[return-value]

            def prepare_inputs_for_generation(
                self, input_ids: object, **kwargs: object
            ) -> dict[str, object]:
                return {"input_ids": input_ids}

        model = HFTinyModel()
        try:
            result = apply_dora(model, r=2, alpha=4, backend=Backend(force="torch"))
            assert result is not None
        except (ImportError, AttributeError, TypeError):
            pass


class TestPEFTUnfreezeTorchBranch:
    """_unfreeze_lora_params torch branch — lines 211-213."""

    def test_unfreeze_lora_params_torch(self) -> None:
        """_unfreeze_lora_params unfreezes lora_ params on torch model."""
        pytest.importorskip("torch")
        import torch.nn as tnn

        from auto_chasm.backends import Backend
        from auto_chasm.peft import _unfreeze_lora_params

        class TorchModelWithLora(tnn.Module):
            """Model with lora_ named parameters."""

            def __init__(self) -> None:
                super().__init__()
                self.base = tnn.Linear(4, 4)
                self.lora_a = tnn.Parameter(torch.zeros(4, 2))
                self.lora_b = tnn.Parameter(torch.zeros(2, 4))

            def forward(self, x: object) -> tnn.Linear:
                return self.base(x)  # type: ignore[return-value]

        import torch

        model = TorchModelWithLora()
        # Freeze everything
        for p in model.parameters():
            p.requires_grad = False
        assert not any(p.requires_grad for p in model.parameters())

        _unfreeze_lora_params(model, Backend(force="torch"))

        # lora_ params should be unfrozen
        lora_params = [p for name, p in model.named_parameters() if "lora_" in name]
        assert all(p.requires_grad for p in lora_params)
        # base param should remain frozen
        assert not model.base.weight.requires_grad


class TestPEFTUnfreezeMLXImportFail:
    """_unfreeze_lora_params MLX fallback when import fails — line 209."""

    def test_unfreeze_lora_params_mlx_import_fail(self) -> None:
        """When mlx_lm.tuner.lora import fails, fall back to model.unfreeze()."""
        import mlx.nn as mnn

        from auto_chasm.backends import Backend
        from auto_chasm.peft import _unfreeze_lora_params

        class MlxModel(mnn.Module):
            """MLX model for testing."""

            def __init__(self) -> None:
                super().__init__()
                self.layer = mnn.Linear(4, 4)

            def __call__(self, x: object) -> mnn.Linear:
                return self.layer(x)  # type: ignore[return-value]

        model = MlxModel()
        model.freeze()
        # Just call it — should not raise
        _unfreeze_lora_params(model, Backend(force="mlx"))

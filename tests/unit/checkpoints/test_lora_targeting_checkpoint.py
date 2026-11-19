"""Edge-case tests for LoRA layer targeting and checkpoint code."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.checkpoint import (
    export_checkpoint,
    import_checkpoint,
    load_checkpoint,
)
from auto_chasm.config import LoraConfig
from auto_chasm.model import Model


class _TinyMlp(nn.Module):  # type: ignore[misc]
    """Tiny MLP for checkpoint tests."""

    def __init__(self, hidden_dim: int = 8, vocab_size: int = 16, num_layers: int = 2) -> None:
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


class _DummyTokenizer:
    """Minimal tokenizer for testing."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 16 for c in text[:5]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids if i > 0)


def _model_with_config() -> Model:
    """Create a TinyMlp wrapped in a Model with config."""
    base = _TinyMlp()

    class Cfg:
        """Minimal model config for checkpoint tests."""

        hidden_size = 8
        num_hidden_layers = 2

    base.config = Cfg()
    return Model(base, _DummyTokenizer(), "mlx")


# ===========================================================================
# _extract_layer_index edge cases
# ===========================================================================


class TestExtractLayerIndexEdgeCases:
    """Edge cases for _extract_layer_index."""

    def test_large_index(self) -> None:
        from auto_chasm.peft import _extract_layer_index

        assert _extract_layer_index("model.layers.999.self_attn.q_proj") == 999

    def test_layer_zero_dot_suffix(self) -> None:
        from auto_chasm.peft import _extract_layer_index

        assert _extract_layer_index("model.layers.0.") == 0

    def test_negative_index_returns_none(self) -> None:
        from auto_chasm.peft import _extract_layer_index

        assert _extract_layer_index("model.layers.-1.self_attn.q_proj") is None


# ===========================================================================
# _filter_lora_targets edge cases
# ===========================================================================


class TestFilterLoraTargetsEdgeCases:
    """Edge cases for _filter_lora_targets."""

    MODULES = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.self_attn.q_proj",
    ]

    def test_until_layer_zero_excludes_all(self) -> None:
        from auto_chasm.peft import _filter_lora_targets

        result = _filter_lora_targets(self.MODULES, until_layer=0)
        assert result == []

    def test_after_layer_large_includes_nothing(self) -> None:
        from auto_chasm.peft import _filter_lora_targets

        result = _filter_lora_targets(self.MODULES, after_layer=999999)
        assert result == []

    def test_target_layers_with_until_layer(self) -> None:
        from auto_chasm.peft import _filter_lora_targets

        result = _filter_lora_targets(self.MODULES, target_layers=[0], until_layer=1)
        assert result == ["model.layers.0.self_attn.q_proj"]

    def test_non_extractable_with_empty_target_layers(self) -> None:
        from auto_chasm.peft import _filter_lora_targets

        modules = ["embed_tokens.weight", "model.layers.0.self_attn.q_proj"]
        result = _filter_lora_targets(modules, target_layers=[], until_layer=5)
        assert "embed_tokens.weight" not in result

    def test_non_extractable_always_included(self) -> None:
        from auto_chasm.peft import _filter_lora_targets

        modules = ["embed_tokens.weight"]
        result = _filter_lora_targets(modules, target_layers=[0])
        assert result == ["embed_tokens.weight"]

    def test_until_layer_zero_excludes_layer_zero(self) -> None:
        from auto_chasm.peft import _filter_lora_targets

        result = _filter_lora_targets(self.MODULES[:1], until_layer=0)
        assert result == []

    def test_after_layer_zero_includes_layer_zero(self) -> None:
        from auto_chasm.peft import _filter_lora_targets

        result = _filter_lora_targets(self.MODULES[:1], after_layer=0)
        assert result == ["model.layers.0.self_attn.q_proj"]

    def test_non_extractable_bypasses_after_layer(self) -> None:
        from auto_chasm.peft import _filter_lora_targets

        modules = ["embed_tokens.weight", "model.layers.0.self_attn.q_proj"]
        result = _filter_lora_targets(modules, after_layer=999999)
        assert "embed_tokens.weight" in result


# ===========================================================================
# get_num_layers edge cases
# ===========================================================================


class TestGetNumLayersEdgeCases:
    """Edge cases for get_num_layers."""

    def test_empty_list_returns_zero(self) -> None:
        from auto_chasm.peft import get_num_layers

        model: Any = type("EmptyLayers", (), {})()
        model.layers = []
        assert get_num_layers(model) == 0

    def test_dict_layers(self) -> None:
        from auto_chasm.peft import get_num_layers

        model: Any = type("DictLayers", (), {})()
        model.layers = {"a": 1, "b": 2, "c": 3}
        assert get_num_layers(model) == 3

    def test_list_like_layers(self) -> None:
        from auto_chasm.peft import get_num_layers

        class ListLike:
            """An object with list-like access."""

            def __init__(self) -> None:
                self._items = [1, 2, 3, 4]

            def __len__(self) -> int:
                return len(self._items)

            def __getitem__(self, idx: int) -> int:
                return self._items[idx]

        model: Any = type("ListLikeModel", (), {})()
        model.layers = ListLike()
        assert get_num_layers(model) == 4

    def test_num_layers_no_probes(self) -> None:
        model = _model_with_config()
        n = model.num_layers
        assert n == 2


# ===========================================================================
# apply_lora with target_layers
# ===========================================================================


class TestApplyLoraWithTargetLayers:
    """apply_lora delegate filtering to backend."""

    def test_apply_lora_target_layers_zero(self) -> None:
        from auto_chasm.peft import apply_lora

        mock_backend: Any = MagicMock()
        mock_backend.name = "mlx"

        apply_lora(
            MagicMock(),
            r=4,
            target_layers=[0],
            target_modules=[
                "model.layers.0.self_attn.q_proj",
                "model.layers.1.self_attn.q_proj",
            ],
            backend=mock_backend,
        )

        call_args = mock_backend.wrapping.apply_adapters.call_args
        assert call_args is not None
        filtered = call_args[0][2]
        assert filtered == ["model.layers.0.self_attn.q_proj"]


# ===========================================================================
# Checkpoint edge cases
# ===========================================================================


class TestCheckpointEdgeCases:
    """Edge cases for checkpoint save/load/export/import."""

    def test_lora_target_layers_survives_serialization(self) -> None:
        original = LoraConfig(
            rank=4,
            target_layers=[0],
            until_layer=5,
            after_layer=None,
            peft_method="qlora",
        )

        manifest_lora = {
            "rank": original.rank,
            "alpha": original.alpha,
            "dropout": original.dropout,
            "target_modules": original.target_modules,
            "target_layers": original.target_layers,
            "until_layer": original.until_layer,
            "after_layer": original.after_layer,
            "peft_method": original.peft_method,
        }

        restored = LoraConfig(
            rank=manifest_lora.get("rank", 8),  # type: ignore[arg-type]
            alpha=manifest_lora.get("alpha", 16),  # type: ignore[arg-type]
            dropout=manifest_lora.get("dropout", 0.0),  # type: ignore[arg-type]
            target_modules=manifest_lora.get("target_modules"),  # type: ignore[arg-type]
            target_layers=manifest_lora.get("target_layers"),  # type: ignore[arg-type]
            until_layer=manifest_lora.get("until_layer"),  # type: ignore[arg-type]
            after_layer=manifest_lora.get("after_layer"),  # type: ignore[arg-type]
            peft_method=manifest_lora.get("peft_method", "lora"),  # type: ignore[arg-type]
        )

        assert restored.target_layers == original.target_layers
        assert restored.until_layer == original.until_layer
        assert restored.peft_method == original.peft_method

    def test_load_checkpoint_corrupted_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "manifest.json").write_text("not valid json{{{")
            with pytest.raises(json.JSONDecodeError):
                load_checkpoint(tmp)

    def test_import_checkpoint_nonexistent(self) -> None:
        with pytest.raises(FileNotFoundError):
            import_checkpoint("/nonexistent/archive.auto_chasm", "/tmp")

    def test_export_checkpoint_missing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp, "ckpt")
            ckpt.mkdir()
            Path(ckpt, "manifest.json").write_text('{"test": true}')
            output = Path(tmp, "nonexistent_parent", "out.auto_chasm")
            export_checkpoint(str(ckpt), str(output))
            assert output.exists()

    def test_save_checkpoint_overwrite_existing_dir(self) -> None:
        model = _model_with_config()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp, "ckpt")
            model.save_checkpoint(str(ckpt))
            model.save_checkpoint(str(ckpt))
            assert Path(ckpt, "manifest.json").exists()


# ===========================================================================
# Config edge cases
# ===========================================================================


class TestLoraConfigEdgeCases:
    """Edge cases for LoraConfig."""

    def test_lora_config_target_layers_empty(self) -> None:
        cfg = LoraConfig(target_layers=[], until_layer=None, after_layer=None)
        assert cfg.target_layers == []
        assert cfg.until_layer is None
        assert cfg.after_layer is None

    def test_lora_config_dropout_extreme(self) -> None:
        cfg = LoraConfig(dropout=1.0)
        assert cfg.dropout == 1.0

    def test_lora_config_qlora_target_layers(self) -> None:
        cfg = LoraConfig(peft_method="qlora", target_layers=[0])
        assert cfg.peft_method == "qlora"
        assert cfg.target_layers == [0]

    def test_lora_config_dora_until_layer(self) -> None:
        cfg = LoraConfig(peft_method="dora", until_layer=5)
        assert cfg.peft_method == "dora"
        assert cfg.until_layer == 5

"""Tests for LoRA layer targeting: _extract_layer_index, _filter_lora_targets, get_num_layers."""

from __future__ import annotations

from typing import Any

import mlx.nn as nn
import pytest


class TestExtractLayerIndex:
    """Tests for _extract_layer_index."""

    def test_standard_layer(self) -> None:
        """Standard 'model.layers.3.self_attn.q_proj' extracts index 3."""
        from auto_chasm.peft import _extract_layer_index

        assert _extract_layer_index("model.layers.3.self_attn.q_proj") == 3

    def test_no_layer_returns_none(self) -> None:
        """Module name without a layer index returns None."""
        from auto_chasm.peft import _extract_layer_index

        assert _extract_layer_index("embed_tokens.weight") is None

    def test_huggingface_format(self) -> None:
        """HuggingFace 'transformer.h.5.attention.q_proj' extracts index 5."""
        from auto_chasm.peft import _extract_layer_index

        assert _extract_layer_index("transformer.h.5.attention.q_proj") == 5

    def test_blocks_format(self) -> None:
        """Blocks format 'model.blocks.2.fc1' extracts index 2."""
        from auto_chasm.peft import _extract_layer_index

        assert _extract_layer_index("model.blocks.2.fc1") == 2

    def test_layer_zero(self) -> None:
        """'model.layers.0.' extracts index 0."""
        from auto_chasm.peft import _extract_layer_index

        assert _extract_layer_index("model.layers.0.self_attn.q_proj") == 0

    def test_no_match_returns_none(self) -> None:
        """Completely unmatched pattern returns None."""
        from auto_chasm.peft import _extract_layer_index

        assert _extract_layer_index("encoder.block.5.layer_norm") is None


class TestFilterLoraTargets:
    """Tests for _filter_lora_targets."""

    MODULES = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.2.self_attn.q_proj",
        "model.layers.3.self_attn.q_proj",
        "model.layers.4.self_attn.q_proj",
    ]

    def test_no_filters_returns_all(self) -> None:
        """No filters should return all modules unchanged."""
        from auto_chasm.peft import _filter_lora_targets

        result = _filter_lora_targets(self.MODULES)
        assert result == self.MODULES

    def test_all_none_returns_all(self) -> None:
        """All filter params None returns all modules."""
        from auto_chasm.peft import _filter_lora_targets

        result = _filter_lora_targets(
            self.MODULES, target_layers=None, until_layer=None, after_layer=None
        )
        assert result == self.MODULES

    def test_target_layers_filters(self) -> None:
        """target_layers=[0, 2] returns only layers 0 and 2."""
        from auto_chasm.peft import _filter_lora_targets

        result = _filter_lora_targets(self.MODULES, target_layers=[0, 2])
        assert len(result) == 2
        assert all("layers.0." in m or "layers.2." in m for m in result)

    def test_until_layer_excludes(self) -> None:
        """until_layer=3 returns layers 0, 1, 2 (indices < 3)."""
        from auto_chasm.peft import _filter_lora_targets

        result = _filter_lora_targets(self.MODULES, until_layer=3)
        assert len(result) == 3
        assert all("layers.0." in m or "layers.1." in m or "layers.2." in m for m in result)

    def test_after_layer_includes(self) -> None:
        """after_layer=3 returns layers 3, 4 (indices >= 3)."""
        from auto_chasm.peft import _filter_lora_targets

        result = _filter_lora_targets(self.MODULES, after_layer=3)
        assert len(result) == 2
        assert all("layers.3." in m or "layers.4." in m for m in result)

    def test_combined_filters(self) -> None:
        """Combined target_layers + until_layer."""
        from auto_chasm.peft import _filter_lora_targets

        result = _filter_lora_targets(self.MODULES, target_layers=[0, 2, 3], until_layer=3)
        assert len(result) == 2
        assert all("layers.0." in m or "layers.2." in m for m in result)

    def test_non_extractable_included(self) -> None:
        """Modules without extractable layer index are always included."""
        from auto_chasm.peft import _filter_lora_targets

        modules = ["embed_tokens.weight", "model.layers.0.self_attn.q_proj"]
        result = _filter_lora_targets(modules, target_layers=[0])
        assert "embed_tokens.weight" in result
        assert "model.layers.0.self_attn.q_proj" in result

    def test_empty_target_layers(self) -> None:
        """Empty target_layers list returns no layer modules."""
        from auto_chasm.peft import _filter_lora_targets

        result = _filter_lora_targets(self.MODULES, target_layers=[])
        assert len(result) == 0


class TestGetNumLayers:
    """Tests for get_num_layers and Model.num_layers."""

    def test_get_num_layers_mlx(self) -> None:
        """get_num_layers returns correct count for MLX model."""
        from auto_chasm.peft import get_num_layers

        class TinyModel(nn.Module):
            """Model with layers attribute."""

            def __init__(self) -> None:
                super().__init__()
                self.layers = [nn.Linear(4, 4) for _ in range(6)]

            def __call__(self, x: object) -> nn.Linear:
                return self.layers[0](x)  # type: ignore[return-value]

        n = get_num_layers(TinyModel())
        assert n == 6

    def test_model_num_layers_property(self, model_wrapper: Any) -> None:
        """Model.num_layers returns layer count."""
        n = model_wrapper.num_layers
        assert n == 4  # TinyMlp has 4 layers

    def test_no_layers_raises(self) -> None:
        """get_num_layers raises ValueError when no layers found."""
        from auto_chasm.peft import get_num_layers

        with pytest.raises(ValueError, match="Cannot determine"):
            get_num_layers(object())

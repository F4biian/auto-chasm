"""Regression tests for PEFT targeting + dead config fields.

Covers four confirmed bugs:

1. ``_extract_layer_index`` ignored top-level ``layers`` models (regex needed a
   leading dot), so LoRA layer targeting was silently a no-op there.
2. The same ``target_modules=["q_proj"]`` wrapped 0 LoRA layers on MLX while
   wrapping several on torch; a 0-match was silent. Matching is now normalized
   against the model's real modules and a 0-match raises.
3. ``TrainingConfig.mixed_precision`` was a dead field (fp16/bf16 silently ran
   in fp32). It is now wired: bf16 casts the frozen base (both backends), fp16
   uses torch autocast + a GradScaler; an invalid mode raises ``ValueError``.
4. ``TrainingConfig.num_epochs`` / ``save_best_only`` were dead fields. They now
   raise ``NotImplementedError`` on a non-default value instead of being ignored.
"""

from __future__ import annotations

from typing import Any

import pytest

from auto_chasm.peft import (
    _extract_layer_index,
    _filter_lora_targets,
    _resolve_target_modules,
    apply_lora,
)


class _FakeModule:
    """Minimal stand-in exposing only ``named_modules`` for resolution tests."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def named_modules(self) -> list[tuple[str, Any]]:
        """Yield ``(name, module)`` pairs (module is irrelevant here)."""
        return [(n, object()) for n in self._names]


class TestExtractLayerIndexBareLayers:
    """Bug 1: layer index extraction must work without a leading dot."""

    def test_bare_layers_returns_zero(self) -> None:
        """'layers.0.self_attn.q_proj' (no leading dot) extracts index 0."""
        assert _extract_layer_index("layers.0.self_attn.q_proj") == 0

    def test_bare_blocks(self) -> None:
        """'blocks.7.fc1' extracts index 7."""
        assert _extract_layer_index("blocks.7.fc1") == 7

    def test_bare_h(self) -> None:
        """'h.9.attn.q_proj' extracts index 9."""
        assert _extract_layer_index("h.9.attn.q_proj") == 9

    def test_prefixed_still_works(self) -> None:
        """Leading prefixes still resolve (regression guard)."""
        assert _extract_layer_index("model.layers.3.self_attn.q_proj") == 3
        assert _extract_layer_index("transformer.h.5.attn.q_proj") == 5

    def test_non_layer_returns_none(self) -> None:
        """Names without a layer container return None."""
        assert _extract_layer_index("embed_tokens.weight") is None


class TestLayerTargetingFiltersBareLayers:
    """Bug 1: layer targeting must actually filter top-level-``layers`` names."""

    MODULES = [
        "layers.0.self_attn.q_proj",
        "layers.1.self_attn.q_proj",
        "layers.2.self_attn.q_proj",
        "layers.3.self_attn.q_proj",
    ]

    def test_target_layers_filters(self) -> None:
        """target_layers=[0] keeps only layer 0 (was: kept all four)."""
        result = _filter_lora_targets(self.MODULES, target_layers=[0])
        assert result == ["layers.0.self_attn.q_proj"]

    def test_until_layer_filters(self) -> None:
        """until_layer=2 keeps layers 0 and 1 only."""
        result = _filter_lora_targets(self.MODULES, until_layer=2)
        assert result == [
            "layers.0.self_attn.q_proj",
            "layers.1.self_attn.q_proj",
        ]

    def test_after_layer_filters(self) -> None:
        """after_layer=2 keeps layers 2 and 3 only."""
        result = _filter_lora_targets(self.MODULES, after_layer=2)
        assert result == [
            "layers.2.self_attn.q_proj",
            "layers.3.self_attn.q_proj",
        ]


class TestResolveTargetModules:
    """Bug 2: cross-backend match normalization + no silent 0-match."""

    def test_bare_name_expands_to_full_paths(self) -> None:
        """'q_proj' resolves to every matching full module path."""
        model = _FakeModule(
            [
                "layers.0.self_attn.q_proj",
                "layers.0.self_attn.v_proj",
                "layers.1.self_attn.q_proj",
                "layers.1.self_attn.v_proj",
            ]
        )
        result = _resolve_target_modules(model, ["q_proj"])
        assert result == [
            "layers.0.self_attn.q_proj",
            "layers.1.self_attn.q_proj",
        ]

    def test_consistent_across_naming_conventions(self) -> None:
        """Bare name resolves identically whether or not paths carry a prefix."""
        bare = _FakeModule(["layers.0.self_attn.q_proj", "layers.1.self_attn.q_proj"])
        prefixed = _FakeModule(
            ["model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj"]
        )
        assert len(_resolve_target_modules(bare, ["q_proj"])) == 2
        assert len(_resolve_target_modules(prefixed, ["q_proj"])) == 2

    def test_no_match_raises(self) -> None:
        """A target matching nothing raises rather than silently doing nothing."""
        model = _FakeModule(["layers.0.self_attn.q_proj"])
        with pytest.raises(ValueError, match="matched no modules"):
            _resolve_target_modules(model, ["does_not_exist"])

    def test_partial_match_warns_but_keeps_rest(self, caplog: Any) -> None:
        """A partially-unmatched request warns and keeps the matched names."""
        model = _FakeModule(["layers.0.self_attn.q_proj", "layers.1.self_attn.q_proj"])
        import logging

        with caplog.at_level(logging.WARNING):
            result = _resolve_target_modules(model, ["q_proj", "missing"])
        assert len(result) == 2
        assert any("missing" in rec.message for rec in caplog.records)

    def test_no_inspectable_modules_passes_through(self) -> None:
        """A model without named_modules passes the request through unchanged."""

        class Bare:
            """A model with no named_modules()."""

        assert _resolve_target_modules(Bare(), ["q_proj"]) == ["q_proj"]


class TestApplyLoraNonMatchRaises:
    """Bug 2 (end-to-end): apply_lora must not silently wrap nothing."""

    def test_apply_lora_no_match_raises(self) -> None:
        """apply_lora with an unmatched target raises before touching a backend."""
        model = _FakeModule(["layers.0.self_attn.q_proj"])
        with pytest.raises(ValueError, match="matched no modules"):
            apply_lora(model, target_modules=["nonexistent_proj"], backend=object())

    def test_apply_lora_empty_after_layer_filter_raises(self) -> None:
        """apply_lora raises when layer targeting filters everything out."""
        model = _FakeModule(["layers.0.self_attn.q_proj", "layers.1.self_attn.q_proj"])
        with pytest.raises(ValueError, match="filtered out"):
            apply_lora(
                model,
                target_modules=["q_proj"],
                target_layers=[99],
                backend=object(),
            )


@pytest.mark.real_model
class TestApplyLoraMlxWrapsLayers:
    """Bug 2: bare 'q_proj' wraps real LoRA layers on the cached MLX model."""

    def test_mlx_bare_qproj_wraps_layers(self) -> None:
        """target_modules=['q_proj'] wraps every q_proj on MLX (was: 0)."""
        mlx_lm = pytest.importorskip("mlx_lm")
        from mlx_lm.tuner.lora import LoRALinear

        from auto_chasm.backends import Backend

        try:
            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            model, _ = mlx_lm.load("mlx-community/gemma-3-270m-it-8bit")
        except Exception:
            pytest.skip("cached MLX model unavailable")

        n_qproj = sum(1 for n, _ in model.named_modules() if n.endswith("q_proj"))
        apply_lora(model, r=4, alpha=8, target_modules=["q_proj"], backend=Backend(force="mlx"))
        n_lora = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALinear))
        assert n_lora == n_qproj
        assert n_lora > 0

    def test_mlx_layer_targeting_filters_wrapping(self) -> None:
        """target_layers=[0, 1] wraps exactly those two layers on MLX."""
        mlx_lm = pytest.importorskip("mlx_lm")
        from mlx_lm.tuner.lora import LoRALinear

        from auto_chasm.backends import Backend

        try:
            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            model, _ = mlx_lm.load("mlx-community/gemma-3-270m-it-8bit")
        except Exception:
            pytest.skip("cached MLX model unavailable")

        apply_lora(
            model,
            r=4,
            alpha=8,
            target_modules=["q_proj"],
            target_layers=[0, 1],
            backend=Backend(force="mlx"),
        )
        n_lora = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALinear))
        assert n_lora == 2


class TestMixedPrecisionHonest:
    """Bug 3: mixed_precision must not silently degrade to fp32."""

    def test_fp32_default_ok(self) -> None:
        """The default fp32 constructs cleanly."""
        from auto_chasm.config import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.mixed_precision == "fp32"

    def test_fp16_is_valid_config_torch_only(self) -> None:
        """fp16 is a valid config now (torch autocast + GradScaler); junk still raises."""
        from auto_chasm.config import TrainingConfig

        assert TrainingConfig(mixed_precision="fp16").mixed_precision == "fp16"
        with pytest.raises(ValueError, match="not valid"):
            TrainingConfig(mixed_precision="fp8")  # type: ignore[arg-type]

    def test_bf16_is_supported(self) -> None:
        """bf16 is now a real, supported mode (frozen-base mixed precision)."""
        from auto_chasm.config import TrainingConfig

        assert TrainingConfig(mixed_precision="bf16").mixed_precision == "bf16"


class TestNumEpochsHonest:
    """Bug 4: num_epochs must not be a silently-ignored dead field."""

    def test_default_ok(self) -> None:
        """The default num_epochs=3 constructs cleanly."""
        from auto_chasm.config import TrainingConfig

        assert TrainingConfig().num_epochs == 3

    def test_non_default_raises(self) -> None:
        """A non-default num_epochs raises NotImplementedError."""
        from auto_chasm.config import TrainingConfig

        with pytest.raises(NotImplementedError, match="num_epochs"):
            TrainingConfig(num_epochs=5)


class TestSaveBestOnlyHonest:
    """Bug 4: save_best_only must not be a silently-ignored dead field."""

    def test_default_ok(self) -> None:
        """The default save_best_only=True constructs cleanly."""
        from auto_chasm.config import TrainingConfig

        assert TrainingConfig().save_best_only is True

    def test_non_default_raises(self) -> None:
        """save_best_only=False raises NotImplementedError."""
        from auto_chasm.config import TrainingConfig

        with pytest.raises(NotImplementedError, match="save_best_only"):
            TrainingConfig(save_best_only=False)

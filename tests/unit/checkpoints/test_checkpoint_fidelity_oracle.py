"""Oracle tests for lossless checkpoint round-trips.

Guards two bugs:
* loading dropped LoRA layer-targeting + peft_method, so a DoRA-on-some-
  layers checkpoint reloaded as plain LoRA on all layers;
* steering serialization dropped the head weight + Fisher stats and always
  restored MLX tensors regardless of the model's backend.
"""

from __future__ import annotations

from auto_chasm.checkpoint import _lora_from_manifest
from auto_chasm.config import LoraConfig, SteeringConfig
from auto_chasm.steering import SteeringHook
from auto_chasm.utils import tensor_backend


def _manifest_lora(cfg: LoraConfig) -> dict:
    """Replicate the manifest['lora'] block that save_checkpoint writes."""
    return {
        "rank": cfg.rank,
        "alpha": cfg.alpha,
        "dropout": cfg.dropout,
        "target_modules": cfg.target_modules,
        "target_layers": cfg.target_layers,
        "until_layer": cfg.until_layer,
        "after_layer": cfg.after_layer,
        "peft_method": cfg.peft_method,
    }


def test_lora_config_round_trips_all_fields():
    original = LoraConfig(
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        peft_method="dora",
        target_layers=[0, 2, 4],
        until_layer=8,
        after_layer=2,
    )
    restored = _lora_from_manifest(_manifest_lora(original))
    assert restored.rank == 16
    assert restored.alpha == 32
    assert restored.dropout == 0.05
    assert restored.target_modules == ["q_proj", "v_proj"]
    assert restored.peft_method == "dora"  # was silently lost before
    assert restored.target_layers == [0, 2, 4]  # was silently lost before
    assert restored.until_layer == 8
    assert restored.after_layer == 2


def _make_hook_with_geometry():
    import mlx.core as mx

    hook = SteeringHook("p", SteeringConfig(method="nullify", scale=1.5))
    hook._mean_0 = mx.array([0.0, 0.0, 0.0])
    hook._mean_1 = mx.array([1.0, 2.0, 3.0])
    hook._direction = hook._mean_1 - hook._mean_0
    hook._head_weight = mx.array([0.5, -0.5, 0.25])
    hook._head_bias = mx.array([0.1])
    hook._head_norm = 0.75
    hook._fisher_along = 0.42
    return hook


def test_steering_serialization_preserves_head_weight_and_fisher():
    hook = _make_hook_with_geometry()
    restored = SteeringHook.from_dict(hook.to_dict(), backend="mlx")
    assert restored._head_weight is not None  # was dropped before
    assert restored._fisher_along == 0.42  # was dropped before
    assert restored._head_norm == 0.75
    assert restored.config.scale == 1.5


def test_steering_restore_respects_backend():
    data = _make_hook_with_geometry().to_dict()
    mlx_hook = SteeringHook.from_dict(data, backend="mlx")
    torch_hook = SteeringHook.from_dict(data, backend="torch")
    assert tensor_backend(mlx_hook._mean_0) == "mlx"
    assert tensor_backend(torch_hook._mean_0) == "torch"

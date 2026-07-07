"""Integration oracle: a checkpoint round-trips LoRA adapters at a *probed* layer.

Regression for a demo-found bug: a ``hidden`` probe wraps its transformer layer,
so LoRA adapters trained at that layer are saved under the wrapper's
``model.layers.N.layer.self_attn.*`` keys. ``load_checkpoint`` used to load
adapters *before* attaching probes, so those wrapped keys silently failed to match
(``load_weights(strict=False)``) and that layer's adapters reloaded at zero-init —
corrupting the model. The fix attaches probes (re-wrapping the layer) before
loading adapters, mirroring the save-time order. This test uses the real (cached)
SmolLM2-135M because ``from_checkpoint`` reloads the base model by name.
"""

from __future__ import annotations

import tempfile

import mlx.core as mx
import pytest
from mlx.utils import tree_map

from auto_chasm import LoraConfig, Model, ProbeConfig

MODEL = "HuggingFaceTB/SmolLM2-135M"

# Whole module reloads a real base model by name — gated behind --run-real-model.
pytestmark = pytest.mark.real_model


def test_lora_adapters_at_probed_layer_survive_checkpoint_roundtrip() -> None:
    model = Model.from_pretrained(MODEL, lora=LoraConfig(rank=4, alpha=8))
    model.attach_probe(ProbeConfig(name="p", layers=[2], source="hidden"))

    # Make the adapters non-trivial (lora_b is zero-init), so a dropped adapter at
    # the probed layer would visibly change the output.
    model.model.update(tree_map(lambda v: v + 0.3, model.model.trainable_parameters()))
    mx.eval(model.model.parameters())

    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    before = model.forward(ids).lm_logits

    with tempfile.TemporaryDirectory() as tmp:
        model.save_checkpoint(tmp)
        reloaded = Model.from_checkpoint(tmp)
    after = reloaded.forward(ids).lm_logits

    max_diff = float(mx.max(mx.abs(before - after)).item())
    assert max_diff < 1e-3, (
        f"reloaded LM logits differ (max|diff|={max_diff:.4f}) — the probed layer's "
        "LoRA adapters were dropped on reload (wrapped-key mismatch)."
    )

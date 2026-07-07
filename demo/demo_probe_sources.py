"""Read probes from different activation sources and multiple layers.

A probe can read the block output (``hidden``), the MLP or attention sub-module,
the residual stream, the embeddings, or the output logits — and can span several
layers at once. This attaches one of each and runs a single forward pass to show
they all produce head outputs. Backend-agnostic.

    python demo/demo_probe_sources.py
"""

from __future__ import annotations

from auto_chasm import Model, ProbeConfig

MODEL = "HuggingFaceTB/SmolLM2-135M"

model = Model.from_pretrained(MODEL)
last = model.num_layers - 1

model.add_probes([
    ProbeConfig(name="hidden", layers=[last], source="hidden"),        # the block output
    ProbeConfig(name="mlp", layers=[last], source="mlp"),              # the MLP sub-module output
    ProbeConfig(name="attention", layers=[last], source="attention"),  # the attention sub-module output
    ProbeConfig(name="multi", layers=[0, last // 2, last], aggregation="concat"),  # three layers, concatenated
])

out = model.forward([model.tokenizer.encode("The quick brown fox jumps.")])
print("probe                 output shape")
for name in ("hidden", "mlp", "attention", "multi"):
    print(f"{name:<20}  {tuple(out.probes[name].logits.shape)}")

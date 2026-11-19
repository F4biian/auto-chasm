"""Inspect a model's architecture and parameter counts.

``model.stats()`` returns a dict of the layer count, hidden size, vocabulary size,
attention heads, MLP width, total and trainable parameter counts, and per-probe
parameter counts. Individual pieces are also available as accessors. Backend-agnostic.

    python demo/demo_model_stats.py
"""

from __future__ import annotations

import json

from auto_chasm import Model, ProbeConfig

MODEL = "HuggingFaceTB/SmolLM2-135M"

model = Model.from_pretrained(MODEL)
model.attach_probe(ProbeConfig(name="probe", layers=[-1]))

print(json.dumps(model.stats(), indent=2))
print(
    f"\nhidden_size={model.hidden_size}  vocab_size={model.vocab_size}  "
    f"layers={model.num_layers}\ntotal params={model.num_parameters():,}  "
    f"trainable={model.num_parameters(trainable=True):,}"
)

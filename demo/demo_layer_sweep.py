"""Find which layer best encodes a property, by probing several layers at once.

LayerSweep attaches one head per layer, trains them together in a single frozen-base
pass, snapshots each head at its own best-validation step, and reports each layer's
score. Backend-agnostic.

    python demo/demo_layer_sweep.py
"""

from __future__ import annotations

from auto_chasm import Dataset, JointLoss, LayerSweep, Model, Task

MODEL = "HuggingFaceTB/SmolLM2-135M"

model = Model.from_pretrained(MODEL)

texts = [
    "Is it raining?", "The sky is grey.", "What time is it?", "The clock is slow.",
    "Where are we?", "The map is old.", "How far is it?", "The trip is long.",
    "Who is there?", "The room is dark.", "Why is that?", "The light is on.",
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]   # 1 = question, 0 = statement
data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="probe",
                          label_site="response", append_eos=True)

# Deterministic train / val / test split (LayerSweep selects each layer's head by val).
rest, test = data.split(0.25, seed=0)
train, val = rest.split(0.25, seed=0)

# task=Task.binary() sizes every layer's head and derives the accuracy metrics.
sweep = LayerSweep(model, task=Task.binary(), layers=[0, model.num_layers // 2, model.num_layers - 1])
result = sweep.run(
    train, val, test,
    loss_fn=JointLoss(weights={"lm_head": 0.0}),   # pure-probe; every layer head uses the default BCE
    num_iters=40,
    eval_every=10,
)

print(f"\nbest layer: {result.best_layer()}")
for layer, scores in result.best.items():
    print(f"  layer {layer:>2}: val_acc={scores['val_acc']:.2f}  test_acc={scores['test_acc']:.2f}")

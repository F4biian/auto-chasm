"""Train a probe that REGRESSES a continuous value instead of classifying.

The label is a float (here the word count of each text), the head is a width-1
scalar, and the loss is "mse". `from_texts` keeps float labels as floats.
Backend-agnostic.

    python demo/demo_regression.py
"""

from __future__ import annotations

from auto_chasm import Dataset, JointLoss, Model, ProbeConfig, Trainer

MODEL = "HuggingFaceTB/SmolLM2-135M"

model = Model.from_pretrained(MODEL)

texts = [
    "Hello.", "The cat sat.", "A", "Birds fly south in winter.",
    "Yes.", "This is a slightly longer sentence here.", "Go now.", "Two words.",
]
# The regression target is the word count of each text — a continuous (float) value.
word_count = [float(len(t.split())) for t in texts]
data = Dataset.from_texts(texts, word_count, model.tokenizer, probe_name="length",
                          label_site="response", append_eos=True)

model.attach_probe(ProbeConfig(name="length", layers=[-1]))   # width-1 head = a scalar regressor
model.prepare_for_joint_training()

Trainer(
    model=model,
    loss_fn=JointLoss(weights={"lm_head": 0.0, "length": 1.0}, losses={"length": "mse"}),
    num_iters=60,
    batch_size=4,
).train(data)

# The head now predicts a continuous value (no sigmoid — it is a regressor).
print("\ntext                              predicted / true words")
for text in ["Hi.", "One two three four five words."]:
    pred = float(model.forward([model.tokenizer.encode(text)]).probes["length"].logits[0, -1].item())
    print(f"{text!r:<34}  {pred:5.1f} / {len(text.split())}")

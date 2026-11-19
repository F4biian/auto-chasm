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
    "Yes.", "Go now.", "The cat sat.", "Birds fly south.", "I like green tea.",
    "We walked to the park.", "She reads a book every night.", "Hello.",
    "Two words.", "The sun is very bright today.", "My brother plays the guitar well.",
    "Rain again.", "They built a small wooden house.", "Dogs bark.",
    "He drinks coffee every single morning before work.", "It works.",
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
    num_iters=150,
    batch_size=4,
    learning_rate=1e-3,
).train(data)

# The head now predicts a continuous value (no sigmoid — it is a regressor).
print("\ntext                              predicted / true words")
for text in ["Good night.", "The dog sleeps.", "We ate dinner at a quiet place."]:
    out = model.forward([model.tokenizer.encode(text)])
    pred = float(out.probes["length"].logits[0, -1].item())
    print(f"{text!r:<34}  {pred:5.1f} / {len(text.split())}")

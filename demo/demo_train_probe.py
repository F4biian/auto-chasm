"""Train a probe that detects questions, then read it live while generating.

Copy-paste runnable: downloads a small model (~270 MB) on first run and needs no
other setup. The same code runs on the MLX and PyTorch backends.

    python demo/demo_train_probe.py
"""

from __future__ import annotations

import math

from auto_chasm import Dataset, JointLoss, Model, ProbeConfig, Trainer

MODEL = "HuggingFaceTB/SmolLM2-135M"  # small standard-Llama arch; loads on both backends

model = Model.from_pretrained(MODEL)

# One class label per text: 1 = question, 0 = statement (the label semantics).
# label_site="response" labels the whole text; append_eos supervises the hidden
# state AFTER the model has read every token, not the second-to-last one.
texts = [
    "The sky is blue.", "Is it raining?", "Cats purr.", "Are you there?",
    "It is warm today.", "What time is it?", "Birds can fly.", "Do you agree?",
]
labels = [0, 1, 0, 1, 0, 1, 0, 1]
data = Dataset.from_texts(
    texts, labels, model.tokenizer, probe_name="is_question",
    label_site="response", append_eos=True,
)

# A linear head on the last layer is a binary classifier.
model.attach_probe(ProbeConfig(name="is_question", layers=[-1]))
model.prepare_for_joint_training()

# Train the probe only: the language-model term is switched off.
Trainer(
    model=model,
    loss_fn=JointLoss(weights={"lm_head": 0.0, "is_question": 1.0}),
    num_iters=30,
    batch_size=4,
).train(data)

# Read the probe's prediction for each generated token.
print("\ntoken            p(question)")
for step in model.generate_with_probes("Tell me something:", max_tokens=8, temperature=0.0):
    logit = float(step.probes["is_question"].logits[0, -1].item())
    print(f"{step.token_str!r:<16} {1 / (1 + math.exp(-logit)):.2f}")

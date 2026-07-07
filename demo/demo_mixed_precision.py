"""Train a probe in bf16 mixed precision and watch it actually learn.

The frozen base runs in bfloat16 while the trainable probe (and the optimizer)
stay in float32. bf16 shares fp32's exponent range, so no loss scaling is needed;
it halves the base's memory with no accuracy loss here. This runs the *same* code
on MLX and PyTorch. (For fp16 on torch, pass mixed_precision="fp16" — it adds a
GradScaler automatically; fp16 is not supported on MLX, where bf16 is preferred.)

Copy-paste runnable: downloads a small model (~270 MB) on first run.

    python demo/demo_mixed_precision.py
"""

from __future__ import annotations

import math

from auto_chasm import Dataset, JointLoss, Model, ProbeConfig, Trainer, TrainingConfig

MODEL = "HuggingFaceTB/SmolLM2-135M"  # small standard-Llama arch; loads on both backends

model = Model.from_pretrained(MODEL)

# A tiny question-vs-statement task: 1 = question, 0 = statement.
train_texts = [
    "The sky is blue.",
    "Is it raining today?",
    "Cats like to sleep.",
    "Are you feeling well?",
    "It is warm outside.",
    "What time does it start?",
    "Birds build nests.",
    "Do you like coffee?",
    "The train was late.",
    "Where did you go?",
    "Water boils at 100 degrees.",
    "Can I help you?",
    "My favorite color is green.",
    "How does this work?",
    "The meeting is at noon.",
    "Would you agree?",
    "The garden is quiet.",
    "Why are the leaves green?",
    "She reads every night.",
    "Which one is faster?",
]
train_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

# Held-out examples the probe never sees during training.
eval_texts = [
    "The dog is barking.",
    "Is the store open?",
    "Leaves fall in autumn.",
    "Have you eaten yet?",
    "The book was long.",
    "Why is the sky dark?",
]
eval_labels = [0, 1, 0, 1, 0, 1]

data = Dataset.from_texts(
    train_texts,
    train_labels,
    model.tokenizer,
    probe_name="is_question",
    label_site="response",
    append_eos=True,
)

# granularity="response" pools the response into ONE prediction per text — a clean
# signal for whole-sentence classification.
model.attach_probe(
    ProbeConfig(name="is_question", layers=[-1], granularity="response", aggregation="mean")
)
model.prepare_for_joint_training()


def p_question(text: str) -> float:
    """Probability the probe assigns to 'this text is a question'."""
    ids = model.tokenizer.encode(text)
    if ids[-1] != model.tokenizer.eos_token_id:
        ids = ids + [model.tokenizer.eos_token_id]
    logit = float(model.forward([ids]).probes["is_question"].logits.reshape(-1)[-1].item())
    return 1.0 / (1.0 + math.exp(-logit))


def accuracy() -> float:
    """Held-out accuracy of the probe (threshold at p > 0.5)."""
    pairs = zip(eval_texts, eval_labels, strict=True)
    return sum((p_question(t) > 0.5) == bool(y) for t, y in pairs) / len(eval_labels)


print(f"held-out accuracy BEFORE training: {accuracy():.0%}")

# The base is cast to bf16 for its forward; the probe + optimizer stay fp32.
Trainer(
    model=model,
    loss_fn=JointLoss(weights={"lm_head": 0.0, "is_question": 1.0}),
    config=TrainingConfig(mixed_precision="bf16", batch_size=4),
    num_iters=40,
    learning_rate=5e-4,
    verbose=False,
).train(data)

print(f"held-out accuracy AFTER bf16 training:  {accuracy():.0%}")
print("\nprobe predictions on unseen text:")
for text, label in zip(eval_texts, eval_labels, strict=True):
    print(
        f"  {'question ' if label else 'statement'}  p(question)={p_question(text):.2f}  {text!r}"
    )

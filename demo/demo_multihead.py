"""Train several probe heads at once, each with its own labels and loss.

One binary head (is it a question?) and one 3-class head (topic) share the model
and train jointly, each on independent per-head labels. Backend-agnostic; needs
only a small model download.

    python demo/demo_multihead.py
"""

from __future__ import annotations

from auto_chasm import Dataset, JointLoss, Model, ProbeConfig, Trainer

MODEL = "HuggingFaceTB/SmolLM2-135M"

model = Model.from_pretrained(MODEL)

texts = [
    "Is it raining outside?", "The cat sleeps.", "What is the price?", "Dogs bark loudly.",
    "Where did you go?", "The sun is bright.", "How does this work?", "Rivers flow downhill.",
]
is_question = [1, 0, 1, 0, 1, 0, 1, 0]          # binary label per text
topic = [2, 0, 2, 0, 1, 0, 1, 0]                # 3-class label per text (0/1/2)

# Build each head's per-token labels, then pair them on the SAME tokens so every
# sample carries a {head: labels} dict — each head learns only its own target.
dq = Dataset.from_texts(texts, is_question, model.tokenizer, probe_name="is_question",
                        label_site="response", append_eos=True)
dt = Dataset.from_texts(texts, topic, model.tokenizer, probe_name="topic",
                        label_site="response", append_eos=True)
data = [
    {"tokens": a["tokens"], "labels": {"is_question": a["labels"], "topic": b["labels"]}}
    for a, b in zip(dq, dt, strict=True)
]

model.add_probes([
    ProbeConfig(name="is_question", layers=[-1]),                              # width 1 -> BCE
    ProbeConfig(name="topic", layers=[-1], module_config={"out_features": 3}),  # width 3 -> CE
])
model.prepare_for_joint_training()

Trainer(
    model=model,
    loss_fn=JointLoss(
        weights={"lm_head": 0.0, "is_question": 1.0, "topic": 1.0},
        losses={"topic": "ce"},   # is_question uses the default "bce"
    ),
    num_iters=40,
    batch_size=4,
).train(data)

# Read both heads at the last token of a fresh text.
out = model.forward([model.tokenizer.encode("How much does it cost?")])
q_logit = float(out.probes["is_question"].logits[0, -1].item())
topic_logits = out.probes["topic"].logits[0, -1].tolist()
print(f"\nis_question logit = {q_logit:+.2f}  (>0 => question)")
print(f"topic class       = {max(range(3), key=lambda i: topic_logits[i])}  from logits {topic_logits}")

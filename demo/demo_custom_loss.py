"""Write your OWN per-probe loss from the raw probe outputs, combined via JointLoss.

A probe's loss can be any callable ``(probe, target)`` — not just the built-in
bce/ce/mse/mae. Here a soft-margin loss computed from the raw head logits trains one
head, while a second head uses the built-in BCE; JointLoss sums them by weight. The
``probe`` argument exposes ``.logits``, ``.softmax()``, ``.log_softmax()`` and
``.reduce()`` (which averages over the valid, non-padding positions), so you can build
KL, margin, focal, ... losses that stay backend-agnostic. Backend-agnostic.

    python demo/demo_custom_loss.py
"""

from __future__ import annotations

from auto_chasm import Dataset, JointLoss, Model, ProbeConfig, Trainer, ops

MODEL = "HuggingFaceTB/SmolLM2-135M"

model = Model.from_pretrained(MODEL)

texts = [
    "Is it far?", "The road is long.", "Are we there?", "The car is red.",
    "Do you know?", "Rivers run downhill.", "Can you help?", "The sky is clear.",
]
question = [1, 0, 1, 0, 1, 0, 1, 0]   # head "q": is it a question?
short = [1, 0, 1, 0, 1, 1, 1, 0]      # head "s": is it a short (<= 4 word) text?
dq = Dataset.from_texts(texts, question, model.tokenizer, probe_name="q",
                        label_site="response", append_eos=True)
ds = Dataset.from_texts(texts, short, model.tokenizer, probe_name="s",
                        label_site="response", append_eos=True)
data = [
    {"tokens": a["tokens"], "labels": {"q": a["labels"], "s": b["labels"]}}
    for a, b in zip(dq, ds, strict=True)
]

model.add_probes([ProbeConfig(name="q", layers=[-1]), ProbeConfig(name="s", layers=[-1])])
model.prepare_for_joint_training()


# A custom per-probe loss from the RAW logits: soft-margin (softplus of the signed
# margin). `probe.reduce` averages over the valid (non-padding) positions for you.
def soft_margin(probe: object, target: object) -> object:
    signed = 2.0 * target - 1.0  # {0, 1} labels -> {-1, +1}
    return probe.reduce(ops.softplus(-signed * probe.logits))


# "q" uses your custom loss; "s" uses the built-in BCE. JointLoss combines them.
# (For a NON-sum total, pass combine=lambda t: ... over the named terms instead.)
loss = JointLoss(weights={"lm_head": 0.0, "q": 1.0, "s": 1.0}, losses={"q": soft_margin})
Trainer(model=model, loss_fn=loss, num_iters=40, batch_size=4).train(data)

for text, want in [("Will it rain?", "q=1"), ("The grass is green.", "q=0")]:
    logit = float(model.forward([model.tokenizer.encode(text)]).probes["q"].logits[0, -1].item())
    print(f"{text:<22} q_logit = {logit:+5.2f}  (want {want})")

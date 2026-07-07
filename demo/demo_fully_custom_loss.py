"""Write a FULLY custom loss from scratch — total control over every probe.

Instead of JointLoss, pass Trainer any callable
``loss_fn(model, batch, labels, lengths) -> (total, ntoks, components)``. It runs the
model, reads the RAW probe outputs, and returns a scalar total plus a ``components``
dict for logging. ``ops`` keeps the math backend-agnostic. Backend-agnostic.

    python demo/demo_fully_custom_loss.py
"""

from __future__ import annotations

from auto_chasm import Dataset, Model, ProbeConfig, Trainer, ops

MODEL = "HuggingFaceTB/SmolLM2-135M"

model = Model.from_pretrained(MODEL)

texts = ["Is it far?", "The road is long.", "Are we there?", "The car is red.",
         "Do you know?", "The sky is clear.", "Can you help?", "Birds fly south."]
is_question = [1, 0, 1, 0, 1, 0, 1, 0]
data = Dataset.from_texts(texts, is_question, model.tokenizer, probe_name="p",
                          label_site="response", append_eos=True)

model.attach_probe(ProbeConfig(name="p", layers=[-1]))
model.prepare_for_joint_training()


def my_loss(model, batch, labels, lengths):  # noqa: ANN001, ANN201
    """A squared-hinge loss on the raw probe logits, written from scratch."""
    _, probes = model(batch[:, :-1])  # run the (wrapped) model; read the raw head output
    logits = probes["p"]  # [B, T-1] for a binary head
    target = labels[:, 1:]
    valid = target != -100  # exclude ignored / padding positions
    signed = 2.0 * target - 1.0  # {0, 1} -> {-1, +1}
    per_token = ops.clamp(1.0 - signed * logits, lo=0.0) ** 2  # squared hinge
    loss = ops.masked_mean(per_token, valid)  # scalar mean over valid positions
    return loss, ops.sum(valid), {"squared_hinge": loss}  # (total, ntoks, components)


result = Trainer(model=model, loss_fn=my_loss, num_iters=40, batch_size=4).train(data)
losses = result["history"].train_losses
print(f"\nsquared-hinge loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
for text, want in [("Are you sure?", "question"), ("It is late.", "statement")]:
    logit = float(model.forward([model.tokenizer.encode(text)]).probes["p"].logits[0, -1].item())
    print(f"{text:<18} logit = {logit:+5.2f}  (want {want})")

"""Save a trained probe and reload it, verifying the outputs are identical.

save_checkpoint bundles the probe weights, any adapters, the steering geometry, and
the config into one folder; from_checkpoint restores all of it. Backend-agnostic.

    python demo/demo_checkpoint.py
"""

from __future__ import annotations

from pathlib import Path

from auto_chasm import Dataset, JointLoss, Model, ProbeConfig, Trainer

MODEL = "HuggingFaceTB/SmolLM2-135M"
CKPT = str(Path(__file__).parent / "demo_checkpoint_output")

model = Model.from_pretrained(MODEL)

texts = ["Is it far?", "The road is long.", "Are we there?", "The car is red."]
labels = [1, 0, 1, 0]   # 1 = question, 0 = statement
data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="is_question",
                          label_site="response", append_eos=True)

model.attach_probe(ProbeConfig(name="is_question", layers=[-1]))
model.prepare_for_joint_training()
Trainer(
    model=model,
    loss_fn=JointLoss(weights={"lm_head": 0.0, "is_question": 1.0}),
    num_iters=20, batch_size=2,
).train(data)

probe_ids = [model.tokenizer.encode("Do you know?")]
before = float(model.forward(probe_ids).probes["is_question"].logits[0, -1].item())

model.save_checkpoint(CKPT)
restored = Model.from_checkpoint(CKPT)
after = float(restored.forward(probe_ids).probes["is_question"].logits[0, -1].item())

print(f"\nprobe logit before save = {before:+.5f}")
print(f"probe logit after  load = {after:+.5f}")
print("identical after round-trip." if abs(before - after) < 1e-3 else "MISMATCH!")

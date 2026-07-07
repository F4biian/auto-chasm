"""Train a probe, then steer generation along the direction it learned.

Steering edits the model's activations at inference along the axis a probe encodes.
It needs the per-class average activations ("class means"), computed once from data.
This trains a positive/negative sentiment probe and generates with steering off,
then on. Backend-agnostic.

    python demo/demo_steering.py
"""

from __future__ import annotations

from auto_chasm import Dataset, JointLoss, Model, ProbeConfig, SteeringConfig, Trainer

MODEL = "HuggingFaceTB/SmolLM2-135M"

model = Model.from_pretrained(MODEL)

texts = ["I love this, it is wonderful.", "This is awful and I hate it.",
         "What a delightful morning.", "Everything is broken and sad.",
         "A truly fantastic result.", "A miserable, hopeless day."]
sentiment = [1, 0, 1, 0, 1, 0]   # 1 = positive, 0 = negative
data = Dataset.from_texts(texts, sentiment, model.tokenizer, probe_name="sentiment",
                          label_site="response", append_eos=True)

model.attach_probe(ProbeConfig(name="sentiment", layers=[-1]))
model.prepare_for_joint_training()
Trainer(
    model=model,
    loss_fn=JointLoss(weights={"lm_head": 0.0, "sentiment": 1.0}),
    num_iters=30, batch_size=2,
).train(data)

# The steering geometry: the mean activation for each class at the probe's layer.
class_means = model.compute_class_means(data)

prompt = "Today the weather is"
print("no steering :", model.generate(prompt, max_tokens=12))
model.enable_steering("sentiment", config=SteeringConfig(method="push_to_mean", scale=8.0),
                      class_means=class_means)
print("steered     :", model.generate(prompt, max_tokens=12))
model.disable_steering("sentiment")

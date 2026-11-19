# 5. Training

`Trainer` runs the training loop. `JointLoss` decides what is trained. Both work the same
on MLX and PyTorch.

[All guides](README.md) · Previous: [4. Probes](04-probes.md) · Next: [6. Custom losses](06-custom-losses.md)

## Setup

```python
from auto_chasm import Dataset, JointLoss, Model, ProbeConfig, Trainer

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")

texts = [
    "Is it raining?", "The sky is blue.", "Where are you?", "Cats purr.",
    "What time is it?", "I like tea.", "Can birds fly?", "The door is open.",
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]
data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="question", append_eos=True)
train, val = data.split(0.25, seed=0)

model.attach_probe(ProbeConfig(name="question", layers=[-1]))
model.prepare_for_joint_training()
```

## A typical run

```python
trainer = Trainer(
    model=model,
    loss_fn=JointLoss(weights={"lm_head": 0.0, "question": 1.0}),
    num_iters=40,
    batch_size=4,
    learning_rate=1e-3,
    eval_steps=10,
)
result = trainer.train(train, val_data=val)
print(result["history"][-1])
```

`eval_steps=10` evaluates on `val` every 10 steps. `train` returns a dict with the
`"history"` of all steps, the `"test_metrics"` (if you pass `test_data`) and the
`"output_dir"`. The run also writes `training_history.json` and
`training_manifest.json` to `output_dir` (by default `./checkpoints`).

## Choosing the loss

`JointLoss` has one term per name: `"lm_head"` for the language model, and one per
probe. By default it adds them up with their weights:

```python
probe_only = JointLoss(weights={"lm_head": 0.0, "question": 1.0})
both = JointLoss(weights={"lm_head": 1.0, "question": 2.0})
```

Each probe can use its own loss:

| Loss | For |
|---|---|
| `"bce"` (default) | binary classification (one output) |
| `"ce"` | multi-class classification |
| `"mse"`, `"mae"` | regression |

```python
JointLoss(losses={"question": "bce"})
```

A term with weight `0` is skipped. To write your own loss, see
[6. Custom losses](06-custom-losses.md).

## Early stopping

By default a run trains for all `num_iters` steps and keeps the final weights. Two
options change that, and they are independent:

- `early_stopping_patience=4` stops after 4 evaluations without improvement.
- `restore_best_weights=True` ends with the weights of the best evaluation.

```python
trainer = Trainer(
    model=model,
    loss_fn=JointLoss(weights={"lm_head": 0.0, "question": 1.0}),
    num_iters=40,
    eval_steps=5,
    early_stopping_patience=4,
    restore_best_weights=True,
)
trainer.train(train, val_data=val)
```

To stop on a probe metric such as F1 instead of the loss, compute it with
`eval_metrics_fn` and name it in `early_stopping_metric`:

```python
from auto_chasm.trainers import default_binary_metrics

trainer = Trainer(
    model=model,
    loss_fn=JointLoss(weights={"lm_head": 0.0, "question": 1.0}),
    num_iters=40,
    eval_steps=5,
    eval_metrics_fn=default_binary_metrics,
    early_stopping_metric="val_question_f1",
    early_stopping_higher_is_better=True,
    early_stopping_patience=4,
)
trainer.train(train, val_data=val)
```

The metric names follow the pattern `val_<probe>_<metric>`.

## Running out of memory

If training is killed by the operating system, or fails with
`[metal::malloc] Resource limit (499000) exceeded`, turn on gradient checkpointing
before training:

```python
model.enable_gradient_checkpointing()
```

It keeps only each layer's input and recomputes the rest during the backward pass.
On Qwen3.5-0.8B this cut peak memory by about 3.5 times, at 15 to 25 percent more time
per step.

If that is not enough, lower `max_seq_length`, then `batch_size`.

<details>
<summary>Why some models need much more memory than their size suggests</summary>

Memory grows with sequence length times batch size, and the factor depends on the
architecture. Models with linear-attention or state-space layers (Qwen3.5, for example)
can need ten times more than a classic transformer of the same size, because their fast
kernel has no backward pass and training falls back to a step-by-step loop. The trainer
detects such layers and prints a `[memory]` note before the first step.

The `metal::malloc` error is a limit on the number of buffers, not on memory, so free
RAM does not help. Reduce `max_seq_length` first, then `batch_size`. For the same
reason, the trainer does not compile the training step for these models; pass
`compile_step=True` or `False` to decide yourself.
</details>

## Mixed precision

Run the frozen model in bfloat16, while the trainable weights stay in float32:

```python
from auto_chasm import TrainingConfig

trainer = Trainer(
    model=model,
    loss_fn=JointLoss(weights={"lm_head": 0.0, "question": 1.0}),
    num_iters=10,
    config=TrainingConfig(mixed_precision="bf16"),
)
trainer.train(train)
```

`"bf16"` works on both backends. `"fp16"` is available on PyTorch only. See
[`demo/demo_mixed_precision.py`](../demo/demo_mixed_precision.py).

## Class imbalance

If one class is rare, give it a higher weight. Class weights work with the `"ce"` loss,
so the probe needs one output per class:

```python
model.attach_probe(ProbeConfig(name="kind", layers=[-1], module_config={"out_features": 2}))
kind_data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="kind", append_eos=True)

weights = kind_data.class_weights(num_classes=2)
loss = JointLoss(weights={"lm_head": 0.0, "question": 0.0}, losses={"kind": "ce"}, class_weights=weights)
Trainer(model=model, loss_fn=loss, num_iters=10, batch_size=4).train(kind_data)
```

`class_weights="balanced"` computes the weights from the training data for you.

<details>
<summary>All <code>Trainer</code> options</summary>

| Option | Default | Meaning |
|---|---|---|
| `learning_rate` | `2e-4` | peak learning rate |
| `num_iters` | `500` | training steps |
| `batch_size` | `8` | samples per step |
| `max_seq_length` | `256` | longer samples are cut |
| `grad_accum_steps` | `1` | steps per optimizer update |
| `weight_decay` | `0.0` | AdamW weight decay |
| `grad_clip_norm` | `1.0` | gradient clipping |
| `lr_schedule` | `"cosine"` | `"cosine"`, `"linear"` or `"constant"` |
| `warmup_ratio` | `0.0` | share of updates used for warmup |
| `eval_steps` | `None` | evaluate every N steps |
| `eval_metrics_fn` | `None` | extra validation metrics |
| `early_stopping_patience` | `0` | `0` turns early stopping off |
| `early_stopping_metric` | `"val_loss"` | the metric to watch |
| `early_stopping_higher_is_better` | `False` | `True` for accuracy, F1, AUROC |
| `min_delta` | `1e-4` | smallest change that counts as improvement |
| `restore_best_weights` | `False` | end with the best weights |
| `logging_steps` | `25` | print the loss every N steps |
| `save_steps` | `100` | save every N steps |
| `keep_best_only` | `False` | keep only the best saved step |
| `output_dir` | `"./checkpoints"` | where logs and saves go |
| `save_history` | `True` | write `training_history.json` |
| `callbacks` | `None` | functions called during training |
| `verbose` | `True` | print progress |
| `seed` | `None` | seed for the batch order |
| `compile_step` | `None` | compile the MLX training step; `None` decides per model |
| `config` | `None` | a `TrainingConfig`; explicit options win over it |
</details>

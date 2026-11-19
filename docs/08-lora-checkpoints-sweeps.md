# 8. LoRA, checkpoints and layer sweeps

[All guides](README.md) · Previous: [7. Generation and steering](07-generation-and-steering.md) · Next: [9. Mass-mean probes and evaluation](09-mass-mean-and-evaluation.md)

## Setup

```python
from auto_chasm import Dataset, JointLoss, LoraConfig, Model, ProbeConfig, Trainer

texts = [
    "Is it raining?", "The sky is blue.", "Where are you?", "Cats purr.",
    "What time is it?", "I like tea.", "Can birds fly?", "The door is open.",
    "Who is there?", "Dogs bark.", "Why is it cold?", "The tea is hot.",
]
labels = [1, 0] * 6
```

## LoRA: train the model together with the probes

Load the model with LoRA adapters. After `prepare_for_joint_training`, the base model
is frozen, and the adapters and probes are trainable:

```python
model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M", lora=LoraConfig(rank=16, alpha=32))
data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="question", append_eos=True)

model.attach_probe(ProbeConfig(name="question", layers=[-1]))
model.prepare_for_joint_training()

loss = JointLoss(weights={"lm_head": 1.0, "question": 1.0})
Trainer(model=model, loss_fn=loss, num_iters=20, batch_size=4).train(data)
```

Both losses now update the adapters, and the probe loss also trains the probe head.

`LoraConfig` options:

| Option | Default | Meaning |
|---|---|---|
| `rank` | `8` | size of the adapters |
| `alpha` | `16` | scaling of the adapters |
| `peft_method` | `"lora"` | `"lora"`, `"qlora"` (quantized base) or `"dora"` |
| `target_modules` | `None` | which layers get adapters; `None` means every linear layer in attention and MLP blocks |

To see which modules can get adapters:

```python
print(model.lora_targetable_modules[:3])
```

## Checkpoints

`save_checkpoint` writes one folder with the probes, the adapters, the steering setup
and all settings. `from_checkpoint` loads all of it, including the base model:

```python
model.save_checkpoint("runs/question-lora")
restored = Model.from_checkpoint("runs/question-lora")
```

`from_checkpoint` loads on the backend the checkpoint was saved with. Pass
`backend_name="mlx"` or `"torch"` to switch, and `load_steering=False` to skip the
steering setup. See [`demo/demo_checkpoint.py`](../demo/demo_checkpoint.py).

## Layer sweeps

Which layer knows the most about your property? `LayerSweep` trains one probe per
layer in a single pass over a frozen model, and keeps each layer's probe at its best
validation step:

```python
from auto_chasm import LayerSweep, Task

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="question", append_eos=True)
train, rest = data.split(0.5, seed=0)
val, test = rest.split(0.5, seed=0)

sweep = LayerSweep(model, task=Task.binary())
result = sweep.run(
    train, val, test,
    loss_fn=JointLoss(weights={"lm_head": 0.0}),
    num_iters=20,
    eval_every=5,
)
print(result.best_layer())
result.to_csv("sweep.csv")
```

`result.plot("sweep.png")` draws the metrics per layer. For error bars on the result,
see [AUROC with confidence intervals](09-mass-mean-and-evaluation.md#auroc-with-confidence-intervals).
Runnable version: [`demo/demo_layer_sweep.py`](../demo/demo_layer_sweep.py).

<details>
<summary>All <code>LayerSweep</code> options</summary>

| Option | Default | Meaning |
|---|---|---|
| `task` | `None` | a `Task`; sets the head size, classes and metrics |
| `out_features` | `None` | the head size, if you do not pass a `task` |
| `layers` | `None` | the layers to sweep; `None` means all |
| `module_spec` | `None` | a `ModuleSpec` for the heads |
| `score_metric` | `"val_loss"` | picks each layer's best step, e.g. `"val_acc"` or `"val_auroc"` |
| `higher_is_better` | `False` | set `True` for accuracy, F1 or AUROC |
| `early_stopping_patience` | `0` | stop a layer after this many evaluations without improvement |
| `min_delta` | `0.0` | smallest change that counts as improvement |
| `eval_metrics_fn` | `None` | extra metrics |

`run(train, val, test=None, *, loss_fn, num_iters, eval_every, **trainer_options)`
passes other options on to `Trainer`. Pass `test=None` to skip the test pass.
Snapshots are kept in memory, not on disk.
</details>

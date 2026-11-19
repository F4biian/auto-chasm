# 10. API reference

Everything here is imported with `from auto_chasm import ...`. Each class and function
has a docstring, so `help(Model.generate)` shows all its options.

[All guides](README.md) · Previous: [9. Mass-mean probes and evaluation](09-mass-mean-and-evaluation.md)

## Public names

| Name | What it is | Guide |
|---|---|---|
| `Model` | load a model, attach probes, train, generate, steer, save | [1](01-getting-started.md) |
| `Dataset` | labeled data from texts or conversations | [2](02-data.md) |
| `ProbeConfig` | describes one probe | [4](04-probes.md) |
| `ModuleSpec` | a head architecture that works on both backends | [4](04-probes.md#your-own-head) |
| `Probe` | an attached probe, found in `model.probes[name]` | [4](04-probes.md) |
| `Trainer` | the training loop | [5](05-training.md) |
| `TrainingConfig` | training options as one object | [5](05-training.md#mixed-precision) |
| `SFTTrainer` | a supervised fine-tuning loop (MLX only) | |
| `JointLoss` | the combined language-model and probe loss | [5](05-training.md#choosing-the-loss) |
| `ops` | math functions for custom losses on both backends | [6](06-custom-losses.md) |
| `GenerationConfig` | generation options as one object | [7](07-generation-and-steering.md) |
| `SteeringConfig` | steering method and strength | [7](07-generation-and-steering.md#steer-the-model) |
| `LoraConfig` | LoRA, QLoRA and DoRA settings | [8](08-lora-checkpoints-sweeps.md) |
| `LayerSweep`, `SweepResult` | one probe per layer, and its results | [8](08-lora-checkpoints-sweeps.md#layer-sweeps) |
| `Task` | binary, multi-class or regression | [8](08-lora-checkpoints-sweeps.md#layer-sweeps) |
| `classification_metrics`, `regression_metrics` | metric helpers | |
| `no_grad` | turns off gradients on either backend | |
| `set_default_thinking` | turns the `<think>` block of reasoning models on or off | [2](02-data.md#chat-templates) |

More specialized tools live in submodules, for example
`auto_chasm.trainers.default_binary_metrics` or `auto_chasm.data.build_dataset`.

## Model information

```python
from auto_chasm import Model, no_grad

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")

print(model.num_layers, model.hidden_size)
print(model.num_parameters(), model.num_parameters(trainable=True))
print(model.stats())
```

`no_grad` saves memory when you only run the model:

```python
with no_grad(model):
    out = model.forward([model.tokenizer.encode("Hello")])
```

## Backends

The same code runs on MLX (Apple silicon) and PyTorch (CUDA or CPU). The backend is
picked from what is installed; `backend_name="mlx"` or `"torch"` chooses one. Models load
through `mlx-lm` on MLX and through `transformers` on PyTorch. The package ships type
hints, so your editor can complete the whole API.

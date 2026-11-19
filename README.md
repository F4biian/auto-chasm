# auto-chasm

Attach small **probe heads** to a language model, **train them together** with the
model, and **read them live** while the model writes. The same code runs on Apple's
**MLX** and on **PyTorch**.

> [!NOTE]
> auto-chasm was built with AI. Code, tests and documentation were written with the
> support of [Claude Code](https://claude.com/claude-code). Please double-check results
> that matter to you, and report bugs as issues.

## What is a probe?

A probe is a tiny classifier or regressor. It looks at the hidden activations of one
layer and predicts a property of the text, for example "is this a question?".

With auto-chasm you can:

- build labeled data from plain texts or chat conversations,
- attach probes to any layer,
- train probes alone, or together with the model through one joint loss,
- use built-in losses or write your own,
- read every probe token by token while the model generates,
- steer the model along what a probe has learned,
- compare layers and report AUROC with confidence intervals.

## Install

On Apple silicon (MLX):

```bash
pip install "auto-chasm[mlx] @ git+https://github.com/F4biian/auto-chasm.git"
```

On CUDA or CPU (PyTorch):

```bash
pip install "auto-chasm[torch] @ git+https://github.com/F4biian/auto-chasm.git"
```

## Quickstart

Teach a probe to recognize questions, while the model itself is fine-tuned with LoRA:

```python
from auto_chasm import Dataset, JointLoss, LoraConfig, Model, ProbeConfig, Trainer

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M", lora=LoraConfig(rank=8))

texts = [
    "Is it raining?", "The sky is blue.", "Where are you?", "Cats purr.",
    "What time is it?", "I like tea.", "Can birds fly?", "The door is open.",
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]
data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="question", append_eos=True)

model.attach_probe(ProbeConfig(name="question", layers=[-1]))
model.prepare_for_joint_training()

loss = JointLoss(weights={"lm_head": 1.0, "question": 1.0})
Trainer(model=model, loss_fn=loss, num_iters=40, batch_size=4).train(data)
```

What happens here:

1. `Dataset.from_texts` gives each text its label (1 = question).
2. `attach_probe` adds a linear head that reads the last layer.
3. `JointLoss` adds up the language-model loss (`"lm_head"`) and the probe loss, so
   the model and the probe train together.

Now ask the probe about new sentences. A higher score means "more likely a question":

```python
for text in ["Do you like music?", "Music is nice."]:
    ids = model.tokenizer.encode(text)
    score = model.forward([ids]).probes["question"].logits[0, -1].item()
    print(f"{text:20} {score:+.2f}")
```

### Use your own loss

A probe loss is any function of the probe output and the target. This one is the
squared error of the predicted probability:

```python
from auto_chasm import ops

def squared_error(probe, target):
    probability = ops.sigmoid(probe.logits)
    return probe.reduce((probability - target) ** 2)

loss = JointLoss(weights={"lm_head": 1.0, "question": 1.0}, losses={"question": squared_error})
Trainer(model=model, loss_fn=loss, num_iters=40, batch_size=4).train(data)
```

`probe.reduce` averages over the labeled tokens, and `ops` works on MLX and PyTorch
alike. More options are in [Custom losses](docs/06-custom-losses.md).

## Documentation

Start with [Getting started](docs/01-getting-started.md). Every guide is short and has
copy-paste examples.

| Guide | You learn how to |
|---|---|
| [1. Getting started](docs/01-getting-started.md) | go from a model to a trained, saved probe in six steps |
| [2. Data and labels](docs/02-data.md) | label texts, chat conversations and single tokens |
| [3. LM token weights](docs/03-lm-token-weights.md) | train the model only on some tokens, or make it unlearn them |
| [4. Probes](docs/04-probes.md) | read other layers, use several probes, build your own head |
| [5. Training](docs/05-training.md) | choose losses, handle class imbalance, stop early, save memory |
| [6. Custom losses](docs/06-custom-losses.md) | write your own loss, from one probe term to a full loss function |
| [7. Generation and steering](docs/07-generation-and-steering.md) | generate text, read probes per token, steer the model |
| [8. LoRA, checkpoints, layer sweeps](docs/08-lora-checkpoints-sweeps.md) | fine-tune with adapters, save runs, find the best layer |
| [9. Mass-mean probes and evaluation](docs/09-mass-mean-and-evaluation.md) | fit probes without training, report AUROC with error bars |
| [10. API reference](docs/10-api-reference.md) | look up any public name |

## Examples

Each script in [`demo/`](demo/) runs on its own:

| Script | Shows |
|---|---|
| [`demo_train_probe.py`](demo/demo_train_probe.py) | train a probe and read it during generation |
| [`demo_multihead.py`](demo/demo_multihead.py) | several probes with different tasks |
| [`demo_probe_sources.py`](demo/demo_probe_sources.py) | probes on attention, MLP and several layers |
| [`demo_custom_head.py`](demo/demo_custom_head.py) | your own head architecture |
| [`demo_custom_loss.py`](demo/demo_custom_loss.py) | a custom probe loss |
| [`demo_fully_custom_loss.py`](demo/demo_fully_custom_loss.py) | a loss function written from scratch |
| [`demo_regression.py`](demo/demo_regression.py) | a probe that predicts a number |
| [`demo_mixed_precision.py`](demo/demo_mixed_precision.py) | training in bf16 |
| [`demo_steering.py`](demo/demo_steering.py) | steering generation with a probe |
| [`demo_layer_sweep.py`](demo/demo_layer_sweep.py) | finding the best layer |
| [`demo_checkpoint.py`](demo/demo_checkpoint.py) | saving and restoring a run |
| [`demo_kv_cache.py`](demo/demo_kv_cache.py) | fast cached generation |
| [`demo_model_stats.py`](demo/demo_model_stats.py) | architecture and parameter counts |

```bash
python demo/demo_train_probe.py
```

## Development

```bash
uv sync --all-extras
uv run pytest
```

Add `--run-real-model` to also run the tests that load real models. The test layout is
described in [`tests/README.md`](tests/README.md).

## License

MIT, see [LICENSE](LICENSE).

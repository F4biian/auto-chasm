# 6. Custom losses

The built-in losses cover the common cases. When you need something else, pick the
smallest of three options:

1. a custom loss for one probe,
2. a custom formula that combines the loss terms,
3. a loss function written from scratch.

All three use `ops`, a small math module that works on MLX and PyTorch, so the same
loss runs on both.

[All guides](README.md) · Previous: [5. Training](05-training.md) · Next: [7. Generation and steering](07-generation-and-steering.md)

## Setup

```python
from auto_chasm import Dataset, JointLoss, Model, ProbeConfig, Trainer, ops

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")

texts = [
    "Is it raining?", "The sky is blue.", "Where are you?", "Cats purr.",
    "What time is it?", "I like tea.", "Can birds fly?", "The door is open.",
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]
data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="question", append_eos=True)

model.attach_probe(ProbeConfig(name="question", layers=[-1]))
model.prepare_for_joint_training()
```

## Option 1: A loss for one probe

Write a function that takes the probe output and the target, and pass it in `losses`:

```python
def squared_error(probe, target):
    probability = ops.sigmoid(probe.logits)
    return probe.reduce((probability - target) ** 2)

loss = JointLoss(weights={"lm_head": 0.0, "question": 1.0}, losses={"question": squared_error})
Trainer(model=model, loss_fn=loss, num_iters=40, batch_size=4).train(data)
```

The `probe` argument gives you:

| Attribute | What it is |
|---|---|
| `probe.logits` | the raw head output |
| `probe.softmax()`, `probe.log_softmax()` | normalized outputs for multi-class heads |
| `probe.reduce(values)` | the mean over all labeled tokens (padding is ignored) |

Other probes keep their built-in loss, and `JointLoss` adds everything up as usual.
See [`demo/demo_custom_loss.py`](../demo/demo_custom_loss.py).

## Option 2: A custom formula

By default the total loss is a weighted sum. To combine the terms differently, pass a
function to `combine`. It receives all terms by name:

```python
loss = JointLoss(combine=lambda terms: terms.lm_head + 2.0 * terms.question ** 2)
```

## Option 3: A loss from scratch

`Trainer` also accepts any function `loss_fn(model, batch, labels, lengths)`. It runs
the model itself and returns three things: the loss, the number of tokens, and a dict
of values to log.

```python
def hinge_loss(model, batch, labels, lengths):
    _, probes = model(batch[:, :-1])
    logits, target = probes["question"], labels[:, 1:]
    valid = target != -100
    per_token = ops.clamp(1.0 - (2.0 * target - 1.0) * logits, lo=0.0)
    total = ops.masked_mean(per_token, valid)
    return total, ops.sum(valid), {"hinge": total}

Trainer(model=model, loss_fn=hinge_loss, num_iters=40, batch_size=4).train(data)
```

Two details to know:

- The model gets `batch[:, :-1]` and the labels are `labels[:, 1:]`, because each
  position predicts the next token.
- Labels of `-100` mark positions without a label (and padding). Leave them out.

See [`demo/demo_fully_custom_loss.py`](../demo/demo_fully_custom_loss.py).

<details>
<summary>What <code>ops</code> offers</summary>

`abs`, `arange`, `argmax`, `clamp`, `exp`, `log`, `log_softmax`, `logsumexp`,
`masked_mean`, `max`, `mean`, `pow`, `sigmoid`, `softmax`, `softplus`, `sqrt`, `sum`,
`where` and `zeros_like`. Each one runs on MLX or PyTorch, depending on its input.
</details>

## Regression

A probe with one output and the `"mse"` loss predicts a number. Here it learns the word
count of a text:

```python
sentences = [
    "Yes.", "Go now.", "The cat sat.", "Birds fly south.", "I like green tea.",
    "We walked to the park.", "She reads a book every night.", "Rain again.",
]
counts = [float(len(s.split())) for s in sentences]
words = Dataset.from_texts(sentences, counts, model.tokenizer, probe_name="words", append_eos=True)

model.attach_probe(ProbeConfig(name="words", layers=[-1]))
model.prepare_for_joint_training()

loss = JointLoss(weights={"lm_head": 0.0, "words": 1.0}, losses={"words": "mse"})
Trainer(model=model, loss_fn=loss, num_iters=100, batch_size=4, learning_rate=1e-3).train(words)

ids = model.tokenizer.encode("The dog sleeps.")
print(model.forward([ids]).probes["words"].logits[0, -1].item())
```

See [`demo/demo_regression.py`](../demo/demo_regression.py) for a longer version.

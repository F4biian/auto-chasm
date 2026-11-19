# 3. LM token weights: mask and unlearn tokens

Probes have their own labels. The language model has a label channel too: the special
key `"lm_head"`. With it you decide, token by token, whether the model should learn a
token, ignore it, or learn to avoid it.

[All guides](README.md) · Previous: [2. Data and labels](02-data.md) · Next: [4. Probes](04-probes.md)

## What a weight does

| Weight | Effect on that token |
|---|---|
| `1.0` (default) | trained normally |
| `2.0`, `0.5`, ... | trained, counted more or less |
| `0.0` | ignored |
| `-1.0`, `-2.0`, ... | unlearned: the model is pushed to make it less likely |

Negative weights use [unlikelihood training](https://arxiv.org/abs/1908.04319). Its
loss goes to zero once a token is unlikely, so unlearning cannot run away.

## Setup

```python
from auto_chasm import Dataset, JointLoss, Model, ProbeConfig, Trainer

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
```

## Step 1: Train only on the assistant's replies

`lm_train_on` sets a weight per message: `1.0` for the roles you name, `0.0` for all
other roles.

```python
conversations = [[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What year did the war end?"},
    {"role": "assistant", "content": "It ended in 1945."},
]]
data = Dataset.from_conversations(conversations, model.tokenizer, lm_train_on="assistant")
```

> [!IMPORTANT]
> `lm_train_on` also masks the `system` role. To train it as well, name it:
> `lm_train_on=("assistant", "system")`. Without `lm_train_on`, every token trains the
> model.

## Step 2: Change single tokens

Inside a message, list weights under `"lm_head"`. They replace the message's weight for
the tokens they cover:

```python
conversations = [[
    {"role": "user", "content": "Who invented the telephone?"},
    {"role": "assistant", "content": "It was invented by Thomas Edison in 1876.",
     "labels": {"lm_head": [
         {"text": "Thomas Edison", "weight": -1.0},
         {"regex": r"\d{4}", "weight": 0.0},
     ]}},
]]
data = Dataset.from_conversations(conversations, model.tokenizer, lm_train_on="assistant")
```

Here the model unlearns the wrong name and ignores the year. A spec can select tokens in
four ways:

| Spec | Selects |
|---|---|
| `{"start": 10, "end": 20, "weight": w}` | a character span |
| `{"text": "Edison", "weight": w}` | every occurrence of a text |
| `{"regex": r"\d{4}", "weight": w}` | every match of a regular expression |
| `{"token_ids": [1234, 567], "weight": w}` | an exact token sequence |

## Step 3: Train

There is nothing extra to configure. `Trainer` and `JointLoss` read the weights from the
data:

```python
Trainer(model=model, loss_fn=JointLoss(), num_iters=5, batch_size=1).train(data)
```

The same messages can carry probe labels as well. Probe labels and `"lm_head"` weights
are independent.

## Rules worth knowing

- **Message weight first, then your specs.** Specs override the role's weight for the
  tokens they cover, in both directions: a `1.0` span on a user message trains it.
- **Overlapping specs take the smallest weight**, so the strongest intervention wins.
- **Chat template tokens.** The tokens that open a turn are never trained. The token
  that closes a turn gets the role's weight, so the model still learns to stop.
- **No custom LM loss on top.** `JointLoss(losses={"lm_head": my_fn})` together with
  weights raises an error, because the weights would be ignored otherwise.

<details>
<summary>How the weights are normalized</summary>

By default, weights are absolute: the token losses of a batch are added up and divided
by the number of tokens with a weight other than `0`. A token with weight `-64` then
always counts 64 times as much as one with `-1`.

`JointLoss(lm_weight_normalization="weights")` divides by the sum of the absolute
weights instead. Then a weight only counts relative to the other tokens in its batch.
</details>

<details>
<summary>Limits of unlearning</summary>

A token that is already unlikely adds almost nothing to the loss, and one token's loss
is capped at about 11.5. What unlikelihood does not control is where the probability
goes instead: it spreads over the rest of the vocabulary. Prefer small, targeted spans,
and watch the loss on normal tokens.
</details>

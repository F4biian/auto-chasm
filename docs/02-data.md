# 2. Data and labels

You say *where* a label belongs in the text, and auto-chasm puts it on the right
tokens. There are two ways to build a dataset:

- `Dataset.from_texts`: plain texts with one label each. Start here.
- `Dataset.from_conversations`: chat conversations with labels on parts of the text.

[All guides](README.md) · Previous: [1. Getting started](01-getting-started.md) · Next: [3. LM token weights](03-lm-token-weights.md)

## Setup

```python
from auto_chasm import Dataset, Model

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
```

## One label per text

```python
texts = [
    "Is it raining?", "The sky is blue.", "Where are you?", "Cats purr.",
    "What time is it?", "I like tea.", "Can birds fly?", "The door is open.",
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]

data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="question", append_eos=True)
```

Labels are class numbers (`0`, `1`, `2`, ...) for classification, or floats for
[regression](06-custom-losses.md#regression).

`label_site` decides which tokens get the label:

| `label_site` | The label goes on | Good for |
|---|---|---|
| `"response"` (default) | the end of the text | properties of the whole text |
| `"token"` | every token after the first `warmup_chars` characters | properties of each token |
| `"sentence"` | the end of each sentence | properties of each sentence |

> [!TIP]
> Use `append_eos=True` when there is one label per text. During training, each
> position predicts the *next* token, so the label on the last token would never be
> used. `append_eos=True` adds an end token and moves the label there, so the probe
> learns from the state after the whole text.

<details>
<summary>All options of <code>from_texts</code></summary>

| Option | Default | Meaning |
|---|---|---|
| `probe_name` | `"probe"` | which probe these labels train |
| `label_site` | `"response"` | `"response"`, `"token"` or `"sentence"` |
| `append_eos` | `False` | add an end token and move a `"response"` label onto it |
| `warmup_chars` | `50` | for `"token"`: the first labeled character |
| `sentence_delimiters` | `None` | for `"sentence"`, e.g. `[".", "!", "?"]` |
| `default_label` | `None` | label for unlabeled tokens; `None` leaves them out |
| `offset` | `0` | shift all labels by this many tokens |
| `groups` | `None` | one group key per text, used by `split(groups="group")` |

If the tokenizer has a chat template, each text is rendered as one user message.
</details>

## Conversations with labeled spans

For more control, write each example as a conversation. A message can label a part of
its text with a character span:

```python
conversations = [
    [{"role": "user", "content": "I have 3 cats",
      "labels": {"digit": [{"start": 7, "end": 8, "label": 1}]}}],
    [{"role": "user", "content": "I have no cats",
      "labels": {"digit": []}}],
]
spans = Dataset.from_conversations(conversations, model.tokenizer, default_label=0)
```

Here the character `"3"` (positions 7 to 8) is labeled `1`. How the other tokens are
treated:

- With `default_label=0`, all other tokens of a labeled message become `0`. Without it,
  they are left out of training.
- An empty list (`"digit": []`) means "this message has no positive tokens". Use it for
  negative examples.
- A message without the probe's key is left out for that probe.

## Labeling the tokens between two turns

A chat template adds tokens between the turns, for example
`<|im_end|>\n<|im_start|>assistant\n`. These tokens are interesting: the user has
asked, but the model has not answered yet. They belong to no message, so you label
them by naming a *site*:

```python
conversations = [[
    {"role": "user", "content": "What is 2 + 2?",
     "labels": {"answerable": [{"at": "after", "index": -1, "label": 1}]}},
    {"role": "assistant", "content": "4"},
]]
gap = Dataset.from_conversations(conversations, model.tokenizer)
```

This labels the last token before the answer starts.

| Site | Tokens |
|---|---|
| `"content"` | the message text |
| `"prefix"` / `"suffix"` | the template tokens at the start / end of this message |
| `"before"` / `"after"` | everything between this message's text and the previous / next one |
| `"message"` | prefix, content and suffix together |

<details>
<summary>More about sites</summary>

- Sites follow the template, so `"after"` on a user turn is the same position for
  every model. For code that should work across models, prefer `"before"` and
  `"after"` over `"prefix"` and `"suffix"`.
- `"index"` picks one token of the site: `0` is the first, `-1` the last.
- `"token_ids": [...]` matches an exact token sequence inside the site.
- A site labels only what it names. The message text stays unlabeled, even with
  `default_label=0`.
- A site that matches nothing logs a warning.
</details>

## Chat templates

`from_conversations` renders conversations with the tokenizer's chat template, so the
model sees the same format in training and later in use. The template tokens and the
message text are tokenized separately, so character spans always stay correct.
To join the raw texts without a template, pass `chat_template=False`.

**Reasoning models.** Some models (Qwen3, for example) add a `<think>` block. Switch it
on or off once, and both data and generation follow:

```python
from auto_chasm import set_default_thinking

set_default_thinking(False)
```

## Split the data

```python
train, val = data.split(0.25, seed=0)
train, val = data.split(0.25, stratify="label", seed=0)
```

The second line keeps the class balance equal on both sides. With `groups=` in
`from_texts`, `data.split(0.25, groups="group")` keeps every group on one side.
Splits always keep whole texts together.

Two more helpers:

```python
weights = data.class_weights(num_classes=2)
task = data.infer_task()
print(weights, task)
```

`class_weights` gives rarer classes a higher weight (see
[class imbalance](05-training.md#class-imbalance)), and `infer_task` detects
binary, multi-class or regression from the labels.

Next: [3. LM token weights](03-lm-token-weights.md) shows how to control which tokens
train the language model itself.

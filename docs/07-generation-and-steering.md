# 7. Generation and steering

Generate text, read every probe for each new token, and steer the model with what a
probe has learned.

[All guides](README.md) · Previous: [6. Custom losses](06-custom-losses.md) · Next: [8. LoRA, checkpoints, layer sweeps](08-lora-checkpoints-sweeps.md)

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

model.attach_probe(ProbeConfig(name="question", layers=[-1]))
model.prepare_for_joint_training()
Trainer(model=model, loss_fn=JointLoss(weights={"lm_head": 0.0}), num_iters=40, batch_size=4).train(data)
```

## Generate text

```python
print(model.generate("Once upon a time", max_tokens=20))
print(model.generate("Once upon a time", max_tokens=20, temperature=0.7))
```

The default `temperature=0` always picks the most likely token. A higher temperature
samples, so the text varies.

To get the text piece by piece as it is written:

```python
for piece in model.generate_stream("Once upon a time", max_tokens=20):
    print(piece, end="", flush=True)
```

To stop early, pass a stop text or stop token ids:

```python
print(model.generate("List three colors:", max_tokens=50, stop_sequences=["\n\n"]))
```

**Chat models.** Instruct models can take a list of messages instead of a prompt:

```python
chat_model = Model.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
messages = [{"role": "user", "content": "Say hello in French."}]
print(chat_model.generate(messages=messages, max_tokens=20))
```

<details>
<summary>All generation options</summary>

| Option | Default | Meaning |
|---|---|---|
| `max_tokens` | `256` | the maximum number of new tokens |
| `temperature` | `0.0` | `0` = always the most likely token |
| `top_p`, `top_k` | off | limit sampling to the most likely tokens |
| `repetition_penalty` | off | make repeated tokens less likely |
| `stop_sequences` | `None` | stop when one of these texts appears |
| `stop_tokens` | `None` | stop at one of these token ids |
| `max_repeat` | `256` | stop when the same token repeats this often; `None` turns it off |
| `messages` | `None` | chat messages instead of a prompt (needs a chat template) |
| `use_cache` | `True` | reuse earlier computations; same output, much faster |

A `GenerationConfig` holds these options as one object: `model.generate(prompt, config=...)`.
See [`demo/demo_kv_cache.py`](../demo/demo_kv_cache.py) for the speed of the cache.
</details>

## Read probes while generating

`generate_with_probes` gives you one step per new token, with the token and the output of
every probe:

```python
for step in model.generate_with_probes("Tell me, do you", max_tokens=8):
    score = step.probes["question"].logits[0, -1].item()
    print(f"{step.token_str!r:12} {score:+.2f}")
```

Each step has `.token_id`, `.token_str`, `.next_logits` and `.probes`.

To read the probes on a fixed text instead, run one forward pass:

```python
ids = model.tokenizer.encode("Is this a question?")
out = model.forward([ids])
print(out.probes["question"].logits.shape)
```

`out.probes[name].logits` has the shape `[batch, tokens, outputs]`, and
`out.lm_logits` holds the language model's logits.

## Steer the model

Steering changes the hidden state at the probe's layer while the model generates. It
needs the average activation of each class, which you compute once from labeled data:

```python
from auto_chasm import SteeringConfig

class_means = model.compute_class_means(data)

model.enable_steering(
    "question",
    config=SteeringConfig(method="push_to_mean", scale=4.0),
    class_means=class_means,
)
print(model.generate("Tell me, do you", max_tokens=12))
model.disable_steering("question")
```

The methods, each applied to the newest token:

| Method | What it does |
|---|---|
| `"nullify"` | if the probe score is positive, moves the state until the score is 0 |
| `"push_to_mean"` | if the probe score is positive, moves it to the score of the class-0 average |
| `"boundary"` | moves the state up to the midpoint between the two class averages |
| `"custom"` | uses your own function: `model.enable_steering(name, steer_fn=...)` |

`scale` controls the strength; `scale=0` changes nothing. `"nullify"` and
`"push_to_mean"` leave a token alone when the probe already predicts class 0. See
[`demo/demo_steering.py`](../demo/demo_steering.py).

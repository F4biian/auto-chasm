# auto-chasm

Attach small **probe heads** to a language model and **train them jointly** with
the model (or on their own), then read their predictions **live during
generation** — all through one API that runs on **Apple's MLX** and **PyTorch**
without code changes. Probes can additionally **steer** the model's activations
along what they learned.

```python
from auto_chasm import Model

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
print(model.generate("2 + 2 =", max_tokens=4))   # works before any training
```

A *probe* is a tiny classifier or regressor that reads a hidden layer's activations
and predicts a property of the text — is this a question, does it contain a digit,
what sentiment, how confident. `auto-chasm` makes attaching, training, inspecting,
and steering with such probes a few lines each.

## Contents

[Install](#install) · [Quickstart](#quickstart--the-whole-pipeline) ·
[Data & labels](#data--labels) · [Probes](#probes) ·
[Training](#training) · [Custom losses](#custom-losses) ·
[Generation & inspection](#generation--probe-inspection) · [Steering](#steering) ·
[LoRA, checkpoints & layer sweeps](#lora-checkpoints--layer-sweeps) ·
[Model stats & backends](#model-stats--backends) · [API](#api-at-a-glance)

Each section leads with a runnable example; the details fold into expandable
sections so the page stays scannable. Every example uses a small model that loads on
both backends. The runnable scripts under [`demo/`](demo/) mirror each section.

---

## Install

Install directly from the Git repository, choosing a backend via extras:

```bash
pip install "auto-chasm[mlx]   @ git+https://github.com/F4biian/auto-chasm.git"  # Apple silicon (MLX)
pip install "auto-chasm[torch] @ git+https://github.com/F4biian/auto-chasm.git"  # CUDA / CPU (PyTorch)
```

`import auto_chasm` works with **either** backend installed alone — the same code
runs on both. The backend is auto-detected from what is installed; pass
`backend_name="mlx"` or `"torch"` to force one.

---

## Quickstart — the whole pipeline

Each step is one block and builds on the previous. Run them in order.

**1 — Load a model.** Generation works immediately, no training required.

```python
from auto_chasm import Model

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
```

**2 — Describe labeled data.** `Dataset.from_texts` takes texts and one label each,
placed at a *site* in the text. `label_site="response"` labels the whole text (the
last token's state); `append_eos=True` moves that label onto an appended end token so
the probe reads the *entire* text.

```python
from auto_chasm import Dataset

texts  = ["The sky is blue.", "Is it raining?", "Cats purr.", "Where are you?"]
labels = [0, 1, 0, 1]                      # 1 = question, 0 = statement — the label semantics

data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="is_question",
                          label_site="response", append_eos=True)
```

**3 — Attach a probe.** A `ProbeConfig` names the head and picks which layer(s) it
reads. The default head is a single linear unit (a binary classifier).

```python
from auto_chasm import ProbeConfig

model.attach_probe(ProbeConfig(name="is_question", layers=[-1]))   # -1 = last layer
model.prepare_for_joint_training()                                 # freeze base; train probes (+ LoRA)
```

**4 — Train.** `Trainer` runs the loop; `JointLoss` combines the language-model loss
and each probe's loss by name. Here the language-model term is off (`"lm_head": 0.0`).

```python
from auto_chasm import Trainer, JointLoss

Trainer(
    model=model,
    loss_fn=JointLoss(weights={"lm_head": 0.0, "is_question": 1.0}),
    num_iters=20,
    batch_size=2,
).train(data)
```

**5 — Inspect during generation.** `generate_with_probes` yields one step per token,
each carrying the token and every probe's output at that step.

```python
import math

for step in model.generate_with_probes("The weather today is", max_tokens=5):
    logit = float(step.probes["is_question"].logits[0, -1].item())
    print(f"{step.token_str!r}  p(question)={1 / (1 + math.exp(-logit)):.2f}")
```

**6 — Steer, then save.** Steering nudges activations along the axis the probe learned
(it needs the per-class average activations). `save_checkpoint` bundles everything.

```python
from auto_chasm import SteeringConfig

class_means = model.compute_class_means(data)
model.enable_steering("is_question", config=SteeringConfig(method="nullify"), class_means=class_means)
print(model.generate("The weather today is", max_tokens=6))
model.disable_steering("is_question")

model.save_checkpoint("./runs/is-question")
restored = Model.from_checkpoint("./runs/is-question")   # probes + adapters + steering
```

---

## Data & labels

You give the library **labels placed on characters**, and it maps them to the right
tokens. There are two entry points.

### `Dataset.from_texts` — one label per text

```python
data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="p", label_site="response")
```

`label_site` decides where the single label lands:

| `label_site` | Labels… | Use for |
|---|---|---|
| `"response"` (default) | the last character (whole-text representation) | text-level classification / regression |
| `"token"` | every character from `warmup_chars` onward | per-token properties after a warm-up |
| `"sentence"` | each sentence-ending delimiter | per-sentence properties (needs `sentence_delimiters=[".", "!", "?"]`) |

Labels may be **int class indices** (classification) or **floats** (regression — see
[Custom losses & regression](#custom-losses)).

<details>
<summary>All <code>from_texts</code> parameters</summary>

```python
Dataset.from_texts(
    texts,                    # Sequence[str]
    labels,                   # Sequence[float] — int class indices, or floats for regression
    tokenizer,                # your model's tokenizer
    *,
    label_site="response",    # "response" | "token" | "sentence"
    warmup_chars=50,          # for "token": first labeled character
    sentence_delimiters=None, # for "sentence": e.g. [".", "!", "?"]
    probe_name="probe",       # which head these labels train
    offset=0,                 # shift labels by N tokens
    default_label=None,       # label for unmarked tokens; None masks them (-100)
    append_eos=False,         # append EOS and move a "response" label onto it (recommended)
    groups=None,              # per-text group key -> split(groups="group") keeps a group together
)
```

**`append_eos=True` is strongly recommended for `label_site="response"`.** Without it,
the response label sits on the last content token, whose state the loss never reads
(it is dropped as the final input under next-token alignment). Appending an EOS and
moving the label onto it supervises the state *after* reading the whole text.
</details>

### `build_dataset` / `Dataset.from_conversations` — character spans

For finer control (per-character labels, multiple heads, chat turns) mark **spans**
`{"start", "end", "label"}` per probe, per message:

```python
from auto_chasm.data import build_dataset

conversations = [
    [{"role": "user", "content": "I have 3 cats",
      "labels": {"digit": [{"start": 7, "end": 8, "label": 1}]}}],
]
data = build_dataset(conversations, model.tokenizer, default_label=0)
```

- **Only the spans you mark are labeled.** Everything else is masked (`-100`, not
  trained) — unless you pass `default_label=0`, which makes unmarked tokens the
  negative class inside any message that has at least one span.
- A message with `"labels": {}` is skipped for that probe.
- `aggregation` (`"max"`/`"min"`/`"mean"`) resolves overlapping spans on one token.

<details>
<summary>Split, class balance, and inferring the task from labels</summary>

```python
train, val = data.split(0.15, seed=42)                    # deterministic 85/15 split (two Datasets)
train, val = data.split(0.15, stratify="label")           # keep class balance across the split
train, val = data.split(0.15, groups="group")             # keep a group entirely on one side (no leakage)

weights = data.class_weights(num_classes=3)   # inverse-frequency ("balanced") per-class weights
task    = data.infer_task()                   # Task.binary()/multiclass(k)/regression() from the labels
```

`split` is always at the sample (or group) level, never the token level, so one
text's tokens are never divided across train and val. `class_weights` feeds
[class-imbalance handling](#training); `infer_task` sizes a head and picks its metrics
automatically (used by [`LayerSweep`](#lora-checkpoints--layer-sweeps)).
</details>

### Chat templates & reasoning mode

Conversations are rendered through the tokenizer's **chat template**, so training
sees the same format as inference — role markers, turn delimiters, and any system
turn the template injects:

```python
data = Dataset.from_conversations(conversations=convos, tokenizer=model.tokenizer,
                                  lm_train_on="assistant")
# tokens decode to:
# '<|im_start|>user\nWho invented it?<|im_end|>\n<|im_start|>assistant\nIt was ...<|im_end|>\n'
```

Character spans stay valid: the scaffolding is tokenized separately from the
content and spliced around it, never rendered into one string. LM weights follow
the turn structure — a turn's **opener is masked** (it is prompt, not a target)
while its **closing tag takes the role's weight**, so `lm_train_on="assistant"`
still teaches the model to emit `<|im_end|>` and stop.

Pass `chat_template=False` to concatenate raw message text instead (the pre-0.3
behaviour, needed to reproduce datasets built before this existed).

**Reasoning mode.** Set it once; data prep and generation both honour it:

```python
from auto_chasm import set_default_thinking
set_default_thinking(False)          # closes the <think> block everywhere
```

Do not rely on the template's own default. For Qwen3.5 a plain `AutoTokenizer`
renders a **closed** `<think></think>` (reasoning off) while mlx-lm's
`TokenizerWrapper` leaves it **open** (reasoning on) — from the identical template
string. Per-call `enable_thinking=` overrides the global on both
`Dataset.from_conversations` and `build_dataset`.

If your assistant messages contain no reasoning traces, use `False`: with `True`
the training turns still render an empty closed block while generation opens one,
which is the mismatch this setting exists to prevent.

**Empty span list = all negative.** `labels={"probe": []}` *declares* the probe for
that message and marks nothing positive, so every token takes `default_label`.
Omitting the probe key entirely is what masks the message. The distinction matters:
in a span-annotated corpus the negative-only examples are the clean ones, and
reading `[]` as "unlabeled" drops them all.

### Per-token LM-loss weights — mask & unlearn tokens

The reserved label key **`"lm_head"`** controls how each token trains the
**language-model head** (probe labels are untouched):

| weight | objective at that token | effect |
|---|---|---|
| `1.0` (default) | `-log p` (cross-entropy) | train normally |
| `w > 0` | `w · -log p` | train, scaled |
| `0.0` | — | masked, exactly like a `-100` label |
| `w < 0` | `\|w\| · -log(1 - p)` | **unlikelihood training** — push the token DOWN |

Negative weights run [unlikelihood training (Welleck et al.,
2019)](https://arxiv.org/abs/1908.04319) on that token, not naive gradient
ascent: `-log(1-p)` decays to 0 once the token is unlikely, so the term cannot
run away and the total loss stays bounded below by 0 like any ordinary loss.
`|w|` is precisely the paper's `alpha` mixing coefficient, so `-5.0` means five
times the unlikelihood pressure of `-1.0` — the same objective, scaled. Two ways
to declare it, designed to be used **together**:

```python
# 1) Role-based — the chat-SFT switch. Sets each message's BASELINE weight.
#    "assistant" => assistant tokens 1.0, EVERY other role 0.0 (system too!).
# 2) Explicit per-message specs — applied ON TOP, overriding that baseline.
conversations = [[
    {"role": "system", "content": "You are ..."},                      # baseline 0.0 (masked)
    {"role": "user", "content": "Question ..."},                       # baseline 0.0 (masked)
    {"role": "assistant", "content": "Answer with a hallucination.",   # baseline 1.0 (trained)
     "labels": {
         "halluc":  [{"start": 14, "end": 31, "label": 1}],            # a probe, as usual
         "lm_head": [                                                  # the LM-weight channel
             {"start": 14, "end": 31, "weight": -1.0},                 # char span -> unlearn
             {"text": "hallucination", "weight": 0.0},                 # every occurrence -> mask
             {"regex": r"\d{4}", "weight": 0.0},                       # every match
             {"token_ids": [1234, 567], "weight": -2.0},               # token subsequence, alpha=2
         ],
     }},
]]
data = Dataset.from_conversations(conversations, model.tokenizer,
                                  lm_train_on="assistant")     # or ("assistant", "system")
```

That is the whole API — `Trainer`/`JointLoss` pick the channel up from the data
automatically (no loss configuration). Semantics worth knowing:

- **Default = today's behavior.** Without `lm_train_on`/specs nothing changes:
  every token trains the LM head (there is NO automatic user-token masking).
- ⚠️ **`lm_train_on` masks every role you do not name — `system` included.**
  `"assistant"` is the standard chat-SFT choice (you almost never want the model
  learning to *emit* its own system prompt), but if you do want system tokens
  trained, name them: `lm_train_on=("assistant", "system")`.
- **The two compose: role = baseline, specs = override.** `lm_train_on` sets
  each message's baseline (named role → `1.0`, everything else → `0.0`); specs
  then override the baseline for the tokens they cover, in either direction — a
  `weight: -1.0` span unlearns part of an assistant reply, and a `weight: 1.0`
  span on a user message trains it despite the role baseline. Tokens no spec
  covers keep their role baseline.
- Overlapping **specs** aggregate with **min** — the most aggressive
  intervention wins (`-5 < -1 < 0 < 1`). The role baseline does *not* take part
  in that min, which is what lets a spec override a masked role.
- With an active channel, `labels` is always a per-probe dict + the float
  `"lm_head"` array; attach probes under the same names your spans use.
- A custom `losses={"lm_head": fn}` override + the channel raises (the weights
  would be silently ignored otherwise).
- **Unlearning is bounded, but not free.** Every term is `>= 0` and a token
  already at `p ~ 0` contributes ~nothing, so unlearning cannot run away the
  way negated CE does. `1 - p` is floored at `1e-5` (as in the paper's
  reference implementation), which caps one token's loss at `~11.5` and its
  gradient with it. What unlikelihood does NOT decide for you is what the
  probability mass moves *to*: suppressing a token redistributes it over the
  rest of the vocabulary, so still prefer targeted spans over broad ones and
  watch the LM loss on ordinary tokens.

---

## Probes

A `ProbeConfig` is a small, declarative description of a head. The only required
fields are `name` and `layers`.

```python
from auto_chasm import ProbeConfig

ProbeConfig(name="sentiment", layers=[-1])                                # linear binary head, last layer
ProbeConfig(name="topic", layers=[8], module_config={"out_features": 5})  # 5-class head on layer 8
ProbeConfig(name="score", layers=[-1])                                    # width-1 head -> a regressor
```

### Read several layers at once

Pass multiple `layers`; `aggregation` decides how they combine before the head:

```python
ProbeConfig(name="deep", layers=[6, 9, 12], aggregation="concat")   # concatenate the three layers
ProbeConfig(name="avg",  layers=[6, 9, 12], aggregation="mean")     # average them
```

`"concat"` (default) sizes the head to `len(layers) * hidden`; `"mean"`/`"max"`/`"last"`
keep it at `hidden`; a callable `(list_of_states) -> tensor` does anything else. All
captured layers are used for both training and evaluation.

### Custom heads

`module_type` accepts three levels of control:

```python
from auto_chasm import ModuleSpec

# 1. A built-in name — sized via module_config.
ProbeConfig(name="p", layers=[-1], module_type="mlp", module_config={"hidden_dims": [128, 64]})

# 2. A ModuleSpec — a declarative, BACKEND-AGNOSTIC head (depth, activation, dropout,
#    layer-norm). The library builds the concrete module on whichever backend you run,
#    and it checkpoints/reloads cleanly.
ProbeConfig(name="p", layers=[-1],
            module_type=ModuleSpec.mlp(hidden_dims=[128, 64], activation="gelu", dropout=0.1))

# 3. A callable (in_features, cfg) -> module — return ANY framework module, so you can
#    build absolutely anything the framework supports. This is your escape hatch and is
#    NOT backend-agnostic (you construct a concrete torch/mlx module yourself).
def build_head(in_features, cfg):
    import torch.nn as nn
    return nn.Sequential(nn.LayerNorm(in_features), nn.Linear(in_features, cfg.get("out_features", 1)))

ProbeConfig(name="p", layers=[-1], module_type=build_head)
```

The probe applies its own layer aggregation and granularity pooling *around* the head, so
a head is just an `in_features -> out_features` module. See
[`demo/demo_custom_head.py`](demo/demo_custom_head.py) — a `ModuleSpec` head and a raw
`torch.nn.Module` head trained on the real model (`Model.from_pretrained(..., backend_name="torch")`).

<details>
<summary>Every <code>ProbeConfig</code> field and its options</summary>

| Field | Values | Meaning |
|---|---|---|
| `name` | any string except `"lm_head"` | the head's key everywhere (loss weights, outputs) |
| `layers` | `list[int]` (negatives allowed) | which layer(s) the head reads; `[-1]` is the last |
| `source` | `"hidden"` (default), `"residual"`, `"attention"`, `"mlp"`, `"embedding"`, `"logits"` | which activation to read. `hidden`/`residual`/`attention`/`mlp` are per-layer (may span several `layers`); `embedding`/`logits` read a single site (one layer) |
| `granularity` | `"token"` (default), `"response"`, `"sentence"`, `"custom"` | one prediction per token, per whole text (mean-pooled), per sentence, or a custom pooler |
| `module_type` | `"linear"` (default), `"mlp"`, a `ModuleSpec`, or a callable | the head architecture (see [Custom heads](#custom-heads)) |
| `module_config` | `dict` | head sizing, e.g. `{"out_features": 5}` for 5 classes, or MLP `hidden_dims` |
| `aggregation` | `"concat"` (default), `"mean"`, `"max"`, `"last"`, or a callable | how multiple `layers` combine before the head |
| `pooling` | callable | custom time pooler for `granularity="custom"` |
| `layer_norm` | `bool` | apply a layer-norm to captured activations before the head |

The output width is the head width: `1` (default) is binary/regression; `{"out_features": k}` makes a `k`-class head.
</details>

<details>
<summary>Multiple heads at once</summary>

```python
model.add_probes([
    ProbeConfig(name="is_question", layers=[-1]),                             # binary (BCE)
    ProbeConfig(name="topic", layers=[8], module_config={"out_features": 5}), # 5-class (CE)
    ProbeConfig(name="length", layers=[-1]),                                  # regression (MSE)
])
```

Each head trains on its own labels — pass a `{probe_name: labels}` dict per sample,
or build each probe's data with its own `probe_name` and pair them on the same tokens
(see [`demo/demo_multihead.py`](demo/demo_multihead.py)) — and its own loss (below).
</details>

---

## Training

`Trainer` runs the loop; the **loss function** decides what is optimized. The same
`Trainer` and `JointLoss` run on both backends.

```python
from auto_chasm import Trainer, JointLoss

trainer = Trainer(
    model=model,
    loss_fn=JointLoss(weights={"lm_head": 1.0, "is_question": 2.0}),  # LM loss + 2× the probe loss
    num_iters=200,
    batch_size=4,
    learning_rate=2e-4,
    eval_steps=50,                       # evaluate val every 50 steps
    early_stopping_patience=4,           # opt in: stop after 4 evals without improvement
    restore_best_weights=True,           # opt in: end on the best checkpoint, not the last step
)
result = trainer.train(train, val_data=val, test_data=test)
history = result["history"]              # per-step losses, val metrics, best checkpoint
```

`training_history.json` holds **one row per step** — validation and throughput are
reported from different points in an iteration and are merged into the same entry,
rather than emitting two half-empty rows. Fields that were not measured at a step
are omitted instead of written as `null`, so a real `null` still means "computed,
undefined". When `test_data` is given, its metrics are recorded into the final
row (and the file re-saved) rather than living only in the returned dict.

The loss over `{"lm_head"} ∪ {probe names}` is a weighted sum by default; each term
picks its own loss:

```python
JointLoss(weights={"lm_head": 0.0, "sentiment": 1.0})     # probe-only (LM term off)
JointLoss(losses={"topic": "ce", "length": "mse"})        # per-probe loss choice
```

Built-in loss names: **`"bce"`** (binary, the default), **`"ce"`** (multi-class),
**`"mse"`** and **`"mae"`** (regression). A term with weight `≤ 0` is skipped. For your
own losses see [Custom losses](#custom-losses).

### Class imbalance

Weight the cross-entropy so rare classes count more. Compute inverse-frequency weights
from the data, or let the trainer resolve `"balanced"`:

```python
weights = train.class_weights(num_classes=3)              # e.g. [0.6, 2.5, 1.9]
JointLoss(losses={"topic": "ce"}, class_weights=weights)

JointLoss(losses={"topic": "ce"}, class_weights="balanced")   # Trainer.train computes it from train_data
```

### Early stopping and best-checkpoint restore

**Both are off by default.** A run trains for the full `num_iters` and you keep the
**final-step weights** — which is what a fixed-budget run means, and what you want
whenever the monitored metric is not expected to fall monotonically (unlikelihood /
unlearning runs are the clear case: the val loss there rises by construction).

They are two independent switches:

```python
Trainer(model=model, loss_fn=..., eval_steps=50,
        early_stopping_patience=4,             # 0 (default) = never stop early
        restore_best_weights=True,             # False (default) = keep the final step
        early_stopping_metric="val_loss",      # what to monitor
        min_delta=1e-4)
```

`early_stopping_patience` decides whether training *stops sooner*;
`restore_best_weights` decides *which weights you end up with*. Enabling early
stopping alone leaves you at the stopping point, not at the best step — set both if
you want the classic behaviour.

Best-val tracking runs whenever `val_data` and `eval_steps` are given, regardless of
either flag, so `best_iter` is always reported in `training_manifest.json`
(alongside `restore_best_weights`, recording which weights the manifest describes).
Only the rollback is opt-in. To skip the tracking entirely, pass `eval_steps=0` or
omit `val_data`.

To early-stop on a probe metric (accuracy, F1) instead of the loss, pass an
`eval_metrics_fn` and monitor its key with the right direction:

```python
from auto_chasm.trainers import default_binary_metrics   # -> <probe>_accuracy/_precision/_recall/_f1

Trainer(model=model, loss_fn=..., eval_metrics_fn=default_binary_metrics,
        early_stopping_metric="val_is_question_f1",   # "val_<probe>_<metric>"
        early_stopping_higher_is_better=True)          # F1/accuracy: maximize
```

For **per-layer** early stopping — each layer's head kept at *its own* best step — use
[`LayerSweep`](#lora-checkpoints--layer-sweeps), which snapshots every layer
independently.

### Activation memory & gradient checkpointing

If a run dies with `zsh: killed` (the OS OOM killer) or
`[metal::malloc] Resource limit (499000) exceeded`, the cause is almost always
**activation memory**, not parameter count:

```python
model.enable_gradient_checkpointing()      # before training; call once
```

Only each block's input is kept and the interior is recomputed during backward.
Measured on Qwen3.5-0.8B (MLX, LoRA r=8, batch 1):

| sequence | peak, off | peak, on | saving | step time |
|---|---|---|---|---|
| 253 tok | 11.95 GB | 3.28 GB | **3.6×** | 1.26× slower |
| 628 tok | 29.68 GB | 7.78 GB | **3.8×** | 1.15× slower |
| 1243 tok | 64.62 GB | 21.58 GB | **3.0×** | *faster* — the un-checkpointed run was swapping |

<details>
<summary>Why some models need far more memory than their size suggests</summary>

Peak memory scales with **sequence length × batch size**, and the constant depends
on the architecture. A dense model is modest — Qwen2.5-0.5B measures ~5 MB/token.
A **linear-attention / state-space** model can be an order of magnitude worse:
Qwen3.5-0.8B measures **~48 MB/token**, so a 0.8B model exhausts 64 GB at ~1300
tokens.

The reason is not the loss and not gradient accumulation (measured: unlikelihood
weights cost 0.00 GB extra; going from 1 to 4 accumulation steps costs 0.6 GB).
Those blocks are implemented with a fused kernel that has **no backward**, so the
differentiable path is an unrolled loop over timesteps — in mlx-lm,
`gated_delta_update(..., use_kernel=not self.training)`. Each timestep's
`[B, heads, Dv, Dk]` state is then retained for backward: for Qwen3.5-0.8B that is
1.0 MB per timestep per layer across 18 such layers, i.e. 18 MB/token of state
before counting the other per-step intermediates.

The same unrolled loop multiplies the number of distinct buffers in one graph,
which is what surfaces as `[metal::malloc] Resource limit (499000) exceeded`.
That one is a **count** limit, not memory — free RAM does not prevent it, and
gradient checkpointing does **not** lower it (checkpointing recomputes, so the op
count stays). Only a smaller graph helps: lower `max_seq_length` first (the count
is roughly linear in it), then `batch_size`. The trainer catches this error and
re-raises it with those levers and their current values, rather than letting the
raw `metal::malloc` message through.

The trainer detects such blocks and prints a `[memory]` line naming the cause
before the first step, rather than letting the run die unexplained. The PyTorch
path is not affected the same way: transformers implements these layers with a
chunked algorithm in plain autograd-differentiable ops.

**The same architecture also breaks `mx.compile`.** The MLX training step is
compiled, and `mx.compile` retains one graph per input *shape*; batches are padded
to a 32-token boundary, so a run sees roughly `max_seq_length / 32` shapes. For a
dense model those graphs are small. For an unrolled recurrence each is
`T x n_blocks` nodes, and the retained set crosses Metal's 499000-buffer ceiling a
few hundred iterations in — with tens of GB still free, and only after any short
smoke test has passed. Measured on Qwen3.5-0.8B (18 such blocks,
`max_seq_length=1280`, ~40 shapes): dies at iteration ~170 compiled, completes
uncompiled. The trainer therefore skips compilation on these models and says so;
`compile_step=True` forces it back on, `compile_step=False` off. Uncompiled costs
memory — 46 GB peak versus 20 GB — because compilation fuses intermediates away.

**Other levers**, in order of effect: shorter `max_seq_length` (peak is linear in
it), `batch_size=1` (also linear), then checkpointing. Raising `grad_accum_steps`
does **not** trade memory here and high values can trip the buffer-count limit.
</details>

### Mixed precision

Train the frozen base in half precision while the trainable probe/adapter params and
the optimizer stay `fp32`:

```python
from auto_chasm import TrainingConfig

Trainer(model=model, loss_fn=..., config=TrainingConfig(mixed_precision="bf16"))
```

`"bf16"` runs on **both backends** — it casts the base to bfloat16 and, because bf16
shares fp32's exponent range, needs no loss scaling. `"fp16"` is **torch-only**: it keeps
weights in fp32 and runs the forward under `torch.autocast` + a `GradScaler` (fp16's
narrow range needs loss scaling); it raises on MLX, where bf16 is preferred. See
[`demo/demo_mixed_precision.py`](demo/demo_mixed_precision.py) — a probe trained in bf16
on the real model that generalizes to held-out text.

<details>
<summary>All the training knobs (Trainer)</summary>

```python
Trainer(
    model, loss_fn,
    learning_rate=2e-4, weight_decay=0.0, grad_clip_norm=1.0,
    num_iters=500, batch_size=8, max_seq_length=256, grad_accum_steps=1,
    logging_steps=25, save_steps=100, eval_steps=None,
    early_stopping_patience=0,                       # 0 = disabled (default)
    restore_best_weights=False,                      # False = keep final-step weights (default)
    early_stopping_metric="val_loss",
    early_stopping_higher_is_better=False, min_delta=1e-4,
    keep_best_only=False, save_history=True, output_dir="./checkpoints",
    lr_schedule="cosine", warmup_ratio=0.0,          # "cosine" | "linear" | "constant"
    eval_metrics_fn=None,                            # custom val metrics (probe F1/accuracy/...)
    callbacks=None, verbose=True,
    config=None,                                     # a TrainingConfig; explicit kwargs override it
)
```

`train(train_data, val_data=None, test_data=None)` returns
`{"history", "test_metrics", "output_dir"}`. Every explicit keyword argument wins over
a `config=TrainingConfig(...)`, even when it equals the default.
</details>

---

## Custom losses

You are not limited to the built-in loss names. Write a loss **per probe** and let
`JointLoss` combine it with the others, or write **one loss for everything** from
scratch.

### Per-probe: any callable `(probe, target)`

The `probe` argument is a bound output exposing `.logits`, `.softmax()`,
`.log_softmax()`, and `.reduce()` (which averages over the valid, non-padding
positions for you). Build KL, margin, focal, … — anything — and stay backend-agnostic:

```python
from auto_chasm import ops

def soft_margin(probe, target):                       # a loss the built-ins don't cover
    signed = 2.0 * target - 1.0                        # {0,1} labels -> {-1,+1}
    return probe.reduce(ops.softplus(-signed * probe.logits))

# "q" uses your loss, "s" uses the built-in BCE; JointLoss sums them by weight.
JointLoss(weights={"lm_head": 0.0, "q": 1.0, "s": 1.0}, losses={"q": soft_margin})
```

For a total that is **not** a weighted sum, pass `combine=` — a lambda over the named
terms, which compose with ordinary Python operators (`+ - * / **`):

```python
JointLoss(combine=lambda t: t.lm_head + 2.0 * t.sentiment ** 2)   # weight the squared probe term
```

### Fully custom: your own `loss_fn`

Pass `Trainer` any callable `loss_fn(model, batch, labels, lengths) -> (total, ntoks,
components)`. It runs the model, reads the raw probe outputs, and returns the scalar
total plus a `components` dict for logging:

```python
def my_loss(model, batch, labels, lengths):
    _, probes = model(batch[:, :-1])                   # run the model; read raw head outputs
    logits, target = probes["p"], labels[:, 1:]
    valid = target != -100                             # exclude ignored / padding positions
    per_token = ops.clamp(1.0 - (2.0 * target - 1.0) * logits, lo=0.0) ** 2   # squared hinge
    total = ops.masked_mean(per_token, valid)
    return total, ops.sum(valid), {"squared_hinge": total}

Trainer(model=model, loss_fn=my_loss, num_iters=200, batch_size=4).train(data)
```

`ops` is the backend-agnostic math facade (`ops.exp`, `ops.clamp`, `ops.softmax`,
`ops.masked_mean`, …) so one custom loss runs on MLX and PyTorch.

### Regression

A width-1 head with the `"mse"` (or `"mae"`) loss regresses a continuous target.
`from_texts` keeps float labels as floats:

```python
scores = [0.2, 3.5, 1.1, 2.8]   # continuous targets
data = Dataset.from_texts(texts, scores, model.tokenizer, probe_name="score",
                          label_site="response", append_eos=True)
model.attach_probe(ProbeConfig(name="score", layers=[-1]))          # width-1 = scalar regressor
Trainer(model=model, loss_fn=JointLoss(weights={"lm_head": 0.0, "score": 1.0},
                                       losses={"score": "mse"}), num_iters=100).train(data)
pred = float(model.forward([model.tokenizer.encode("some text")]).probes["score"].logits[0, -1].item())
```

See [`demo/demo_custom_loss.py`](demo/demo_custom_loss.py),
[`demo/demo_fully_custom_loss.py`](demo/demo_fully_custom_loss.py), and
[`demo/demo_regression.py`](demo/demo_regression.py).

---

## Generation & probe inspection

```python
model.generate("Once upon a time", max_tokens=50, temperature=0.7)       # sample
"".join(model.generate_stream("Once upon a time", max_tokens=50))         # stream token pieces
model.chat([{"role": "user", "content": "Hello!"}])                       # chat-templated (instruct models)
```

Stop control and the repetition guard are keyword arguments:

```python
model.generate("List three colors:", max_tokens=100,
               stop_sequences=["\n\n"],   # stop when this text appears
               stop_tokens=[13],          # …or on these token ids
               max_repeat=None)           # None disables the repeat guard; an int caps identical repeats
```

`generate_with_probes` streams a `GenerationStep` per token — `.token_id`,
`.token_str`, `.next_logits`, and `.probes[name]` (each probe's output at that step).

Streaming and custom-stop generation use an incremental **KV cache** (`use_cache=True`,
the default) so decoding is O(n), not O(n²) — see
[`demo/demo_kv_cache.py`](demo/demo_kv_cache.py). It is an optimization only: the output
is bit-identical to full-forward in fp32 (numerically equivalent in bf16). The cache is
disabled automatically while steering is active, since steering re-derives the hidden
states a cache would freeze.

<details>
<summary>All generation parameters</summary>

```python
model.generate(prompt=None, max_tokens=256, temperature=0.0, messages=None,
               max_repeat=256, top_p=..., top_k=..., use_cache=True,
               stop_sequences=..., stop_tokens=...)
```

`temperature=0` is greedy; a negative temperature raises. Pass either `prompt=` (a
string) or `messages=` (chat turns — requires the tokenizer's chat template; base
models have none, so use `prompt=`). A `GenerationConfig` bundles these defaults.
`use_cache=False` forces a full re-forward each step (rarely needed).
</details>

---

## Steering

Steering edits activations at inference along the direction a probe encodes. Compute
the per-class means once, then toggle steering per probe.

```python
from auto_chasm import SteeringConfig

class_means = model.compute_class_means(data)   # {probe: {"mean_0": ..., "mean_1": ...}}
model.enable_steering("is_question",
                      config=SteeringConfig(method="push_to_mean", scale=6.0),
                      class_means=class_means)
print(model.generate("The weather today is", max_tokens=20))
model.disable_steering("is_question")
```

`method` is `"nullify"` (remove the probe direction), `"push_to_mean"` (move toward a
class), `"boundary"` (push across the decision boundary), or `"custom"` (your own
`steer_fn`). `scale` sets the intensity; `scale=0` is a no-op.

---

## LoRA, checkpoints & layer sweeps

**LoRA / PEFT** fine-tunes the base model cheaply alongside the probes.
`prepare_for_joint_training` freezes the base and unfreezes the adapters and heads:

```python
from auto_chasm import LoraConfig

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M", lora=LoraConfig(rank=16, alpha=32))
model.attach_probe(ProbeConfig(name="p", layers=[-1]))
model.prepare_for_joint_training()   # LM loss now trains the LoRA adapters, the probe loss the head
```

`LoraConfig(peft_method=...)` selects `"lora"` (default), `"qlora"` (quantized), or
`"dora"`. `target_modules=None` adapts **every adaptable linear module** — all
attention *and* MLP projections of every layer; only the LM head is excluded
(the "all-linear" convention: with tied embeddings, adapting the head would
double-adapt the input embedding). Pass an explicit list to narrow the scope.
The exact set a model exposes:

```python
model.lora_targetable_modules   # ["model.layers.0.self_attn.q_proj", ...] — the
                                # default target set, stable before/after adapters
model.stats()                   # includes it under "lora_targetable_modules"
```

**Checkpoints.** `save_checkpoint` writes one folder with the probe weights, any
adapters, the steering geometry, and the config; `from_checkpoint` restores all of it:

```python
model.save_checkpoint("./runs/exp")
restored = Model.from_checkpoint("./runs/exp")
```

**Layer sweeps.** To find *which layer* best encodes a property, `LayerSweep` trains
one head per layer in a single frozen-base pass and keeps **each layer's head at its
own best-validation step**:

```python
from auto_chasm import LayerSweep, Task

sweep = LayerSweep(model, task=Task.binary())               # one head per layer, sized from the task
result = sweep.run(train, val, test,
                   loss_fn=JointLoss(weights={"lm_head": 0.0}),
                   num_iters=200, eval_every=50)
print(result.best_layer())      # the best layer by test-adjusted accuracy
result.to_csv("sweep.csv")      # per-layer metrics; result.plot("sweep.png") draws the chart
```

<details>
<summary>LayerSweep options (and per-layer selection metric)</summary>

```python
LayerSweep(model, *, task=None, out_features=None, module_spec=None, layers=None,
           num_classes=None, score_metric="val_loss", higher_is_better=False,
           early_stopping_patience=0, min_delta=0.0, ordinal_tol=1, eval_metrics_fn=None)
```

**Early stopping is PER LAYER.** `early_stopping_patience` counts evals without
improvement for each layer separately, on its own `score_metric`; `min_delta` is the
margin an improvement must clear so noise around a plateau does not reset the
counter. A stopped layer keeps the best snapshot it had, and once *every* layer has
plateaued the run ends — the heads share one forward pass, so no single layer can
stop the pass on its own. Each layer's plateau step is reported as `stopped_at`
(`nan` = still improving at the end). `0` (default) disables it.

Pass a `task=` (which derives head width, classes, and metrics) *or* `out_features=`
explicitly. `layers=None` sweeps every layer. `score_metric` picks each layer's best
snapshot — `"val_loss"` (default), `"val_acc"`, `"val_macro_f1"`, … — paired with
`higher_is_better` (set `True` for accuracy/F1). An unknown metric raises with the
available names listed. `run(train, val, test=None, *, loss_fn, num_iters, eval_every)`. Pass `test_data=None` to SKIP the internal test pass when the reported numbers come from `model.probe_scores` instead — it needs its own pass anyway and adds confidence intervals, so evaluating the same set twice is wasted work.

A binary `task=` also yields `"val_auroc"` — threshold-free, and invariant to the
head's scale and bias. Remaining `run(...)` keywords are forwarded to `Trainer`,
**except** `eval_steps` / `save_steps` / `early_stopping_patience`, which the sweep
owns: it validates on `eval_every` and keeps each layer at its own best step, so a
global early stop would halt every layer when one plateaued. Passing them raises.

Snapshots are held in memory (never written to disk), restored per layer before the
test pass, so `to_csv` reports each layer's own best checkpoint. Columns are derived
from the metrics actually produced, so a custom `eval_metrics_fn` reaches the file.
</details>

---

## Mass-mean probes & hidden states

A mass-mean probe needs no training: the direction is `theta = mean_1 - mean_0`
over token-level hidden states. `fit_mass_mean` computes it in ONE streaming pass
and writes it **into** each linear head, so it becomes an ordinary linear probe
whose weights were computed rather than learned — and every scoring tool works on
it unchanged:

```python
model.add_probes([ProbeConfig(name=f"L{i}", layers=[i], module_config={"out_features": 1})
                  for i in range(model.num_layers)])

means = model.fit_mass_mean(train_data)     # {layer: {"mean_0","mean_1","theta"}}
ps = model.probe_scores(test_data)          # same API as a trained sweep
ps.to_csv("massmean_ci.csv", n_boot=1000, ci=95.0, seed=0)
```

Memory is O(hidden) per probe — sums stream, states are never stored — so corpus
size is irrelevant, and all layers are filled from one pass. AUROC depends only on
`theta/|theta|`, so the scale and the bias set where the threshold sits and never
the ranking; the bias is placed so the midpoint of the two class means scores 0.

**Whitening, opt-in.** When a mass-mean probe sits far below a trained linear
one, the usual cause is that hidden states are strongly anisotropic: `μ₁ − μ₀`
picks up whatever high-variance nuisance direction happens to lie between the
centroids, and a plain projection cannot discount it.

```python
model.fit_mass_mean(train_data, whiten=True)      # scores Sigma^-1/2 (h - mu)
```

Fitting adds two quantities describing the hidden states — the overall mean `μ`
and covariance `Σ` — accumulated in the SAME pass as the class means. Every state
is then centered and whitened before the projection:

```
h_white = Σ^(-1/2) (h − μ)
```

Both are **label-free**: `μ` and `Σ` know nothing about the classes, so the
transform applies to any state, including unlabelled tokens at generation time.
Only the direction uses labels, exactly as the plain probe already does.

This is deliberately not LDA (which whitens by the *within-class* covariance).
For two classes the two give **identical AUROC** — `S_total = S_within +
(n₀n₁/n)·θθᵀ`, a rank-one term along θ, so by Sherman–Morrison `Σ_t⁻¹θ` is a
positive multiple of `Σ_w⁻¹θ`: same ranking, different length. The label-free
version is therefore free, and reaches further. Both facts are pinned by tests.

The transform is stored on the probe and saved with the checkpoint, so it
outlives the process:

```python
model.fit_mass_mean(train_data, whiten=True)
model.save_checkpoint("ck/")

m = Model.from_checkpoint("ck/")
m.probes["L20"].whitening["mean"]        # mu, Sigma^-1/2 and Sigma
m.probes["L20"].whiten(hidden_states)    # apply it to any states yourself
```

Scoring needs no extra work at inference: the transform folds into the head's
weight and bias, so `probe_scores` and the sweeps are unchanged. It costs one
`hidden × hidden` matrix per probe while fitting (26 MB at hidden=2560, so ~950 MB
across 36 layers) and needs far more states than dimensions to estimate well — it
warns when they are scarce, and `shrinkage` ridges `Σ` before the inverse root.
Off by default: the plain probe stays the plain difference of means.

By default the probe is exactly that projection — **no scale, no bias**. The
head's bias is actively zeroed (it arrives randomly initialised, which would offset
every score by an arbitrary constant). Two opt-in knobs exist for when the
*threshold* metrics matter:

```python
model.fit_mass_mean(train_data, calibrate_scale=True, calibrate_bias=True)
```

`calibrate_scale` puts the class means at logits `±2`; `calibrate_bias` puts their
midpoint at 0. **Neither changes AUROC** — it reads only the ranking and is
invariant to any positive rescale or shift:

| | `w == θ` | bias | AUROC | loss |
|---|---|---|---|---|
| default | yes | 0.000 | 0.9286 | 162.39 |
| `calibrate_scale` | no | 0.000 | **0.9286** | 0.57 |
| both | no | −1.886 | **0.9286** | 0.30 |

So they only make `loss` (and via the bias, `acc`/`macro_f1`) interpretable —
uncalibrated, `|θ|` is 38–67 on a 576-dim model, so cross-entropy lands in the
tens. Leave them off unless you need those columns.

**One comparable table across probe types.** `evaluate_probes` scores every probe
on every split and merges them into one row each — same columns whether the head
was gradient-trained or fitted in closed form:

```python
report = model.evaluate_probes({"val": val_data, "test": test_data},
                               n_boot=1000, ci=95.0, seed=SEED)
report.to_csv("probes.csv")
# probe,layer,val_loss,val_acc,val_macro_f1,val_auroc,val_auroc_lo,val_auroc_hi,
#             val_n_tokens,val_n_groups,test_… (same)
```

`n_boot=0` skips the intervals for a quick pass. The per-token scores behind each
row are **kept** on `report.scores[split]`, so plotting or re-bootstrapping needs
no second forward pass:

```python
report.scores["test"].to_csv("ci.csv", stats=...)   # already computed
```

Training-only facts (which
iteration a layer peaked at, where it plateaued) stay in `SweepResult`, because a
closed-form fit has none.

**Hidden states for plotting**, subsampled so a corpus cannot OOM you:

```python
hs = model.hidden_states(train_data, layers=[13], max_tokens=50_000, seed=0)
hs.states[13]        # [N, hidden]      hs.labels / hs.groups   # [N]
hs.class_means(13)   # {"mean_0","mean_1","theta"} from the retained sample
```

Sampling happens DURING the pass, so peak memory is set by `max_tokens`, not by
corpus size (78k tokens x 24 layers x 896 dims would be ~6.7 GB kept whole).
Capture runs through attached probes, so one must exist at each requested layer —
there is no per-probe detach, only `restore_original_layers()`, which would
discard a sweep's trained heads.

---

## Error bars — per-token scores & clustered bootstrap

An eval loop reports an aggregate and discards the `(score, label)` pairs, so
there is nothing left to put a confidence interval around. `probe_scores` runs the
dataset once and keeps them, for **every attached probe from the same forward
pass**:

```python
ps = model.probe_scores(test_data)          # after LayerSweep: each layer's BEST head
ps.aurocs()                                 # {"L0": 0.71, "L1": 0.74, ...} corpus AUROC
ps.bootstrap()                              # {"L0": (point, lo, hi), ...} 95% CI
ps.to_csv("ci.csv")                         # probe,auroc,ci_lo,ci_hi,n_tokens,n_groups
```

**It resamples RESPONSES, not tokens.** Tokens inside one response share a prompt,
a model, and a hallucination span — they are nowhere near independent. Resampling
tokens pretends a corpus of ~1200 responses is ~78k independent observations and
reports intervals several times too narrow (measured on realistic correlated data:
0.026 wide token-level vs 0.070 clustered — a 2.7x understatement). Build the
dataset with `groups=` (the prompt id, say) to cluster a level higher still;
without one, each sample is its own cluster.

Every probe is bootstrapped on the **same** resampled draws, so adjacent layers
share their sampling noise and the curves are comparable — bootstrapping each
independently makes every layer wobble on its own and hides whether a peak is real.

<details>
<summary>All bootstrap options</summary>

```python
ps.bootstrap(
    name=None,              # one probe, or None for all
    n_boot=1000,            # resamples
    ci=95.0,                # central interval width, percent
    seed=0,                 # reproducible resampling
    cluster=True,           # resample GROUPS (correct); False resamples tokens
    method="percentile",    # or "basic" (reverse percentile)
    statistic=None,         # (scores, labels) -> float; None = AUROC
)
```

`method="basic"` reflects the draws through the point estimate,
`[2t - q_hi, 2t - q_lo]`, correcting first-order bias when the draws sit
systematically off it — worth checking when an interval looks lopsided.

`statistic=` bootstraps anything, not just AUROC:

```python
def accuracy(scores, labels):
    return float(((scores > 0) == labels).mean())

ps.bootstrap(statistic=accuracy)
```

`ps.to_csv(path, ...)` takes the same options **spelled out**, so an editor
completes them there too. `bootstrap()` is a pure function, not a setting — calling
it and then calling `to_csv()` with no options writes the DEFAULT interval and
discards the one just computed. Either pass the options to `to_csv`, or hand the
dict back:

```python
stats = ps.bootstrap(n_boot=4200, ci=90.0, seed=SEED)
ps.to_csv("ci.csv", stats=stats)          # writes exactly that run
ps.to_csv("ci.csv", n_boot=4200, ci=90.0, seed=SEED)   # or compute it here
```

Combining `stats=` with an option raises rather than silently ignoring it.
`ps.statistic(name, fn)` gives the corpus value of any `fn` without bootstrapping.

`collect_probe_scores` / `model.probe_scores` take `probe_names=` (a subset),
`batch_size=` and `max_seq_length=`. The result is invariant to `batch_size` —
padding changes, the masked output does not.
</details>

The point estimate is the **corpus** AUROC, not the token-weighted mean of
per-batch AUROCs an eval loop reports. Close, but only the corpus value is the
thing a confidence interval is around.

---

## Model stats & backends

Inspect a model's architecture and parameter counts from the facade:

```python
model.stats()                         # {backend, num_layers, hidden_size, vocab_size,
                                      #  num_attention_heads, intermediate_size,
                                      #  num_parameters, num_trainable_parameters,
                                      #  num_probes, probe_parameters}
model.hidden_size                     # 576
model.num_layers                      # 30
model.num_parameters()                # total (base + probes)
model.num_parameters(trainable=True)  # trainable only (after prepare_for_joint_training)
```

The same code runs on MLX (Apple silicon) and PyTorch (CUDA / CPU); the backend is
auto-detected and can be forced with `backend_name="torch"`. A standard Hugging Face
model (Llama, Qwen, Gemma, … architectures) loads on both — MLX through `mlx-lm`,
PyTorch through `transformers`. `py.typed` ships full type information, so editors
autocomplete the whole API.

---

## API at a glance

| Import | What it is |
|---|---|
| `Model` | the model facade: probes, training prep, generation, steering, checkpoints, `stats()` |
| `Dataset`, `Task` | labeled data (`from_texts`/`from_conversations`, `split`, `class_weights`, `infer_task`) and the inferred task |
| `ProbeConfig`, `ModuleSpec`, `Probe` | describe, size, and hold a probe head |
| `Trainer`, `SFTTrainer`, `TrainingConfig` | training loops and their config |
| `JointLoss`, `ops` | the joint LM+probe loss and the backend-agnostic math facade |
| `LayerSweep`, `SweepResult` | per-layer probing sweeps and their results |
| `GenerationConfig`, `SteeringConfig`, `LoraConfig` | feature configs |
| `classification_metrics`, `regression_metrics` | metric helpers (binary heads also get `_auroc`) |

The runnable, backend-free scripts under [`demo/`](demo/) exercise each of these
end-to-end.

---

## Development

```bash
uv sync --all-extras --group dev    # install with both backends + dev tools
uv run pytest                       # test suite
```

Contributions keep the single-README documentation and the `demo/` scripts in sync
with the code. Licensed under the MIT License.
</content>

# 9. Mass-mean probes and evaluation

This guide shows probes that need no training, and how to report how good a probe is,
with honest error bars.

[All guides](README.md) · Previous: [8. LoRA, checkpoints, layer sweeps](08-lora-checkpoints-sweeps.md) · Next: [10. API reference](10-api-reference.md)

## Setup

```python
from auto_chasm import Dataset, Model, ProbeConfig

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")

texts = [
    "Is it raining?", "The sky is blue.", "Where are you?", "Cats purr.",
    "What time is it?", "I like tea.", "Can birds fly?", "The door is open.",
    "Who is there?", "Dogs bark.", "Why is it cold?", "The tea is hot.",
]
labels = [1, 0] * 6
data = Dataset.from_texts(texts, labels, model.tokenizer, probe_name="question", append_eos=True)
train, test = data.split(0.5, stratify="label", seed=0)
```

## Mass-mean probes

A mass-mean probe needs no training. Its direction is the difference between the two
class averages of the hidden states: `theta = mean_1 - mean_0`. `fit_mass_mean`
computes it in one pass over the data and writes it into the probe's linear head:

```python
model.add_probes([ProbeConfig(name=f"L{i}", layers=[i]) for i in range(model.num_layers)])
model.fit_mass_mean(train)
```

This fits one probe per layer at once. The result is an ordinary linear probe, so
everything below works with it. Memory use does not grow with the size of the dataset.

### Keep the direction fixed

A normal linear head changes its direction when you train it further. To keep the
fitted direction and train only its scale and bias, use `module_type="mass_mean"`:

```python
model.attach_probe(ProbeConfig(name="fixed", layers=[15], module_type="mass_mean"))
model.fit_mass_mean(train, probe_names=["fixed"])
```

Now `"fixed"` can be trained jointly (see [Training](05-training.md)) without losing its
direction, and the direction is saved with checkpoints.

### Whitening

If a mass-mean probe is much worse than a trained probe, the hidden states are probably
stretched in directions that have nothing to do with the classes. Whitening corrects
for that:

```python
model.fit_mass_mean(train, probe_names=["L15"], whiten=True, shrinkage="auto")
```

It uses the overall mean and covariance of the hidden states, computed in the same pass
and without labels. `shrinkage="auto"` picks the regularization of the covariance for
you ([Ledoit-Wolf](https://www.sciencedirect.com/science/article/pii/S0047259X03000964)).

<details>
<summary>More about whitening and calibration</summary>

- Whitening needs many more hidden states than the model has dimensions. It warns
  when there are too few.
- The transform is folded into the head and saved with the checkpoint.
  `model.probes["L15"].whitening` holds it, and `model.probes["L15"].whiten(states)`
  applies it to any states.
- For two classes, this gives the same ranking (and AUROC) as LDA.
- By default the probe has no scale and a bias of 0. AUROC does not care, but loss,
  accuracy and F1 are hard to read then. `calibrate_scale=True` puts the class averages
  at scores `-2` and `+2`, and `calibrate_bias=True` puts their midpoint at 0.
</details>

## AUROC with confidence intervals

`probe_scores` runs the data once and keeps the score and label of every token, for
every probe:

```python
scores = model.probe_scores(test)
print(scores.aurocs())
print(scores.bootstrap(n_boot=200))
scores.to_csv("auroc.csv", n_boot=200)
```

`aurocs()` gives the AUROC per probe. `bootstrap()` gives `(estimate, lower, upper)`
with a 95% confidence interval.

> [!IMPORTANT]
> The bootstrap resamples whole texts, not single tokens. Tokens of one text are not
> independent, and resampling them would make the intervals far too narrow. If several
> texts share a source (for example the same prompt), pass `groups=` when you build the
> dataset to resample at that level.

All probes are resampled with the same draws, so neighboring layers share their noise
and can be compared directly.

<details>
<summary>All bootstrap options</summary>

| Option | Default | Meaning |
|---|---|---|
| `name` | `None` | one probe, or `None` for all |
| `n_boot` | `1000` | number of resamples |
| `ci` | `95.0` | interval width in percent |
| `seed` | `0` | random seed |
| `cluster` | `True` | resample whole texts (or groups); `False` resamples tokens |
| `method` | `"percentile"` | or `"basic"` |
| `statistic` | `None` | any `(scores, labels) -> float`; `None` means AUROC |

`to_csv` takes the same options. `bootstrap()` stores nothing, so pass its result on if
you want exactly that run in the file: `scores.to_csv(path, stats=result)`.
</details>

## One table for all probes

`evaluate_probes` scores every probe on every split and returns one row per probe:

```python
report = model.evaluate_probes({"train": train, "test": test}, n_boot=200)
report.to_csv("probes.csv")
```

The table has the same columns for trained and fitted probes: loss, accuracy, macro F1
and AUROC with its interval, per split. `n_boot=0` skips the intervals.

## Hidden states for plots

```python
states = model.hidden_states(train, layers=[13], max_tokens=5000)
print(states.states[13].shape)
```

`states.labels` holds the matching labels. Tokens are sampled during the pass, so memory
is bounded by `max_tokens`. A probe must be attached at each layer you ask for.

# 4. Probes

A probe is described by a `ProbeConfig`. Only two fields are required: a `name` and
the `layers` it reads.

[All guides](README.md) · Previous: [3. LM token weights](03-lm-token-weights.md) · Next: [5. Training](05-training.md)

## Setup

```python
from auto_chasm import Model, ProbeConfig

model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
```

## Attach a probe

```python
model.attach_probe(ProbeConfig(name="sentiment", layers=[-1]))
```

This adds a linear head with one output on the last layer. One output means binary
classification (with the default `"bce"` loss) or regression (with `"mse"`).

For more than two classes, set the number of outputs:

```python
model.attach_probe(ProbeConfig(name="topic", layers=[8], module_config={"out_features": 5}))
```

## Several probes at once

```python
model.add_probes([
    ProbeConfig(name="is_question", layers=[-1]),
    ProbeConfig(name="language", layers=[10], module_config={"out_features": 3}),
    ProbeConfig(name="length", layers=[-1]),
])
print(list(model.probes))
```

Each probe trains on its own labels and can use its own loss, for example
`JointLoss(losses={"language": "ce", "length": "mse"})`. See
[`demo/demo_multihead.py`](../demo/demo_multihead.py) for a full run.

## Read several layers

```python
model.attach_probe(ProbeConfig(name="deep", layers=[6, 9, 12]))
model.attach_probe(ProbeConfig(name="avg", layers=[6, 9, 12], aggregation="mean"))
```

By default the layers are concatenated, so the head gets `3 x hidden_size` inputs.
`aggregation="mean"` (or `"max"`, `"last"`) combines them into one vector first.

## Read other parts of the model

By default a probe reads the output of a layer. `source` changes that:

```python
last = model.num_layers - 1
model.attach_probe(ProbeConfig(name="mlp_out", layers=[last], source="mlp"))
model.attach_probe(ProbeConfig(name="attn_out", layers=[last], source="attention"))
```

| `source` | Reads |
|---|---|
| `"hidden"` (default) | the output of the layer |
| `"residual"` | the input of the layer |
| `"attention"` | the output of the attention block |
| `"mlp"` | the output of the MLP block |
| `"embedding"` | the token embeddings (one layer only) |
| `"logits"` | the model's output logits (one layer only) |

See [`demo/demo_probe_sources.py`](../demo/demo_probe_sources.py).

## Your own head

The default head is linear. There are three ways to use another one.

**A built-in MLP:**

```python
model.attach_probe(ProbeConfig(
    name="mlp_head", layers=[-1],
    module_type="mlp", module_config={"hidden_dims": [128]},
))
```

**A `ModuleSpec`**, which describes an MLP independently of the backend and is saved
with checkpoints:

```python
from auto_chasm import ModuleSpec

head = ModuleSpec.mlp(hidden_dims=[128, 64], activation="gelu", dropout=0.1)
model.attach_probe(ProbeConfig(name="spec_head", layers=[-1], module_type=head))
```

**Any module you build yourself.** Pass a function that receives the input size and the
`module_config`, and returns a module of your backend (here PyTorch):

```python
import torch.nn as nn

def build_head(in_features, config):
    return nn.Sequential(nn.LayerNorm(in_features), nn.Linear(in_features, 1))

torch_model = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M", backend_name="torch")
torch_model.attach_probe(ProbeConfig(name="my_head", layers=[-1], module_type=build_head))
```

The probe still handles layer selection and pooling, so your module only maps
`in_features` to the number of outputs. See [`demo/demo_custom_head.py`](../demo/demo_custom_head.py).

There is also `module_type="mass_mean"`, a head with a fixed direction, explained in
[9. Mass-mean probes](09-mass-mean-and-evaluation.md#keep-the-direction-fixed).

<details>
<summary>All <code>ProbeConfig</code> fields</summary>

| Field | Default | Meaning |
|---|---|---|
| `name` | required | the probe's name in losses and outputs (not `"lm_head"`) |
| `layers` | required | the layers to read; negative numbers count from the end |
| `source` | `"hidden"` | which activation to read (see the table above) |
| `module_type` | `"linear"` | `"linear"`, `"mlp"`, `"mass_mean"`, a `ModuleSpec`, or a function |
| `module_config` | `{}` | head size, e.g. `{"out_features": 5}` or `{"hidden_dims": [128]}` |
| `aggregation` | `"concat"` | how several layers are combined: `"concat"`, `"mean"`, `"max"`, `"last"`, or a function |
| `granularity` | `"token"` | one prediction per `"token"`, per `"response"`, per `"sentence"`, or `"custom"` |
| `pooling` | `None` | the pooling function for `granularity="custom"` |
| `layer_norm` | `False` | normalize the activations before the head |
</details>

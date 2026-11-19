"""Regression: a ModuleSpec probe head must be checkpointable.

save_checkpoint serialized module_type via isinstance(x, str), so the library's own
declarative ModuleSpec head (callable, not a str) collapsed to the "__callable__"
sentinel and reload raised — a first-class, reconstructable head was un-reloadable.
It is now stored structurally and rebuilt on load.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

from auto_chasm import Model, ProbeConfig
from auto_chasm._checkpoint_weights import load_probe_weights
from auto_chasm.checkpoint import (
    _check_probe_reconstructable,
    _deserialize_module_type,
    _serialize_module_type,
)
from auto_chasm.modules import ModuleSpec


def test_serialize_string_and_lambda() -> None:
    assert _serialize_module_type("linear") == "linear"
    assert _serialize_module_type(lambda in_f, cfg: None) == "__callable__"


def test_module_spec_round_trips() -> None:
    """A ModuleSpec with a string activation serializes structurally and rebuilds."""
    spec = ModuleSpec.mlp(hidden_dims=[16, 8], out_features=3, activation="gelu", dropout=0.1)
    stored = _serialize_module_type(spec)
    assert isinstance(stored, dict) and "__module_spec__" in stored
    back = _deserialize_module_type(stored)
    assert isinstance(back, ModuleSpec)
    assert back.hidden_dims == (16, 8)  # tuple restored, not a list
    assert back.out_features == 3 and back.activation == "gelu" and back.dropout == 0.1
    # A stored spec is NOT rejected by the reconstructable check.
    _check_probe_reconstructable("p", {"module_type": stored})


def test_module_spec_with_callable_activation_stays_callable() -> None:
    """A ModuleSpec whose activation is a callable is not serializable -> sentinel."""
    spec = ModuleSpec.mlp(hidden_dims=[8], out_features=2, activation=lambda x: x)
    assert _serialize_module_type(spec) == "__callable__"


class _TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = [nn.Linear(8, 8) for _ in range(2)]
        self.output_proj = nn.Linear(8, 16)

    def __call__(self, x: mx.array, **k: object) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = 8
    num_hidden_layers = 2


def test_module_spec_probe_weights_round_trip(tmp_path) -> None:  # noqa: ANN001
    """A ModuleSpec probe saves and its weights reload into a rebuilt spec head."""
    spec = ModuleSpec.mlp(hidden_dims=[16], out_features=3, activation="gelu")
    src = Model(_TinyMlp(), None, "mlx")
    src.model.config = _Cfg()
    src.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden", module_type=spec))
    # Perturb the probe so the round-trip is meaningful.
    src._probes["p"].module.update(
        tree_map(lambda a: a + 0.5, src._probes["p"].module.parameters())
    )
    mx.eval(src._probes["p"].module.parameters())
    src.save_checkpoint(str(tmp_path))

    import json

    with open(tmp_path / "manifest.json") as f:
        stored = json.load(f)["probes"]["p"]["module_type"]
    recon = _deserialize_module_type(stored)  # rebuild the head spec from the manifest
    dst = Model(_TinyMlp(), None, "mlx")
    dst.model.config = _Cfg()
    dst.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden", module_type=recon))
    load_probe_weights(dst._probes["p"], tmp_path / "probes" / "p.safetensors", dst.backend)

    a = dict(tree_flatten(src._probes["p"].module.parameters()))
    b = dict(tree_flatten(dst._probes["p"].module.parameters()))
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(np.array(a[k]), np.array(b[k]))

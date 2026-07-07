"""Oracle + parity tests for ModuleSpec, build_module, and the layer_norm fix.

Pins: (1) the backend-agnostic head builder produces the exact forward math
(hand recompute) on MLX and matches torch within tolerance; (2) unknown
activation names raise and callable activations are honored; (3) custom
``(in_features, cfg) -> module`` callables still attach (no architecture cap);
(4) reusing a spec mints distinct per-head params; (5) ``ProbeConfig.layer_norm``
(a previously dead field) now applies an input LayerNorm, with a negative
control and a numpy LayerNorm oracle.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm import Model, ModuleSpec, ProbeConfig
from auto_chasm.metrics import to_numpy


class _TinyMlp(nn.Module):
    def __init__(self, h: int = 16, v: int = 32, layers: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(v, h)
        self.layers = [nn.Linear(h, h) for _ in range(layers)]
        self.output_proj = nn.Linear(h, v)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = 16
    num_hidden_layers = 4


def _attach(**probe_kwargs):
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    return m.attach_probe(ProbeConfig(name="p", layers=[0], aggregation="last", **probe_kwargs))


def test_mlp_forward_oracle_mlx() -> None:
    """A 1-hidden-layer MLP head equals an independent fc0 -> relu -> fc1 recompute."""
    head = ModuleSpec.mlp((8,), out_features=3, activation="relu")(4, {"_backend_name": "mlx"})
    x = mx.random.normal((2, 5, 4))
    out = head(x)
    assert out.shape == (2, 5, 3)
    w0, b0 = head.linears[0].weight, head.linears[0].bias
    w1, b1 = head.linears[1].weight, head.linears[1].bias
    expected = nn.relu(x @ w0.T + b0) @ w1.T + b1
    assert mx.allclose(out, expected, atol=1e-5)


def test_linear_spec_is_single_layer() -> None:
    """hidden_dims=() builds exactly one Linear (in -> out)."""
    head = ModuleSpec.linear(out_features=3)(4, {"_backend_name": "mlx"})
    assert len(head.linears) == 1
    x = mx.random.normal((1, 3, 4))
    expected = x @ head.linears[0].weight.T + head.linears[0].bias
    assert mx.allclose(head(x), expected, atol=1e-5)


def test_out_features_from_cfg_overrides_spec() -> None:
    """module_config out_features overrides the spec's default."""
    head = ModuleSpec.mlp((8,), out_features=3)(4, {"_backend_name": "mlx", "out_features": 7})
    assert head(mx.zeros((1, 2, 4))).shape == (1, 2, 7)


def test_unknown_activation_raises() -> None:
    """An unknown activation name raises ValueError (no silent fallback)."""
    with pytest.raises(ValueError, match="Unknown activation"):
        ModuleSpec.mlp((8,), out_features=3, activation="bogus")(4, {"_backend_name": "mlx"})


def test_callable_activation_is_honored() -> None:
    """A callable activation is used verbatim (here: zero it out)."""
    head = ModuleSpec.mlp((8,), out_features=3, activation=lambda x: x * 0.0)(
        4, {"_backend_name": "mlx"}
    )
    # The hidden activation is forced to 0, so the output is just fc1's bias.
    out = head(mx.random.normal((1, 2, 4)))
    expected = mx.broadcast_to(head.linears[1].bias, out.shape)
    assert mx.allclose(out, expected, atol=1e-5)


def test_reusing_spec_makes_distinct_params() -> None:
    """The same spec, called twice, builds independent heads (not shared weights)."""
    spec = ModuleSpec.mlp((8,), out_features=3)
    a = spec(4, {"_backend_name": "mlx"})
    b = spec(4, {"_backend_name": "mlx"})
    assert a is not b
    assert not mx.array_equal(a.linears[0].weight, b.linears[0].weight)


def test_custom_callable_module_type_still_attaches() -> None:
    """A power-user (in_features, cfg) -> module callable still works end to end."""

    def my_head(in_features, cfg):
        import mlx.nn as _nn

        return _nn.Linear(in_features, cfg.get("out_features", 2))

    p = _attach(module_type=my_head, module_config={"out_features": 4})
    assert p.module(mx.random.normal((1, 3, 16))).shape == (1, 3, 4)


def test_layer_norm_applies_with_numpy_oracle() -> None:
    """layer_norm=True wraps the head; output == inner(numpy LayerNorm(x))."""
    from auto_chasm._mlx_mlp import _NormThenModule

    p = _attach(module_config={"out_features": 3}, layer_norm=True)
    assert isinstance(p.module, _NormThenModule)
    x = mx.random.normal((2, 4, 16))
    out = to_numpy(p.module(x))

    xn = to_numpy(x)
    mean = xn.mean(-1, keepdims=True)
    var = xn.var(-1, keepdims=True)  # ddof=0, matches LayerNorm
    normed = (xn - mean) / np.sqrt(var + 1e-5)  # default affine weight=1, bias=0
    w = to_numpy(p.module.inner.weight)
    b = to_numpy(p.module.inner.bias)
    expected = normed @ w.T + b
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_layer_norm_false_is_unwrapped() -> None:
    """The default (layer_norm=False) leaves the head unwrapped (negative control)."""
    from auto_chasm._mlx_mlp import _NormThenModule

    p = _attach(module_config={"out_features": 3}, layer_norm=False)
    assert not isinstance(p.module, _NormThenModule)
    x = mx.random.normal((1, 3, 16))
    expected = x @ p.module.weight.T + p.module.bias
    assert mx.allclose(p.module(x), expected, atol=1e-5)


def test_layer_norm_params_are_trainable() -> None:
    """The wrapping LayerNorm's weight and bias are in the trainable tree."""
    from mlx.utils import tree_flatten

    p = _attach(module_config={"out_features": 3}, layer_norm=True)
    keys = {k for k, _ in tree_flatten(p.module.trainable_parameters())}
    assert any("norm.weight" in k for k in keys)
    assert any("norm.bias" in k for k in keys)
    assert any("inner.weight" in k for k in keys)


def test_mlx_torch_forward_parity() -> None:
    """Same spec + same weights -> identical forward on MLX and torch."""
    torch = pytest.importorskip("torch")
    spec = ModuleSpec.mlp((6,), out_features=3, activation="gelu", input_layer_norm=True)
    mhead = spec(4, {"_backend_name": "mlx"})
    thead = spec(4, {"_backend_name": "torch"})

    # Copy MLX weights into the torch head (same shapes on both backends).
    thead.norm.weight.data = torch.tensor(to_numpy(mhead.norm.weight))
    thead.norm.bias.data = torch.tensor(to_numpy(mhead.norm.bias))
    for tl, ml in zip(thead.linears, mhead.linears, strict=True):
        tl.weight.data = torch.tensor(to_numpy(ml.weight))
        tl.bias.data = torch.tensor(to_numpy(ml.bias))

    x = np.random.randn(2, 5, 4).astype(np.float32)
    thead.eval()
    mout = to_numpy(mhead(mx.array(x)))
    tout = to_numpy(thead(torch.tensor(x)))
    np.testing.assert_allclose(mout, tout, atol=1e-4)

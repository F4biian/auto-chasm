"""Oracle tests for the sub-block probe sources: attention / mlp / residual.

These pin ``ProbeConfig(source=...)`` (src/auto_chasm/probe.py) for the three
sub-block sources against an INDEPENDENT ground-truth recomputation, not "it
runs".  Each hooks a different point inside a transformer block:

* ``attention`` -> the block's ``self_attn`` submodule output,
* ``mlp``       -> the block's ``mlp`` submodule output (fed the post-attention
  residual, not the block input),
* ``residual``  -> the residual stream *entering* the block (its input).

The model uses an explicit ``_AttnMlpBlock`` so each submodule is individually
hookable and the residual math (``x = x + attn(x); x = x + mlp(x)``) is known,
which lets every capture be checked against a hand recompute.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from auto_chasm import Model, ProbeConfig
from auto_chasm.trainers.trainable import _TrainableModel

_H = 8  # hidden size
_V = 16  # vocab size
_L = 3  # number of transformer blocks


class _AttnMlpBlock(nn.Module):
    """A transformer block with separate self_attn and mlp submodules + residuals."""

    def __init__(self, h: int) -> None:
        super().__init__()
        self.self_attn = nn.Linear(h, h)
        self.mlp = nn.Linear(h, h)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class _TinyTransformer(nn.Module):
    """Embedding -> N attn/mlp blocks -> output projection."""

    def __init__(self, h: int = _H, v: int = _V, layers: int = _L) -> None:
        super().__init__()
        self.embedding = nn.Embedding(v, h)
        self.layers = [_AttnMlpBlock(h) for _ in range(layers)]
        self.output_proj = nn.Linear(h, v)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h)


class _Cfg:
    """Minimal config exposing hidden_size / depth / vocab_size to the probe."""

    hidden_size = _H
    num_hidden_layers = _L
    vocab_size = _V


def _build(source: str, idx: int) -> Model:
    m = Model(_TinyTransformer(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(ProbeConfig(name="p", layers=[idx], source=source, aggregation="last"))
    return m


def _unwrap(mod: object) -> object:
    """A captured module is a LayerCapture wrapping the original under `.layer`."""
    return mod.layer if hasattr(mod, "layer") else mod  # type: ignore[attr-defined]


def _in_features(probe_module: object) -> int:
    return int(probe_module.weight.shape[1])  # type: ignore[attr-defined]


def _block_input(m: Model, x: mx.array, idx: int) -> mx.array:
    """The residual stream entering block ``idx`` (embedding + blocks < idx)."""
    h = _unwrap(m.model.embedding)(x)  # type: ignore[operator]
    for i in range(idx):
        h = _unwrap(m.model.layers[i])(h)  # type: ignore[operator]
    return h


class TestAttentionSource:
    """source='attention' captures the block's self_attn output (dim = hidden)."""

    def test_in_features_is_hidden(self) -> None:
        assert _in_features(_build("attention", 1)._probes["p"].module) == _H

    def test_captured_equals_self_attn_output(self) -> None:
        idx = 1
        m = _build("attention", idx)
        x = mx.array([[1, 2, 3]])
        _TrainableModel(m.model, m._probes)(x)
        captured = m._probes["p"].get_captured_states()[0]

        # Ground truth: self_attn applied to the block's input residual stream.
        h_in = _block_input(m, x, idx)
        attn = _unwrap(m.model.layers[idx].self_attn)
        expected = attn(h_in)  # type: ignore[operator]
        assert captured.shape == (1, 3, _H)
        assert float(mx.max(mx.abs(captured - expected))) == 0.0


class TestMlpSource:
    """source='mlp' captures the block's mlp output, fed the post-attention residual."""

    def test_in_features_is_hidden(self) -> None:
        assert _in_features(_build("mlp", 1)._probes["p"].module) == _H

    def test_captured_equals_mlp_output_of_post_attention_residual(self) -> None:
        idx = 1
        m = _build("mlp", idx)
        x = mx.array([[1, 2, 3]])
        _TrainableModel(m.model, m._probes)(x)
        captured = m._probes["p"].get_captured_states()[0]

        # The mlp sees x + self_attn(x) (self_attn is NOT wrapped for this source).
        h_in = _block_input(m, x, idx)
        mlp_input = h_in + m.model.layers[idx].self_attn(h_in)
        expected = _unwrap(m.model.layers[idx].mlp)(mlp_input)  # type: ignore[operator]
        assert captured.shape == (1, 3, _H)
        assert float(mx.max(mx.abs(captured - expected))) == 0.0


class TestResidualSource:
    """source='residual' captures the residual stream ENTERING the block."""

    def test_in_features_is_hidden(self) -> None:
        assert _in_features(_build("residual", 1)._probes["p"].module) == _H

    def test_captured_equals_block_input(self) -> None:
        idx = 1
        m = _build("residual", idx)
        x = mx.array([[1, 2, 3]])
        _TrainableModel(m.model, m._probes)(x)
        captured = m._probes["p"].get_captured_states()[0]

        expected = _block_input(m, x, idx)  # block input = embedding + blocks < idx
        assert captured.shape == (1, 3, _H)
        assert float(mx.max(mx.abs(captured - expected))) == 0.0

    def test_residual_differs_from_hidden_output(self) -> None:
        # The residual (input) must NOT equal the block's output (the 'hidden'
        # source) — otherwise capture_input did nothing.
        idx = 1
        m = _build("residual", idx)
        x = mx.array([[1, 2, 3]])
        _TrainableModel(m.model, m._probes)(x)
        captured_in = m._probes["p"].get_captured_states()[0]
        block_out = _unwrap(m.model.layers[idx])(captured_in)  # type: ignore[operator]
        assert float(mx.max(mx.abs(captured_in - block_out))) > 1e-4


class TestRestoreUnwrapsSubmodules:
    """restore_original_layers fully removes attention/mlp submodule captures."""

    def test_restore_unwraps_attention_submodule(self) -> None:
        m = _build("attention", 1)
        assert type(m.model.layers[1].self_attn).__name__ == "_MLXLayerCapture"
        m.restore_original_layers()
        # The wrapper is gone; the genuine Linear submodule is back.
        assert type(m.model.layers[1].self_attn).__name__ == "Linear"

    def test_restore_unwraps_mlp_submodule(self) -> None:
        m = _build("mlp", 0)
        assert type(m.model.layers[0].mlp).__name__ == "_MLXLayerCapture"
        m.restore_original_layers()
        assert type(m.model.layers[0].mlp).__name__ == "Linear"


class TestSharedLayerCoexistence:
    """A hidden probe and an attention probe at the SAME layer both capture."""

    def test_hidden_then_attention_same_layer(self) -> None:
        # Attaching 'hidden' wraps the block; attaching 'attention' at the same
        # layer must unwrap to the real block and wrap its submodule — both fire.
        m = Model(_TinyTransformer(), None, "mlx")
        m.model.config = _Cfg()
        m.attach_probe(ProbeConfig(name="h", layers=[1], source="hidden", aggregation="last"))
        m.attach_probe(ProbeConfig(name="a", layers=[1], source="attention", aggregation="last"))
        x = mx.array([[1, 2, 3]])
        _TrainableModel(m.model, m._probes)(x)

        cap_h = m._probes["h"].get_captured_states()[0]
        cap_a = m._probes["a"].get_captured_states()[0]
        # hidden = block output; attention = self_attn output of the block input.
        h_in = _block_input(m, x, 1)
        attn = _unwrap(_unwrap(m.model.layers[1]).self_attn)
        assert float(mx.max(mx.abs(cap_a - attn(h_in)))) == 0.0  # type: ignore[operator]
        block_out = _unwrap(m.model.layers[1])(h_in)  # type: ignore[operator]
        assert float(mx.max(mx.abs(cap_h - block_out))) == 0.0
        # The two heads captured different tensors.
        assert float(mx.max(mx.abs(cap_h - cap_a))) > 1e-4


class TestSubBlockSourcesDiffer:
    """attention, mlp, and residual read three genuinely different tensors."""

    def test_three_sources_distinct(self) -> None:
        x = mx.array([[2, 5, 7]])
        caps = []
        for src in ("attention", "mlp", "residual"):
            m = _build(src, 1)
            _TrainableModel(m.model, m._probes)(x)
            caps.append(m._probes["p"].get_captured_states()[0])
        a, mlp_c, r = caps
        assert float(mx.max(mx.abs(a - mlp_c))) > 1e-4
        assert float(mx.max(mx.abs(a - r))) > 1e-4
        assert float(mx.max(mx.abs(mlp_c - r))) > 1e-4

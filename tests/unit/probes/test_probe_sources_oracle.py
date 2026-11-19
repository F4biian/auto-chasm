"""Oracle tests for the three implemented probe sources: hidden / embedding / logits.

These pin ``ProbeConfig(source=...)`` (src/auto_chasm/probe.py) against an
INDEPENDENT ground-truth recomputation, not merely "it runs".  Each source is
supposed to read a specific submodule of the model:

* ``hidden``    -> the wrapped transformer block's output  (dim = hidden_size)
* ``embedding`` -> the token embedding lookup ``embedding(x)`` (dim = hidden_size)
* ``logits``    -> the LM-head / output projection output  (dim = vocab_size)

For each, we capture the probe's input activation by running a real forward via
``_TrainableModel`` and assert it equals a hand-computed forward of the exact
submodule that source hooks.  We also assert the probe head's ``in_features``
equals the dimension that source is required to feed it.

``_TinyMlp`` puts the activation *inside* a ``_Block`` module (the thing that
gets wrapped), so wrapping ``layers[idx]`` captures the genuine post-activation
block output -- mirroring how the real probe hooks a transformer block.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from auto_chasm import Model, ProbeConfig
from auto_chasm.trainers.trainable import _TrainableModel

_H = 8  # hidden size
_V = 16  # vocab size
_L = 3  # number of transformer blocks


class _Block(nn.Module):
    """A transformer block stand-in: linear + activation, captured as one unit."""

    def __init__(self, h: int) -> None:
        super().__init__()
        self.linear = nn.Linear(h, h)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.linear(x))


class _TinyMlp(nn.Module):
    """Embedding -> N blocks -> output projection; submodules are individually hookable."""

    def __init__(self, h: int = _H, v: int = _V, layers: int = _L) -> None:
        super().__init__()
        self.embedding = nn.Embedding(v, h)
        self.layers = [_Block(h) for _ in range(layers)]
        self.output_proj = nn.Linear(h, v)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h)


class _Cfg:
    """Minimal config exposing hidden_size / depth / vocab_size to the probe machinery."""

    hidden_size = _H
    num_hidden_layers = _L
    vocab_size = _V


def _build(source: str, layers: list[int]) -> tuple[Model, ProbeConfig]:
    cfg = ProbeConfig(name="p", layers=layers, source=source, aggregation="last")
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(cfg)
    return m, cfg


def _unwrap(layer: object) -> object:
    # A captured layer is an _MLXLayerCapture wrapping the original under `.layer`.
    return layer.layer if hasattr(layer, "layer") else layer


def _in_features(probe_module: object) -> int:
    # MLX nn.Linear weight is (out_features, in_features).
    return int(probe_module.weight.shape[1])


class TestHiddenSource:
    """source='hidden' captures the wrapped block's output; head in-dim is hidden_size."""

    def test_in_features_equals_hidden_size(self) -> None:
        m, _ = _build("hidden", layers=[1])
        assert _in_features(m._probes["p"].module) == _H

    def test_captured_state_equals_independent_block_forward(self) -> None:
        idx = 1
        m, _ = _build("hidden", layers=[idx])
        tm = _TrainableModel(m.model, m._probes)
        x = mx.array([[1, 2, 3]])
        tm(x)

        captured = m._probes["p"].get_captured_states()[0]

        # Independent ground truth: embed, then run the genuine (unwrapped) block
        # modules up to and including `idx`. This is what the wrapped layer emits.
        blocks = [_unwrap(m.model.layers[i]) for i in range(_L)]
        h = m.model.embedding(x)
        for i in range(idx + 1):
            h = blocks[i](h)

        assert captured.shape == (1, 3, _H)
        assert float(mx.max(mx.abs(captured - h))) == 0.0

    def test_captured_state_is_post_activation_not_pre(self) -> None:
        # Guard: the capture is the FULL block output (post-gelu), not the raw
        # linear pre-activation. A regression that hooked the wrong tensor would
        # match `pre` instead.
        idx = 0
        m, _ = _build("hidden", layers=[idx])
        tm = _TrainableModel(m.model, m._probes)
        x = mx.array([[1, 2, 3]])
        tm(x)
        captured = m._probes["p"].get_captured_states()[0]

        block = _unwrap(m.model.layers[idx])
        h_in = m.model.embedding(x)
        pre = block.linear(h_in)
        post = nn.gelu(pre)
        assert float(mx.max(mx.abs(captured - post))) == 0.0
        assert float(mx.max(mx.abs(captured - pre))) > 1e-3


class TestEmbeddingSource:
    """source='embedding' captures embedding(x); head in-dim is hidden_size."""

    def test_in_features_equals_hidden_size(self) -> None:
        m, _ = _build("embedding", layers=[0])
        assert _in_features(m._probes["p"].module) == _H

    def test_captured_state_equals_embedding_lookup(self) -> None:
        m, _ = _build("embedding", layers=[0])
        tm = _TrainableModel(m.model, m._probes)
        x = mx.array([[1, 2, 3, 4]])
        tm(x)

        captured = m._probes["p"].get_captured_states()[0]

        # Ground truth: the raw embedding lookup, before any block runs. The
        # embedding module is now a capture wrapper, so unwrap it.
        embed = _unwrap(m.model.embedding)
        expected = embed(x)
        assert captured.shape == (1, 4, _H)
        assert float(mx.max(mx.abs(captured - expected))) == 0.0


class TestLogitsSource:
    """source='logits' captures the LM-head output; head in-dim is vocab_size."""

    def test_in_features_equals_vocab_size(self) -> None:
        m, _ = _build("logits", layers=[0])
        assert _in_features(m._probes["p"].module) == _V

    def test_captured_state_equals_model_output_logits(self) -> None:
        m, _ = _build("logits", layers=[0])
        tm = _TrainableModel(m.model, m._probes)
        x = mx.array([[1, 2, 3]])
        lm_logits, _probes = tm(x)

        captured = m._probes["p"].get_captured_states()[0]

        # Ground truth: the model's own output logits. The logits source hooks
        # the output projection, so its captured tensor IS the LM logits.
        assert captured.shape == (1, 3, _V)
        assert float(mx.max(mx.abs(captured - lm_logits))) == 0.0


class TestSourcesProduceDistinctInputs:
    """The three sources read genuinely different tensors (no aliasing/mixup)."""

    def test_hidden_embedding_logits_differ(self) -> None:
        x = mx.array([[2, 5, 7]])

        m_h, _ = _build("hidden", layers=[1])
        _TrainableModel(m_h.model, m_h._probes)(x)
        cap_h = m_h._probes["p"].get_captured_states()[0]

        m_e, _ = _build("embedding", layers=[0])
        _TrainableModel(m_e.model, m_e._probes)(x)
        cap_e = m_e._probes["p"].get_captured_states()[0]

        m_l, _ = _build("logits", layers=[0])
        _TrainableModel(m_l.model, m_l._probes)(x)
        cap_l = m_l._probes["p"].get_captured_states()[0]

        # hidden vs embedding share the hidden dim but must hold different values.
        assert cap_h.shape == cap_e.shape == (1, 3, _H)
        assert float(mx.max(mx.abs(cap_h - cap_e))) > 1e-4
        # logits lives in vocab space.
        assert cap_l.shape == (1, 3, _V)

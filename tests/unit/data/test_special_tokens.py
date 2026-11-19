"""Tests for Model.add_special_tokens (tokenizer + embedding stay in sync)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from auto_chasm.model import Model


class _MockTok:
    def __init__(self) -> None:
        self.vocab = 32
        self.extra: list[str] = []

    def add_tokens(self, toks, special_tokens=False):
        new = [t for t in toks if t not in self.extra]
        self.extra += new
        self.vocab += len(new)
        return len(new)

    def __len__(self) -> int:
        return self.vocab


class _TinyMlp(nn.Module):
    def __init__(self, h: int = 16, v: int = 32, layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(v, h)
        self.layers = [nn.Linear(h, h) for _ in range(layers)]
        self.output_proj = nn.Linear(h, v)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


def _model() -> Model:
    m = Model(_TinyMlp(), _MockTok(), "mlx")

    class Cfg:
        """Minimal model config."""

        hidden_size = 16
        num_hidden_layers = 2

    m.model.config = Cfg()
    return m


def test_add_grows_embedding_and_head_and_accepts_new_ids():
    m = _model()
    assert m.model.embedding.weight.shape[0] == 32
    added = m.add_special_tokens(["<probe>", "<end>"])
    assert added == 2
    assert m.model.embedding.weight.shape[0] == 34
    assert m.model.output_proj.weight.shape[0] == 34
    # A forward pass using the brand-new token ids must not index out of range.
    out = m.model(mx.array([[32, 33, 1]]))
    assert out.shape == (1, 3, 34)


def test_adding_existing_token_is_noop():
    m = _model()
    m.add_special_tokens(["<x>"])
    assert m.add_special_tokens(["<x>"]) == 0


def test_quantized_mlx_embedding_grows_losslessly():
    """Adding tokens to a quantized MLX embedding works and keeps existing rows exact."""

    class QMlp(nn.Module):
        """Tiny model with a quantized embedding (dim divisible by group size)."""

        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.QuantizedEmbedding(32, 64, group_size=64, bits=4)

        def __call__(self, x: mx.array) -> mx.array:
            return self.embed_tokens(x)

    m = Model(QMlp(), _MockTok(), "mlx")
    qe = m.model.embed_tokens
    old_w, old_s, old_b = qe.weight, qe.scales, qe.biases

    added = m.add_special_tokens(["<x>", "<y>"])
    assert added == 2
    assert qe.num_embeddings == 34
    # Existing rows are BYTE-IDENTICAL — the original tokens lose no precision;
    # only the two new rows were quantized and appended.
    assert bool(mx.array_equal(qe.weight[:32], old_w))
    assert bool(mx.array_equal(qe.scales[:32], old_s))
    assert bool(mx.array_equal(qe.biases[:32], old_b))
    # The brand-new ids look up without indexing out of range.
    out = qe(mx.array([[32, 33, 1]]))
    assert out.shape == (1, 3, 64)


def test_quantized_untied_output_head_also_grows():
    """An untied QuantizedLinear head grows with the (quantized) embedding."""

    class QHeadModel(nn.Module):
        """Quantized embedding + an untied quantized output head."""

        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.QuantizedEmbedding(32, 64, group_size=64, bits=4)
            self.lm_head = nn.QuantizedLinear(64, 32, group_size=64, bits=4)

        def __call__(self, x: mx.array) -> mx.array:
            return self.lm_head(self.embed_tokens(x))

    m = Model(QHeadModel(), _MockTok(), "mlx")
    m.add_special_tokens(["<x>", "<y>"])
    assert m.model.embed_tokens.num_embeddings == 34
    # The output head now produces logits over the grown vocab (34).
    out = m.model(mx.array([[0, 32, 33]]))
    assert out.shape == (1, 3, 34)


def test_add_special_tokens_syncs_config_vocab_size_for_logits_probe():
    """add_special_tokens updates config.vocab_size so a logits probe sizes correctly.

    Regression: the MLX resize grew the tables but left config.vocab_size stale, so a
    source="logits" probe (sized from config) died with a cryptic matmul shape error.
    """
    from auto_chasm import ProbeConfig

    m = Model(_TinyMlp(), _MockTok(), "mlx")

    class Cfg:
        """Config that declares a vocab size."""

        hidden_size = 16
        num_hidden_layers = 2
        vocab_size = 32

    m.model.config = Cfg()
    m.add_special_tokens(["<probe>"])
    assert m.model.config.vocab_size == 33  # synced with the grown embedding/head

    # A logits probe attached AFTER the resize sizes its in-dim from the new vocab
    # and forwards on the new id without an out-of-range / matmul crash.
    m.attach_probe(ProbeConfig(name="lp", layers=[0], source="logits"))
    outputs = m.forward([[32, 1, 2]])
    assert "lp" in outputs.probes

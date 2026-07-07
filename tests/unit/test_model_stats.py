"""Tests for the Model stats accessors (hidden_size, params, stats())."""

from __future__ import annotations

import mlx.nn as nn

from auto_chasm import Model, ProbeConfig


class _TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.layers = [nn.Linear(16, 16) for _ in range(4)]
        self.output_proj = nn.Linear(16, 32)

    def __call__(self, x, **k):  # noqa: ANN001, ANN003, ANN204
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = 16
    num_hidden_layers = 4
    vocab_size = 32
    num_attention_heads = 4
    intermediate_size = 64


def _model() -> Model:
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    return m


def test_facade_stat_accessors() -> None:
    """hidden_size / vocab_size / num_layers / num_parameters are exposed and correct."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[-1]))
    assert m.hidden_size == 16
    assert m.vocab_size == 32
    assert m.num_layers == 4
    assert m.num_parameters() > 0
    # A frozen base + trainable probe: trainable count is smaller than the total.
    m.prepare_for_joint_training()
    assert 0 < m.num_parameters(trainable=True) < m.num_parameters()


def test_stats_dict_has_all_fields() -> None:
    """stats() returns architecture + parameter fields, including per-probe params."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[-1]))
    s = m.stats()
    for key in (
        "backend",
        "num_layers",
        "hidden_size",
        "vocab_size",
        "num_attention_heads",
        "intermediate_size",
        "num_parameters",
        "num_trainable_parameters",
        "num_probes",
        "probe_parameters",
    ):
        assert key in s
    assert s["num_attention_heads"] == 4 and s["intermediate_size"] == 64
    assert s["num_probes"] == 1 and "p" in s["probe_parameters"]


def test_vocab_size_falls_back_to_embedding() -> None:
    """vocab_size uses the embedding's row count when the config omits a vocab field."""

    class _CfgNoVocab:
        hidden_size = 16
        num_hidden_layers = 4

    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _CfgNoVocab()
    assert m.vocab_size == 32  # from the embedding (32 rows), not the config

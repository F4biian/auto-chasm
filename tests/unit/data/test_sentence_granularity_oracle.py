"""Oracle tests for granularity='sentence' (per-sentence pooling, broadcast back).

Pins ``_probe_agg.sentence_pool`` (and the probe wiring) against a hand-computed
per-sentence mean. A sentence ends at (and includes) a delimiter token; each
token's output becomes the mean of its sentence's logits over the valid mask,
broadcast back — so the ``[B, T, out]`` shape and per-token labels still apply.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import Model, ProbeConfig
from auto_chasm._probe_agg import sentence_pool
from auto_chasm.trainers.trainable import _TrainableModel

# logits 1,2,3 | 10,20 with delimiter id 9 ending the first sentence at index 2.
_LOGITS = mx.array([[[1.0], [2.0], [3.0], [10.0], [20.0]]])  # [1, 5, 1]
_IDS = mx.array([[5, 5, 9, 5, 5]])  # delimiter=9 at index 2 → seg = [0,0,0,1,1]


class TestSentencePoolMath:
    """sentence_pool returns the per-sentence mean broadcast to each token."""

    def test_mean_broadcast_no_mask(self) -> None:
        out = sentence_pool(_LOGITS, _IDS, [9], None, "mlx")
        # sent 0 = mean(1,2,3)=2 over idx 0..2; sent 1 = mean(10,20)=15 over idx 3..4.
        assert out.tolist() == [[[2.0], [2.0], [2.0], [15.0], [15.0]]]

    def test_mask_excludes_position_from_mean(self) -> None:
        mask = mx.array([[True, True, False, True, True]])  # drop the delimiter token
        out = sentence_pool(_LOGITS, _IDS, [9], mask, "mlx")
        # sent 0 = mean(1,2)=1.5 (idx 2 excluded); still broadcast to idx 0,1,2.
        assert out.tolist() == [[[1.5], [1.5], [1.5], [15.0], [15.0]]]

    def test_single_sentence_equals_response_mean(self) -> None:
        # No delimiter present → one sentence → every token gets the global mean.
        ids = mx.array([[5, 5, 5, 5, 5]])
        out = sentence_pool(_LOGITS, ids, [9], None, "mlx")
        vals = [v[0] for v in out[0].tolist()]
        assert vals == pytest.approx([7.2] * 5)  # mean(1,2,3,10,20)=7.2

    def test_mlx_torch_parity(self) -> None:
        pytest.importorskip("torch")
        import torch

        t_logits = torch.tensor([[[1.0], [2.0], [3.0], [10.0], [20.0]]])
        t_ids = torch.tensor([[5, 5, 9, 5, 5]])
        t_out = sentence_pool(t_logits, t_ids, [9], None, "torch")
        m_out = sentence_pool(_LOGITS, _IDS, [9], None, "mlx")
        assert t_out.flatten().tolist() == pytest.approx(
            [v[0] for row in m_out.tolist() for v in row]
        )


class _TinyMlp(nn.Module):
    """Embedding -> linear blocks -> output projection for probe wiring tests."""

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


class _Cfg:
    """Minimal model config for the probe machinery."""

    hidden_size = 16
    num_hidden_layers = 2
    vocab_size = 32


class TestSentenceProbeWiring:
    """The probe routes granularity='sentence' through sentence_pool end to end."""

    def test_probe_forward_pools_per_sentence(self) -> None:
        m = Model(_TinyMlp(), None, "mlx")
        m.model.config = _Cfg()
        m.attach_probe(
            ProbeConfig(
                name="s",
                layers=[1],
                granularity="sentence",
                module_config={"sentence_delimiters": [9]},
            )
        )
        ids = mx.array([[5, 5, 9, 5, 5]])
        _TrainableModel(m.model, m._probes)(ids)
        probe = m._probes["s"]
        out = probe.forward(input_ids=ids)
        # Output keeps the per-token shape, and is constant within each sentence.
        assert out.shape == (1, 5, 1)
        vals = [v[0] for v in out[0].tolist()]
        assert vals[0] == vals[1] == vals[2]  # sentence 0 constant
        assert vals[3] == vals[4]  # sentence 1 constant
        assert vals[0] != vals[3]  # different sentences differ

    def test_sentence_without_input_ids_raises(self) -> None:
        m = Model(_TinyMlp(), None, "mlx")
        m.model.config = _Cfg()
        m.attach_probe(
            ProbeConfig(
                name="s",
                layers=[1],
                granularity="sentence",
                module_config={"sentence_delimiters": [9]},
            )
        )
        _TrainableModel(m.model, m._probes)(mx.array([[5, 5, 9, 5, 5]]))
        with pytest.raises(ValueError, match="input_ids"):
            m._probes["s"].forward()  # no input_ids passed

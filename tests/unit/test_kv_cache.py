"""KV-cache correctness for the manual / streaming generation loops (both backends).

The KV cache must be an optimization only: greedy generation WITH a cache must be
byte-for-byte identical to full-forward WITHOUT one. These tests use tiny *real*
transformers (with attention, so the cache actually matters), prove the decoders
feed incrementally, prove they fall back transparently for cacheless models, and
prove the Model facade disables caching while steering is active (which would
otherwise diverge — a cache freezes the past hidden states steering re-derives).
"""

from __future__ import annotations

import mlx.core as mx

from auto_chasm._gen_cache import MlxDecoder, TorchDecoder
from auto_chasm.generation import (
    _generate_manual_mlx,
    _generate_manual_torch,
    _generate_stream_mlx,
    _generate_stream_torch,
)


class _Tok:
    """Trivial tokenizer: a fixed prompt, id->char decode, an out-of-range eos."""

    eos_token_id = 999

    def __init__(self, prompt: list[int] | None = None) -> None:
        self._prompt = prompt or [5, 6, 7, 8]

    def encode(self, text: str) -> list[int]:
        return list(self._prompt)

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


def _tiny_mlx_llama():  # noqa: ANN202
    from mlx_lm.models.llama import Model, ModelArgs

    args = ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=100,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
    )
    return Model(args)


def _tiny_torch_llama():  # noqa: ANN202
    import pytest

    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(cfg).eval()


# --- Correctness: cache == full-forward (greedy) ---------------------------------


def test_mlx_cache_matches_full_forward() -> None:
    """MLX manual + stream: cache output is byte-identical to full-forward."""
    m, tok = _tiny_mlx_llama(), _Tok()
    man_c = _generate_manual_mlx(m, tok, "x", 24, 0.0, use_cache=True)
    man_nc = _generate_manual_mlx(m, tok, "x", 24, 0.0, use_cache=False)
    assert man_c == man_nc and len(man_c) == 24
    str_c = "".join(_generate_stream_mlx(m, tok, "x", 24, 0.0, use_cache=True))
    str_nc = "".join(_generate_stream_mlx(m, tok, "x", 24, 0.0, use_cache=False))
    assert str_c == str_nc == man_c  # manual and stream agree too


def test_torch_cache_matches_full_forward() -> None:
    """Torch manual + stream: cache output is byte-identical to full-forward."""
    m, tok = _tiny_torch_llama(), _Tok()
    man_c = _generate_manual_torch(m, tok, "x", 24, 0.0, use_cache=True)
    man_nc = _generate_manual_torch(m, tok, "x", 24, 0.0, use_cache=False)
    assert man_c == man_nc and len(man_c) == 24
    str_c = "".join(_generate_stream_torch(m, tok, "x", 24, 0.0, use_cache=True))
    str_nc = "".join(_generate_stream_torch(m, tok, "x", 24, 0.0, use_cache=False))
    assert str_c == str_nc == man_c


def test_cache_matches_with_stop_sequence() -> None:
    """A stop sequence truncates identically with and without the cache."""
    m, tok = _tiny_mlx_llama(), _Tok()
    # Pick a stop string that actually occurs in the (deterministic) output.
    full = _generate_manual_mlx(m, tok, "x", 40, 0.0, use_cache=False)
    stop = full[3:6]
    c = _generate_manual_mlx(m, tok, "x", 40, 0.0, stop_sequences=[stop], use_cache=True)
    nc = _generate_manual_mlx(m, tok, "x", 40, 0.0, stop_sequences=[stop], use_cache=False)
    assert c == nc
    assert stop not in c  # stopped before the sequence


def test_max_tokens_one_with_cache() -> None:
    """The single-token edge case works with the cache (prime + one read)."""
    m, tok = _tiny_torch_llama(), _Tok()
    assert _generate_manual_torch(m, tok, "x", 1, 0.0, use_cache=True) == _generate_manual_torch(
        m, tok, "x", 1, 0.0, use_cache=False
    )


# --- The decoders really feed incrementally --------------------------------------


def test_mlx_decoder_feeds_only_new_tokens() -> None:
    """After priming the prompt, the MLX decoder feeds one token per step (O(n))."""
    dec = MlxDecoder(_tiny_mlx_llama(), use_cache=True)
    assert dec.cache is not None
    dec.next_logits([1, 2, 3, 4])
    assert dec._fed == 4  # whole prompt consumed once
    dec.next_logits([1, 2, 3, 4, 5])
    assert dec._fed == 5  # only the new token fed


def test_torch_decoder_feeds_only_new_tokens() -> None:
    """The torch decoder threads past_key_values and feeds one token per step."""
    m = _tiny_torch_llama()
    dec = TorchDecoder(m, use_cache=True)
    dec.next_logits([1, 2, 3, 4], "cpu")
    assert dec._fed == 4 and dec.past is not None
    dec.next_logits([1, 2, 3, 4, 5], "cpu")
    assert dec._fed == 5


# --- Fallback for cacheless models -----------------------------------------------


def test_mlx_fallback_for_plain_callable() -> None:
    """A plain callable (no mlx_lm cache) falls back to full-forward, unchanged output."""
    import numpy as np

    base = np.full(100, -10.0, dtype=np.float32)
    base[42] = 10.0

    def model(x: mx.array) -> mx.array:
        return mx.broadcast_to(mx.array(base), (1, x.shape[1], 100))

    dec = MlxDecoder(model, use_cache=True)
    assert dec.cache is None  # make_prompt_cache could not build one
    tok = _Tok()
    c = _generate_manual_mlx(model, tok, "x", 5, 0.0, use_cache=True)
    nc = _generate_manual_mlx(model, tok, "x", 5, 0.0, use_cache=False)
    assert c == nc


def test_torch_fallback_for_cacheless_model() -> None:
    """A torch model that rejects the cache kwargs falls back and matches full-forward."""
    import torch

    base = torch.full((100,), -10.0)
    base[42] = 10.0

    class _NoCache:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return base.expand(1, x.shape[1], 100)

    dec = TorchDecoder(_NoCache(), use_cache=True)
    dec.next_logits([1, 2, 3], "cpu")
    assert dec._ok is False  # detected the model cannot cache
    tok = _Tok()
    c = _generate_manual_torch(_NoCache(), tok, "x", 5, 0.0, use_cache=True)
    nc = _generate_manual_torch(_NoCache(), tok, "x", 5, 0.0, use_cache=False)
    assert c == nc


# --- The Model facade disables caching while steering is active -------------------


def test_model_disables_cache_under_steering() -> None:
    """Model.generate defaults use_cache off while a steering hook is enabled."""
    import mlx.nn as nn

    from auto_chasm import Model, ProbeConfig, SteeringConfig

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

    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden"))

    # No steering: caching is left to the generation default (not forced off).
    _, _, kw = m._apply_gen_config(None, 10, 0.0, {})
    assert "use_cache" not in kw

    m.enable_steering("p", config=SteeringConfig(method="custom"), steer_fn=lambda h, hd, lg: h)
    _, _, kw2 = m._apply_gen_config(None, 10, 0.0, {})
    assert kw2["use_cache"] is False  # steering forces the cache off

    # Steering correctness is NOT user-overridable: an explicit use_cache=True is
    # forced back to False (a cache would freeze the hidden states steering re-derives).
    _, _, kw3 = m._apply_gen_config(None, 10, 0.0, {"use_cache": True})
    assert kw3["use_cache"] is False

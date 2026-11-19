"""Generation regressions: sampling params, stop hold-back, num_return, space.

GEN-1 top_p/top_k/repetition_penalty were silently ignored in the manual/stream
loops. GEN-2 streaming leaked partial stop-sequence prefixes before the stop fired.
GEN-3 torch manual/stream did not strip the leading sub-word space MLX strips.
GEN-4 num_return_sequences>1 was silently dropped (one sequence returned anyway).
"""

from __future__ import annotations

import numpy as np
import pytest

from auto_chasm._generation_utils import filter_logits, held_back_len, stream_flush

# --- GEN-1: filter_logits applies repetition penalty / top-k / top-p --------------


def test_filter_logits_repetition_penalty_can_flip_argmax() -> None:
    """A seen token's logit is penalised, flipping the greedy choice."""
    logits = np.array([2.0, 1.9, -1.0])
    out = filter_logits(logits, 0.0, None, None, 2.0, [0])  # penalise token 0
    assert out[0] == pytest.approx(1.0)  # 2.0 / 2.0
    assert int(np.argmax(out)) == 1  # token 1 now wins (was token 0)


def test_filter_logits_top_k_keeps_exactly_k() -> None:
    """top_k leaves exactly k finite logits; the rest are -inf."""
    logits = np.array([3.0, 2.0, 1.0, 0.0])
    out = filter_logits(logits, 1.0, 2, None, None, [])
    finite = np.isfinite(out)
    assert int(finite.sum()) == 2
    assert finite[0] and finite[1] and not finite[2] and not finite[3]


def test_filter_logits_top_p_nucleus() -> None:
    """top_p keeps the smallest set whose cumulative prob crosses the threshold."""
    logits = np.array([2.0, 1.0, 0.0, -10.0])  # softmax ~ [.705, .259, .095, ~0]
    out = filter_logits(logits, 1.0, None, 0.9, None, [])
    finite = np.isfinite(out)
    assert finite[0] and finite[1]  # top two cross 0.9
    assert not finite[2] and not finite[3]


def test_filter_logits_greedy_ignores_top_k_top_p() -> None:
    """Greedy (temperature 0) does not apply top-k/top-p — only the penalty."""
    logits = np.array([1.0, 2.0, 3.0])
    out = filter_logits(logits, 0.0, 1, 0.1, None, [])
    assert np.all(np.isfinite(out))  # nothing masked to -inf


# --- GEN-2: stream_flush never leaks a partial stop sequence ----------------------


def test_stream_flush_holds_partial_stop_prefix() -> None:
    """A trailing prefix of a stop sequence is held back, not emitted."""
    assert stream_flush("ab", ["STOP"]) == ("ab", "", False)
    assert stream_flush("abS", ["STOP"]) == ("ab", "S", False)
    assert stream_flush("abST", ["STOP"]) == ("ab", "ST", False)
    assert stream_flush("abSTOP", ["STOP"]) == ("ab", "", True)  # full match: stop
    assert stream_flush("abc", None) == ("abc", "", False)  # no stops -> emit all


def test_held_back_len_longest_prefix() -> None:
    """held_back_len returns the longest suffix that could still begin a stop."""
    assert held_back_len("xxST", ["STOP"]) == 2
    assert held_back_len("xxSTO", ["STOP", "STONE"]) == 3
    assert held_back_len("xxq", ["STOP"]) == 0


# --- GEN-4: num_return_sequences > 1 is rejected, not silently dropped -------------


def test_reject_num_return_sequences_over_one() -> None:
    """generate() raises for num_return_sequences > 1 rather than dropping extras."""
    from auto_chasm.generation import generate

    with pytest.raises(ValueError, match="num_return_sequences"):
        generate(object(), object(), "hi", num_return_sequences=2)


def test_num_return_sequences_one_is_allowed() -> None:
    """num_return_sequences == 1 (or unset) does not raise."""
    from auto_chasm._generation_utils import reject_num_return_sequences

    reject_num_return_sequences({"num_return_sequences": 1})
    reject_num_return_sequences({})  # unset


# --- GEN-1 / GEN-2 / GEN-3 end-to-end through the real MLX loops -------------------


class _CharTok:
    """Fake tokenizer: fixed 1-token prompt, id->char decode, distinct eos."""

    eos_token_id = 99

    def __init__(self, mapping: dict[int, str], prompt_ids: list[int]) -> None:
        self.mapping = mapping
        self.prompt_ids = prompt_ids

    def encode(self, text: str) -> list[int]:
        return list(self.prompt_ids)

    def decode(self, ids: list[int]) -> str:
        return "".join(self.mapping.get(int(i), "") for i in ids)


def _fixed_logits_model(peaks: dict[int, float], vocab: int):  # noqa: ANN202
    """Return a callable MLX model whose next-token logits are fixed per step.

    ``peaks`` maps token id -> logit; a per-call counter is not needed because the
    same fixed distribution is returned every step (used for penalty/argmax tests).
    """
    import mlx.core as mx

    base = np.full(vocab, -10.0, dtype=np.float32)
    for tid, val in peaks.items():
        base[tid] = val

    def model(x: mx.array) -> mx.array:
        seq = x.shape[1]
        return mx.broadcast_to(mx.array(base), (1, seq, vocab))

    return model


def _scripted_model(script: list[int], vocab: int):  # noqa: ANN202
    """Return a stateful MLX model that peaks token ``script[i]`` on call ``i``."""
    import mlx.core as mx

    state = {"i": 0}

    def model(x: mx.array) -> mx.array:
        peak = script[min(state["i"], len(script) - 1)]
        state["i"] += 1
        seq = x.shape[1]
        base = np.full(vocab, -10.0, dtype=np.float32)
        base[peak] = 10.0
        return mx.broadcast_to(mx.array(base), (1, seq, vocab))

    return model


def test_gen1_repetition_penalty_threaded_into_manual_loop() -> None:
    """repetition_penalty reaches the manual MLX loop and changes the greedy pick."""
    from auto_chasm.generation import _generate_manual_mlx

    tok = _CharTok({1: "A", 2: "B"}, prompt_ids=[1])  # token 1 (A) is in the context
    model = _fixed_logits_model({1: 10.0, 2: 9.0}, vocab=100)
    assert _generate_manual_mlx(model, tok, "x", 1, 0.0) == "A"  # greedy picks A
    penalised = _generate_manual_mlx(model, tok, "x", 1, 0.0, repetition_penalty=2.0)
    assert penalised == "B"  # A penalised (10/2 < 9) -> B (was silently ignored)


def test_gen2_stream_does_not_leak_partial_stop() -> None:
    """Streaming holds back 'S','ST','STO' and stops at 'STOP' without leaking them."""
    from auto_chasm.generation import _generate_stream_mlx

    mapping = {1: "a", 2: "b", 3: "S", 4: "T", 5: "O", 6: "P", 7: "x"}
    tok = _CharTok(mapping, prompt_ids=[0])
    model = _scripted_model([1, 2, 3, 4, 5, 6, 7], vocab=100)
    pieces = list(_generate_stream_mlx(model, tok, "x", 10, 0.0, stop_sequences=["STOP"]))
    assert "".join(pieces) == "ab"  # nothing after the stop, no partial leak
    assert all(ch not in "".join(pieces) for ch in "STOP")


def test_gen3_mlx_manual_strips_leading_subword_space() -> None:
    """The manual MLX loop strips the one leading sub-word space (parity with mlx_lm)."""
    from auto_chasm.generation import _generate_manual_mlx

    tok = _CharTok({1: " A"}, prompt_ids=[0])  # first sub-word decodes with a space
    model = _fixed_logits_model({1: 10.0}, vocab=100)
    assert _generate_manual_mlx(model, tok, "x", 1, 0.0) == "A"  # leading space stripped

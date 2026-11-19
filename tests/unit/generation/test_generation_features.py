"""Tests for generation features: stop tokens, repetition detection."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def _make_model(predict_token: int = 5, vocab_size: int = 16) -> Any:
    """Create a tiny model that always predicts a specific token."""

    class FixedModel:
        """Returns logits with the target token as argmax."""

        def __init__(self) -> None:
            self._vocab_size = max(vocab_size, predict_token + 1)
            self._predict_token = predict_token

        def __call__(self, x: mx.array) -> tuple:
            logits = mx.full((x.shape[0], x.shape[1], self._vocab_size), -100.0)
            logits[:, -1, self._predict_token] = mx.array(100.0)
            return (logits,)

    return FixedModel()


class FixedTokenizer:
    """Tokenizer that returns fixed token sequences."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids if i > 0)


class VocabTokenizer(FixedTokenizer):
    """Tokenizer with vocab for auto-detection tests."""

    def get_vocab(self) -> dict[str, int]:
        return {"<end_of_turn>": 99, "hello": 1, "world": 2}


class TestExtractStopTokens:
    """Tests for _extract_stop_tokens helper."""

    def test_empty_kwargs_returns_none(self) -> None:
        """No stop_tokens/stop_sequences in kwargs returns None."""
        from auto_chasm.generation import _extract_stop_tokens

        result = _extract_stop_tokens(FixedTokenizer(), {})
        assert result is None

    def test_explicit_stop_tokens(self) -> None:
        """Explicit stop_tokens list is returned."""
        from auto_chasm.generation import _extract_stop_tokens

        result = _extract_stop_tokens(FixedTokenizer(), {"stop_tokens": [42, 99]})
        assert result == [42, 99]

    def test_stop_sequences_left_for_substring_matching(self) -> None:
        """stop_sequences is NOT reduced to token ids here.

        A multi-token string is matched as a substring of the decoded output by
        the generation loops, so ``_extract_stop_tokens`` neither consumes nor
        tokenizes ``stop_sequences`` -- it only handles ``stop_tokens``.
        """
        from auto_chasm.generation import _extract_stop_tokens

        kwargs = {"stop_sequences": ["hello"]}
        result = _extract_stop_tokens(FixedTokenizer(), kwargs)
        assert result is None  # no explicit stop_tokens; no vocab to auto-detect
        assert kwargs == {"stop_sequences": ["hello"]}  # left untouched

    def test_auto_detect_from_vocab(self) -> None:
        """Auto-detects common turn-end tokens from tokenizer vocab."""
        from auto_chasm.generation import _extract_stop_tokens

        result = _extract_stop_tokens(VocabTokenizer(), {})
        assert result is not None
        assert 99 in result  # <end_of_turn> detected

    def test_auto_detect_excludes_eos(self) -> None:
        """Auto-detect should not include the EOS token."""
        from auto_chasm.generation import _extract_stop_tokens

        tok = VocabTokenizer()
        result = _extract_stop_tokens(tok, {})
        assert result is not None
        assert tok.eos_token_id not in result

    def test_no_vocab_returns_none(self) -> None:
        """Tokenizer without get_vocab returns None."""
        from auto_chasm.generation import _extract_stop_tokens

        class NoVocabTokenizer(FixedTokenizer):
            """Tokenizer that raises on get_vocab access."""

            def get_vocab(self) -> dict[str, int]:
                raise AttributeError

        result = _extract_stop_tokens(NoVocabTokenizer(), {})
        assert result is None


class TestRepetitionDetection:
    """Tests for 50-token repetition guard in manual generation."""

    def test_repetition_stops_early(self) -> None:
        """Generation with max_tokens=200 stops early due to repetition."""
        from auto_chasm.generation import _generate_manual_mlx

        model = _make_model(predict_token=42)
        result = _generate_manual_mlx(model, FixedTokenizer(), "test", 200, 0.0)
        generated = len(FixedTokenizer().encode(result))
        assert generated <= 60, f"Expected ~50 tokens, got {generated}"

    def test_49_repeats_does_not_stop(self) -> None:
        """49 consecutive same tokens should NOT trigger repetition guard."""
        from auto_chasm.generation import _generate_manual_mlx

        call_count = [0]

        vocab = 100

        class CountingModel:
            """Counts calls, always predicts token 42."""

            def __call__(self, x: mx.array) -> tuple:
                call_count[0] += 1
                logits = mx.full((x.shape[0], x.shape[1], vocab), -100.0)
                logits[:, -1, 42] = mx.array(100.0)
                return (logits,)

        _generate_manual_mlx(CountingModel(), FixedTokenizer(), "test", 49, 0.0)
        assert call_count[0] >= 47, f"Should generate ~49 tokens, got {call_count[0]}"

    def test_repetition_guard_is_configurable(self) -> None:
        """The guard no longer silently truncates at 50; it is configurable."""
        from auto_chasm.generation import _generate_manual_mlx

        call_count = [0]
        vocab = 100

        class CountingModel:
            """Counts calls, always predicts token 42."""

            def __call__(self, x: mx.array) -> tuple:
                call_count[0] += 1
                logits = mx.full((x.shape[0], x.shape[1], vocab), -100.0)
                logits[:, -1, 42] = mx.array(100.0)
                return (logits,)

        # The default does NOT cut a legitimate 60-token run off at 50.
        _generate_manual_mlx(CountingModel(), FixedTokenizer(), "test", 60, 0.0)
        assert call_count[0] > 56, "default guard must not truncate legitimate repeats at 50"

        # An explicit max_repeat caps the run near that value.
        call_count[0] = 0
        _generate_manual_mlx(CountingModel(), FixedTokenizer(), "test", 200, 0.0, max_repeat=10)
        assert call_count[0] <= 25, f"max_repeat=10 should stop near 10, got {call_count[0]}"


class TestStopTokenGuard:
    """Tests that stop_tokens halt generation."""

    def test_stop_token_halts_generation(self) -> None:
        """Generation stops immediately when stop token is predicted."""
        from auto_chasm.generation import _generate_manual_mlx

        model = _make_model(predict_token=99)
        result = _generate_manual_mlx(model, FixedTokenizer(), "test", 100, 0.0, stop_tokens=[99])
        # Should stop at first token (stop token 99)
        assert len(result) < 20, f"Expected short output, got {len(result)} chars"


class _SeqModel:
    """Emits a fixed token sequence (one id per step, greedy argmax)."""

    def __init__(self, seq: list[int], prompt_len: int = 3, vocab: int = 64) -> None:
        self._seq = seq
        self._prompt_len = prompt_len
        self._vocab = vocab

    def __call__(self, x: mx.array) -> tuple:
        step = int(x.shape[1]) - self._prompt_len
        tok = self._seq[min(step, len(self._seq) - 1)]
        logits = mx.full((x.shape[0], x.shape[1], self._vocab), -100.0)
        logits[:, -1, tok] = mx.array(100.0)
        return (logits,)


class _LetterTok(FixedTokenizer):
    """Decodes 10..13 to A..D so streamed text has known substrings."""

    _map = {10: "A", 11: "B", 12: "C", 13: "D"}

    def decode(self, ids: list[int]) -> str:
        return "".join(self._map.get(i, "") for i in ids if i > 0)


class TestStopControlOnPrimaryPath:
    """M6: explicit stop control is honoured on the primary generate() entry.

    Regression: the fast mlx_lm / HF paths silently dropped ``stop_tokens`` and
    ``stop_sequences`` -- only the manual fallback honoured them.
    """

    def test_generate_honours_stop_tokens(self) -> None:
        """generate() stops on a stop-token id (routed to the manual loop)."""
        from auto_chasm.generation import generate

        model = _make_model(predict_token=99)  # always predicts token 99
        out = generate(
            model, FixedTokenizer(), "test", max_tokens=100, temperature=0.0, stop_tokens=[99]
        )
        assert out == ""  # the very first token is a stop token -> no continuation

    def test_generate_truncates_at_stop_sequence(self) -> None:
        """generate() cuts the output before a stop_sequences substring."""
        from auto_chasm.generation import generate

        model = _SeqModel([10, 11, 12, 13])  # emits A, B, C, D
        out = generate(
            model, _LetterTok(), "test", max_tokens=10, temperature=0.0, stop_sequences=["C"]
        )
        assert out == "AB"  # stopped before "C"

    def test_stream_truncates_at_stop_sequence(self) -> None:
        """Streaming stops before a stop_sequences substring and never leaks it."""
        from auto_chasm.generation import _generate_stream_mlx

        pieces = list(
            _generate_stream_mlx(
                _SeqModel([10, 11, 12, 13]),
                _LetterTok(),
                "test",
                max_tokens=10,
                temperature=0.0,
                stop_sequences=["C"],
            )
        )
        assert "".join(pieces) == "AB"
        assert "C" not in "".join(pieces)


class TestTorchStopControlWiring:
    """M6 (torch): stop control reaches HF generate() natively, not dropped."""

    def test_stop_tokens_folded_into_eos_and_stop_strings_set(self) -> None:
        """stop_tokens extend eos_token_id; stop_sequences become HF stop_strings."""
        import pytest

        torch = pytest.importorskip("torch")

        from auto_chasm.generation import _generate_torch

        captured: dict = {}

        class _RecModel:
            """Records the kwargs HF generate() is called with, then emits EOS."""

            device = "cpu"

            def generate(self, input_ids, **kw):  # noqa: ANN001, ANN003
                captured.update(kw)
                return torch.cat([input_ids, torch.tensor([[0]])], dim=1)

        class _HFTok(FixedTokenizer):
            """Callable HF-style tokenizer returning an input_ids tensor."""

            def __call__(self, text: str, return_tensors: str = "pt") -> dict:
                return {"input_ids": torch.tensor([self.encode(text)])}

            def decode(self, ids, skip_special_tokens: bool = False) -> str:  # noqa: ANN001
                return ""

        _generate_torch(
            _RecModel(), _HFTok(), "hi", 5, 0.0, stop_tokens=[7, 8], stop_sequences=["END"]
        )
        assert captured["eos_token_id"] == [0, 7, 8]  # base eos 0 + the two stop tokens
        assert captured["stop_strings"] == ["END"]

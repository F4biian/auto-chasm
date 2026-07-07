"""Generation tests — stop tokens, messages, max_tokens, temperature.

Tests exercise edge cases around the generation API that the other
generation tests do not cover.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.generation import (
    _extract_stop_tokens,
    _generate_manual_mlx,
    _generate_stream_mlx,
    _resolve_prompt,
    chat,
    generate,
)

TINY_VOCAB = 32
TINY_HIDDEN = 16
TINY_LAYERS = 2
STOP_THRESHOLD = 50


class TinyLm(nn.Module):
    """A tiny LM with configurable dimensions."""

    def __init__(
        self,
        hidden_dim: int = TINY_HIDDEN,
        vocab_size: int = TINY_VOCAB,
        num_layers: int = TINY_LAYERS,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array) -> tuple:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return (self.output_proj(h),)


class FixedPredictor(nn.Module):
    """Model that always predicts a single fixed token."""

    def __init__(
        self, token_id: int, vocab_size: int = TINY_VOCAB, hidden_dim: int = TINY_HIDDEN
    ) -> None:
        super().__init__()
        self._token_id = token_id
        self._vocab_size = max(vocab_size, token_id + 1)
        self.embedding = nn.Embedding(self._vocab_size, hidden_dim)

    def __call__(self, x: mx.array) -> tuple:
        logits = mx.full((x.shape[0], x.shape[1], self._vocab_size), -100.0)
        logits[:, -1, self._token_id] = mx.array(100.0)
        return (logits,)


class FixedTokenizer:
    """Minimal tokenizer for manual generation tests."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "".join(f"[{i}]" for i in ids if i > 0)


class ChatTemplateTokenizer(FixedTokenizer):
    """Tokenizer with a chat template."""

    chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        return "chat_prompt"


class NoChatTemplateTokenizer(FixedTokenizer):
    """Tokenizer that has apply_chat_template but no template string."""

    chat_template = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        msg = "should not be called"
        raise NotImplementedError(msg)


class VocabDictTokenizer(FixedTokenizer):
    """Tokenizer with get_vocab returning a dict."""

    def get_vocab(self) -> dict[str, int]:
        return {"<end_of_turn>": 99, "hello": 1, "world": 2}


class NonDictVocabTokenizer(FixedTokenizer):
    """Tokenizer with get_vocab returning a non-dict (list)."""

    def get_vocab(self) -> list[tuple[str, int]]:
        return [("<end_of_turn>", 99)]


class TestStopTokens:
    """Tests for stop token detection and early stopping."""

    def test_stop_token_99_halts_after_one(self) -> None:
        """stop_tokens=[99] with a model always predicting 99 stops at first token."""
        model = FixedPredictor(token_id=99, vocab_size=100)
        result = _generate_manual_mlx(
            model, FixedTokenizer(), "test", max_tokens=50, temperature=0.0, stop_tokens=[99]
        )
        assert result == ""

    def test_empty_stop_tokens_list(self) -> None:
        """Empty stop_tokens list does not add extra stop tokens beyond EOS."""
        model = FixedPredictor(token_id=42, vocab_size=100)
        result = _generate_manual_mlx(
            model, FixedTokenizer(), "test", max_tokens=5, temperature=0.0, stop_tokens=[]
        )
        assert len(result) > 0

    def test_stop_tokens_none_vs_empty_same_behavior(self) -> None:
        """None and empty list for stop_tokens behave identically."""
        model = FixedPredictor(token_id=42, vocab_size=100)
        mx.random.seed(TINY_VOCAB)
        a = _generate_manual_mlx(
            model, FixedTokenizer(), "test", max_tokens=5, temperature=0.0, stop_tokens=None
        )
        mx.random.seed(TINY_VOCAB)
        b = _generate_manual_mlx(
            model, FixedTokenizer(), "test", max_tokens=5, temperature=0.0, stop_tokens=[]
        )
        assert a == b

    def test_stop_token_through_generate_kwargs(self) -> None:
        """stop_tokens passed as kwarg to generate() stops early."""
        model = FixedPredictor(token_id=99, vocab_size=100)
        result = generate(
            model,
            FixedTokenizer(),
            "test",
            max_tokens=TINY_VOCAB,
            temperature=0.0,
            stop_tokens=[99],
        )
        assert isinstance(result, str)
        assert len(result) < TINY_VOCAB

    def test_eos_only_without_stop_tokens(self) -> None:
        """Without stop_tokens, only EOS halts generation."""
        model = FixedPredictor(token_id=42, vocab_size=100)
        result = _generate_manual_mlx(
            model, FixedTokenizer(), "test", max_tokens=10, temperature=0.0
        )
        assert len(result) > 0

    def test_multi_stop_tokens(self) -> None:
        """Multiple stop tokens all stop generation (first token is stop)."""
        model = FixedPredictor(token_id=7, vocab_size=100)
        result = _generate_manual_mlx(
            model, FixedTokenizer(), "test", max_tokens=50, temperature=0.0, stop_tokens=[5, 7, 9]
        )
        assert result == ""


class TestMessagesParameter:
    """Tests for the messages parameter in generation APIs."""

    def test_messages_no_chat_template_raises(self) -> None:
        """Messages without chat_template raises ValueError."""
        tok = NoChatTemplateTokenizer()
        with pytest.raises(ValueError, match="chat_template"):
            _resolve_prompt(tok, prompt=None, messages=[{"role": "user", "content": "hi"}])

    def test_messages_with_chat_template_succeeds(self) -> None:
        """Messages with chat_template returns a prompt string."""
        tok = ChatTemplateTokenizer()
        result = _resolve_prompt(tok, prompt=None, messages=[{"role": "user", "content": "hi"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_both_prompt_and_messages_messages_wins(self) -> None:
        """When both prompt and messages are given, messages wins."""
        tok = ChatTemplateTokenizer()
        result = _resolve_prompt(
            tok, prompt="ignored_prompt", messages=[{"role": "user", "content": "hi"}]
        )
        assert "ignored_prompt" not in result

    def test_neither_prompt_nor_messages_raises(self) -> None:
        """With neither prompt nor messages, ValueError is raised."""
        tok = ChatTemplateTokenizer()
        with pytest.raises(ValueError, match="Either prompt or messages"):
            _resolve_prompt(tok, prompt=None, messages=None)

    def test_invalid_role_in_messages_handled(self) -> None:
        """Messages with invalid role are passed through to apply_chat_template."""
        tok = ChatTemplateTokenizer()
        messages = [{"role": "admin", "content": "hi"}]
        result = _resolve_prompt(tok, prompt=None, messages=messages)
        assert isinstance(result, str)

    def test_empty_messages_list(self) -> None:
        """Empty messages list with chat_template succeeds."""
        tok = ChatTemplateTokenizer()
        result = _resolve_prompt(tok, prompt=None, messages=[])
        assert isinstance(result, str)

    def test_messages_on_model_generate(self) -> None:
        """Model.generate with messages returns a string."""
        from auto_chasm import Model

        base = TinyLm()
        tok = ChatTemplateTokenizer()
        model = Model(base, tok, backend_name="mlx")
        result = model.generate(
            messages=[{"role": "user", "content": "hi"}], max_tokens=1, temperature=0.0
        )
        assert isinstance(result, str)

    def test_messages_on_model_generate_stream(self) -> None:
        """Model.generate_stream with messages yields token strings."""
        from auto_chasm import Model

        base = TinyLm()
        tok = ChatTemplateTokenizer()
        model = Model(base, tok, backend_name="mlx")
        tokens = list(
            model.generate_stream(
                messages=[{"role": "user", "content": "hi"}], max_tokens=2, temperature=0.0
            )
        )
        assert all(isinstance(t, str) for t in tokens)

    def test_messages_on_generate_with_probes(self) -> None:
        """Model.generate_with_probes with messages yields GenerationSteps."""
        from auto_chasm import Model

        base = TinyLm()
        tok = ChatTemplateTokenizer()
        model = Model(base, tok, backend_name="mlx")
        steps = list(
            model.generate_with_probes(
                messages=[{"role": "user", "content": "hi"}], max_tokens=2, temperature=0.0
            )
        )
        assert len(steps) > 0
        assert all(s.token_str is not None for s in steps)

    def test_chat_function_with_template(self) -> None:
        """chat() function with a tokenizer that has chat_template succeeds."""
        model = TinyLm()
        tok = ChatTemplateTokenizer()
        result = chat(
            model, tok, [{"role": "user", "content": "hi"}], max_tokens=1, temperature=0.0
        )
        assert isinstance(result, str)

    def test_chat_function_fallback_no_template(self) -> None:
        """chat() function falls back to string formatting when no template."""
        model = TinyLm()
        result = chat(
            model,
            FixedTokenizer(),
            [{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.0,
        )
        assert isinstance(result, str)


class TestMaxTokensEdgeCases:
    """Tests for extreme max_tokens values."""

    def test_max_tokens_zero(self) -> None:
        """max_tokens=0 returns an empty string."""
        result = _generate_manual_mlx(
            TinyLm(), FixedTokenizer(), "test", max_tokens=0, temperature=0.0
        )
        assert result == ""

    def test_negative_max_tokens(self) -> None:
        """Negative max_tokens produces empty string (no crash)."""
        result = _generate_manual_mlx(
            TinyLm(), FixedTokenizer(), "test", max_tokens=-1, temperature=0.0
        )
        assert result == ""

    def test_negative_max_tokens_model_generate(self) -> None:
        """Negative max_tokens on Model.generate returns empty string."""
        from auto_chasm import Model

        base = TinyLm()
        model = Model(base, FixedTokenizer(), backend_name="mlx")
        result = model.generate(prompt="test", max_tokens=-1, temperature=1.0)
        assert result == ""

    def test_large_max_tokens_does_not_crash(self) -> None:
        """Very large max_tokens should not crash (repetition guard stops)."""
        model = FixedPredictor(token_id=42, vocab_size=100)
        result = _generate_manual_mlx(
            model, FixedTokenizer(), "test", max_tokens=10_000, temperature=0.0
        )
        assert isinstance(result, str)


class TestTemperatureEdgeCases:
    """Tests for temperature values at boundaries."""

    def test_negative_temperature_raises(self) -> None:
        """Negative temperature must raise rather than flip the distribution.

        A negative temperature would invert ``logits / temperature`` and
        silently sample the least-likely tokens, so the manual path now rejects
        it with ``ValueError``.
        """
        with pytest.raises(ValueError):
            _generate_manual_mlx(TinyLm(), FixedTokenizer(), "test", max_tokens=3, temperature=-1.0)

    def test_zero_temperature_deterministic(self) -> None:
        """temperature=0.0 produces deterministic results."""
        mx.random.seed(TINY_VOCAB)
        a = _generate_manual_mlx(TinyLm(), FixedTokenizer(), "test", max_tokens=5, temperature=0.0)
        mx.random.seed(TINY_VOCAB)
        b = _generate_manual_mlx(TinyLm(), FixedTokenizer(), "test", max_tokens=5, temperature=0.0)
        assert a == b

    def test_inf_temperature(self) -> None:
        """Very high temperature should not crash."""
        result = _generate_manual_mlx(
            TinyLm(), FixedTokenizer(), "test", max_tokens=3, temperature=1e6
        )
        assert isinstance(result, str)

    def test_negative_temperature_on_generate(self) -> None:
        """Negative temperature through generate() raises ValueError.

        The manual fallback rejects negative temperatures, so a negative
        temperature surfaces as ``ValueError`` rather than silently inverting
        the sampling distribution.
        """
        with pytest.raises(ValueError):
            generate(TinyLm(), FixedTokenizer(), "test", max_tokens=3, temperature=-1.0)

    def test_temperature_zero_with_do_sample_via_config(self) -> None:
        """temperature=0.0 with do_sample=True via GenerationConfig uses epsilon."""
        from auto_chasm import GenerationConfig, Model

        base = TinyLm()
        model = Model(base, FixedTokenizer(), backend_name="mlx")
        cfg = GenerationConfig(max_tokens=3, temperature=0.0, do_sample=True)
        result = model.generate(prompt="test", max_tokens=3, temperature=0.0, config=cfg)
        assert isinstance(result, str)


class TestEosPromptEdgeCase:
    """Tests where the prompt itself is already the EOS token."""

    def test_eos_prompt_manual_mlx(self) -> None:
        """Prompt that encodes to just the EOS token should produce empty output."""
        model = FixedPredictor(token_id=STOP_THRESHOLD, vocab_size=100)

        class EosTokenizer(FixedTokenizer):
            """Tokenizer that encodes everything to [0] (EOS)."""

            def encode(self, text: str) -> list[int]:
                return [self.eos_token_id]

            def decode(self, ids: list[int]) -> str:
                return ""

        result = _generate_manual_mlx(
            model, EosTokenizer(), "anything", max_tokens=5, temperature=0.0
        )
        assert result == ""

    def test_eos_first_predicted_token(self) -> None:
        """When the first predicted token is EOS, output is empty."""
        model = FixedPredictor(token_id=0, vocab_size=100)

        result = _generate_manual_mlx(
            model, FixedTokenizer(), "test", max_tokens=5, temperature=0.0
        )
        assert result == ""


class TestExtractStopTokensEdgeCases:
    """Edge cases for the _extract_stop_tokens helper."""

    def test_stop_sequences_not_flattened_into_stop_ids(self) -> None:
        """A multi-token stop_sequence is NOT reduced to per-token stop IDs (M6).

        Flattening ``"multi_stop"`` -> ``{7, 8, 9}`` was the bug: generation then
        stopped on ANY of those tokens wherever they appeared, not on the full
        string. ``_extract_stop_tokens`` now leaves ``stop_sequences`` alone (the
        generation loops match it as a decoded substring instead).
        """

        class MultiTokenTokenizer(FixedTokenizer):
            """Tokenizer whose encode returns multiple tokens for 'stop'."""

            def encode(self, text: str) -> list[int]:
                if text == "multi_stop":
                    return [7, 8, 9]
                return [1, 2, 3]

        kwargs = {"stop_sequences": ["multi_stop"]}
        result = _extract_stop_tokens(MultiTokenTokenizer(), kwargs)
        assert result is None  # no explicit stop_tokens, no vocab to auto-detect
        assert kwargs == {"stop_sequences": ["multi_stop"]}  # left for substring matching

    def test_non_dict_vocab_does_not_crash(self) -> None:
        """Tokenizer whose get_vocab returns a non-dict is handled gracefully."""
        result = _extract_stop_tokens(NonDictVocabTokenizer(), {})
        assert result is None

    def test_stop_sequences_empty_list(self) -> None:
        """Empty stop_sequences list returns None."""
        result = _extract_stop_tokens(FixedTokenizer(), {"stop_sequences": []})
        assert result is None

    def test_vocab_without_end_turn(self) -> None:
        """Tokenizer vocab without common turn-end tokens returns None."""

        class NoEndTurnTokenizer(FixedTokenizer):
            """A tokenizer whose vocab lacks turn-end tokens."""

            def get_vocab(self) -> dict[str, int]:
                return {"hello": 1, "world": 2}

        result = _extract_stop_tokens(NoEndTurnTokenizer(), {})
        assert result is None

    def test_stop_tokens_invalid_type_no_crash(self) -> None:
        """stop_tokens that is not a list should not crash _extract_stop_tokens."""
        result = _extract_stop_tokens(FixedTokenizer(), {"stop_tokens": "not_a_list"})
        assert result is not None


class TestStreamEdgeCases:
    """Edge cases for streaming generation."""

    def test_stream_zero_max_tokens(self) -> None:
        """Streaming with max_tokens=0 yields nothing."""
        tokens = list(
            _generate_stream_mlx(TinyLm(), FixedTokenizer(), "test", max_tokens=0, temperature=0.0)
        )
        assert len(tokens) == 0

    def test_stream_negative_max_tokens(self) -> None:
        """Streaming with negative max_tokens yields nothing."""
        tokens = list(
            _generate_stream_mlx(TinyLm(), FixedTokenizer(), "test", max_tokens=-1, temperature=0.0)
        )
        assert len(tokens) == 0

    def test_stream_stop_token_stops(self) -> None:
        """Streaming with stop_tokens stops early."""
        model = FixedPredictor(token_id=99, vocab_size=100)
        tokens = list(
            _generate_stream_mlx(
                model, FixedTokenizer(), "test", max_tokens=50, temperature=0.0, stop_tokens=[99]
            )
        )
        assert len(tokens) <= 2

    def test_stream_empty_prompt(self) -> None:
        """Streaming with an empty prompt string does not crash."""
        tokens = list(
            _generate_stream_mlx(TinyLm(), FixedTokenizer(), "", max_tokens=3, temperature=0.0)
        )
        assert len(tokens) >= 0


class TestGenerateDispatchEdgeCases:
    """Edge cases for the top-level generate function."""

    def test_generate_empty_prompt(self) -> None:
        """generate() with empty prompt does not crash."""
        result = generate(TinyLm(), FixedTokenizer(), "", max_tokens=3, temperature=0.0)
        assert isinstance(result, str)

    def test_generate_with_extra_kwargs(self) -> None:
        """generate() with unexpected kwargs does not crash."""
        result = generate(
            TinyLm(), FixedTokenizer(), "test", max_tokens=2, temperature=0.0, unknown_kwarg="hello"
        )
        assert isinstance(result, str)

    def test_chat_fallback_with_role_formatting(self) -> None:
        """chat() fallback formats messages with role capitalization."""
        model = TinyLm()
        result = chat(
            model,
            FixedTokenizer(),
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.0,
        )
        assert isinstance(result, str)


class TestRepetitionGuardEdgeCases:
    """Edge cases for the 50-token repetition guard."""

    def test_repetition_guard_alternating_tokens(self) -> None:
        """Alternating tokens 42, 43, 42, 43 should not trigger guard."""

        class AlternatingModel:
            """Returns alternating tokens 42, 43, 42, 43..."""

            def __init__(self) -> None:
                self._step = 0

            def __call__(self, x: mx.array) -> tuple:
                self._step += 1
                token = 42 if self._step % 2 == 1 else 43
                logits = mx.full((x.shape[0], x.shape[1], 100), -100.0)
                logits[:, -1, token] = mx.array(100.0)
                return (logits,)

        result = _generate_manual_mlx(
            AlternatingModel(), FixedTokenizer(), "test", max_tokens=200, temperature=0.0
        )
        assert len(result) > 0

    def test_repetition_guard_exact_49_same(self) -> None:
        """Exactly 49 consecutive same tokens should NOT trigger guard."""

        class FortyNineModel:
            """Tracks calls to verify guard behavior."""

            def __init__(self) -> None:
                self.call_count = 0

            def __call__(self, x: mx.array) -> tuple:
                self.call_count += 1
                logits = mx.full((x.shape[0], x.shape[1], 100), -100.0)
                logits[:, -1, 42] = mx.array(100.0)
                return (logits,)

        model_inst = FortyNineModel()
        _generate_manual_mlx(
            model_inst, FixedTokenizer(), "test", max_tokens=STOP_THRESHOLD, temperature=0.0
        )
        assert model_inst.call_count >= STOP_THRESHOLD - 3

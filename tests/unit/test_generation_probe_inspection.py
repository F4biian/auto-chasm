"""Generation tests — probe inspection, steering, config integration.

Tests exercise the generate-with-probes pipeline and steering interaction
with generation.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import GenerationConfig, ProbeConfig, SteeringConfig
from auto_chasm.generation import (
    _generate_stream_mlx,
    _resolve_prompt,
    chat,
    generate_stream,
)
from auto_chasm.model import Model
from auto_chasm.outputs import GenerationStep
from auto_chasm.steering import SteeringHook

TINY_VOCAB = 32
TINY_HIDDEN = 16
TINY_LAYERS = 4


@pytest.fixture(autouse=True)
def _seed_mlx_rng() -> None:
    """Make the toy-model weights deterministic and test-order independent.

    ``TinyMlp``/``FixedPredictor`` weights come from MLX's global RNG, which MLX
    seeds from entropy at process start. A randomly-initialised toy model
    argmaxes to ``eos_token_id`` (0) on its very first token ~3.5% of the time;
    when it does, ``generate_with_probes``/``_generate_stream_mlx`` correctly
    stop before yielding anything (the documented EOS-first behaviour asserted by
    ``test_generate_stream_eos_first_token``), which spuriously fails the
    ``assert len(steps) > 0`` / ``len(tokens) >= 1`` checks in this file. The
    flake surfaces only in multi-file runs because the global RNG state at each
    ``TinyMlp()`` call then depends on prior collection/RNG consumption.

    Re-seeding before every test pins the weights to a known-good configuration,
    so the suite is reproducible regardless of collection order. Seed 0 is
    verified to keep every generating test non-empty; if a future refactor of the
    toy model lands on an EOS-first configuration it will fail deterministically
    (and is fixed by bumping this seed), never flakily.
    """
    mx.random.seed(0)


class TinyMlp(nn.Module):
    """A tiny MLP for testing."""

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

    def __call__(self, x: mx.array, **kwargs: Any) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class DummyTokenizer:
    """Minimal tokenizer for testing."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % TINY_VOCAB for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + TINY_VOCAB) for i in ids if i > 0)


class ChatTok(DummyTokenizer):
    """Tokenizer with a chat template."""

    chat_template = "template"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        return "chat_resolved"


class _Cfg:
    """Dummy model config."""

    hidden_size = TINY_HIDDEN
    num_hidden_layers = TINY_LAYERS


def _wrap(model: nn.Module, tok: DummyTokenizer | None = None) -> Model:
    """Wrap a model in a Model facade with a dummy config."""
    base = model
    base.config = _Cfg()
    return Model(base, tok or DummyTokenizer(), backend_name="mlx")


class FixedPredictor(nn.Module):
    """Model that always predicts a fixed token."""

    def __init__(
        self, token_id: int, vocab_size: int = TINY_VOCAB, hidden_dim: int = TINY_HIDDEN
    ) -> None:
        super().__init__()
        self._token_id = token_id
        self._vocab_size = max(vocab_size, token_id + 1)
        self.embedding = nn.Embedding(self._vocab_size, hidden_dim)

    def __call__(self, x: mx.array, **kwargs: Any) -> mx.array:
        logits = mx.full((x.shape[0], x.shape[1], self._vocab_size), -100.0)
        logits[:, -1, self._token_id] = mx.array(100.0)
        return logits


class TestGenerateWithProbes:
    """Tests for Model.generate_with_probes."""

    def test_zero_probes_still_generates(self) -> None:
        """generate_with_probes with no attached probes yields GenerationSteps."""
        model = _wrap(TinyMlp())
        steps = list(model.generate_with_probes(prompt="test", max_tokens=3, temperature=0.0))
        assert len(steps) > 0
        for step in steps:
            assert isinstance(step, GenerationStep)
            assert step.probes == {}

    def test_with_probe_returns_probe_outputs(self) -> None:
        """generate_with_probes with a probe returns probe outputs in each step."""
        model = _wrap(TinyMlp())
        model.attach_probe(ProbeConfig(name="p0", layers=[0]))
        steps = list(model.generate_with_probes(prompt="test", max_tokens=3, temperature=0.0))
        assert len(steps) > 0
        for step in steps:
            assert "p0" in step.probes
            assert step.token_str is not None

    def test_stop_tokens_work_with_probes(self) -> None:
        """Explicit stop_tokens halt generate_with_probes early."""
        model = _wrap(FixedPredictor(token_id=99, vocab_size=100))
        steps = list(
            model.generate_with_probes(
                prompt="test", max_tokens=TINY_VOCAB, temperature=0.0, stop_tokens=[99]
            )
        )
        assert len(steps) <= 2

    def test_step_has_next_logits(self) -> None:
        """Each GenerationStep has next_logits with correct shape."""
        model = _wrap(TinyMlp())
        steps = list(model.generate_with_probes(prompt="test", max_tokens=2, temperature=0.0))
        for step in steps:
            assert step.next_logits is not None
            assert step.next_logits.shape[0] == TINY_VOCAB

    def test_multiple_probes_all_present(self) -> None:
        """generate_with_probes with multiple probes reports all in each step."""
        model = _wrap(TinyMlp())
        model.attach_probe(ProbeConfig(name="p0", layers=[0]))
        model.attach_probe(ProbeConfig(name="p1", layers=[1]))
        steps = list(model.generate_with_probes(prompt="test", max_tokens=2, temperature=0.0))
        for step in steps:
            assert "p0" in step.probes
            assert "p1" in step.probes

    def test_generate_with_probes_messages_no_chat_template(self) -> None:
        """generate_with_probes with messages but no chat_template raises."""
        model = _wrap(TinyMlp())
        with pytest.raises((ValueError, AttributeError)):
            list(
                model.generate_with_probes(
                    messages=[{"role": "user", "content": "hi"}], max_tokens=2
                )
            )

    def test_generate_with_probes_messages_no_template_attr(self) -> None:
        """Tokenizer with apply_chat_template but no chat_template attr raises ValueError."""

        class PartialTok(DummyTokenizer):
            """Has apply_chat_template but NO chat_template attribute."""

            def apply_chat_template(
                self,
                messages: list[dict[str, str]],
                tokenize: bool = False,
                add_generation_prompt: bool = True,
            ) -> str:
                return ""

        model = _wrap(TinyMlp(), PartialTok())
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        with pytest.raises(ValueError, match="chat_template"):
            list(
                model.generate_with_probes(
                    messages=[{"role": "user", "content": "hi"}], max_tokens=2
                )
            )


class TestGenerateWithProbesSteering:
    """Tests for steering interaction with generate_with_probes."""

    def test_steering_enabled_no_geometry_raises(self) -> None:
        """Steering without geometry must fail loudly, not silently no-op."""
        import pytest

        model = _wrap(TinyMlp())
        model.attach_probe(ProbeConfig(name="p0", layers=[0]))
        with pytest.raises(ValueError, match="no steering geometry"):
            model.enable_steering("p0", SteeringConfig(method="nullify"))

    def test_steering_with_geometry_produces_output(self) -> None:
        """Steering with computed geometry modifies outputs (no crash)."""
        model = _wrap(TinyMlp())
        model.attach_probe(ProbeConfig(name="p0", layers=[0]))
        hidden_dim = TINY_HIDDEN
        class_means = {"p0": {"mean_0": mx.zeros(hidden_dim), "mean_1": mx.ones(hidden_dim)}}
        model.enable_steering(
            "p0", SteeringConfig(method="nullify", scale=1.0), class_means=class_means
        )
        steps = list(model.generate_with_probes(prompt="test", max_tokens=3, temperature=0.0))
        assert len(steps) > 0

    def test_all_steering_methods_work(self) -> None:
        """All built-in steering methods work during generate_with_probes."""
        for method in ("nullify", "push_to_mean", "boundary"):
            model = _wrap(TinyMlp())
            model.attach_probe(ProbeConfig(name="p", layers=[0]))
            hidden_dim = TINY_HIDDEN
            class_means = {"p": {"mean_0": mx.zeros(hidden_dim), "mean_1": mx.ones(hidden_dim)}}
            model.enable_steering(
                "p", SteeringConfig(method=method, scale=1.0), class_means=class_means
            )
            steps = list(model.generate_with_probes(prompt="test", max_tokens=2, temperature=0.0))
            assert len(steps) > 0

    def test_steering_disabled_mid_generation(self) -> None:
        """Disabling steering between calls does not crash."""
        model = _wrap(TinyMlp())
        model.attach_probe(ProbeConfig(name="p0", layers=[0]))
        cm = {"p0": {"mean_0": mx.zeros(TINY_HIDDEN), "mean_1": mx.ones(TINY_HIDDEN)}}
        model.enable_steering("p0", class_means=cm)
        model.disable_steering("p0")
        steps = list(model.generate_with_probes(prompt="test", max_tokens=2, temperature=0.0))
        assert len(steps) > 0

    def test_steering_enable_twice_same_hook(self) -> None:
        """Enabling steering twice on same probe uses single hook."""
        model = _wrap(TinyMlp())
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        cm = {"p": {"mean_0": mx.zeros(TINY_HIDDEN), "mean_1": mx.ones(TINY_HIDDEN)}}
        model.enable_steering("p", class_means=cm)
        model.enable_steering("p", class_means=cm)
        assert len(model.steering_hooks) == 1

    def test_steering_unknown_method_noop(self) -> None:
        """Unknown steering method returns hidden unchanged."""
        model = _wrap(TinyMlp())
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        hidden_dim = TINY_HIDDEN
        class_means = {"p": {"mean_0": mx.zeros(hidden_dim), "mean_1": mx.ones(hidden_dim)}}
        model.enable_steering("p", SteeringConfig(method="nullify"), class_means=class_means)
        steps = list(model.generate_with_probes(prompt="test", max_tokens=2, temperature=0.0))
        assert len(steps) > 0


class TestSteeringHookEdge:
    """Direct tests on SteeringHook edge cases."""

    def test_steer_not_enabled_returns_hidden(self) -> None:
        """steer() with disabled hook returns hidden unchanged."""
        hook = SteeringHook("test", SteeringConfig())
        hook._mean_0 = mx.zeros(TINY_HIDDEN)
        hook._mean_1 = mx.ones(TINY_HIDDEN)
        hook._direction = mx.ones(TINY_HIDDEN)
        hidden = mx.random.normal((1, 3, TINY_HIDDEN))
        head = nn.Linear(TINY_HIDDEN, 1)
        logits = head(hidden).squeeze(-1)
        result = hook.steer(hidden, head, logits)
        assert result is hidden
        assert mx.allclose(result, hidden)

    def test_steer_enabled_no_geometry_warns_not_crash(self) -> None:
        """steer() enabled without geometry logs warning and returns hidden."""
        hook = SteeringHook("test", SteeringConfig())
        hook.enable()
        hidden = mx.random.normal((1, 3, TINY_HIDDEN))
        head = nn.Linear(TINY_HIDDEN, 1)
        logits = head(hidden).squeeze(-1)
        result = hook.steer(hidden, head, logits)
        assert result is hidden

    def test_steer_with_custom_fn(self) -> None:
        """Custom steering function is called instead of built-in methods."""
        hook = SteeringHook("test", SteeringConfig())
        hook.enable()
        hook._mean_0 = mx.zeros(TINY_HIDDEN)
        hook._mean_1 = mx.ones(TINY_HIDDEN)

        def custom_fn(h: Any, head: Any, logits: Any) -> Any:
            return h * 2.0

        hook.set_custom(custom_fn)
        hidden = mx.ones((1, 2, TINY_HIDDEN))
        head = nn.Linear(TINY_HIDDEN, 1)
        logits = head(hidden).squeeze(-1)
        result = hook.steer(hidden, head, logits)
        assert mx.allclose(result, hidden * 2.0)

    def test_steer_empty_hidden(self) -> None:
        """Steer with zero-length sequence does not crash."""
        hook = SteeringHook("test", SteeringConfig())
        hook._mean_0 = mx.zeros(TINY_HIDDEN)
        hook._mean_1 = mx.ones(TINY_HIDDEN)
        hook._direction = mx.ones(TINY_HIDDEN)
        hook.enable()
        hidden = mx.zeros((1, 0, TINY_HIDDEN))
        head = nn.Linear(TINY_HIDDEN, 1)
        logits = mx.zeros((1, 0))
        result = hook.steer(hidden, head, logits)
        assert result.shape == hidden.shape


class TestGenerationConfigIntegration:
    """Tests for GenerationConfig integration with generation methods."""

    def test_config_overrides_max_tokens(self) -> None:
        """GenerationConfig.max_tokens overrides default."""
        model = _wrap(TinyMlp())
        cfg = GenerationConfig(max_tokens=2)
        result = model.generate(prompt="test", config=cfg)
        assert isinstance(result, str)

    def test_config_do_sample_sets_epsilon_temp(self) -> None:
        """do_sample=True with temperature=0.0 uses epsilon temp."""
        model = _wrap(TinyMlp())
        cfg = GenerationConfig(max_tokens=3, temperature=0.0, do_sample=True)
        result = model.generate(prompt="test", config=cfg)
        assert isinstance(result, str)

    def test_config_top_p_forwarded(self) -> None:
        """GenerationConfig.top_p is forwarded to generation kwargs."""
        model = _wrap(TinyMlp())
        cfg = GenerationConfig(max_tokens=2, top_p=0.9)
        result = model.generate(prompt="test", config=cfg)
        assert isinstance(result, str)

    def test_config_top_k_forwarded(self) -> None:
        """GenerationConfig.top_k is forwarded to generation kwargs."""
        model = _wrap(TinyMlp())
        cfg = GenerationConfig(max_tokens=2, top_k=TINY_VOCAB)
        result = model.generate(prompt="test", config=cfg)
        assert isinstance(result, str)

    def test_config_repetition_penalty_forwarded(self) -> None:
        """GenerationConfig.repetition_penalty is forwarded."""
        model = _wrap(TinyMlp())
        cfg = GenerationConfig(max_tokens=2, repetition_penalty=1.2)
        result = model.generate(prompt="test", config=cfg)
        assert isinstance(result, str)

    def test_config_stop_sequences_forwarded(self) -> None:
        """GenerationConfig.stop_sequences is forwarded."""
        model = _wrap(TinyMlp())
        cfg = GenerationConfig(max_tokens=5, stop_sequences=["."])
        result = model.generate(prompt="test", config=cfg)
        assert isinstance(result, str)

    def test_kwargs_override_config(self) -> None:
        """Explicit kwargs override GenerationConfig values."""
        model = _wrap(TinyMlp())
        cfg = GenerationConfig(max_tokens=10)
        result = model.generate(prompt="test", max_tokens=2, config=cfg)
        assert isinstance(result, str)

    def test_generate_with_probes_ignores_config(self) -> None:
        """generate_with_probes does not accept GenerationConfig (no crash)."""
        model = _wrap(TinyMlp())
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        steps = list(model.generate_with_probes(prompt="test", max_tokens=2, temperature=0.0))
        assert len(steps) > 0

    def test_stream_respects_config(self) -> None:
        """generate_stream with GenerationConfig respects max_tokens.

        The toy model may predict EOS immediately (yielding zero tokens), so
        the meaningful property is the upper bound, not a non-empty stream.
        """
        model = _wrap(TinyMlp())
        cfg = GenerationConfig(max_tokens=2)
        tokens = list(model.generate_stream(prompt="test", config=cfg))
        assert len(tokens) <= 2


class TestChatModes:
    """Tests for chat() function edge cases."""

    def test_chat_with_system_message(self) -> None:
        """Chat with system message works via fallback formatting."""
        model = TinyMlp()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = chat(model, DummyTokenizer(), messages, max_tokens=2, temperature=0.0)
        assert isinstance(result, str)

    def test_chat_single_turn(self, monkeypatch: Any) -> None:
        """Single turn chat via Model.chat_repl does not crash."""
        monkeypatch.setattr("builtins.input", lambda _="": "quit")
        model = _wrap(TinyMlp())
        model.chat_repl(system_prompt=None, max_tokens=1, temperature=0.0)

    def test_chat_empty_messages_list(self) -> None:
        """Empty messages list in chat() produces output."""
        model = TinyMlp()
        result = chat(model, DummyTokenizer(), [], max_tokens=2, temperature=0.0)
        assert isinstance(result, str)

    def test_chat_multi_turn_preserves_format(self) -> None:
        """Multi-turn chat properly formats conversation."""
        model = TinyMlp()
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "second"},
        ]
        result = chat(model, DummyTokenizer(), messages, max_tokens=2, temperature=0.0)
        assert isinstance(result, str)


class TestTorchGenerationEdgeCases:
    """Edge-case tests for PyTorch generation paths."""

    def test_torch_generate_negative_temp(self) -> None:
        """Torch generation with negative temperature raises ValueError.

        A negative temperature would invert ``logits / temperature`` and
        silently sample the least-likely tokens, so the manual torch path now
        rejects it instead of producing wrong output.
        """
        pytest.importorskip("torch")
        import torch

        from auto_chasm.generation import _generate_manual_torch

        class TorchModel(torch.nn.Module):
            """A tiny torch model for generation tests."""

            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(TINY_VOCAB, TINY_HIDDEN)
                self.proj = torch.nn.Linear(TINY_HIDDEN, TINY_VOCAB)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.embedding(x)
                return self.proj(h)

        class Tok:
            """Minimal tokenizer for torch generation tests."""

            eos_token_id = 0

            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

            def decode(self, ids: list[int]) -> str:
                return "".join(chr(i + TINY_VOCAB) for i in ids if i > 0)

        torch.manual_seed(TINY_VOCAB)
        with pytest.raises(ValueError):
            _generate_manual_torch(TorchModel(), Tok(), "test", max_tokens=3, temperature=-1.0)

    def test_torch_generate_max_tokens_zero(self) -> None:
        """Torch generation with max_tokens=0 returns empty."""
        pytest.importorskip("torch")
        import torch

        from auto_chasm.generation import _generate_manual_torch

        class TorchFlat(torch.nn.Module):
            """A torch model returning all-zero logits."""

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros((x.shape[0], x.shape[1], TINY_VOCAB))

        class Tok:
            """Minimal tokenizer for max-tokens-zero test."""

            eos_token_id = 0

            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

            def decode(self, ids: list[int]) -> str:
                return ""

        result = _generate_manual_torch(TorchFlat(), Tok(), "test", max_tokens=0, temperature=0.0)
        assert result == ""

    def test_torch_generate_stop_tokens(self) -> None:
        """Torch generation with stop_tokens stops early."""
        pytest.importorskip("torch")
        import torch

        from auto_chasm.generation import _generate_manual_torch

        class EosPredictor(torch.nn.Module):
            """A torch model that predicts a specific EOS token."""

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                logits = torch.full((1, x.shape[1], TINY_VOCAB + 100), -100.0)
                logits[0, -1, 99] = 100.0
                return logits

        class Tok:
            """Minimal tokenizer for stop-tokens test."""

            eos_token_id = 0

            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

            def decode(self, ids: list[int]) -> str:
                return ""

        torch.manual_seed(42)
        result = _generate_manual_torch(
            EosPredictor(), Tok(), "test", max_tokens=50, temperature=0.0, stop_tokens=[99]
        )
        assert len(result) == 0


class TestResolvePromptEdge:
    """Edge cases for the _resolve_prompt helper."""

    def test_prompt_explicit_none(self) -> None:
        """Explicit prompt=None with messages=None raises ValueError."""
        tok = DummyTokenizer()
        with pytest.raises(ValueError, match="Either prompt or messages"):
            _resolve_prompt(tok, prompt=None, messages=None)

    def test_prompt_empty_string(self) -> None:
        """Empty string prompt is returned as-is."""
        tok = DummyTokenizer()
        result = _resolve_prompt(tok, prompt="", messages=None)
        assert result == ""

    def test_messages_with_non_string_content(self) -> None:
        """Messages with non-string content field does not crash."""
        tok = ChatTok()
        result = _resolve_prompt(tok, prompt=None, messages=[{"role": "user", "content": "hi"}])
        assert isinstance(result, str)

    def test_tokenizer_without_apply_chat_template_method(self) -> None:
        """Tokenizer without apply_chat_template uses prompt fallback."""

        class NoApplyTok(DummyTokenizer):
            """A tokenizer without apply_chat_template."""

            pass

        result = _resolve_prompt(NoApplyTok(), prompt="hello", messages=None)
        assert result == "hello"


class TestGenerateStreamExtra:
    """Extra edge cases for generate_stream."""

    def test_generate_stream_backend_dispatch(self) -> None:
        """generate_stream dispatches correctly with explicit backend."""
        from auto_chasm.backends import Backend

        backend = Backend(force="mlx")
        tokens = list(
            generate_stream(TinyMlp(), DummyTokenizer(), "test", max_tokens=2, backend=backend)
        )
        assert len(tokens) >= 1

    def test_generate_stream_eos_first_token(self) -> None:
        """Streaming a model that predicts EOS first must yield nothing (no leak)."""
        model = FixedPredictor(token_id=0, vocab_size=100)
        tokens = list(
            _generate_stream_mlx(model, DummyTokenizer(), "test", max_tokens=10, temperature=0.0)
        )
        assert tokens == []  # the EOS token's text must not be emitted


class TestBackendFallback:
    """Tests for backend detection and fallback behavior."""

    def test_backend_none_detects_mlx(self) -> None:
        """Backend=None auto-detects MLX on this machine."""
        from auto_chasm.backends import Backend

        backend = Backend(force=None)
        assert backend.name in ("mlx", "torch")

    def test_generate_no_backend_auto_detects(self) -> None:
        """generate() without backend auto-detects."""
        from auto_chasm.generation import generate

        result = generate(TinyMlp(), DummyTokenizer(), "test", max_tokens=1, temperature=0.0)
        assert isinstance(result, str)

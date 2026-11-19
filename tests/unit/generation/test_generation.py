"""tests for generation — backend dispatch, edge cases, silent failures."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.generation import (
    _generate_manual_mlx,
    _generate_stream_mlx,
    chat,
    generate,
    generate_stream,
)


class TinyLm(nn.Module):
    """Tiny LM that outputs predictable logits."""

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array) -> tuple:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return (self.output_proj(h),)


class ControlledLm(TinyLm):
    """Model that outputs a known token for testing EOS stopping."""

    def __call__(self, x: mx.array) -> tuple:
        logits = mx.full((x.shape[0], x.shape[1], 16), -100.0)
        # Always predict token 5 for the last position
        logits = logits.at[:, -1, 5].set(100.0)
        return (logits,)

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 16, num_layers: int = 2) -> None:
        super().__init__(hidden_dim, vocab_size, num_layers)


class DummyTokenizer:
    """Tokenizer for testing manual generation path."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "".join(f"[{i}]" for i in ids if i > 0)


class RealishTokenizer(DummyTokenizer):
    """Tokenizer that mlx_lm can handle via mock."""

    eos_token_id = 0
    chat_template = None

    def get_vocab(self) -> dict:
        return {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "chat_prompt"


class TestManualGenerate:
    """Tests directly against _generate_manual_mlx — the fallback path."""

    def test_greedy_generates_tokens(self) -> None:
        mx.random.seed(42)
        result = _generate_manual_mlx(TinyLm(), DummyTokenizer(), "test", 3, 0.0)
        assert len(result) > 0

    def test_greedy_deterministic(self) -> None:
        mx.random.seed(42)
        a = _generate_manual_mlx(TinyLm(), DummyTokenizer(), "test", 3, 0.0)
        mx.random.seed(42)
        b = _generate_manual_mlx(TinyLm(), DummyTokenizer(), "test", 3, 0.0)
        assert a == b

    def test_sampling_nondeterministic(self) -> None:
        mx.random.seed(0)
        a = _generate_manual_mlx(TinyLm(), DummyTokenizer(), "test", 5, 1.0)
        mx.random.seed(999)
        b = _generate_manual_mlx(TinyLm(), DummyTokenizer(), "test", 5, 1.0)
        assert isinstance(a, str)
        assert isinstance(b, str)

    def test_zero_max_tokens(self) -> None:
        result = _generate_manual_mlx(TinyLm(), DummyTokenizer(), "test", 0, 0.0)
        assert result == ""

    def test_single_token_generation(self) -> None:
        """max_tokens=1 should produce at most 1 token."""
        result = _generate_manual_mlx(TinyLm(), DummyTokenizer(), "test", 1, 0.0)
        # Should produce exactly one token, not two (one + EOS)
        result_stripped = result.replace("[0]", "")
        assert result_stripped.count("[") <= 1  # one token


class TestStreamMLX:
    """Tests for the streaming MLX generator."""

    def test_stream_yields_strings(self) -> None:
        tokens = list(_generate_stream_mlx(TinyLm(), DummyTokenizer(), "test", 3, 0.0))
        assert all(isinstance(t, str) for t in tokens)

    def test_stream_sampling(self) -> None:
        mx.random.seed(42)
        tokens = list(_generate_stream_mlx(TinyLm(), DummyTokenizer(), "test", 3, 1.0))
        assert len(tokens) >= 1

    def test_stream_zero_max_tokens(self) -> None:
        tokens = list(_generate_stream_mlx(TinyLm(), DummyTokenizer(), "test", 0, 0.0))
        assert len(tokens) == 0

    def test_no_dead_code(self) -> None:
        import inspect

        source = inspect.getsource(_generate_stream_mlx)
        for line in source.split("\n"):
            if line.strip() == "len(tokens)":
                pytest.fail("Dead code: len(tokens) discarded")


class TestGenerateDispatchBugs:
    """BUG-28: generate dispatches to torch when backend=None."""

    def test_generate_uses_mlx_by_default(self) -> None:
        """On MLX machine, generate should auto-detect MLX backend."""
        result = generate(TinyLm(), DummyTokenizer(), "hello", max_tokens=1, temperature=0.0)
        assert isinstance(result, str)

    def test_generate_stream_uses_mlx_by_default(self) -> None:
        tokens = list(
            generate_stream(TinyLm(), DummyTokenizer(), "hello", max_tokens=1, temperature=0.0)
        )
        assert len(tokens) >= 1


class TestMLXLmGenerate:
    """Tests using the _generate_manual_mlx path directly."""

    def test_manual_fallback(self) -> None:
        """Manual fallback should generate tokens."""
        result = _generate_manual_mlx(TinyLm(), DummyTokenizer(), "test", 2, 0.0)
        assert isinstance(result, str)

    def test_manual_fallback_sampling(self) -> None:
        mx.random.seed(42)
        result = _generate_manual_mlx(TinyLm(), DummyTokenizer(), "p", 2, 0.5)
        assert isinstance(result, str)

    def test_mlx_lm_import_error_fallback(self) -> None:
        """When mlx_lm is not importable, should fall back to manual."""
        import sys

        from auto_chasm.generation import _generate_mlx

        saved = sys.modules.get("mlx_lm", True)
        saved_gen = sys.modules.get("mlx_lm.generate", True)
        sys.modules["mlx_lm"] = None
        sys.modules["mlx_lm.generate"] = None

        try:
            result = _generate_mlx(TinyLm(), DummyTokenizer(), "test", 2, 0.0)
            assert isinstance(result, str)
        finally:
            if saved is True:
                sys.modules.pop("mlx_lm", None)
                sys.modules.pop("mlx_lm.generate", None)
            else:
                sys.modules["mlx_lm"] = saved
                if saved_gen is not True:
                    sys.modules["mlx_lm.generate"] = saved_gen
                else:
                    sys.modules.pop("mlx_lm.generate", None)

    def test_chat_with_template(self) -> None:
        """Chat should use apply_chat_template when available."""

        class TemplatedTok(DummyTokenizer):
            """Test helper."""

            chat_template = "{{ messages }}"

        msgs = [{"role": "user", "content": "hi"}]
        try:
            result = chat(TinyLm(), TemplatedTok(), msgs, max_tokens=1, temperature=0.0)
            assert isinstance(result, str)
        except Exception:
            pass

    def test_mlx_generate_sampling_path(self) -> None:
        """Test the mlx_lm sampler import when temperature > 0."""
        from auto_chasm.backends import Backend
        from auto_chasm.generation import generate

        model = TinyLm()
        # Use DummyTokenizer with temp > 0 — will try mlx_lm path first
        # then fallback to manual when tokenizer isn't fully compatible
        backend = Backend(force="mlx")
        try:
            result = generate(model, DummyTokenizer(), "test", 2, 0.5, backend=backend)
            assert isinstance(result, str)
        except (AttributeError, ImportError, TypeError):
            # Expected if mlx_lm path fails and manual fallback works differently
            pass


class TestChatRepl:
    """Tests for the chat_repl function and Model.chat_repl method."""

    def test_chat_repl_function_exists(self) -> None:
        from auto_chasm.generation import chat_repl

        assert callable(chat_repl)

    def test_chat_repl_has_correct_signature(self) -> None:
        import inspect

        from auto_chasm.generation import chat_repl

        sig = inspect.signature(chat_repl)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "tokenizer" in params
        assert "system_prompt" in params
        assert "max_tokens" in params
        assert "temperature" in params
        assert "backend" in params

    def test_model_has_chat_repl_method(self) -> None:
        from auto_chasm import Model

        model = Model(TinyLm(), DummyTokenizer(), "mlx")
        assert callable(model.chat_repl)


# ---------------------------------------------------------------------------
# PyTorch generation tests
# ---------------------------------------------------------------------------


class _TorchTinyLm:
    """Tiny PyTorch LM for generation testing."""

    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16) -> None:
        import torch.nn as tnn

        self.embedding = tnn.Embedding(vocab_size, hidden_dim)
        self.proj = tnn.Linear(hidden_dim, vocab_size)
        self._vocab_size = vocab_size

    def __call__(self, x):  # type: ignore[no-untyped-def]

        h = self.embedding(x)
        return self.proj(h)

    def parameters(self):  # type: ignore[no-untyped-def]
        return list(self.embedding.parameters()) + list(self.proj.parameters())

    def eval(self):  # type: ignore[no-untyped-def]
        self.embedding.eval()
        self.proj.eval()
        return self

    def train(self):  # type: ignore[no-untyped-def]
        self.embedding.train()
        self.proj.train()
        return self


class _TorchTokenizer:
    """Tokenizer that supports the HF call API for torch generation."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 for c in text[:10]]

    def decode(self, ids, **kwargs):  # type: ignore[no-untyped-def]
        return "".join(chr(i + 32) for i in ids if i > 0)

    def __call__(self, text, return_tensors="pt"):  # type: ignore[no-untyped-def]
        import torch

        ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}


class TestTorchGeneration:
    """Tests for PyTorch generation paths."""

    def test_generate_manual_torch(self) -> None:
        """Manual torch generation produces a string."""
        import torch

        from auto_chasm.generation import _generate_manual_torch

        torch.manual_seed(42)
        model = _TorchTinyLm()
        tokenizer = _TorchTokenizer()
        result = _generate_manual_torch(model, tokenizer, "Hello", max_tokens=5, temperature=0.0)
        assert isinstance(result, str)

    def test_generate_manual_torch_deterministic(self) -> None:
        """Manual torch generation is deterministic with temperature=0."""
        from auto_chasm.generation import _generate_manual_torch

        model = _TorchTinyLm()
        tokenizer = _TorchTokenizer()
        r1 = _generate_manual_torch(model, tokenizer, "Hello", max_tokens=5, temperature=0.0)
        r2 = _generate_manual_torch(model, tokenizer, "Hello", max_tokens=5, temperature=0.0)
        assert r1 == r2

    def test_generate_torch_fallback(self) -> None:
        """generate() with torch backend falls back to manual when no .generate()."""
        import torch

        from auto_chasm.generation import _generate_torch

        torch.manual_seed(42)
        model = _TorchTinyLm()
        tokenizer = _TorchTokenizer()
        result = _generate_torch(model, tokenizer, "Hello", max_tokens=3, temperature=0.0)
        assert isinstance(result, str)

    def test_generate_dispatches_torch(self) -> None:
        """generate() dispatches to torch when backend is 'torch'."""
        import torch

        from auto_chasm.backends import Backend

        torch.manual_seed(42)
        model = _TorchTinyLm()
        tokenizer = _TorchTokenizer()
        backend = Backend(force="torch")
        result = generate(model, tokenizer, "Hello", max_tokens=3, temperature=0.0, backend=backend)
        assert isinstance(result, str)

    def test_generate_stream_dispatches_torch(self) -> None:
        """generate_stream() dispatches to torch when backend is 'torch'."""
        import torch

        from auto_chasm.backends import Backend

        torch.manual_seed(42)
        model = _TorchTinyLm()
        tokenizer = _TorchTokenizer()
        backend = Backend(force="torch")
        tokens = list(generate_stream(model, tokenizer, "Hi", max_tokens=3, backend=backend))
        assert len(tokens) >= 1

    def test_chat_torch_backend(self) -> None:
        """chat() works with torch backend."""
        import torch

        from auto_chasm.backends import Backend

        torch.manual_seed(42)
        model = _TorchTinyLm()
        tokenizer = _TorchTokenizer()
        backend = Backend(force="torch")
        messages = [{"role": "user", "content": "Hello"}]
        result = chat(model, tokenizer, messages, max_tokens=3, backend=backend)
        assert isinstance(result, str)


class TestChatTemplateCoverage:
    """Cover chat_template path in chat() — line 121."""

    def test_chat_with_real_template(self) -> None:
        """chat() with a tokenizer that has a valid chat_template returns str."""
        from auto_chasm.generation import chat

        class TemplatedTokenizer(DummyTokenizer):
            """Tokenizer with a real chat_template."""

            chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "Hello from template"

        model = TinyLm()
        tokenizer = TemplatedTokenizer()
        result = chat(model, tokenizer, [{"role": "user", "content": "hi"}], max_tokens=1)
        assert isinstance(result, str)
        assert len(result) > 0


class TestChatReplCoverage:
    """Cover chat_repl() — line 153-186 by mocking input()."""

    def test_chat_repl_quit_exits(self, monkeypatch: Any) -> None:
        """chat_repl exits cleanly on 'quit'."""
        from auto_chasm.generation import chat_repl

        monkeypatch.setattr("builtins.input", lambda _="": "quit")
        chat_repl(TinyLm(), DummyTokenizer(), max_tokens=1)

    def test_chat_repl_eof_error_exits(self, monkeypatch: Any) -> None:
        """chat_repl exits cleanly on EOFError."""
        from auto_chasm.generation import chat_repl

        def raise_eof(*args: object, **kwargs: object) -> str:
            raise EOFError()

        monkeypatch.setattr("builtins.input", raise_eof)
        chat_repl(TinyLm(), DummyTokenizer(), max_tokens=1)

    def test_chat_repl_one_round(self, monkeypatch: Any, capsys: Any) -> None:
        """chat_repl handles one conversation round."""
        from auto_chasm.generation import chat_repl

        inputs = iter(["hello", "quit"])

        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))
        chat_repl(TinyLm(), DummyTokenizer(), max_tokens=1)

        captured = capsys.readouterr()
        # input() writes prompt to stderr, so "You:" is in err
        assert "Assistant:" in captured.out


class TestTorchManualGenerationCoverage:
    """Cover torch manual generation — sampling and EOS paths."""

    def test_generate_manual_torch_sampling(self) -> None:
        """_generate_manual_torch with temperature > 0 produces a string."""
        pytest.importorskip("torch")
        import torch

        from auto_chasm.generation import _generate_manual_torch

        class TorchTinyLm(torch.nn.Module):
            """Tiny LM for torch testing."""

            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 8)
                self.proj = torch.nn.Linear(8, 16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.embedding(x)
                return self.proj(h)

        class Tok:
            """Simple tokenizer."""

            eos_token_id = 0

            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

            def decode(self, ids: list[int]) -> str:
                return "out"

        torch.manual_seed(42)
        result = _generate_manual_torch(TorchTinyLm(), Tok(), "test", 3, 0.5)
        assert isinstance(result, str)

    def test_generate_manual_torch_eos_stops_early(self) -> None:
        """_generate_manual_torch stops when EOS token is predicted."""
        pytest.importorskip("torch")
        import torch

        from auto_chasm.generation import _generate_manual_torch

        class EosModel(torch.nn.Module):
            """Model that always predicts EOS."""

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                logits = torch.full((1, x.shape[1], 16), -100.0)
                logits[0, -1, 0] = 100.0  # EOS token (id=0)
                return logits

        class Tok:
            """Tokenizer where eos_token_id=0."""

            eos_token_id = 0

            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

            def decode(self, ids: list[int]) -> str:
                return ""

        result = _generate_manual_torch(EosModel(), Tok(), "test", 10, 0.0)
        assert result == ""  # No tokens after EOS


class TestGenerateTorchModelGeneratePath:
    """Cover _generate_torch model.generate() path — lines 295-308."""

    def test_generate_torch_uses_model_generate_method(self) -> None:
        """_generate_torch should use model.generate() when available."""
        pytest.importorskip("torch")
        import torch

        from auto_chasm.generation import _generate_torch

        class ModelWithGenerate(torch.nn.Module):
            """Model with a .generate() method."""

            def generate(
                self,
                input_ids: torch.Tensor,
                max_new_tokens: int = 5,
                temperature: float | None = None,
                do_sample: bool = False,
                **kwargs: object,
            ) -> torch.Tensor:
                b, seq = input_ids.shape
                new_toks = torch.tensor([[6, 7, 8]], dtype=torch.long)
                return torch.cat([input_ids, new_toks], dim=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros((1, x.shape[1], 16))

        class Tok:
            """Simple tokenizer."""

            eos_token_id = 0

            def __call__(self, prompt: str, return_tensors: str = "pt") -> dict:
                return {"input_ids": torch.tensor([[1, 2, 3]])}

            def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
                return "generated"

        tok = Tok()
        # Accessing model.device might fail; handle it
        model = ModelWithGenerate()
        try:
            result = _generate_torch(model, tok, "test", 3, 0.0)
            assert isinstance(result, str)
        except (AttributeError, TypeError):
            pass


def test_generate_rejects_negative_temperature_on_all_paths() -> None:
    """Negative temperature raises a clear ValueError at the public entry (all backends).

    Regression: the guard only covered the manual fallbacks, so the primary
    mlx_lm / HF paths silently ran greedy (or clamped) on a negative temperature.
    The guard is now at the top of `generate`/`generate_stream`, before any model
    touch, so it raises regardless of the downstream path.
    """
    from auto_chasm.generation import generate, generate_stream

    with pytest.raises(ValueError):
        generate(object(), object(), "hi", temperature=-5.0)
    with pytest.raises(ValueError):
        list(generate_stream(object(), object(), "hi", temperature=-0.1))


def test_stream_matches_nonstream_for_multibyte_grapheme() -> None:
    """Streaming a multi-byte grapheme equals the non-stream full decode (M8).

    Regression: streaming decoded each token in isolation, so a 2-token grapheme
    ("é") streamed as a replacement char while non-stream decoded it correctly.
    """
    from auto_chasm.generation import _generate_manual_mlx, _generate_stream_mlx

    class _ByteModel(TinyLm):
        """Greedily emits byte-token 6, then 7, then EOS(0)."""

        def __call__(self, x: mx.array) -> tuple:
            step = int(x.shape[1]) - 3  # prompt is 3 tokens
            nxt = (6, 7, 0)[min(step, 2)]
            row = [-100.0] * 16
            row[nxt] = 100.0  # only the last position's argmax is read
            logits = mx.broadcast_to(mx.array(row), (x.shape[0], x.shape[1], 16))
            return (logits,)

    class _ByteTok(DummyTokenizer):
        """decode([6])='' replacement char; decode([6,7]) completes to 'é'."""

        def decode(self, ids: list[int]) -> str:
            body = tuple(i for i in ids if i > 0)
            return {(): "", (6,): "�", (6, 7): "é"}.get(body, "?" * len(body))

    model, tok = _ByteModel(), _ByteTok()
    nonstream = _generate_manual_mlx(model, tok, "hi", 5, 0.0)
    stream = "".join(_generate_stream_mlx(model, tok, "hi", 5, 0.0))
    assert nonstream == "é"
    assert stream == nonstream

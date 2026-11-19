"""Tests for generation and history.

Covers ``src/auto_chasm/generation.py``, ``src/auto_chasm/history.py``,
and ``GenerationConfig`` validation in ``src/auto_chasm/config.py``.

Tests named ``test_BUG_*`` are regression tests for specific past defects; the
rest are general regression coverage.
"""

from __future__ import annotations

import contextlib
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import GenerationConfig, Model, ProbeConfig
from auto_chasm.generation import (
    _extract_stop_tokens,
    _generate_manual_mlx,
    _generate_mlx,
    _generate_stream_mlx,
    chat,
)
from auto_chasm.history import History, HistoryEntry

# ---------------------------------------------------------------------------
# Tiny deterministic models / tokenizers
# ---------------------------------------------------------------------------


class _VariedModel(nn.Module):
    """A tiny model whose argmax varies with the input (deterministic)."""

    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.output_proj = nn.Linear(hidden, vocab)
        self.vocab = vocab

    def __call__(self, x: mx.array, **kwargs: Any) -> mx.array:
        return self.output_proj(self.embedding(x))


class _ConstModel(nn.Module):
    """A model that always makes one token the argmax (forces repeat guard)."""

    def __init__(self, vocab: int = 32, const: int = 7) -> None:
        super().__init__()
        self.vocab = vocab
        self.const = const

    def __call__(self, x: mx.array, **kwargs: Any) -> mx.array:
        b, t = x.shape
        onehot = [0.0] * self.vocab
        onehot[self.const] = 10.0
        return mx.zeros((b, t, self.vocab)) + mx.array(onehot)


class _Tok:
    """Minimal tokenizer; never emits eos (id 0) for normal chars."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % 31) + 1 for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids if i > 0)


def _config_for(model: Any, vocab: int = 32) -> Any:
    class _Cfg:
        hidden_size = 16
        num_hidden_layers = 4
        vocab_size = vocab

    model.config = _Cfg()
    return model


# ===========================================================================
# BUG 1 — streaming emits one extra token vs non-streaming when the repeat
#          guard fires (streaming vs non-streaming parity violation).
# ===========================================================================


def test_BUG_stream_repeat_guard_off_by_one_vs_manual() -> None:
    """Streaming and non-streaming must stop at the SAME token count.

    The streaming path yields the token *before* the repeat-guard check, while
    the manual (non-stream) path checks the guard *before* appending. For a
    model that always repeats, streaming emits exactly one more token than the
    non-streaming path for the same ``max_repeat`` — a streaming/non-streaming
    parity violation.
    """
    model = _ConstModel(const=7)
    tok = _Tok()
    for max_repeat in (1, 2, 3, 5):
        n_stream = len(list(_generate_stream_mlx(model, tok, "hi", 50, 0.0, max_repeat=max_repeat)))
        manual = _generate_manual_mlx(model, tok, "hi", 50, 0.0, max_repeat=max_repeat)
        n_manual = len(manual)
        assert n_stream == n_manual, (
            f"max_repeat={max_repeat}: stream emitted {n_stream} tokens but "
            f"manual emitted {n_manual} — streaming/non-streaming parity broken"
        )


def test_BUG_stream_repeat_guard_off_by_one_vs_probes() -> None:
    """``generate_with_probes`` and ``generate_stream`` must agree on length.

    Same root cause as the manual comparison: the streaming path emits the
    repeated token before breaking, so it returns one more token than
    ``generate_with_probes`` for the same ``max_repeat``.
    """
    model = _ConstModel(const=7)
    tok = _Tok()
    _config_for(model)
    wrapper = Model(model, tok, backend_name="mlx")
    for max_repeat in (1, 2, 3):
        n_stream = len(list(_generate_stream_mlx(model, tok, "hi", 50, 0.0, max_repeat=max_repeat)))
        n_probes = len(
            list(wrapper.generate_with_probes("hi", max_tokens=50, max_repeat=max_repeat))
        )
        assert n_stream == n_probes, (
            f"max_repeat={max_repeat}: stream emitted {n_stream}, "
            f"generate_with_probes emitted {n_probes}"
        )


# ===========================================================================
# BUG 2 — torch generation crashes on the documented ``do_sample`` kwarg,
#          while MLX accepts it (cross-backend divergence + crash on valid
#          input). ``do_sample`` is a documented GenerationConfig field.
# ===========================================================================


class _StubHF:
    """An HF-like model exposing ``.generate`` with the standard signature."""

    device = "cpu"

    def generate(
        self,
        input_ids: Any,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool = False,
        **kw: Any,
    ) -> Any:
        import torch

        new = torch.zeros((input_ids.shape[0], 2), dtype=input_ids.dtype)
        return torch.cat([input_ids, new], dim=1)


class _HFTok:
    eos_token_id = 0

    def __call__(self, text: str, return_tensors: str | None = None) -> dict[str, Any]:
        import torch

        return {"input_ids": torch.tensor([[(ord(c) % 31) + 1 for c in text[:10]]])}

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % 31) + 1 for c in text[:10]]

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        with contextlib.suppress(AttributeError):
            ids = ids.tolist()
        return "".join(chr(int(i) + 32) for i in ids if int(i) > 0)


def test_BUG_torch_do_sample_kwarg_crashes_but_mlx_accepts() -> None:
    """Passing ``do_sample`` should behave the same on both backends.

    ``do_sample`` is a documented ``GenerationConfig`` field. The MLX path pops
    it and generates fine; the torch path forwards it to ``model.generate``,
    which already receives ``do_sample=temperature>0``, raising
    ``TypeError: got multiple values for keyword argument 'do_sample'``.
    """
    pytest.importorskip("torch")
    from auto_chasm.generation import _generate_torch

    # MLX: accepts do_sample gracefully.
    mlx_model = _VariedModel()
    mlx_out = _generate_mlx(mlx_model, _Tok(), "abc", 4, 0.0, do_sample=True)
    assert isinstance(mlx_out, str)

    # torch: must NOT crash on the same documented kwarg.
    torch_out = _generate_torch(_StubHF(), _HFTok(), "abc", 4, 0.0, do_sample=True)
    assert isinstance(torch_out, str)


# ===========================================================================
# BUG 3 — GenerationConfig performs no validation: it silently accepts
#          nonsensical params (negative temperature, top_p > 1, negative
#          top_k, negative max_tokens) that yield silently-wrong sampling.
# ===========================================================================


def test_BUG_generation_config_rejects_invalid_params() -> None:
    """``GenerationConfig`` should reject clearly invalid sampling params.

    A negative temperature silently flips/sharpens the *anti*-distribution
    (categorical(logits / negative_T)); top_p>1, negative top_k and negative
    max_tokens are equally meaningless. For a scientific audience these should
    raise at construction, not silently produce wrong samples.
    """
    with pytest.raises((ValueError, AssertionError)):
        GenerationConfig(temperature=-1.0)
    with pytest.raises((ValueError, AssertionError)):
        GenerationConfig(top_p=5.0)
    with pytest.raises((ValueError, AssertionError)):
        GenerationConfig(top_k=-3)
    with pytest.raises((ValueError, AssertionError)):
        GenerationConfig(max_tokens=-10)


def test_BUG_negative_temperature_should_error_not_flip_distribution() -> None:
    """A negative temperature must error rather than silently flip sampling.

    ``temperature > 0`` is True for negatives, so ``logits / temperature``
    inverts the distribution and the model samples its *least* likely tokens
    with no warning — a silent-wrong-numbers footgun.
    """
    model = _VariedModel()
    with pytest.raises((ValueError, AssertionError)):
        _generate_manual_mlx(model, _Tok(), "abc", 5, -1.0)


# ===========================================================================
# BUG 4 — HistoryEntry.from_dict aliases the caller's nested dicts, so the
#          deserialized entry shares mutable state with the input (mutation of
#          caller-owned input / shared-state corruption).
# ===========================================================================


def test_BUG_history_from_dict_does_not_alias_caller_nested_dicts() -> None:
    """``from_dict`` must copy nested dicts, not alias the caller's input.

    ``to_dict`` correctly copies via ``dict(...)`` but ``from_dict`` stores the
    caller's dict objects by reference. Mutating the resulting entry then
    silently corrupts the source dict (and vice versa).
    """
    src = {"step": 1, "loss_components": {"lm_ce": 0.5}, "val_metrics": {"acc": 0.9}}
    entry = HistoryEntry.from_dict(src)
    entry.loss_components["lm_ce"] = 999.0
    entry.val_metrics["acc"] = -1.0
    assert src["loss_components"]["lm_ce"] == 0.5, "from_dict aliased loss_components"
    assert src["val_metrics"]["acc"] == 0.9, "from_dict aliased val_metrics"


# ===========================================================================
# BUG 5 — best_val_loss / best_val_metric select a NaN entry as "best" when a
#          NaN is present (order-dependent). This silently picks a degenerate
#          checkpoint for early stopping / model selection.
# ===========================================================================


def test_BUG_best_val_loss_ignores_nan() -> None:
    """A NaN val_loss must never be chosen as the best checkpoint.

    ``min(..., key=val_loss)`` with a NaN present is order-dependent and here
    returns the NaN entry as 'best', so checkpoint selection silently picks a
    diverged model.
    """
    h = History()
    h.append(HistoryEntry(step=0, val_loss=float("nan")))
    h.append(HistoryEntry(step=1, val_loss=0.5))
    h.append(HistoryEntry(step=2, val_loss=0.3))
    best = h.best_val_loss()
    assert best is not None
    assert best.step == 2, f"best_val_loss picked step {best.step} (val={best.val_loss})"


def test_BUG_best_val_metric_ignores_nan() -> None:
    """A NaN metric must never be selected as the best metric value."""
    h = History()
    h.append(HistoryEntry(step=0, val_metrics={"acc": float("nan")}))
    h.append(HistoryEntry(step=1, val_metrics={"acc": 0.9}))
    h.append(HistoryEntry(step=2, val_metrics={"acc": 0.95}))
    result = h.best_val_metric("acc", higher_is_better=True)
    assert result is not None
    step, value = result
    assert step == 2 and value == 0.95, f"best_val_metric returned step={step}, value={value}"


# ===========================================================================
# Regression coverage — behaviors verified correct (these should PASS).
# ===========================================================================


def test_greedy_generation_is_deterministic() -> None:
    """Greedy (temperature=0) generation must be identical across two calls."""
    model = _VariedModel()
    tok = _Tok()
    a = _generate_manual_mlx(model, tok, "hello", 8, 0.0)
    b = _generate_manual_mlx(model, tok, "hello", 8, 0.0)
    assert a == b


def test_stream_matches_manual_when_no_guard() -> None:
    """When the repeat guard never fires, stream and manual outputs match."""
    model = _VariedModel()
    tok = _Tok()
    manual = _generate_manual_mlx(model, tok, "abc", 6, 0.0)
    stream = "".join(_generate_stream_mlx(model, tok, "abc", 6, 0.0))
    assert manual == stream


def test_max_tokens_zero_returns_empty_not_crash() -> None:
    """``max_tokens=0`` must yield an empty continuation, not raise."""
    model = _VariedModel()
    tok = _Tok()
    assert _generate_manual_mlx(model, tok, "abc", 0, 0.0) == ""
    assert list(_generate_stream_mlx(model, tok, "abc", 0, 0.0)) == []


def test_probe_inspection_matches_separate_forward() -> None:
    """Per-step probe logits must match a fresh forward on the same prefix."""
    mx.random.seed(3)
    model = _VariedModel()
    tok = _Tok()
    _config_for(model)
    wrapper = Model(model, tok, backend_name="mlx")
    wrapper.attach_probe(
        ProbeConfig(name="p", layers=[0], source="logits", module_config={"out_dim": 2})
    )
    steps = list(wrapper.generate_with_probes("abc", max_tokens=5, temperature=0.0))
    assert steps, "expected at least one generation step"
    toks = list(tok.encode("abc"))
    for step in steps:
        out = wrapper.forward([toks])
        oracle = out.probes["p"].logits[0, -1, :]
        observed = step.probes["p"].logits[0, -1, :]
        assert bool(mx.allclose(oracle, observed).item())
        toks.append(step.token_id)


def test_probe_inspection_does_not_change_tokens() -> None:
    """Attaching a (non-steering) probe must not change generated tokens."""
    mx.random.seed(3)
    model = _VariedModel()
    tok = _Tok()
    _config_for(model)

    plain = Model(model, tok, backend_name="mlx")
    base_ids = [s.token_id for s in plain.generate_with_probes("abc", max_tokens=6)]

    model2 = _VariedModel()
    model2.update(model.parameters())  # identical weights
    _config_for(model2)
    probed = Model(model2, tok, backend_name="mlx")
    probed.attach_probe(
        ProbeConfig(name="p", layers=[0], source="logits", module_config={"out_dim": 2})
    )
    probed_ids = [s.token_id for s in probed.generate_with_probes("abc", max_tokens=6)]
    assert base_ids == probed_ids


def test_stream_then_generate_no_state_leak() -> None:
    """Streaming prompt A must not contaminate a later generation of prompt B."""
    model = _VariedModel()
    tok = _Tok()
    fresh_b = _generate_manual_mlx(model, tok, "xyz", 6, 0.0)
    list(_generate_stream_mlx(model, tok, "abc", 6, 0.0))
    after = _generate_manual_mlx(model, tok, "xyz", 6, 0.0)
    assert fresh_b == after


def test_extract_stop_tokens_pops_caller_kwargs() -> None:
    """``_extract_stop_tokens`` pops its keys (documented behavior)."""
    tok = _Tok()
    kwargs = {"stop_tokens": [5, 6], "temperature": 0.5}
    out = _extract_stop_tokens(tok, kwargs)
    assert out == [5, 6]
    assert "stop_tokens" not in kwargs
    assert kwargs == {"temperature": 0.5}


def test_chat_fallback_without_template_does_not_crash() -> None:
    """``chat`` must format messages even when the tokenizer has no template."""
    model = _VariedModel()
    tok = _Tok()  # no apply_chat_template / chat_template
    out = chat(model, tok, [{"role": "user", "content": "hi"}], max_tokens=4)
    assert isinstance(out, str)


def test_history_roundtrip_preserves_values() -> None:
    """``to_dict``/``from_dict`` must round-trip values faithfully."""
    h = History()
    h.append(HistoryEntry(step=0, train_loss=1.0, loss_components={"a": 0.1}))
    h.append(HistoryEntry(step=1, val_loss=0.3, val_metrics={"acc": 0.9}))
    h2 = History.from_dict(h.to_dict())
    assert len(h2) == 2
    assert h2[0].loss_components == {"a": 0.1}
    assert h2[1].val_metrics == {"acc": 0.9}

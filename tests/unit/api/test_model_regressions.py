"""Regression tests for edge cases (Model generation facade).

- M7: generate_with_probes sets each probe's prompt_len for the run (response
  probes pool the response-only region, matching training) and resets it after.
- m6: plain generate/generate_stream clear attached probes' captured states, so
  no stale pile lingers for a later probe.forward().
- m7: max_repeat=None on the facade DISABLES the repetition guard (previously it
  silently fell back to the backend default, so None could not disable it).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from auto_chasm import Model, ProbeConfig


class _Fixed5(nn.Module):
    """Two-layer MLP that always argmaxes to token 5 (never EOS=0)."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = [nn.Linear(8, 8) for _ in range(2)]
        self.output_proj = nn.Linear(8, 16)

    def __call__(self, x: mx.array, **kwargs: object) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:  # run layers so probe capture hooks fire
            h = nn.gelu(layer(h))
        row = [-100.0] * 16
        row[5] = 100.0  # deterministic argmax = token 5
        return mx.broadcast_to(mx.array(row), (x.shape[0], x.shape[1], 16))


class _Tok:
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]  # prompt is 3 tokens

    def decode(self, ids: list[int], **kwargs: object) -> str:
        return "x"


class _Cfg:
    hidden_size = 8
    num_hidden_layers = 2


def _model() -> Model:
    base = _Fixed5()
    base.config = _Cfg()
    return Model(base, _Tok(), "mlx")


def test_M7_generate_with_probes_sets_and_resets_prompt_len() -> None:
    """generate_with_probes sets probe.prompt_len during the run and clears it after (M7)."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden", granularity="response"))
    probe = m._probes["p"]
    assert probe.prompt_len is None

    gen = m.generate_with_probes("hello", max_tokens=3, temperature=0.0)
    next(gen)  # execute one step: we are now inside the loop
    assert probe.prompt_len == 3  # M7: set to the prompt's token count during generation
    gen.close()  # trigger the finally
    assert probe.prompt_len is None  # reset so a later forward() is unaffected


def test_m6_plain_generate_clears_probe_captures() -> None:
    """Plain generate leaves no accumulated probe captures behind (m6)."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden"))
    m.generate("hi", max_tokens=4, temperature=0.0)
    # Base-model hooks fired every step; without the fix the pile would remain.
    assert m._probes["p"].get_captured_states() == []


def test_m7_max_repeat_none_disables_guard() -> None:
    """max_repeat=None disables the repetition guard; a cap stops the repeat (m7)."""
    m = _model()  # always emits token 5 -> a pure repetition
    disabled = list(m.generate_with_probes("hi", max_tokens=20, temperature=0.0, max_repeat=None))
    assert len(disabled) == 20  # guard OFF -> runs to max_tokens

    capped = list(m.generate_with_probes("hi", max_tokens=20, temperature=0.0, max_repeat=3))
    assert len(capped) == 3  # guard fires after 3 identical tokens

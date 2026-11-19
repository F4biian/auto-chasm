"""Regression: steering must never run through a KV cache.

A KV cache freezes each past token's hidden state (steered as it passed through as
the last position); the full-forward path re-steers only the current last position,
so caching under steering produces wrong output. Two holes are closed here:
(1) the Model guard used setdefault, so a user's use_cache=True re-enabled it;
(2) on MLX, steered generation with no explicit stops went to mlx_lm.generate, which
caches internally and cannot honour use_cache=False.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

import auto_chasm.generation as gen
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


def _steered_model() -> Model:
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden"))
    m.enable_steering("p", config=SteeringConfig(method="custom"), steer_fn=lambda h, hd, lg: h)
    return m


def test_steering_forces_cache_off_over_user_true() -> None:
    """Under steering, an explicit use_cache=True is overridden to False (correctness)."""
    m = _steered_model()
    _, _, kw = m._apply_gen_config(None, 10, 0.0, {"use_cache": True})
    assert kw["use_cache"] is False


def test_steering_defaults_cache_off() -> None:
    m = _steered_model()
    _, _, kw = m._apply_gen_config(None, 10, 0.0, {})
    assert kw["use_cache"] is False


def test_no_steering_does_not_force_cache() -> None:
    m = _steered_model()
    m.disable_steering("p")
    _, _, kw = m._apply_gen_config(None, 10, 0.0, {})
    assert "use_cache" not in kw  # left to the generation default (True)


class _Tok:
    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "x" * len(ids)


def test_mlx_use_cache_false_routes_to_manual_loop(monkeypatch) -> None:  # noqa: ANN001
    """_generate_mlx routes to the full-forward manual loop when caching is off.

    mlx_lm.generate always caches internally, so a steered (use_cache=False) run must
    not reach it — it must go through _generate_manual_mlx (which honours use_cache).
    """
    seen: dict[str, bool] = {"manual": False}
    orig = gen._generate_manual_mlx

    def spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        seen["manual"] = True
        return orig(*args, **kwargs)

    monkeypatch.setattr(gen, "_generate_manual_mlx", spy)
    gen._generate_mlx(_TinyMlp(), _Tok(), "x", 3, 0.0, use_cache=False)
    assert seen["manual"] is True  # full-forward path taken, not mlx_lm.generate

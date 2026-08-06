"""MLX's buffer cache must be bounded when the MLX backend is selected.

Nothing in this library ever released it. MLX retains freed buffers for every
distinct tensor shape it has seen, so a run with variable-length sequences grows
without limit -- and because Metal's unified memory does not appear in `ps` RSS,
neither the process nor an RSS-based watchdog can see it coming. Observed: a
0.5B-parameter run drove a 64 GB machine to 0.4 GB free while RSS read 2.7 GB.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from auto_chasm.backends.mlx_backend import configure_mlx_memory  # noqa: E402


def test_cache_limit_is_applied_by_default() -> None:
    """The default cap must actually reach MLX (set_cache_limit returns the OLD value)."""
    configure_mlx_memory()
    # Setting it again returns what it was, i.e. what configure_mlx_memory set.
    previous = mx.set_cache_limit(2 * 1024**3)
    assert previous > 0, "no cache limit was in force"
    mx.set_cache_limit(previous)  # restore


def test_env_var_overrides_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTO_CHASM_MLX_CACHE_LIMIT_GB", "1")
    configure_mlx_memory()
    previous = mx.set_cache_limit(2 * 1024**3)
    assert previous == 1024**3
    mx.set_cache_limit(previous)


def test_zero_disables_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """0 must mean "no cap" -- someone with a big machine may not want one."""
    monkeypatch.setenv("AUTO_CHASM_MLX_CACHE_LIMIT_GB", "1")
    configure_mlx_memory()
    monkeypatch.setenv("AUTO_CHASM_MLX_CACHE_LIMIT_GB", "0")
    configure_mlx_memory()
    previous = mx.set_cache_limit(2 * 1024**3)
    assert previous == 1024**3, "a 0 setting must leave the previous limit untouched"
    mx.set_cache_limit(previous)


def test_bad_value_warns_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """A memory HINT must never be the thing that kills a training run."""
    monkeypatch.setenv("AUTO_CHASM_MLX_CACHE_LIMIT_GB", "not-a-number")
    configure_mlx_memory()  # must not raise


def test_selecting_the_mlx_backend_applies_it() -> None:
    """The cap must be wired into the one path every MLX run passes through."""
    from auto_chasm.backends.base import Backend

    mx.set_cache_limit(0)  # start from "no cap"
    backend = Backend(force="mlx")
    assert backend.name == "mlx"
    previous = mx.set_cache_limit(2 * 1024**3)
    assert previous > 0, "Backend(force='mlx') did not bound the cache"
    mx.set_cache_limit(previous)

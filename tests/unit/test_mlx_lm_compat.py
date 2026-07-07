"""Compatibility shim so ``import mlx_lm`` works on transformers >= 5.9.

mlx-lm registers a custom tokenizer under a *string* key at import time, which
transformers >= 5.9 rejects (it reads ``key.__module__``). ``load_mlx`` installs
a shim first; these tests reproduce that failure against a stand-in mapping and
assert the shim neutralizes it, independent of the installed transformers version.
"""

from __future__ import annotations

import pytest


def test_shim_neutralizes_string_key_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """A str key registers without raising, while a class key keeps the stock path."""
    pytest.importorskip("transformers")
    from transformers.models.auto import auto_factory

    from auto_chasm.backends import loaders

    class FakeMapping:
        """Mimics transformers >= 5.9: ``register`` reads ``key.__module__``."""

        def __init__(self) -> None:
            self._extra_content: dict[object, object] = {}

        def register(self, key: object, value: object, exist_ok: bool = False) -> None:
            if key.__module__.startswith("transformers."):  # type: ignore[attr-defined]
                return
            self._extra_content[key] = value

    monkeypatch.setattr(auto_factory, "_LazyAutoMapping", FakeMapping)

    # Baseline: the stock register crashes on mlx-lm's string key.
    with pytest.raises(AttributeError):
        FakeMapping().register("NewlineTokenizer", object())

    loaders._ensure_mlx_lm_import_compat()  # patches FakeMapping.register

    inst = FakeMapping()
    inst.register("NewlineTokenizer", "tok")  # must no longer raise
    assert inst._extra_content["NewlineTokenizer"] == "tok"

    class Cfg:
        """Stand-in config class (a real ``type`` key)."""

    inst.register(Cfg, "cfgval")  # a real class still flows through the original path
    assert inst._extra_content[Cfg] == "cfgval"


def test_shim_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling the shim twice does not double-wrap ``register``."""
    pytest.importorskip("transformers")
    from transformers.models.auto import auto_factory

    from auto_chasm.backends import loaders

    class FakeMapping:
        """Minimal stand-in whose ``register`` the shim wraps."""

        def __init__(self) -> None:
            self._extra_content: dict[object, object] = {}

        def register(self, key: object, value: object, exist_ok: bool = False) -> None:
            """Store ``key`` -> ``value`` unconditionally."""
            self._extra_content[key] = value

    monkeypatch.setattr(auto_factory, "_LazyAutoMapping", FakeMapping)
    loaders._ensure_mlx_lm_import_compat()
    first = FakeMapping.register
    loaders._ensure_mlx_lm_import_compat()
    assert FakeMapping.register is first

"""Compatibility shim so ``import mlx_lm`` works on transformers >= 5.9.

mlx-lm registers a custom tokenizer under a *string* key at import time, which
transformers >= 5.9 rejects (it reads ``key.__module__``). Every auto_chasm code
path that imports mlx_lm calls the shim first; these tests reproduce that failure
against a stand-in mapping and assert the shim neutralizes it, independent of the
installed transformers version.
"""

from __future__ import annotations

import pytest


def test_shim_neutralizes_string_key_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """A str key registers without raising, while a class key keeps the stock path."""
    pytest.importorskip("transformers")
    from transformers.models.auto import auto_factory

    from auto_chasm._mlx_compat import ensure_mlx_lm_compat

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

    ensure_mlx_lm_compat()  # patches FakeMapping.register

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

    from auto_chasm._mlx_compat import ensure_mlx_lm_compat

    class FakeMapping:
        """Minimal stand-in whose ``register`` the shim wraps."""

        def __init__(self) -> None:
            self._extra_content: dict[object, object] = {}

        def register(self, key: object, value: object, exist_ok: bool = False) -> None:
            """Store ``key`` -> ``value`` unconditionally."""
            self._extra_content[key] = value

    monkeypatch.setattr(auto_factory, "_LazyAutoMapping", FakeMapping)
    ensure_mlx_lm_compat()
    first = FakeMapping.register
    ensure_mlx_lm_compat()
    assert FakeMapping.register is first


def test_direct_mlx_lm_import_path_does_not_crash() -> None:
    """A code path importing mlx_lm *without* load_mlx (``_lora_module_types``)
    imports cleanly — this is the site that crashed under transformers >= 5.9.
    """
    pytest.importorskip("mlx_lm")
    from auto_chasm.backends.mlx_backend import _lora_module_types

    types = _lora_module_types()  # imports mlx_lm.tuner.* directly
    assert len(types) >= 1
    assert all(isinstance(t, type) for t in types)

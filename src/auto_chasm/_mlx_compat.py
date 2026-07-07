"""Compatibility shim so ``import mlx_lm`` works on transformers >= 5.9.

mlx-lm (through 0.31.3, the latest release) registers a custom ``NewlineTokenizer``
under a *string* key at import time via ``AutoTokenizer.register``. transformers
>= 5.9 tightened the auto-mapping registration to read ``key.__module__``, which
raises ``AttributeError`` on a ``str``. :func:`ensure_mlx_lm_compat` wraps
``_LazyAutoMapping.register`` to store non-class keys directly (exactly the pre-5.9
behaviour), so mlx-lm imports on both older and newer transformers.

Every auto_chasm code path that imports ``mlx_lm`` calls :func:`ensure_mlx_lm_compat`
first. It is idempotent and cheap after the first call, and a harmless no-op once
the two libraries resolve this upstream.
"""

from __future__ import annotations

import contextlib
from typing import Any


def ensure_mlx_lm_compat() -> None:
    """Patch transformers so a subsequent ``import mlx_lm`` cannot crash. Idempotent."""
    try:
        from transformers.models.auto import auto_factory
    except Exception:
        return  # transformers absent or restructured — nothing to patch

    mapping = getattr(auto_factory, "_LazyAutoMapping", None)
    original = getattr(mapping, "register", None)
    if mapping is None or original is None or getattr(original, "_auto_chasm_patched", False):
        return

    def register(self: Any, key: Any, value: Any, exist_ok: bool = False) -> None:
        """Register ``key`` -> ``value``, tolerating mlx-lm's non-class string keys."""
        if isinstance(key, type):  # a real config class → stock path, unchanged
            original(self, key, value, exist_ok=exist_ok)
            return
        # A non-class key (mlx-lm's str) would trip the newer ``key.__module__``
        # guard; store it directly, as transformers < 5.9 did.
        with contextlib.suppress(Exception):  # pragma: no cover - upstream layout change
            self._extra_content[key] = value

    register._auto_chasm_patched = True  # type: ignore[attr-defined]
    mapping.register = register

"""Generation-parameter resolution shared by the ``Model.generate*`` methods."""

from __future__ import annotations

from typing import Any

from auto_chasm.config import GenerationConfig


def apply_gen_config(
    config: GenerationConfig | None,
    max_tokens: int | None,
    temperature: float | None,
    kwargs: dict[str, Any],
) -> tuple[int, float, dict[str, Any]]:
    """Resolve generation params; explicit (non-``None``) args always win.

    An explicit ``max_tokens``/``temperature`` is used verbatim — even when it
    equals a default (``0.0``/``256``).  Only when an arg is ``None`` does the
    value fall back to ``config`` and then to the built-in default.

    Args:
        config: ``GenerationConfig`` overrides, or ``None``.
        max_tokens: Explicit max tokens, or ``None`` to resolve from config.
        temperature: Explicit temperature, or ``None`` to resolve from config.
        kwargs: Generation kwargs to enrich with non-default config fields.

    Returns:
        Tuple of ``(max_tokens, temperature, kwargs)``.
    """
    if max_tokens is None:
        max_tokens = config.max_tokens if config is not None else 256
    if temperature is None:
        temperature = config.temperature if config is not None else 0.0
    if config is not None:
        if config.do_sample and temperature == 0.0:
            temperature = 1e-3
        # (config field, default it must differ from, kwarg name).
        for attr, default, key in (
            ("top_p", 1.0, "top_p"),
            ("top_k", 0, "top_k"),
            ("repetition_penalty", 1.0, "repetition_penalty"),
            ("stop_sequences", [], "stop_sequences"),
            ("num_return_sequences", 1, "num_return_sequences"),
            ("do_sample", False, "do_sample"),
        ):
            value = getattr(config, attr)
            if value != default:
                kwargs.setdefault(key, value)
    return max_tokens, temperature, kwargs

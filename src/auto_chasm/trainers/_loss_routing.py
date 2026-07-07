"""Loss dispatch helpers: which code path a probe loss should take.

Two independent routing decisions live here so ``loss.py`` stays focused and
under the file-length cap:

- :func:`_sequence_level` — per-token vs sequence-level, driven by the probe's
  declared ``granularity`` (reliable metadata) with a tensor-shape fallback only
  for ``custom`` poolers, whose output shape is unknowable in advance.
- :func:`_required_positional_arity` — how many positional arguments a custom
  loss callable *requires*, used to pick the modern ``(probe, target)`` vs the
  legacy ``(logits, target, mask)`` signature.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def _probe_granularities(model: Any) -> dict[str, str]:
    """Best-effort ``{probe_name: granularity}`` from the trainable model.

    Reads the probe heads the trainer wrappers expose (``_probes`` on the torch
    wrapper, ``_probe_captures`` on the MLX ``_TrainableModel``).  Returns an
    empty dict for legacy/custom callables that expose no probe heads, so the
    caller falls back to the shape heuristic.

    Args:
        model: The trainable model (wrapper) or a bare callable.

    Returns:
        A ``{probe_name: granularity}`` mapping (possibly empty).
    """
    probes = getattr(model, "_probes", None)
    if not isinstance(probes, dict):
        probes = getattr(model, "_probe_captures", None)
    if not isinstance(probes, dict):
        return {}
    out: dict[str, str] = {}
    for name, probe in probes.items():
        gran = getattr(getattr(probe, "config", None), "granularity", None)
        if isinstance(gran, str):
            out[name] = gran
    return out


def _sequence_level(probe_logits: Any, n_time: int, granularity: str | None) -> bool:
    """Return whether a probe output is pooled to one prediction per sequence.

    Prefers the probe's declared ``granularity`` (``"response"`` pools the time
    axis; ``"token"``/``"sentence"`` keep it) because a tensor-shape guess
    misroutes a response multi-class probe ``[B, C]`` whenever a batch pads to
    ``T-1 == C``.  For ``"custom"`` poolers (or when no metadata is available)
    it falls back to comparing the candidate time axis against ``n_time``.

    Args:
        probe_logits: The probe output tensor.
        n_time: The number of target tokens (``targets.shape[1]``).
        granularity: The probe's declared granularity, or ``None`` if unknown.

    Returns:
        ``True`` if the output has no per-token time axis matching ``n_time``.
    """
    if granularity == "response":
        return True
    if granularity in ("token", "sentence"):
        return False
    # Custom / unknown pooler: fall back to the tensor-shape heuristic.
    ndim = probe_logits.ndim
    if ndim >= 3:
        return False  # per-token multi-class [B, T-1, C]
    if ndim <= 1:
        return True  # [B] — already one scalar per sequence
    return int(probe_logits.shape[1]) != int(n_time)


def _required_positional_arity(fn: Callable[..., Any]) -> int:
    """Count the positional parameters a callable *requires* (no default value).

    Used to pick a custom loss's signature: the modern ``(probe, target)`` API
    requires 2, the legacy ``(logits, target, mask)`` requires 3.  Counting *all*
    parameters instead misclassifies ``(probe, target, mask=None)`` as legacy and
    trips over ``functools.partial`` bindings; counting only required positionals
    is robust.  Builtins / C callables expose no signature -> assume the modern
    2-arg API (the documented default).

    Args:
        fn: The custom loss callable.

    Returns:
        The number of required positional parameters (``2`` when unknown).
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 2
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return sum(1 for p in params if p.kind in positional and p.default is inspect.Parameter.empty)

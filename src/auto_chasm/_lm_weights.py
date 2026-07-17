"""Per-token LM-loss weight resolution (the reserved ``labels["lm_head"]`` channel).

Helpers used by :func:`auto_chasm.data.build_dataset` to turn role-based
``lm_train_on`` settings and explicit per-message weight specs (char spans,
substrings, regexes, token-id subsequences) into one float weight per token:
``1.0`` = train, ``0.0`` = mask, negative = unlearn (gradient ascent). See
``build_dataset``'s docstring for the user-facing contract and
``auto_chasm.trainers._loss_ce.weighted_lm_ce`` for how the loss consumes it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from auto_chasm.config import LM_HEAD


def _normalize_lm_train_on(lm_train_on: str | Sequence[str]) -> frozenset[str] | None:
    """Normalize ``lm_train_on`` to a role set (``None`` = train on everything).

    Args:
        lm_train_on: ``"all"`` (default — every token trains the LM head, the
            historical behavior), a single role name (e.g. ``"assistant"``), or
            a sequence of role names.

    Returns:
        ``None`` for ``"all"``, else the frozen set of roles whose tokens train.
    """
    if isinstance(lm_train_on, str):
        return None if lm_train_on == "all" else frozenset({lm_train_on})
    return frozenset(lm_train_on)


def _lm_specs_of(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a message's explicit ``labels["lm_head"]`` weight specs (``[]`` if none)."""
    return list(msg.get("labels", {}).get(LM_HEAD) or [])


def _lm_weights_for_message(
    msg: dict[str, Any],
    msg_tokens: list[int],
    token_offsets: list[tuple[int, int]],
    baseline: float,
) -> list[float]:
    """Resolve a message's LM-weight specs into one weight per token.

    Spec forms (each carries a ``"weight"``; mixing forms is fine):

    - ``{"start": i, "end": j, "weight": w}`` — character span (half-open).
    - ``{"text": s, "weight": w}`` — every occurrence of the substring.
    - ``{"regex": r, "weight": w}`` — every regex match.
    - ``{"token_ids": [...], "weight": w}`` — every contiguous occurrence of
      that token-id subsequence (resolved on the tokenized message directly).

    Overlaps aggregate with **min** — the most aggressive intervention wins
    (``-5 < -1 < 0 < 1``: unlearn beats mask beats train). Tokens no spec
    covers keep ``baseline``.

    Args:
        msg: The message dict (for ``content`` + spec validation errors).
        msg_tokens: The message's token ids.
        token_offsets: Per-token character offsets (same encoding as the ids).
        baseline: Weight for uncovered tokens.

    Returns:
        One float weight per token of the message.

    Raises:
        ValueError: On a malformed spec (missing ``weight``, a probe-style
            ``label`` field, or an unknown form).
    """
    specs = _lm_specs_of(msg)
    if not specs:
        return [baseline] * len(msg_tokens)

    content = msg.get("content", "")
    char_spans: list[dict[str, Any]] = []
    token_ranges: list[tuple[int, int, float]] = []
    for spec in specs:
        if "label" in spec and "weight" not in spec:
            raise ValueError(
                f"labels['{LM_HEAD}'] spans carry a 'weight' (1=train, 0=mask, "
                f"negative=unlearn), not a 'label' — got {spec!r}. Probe spans use "
                "'label'; the LM-weight channel uses 'weight'."
            )
        if "weight" not in spec:
            raise ValueError(f"labels['{LM_HEAD}'] span {spec!r} is missing its 'weight'.")
        weight = float(spec["weight"])
        if "start" in spec or "end" in spec:
            char_spans.append({"start": spec["start"], "end": spec["end"], "label": weight})
        elif "text" in spec:
            needle = spec["text"]
            if not needle:
                raise ValueError(f"labels['{LM_HEAD}'] 'text' spec must be non-empty: {spec!r}.")
            pos = content.find(needle)
            while pos >= 0:
                char_spans.append({"start": pos, "end": pos + len(needle), "label": weight})
                pos = content.find(needle, pos + 1)
        elif "regex" in spec:
            for match in re.finditer(spec["regex"], content):
                if match.end() > match.start():  # zero-width matches cover nothing
                    char_spans.append({"start": match.start(), "end": match.end(), "label": weight})
        elif "token_ids" in spec:
            needle_ids = list(spec["token_ids"])
            if not needle_ids:
                raise ValueError(
                    f"labels['{LM_HEAD}'] 'token_ids' spec must be non-empty: {spec!r}."
                )
            k = len(needle_ids)
            for i in range(len(msg_tokens) - k + 1):
                if msg_tokens[i : i + k] == needle_ids:
                    token_ranges.append((i, i + k, weight))
        else:
            raise ValueError(
                f"Unknown labels['{LM_HEAD}'] spec {spec!r}: use start/end, text, "
                "regex, or token_ids (each with a 'weight')."
            )

    from auto_chasm.data import _aggregate_span_labels  # local: avoids circular import

    # Char-form specs -> per-token weights via the shared span aggregator
    # (min = most aggressive wins); uncovered tokens keep the baseline.
    weights = [float(w) for w in _aggregate_span_labels(token_offsets, char_spans, "min", baseline)]
    # Token-id ranges apply on top, with the same min rule.
    for start_idx, end_idx, weight in token_ranges:
        for i in range(start_idx, min(end_idx, len(weights))):
            weights[i] = min(weights[i], weight)
    return weights

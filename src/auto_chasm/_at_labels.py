r"""Positional (``"at"``) label specs: labelling the scaffolding around a message.

A character span addresses a message's content. It cannot address the tokens the chat
template renders *around* that content, ``<|im_end|>\\n<|im_start|>assistant\\n``,
``[/INST]``, ``</s>``, the BOS preamble, because those characters belong to no
message. Yet that is exactly where prompt-level questions live:

- "after this user message, does the model know whether it can answer?": the tokens
  that close the user turn and open the assistant turn, i.e. the last thing the model
  reads before it writes anything;
- "before this answer was generated": the same position, named from the other side.

An ``at`` spec names a site relative to one message instead of a character range::

    {"role": "user", "content": prompt,
     "labels": {"answerable": [{"at": "after", "index": -1, "label": 1}]}}

Sites (every one is a contiguous token range of the rendered conversation):

``"content"``   the message's own text, what a character span covers.
``"prefix"``    scaffolding this message opens with (for message 0 this includes the
                BOS/system preamble).
``"suffix"``    scaffolding that follows this message's content.
``"before"``    everything between the previous message's content and this one's:
                ``suffix`` of message ``i-1`` plus ``prefix`` of message ``i``.
``"after"``     everything between this message's content and the next one's:
                ``suffix`` of message ``i`` plus ``prefix`` of message ``i+1``.
``"message"``   prefix + content + suffix.

``"before"``/``"after"`` exist because templates split that scaffolding differently:
ChatML closes a turn and opens the next in one run of text, Llama-2 writes
``[/INST]``, and a future template may attribute it the other way. Asking for "the
gap between these two turns" is well defined for all of them, while asking for
"message ``i``'s suffix" is only well defined once you know the template. Prefer
``before``/``after`` for cross-model code.

Two optional keys narrow a site:

``"index"``     pick one token of the site (``0`` first, ``-1`` last).
``"token_ids"`` keep only the tokens of occurrences of that id subsequence, which is
                the escape hatch for a template this module cannot reason about:
                pass the ids of the marker you care about (e.g.
                ``tokenizer.encode("<|im_start|>assistant")``) and they are matched
                literally. Applied before ``index``.

Nothing here is model-specific: the sites come from the template's own rendering (see
:mod:`auto_chasm._chat_template`), so they work for any tokenizer whose template
round-trips, and ``token_ids`` covers whatever is left.
"""

from __future__ import annotations

from typing import Any

from auto_chasm.logger import get_logger

logger = get_logger(__name__)

#: Every site an ``at`` spec may name. See the module docstring.
SITES = ("content", "prefix", "suffix", "before", "after", "message")

#: Default site when a spec gives only ``token_ids``: the whole message including its
#: scaffolding, because ids are how users reach the special tokens living there.
_DEFAULT_SITE = "message"


def is_at_spec(spec: Any) -> bool:
    """Whether ``spec`` is a positional spec rather than a character span."""
    return isinstance(spec, dict) and ("at" in spec or "token_ids" in spec)


def validate_at_spec(spec: dict[str, Any], probe_name: str) -> None:
    """Fail loudly on a malformed positional spec.

    Args:
        spec: The spec dict to check.
        probe_name: The probe whose label list holds it (for the message).

    Raises:
        ValueError: If the site is unknown, the spec mixes a character span with a
            site, ``label`` is missing, or ``index``/``token_ids`` are malformed.
    """
    where = f"{probe_name!r} label spec {spec!r}"
    if "start" in spec or "end" in spec:
        raise ValueError(
            f"{where} mixes a character span ('start'/'end') with a positional "
            "key ('at'/'token_ids'). A span addresses the message's content, a "
            "positional spec addresses token sites — use two separate specs."
        )
    site = spec.get("at", _DEFAULT_SITE)
    if site not in SITES:
        raise ValueError(f"{where} has an unknown site at={site!r}; valid: {', '.join(SITES)}.")
    if "label" not in spec:
        raise ValueError(f"{where} has no 'label' key.")
    if "index" in spec and not isinstance(spec["index"], int):
        raise ValueError(f"{where} has a non-integer 'index' ({spec['index']!r}).")
    if "token_ids" in spec:
        ids = spec["token_ids"]
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError(f"{where} has an empty or non-sequence 'token_ids'.")
        if not all(isinstance(i, int) for i in ids):
            raise ValueError(f"{where} has non-integer entries in 'token_ids'.")


def _site_segments(site: str, index: int, n_messages: int) -> list[tuple[int, str]]:
    """The ``(message index, segment key)`` pairs a site covers, in token order."""
    if site == "content":
        return [(index, "content")]
    if site == "prefix":
        return [(index, "pre")]
    if site == "suffix":
        return [(index, "post")]
    if site == "message":
        return [(index, "pre"), (index, "content"), (index, "post")]
    if site == "before":
        return ([(index - 1, "post")] if index > 0 else []) + [(index, "pre")]
    # "after"
    return [(index, "post")] + ([(index + 1, "pre")] if index + 1 < n_messages else [])


def at_positions(
    spec: dict[str, Any],
    index: int,
    layout: list[dict[str, tuple[int, int]]],
    tokens: list[int],
    probe_name: str,
    role: str,
) -> list[int]:
    """Absolute token positions one positional spec selects (``[]`` + a warning if none).

    Args:
        spec: A validated positional spec.
        index: Index of the message the spec is declared on.
        layout: Per message, ``{"pre"/"content"/"post": (start, end)}`` token ranges
            into the rendered conversation.
        tokens: The conversation's token ids (for ``token_ids`` matching).
        probe_name: Probe the spec belongs to (for warnings).
        role: Role of the owning message (for warnings).

    Returns:
        Token positions to label, in ascending order.
    """
    site = spec.get("at", _DEFAULT_SITE)
    positions: list[int] = []
    for msg_index, key in _site_segments(site, index, len(layout)):
        start, end = layout[msg_index][key]
        positions.extend(range(start, end))

    def _empty(reason: str) -> list[int]:
        # Silence here would mean a spec that trains nothing, which is invisible in
        # the data and shows up as a probe that never learns. Say exactly what missed.
        logger.warning(
            "Label spec %r on message %d (role %r) for probe %r selected NO token: %s",
            spec, index, role, probe_name, reason,
        )
        return []

    if not positions:
        return _empty(
            f"site {site!r} is empty here — the chat template renders no scaffolding "
            "there (or chat_template=False, which leaves only 'content')"
        )

    if "token_ids" in spec:
        pattern = list(spec["token_ids"])
        ids = [tokens[p] for p in positions]
        hits: list[int] = []
        for start in range(len(ids) - len(pattern) + 1):
            if ids[start : start + len(pattern)] == pattern:
                hits.extend(positions[start : start + len(pattern)])
        positions = sorted(dict.fromkeys(hits))
        if not positions:
            return _empty(f"token_ids {pattern} do not occur in site {site!r}")

    if "index" in spec:
        at = int(spec["index"])
        if not -len(positions) <= at < len(positions):
            return _empty(f"index {at} is out of range for the {len(positions)} token(s) "
                          f"of site {site!r}")
        positions = [positions[at]]

    return positions

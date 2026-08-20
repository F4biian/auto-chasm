"""Dataset and data-loading utilities.

Handles auto-linking batch columns to probe labels, collation,
and wrapping HuggingFace ``datasets`` objects.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from auto_chasm._data_linking import (  # noqa: F401  (public re-exports)
    JointDataset,
    collate_batches,
    link_columns,
)
from auto_chasm._lm_weights import (
    _lm_specs_of,
    _lm_weights_for_message,
    _normalize_lm_train_on,
)
from auto_chasm.config import LM_HEAD
from auto_chasm.logger import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
"""Sentinel label marking a token as masked: excluded from every probe loss.

This matches PyTorch's ``CrossEntropyLoss(ignore_index=-100)`` convention and is
the value :func:`build_dataset` / :func:`span_labels_to_tokens` assign to any
token you do **not** explicitly label (unless you pass ``default_label=``).
"""


def _label_number(value: Any) -> int | float:
    """Coerce a label to a plain ``int`` (whole number) or ``float`` (fractional).

    Classification labels stay ``int`` class indices; a fractional regression target
    (e.g. ``1.7``) is preserved as a ``float`` instead of being truncated to an int.

    Args:
        value: A user-supplied label (``int``/``float``/numpy scalar).

    Returns:
        An ``int`` when the value is a whole number, else a ``float``.
    """
    number = float(value)
    return int(number) if number.is_integer() else number


def _encode_with_offsets(
    text: str,
    tokenizer: Any,
) -> tuple[list[int] | None, list[tuple[int, int]]]:
    """Tokenize ``text`` once, returning token ids (if available) and offsets.

    Prefers the transformers ``return_offsets_mapping=True`` path so that the
    token ids and the per-token character offsets come from the *same*
    tokenization — this guarantees ``len(ids) == len(offsets)`` and keeps span
    labels aligned to tokens.  Falls back to a character-position heuristic over
    ``tokenizer.encode(text)`` for tokenizers without offset mapping; in that
    fallback the ids are returned too so the caller need not re-tokenize.

    Args:
        text: Input text string.
        tokenizer: A tokenizer object (transformers, MLX wrapper, or
            ``encode``-only).

    Returns:
        Tuple ``(token_ids, token_offsets)``.  ``token_ids`` is ``None`` only
        when the offset-mapping encoding did not expose ``input_ids``; in that
        case the caller should fall back to ``tokenizer.encode(text)``.
    """
    encoding: Any = None
    token_offsets: list[tuple[int, int]] | None = None

    # Try transformers tokenizer callable.
    try:
        encoding = tokenizer(text, return_offsets_mapping=True)
        offset_mapping = encoding.offset_mapping
        if offset_mapping is not None:
            token_offsets = list(offset_mapping)
    except (TypeError, AttributeError, ValueError):
        encoding = None

    # Try underlying _tokenizer (MLX wrapper pattern).
    if token_offsets is None and hasattr(tokenizer, "_tokenizer"):
        try:
            encoding = tokenizer._tokenizer(text, return_offsets_mapping=True)
            offset_mapping = encoding.offset_mapping
            if offset_mapping is not None:
                token_offsets = list(offset_mapping)
        except (TypeError, AttributeError, ValueError):
            encoding = None

    if token_offsets is not None:
        ids = getattr(encoding, "input_ids", None)
        token_ids = list(ids) if ids is not None else None
        return token_ids, token_offsets

    # Fallback heuristic: character-position split over encode() ids.
    token_ids = tokenizer.encode(text)
    n_tokens = len(token_ids)
    text_len = len(text)
    token_offsets = []
    for i in range(n_tokens):
        start = int(i * text_len / n_tokens)
        end = int((i + 1) * text_len / n_tokens) if i < n_tokens - 1 else text_len
        token_offsets.append((start, end))
    return token_ids, token_offsets


def _aggregate_span_labels(
    token_offsets: list[tuple[int, int]],
    spans: list[dict[str, Any]],
    aggregation: str | Callable[..., Any],
    default_label: float,
) -> list[int | float]:
    """Aggregate overlapping span labels into one label per token offset.

    Args:
        token_offsets: Per-token ``(start, end)`` character offsets.
        spans: List of span dicts with ``"start"``, ``"end"``, ``"label"``.
        aggregation: ``"max"``, ``"min"``, ``"mean"``, or a callable.
        default_label: Label assigned when no span overlaps a token.

    Returns:
        List of per-token labels (int or float), one per offset.
    """
    for span in spans:
        start, end = span["start"], span["end"]
        if start < 0 or end < start:
            # Half-open [start, end) with 0 <= start <= end. A negative start still
            # labels from token 0 and an inverted span labels nothing (both emerge
            # from the overlap test below) -- kept for robustness, but warn so a
            # malformed annotation is never silent.
            logger.warning(
                "Malformed label span start=%s, end=%s: expected 0 <= start <= end. "
                "A negative start is clamped to 0; an inverted span labels nothing. "
                "Check your span annotations.",
                start,
                end,
            )
    labels: list[int | float] = []
    covered = 0
    for tok_start, tok_end in token_offsets:
        overlapping = [
            span["label"] for span in spans if span["start"] < tok_end and span["end"] > tok_start
        ]
        # Do not coerce to int: float span labels are valid regression targets.
        if not overlapping:
            labels.append(default_label)
        elif aggregation == "max":
            covered += 1
            labels.append(max(overlapping))
        elif aggregation == "min":
            covered += 1
            labels.append(min(overlapping))
        elif aggregation == "mean":
            covered += 1
            labels.append(sum(overlapping) / len(overlapping))
        elif callable(aggregation):
            covered += 1
            labels.append(aggregation(overlapping))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    # Spans were given but covered no token, and uncovered tokens are masked
    # (default_label is -100): the sample trains on nothing. This is the
    # warmup_chars >= length / empty-text / spans-out-of-range degenerate case.
    if spans and covered == 0 and default_label == IGNORE_INDEX:
        logger.warning(
            "Label spans covered no tokens (e.g. warmup_chars >= text length, empty "
            "text, or spans outside the tokenized range); this sample has zero "
            "supervision."
        )
    return labels


def span_labels_to_tokens(
    text: str,
    tokenizer: Any,
    spans: list[dict[str, Any]],
    aggregation: str | Callable[..., Any] = "max",
    default_label: float | None = None,
) -> list[int | float]:
    """Convert character-level span annotations to per-token labels.

    Tokenizes the text, maps each token's character offsets to
    overlapping spans, and aggregates span labels per token.

    Supports transformers tokenizers (``return_offsets_mapping=True``),
    MLX tokenizer wrappers (accesses ``_tokenizer`` internally), and
    falls back to a character-position heuristic as last resort.

    **Unmarked tokens are masked by default.** A token that no span covers is
    assigned :data:`IGNORE_INDEX` (``-100``), i.e. it is *not* a training target.
    Only tokens you explicitly cover with a span are trained on. To instead
    treat every unmarked token as a concrete class (the common "mark the
    positives, everything else is the negative class ``0``" pattern), pass
    ``default_label=0`` (or any value).

    Args:
        text: Input text string.
        tokenizer: A tokenizer object.
        spans: List of span dicts with ``"start"``, ``"end"``, and
            ``"label"`` keys.
        aggregation: Aggregation strategy for multiple overlapping spans.
            ``"max"``, ``"min"``, ``"mean"``, or a callable
            ``(labels: list) -> float``.
        default_label: Label for tokens no span overlaps. ``None`` (the
            default) masks them with :data:`IGNORE_INDEX` (``-100``); a number
            assigns that label to every unmarked token.

    Returns:
        List of per-token labels (int or float), one per token.
    """
    fill: float = IGNORE_INDEX if default_label is None else default_label
    _token_ids, token_offsets = _encode_with_offsets(text, tokenizer)
    if not token_offsets:
        return []
    return _aggregate_span_labels(token_offsets, spans, aggregation, fill)


def _encode_plain(tokenizer: Any, text: str) -> list[int]:
    """Encode ``text`` WITHOUT letting the tokenizer add its own special tokens.

    The chat template already contains every special token the sequence needs; a
    tokenizer that also prepends BOS would duplicate it.
    """
    if not text:
        return []
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:  # wrappers whose encode() takes no kwargs (e.g. mlx-lm)
        return list(tokenizer.encode(text))


def build_dataset(
    conversations: list[list[dict[str, Any]]],
    tokenizer: Any,
    offset: int = 0,
    aggregation: str | Callable[..., Any] = "max",
    default_label: float | None = None,
    lm_train_on: str | Sequence[str] = "all",
    chat_template: bool | None = None,
    enable_thinking: bool | None = None,
) -> list[dict[str, Any]]:
    """Build a training dataset from conversations with span labels.

    Each conversation is a list of messages with ``"role"``,
    ``"content"``, and optional ``"labels"``.  Labels are dicts
    mapping probe names to lists of character-level spans with
    ``"start"``, ``"end"``, and ``"label"`` keys.

    **Masking — explicit labels only.** Every token you do not explicitly cover
    with a span is masked with :data:`IGNORE_INDEX` (``-100``) and is *not* a
    training target. This holds at two levels:

    - A message with no spans for any probe → all its tokens are ``-100``.
    - A message *with* spans → tokens inside it that no span covers are **also**
      ``-100`` (not silently trained as class ``0``).

    To instead treat unmarked tokens as a concrete class — the common "mark the
    positive spans, everything else is the negative class ``0``" pattern — pass
    ``default_label=0`` (or any value). Then only fully unlabeled messages stay
    masked, and unmarked tokens *inside labeled messages* take ``default_label``.

    **An EMPTY span list declares the probe.** ``{"probe": []}`` means "this probe
    applies here and nothing is positive", so every token takes ``default_label``;
    OMITTING the key is what masks the message. Reading ``[]`` as "unlabeled"
    silently dropped every negative-only example — in a span-annotated corpus,
    exactly the clean ones, leaving a negative class made only of the text
    surrounding positives.

    **Label encoding:** The ``build_dataset`` function passes span
    label values through unchanged.  What the values mean is
    determined by the probe's loss function:

    - ``"bce"`` (binary cross-entropy): labels should be ``0`` or ``1``.
    - ``"mse"`` (mean squared error): labels can be any float (regression).
    - ``"ce"`` (cross-entropy, via custom loss): labels should be class
      indices ``0, 1, 2, ...``.

    **One head or many — per-probe labels.** If every message labels at most one
    probe, each emitted sample carries a single ``labels`` list (shared by every
    attached probe). If any conversation labels **two or more** probes, the
    sample's ``labels`` becomes a ``{probe_name: list}`` dict so each head trains
    on its own independent target — e.g. a hallucination head and a quality head
    from the same text. A message that does not name a probe simply masks that
    probe (``-100``) for its tokens; heads never bleed into one another. The
    ``Trainer`` and :class:`~auto_chasm.JointLoss` route the dict automatically.

    **Per-token LM-loss weights (masking & unlearning).** The reserved label key
    ``"lm_head"`` controls how each token trains the LANGUAGE-MODEL head
    (see ``JointLoss``/``weighted_lm_ce``): weight ``1.0`` = normal training
    (``-log p``), ``0.0`` = masked (like ``-100``), negative = **unlikelihood
    training** (Welleck et al. 2019, arXiv:1908.04319) — ``|w| * -log(1 - p)``,
    which DECREASES the token's likelihood with ``|w|`` as the paper's
    ``alpha`` (e.g. ``-1.0``, or ``-5.0`` for five times the pressure). Two
    ways to declare it, usable TOGETHER (see the composition rule below):

    - ``lm_train_on=`` — role-based baseline: e.g. ``"assistant"`` trains the LM
      only on assistant-message tokens (weight 1) and masks **every other role**
      (weight 0) — including ``system``, not just ``user``. The common chat-SFT
      switch. Pass a sequence to keep more than one role, e.g.
      ``("assistant", "system")``.
    - Explicit per-message specs in ``msg["labels"]["lm_head"]`` — full control:
      ``{"start", "end", "weight"}`` char spans, ``{"text": ..., "weight"}``
      substring occurrences, ``{"regex": ..., "weight"}`` matches, or
      ``{"token_ids": [...], "weight"}`` token subsequences. Overlapping specs
      aggregate with **min** (unlearn beats mask beats train).

    **Composition (the common case).** ``lm_train_on`` sets each message's
    BASELINE weight — ``1.0`` for a named role, ``0.0`` for every other role —
    and explicit specs are then applied ON TOP, OVERRIDING that baseline for
    the tokens they cover. So the typical chat-SFT recipe is simply::

        lm_train_on="assistant"                     # nothing else is trained
        + specs on the assistant message only       # e.g. unlearn a bad span

    A spec can also override in the other direction: a
    ``{"start": 0, "end": 12, "weight": 1.0}`` span on a *user* message trains
    those tokens despite the role baseline of 0. Only the tokens a spec covers
    are overridden; the rest of the message keeps its role baseline. Without
    ``lm_train_on`` the baseline is ``1.0`` everywhere (today's default: no
    automatic user/system masking at all).

    Args:
        conversations: List of conversations.  Each conversation is a
            list of message dicts with keys ``"role"``, ``"content"``,
            and optionally ``"labels"``.
        tokenizer: Tokenizer with an ``encode(text)`` method.
        offset: Shift labels by this many positions.  ``1`` shifts
            right (for next-token prediction), ``-1`` shifts left.
            ``0`` means no shift.  Default ``0``.  Applies to PROBE labels
            only, never to the ``"lm_head"`` weight channel.
        aggregation: Span aggregation strategy (passed to
            :func:`span_labels_to_tokens`).  Default ``"max"``.
        default_label: Label for tokens inside a labeled message that no span
            covers. ``None`` (the default) masks them with :data:`IGNORE_INDEX`
            (``-100``) so only explicitly-marked tokens train; pass ``0`` (or
            any value) to treat unmarked tokens as that class instead.
        chat_template: Render each turn through the tokenizer's chat template
            (role markers, turn delimiters), so training matches what the model
            sees at inference. ``None`` (default) applies it whenever the
            tokenizer HAS a template; ``False`` concatenates raw message text,
            which is the pre-0.3 behaviour and is needed to reproduce datasets
            built before this existed. Character spans stay valid either way: the
            scaffolding is tokenized separately from the content and masked.
        enable_thinking: Reasoning mode for templates that support it —
            ``False`` closes the ``<think>`` block, ``True`` opens it, ``None``
            keeps the template's own default (which is NOT consistent across
            tokenizer wrappers, so prefer passing it explicitly).
        lm_train_on: ``"all"`` (default — every token trains the LM head), a
            role name, or a sequence of role names whose tokens train. EVERY
            other role is LM-masked — ``"assistant"`` masks ``system`` as well
            as ``user``; pass ``("assistant", "system")`` to keep system too.
            Sets the per-message baseline that explicit ``lm_head`` specs then
            override. See above.

    Returns:
        List of ``{"tokens": list[int], "labels": ...}`` dicts ready for the
        ``Trainer``.  If the dataset names **two or more** probes anywhere, EVERY
        sample's ``labels`` is a ``{probe_name: list}`` dict covering all of them
        (``-100`` for probes a given sample doesn't label); if it names **one**
        probe, ``labels`` is a per-token list (a shared stream feeding the head).
        Unspecified positions are ``-100`` (masked); spans supply the rest.
        When an LM-weight channel is active (either mechanism), ``labels`` is
        ALWAYS a dict with the extra ``"lm_head"`` float array — probe labels
        are then keyed by their probe names, so attach probes under the SAME
        names the spans use.

    Raises:
        ValueError: If ``lm_train_on`` is combined with explicit ``"lm_head"``
            specs, or a spec is malformed (see :func:`_lm_weights_for_message`).
    """
    masked = IGNORE_INDEX
    # Fill for unmarked tokens *inside* a labeled message: mask by default
    # (explicit labels only), or the user-chosen class when default_label is set.
    fill: float = masked if default_label is None else default_label
    dataset: list[dict[str, Any]] = []

    # Global probe-name set across the WHOLE dataset (one pre-scan; conversations may
    # be a generator). If two or more probes are named ANYWHERE, every sample emits
    # the full {probe: labels} dict — with -100 rows for probes it does not label — so
    # a sample that labels only one probe can never broadcast its labels onto another
    # head at batch time. A single global probe keeps the plain-list "shared label"
    # contract (from_texts / LayerSweep, where one label stream feeds every head).
    # The reserved "lm_head" key is the per-token LM WEIGHT channel, not a probe.
    conversations = list(conversations)
    global_probes: set[str] = set()
    any_lm_specs = False
    for conversation in conversations:
        for msg in conversation:
            global_probes.update(
                n for n, spans in msg.get("labels", {}).items()
                if spans is not None and n != LM_HEAD
            )
            any_lm_specs = any_lm_specs or bool(_lm_specs_of(msg))
    multi_probe = len(global_probes) >= 2
    global_names = sorted(global_probes)

    lm_roles = _normalize_lm_train_on(lm_train_on)
    emit_lm = lm_roles is not None or any_lm_specs

    from auto_chasm._chat_template import has_chat_template, message_wrappers

    use_template = has_chat_template(tokenizer) if chat_template is None else chat_template
    if use_template and not has_chat_template(tokenizer):
        raise ValueError(
            "chat_template=True but this tokenizer has no chat_template. Pass "
            "chat_template=False, or use a tokenizer that defines one."
        )

    for conversation in conversations:
        all_tokens: list[int] = []
        # Tokenize each message once; remember offsets + spans so every probe's
        # label array is built from the SAME encoding the tokens come from.
        msg_infos: list[tuple[list[int], list[tuple[int, int]], dict[str, Any]]] = []
        conv_probes: set[str] = set()
        lm_weights: list[float] = []
        wrappers: list[tuple[str, str]] = (
            message_wrappers(tokenizer, list(conversation), enable_thinking=enable_thinking)[0]
            if use_template
            else [("", "")] * len(conversation)
        )
        for msg, (prefix, suffix) in zip(conversation, wrappers, strict=True):
            token_ids, token_offsets = _encode_with_offsets(msg["content"], tokenizer)
            msg_tokens = token_ids if token_ids is not None else tokenizer.encode(msg["content"])
            msg_labels_dict = msg.get("labels", {})
            baseline = 1.0 if lm_roles is None or msg.get("role") in lm_roles else 0.0

            # Scaffolding rides along as label-less pseudo-messages, so every probe
            # array masks it automatically and the content keeps its own offsets.
            pre_tokens = _encode_plain(tokenizer, prefix)
            if pre_tokens:
                all_tokens.extend(pre_tokens)
                msg_infos.append((pre_tokens, [], {}))
                if emit_lm:
                    # A turn OPENER is prompt, never a target: even for a role that
                    # trains, the model is GIVEN it rather than writing it.
                    lm_weights.extend([0.0] * len(pre_tokens))

            all_tokens.extend(msg_tokens)
            msg_infos.append((msg_tokens, token_offsets, msg_labels_dict))
            # An empty span list still DECLARES the probe (all-negative message),
            # so it must register here or the probe's array is never built.
            conv_probes.update(
                name for name, spans in msg_labels_dict.items()
                if spans is not None and name != LM_HEAD
            )
            if emit_lm:
                lm_weights.extend(_lm_weights_for_message(msg, msg_tokens, token_offsets, baseline))

            post_tokens = _encode_plain(tokenizer, suffix)
            if post_tokens:
                all_tokens.extend(post_tokens)
                msg_infos.append((post_tokens, [], {}))
                if emit_lm:
                    # The CLOSING tag takes the role's own weight: with
                    # lm_train_on="assistant" the model must learn to emit
                    # <|im_end|> and stop, or generation runs to max_tokens.
                    lm_weights.extend([baseline] * len(post_tokens))

        if emit_lm and lm_roles is not None and all_tokens and not any(lm_weights):
            logger.warning(
                "lm_train_on=%r matched NO message role in a conversation (roles seen: "
                "%s) — its LM weights are all 0, so it trains the LM head on nothing.",
                lm_train_on,
                sorted({str(m.get("role")) for m in conversation}),
            )

        # Whenever labels are emitted as a dict (multi-probe, or the LM-weight
        # channel is active) build EVERY global probe's array — a sample that
        # does not label a probe (including a deliberately EMPTY span list)
        # emits an all-(-100) row for it, so the dict's keys never flicker
        # from sample to sample (a batch of only unlabeled samples would
        # otherwise be missing the probe's key entirely, which per-probe
        # consumers cannot distinguish from "this probe does not exist").
        # Otherwise: just this conversation's single probe, as a plain list.
        probe_names = global_names if (multi_probe or emit_lm) else sorted(conv_probes)

        # Build one full-length label array per probe (independent targets). A
        # message that does not label a probe masks it (-100) for those tokens,
        # so heads never bleed into one another.
        per_probe: dict[str, list[int | float]] = {}
        for name in probe_names:
            seq: list[int | float] = []
            for msg_tokens, token_offsets, msg_labels_dict in msg_infos:
                spans = msg_labels_dict.get(name)
                # ABSENT vs EMPTY are different statements about a message:
                #   name not in labels -> this probe does not apply here -> mask.
                #   name -> []         -> it DOES apply and nothing is positive,
                #                         so every token takes ``fill``.
                # Treating [] as "unlabeled" silently dropped every negative-only
                # example: in a span-annotated corpus that is precisely the clean
                # responses, so the negative class collapsed to whatever sat around
                # the positives, and the model never saw a wholly-clean example.
                if name not in msg_labels_dict:
                    lab = [masked] * len(msg_tokens)
                elif spans and token_offsets:
                    lab = _aggregate_span_labels(token_offsets, spans, aggregation, fill)
                    lab += [masked] * (len(msg_tokens) - len(lab))
                    lab = lab[: len(msg_tokens)]
                else:
                    lab = [fill] * len(msg_tokens)
                seq.extend(lab)
            per_probe[name] = _shift_and_fit(seq, offset, len(all_tokens), masked)

        labels: Any
        if emit_lm:
            # An active LM-weight channel forces dict labels: probe arrays keyed
            # by their probe names (attach probes under those names) + the
            # reserved float "lm_head" weights (offset never applies to it).
            labels = {**per_probe, LM_HEAD: lm_weights}
        elif multi_probe:
            labels = per_probe  # full {probe: labels} dict; -100 for probes not labeled here
        elif conv_probes:
            labels = per_probe[sorted(conv_probes)[0]]  # single global probe: shared list
        else:
            labels = [masked] * len(all_tokens)

        dataset.append({"tokens": all_tokens, "labels": labels})

    return dataset


def _shift_and_fit(
    labels: list[int | float],
    offset: int,
    n_tokens: int,
    masked: int,
) -> list[int | float]:
    """Shift a label sequence for next-token alignment and fit it to ``n_tokens``.

    Args:
        labels: The per-token label sequence (pre-shift).
        offset: Positions to shift — ``>0`` right (next-token), ``<0`` left.
        n_tokens: Target length (the conversation's token count).
        masked: Ignore sentinel used for the gap the shift opens up and any pad.

    Returns:
        A label list of length exactly ``n_tokens``.
    """
    had_labels = any(v != masked for v in labels)
    if offset > 0:
        labels = [masked] * offset + labels[:-offset]
    elif offset < 0:
        labels = labels[-offset:] + [masked] * (-offset)
    labels = labels[:n_tokens]
    labels += [masked] * (n_tokens - len(labels))
    # A shift that pushes the only labeled position off the end silently deletes a
    # sample's supervision (e.g. offset=1 on a response label at the last token).
    if had_labels and all(v == masked for v in labels):
        logger.warning(
            "offset=%d shifted a sample's only labeled position(s) off the end — it now "
            "has NO supervision. Use offset=0 for response/last-token labels.",
            offset,
        )
    return labels


def collapse_response_label(labels: list[Any]) -> list[Any]:
    """Keep ONLY the last labeled token for a response-site sample (warn if none).

    The response char span ``(n-1, n)`` can map to zero tokens (a normalizing
    tokenizer drops a trailing-whitespace last char) or to several (a multi-byte
    final char spans multiple byte-tokens). The intent is one label on the last
    content token: keep the last labeled position, clear earlier ones, and warn when
    the sample ended up with no supervision.

    Args:
        labels: The per-token label list for one response-site sample.

    Returns:
        The list with at most one labeled position (the last).
    """
    out = list(labels)
    labeled = [i for i, v in enumerate(out) if v != IGNORE_INDEX]
    if not labeled:
        logger.warning(
            "A response-site sample produced NO labeled token (the last character's "
            "span mapped to no token — e.g. trailing whitespace on a normalizing "
            "tokenizer). It contributes no supervision."
        )
        return out
    for i in labeled[:-1]:  # a multi-byte final char labeled several tokens -> keep one
        out[i] = IGNORE_INDEX
    return out


def _sample_label_list(sample: Any, probe_name: str | None) -> list[Any]:
    """Extract one sample's per-token label list (handles dict / tuple / dict-labels).

    Args:
        sample: A dataset item — ``{"tokens", "labels"}`` dict or
            ``(tokens, labels)`` tuple; ``labels`` may itself be a
            ``{probe_name: list}`` dict for multi-head data.
        probe_name: Which head's labels to read when ``labels`` is a dict.

    Returns:
        The per-token label list (empty if absent).

    Raises:
        ValueError: If ``labels`` is a per-probe dict but ``probe_name`` is
            ``None`` (ambiguous which head to count).
    """
    if isinstance(sample, (tuple, list)):
        labels = sample[1] if len(sample) > 1 else []
    elif isinstance(sample, dict):
        labels = sample.get("labels", sample.get("binary_labels", []))
    else:
        return []
    if isinstance(labels, dict):
        # The reserved "lm_head" key is the per-token LM WEIGHT channel, never a
        # probe's class labels — exclude it. A dict that then holds exactly ONE
        # probe is unambiguous without probe_name (the common single-probe +
        # lm-weights case).
        probe_keys = [k for k in labels if k != LM_HEAD]
        if probe_name is None:
            if len(probe_keys) == 1:
                return list(labels[probe_keys[0]])
            raise ValueError(
                "This dataset has per-probe (dict) labels; pass probe_name= to "
                "select which head's class distribution to count."
            )
        return list(labels.get(probe_name, []))
    return list(labels)


def label_counts(
    dataset: Any,
    num_classes: int | None = None,
    probe_name: str | None = None,
) -> list[int]:
    """Count per-class label occurrences over a dataset, excluding ``-100``.

    Args:
        dataset: An iterable of ``{"tokens", "labels"}`` samples (e.g. the output
            of :func:`build_dataset`, a list, or a :class:`~auto_chasm.Dataset`).
        num_classes: Number of classes.  ``None`` infers it as ``max label + 1``
            over the (non-``-100``) labels present.
        probe_name: For per-probe (dict) labels, which head to count.

    Returns:
        A list of length ``num_classes`` of integer counts.
    """
    lists = [_sample_label_list(s, probe_name) for s in dataset]
    if num_classes is None:
        max_label = -1
        for lst in lists:
            for lab in lst:
                li = int(lab)
                if li != IGNORE_INDEX and li > max_label:
                    max_label = li
        num_classes = max_label + 1 if max_label >= 0 else 0
    counts = [0] * num_classes
    dropped = 0
    fractional = False
    for lst in lists:
        for lab in lst:
            f = float(lab)
            if f == IGNORE_INDEX:
                continue
            li = int(f)
            if li != f:
                fractional = True  # a non-integer label truncated toward zero
            if 0 <= li < num_classes:
                counts[li] += 1
            else:
                dropped += 1  # a real label outside [0, num_classes): silently lost
    if dropped:
        logger.warning(
            "label_counts: %d label(s) outside [0, %d) were not counted. Check "
            "num_classes, or for stray/negative labels in the data.",
            dropped,
            num_classes,
        )
    if fractional:
        logger.warning(
            "label_counts: non-integer labels were truncated toward zero when counting "
            "classes; class weights may be wrong. Use integer class indices for CE."
        )
    return counts


def balanced_class_weights(
    dataset: Any,
    num_classes: int | None = None,
    probe_name: str | None = None,
) -> list[float]:
    """Inverse-frequency ("balanced") per-class weights over a dataset.

    Returns ``total / (num_classes * max(count_c, 1))`` per class — rare classes
    get weight ``> 1`` and common classes ``< 1`` (mean weight ~1).  This is the
    single canonical "balanced" definition used by ``Dataset.class_weights`` and
    by the trainer when resolving ``class_weights="balanced"``.

    Args:
        dataset: An iterable of ``{"tokens", "labels"}`` samples.
        num_classes: Number of classes.  ``None`` infers ``max label + 1``.
        probe_name: For per-probe (dict) labels, which head to weight.

    Returns:
        A list of per-class float weights.
    """
    counts = label_counts(dataset, num_classes, probe_name)
    n_cls = len(counts)
    total = sum(counts)
    if n_cls == 0 or total == 0:
        return [1.0] * n_cls
    return [total / (n_cls * max(c, 1)) for c in counts]

"""Dataset and data-loading utilities.

Handles auto-linking batch columns to probe labels, collation,
and wrapping HuggingFace ``datasets`` objects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from auto_chasm.config import ProbeConfig
from auto_chasm.logger import get_logger
from auto_chasm.utils import tensor_backend

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


def link_columns(
    batch: dict[str, Any],
    probes: list[ProbeConfig],
    column_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Auto-link dataset columns to probe label keys.

    For each probe, looks for a column named ``{probe_name}_labels``
    (or a custom mapping) in the batch and copies it to ``probe_labels``
    under the probe's name.

    Args:
        batch: A single batch dict from the dataloader.
        probes: List of probe configurations.
        column_map: Optional ``{probe_name: column_name}`` overrides.

    Returns:
        The batch with an added ``"probe_labels"`` dict.
    """
    column_map = column_map or {}
    probe_labels: dict[str, Any] = {}

    for probe in probes:
        col_name = column_map.get(probe.name, f"{probe.name}_labels")
        if col_name in batch:
            probe_labels[probe.name] = batch[col_name]
        else:
            logger.debug(
                "Column %r not found in batch for probe %r; skipping.",
                col_name,
                probe.name,
            )

    batch["probe_labels"] = probe_labels
    return batch


class JointDataset:
    """Wraps a HuggingFace dataset to auto-link columns for joint training.

    Args:
        dataset: The underlying HuggingFace ``Dataset``.
        probes: List of probe configurations.
        column_map: Optional column name overrides.
        tokenizer: Tokenizer for text columns (if pre-tokenization needed).
    """

    def __init__(
        self,
        dataset: Any,
        probes: list[ProbeConfig],
        column_map: dict[str, str] | None = None,
        tokenizer: Any = None,
    ) -> None:
        """Initialize the probe dataset wrapper."""
        self.dataset = dataset
        self.probes = probes
        self.column_map = column_map or {}
        self.tokenizer = tokenizer
        self._validate_probe_columns()

    def _validate_probe_columns(self) -> None:
        """Warn loudly if a probe's label column is absent from the dataset.

        A column missing for *every* sample means the probe would silently
        train on no labels — almost always a typo in the column name.  This
        check fails loud (a warning) rather than letting it pass silently.
        """
        columns: set[str] = set()
        column_names = getattr(self.dataset, "column_names", None)
        if column_names:
            columns = set(column_names)
        elif len(self.dataset) > 0:
            first = self.dataset[0]
            if isinstance(first, dict):
                columns = set(first.keys())
        if not columns:
            return
        for probe in self.probes:
            col = self.column_map.get(probe.name, f"{probe.name}_labels")
            if col not in columns:
                logger.warning(
                    "Probe %r expects label column %r, which is not present in the "
                    "dataset columns %s. This probe would train on NO labels — check "
                    "for a typo or pass column_map={%r: <column>}.",
                    probe.name,
                    col,
                    sorted(columns),
                    probe.name,
                )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single linked sample by index."""
        item = self.dataset[idx]
        return link_columns(item, self.probes, self.column_map)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all linked samples."""
        for i in range(len(self)):
            yield self[i]


def collate_batches(
    batches: list[dict[str, Any]],
    _probes: list[ProbeConfig],
) -> dict[str, Any]:
    """Collate a list of samples into a single batch.

    Stacks tensor fields and groups probe labels under a ``"probe_labels"`` key.

    Args:
        batches: List of individual samples.
        _probes: List of probe configurations (currently unused).

    Returns:
        A single batch dict ready for the training step.
    """
    if not batches:
        return {}

    keys = batches[0].keys()
    result: dict[str, Any] = {}

    for key in keys:
        if key == "probe_labels":
            continue
        values = [b[key] for b in batches]
        result[key] = _stack_backend(values)

    probe_labels: dict[str, list[Any]] = {}
    for b in batches:
        pl = b.get("probe_labels", {})
        for name, label in pl.items():
            probe_labels.setdefault(name, []).append(label)

    result["probe_labels"] = {}
    for name, labels in probe_labels.items():
        result["probe_labels"][name] = _stack_backend(labels)

    return result


def _stack_backend(values: list[Any]) -> Any:
    """Stack a list of tensors along a new leading axis, dispatched by type.

    Dispatch is by the concrete tensor type (never by import availability),
    so a torch batch is not silently routed through MLX on a machine where
    both frameworks are installed.  Non-tensor values are returned as-is.

    Args:
        values: List of same-typed tensors (or arbitrary values).

    Returns:
        A stacked tensor, or the original list if the values are not tensors.
    """
    if not values:
        return values
    if tensor_backend(values[0]) == "torch":
        import torch

        if isinstance(values[0], torch.Tensor):
            return torch.stack(values, dim=0)
        return values
    import mlx.core as mx

    try:
        return mx.stack(values, axis=0)
    except (TypeError, ValueError):
        return values


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


def build_dataset(
    conversations: list[list[dict[str, Any]]],
    tokenizer: Any,
    offset: int = 0,
    aggregation: str | Callable[..., Any] = "max",
    default_label: float | None = None,
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

    Args:
        conversations: List of conversations.  Each conversation is a
            list of message dicts with keys ``"role"``, ``"content"``,
            and optionally ``"labels"``.
        tokenizer: Tokenizer with an ``encode(text)`` method.
        offset: Shift labels by this many positions.  ``1`` shifts
            right (for next-token prediction), ``-1`` shifts left.
            ``0`` means no shift.  Default ``0``.
        aggregation: Span aggregation strategy (passed to
            :func:`span_labels_to_tokens`).  Default ``"max"``.
        default_label: Label for tokens inside a labeled message that no span
            covers. ``None`` (the default) masks them with :data:`IGNORE_INDEX`
            (``-100``) so only explicitly-marked tokens train; pass ``0`` (or
            any value) to treat unmarked tokens as that class instead.

    Returns:
        List of ``{"tokens": list[int], "labels": ...}`` dicts ready for the
        ``Trainer``.  If the dataset names **two or more** probes anywhere, EVERY
        sample's ``labels`` is a ``{probe_name: list}`` dict covering all of them
        (``-100`` for probes a given sample doesn't label); if it names **one**
        probe, ``labels`` is a per-token list (a shared stream feeding the head).
        Unspecified positions are ``-100`` (masked); spans supply the rest.
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
    conversations = list(conversations)
    global_probes: set[str] = set()
    for conversation in conversations:
        for msg in conversation:
            global_probes.update(n for n, spans in msg.get("labels", {}).items() if spans)
    multi_probe = len(global_probes) >= 2
    global_names = sorted(global_probes)

    for conversation in conversations:
        all_tokens: list[int] = []
        # Tokenize each message once; remember offsets + spans so every probe's
        # label array is built from the SAME encoding the tokens come from.
        msg_infos: list[tuple[list[int], list[tuple[int, int]], dict[str, Any]]] = []
        conv_probes: set[str] = set()
        for msg in conversation:
            token_ids, token_offsets = _encode_with_offsets(msg["content"], tokenizer)
            msg_tokens = token_ids if token_ids is not None else tokenizer.encode(msg["content"])
            msg_labels_dict = msg.get("labels", {})
            all_tokens.extend(msg_tokens)
            msg_infos.append((msg_tokens, token_offsets, msg_labels_dict))
            conv_probes.update(name for name, spans in msg_labels_dict.items() if spans)

        # In a multi-probe dataset build EVERY global probe's array (so this sample
        # emits the complete dict, -100 for probes it doesn't label); otherwise just
        # this conversation's single probe.
        probe_names = global_names if multi_probe else sorted(conv_probes)

        # Build one full-length label array per probe (independent targets). A
        # message that does not label a probe masks it (-100) for those tokens,
        # so heads never bleed into one another.
        per_probe: dict[str, list[int | float]] = {}
        for name in probe_names:
            seq: list[int | float] = []
            for msg_tokens, token_offsets, msg_labels_dict in msg_infos:
                spans = msg_labels_dict.get(name)
                if spans and token_offsets:
                    lab = _aggregate_span_labels(token_offsets, spans, aggregation, fill)
                    lab += [masked] * (len(msg_tokens) - len(lab))
                    lab = lab[: len(msg_tokens)]
                else:
                    lab = [masked] * len(msg_tokens)
                seq.extend(lab)
            per_probe[name] = _shift_and_fit(seq, offset, len(all_tokens), masked)

        labels: Any
        if multi_probe:
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
        if probe_name is None:
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

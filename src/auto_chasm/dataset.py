"""OOP dataset front-end for probe training.

:class:`Dataset` is the class-based face of :func:`auto_chasm.data.build_dataset` /
:func:`auto_chasm.data.span_labels_to_tokens`.  It routes through those functions (one
implementation, full back-compat) while giving an object you build with a
classmethod, ``split`` into train/val, derive ``class_weights`` from, and drop
straight into a ``Trainer`` (it is a sequence of ``{"tokens", "labels"}`` dicts).

``from_texts`` builds the character spans for the common label placements so you
never hand-write them:

- ``label_site="response"`` — one label on the last character (whole-text repr).
- ``label_site="token"`` — every character after ``warmup_chars`` (token-level,
  post warm-up).
- ``label_site="sentence"`` — one label at each sentence-ending delimiter
  (requires ``sentence_delimiters``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from auto_chasm._dataset_split import (
    _grouped_stratified_val_indices,
    _sample_stratum,
    _stamp_groups,
)
from auto_chasm.data import (
    IGNORE_INDEX,
    _label_number,
    _sample_label_list,
    balanced_class_weights,
    build_dataset,
    collapse_response_label,
    label_counts,
)
from auto_chasm.logger import get_logger
from auto_chasm.task import Task, TaskSelector

logger = get_logger(__name__)


def _require_hashable(values: Sequence[Any], kind: str) -> None:
    """Raise a friendly ``ValueError`` if any element is unhashable.

    Strata and group keys are used as dict keys, so an unhashable element (a
    list, dict, or ndarray row) would otherwise surface as a raw ``TypeError``
    from deep inside the split. This makes the misuse fail like every other
    bad-argument path: a clear, targeted ``ValueError``.

    Args:
        values: The resolved per-sample strata or group keys.
        kind: ``"stratify"`` or ``"groups"`` (for the message).

    Raises:
        ValueError: If any element is unhashable.
    """
    try:
        for value in values:
            hash(value)
    except TypeError as exc:
        raise ValueError(
            f"{kind} keys must be hashable, but an element is not ({exc}). Use one "
            "scalar key per sample (e.g. int/str/tuple), not a list/array row."
        ) from exc


def _append_eos_to_labels(labels: list[Any], label_site: str) -> list[Any]:
    """Append an EOS slot to a label list, moving a response label onto it.

    For ``label_site="response"`` the single label sits on the last content
    token — whose hidden state the probe never reads, because the loss drops the
    last token from its inputs (next-token alignment). Moving that label onto an
    appended EOS makes the loss supervise the state *after* reading the last
    content token: the whole-text representation. Other sites just gain an
    unlabeled EOS (so the final content token becomes a readable input).

    Args:
        labels: The per-token label list for one sample.
        label_site: The labeling site the list was built for.

    Returns:
        The label list with one EOS slot appended.
    """
    out = list(labels)
    if label_site == "response" and out and out[-1] != IGNORE_INDEX:
        cls_val = out[-1]
        out[-1] = IGNORE_INDEX
        out.append(cls_val)
    else:
        out.append(IGNORE_INDEX)
    return out


def _sentence_end_offsets(text: str, delimiters: Sequence[str]) -> list[int]:
    """Return the char offset of each sentence-ending delimiter occurrence.

    Args:
        text: The text to scan.
        delimiters: Sentence-ending substrings (e.g. ``[".", "!", "?"]``).

    Returns:
        Sorted char offsets of the final character of each delimiter match.
    """
    ends: set[int] = set()
    for delim in delimiters:
        if not delim:
            continue
        start = 0
        while True:
            i = text.find(delim, start)
            if i < 0:
                break
            ends.add(i + len(delim) - 1)
            start = i + 1
    return sorted(ends)


def _spans_for_site(
    text: str,
    cls: float,
    label_site: str,
    warmup_chars: int,
    sentence_delimiters: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Build the character spans that place ``cls`` per ``label_site``.

    Args:
        text: The text being labeled.
        cls: The class index (or a float regression value) to assign.
        label_site: ``"response"``, ``"token"``, or ``"sentence"``.
        warmup_chars: For ``"token"``, the first character labeled.
        sentence_delimiters: For ``"sentence"``, the delimiters to split on.

    Returns:
        A list of ``{"start", "end", "label"}`` span dicts.

    Raises:
        ValueError: If ``label_site`` is unknown, or ``"sentence"`` is requested
            without ``sentence_delimiters``.
    """
    n = len(text)
    if label_site == "response":
        return [{"start": max(n - 1, 0), "end": n, "label": cls}]
    if label_site == "token":
        return [{"start": min(warmup_chars, n), "end": n, "label": cls}]
    if label_site == "sentence":
        if not sentence_delimiters:
            raise ValueError(
                "label_site='sentence' requires sentence_delimiters=[...], e.g. "
                "['.', '!', '?'] — there is no auto-detection."
            )
        return [
            {"start": e, "end": e + 1, "label": cls}
            for e in _sentence_end_offsets(text, sentence_delimiters)
        ]
    raise ValueError(f"Unknown label_site {label_site!r}. Use 'response', 'token', or 'sentence'.")


class Dataset:
    """A probe-training dataset: a sequence of ``{"tokens", "labels"}`` samples.

    Build one with :meth:`from_texts`, :meth:`from_conversations`, or
    :meth:`from_spans`; ``split`` it; derive ``class_weights``; and pass it
    straight to a ``Trainer`` (it supports ``len``, indexing, and iteration).

    Args:
        samples: The per-sample ``{"tokens", "labels"}`` dicts.
        tokenizer: The tokenizer used to build them (kept for reference).
    """

    def __init__(self, samples: Sequence[dict[str, Any]], *, tokenizer: Any = None) -> None:
        """Wrap a list of built samples."""
        self._samples: list[dict[str, Any]] = list(samples)
        self.tokenizer = tokenizer

    @classmethod
    def from_conversations(
        cls,
        conversations: Sequence[Any],
        tokenizer: Any,
        *,
        offset: int = 0,
        aggregation: str = "max",
        default_label: float | None = None,
        groups: Sequence[Any] | None = None,
        lm_train_on: str | Sequence[str] = "all",
        chat_template: bool | None = None,
        enable_thinking: bool | None = None,
    ) -> Dataset:
        """Build from span-annotated conversations (thin wrapper over build_dataset).

        Args:
            conversations: Conversations as accepted by ``build_dataset``.
            tokenizer: Tokenizer with ``encode`` / offset mapping.
            offset: Label shift (see ``build_dataset``).
            aggregation: Span aggregation strategy.
            default_label: Label for unmarked tokens (``None`` masks them).
            groups: Optional group key per conversation, stamped on each sample
                as ``"group"`` so ``split(groups="group")`` keeps a shared prompt
                on one side (e.g. the prompt id when one prompt has several
                answers). Length must equal the number of built samples.
            chat_template: Apply the tokenizer's chat template (role markers,
                turn delimiters) so training matches inference. ``None``
                (default) applies it when the tokenizer has one; ``False``
                concatenates raw message text (pre-0.3 behaviour).
            enable_thinking: Reasoning mode for templates that support it;
                ``None`` keeps the template default, which varies by wrapper.
            lm_train_on: ``"all"`` (default — every token trains the LM head),
                or the role(s) whose tokens train it — e.g. ``"assistant"`` for
                the chat-SFT convention. EVERY other role is LM-masked
                (``system`` as well as ``user``); pass ``("assistant",
                "system")`` to keep system. Composes with explicit ``lm_head``
                specs, which override this baseline where they apply. See
                ``build_dataset`` for the full per-token LM-weight channel
                (masking / unlearning), incl. the explicit-span alternative.

        Returns:
            A :class:`Dataset`.
        """
        samples = build_dataset(
            list(conversations),
            tokenizer,
            offset,
            aggregation,
            default_label,
            lm_train_on=lm_train_on,
            chat_template=chat_template,
            enable_thinking=enable_thinking,
        )
        if groups is not None:
            _stamp_groups(samples, groups)
        return cls(samples, tokenizer=tokenizer)

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        labels: Sequence[float],  # int class indices, or floats for regression
        tokenizer: Any,
        *,
        label_site: str = "response",
        warmup_chars: int = 50,
        sentence_delimiters: Sequence[str] | None = None,
        probe_name: str = "probe",
        offset: int = 0,
        default_label: float | None = None,
        append_eos: bool = False,
        groups: Sequence[Any] | None = None,
    ) -> Dataset:
        """Build from raw texts and one class label each, placing labels by site.

        Args:
            texts: The input texts.
            labels: One class index per text.
            tokenizer: Tokenizer with ``encode`` / offset mapping.
            label_site: Where to place the label — ``"response"`` (last char),
                ``"token"`` (after ``warmup_chars``), or ``"sentence"``.
            warmup_chars: For ``"token"``, the first labeled character.
            sentence_delimiters: For ``"sentence"``, the sentence-ending
                substrings (e.g. ``[".", "!", "?"]``).
            probe_name: The probe name the spans are attached to.
            offset: Label shift passed to ``build_dataset``.
            default_label: Label for unmarked tokens (``None`` masks them).
            append_eos: If ``True``, append the tokenizer's EOS token to every
                sample. For ``label_site="response"`` the label is moved onto the
                EOS so the probe is supervised on the state *after* reading the
                whole text (the true last-token representation) instead of the
                second-to-last token. **Strongly recommended for ``"response"``.**
            groups: Optional group key per text, stamped on each sample as
                ``"group"`` so ``split(groups="group")`` keeps texts that share a
                key on one side (no leakage). Length must equal ``len(texts)``.

        Returns:
            A :class:`Dataset`.
        """
        conversations = [
            [
                {
                    "role": "user",
                    "content": text,
                    "labels": {
                        probe_name: _spans_for_site(
                            text, _label_number(cls), label_site, warmup_chars, sentence_delimiters
                        )
                    },
                }
            ]
            for text, cls in zip(texts, labels, strict=True)
        ]
        dataset = cls.from_conversations(
            conversations, tokenizer, offset=offset, default_label=default_label, groups=groups
        )
        if label_site == "response":
            # Collapse each sample's response label to exactly the last labeled token
            # (a multi-byte / uncovered final char otherwise labels 0 or several).
            for sample in dataset._samples:
                lab = sample["labels"]
                if isinstance(lab, dict):
                    sample["labels"] = {k: collapse_response_label(v) for k, v in lab.items()}
                else:
                    sample["labels"] = collapse_response_label(lab)
        if append_eos:
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is None:
                raise ValueError("append_eos=True requires the tokenizer to define eos_token_id.")
            for sample in dataset._samples:
                sample["tokens"] = list(sample["tokens"]) + [eos]
                lab = sample["labels"]
                if isinstance(lab, dict):
                    sample["labels"] = {
                        k: _append_eos_to_labels(v, label_site) for k, v in lab.items()
                    }
                else:
                    sample["labels"] = _append_eos_to_labels(lab, label_site)
        return dataset

    @classmethod
    def from_spans(
        cls,
        items: Sequence[tuple[str, list[dict[str, Any]]]],
        tokenizer: Any,
        *,
        probe_name: str = "probe",
        offset: int = 0,
        aggregation: str = "max",
        default_label: float | None = None,
        groups: Sequence[Any] | None = None,
    ) -> Dataset:
        """Build from explicit ``(text, spans)`` pairs (the power-user path).

        Args:
            items: ``(text, spans)`` pairs, where ``spans`` is a list of
                ``{"start", "end", "label"}`` dicts.
            tokenizer: Tokenizer with ``encode`` / offset mapping.
            probe_name: The probe name the spans are attached to.
            offset: Label shift passed to ``build_dataset``.
            aggregation: Span aggregation strategy.
            default_label: Label for unmarked tokens (``None`` masks them).
            groups: Optional group key per item, stamped on each sample as
                ``"group"`` for leakage-free grouped splitting. Length must equal
                ``len(items)``.

        Returns:
            A :class:`Dataset`.
        """
        conversations = [
            [{"role": "user", "content": text, "labels": {probe_name: spans}}]
            for text, spans in items
        ]
        return cls.from_conversations(
            conversations,
            tokenizer,
            offset=offset,
            aggregation=aggregation,
            default_label=default_label,
            groups=groups,
        )

    def __len__(self) -> int:
        """Number of samples."""
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample by index."""
        return self._samples[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the samples."""
        return iter(self._samples)

    @property
    def samples(self) -> list[dict[str, Any]]:
        """A copy of the raw ``{"tokens", "labels"}`` sample list."""
        return list(self._samples)

    def split(
        self,
        val_fraction: float = 0.15,
        *,
        seed: int = 42,
        stratify: str | Sequence[Any] | None = None,
        groups: str | Sequence[Any] | None = None,
    ) -> tuple[Dataset, Dataset]:
        """Deterministically split into disjoint ``(train, val)`` datasets.

        The split is always at the **sample** level — and, with ``groups``, at
        the group level — never at the token level, so one text's tokens are
        never divided across the two splits. ``stratify=None, groups=None``
        reproduces the historical plain random split exactly.

        Args:
            val_fraction: Fraction of samples for the validation split.
            seed: Seed for the shuffle (reproducible, disjoint split).
            stratify: Keep class proportions across the splits. ``None``
                (default) does a plain random split; ``"label"`` derives each
                sample's response-level class (its last non-``-100`` label) and
                stratifies on it; a sequence of length ``len(self)`` is used as
                explicit per-sample strata (any hashable key).
            groups: Keep all samples that share a group on the **same** side, so
                a shared prompt cannot leak across the split. ``None`` (default)
                treats every sample independently; ``"group"`` reads each
                sample's stamped ``"group"`` key (see the builders' ``groups=``
                argument); a sequence of length ``len(self)`` is used as explicit
                per-sample group keys. Combine with ``stratify`` for a
                group-pure, class-balanced split (groups win when a group spans
                classes — it is assigned by its majority class).

        Returns:
            ``(train, val)`` as two :class:`Dataset` objects (samples keep their
            original order within each split).

        Raises:
            ValueError: If ``stratify``/``groups`` is an unknown string, a
                sequence whose length differs from ``len(self)``, or
                ``groups="group"`` while a sample lacks a ``"group"`` key.
        """
        strata = self._resolve_strata(stratify)
        group_ids = self._resolve_groups(groups)
        val_idx = _grouped_stratified_val_indices(
            len(self._samples), strata, group_ids, val_fraction, seed
        )
        train = [s for i, s in enumerate(self._samples) if i not in val_idx]
        val = [s for i, s in enumerate(self._samples) if i in val_idx]
        return Dataset(train, tokenizer=self.tokenizer), Dataset(val, tokenizer=self.tokenizer)

    def _resolve_strata(self, stratify: str | Sequence[Any] | None) -> list[Any] | None:
        """Resolve the ``stratify`` argument to per-sample strata (or ``None``)."""
        if stratify is None:
            return None
        if isinstance(stratify, str):
            if stratify != "label":
                raise ValueError(
                    f"Unknown stratify={stratify!r}. Use 'label' (derive the "
                    "response-level class) or a sequence of one stratum per sample."
                )
            return [_sample_stratum(s) for s in self._samples]
        strata = list(stratify)
        if len(strata) != len(self._samples):
            raise ValueError(
                f"stratify has {len(strata)} entries but the dataset has "
                f"{len(self._samples)} samples; provide one stratum per sample."
            )
        _require_hashable(strata, "stratify")
        return strata

    def _resolve_groups(self, groups: str | Sequence[Any] | None) -> list[Any] | None:
        """Resolve the ``groups`` argument to per-sample group keys (or ``None``)."""
        if groups is None:
            return None
        if isinstance(groups, str):
            if groups != "group":
                raise ValueError(
                    f"Unknown groups={groups!r}. Use 'group' (read each sample's "
                    "stamped 'group' key) or a sequence of one group key per sample."
                )
            if any("group" not in s for s in self._samples):
                raise ValueError(
                    "groups='group' requires every sample to carry a 'group' key. "
                    "Stamp it via a builder's groups=[...] argument, or pass an "
                    "explicit groups=[...] sequence to split()."
                )
            stamped = [s["group"] for s in self._samples]
            _require_hashable(stamped, "groups")
            return stamped
        group_ids = list(groups)
        if len(group_ids) != len(self._samples):
            raise ValueError(
                f"groups has {len(group_ids)} entries but the dataset has "
                f"{len(self._samples)} samples; provide one group key per sample."
            )
        _require_hashable(group_ids, "groups")
        return group_ids

    def class_weights(
        self,
        num_classes: int,
        *,
        probe_name: str | None = None,
        scheme: str = "balanced",
    ) -> list[float]:
        """Inverse-frequency ("balanced") per-class weights from this dataset.

        Args:
            num_classes: Number of classes.
            probe_name: For per-probe (dict) labels, which head to weight.
            scheme: Only ``"balanced"`` is supported.

        Returns:
            A list of per-class float weights.

        Raises:
            ValueError: If ``scheme`` is not ``"balanced"``.
        """
        if scheme != "balanced":
            raise ValueError(f"Unknown scheme {scheme!r}. Only 'balanced' is supported.")
        return balanced_class_weights(self._samples, num_classes, probe_name=probe_name)

    def label_counts(
        self, num_classes: int | None = None, *, probe_name: str | None = None
    ) -> list[int]:
        """Per-class token counts over this dataset (excluding ``-100``).

        Args:
            num_classes: Number of classes (``None`` infers ``max label + 1``).
            probe_name: For per-probe (dict) labels, which head to count.

        Returns:
            A list of per-class counts.
        """
        return label_counts(self._samples, num_classes, probe_name=probe_name)

    def infer_task(self, probe_name: str | None = None, *, kind: TaskSelector = "auto") -> Task:
        """Infer (or, for an explicit ``kind``, declare) the :class:`~auto_chasm.Task` for a probe.

        Inspects this dataset's labels for ``probe_name`` and returns the matching
        ``Task`` — the object from which the consistent head width, loss, and
        metrics all follow.

        ``kind="auto"`` reads the label **dtype and range**: floating-point labels
        → ``regression``; integer labels ⊆ {0, 1} → ``binary``; other integer
        labels → ``multiclass`` (``num_classes = max label + 1``).  Pass an
        explicit ``kind`` to override — most usefully ``kind="regression"`` to
        model integer *ordinal* labels (e.g. CEFR A1..C2 as 0..5) as a scalar-MSE
        regression instead of a classifier; the ordinal-bin ``num_classes`` is
        still inferred from the labels.

        Regression ordinal bins are inferred as ``round(max label) + 1``, or
        ``None`` when the labels round to a single value (a genuinely continuous
        target with no natural bins — then :meth:`Task.build_metrics` cannot report
        discretized accuracy). For a continuous target that spans a range (e.g.
        0.0..3.7), the ``round(max)+1`` bins are a rough default; pass an explicit
        ``Task.regression(num_classes=...)`` if you want a specific binning.

        Args:
            probe_name: For per-probe (dict) labels, which head to inspect.
                Required when this dataset carries a ``{probe: labels}`` dict.
            kind: ``"auto"`` (default), ``"classification"`` (binary-or-multiclass
                chosen by the label range), or one of ``"binary"`` /
                ``"multiclass"`` / ``"regression"``.

        Returns:
            The inferred or declared :class:`~auto_chasm.Task`.

        Raises:
            ValueError: If the dataset has no labeled (non-``-100``) positions for
                the probe, or ``kind`` is unknown.
        """
        saw_float = False
        saw_value = False
        max_label = -1
        min_label: int | None = None
        only_binary = True
        for sample in self._samples:
            lst = _sample_label_list(sample, probe_name)
            if not lst:
                continue
            arr = np.asarray(lst)
            if arr.dtype.kind == "f":
                saw_float = True
            for lab in arr.tolist():
                if float(lab) == IGNORE_INDEX:
                    continue
                saw_value = True
                value = int(round(float(lab)))
                max_label = max(max_label, value)
                min_label = value if min_label is None else min(min_label, value)
                if value not in (0, 1):
                    only_binary = False
        if not saw_value:
            suffix = f" for probe {probe_name!r}" if probe_name is not None else ""
            raise ValueError(
                f"Cannot infer a Task: this dataset has no labeled (non-{IGNORE_INDEX}) "
                f"positions{suffix}."
            )
        # Ordinal-bin count for a regression task (a continuous target — max 0 — has none).
        ordinal_bins = max_label + 1 if max_label >= 1 else None
        if kind == "regression" or (kind == "auto" and saw_float):
            return Task.regression(ordinal_bins)
        # Classification labels are class INDICES: a negative value (that is not the
        # -100 sentinel) can never index a head, and would otherwise surface as a
        # misleading "num_classes >= 2, got 0" or a later CE index crash. Reject it now.
        if min_label is not None and min_label < 0:
            raise ValueError(
                f"Cannot infer a classification Task: found a negative class label "
                f"({min_label}). Class labels must be non-negative indices "
                f"0..num_classes-1; use {IGNORE_INDEX} to ignore a position, or "
                f"kind='regression' for signed continuous targets."
            )
        if kind in ("auto", "classification"):
            return Task.binary() if only_binary else Task.multiclass(max_label + 1)
        if kind == "binary":
            return Task.binary()
        if kind == "multiclass":
            return Task.multiclass(max_label + 1)
        raise ValueError(
            f"Unknown task kind selector {kind!r}. Use 'auto', 'classification', "
            f"'binary', 'multiclass', or 'regression'."
        )

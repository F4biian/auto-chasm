"""Data utilities — batching, padding, iteration.

Follows the proven ``make_joint_iterate_batches`` pattern from
``test_joint_sft.py``: sort by length, pad to 32-boundary, yield
``(tokens, labels, lengths)`` triples.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from auto_chasm.logger import get_logger

logger = get_logger(__name__)

# Loss/probe ignore sentinel: positions with this label are masked out and
# do not contribute to any probe loss.  Must match build_dataset() in data.py.
_IGNORE_INDEX = -100


def iterate_batches(
    dataset: Any,
    batch_size: int,
    max_seq_length: int,
    loop: bool = False,
    token_pad: int = 0,
    label_pad: int = -100,
    seed: int | None = None,
) -> Iterator[tuple[Any, Any, Any]]:
    """Iterate over a dataset yielding ``(tokens, labels, lengths)`` batches.

    Follows the exact pattern from ``make_joint_iterate_batches`` in
    ``test_joint_sft.py``: sort by sequence length, pad to 32-boundary,
    yield numpy arrays for backend-agnostic interop.

    Args:
        dataset: Dataset where each item is ``(tokens_list, labels_list)``
            or a dict with ``"tokens"`` and ``"labels"`` keys.
        batch_size: Batch size.
        max_seq_length: Maximum sequence length.
        loop: If ``True``, loop forever.
        token_pad: Padding value for tokens.
        label_pad: Padding value for labels.
        seed: Random seed for shuffling.

    Yields:
        Tuple of ``(tokens, labels, lengths)`` as numpy arrays.
    """

    def _get_item(idx: int) -> tuple[list[int], list[int]]:
        item = dataset[idx]
        if isinstance(item, (tuple, list)):
            return item[0], item[1]
        elif isinstance(item, dict):
            tokens = item.get("tokens", item.get("input_ids", []))
            labels = item.get("labels", item.get("binary_labels", []))
            return tokens, labels  # type: ignore[return-value]
        return item, []

    def _len_fn(idx: int) -> int:
        item = dataset[idx]
        if isinstance(item, (tuple, list)):
            return len(item[0])
        elif isinstance(item, dict):
            return len(item.get("tokens", item.get("input_ids", [])))  # type: ignore[arg-type]
        return 0

    idx = sorted(range(len(dataset)), key=_len_fn)
    effective_batch_size = min(batch_size, len(dataset))

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    # Cover ALL samples, including the trailing remainder.  The final batch may
    # be smaller than ``effective_batch_size``; arrays below are sized by the
    # ACTUAL current batch length so no sample is silently dropped.
    batch_idx = [
        idx[i : i + effective_batch_size] for i in range(0, len(idx), effective_batch_size)
    ]

    # A LOCAL generator: seed=0 must be honored (the old ``if seed:`` treated 0 as
    # "no seed"), and seeding must not stomp the user's global ``np.random`` stream.
    # ``default_rng(None)`` draws fresh entropy for an unseeded shuffle.
    rng = np.random.default_rng(seed)

    while True:
        indices = rng.permutation(len(batch_idx))
        for i in indices:
            batch = [_get_item(j) for j in batch_idx[i]]
            tokens_list, labels_list = zip(*batch, strict=False)
            lengths = [len(t) for t in tokens_list]
            # The final batch may be a partial remainder; size every array by the
            # ACTUAL number of samples in this batch, not effective_batch_size.
            current_batch_size = len(batch)

            if max(lengths) > max_seq_length:
                logger.warning(
                    "Some sequences exceed max_seq_length=%d (longest=%d) and are "
                    "truncated; any labels on tokens past the cutoff are dropped. If a "
                    "sample's only label sits beyond it, that sample trains on nothing "
                    "— raise max_seq_length.",
                    max_seq_length,
                    max(lengths),
                )

            pad_to = 32
            max_length_in_batch = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            truncated = [min(lengths[j], max_seq_length) for j in range(current_batch_size)]

            token_arr = np.full((current_batch_size, max_length_in_batch), token_pad, np.int32)
            for j in range(current_batch_size):
                token_arr[j, : truncated[j]] = tokens_list[j][: truncated[j]]

            # Labels may be a single per-token array (shared by every probe) or a
            # ``{probe_name: labels}`` dict (independent per-probe targets).
            label_out = _build_label_output(
                labels_list, truncated, current_batch_size, max_length_in_batch, label_pad
            )

            offsets = [0] * current_batch_size
            yield (
                token_arr,
                label_out,
                np.array(list(zip(offsets, truncated, strict=False))),
            )

        if not loop:
            break


def _infer_label_dtype(per_sample: Any) -> Any:
    """Infer the numpy dtype for a label matrix from the values present.

    ``np.float32`` if any non-empty label list holds floats; ``np.int32`` if
    concrete integer values are present; ``np.float32`` when there are **no**
    concrete values at all (a fully empty/all-masked batch).

    An empty list is skipped for the float/int decision (``np.asarray([]).dtype``
    is float64, which would otherwise wrongly promote an integer classification
    task to float). But when EVERY list is empty there is nothing to
    disambiguate, so we default to ``float32`` — a probe's label dtype then stays
    **stable across batches** (an all-masked batch must not flip a regression
    probe from ``float32`` to ``int32``). This is harmless: the loss re-casts to
    the dtype it needs (``int`` for cross-entropy), and an all-masked batch is
    entirely ``-100`` regardless of dtype.

    Args:
        per_sample: Iterable of per-sample label lists.

    Returns:
        The numpy dtype to allocate the label matrix with.
    """
    saw_value = False
    for lab in per_sample:
        if len(lab):
            saw_value = True
            if np.asarray(lab).dtype.kind == "f":
                return np.float32
    return np.int32 if saw_value else np.float32


def _pad_label_matrix(
    per_sample: Any,
    truncated: list[int],
    batch_size: int,
    max_length: int,
    label_pad: int,
) -> Any:
    """Pad a list of per-sample label lists into one ``[B, T]`` matrix.

    Short label lists are padded with ``label_pad`` (the ignore sentinel) and
    long ones truncated to the per-sample token length, so a sample never trains
    a probe on phantom positions.

    Args:
        per_sample: List (length ``batch_size``) of per-token label lists.
        truncated: Per-sample valid token length (already capped).
        batch_size: Number of samples in this batch.
        max_length: Padded sequence length of the batch.
        label_pad: Ignore sentinel for padding.

    Returns:
        A ``[batch_size, max_length]`` numpy array.
    """
    arr = np.full((batch_size, max_length), label_pad, _infer_label_dtype(per_sample))
    for j in range(batch_size):
        tl = min(truncated[j], len(per_sample[j]))
        if tl:
            arr[j, :tl] = per_sample[j][:tl]
    return arr


def _build_label_output(
    labels_list: Any,
    truncated: list[int],
    batch_size: int,
    max_length: int,
    label_pad: int,
) -> Any:
    """Build the batch label output: one ``[B, T]`` array, or a per-probe dict.

    If any sample carries a ``{probe_name: labels}`` dict, every probe seen
    across the batch gets its own padded ``[B, T]`` matrix; a sample that does
    not name a probe contributes an all-``label_pad`` (masked) row for it.

    Args:
        labels_list: Per-sample labels — each a list, or a ``{name: list}`` dict.
        truncated: Per-sample valid token length.
        batch_size: Number of samples in this batch.
        max_length: Padded sequence length of the batch.
        label_pad: Ignore sentinel for padding.

    Returns:
        A ``[B, T]`` numpy array, or ``{probe_name: [B, T] array}``.
    """
    if any(isinstance(lab, dict) for lab in labels_list):
        names = sorted({k for lab in labels_list if isinstance(lab, dict) for k in lab})
        out: dict[str, Any] = {}
        for name in names:
            # A plain-list sample (a conversation that labeled <=1 probe) is a
            # SHARED target applied to every head (build_dataset's documented
            # single-list contract). Broadcast it to each probe rather than
            # dropping it to an all-(-100) row, which silently discarded its
            # supervision when batched alongside a per-probe dict sample.
            per = [lab.get(name, []) if isinstance(lab, dict) else lab for lab in labels_list]
            out[name] = _pad_label_matrix(per, truncated, batch_size, max_length, label_pad)
        return out
    return _pad_label_matrix(labels_list, truncated, batch_size, max_length, label_pad)


def labels_to_mlx(labels: Any) -> Any:
    """Convert a numpy label array — or a per-probe dict of them — to MLX arrays.

    Args:
        labels: A numpy array or a ``{probe_name: numpy array}`` dict.

    Returns:
        An ``mx.array`` or a ``{probe_name: mx.array}`` dict (same shape).
    """
    import mlx.core as mx

    if isinstance(labels, dict):
        return {k: mx.array(v) for k, v in labels.items()}
    return mx.array(labels)


def labels_to_torch(labels: Any, device: Any) -> Any:
    """Convert a numpy label array — or a per-probe dict of them — to torch tensors.

    Args:
        labels: A numpy array or a ``{probe_name: numpy array}`` dict.
        device: The torch device to place tensors on.

    Returns:
        A ``torch.Tensor`` or a ``{probe_name: torch.Tensor}`` dict.
    """
    import torch

    if isinstance(labels, dict):
        return {k: torch.from_numpy(v).to(device) for k, v in labels.items()}
    return torch.from_numpy(labels).to(device)


def _pad_labels_to_len(labels: list[Any], n_tokens: int) -> list[Any]:
    """Pad or truncate a per-token label list to ``n_tokens``.

    Short lists are padded with the ignore sentinel ``-100`` (e.g. for an
    appended EOS or trailing tokens the user did not label), NOT a valid class
    like ``0`` — otherwise the probe trains on phantom negatives. Matches
    ``build_dataset()`` in ``data.py``.

    Args:
        labels: The per-token labels for one sample.
        n_tokens: The target length (the sample's token count).

    Returns:
        A list of length ``n_tokens``.
    """
    if len(labels) < n_tokens:
        return labels + [_IGNORE_INDEX] * (n_tokens - len(labels))
    return labels[:n_tokens]


def route_probe_labels(
    sample: dict[str, Any],
    probe_names: list[str],
) -> dict[str, list[int]]:
    """Auto-route ``{probe_name}_labels`` columns to per-probe label arrays.

    Args:
        sample: Dataset sample dict with fields like ``'text'``,
            ``'tokens'``, ``'digit_labels'``, ``'hallucination_labels'``,
            etc.
        probe_names: List of probe names to look for.

    Returns:
        Dict mapping ``probe_name`` → ``label_list``.
    """
    result: dict[str, list[int]] = {}
    for name in probe_names:
        col = f"{name}_labels"
        if col in sample:
            result[name] = list(sample[col])
    return result


class JointTextDataset:
    """Dataset that carries per-token binary labels.

    Follows the proven ``JointTextDataset`` from ``test_joint_sft.py``.
    Supports pre-tokenized data via ``tokens_key`` and auto-routing of
    per-probe labels via ``probe_names``.

    Args:
        data: List of sample dicts.
        tokenizer: Tokenizer (used if ``tokens_key`` is not set).
        text_key: Key for text data.
        labels_key: Key for label data.
        tokens_key: Key for pre-tokenized data.  If ``None``, tokenizes on the fly.
        probe_names: Optional list of probe names for auto-routing
            ``{name}_labels`` columns.  When set, per-probe labels are
            available via the sample dict.
        backend: Backend name (``"mlx"`` or ``"torch"``).  Stored for use by
            ``iterate_batches`` but does not affect dataset return type.
    """

    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenizer: Any,
        text_key: str = "text",
        labels_key: str = "labels",
        tokens_key: str | None = None,
        probe_names: list[str] | None = None,
        backend: str = "mlx",
    ) -> None:
        """Initialize the tokenized dataset."""
        self._data = data
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.labels_key = labels_key
        self.tokens_key = tokens_key
        self.probe_names = probe_names
        self.backend = backend
        self._validate_keys()

    def _validate_keys(self) -> None:
        """Warn loudly if ``labels_key``/``tokens_key`` is absent for most samples.

        A labels column missing from every (or nearly every) sample means the
        probe would silently train on all-``-100`` labels — i.e. on nothing —
        which is almost always a typo in ``labels_key``.  Mirrors the loud-warning
        behavior of :class:`auto_chasm.data.JointDataset`; warn once rather than
        letting the mistake pass silently.
        """
        if not self._data:
            return
        n = len(self._data)
        missing_labels = sum(
            1 for d in self._data if not isinstance(d, dict) or self.labels_key not in d
        )
        if missing_labels > n // 2:
            logger.warning(
                "JointTextDataset: labels_key %r is absent from %d of %d samples. "
                "Those samples train the probe on all-(-100) labels — i.e. on "
                "NOTHING. Check for a typo or pass labels_key=<column>. "
                "Available keys on the first sample: %s.",
                self.labels_key,
                missing_labels,
                n,
                sorted(self._data[0].keys()) if isinstance(self._data[0], dict) else "n/a",
            )
        if self.tokens_key is not None:
            missing_tokens = sum(
                1 for d in self._data if not isinstance(d, dict) or self.tokens_key not in d
            )
            if missing_tokens > n // 2:
                logger.warning(
                    "JointTextDataset: tokens_key %r is absent from %d of %d samples; "
                    "those samples fall back to on-the-fly tokenization of %r. "
                    "Check for a typo or pass the correct tokens_key.",
                    self.tokens_key,
                    missing_tokens,
                    n,
                    self.text_key,
                )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[list[int], Any]:
        """Return tokenized input and label sequences.

        ``labels`` is a per-token list, or a ``{probe_name: list}`` dict for
        independent per-probe targets; each array is padded/truncated to the
        token length with the ignore sentinel ``-100``.
        """
        d = self._data[idx]
        if self.tokens_key and self.tokens_key in d:
            tokens = list(d[self.tokens_key])
        else:
            tokens = self.tokenizer.encode(d[self.text_key])
        if tokens and tokens[-1] != self.tokenizer.eos_token_id:
            tokens.append(self.tokenizer.eos_token_id)

        raw = d.get(self.labels_key, [])
        if isinstance(raw, dict):
            labels: Any = {k: _pad_labels_to_len(list(v), len(tokens)) for k, v in raw.items()}
        else:
            labels = _pad_labels_to_len(list(raw), len(tokens))
        return tokens, labels

"""HF-dataset column linking + collation (re-exported via ``auto_chasm.data``)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from auto_chasm.config import ProbeConfig
from auto_chasm.logger import get_logger
from auto_chasm.utils import tensor_backend

logger = get_logger(__name__)


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

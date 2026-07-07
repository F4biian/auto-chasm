"""Training history — structured metric tracking across training steps.

Provides ``HistoryEntry`` (one logged snapshot) and ``History`` (the full
collection with convenience accessors).  Loss components are stored as
a dynamic ``dict[str, float]`` so any probe configuration works —
binary classification (BCE), regression (MSE), multi-class, etc.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HistoryEntry:
    """A single logged snapshot during training.

    Attributes:
        step: Global training step.
        train_loss: Total training loss for this logging window.
        loss_components: Named loss components (e.g.,
            ``{"lm_ce": 0.5, "probe_bce": 0.3}`` or ``{"probe_mse": 0.2}``).
            Keys are user-defined; the library never hardcodes component names.
        val_loss: Total validation loss (``None`` if no eval this step).
        val_metrics: Validation metrics dict (accuracy, f1, perplexity, etc.).
        test_loss: Total test loss (populated once after training).
        test_metrics: Test metrics dict.
        learning_rate: Current learning rate.
        it_sec: Training iterations per second.
        tokens_sec: Training tokens per second.
        wall_time: Seconds since training start.
        custom: Arbitrary user-defined metrics (populated via callbacks).
    """

    step: int
    train_loss: float | None = None
    loss_components: dict[str, float] = field(default_factory=dict)
    val_loss: float | None = None
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_loss: float | None = None
    test_metrics: dict[str, float] = field(default_factory=dict)
    learning_rate: float | None = None
    it_sec: float | None = None
    tokens_sec: float | None = None
    wall_time: float | None = None
    custom: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Returns:
            Dict with all fields, suitable for ``json.dumps``.
        """
        return {
            "step": self.step,
            "train_loss": self.train_loss,
            "loss_components": dict(self.loss_components),
            "val_loss": self.val_loss,
            "val_metrics": dict(self.val_metrics),
            "test_loss": self.test_loss,
            "test_metrics": dict(self.test_metrics),
            "learning_rate": self.learning_rate,
            "it_sec": self.it_sec,
            "tokens_sec": self.tokens_sec,
            "wall_time": self.wall_time,
            "custom": dict(self.custom),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HistoryEntry:
        """Deserialize from a dict.

        Args:
            d: Dict as produced by ``to_dict``.

        Returns:
            A ``HistoryEntry`` instance.
        """
        return cls(
            step=d["step"],
            train_loss=d.get("train_loss"),
            loss_components=dict(d.get("loss_components", {})),
            val_loss=d.get("val_loss"),
            val_metrics=dict(d.get("val_metrics", {})),
            test_loss=d.get("test_loss"),
            test_metrics=dict(d.get("test_metrics", {})),
            learning_rate=d.get("learning_rate"),
            it_sec=d.get("it_sec"),
            tokens_sec=d.get("tokens_sec"),
            wall_time=d.get("wall_time"),
            custom=dict(d.get("custom", {})),
        )


class History:
    """Collection of ``HistoryEntry`` objects with convenience accessors.

    Supports iteration, indexing, and length.  Provides helpers for
    extracting time-series of any metric.
    """

    def __init__(self) -> None:
        """Initialize an empty history."""
        self._entries: list[HistoryEntry] = []

    def append(self, entry: HistoryEntry) -> None:
        """Add an entry to the history.

        Args:
            entry: The history entry to add.
        """
        self._entries.append(entry)

    @property
    def entries(self) -> list[HistoryEntry]:
        """All history entries."""
        return list(self._entries)

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self._entries)

    def __iter__(self) -> Any:
        """Iterate over entries."""
        return iter(self._entries)

    def __getitem__(self, idx: int) -> HistoryEntry:
        """Get entry by index."""
        return self._entries[idx]

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"History({len(self._entries)} entries)"

    # ---- Time-series extractors ----

    @property
    def steps(self) -> list[int]:
        """List of step numbers."""
        return [e.step for e in self._entries]

    @property
    def train_losses(self) -> list[float]:
        """List of training losses (``None`` entries skipped)."""
        return [e.train_loss for e in self._entries if e.train_loss is not None]

    @property
    def val_losses(self) -> list[float]:
        """List of validation losses (``None`` entries skipped)."""
        return [e.val_loss for e in self._entries if e.val_loss is not None]

    @property
    def val_steps(self) -> list[int]:
        """Step numbers where validation was performed."""
        return [e.step for e in self._entries if e.val_loss is not None]

    def component_series(self, name: str) -> list[float]:
        """Extract a time-series for a specific loss component.

        Args:
            name: Component name (e.g., ``"lm_ce"``, ``"probe_bce"``).

        Returns:
            List of values for entries that have this component.
        """
        return [e.loss_components[name] for e in self._entries if name in e.loss_components]

    def metric_series(self, name: str) -> list[float]:
        """Extract a time-series for a specific validation metric.

        Args:
            name: Metric name (e.g., ``"accuracy"``, ``"f1"``, ``"perplexity"``).

        Returns:
            List of values for entries that have this metric.
        """
        return [e.val_metrics[name] for e in self._entries if name in e.val_metrics]

    def last(self) -> HistoryEntry | None:
        """Return the last entry, or ``None`` if empty."""
        return self._entries[-1] if self._entries else None

    def best_val_loss(self) -> HistoryEntry | None:
        """Return the entry with the best (lowest) validation loss.

        Entries with a non-finite ``val_loss`` (``NaN``/``inf``) are ignored:
        ``min`` with a ``NaN`` present is order-dependent and could otherwise
        select a diverged checkpoint as "best".

        Returns:
            ``HistoryEntry`` with the lowest finite ``val_loss``, or ``None`` if
            no validation was done or every ``val_loss`` is non-finite.
        """

        def _val_loss(entry: HistoryEntry) -> float:
            assert entry.val_loss is not None
            return entry.val_loss

        val_entries = [
            e for e in self._entries if e.val_loss is not None and math.isfinite(e.val_loss)
        ]
        if not val_entries:
            return None
        return min(val_entries, key=_val_loss)

    def best_val_metric(
        self,
        name: str,
        higher_is_better: bool = False,
    ) -> tuple[int, float] | None:
        """Return ``(step, value)`` of the best validation metric.

        Args:
            name: Metric name.
            higher_is_better: If ``True``, maximize; otherwise minimize.

        Entries with a non-finite metric value (``NaN``/``inf``) are ignored,
        since ``min``/``max`` with a ``NaN`` present is order-dependent and
        could otherwise report a diverged step as best.

        Returns:
            Tuple of (step, value) or ``None`` if the metric is absent from
            every entry or all of its values are non-finite.
        """
        entries = [
            e for e in self._entries if name in e.val_metrics and math.isfinite(e.val_metrics[name])
        ]
        if not entries:
            return None
        if higher_is_better:
            best = max(entries, key=lambda e: e.val_metrics[name])
        else:
            best = min(entries, key=lambda e: e.val_metrics[name])
        return (best.step, best.val_metrics[name])

    # ---- JSON serialization ----

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full history to a JSON-compatible dict.

        Returns:
            Dict with ``"entries"`` list, suitable for ``json.dumps``.
        """
        return {"entries": [e.to_dict() for e in self._entries]}

    def save_json(self, path: str | Path) -> None:
        """Save history to a JSON file.

        Args:
            path: File path to write to.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> History:
        """Deserialize from a dict.

        Args:
            d: Dict as produced by ``to_dict``.

        Returns:
            A ``History`` instance.
        """
        h = cls()
        for entry_d in d.get("entries", []):
            h.append(HistoryEntry.from_dict(entry_d))
        return h

    @classmethod
    def load_json(cls, path: str | Path) -> History:
        """Load history from a JSON file.

        Args:
            path: File path to read from.

        Returns:
            A ``History`` instance.
        """
        with open(path) as f:
            return cls.from_dict(json.load(f))

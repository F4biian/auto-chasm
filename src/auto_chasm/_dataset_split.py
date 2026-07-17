"""Stratified/grouped split machinery for :class:`auto_chasm.dataset.Dataset`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from auto_chasm.data import IGNORE_INDEX
from auto_chasm.logger import get_logger

logger = get_logger(__name__)


def _sample_stratum(sample: dict[str, Any]) -> Any:
    """Return a sample's response-level class for stratification.

    The stratum is the sample's last non-``-100`` label — the class at the
    response position (token/sentence datasets place one class per text, so the
    last labeled position carries it). An unlabeled sample gets a ``None`` stratum.
    Splitting is always per sample/group, so this never risks token leakage.

    Args:
        sample: A ``{"tokens", "labels"}`` sample. ``labels`` is a per-token
            list, or a ``{probe: list}`` dict (must hold exactly one probe).

    Returns:
        The response-level class (hashable), or ``None`` if the sample is
        unlabeled.

    Raises:
        ValueError: If ``labels`` is a multi-probe dict (the stratum is then
            ambiguous — pass an explicit ``stratify`` sequence instead).
    """
    labels = sample["labels"]
    if isinstance(labels, dict):
        # The reserved "lm_head" key is the per-token LM WEIGHT channel, never a
        # probe's class labels — exclude it from the ambiguity check.
        probe_keys = [k for k in labels if k != "lm_head"]
        if len(probe_keys) != 1:
            raise ValueError(
                "stratify='label' needs single-probe labels; this sample carries "
                f"{len(probe_keys)} probe label sets, so its class is ambiguous. Pass "
                "an explicit stratify=[...] sequence (one stratum per sample)."
            )
        labels = labels[probe_keys[0]]
    stratum: Any = None
    for value in labels:
        if value != IGNORE_INDEX:
            stratum = value
    return stratum


def _unit_stratum(strata: Sequence[Any], unit: Sequence[int]) -> Any:
    """The representative stratum of a group: the majority over its samples.

    A group (e.g. one prompt with several answers) may span classes; the
    no-leakage guarantee keeps the whole group on one side, so it is assigned by
    its most common stratum (ties broken by first appearance within the group).

    Args:
        strata: Per-sample strata for the whole dataset.
        unit: The sample indices belonging to this group.

    Returns:
        The group's representative stratum.
    """
    counts: dict[Any, int] = {}
    for i in unit:
        counts[strata[i]] = counts.get(strata[i], 0) + 1
    best: Any = None
    best_count = -1
    for i in unit:
        count = counts[strata[i]]
        if count > best_count:
            best_count = count
            best = strata[i]
    return best


def _grouped_stratified_val_indices(
    n: int,
    strata: Sequence[Any] | None,
    group_ids: Sequence[Any] | None,
    val_fraction: float,
    seed: int,
) -> set[int]:
    """Pick the validation indices for a group-pure, optionally stratified split.

    Groups are atomic — no group's samples are split across train/val — so a
    grouped split never leaks a shared prompt. Within each stratum the units are
    shuffled (seeded) and taken to the whole-group count closest to the stratum's
    per-sample validation target (groups are atomic, so an exact target is
    generally unreachable; ties favor the smaller set to avoid overshooting
    ``val_fraction``, so a multi-sample-group split may land slightly under it),
    preserving class proportions when groups are class-pure and
    giving a best-effort balance when a group spans classes. With ``strata=None`` and
    ``group_ids=None`` this reduces exactly to a single seeded permutation over
    the samples (the historical behavior).

    Args:
        n: Number of samples.
        strata: Per-sample stratum, or ``None`` for no stratification.
        group_ids: Per-sample group key, or ``None`` (each sample is its own
            group).
        val_fraction: Target validation fraction (applied per stratum).
        seed: Shuffle seed.

    Returns:
        The set of sample indices assigned to validation.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # Atomic units (groups). Insertion order = first appearance, so the whole
    # procedure is deterministic given the sample order and seed.
    if group_ids is None:
        units: list[list[int]] = [[i] for i in range(n)]
    else:
        by_group: dict[Any, list[int]] = {}
        for i, key in enumerate(group_ids):
            by_group.setdefault(key, []).append(i)
        units = list(by_group.values())

    # Bucket unit indices by their representative stratum (deterministic order).
    strat_order: list[Any] = []
    strat_units: dict[Any, list[int]] = {}
    for unit_idx, unit in enumerate(units):
        stratum = _unit_stratum(strata, unit) if strata is not None else None
        if stratum not in strat_units:
            strat_units[stratum] = []
            strat_order.append(stratum)
        strat_units[stratum].append(unit_idx)

    val_idx: set[int] = set()
    for stratum in strat_order:
        unit_indices = strat_units[stratum]
        total = sum(len(units[u]) for u in unit_indices)
        target = min(total, max(0, round(total * val_fraction)))
        # Honesty guard: a stratified split that rounds a rare class to zero val
        # samples would hide that class's accuracy — surface it, don't omit silently.
        if strata is not None and val_fraction > 0 and total > 0 and target == 0:
            logger.warning(
                "stratify: stratum %r has %d sample(s) but rounds to 0 in the "
                "validation split (val_fraction=%s); it will be ABSENT from val. "
                "Raise val_fraction or merge rare classes.",
                stratum,
                total,
                val_fraction,
            )
        if target <= 0:
            continue  # rounds to no val samples (warned above); leave the stratum out
        taken = 0
        for k, perm_pos in enumerate(rng.permutation(len(unit_indices)).tolist()):
            unit = units[unit_indices[perm_pos]]
            g = len(unit)
            # Groups are atomic: grow the total to the whole-group count CLOSEST to
            # the target (ties favor the smaller set), not the smallest that meets it
            # (overshoots). k==0 is always taken, so target>=1 is never dropped.
            if k >= 1 and abs(taken + g - target) >= abs(taken - target):
                break
            val_idx.update(unit)
            taken += g
    # An oversized atomic group can drag ALL samples into val, emptying train.
    n_total = sum(len(u) for u in units)
    if 0 < val_fraction < 1.0 and n_total > 0 and len(val_idx) >= n_total:
        logger.warning(
            "Grouped split put all %d samples in val (train EMPTY): a group exceeds "
            "val_fraction=%s. Use finer groups.",
            n_total,
            val_fraction,
        )
    return val_idx


def _stamp_groups(samples: list[dict[str, Any]], groups: Sequence[Any]) -> None:
    """Stamp a ``"group"`` key on each built sample (for grouped splitting).

    Args:
        samples: The built ``{"tokens", "labels"}`` samples.
        groups: One group key per built sample.

    Raises:
        ValueError: If ``len(groups) != len(samples)``.
    """
    keys = list(groups)
    if len(keys) != len(samples):
        raise ValueError(
            f"groups has {len(keys)} entries but {len(samples)} samples were "
            "built; provide one group key per input item."
        )
    for sample, key in zip(samples, keys, strict=True):
        sample["group"] = key

"""Oracle tests for stratified, group-aware ``Dataset.split``.

Pins the two guarantees of the new ``split`` knobs and their interaction:

- ``stratify`` keeps class proportions across train/val, computed once per
  sample at *response level* (the last non-``-100`` label) so the split is never
  done at token level.
- ``groups`` is a hard no-leakage constraint: every sample sharing a group key
  lands on the same side (e.g. one prompt with several answers never straddles
  the split). Groups win over stratification when a group spans classes.
- ``stratify=None, groups=None`` reproduces the historical plain random split
  byte-for-byte (same RNG, same rounding), so existing seeds do not move.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

import numpy as np
import pytest

from auto_chasm import Dataset


class _CharTok:
    """One token id per character (matches the build_dataset char-offset path)."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        """Return one id per character."""
        return [ord(c) for c in text]


def _labeled(idx: int, cls: int) -> dict[str, list[int]]:
    """A minimal sample: a unique token id (``idx``) and a single class label."""
    return {"tokens": [idx], "labels": [cls]}


def _val_indices(val: Dataset) -> set[int]:
    """Recover the original indices in a split by the unique token-id marker."""
    return {s["tokens"][0] for s in val}


# --------------------------------------------------------------------------- #
# Backward compatibility: the default path must not move.                      #
# --------------------------------------------------------------------------- #


def test_default_split_matches_legacy_permutation() -> None:
    """stratify=None, groups=None == the old single-permutation split exactly."""
    samples = [_labeled(i, i % 3) for i in range(20)]
    ds = Dataset(samples)
    _, val = ds.split(0.25, seed=0)

    # Reproduce the historical logic verbatim.
    order = np.random.default_rng(0).permutation(20)
    n_val = min(20, max(0, round(20 * 0.25)))
    legacy_val = set(order[:n_val].tolist())

    assert _val_indices(val) == legacy_val


def test_default_split_is_disjoint_complete_deterministic() -> None:
    """The default split still partitions the data and is reproducible."""
    ds = Dataset([_labeled(i, 0) for i in range(20)])
    train, val = ds.split(0.25, seed=7)
    assert len(train) + len(val) == 20
    assert _val_indices(val).isdisjoint(_val_indices(train))
    assert _val_indices(val) | _val_indices(train) == set(range(20))
    _, val2 = ds.split(0.25, seed=7)
    assert _val_indices(val) == _val_indices(val2)


# --------------------------------------------------------------------------- #
# Stratification.                                                              #
# --------------------------------------------------------------------------- #


def test_stratify_label_preserves_class_proportions() -> None:
    """Balanced classes => each class contributes exactly its share to val."""
    # 6 classes x 10 samples each.
    samples = [_labeled(i, i // 10) for i in range(60)]
    ds = Dataset(samples)
    _, val = ds.split(0.2, seed=1, stratify="label")
    per_class: dict[int, int] = {}
    for s in val:
        per_class[s["labels"][0]] = per_class.get(s["labels"][0], 0) + 1
    # round(10 * 0.2) == 2 from every class, no class starved or over-picked.
    assert per_class == dict.fromkeys(range(6), 2)
    assert len(val) == 12


def test_stratify_custom_sequence() -> None:
    """An explicit strata array stratifies on a key unrelated to the labels."""
    samples = [_labeled(i, 0) for i in range(20)]  # all one class
    strata = ["even" if i % 2 == 0 else "odd" for i in range(20)]  # custom axis
    ds = Dataset(samples)
    _, val = ds.split(0.5, seed=3, stratify=strata)
    evens = {i for i in _val_indices(val) if i % 2 == 0}
    odds = {i for i in _val_indices(val) if i % 2 == 1}
    assert len(evens) == 5 and len(odds) == 5  # half of each stratum


def test_stratify_imbalanced_rounds_per_class() -> None:
    """Each class's val count is round(n_c * frac) independently."""
    samples = [_labeled(i, 0) for i in range(8)] + [_labeled(8 + i, 1) for i in range(2)]
    ds = Dataset(samples)
    _, val = ds.split(0.5, seed=2, stratify="label")
    counts = {0: 0, 1: 0}
    for s in val:
        counts[s["labels"][0]] += 1
    assert counts == {0: 4, 1: 1}  # round(8*.5)=4, round(2*.5)=1


def test_stratify_label_on_response_eos_dataset() -> None:
    """The realistic path: response + append_eos, one class per text, stays balanced."""
    texts = [f"text number {i} here" for i in range(24)]
    labels = [i % 4 for i in range(24)]
    ds = Dataset.from_texts(
        texts, labels, _CharTok(), label_site="response", probe_name="cefr", append_eos=True
    )
    train, val = ds.split(0.25, seed=5, stratify="label")
    # 6 per class, round(6*.25)=2 in val from each of the 4 classes.
    val_classes = [next(v for v in s["labels"] if v != -100) for s in val]
    assert sorted(val_classes) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert len(train) == 16 and len(val) == 8


# --------------------------------------------------------------------------- #
# Grouping (no-leakage).                                                       #
# --------------------------------------------------------------------------- #


def test_groups_never_leak_across_split() -> None:
    """No group key appears on both sides of the split."""
    samples = [_labeled(i, 0) for i in range(20)]
    group_of = [i // 2 for i in range(20)]  # 10 groups of 2
    ds = Dataset(samples)
    train, val = ds.split(0.3, seed=4, groups=group_of)
    train_groups = {group_of[i] for i in _val_indices(train)}
    val_groups = {group_of[i] for i in _val_indices(val)}
    assert train_groups.isdisjoint(val_groups)
    # Whole groups move together => an even number of samples.
    assert len(val) % 2 == 0


def test_grouped_split_takes_count_closest_to_target_not_overshoot() -> None:
    """A group split takes the whole-group count CLOSEST to the target, not the
    smallest count that meets-or-exceeds it (which overshoots val_fraction).

    21 samples in 7 groups of 3; val_fraction 0.2 -> target round(21*0.2)=4. The
    closest whole-group count is 3 (one group; |3-4|=1 < |6-4|=2), NOT 6 (two
    groups) as the old take-until-met logic produced. Equal group sizes make the
    count seed-independent.
    """
    ds = Dataset([_labeled(i, 0) for i in range(21)])
    group_of = [i // 3 for i in range(21)]
    for seed in (0, 1, 7):
        _, val = ds.split(0.2, seed=seed, groups=group_of)
        assert len(val) == 3, f"seed={seed}: expected 3 (closest to target 4), got {len(val)}"


def test_groups_via_stamped_key() -> None:
    """groups='group' reads the key stamped by the builder."""
    texts = ["aa", "bb", "cc", "dd"]
    labels = [0, 0, 1, 1]
    # Same prompt id on pairs => those texts must not split apart.
    ds = Dataset.from_texts(
        texts, labels, _CharTok(), label_site="response", groups=["p0", "p0", "p1", "p1"]
    )
    train, val = ds.split(0.5, seed=0, groups="group")
    tr = {s["group"] for s in train}
    va = {s["group"] for s in val}
    assert tr.isdisjoint(va)
    assert tr | va == {"p0", "p1"}


def test_stratified_grouped_combo_pure_groups() -> None:
    """Pure groups (one class each): no leakage AND class proportions preserved."""
    # 12 groups of 2; classes 0,1,2 with 4 groups each.
    samples = []
    group_of = []
    strata = []
    for g in range(12):
        cls = g // 4
        for k in range(2):
            idx = g * 2 + k
            samples.append(_labeled(idx, cls))
            group_of.append(g)
            strata.append(cls)
    ds = Dataset(samples)
    train, val = ds.split(0.5, seed=6, stratify="label", groups=group_of)
    # No leakage.
    assert {group_of[i] for i in _val_indices(train)}.isdisjoint(
        {group_of[i] for i in _val_indices(val)}
    )
    # Balanced: each class has 8 samples => round(8*.5)=4 in val.
    per_class: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for s in val:
        per_class[s["labels"][0]] += 1
    assert per_class == {0: 4, 1: 4, 2: 4}


def test_group_spanning_classes_assigned_whole_no_leak() -> None:
    """A mixed-class group is kept intact (majority rule); still no leakage."""
    # Group 0 = classes [0,0,1] (majority 0); group 1 = [1,1] ; group 2 = [0].
    samples = [
        _labeled(0, 0),
        _labeled(1, 0),
        _labeled(2, 1),
        _labeled(3, 1),
        _labeled(4, 1),
        _labeled(5, 0),
    ]
    group_of = [0, 0, 0, 1, 1, 2]
    ds = Dataset(samples)
    train, val = ds.split(0.4, seed=9, stratify="label", groups=group_of)
    train_groups = {group_of[i] for i in _val_indices(train)}
    val_groups = {group_of[i] for i in _val_indices(val)}
    assert train_groups.isdisjoint(val_groups)
    # group 0's three samples are never divided.
    g0 = {i for i in range(6) if group_of[i] == 0}
    assert g0 <= _val_indices(train) or g0 <= _val_indices(val)
    # Deterministic.
    _, val2 = ds.split(0.4, seed=9, stratify="label", groups=group_of)
    assert _val_indices(val) == _val_indices(val2)


# --------------------------------------------------------------------------- #
# Validation / error paths.                                                    #
# --------------------------------------------------------------------------- #


def test_unknown_stratify_string_raises() -> None:
    """A non-'label' stratify string is rejected."""
    ds = Dataset([_labeled(0, 0)])
    with pytest.raises(ValueError, match="stratify"):
        ds.split(stratify="bogus")


def test_unknown_groups_string_raises() -> None:
    """A non-'group' groups string is rejected."""
    ds = Dataset([_labeled(0, 0)])
    with pytest.raises(ValueError, match="groups"):
        ds.split(groups="bogus")


def test_groups_group_without_key_raises() -> None:
    """groups='group' on samples lacking the key raises a clear error."""
    ds = Dataset([_labeled(0, 0), _labeled(1, 0)])
    with pytest.raises(ValueError, match="'group' key"):
        ds.split(groups="group")


def test_stratify_length_mismatch_raises() -> None:
    """A custom strata array of the wrong length is rejected."""
    ds = Dataset([_labeled(i, 0) for i in range(3)])
    with pytest.raises(ValueError, match="one stratum per sample"):
        ds.split(stratify=[0, 1])


def test_groups_length_mismatch_raises() -> None:
    """A custom groups array of the wrong length is rejected."""
    ds = Dataset([_labeled(i, 0) for i in range(3)])
    with pytest.raises(ValueError, match="one group key per sample"):
        ds.split(groups=[0, 1])


def test_builder_groups_length_mismatch_raises() -> None:
    """from_texts(groups=...) length must equal the number of texts."""
    with pytest.raises(ValueError, match="one group key per input item"):
        Dataset.from_texts(["a", "b"], [0, 1], _CharTok(), groups=["only-one"])


def test_stratify_label_multiprobe_dict_raises() -> None:
    """stratify='label' on multi-probe dict labels is ambiguous => raises."""
    ds = Dataset([{"tokens": [0, 1], "labels": {"a": [0, -100], "b": [-100, 1]}}])
    with pytest.raises(ValueError, match="single-probe"):
        ds.split(stratify="label")


def test_unlabeled_sample_buckets_under_none_stratum() -> None:
    """An all-(-100) sample stratifies into its own None bucket without error."""
    samples = [_labeled(0, 0), _labeled(1, 1), {"tokens": [2], "labels": [-100]}]
    ds = Dataset(samples)
    # frac high enough that the lone unlabeled sample (its own stratum) goes to val.
    _, val = ds.split(1.0, seed=0, stratify="label")
    assert 2 in _val_indices(val)


# --------------------------------------------------------------------------- #
# Honesty guard + audit-surfaced hardening (no silent class drop, robust keys, #
# cross-process determinism).                                                  #
# --------------------------------------------------------------------------- #


def test_round_to_zero_class_warns_and_stays_in_train(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A class whose val target rounds to 0 is logged (not silently dropped)."""
    samples = [_labeled(i, 0) for i in range(9)] + [_labeled(9, 1)]
    ds = Dataset(samples)
    with caplog.at_level(logging.WARNING, logger="auto_chasm.dataset"):
        train, val = ds.split(0.2, seed=1, stratify="label")
    # round(1 * 0.2) == 0 => class 1 is absent from val, wholly in train...
    assert all(s["labels"][0] == 0 for s in val)
    assert any(s["labels"][0] == 1 for s in train)
    # ...and that absence is surfaced, not silent.
    assert "rounds to 0" in caplog.text


def test_string_and_mixed_type_keys_split_correctly() -> None:
    """String strata + string group keys work, never leak, and are deterministic."""
    samples = [_labeled(i, 0) for i in range(12)]
    strata = ["a", "b", "c"] * 4  # string strata
    groups = [f"p{i // 2}" for i in range(12)]  # string group ids, pairs share a key
    ds = Dataset(samples)
    train, val = ds.split(0.5, seed=2, stratify=strata, groups=groups)
    tr_g = {groups[i] for i in _val_indices(train)}
    va_g = {groups[i] for i in _val_indices(val)}
    assert tr_g.isdisjoint(va_g)
    _, val2 = ds.split(0.5, seed=2, stratify=strata, groups=groups)
    assert _val_indices(val) == _val_indices(val2)


def test_unhashable_strata_raises_valueerror() -> None:
    """An unhashable strata element fails as a clear ValueError, not a raw TypeError."""
    ds = Dataset([_labeled(i, 0) for i in range(4)])
    with pytest.raises(ValueError, match="hashable"):
        ds.split(stratify=[[1], [2], [1], [2]])


def test_unhashable_groups_raises_valueerror() -> None:
    """An unhashable group key fails as a clear ValueError, not a raw TypeError."""
    ds = Dataset([_labeled(i, 0) for i in range(4)])
    with pytest.raises(ValueError, match="hashable"):
        ds.split(groups=[[1], [2], [1], [2]])


def test_grouped_split_may_place_whole_class_on_one_side() -> None:
    """Groups win: a class made into one group moves atomically, never leaking."""
    samples = [_labeled(i, 0) for i in range(8)] + [_labeled(8, 1), _labeled(9, 1)]
    group_of = [0] * 8 + [1, 1]  # class 0 is one indivisible group
    ds = Dataset(samples)
    train, val = ds.split(0.25, seed=1, stratify="label", groups=group_of)
    class0 = {i for i in range(10) if samples[i]["labels"][0] == 0}
    assert class0 <= _val_indices(train) or class0 <= _val_indices(val)
    assert {group_of[i] for i in _val_indices(train)}.isdisjoint(
        {group_of[i] for i in _val_indices(val)}
    )


def test_cross_process_determinism_under_hashseed() -> None:
    """The split is byte-identical across processes regardless of PYTHONHASHSEED.

    String strata / group keys are dict-keyed internally; this pins that the
    bucketing order does not depend on hash randomization (a future refactor to
    ``set()``/``sorted()`` would silently break cross-process reproducibility).
    """
    script = (
        "from auto_chasm import Dataset\n"
        "samples=[{'tokens':[i],'labels':[i%4]} for i in range(60)]\n"
        "strata=[f's{i%4}' for i in range(60)]\n"
        "groups=[f'g{i%15}' for i in range(60)]\n"
        "_, val = Dataset(samples).split(0.3, seed=7, stratify=strata, groups=groups)\n"
        "print(sorted(s['tokens'][0] for s in val))\n"
    )

    def run(hashseed: str) -> str:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hashseed},
        )
        return result.stdout.strip()

    out0, out1 = run("0"), run("1")
    assert out0 == out1
    assert out0.startswith("[") and out0 != "[]"  # produced a non-empty val list

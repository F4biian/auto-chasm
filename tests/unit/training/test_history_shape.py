"""One training step must be ONE history row, and test metrics must be in it.

The loop reports validation and throughput from two different points in an
iteration. With ``eval_steps == logging_steps`` that produced two entries per
step -- one with val numbers and the train fields null, one the reverse -- so a
file of N logged steps held 2N half-empty rows and anything plotting loss-vs-step
had to stitch them together. Separately, ``HistoryEntry`` has always advertised
``test_loss``/``test_metrics`` and no trainer ever filled them, so a run that DID
evaluate its test set showed them as null.
"""

from __future__ import annotations

from auto_chasm.history import History, HistoryEntry


def test_same_step_entries_merge_into_one_row() -> None:
    h = History()
    h.record(HistoryEntry(step=5, val_loss=1.5, val_metrics={"loss": 1.5}))
    h.record(HistoryEntry(step=5, train_loss=2.0, learning_rate=1e-4))
    assert len(h) == 1
    e = h.entries[0]
    assert e.train_loss == 2.0 and e.val_loss == 1.5 and e.learning_rate == 1e-4


def test_merge_never_nulls_out_the_earlier_half() -> None:
    """The second half-entry carries None for the first half's fields."""
    h = History()
    h.record(HistoryEntry(step=1, val_loss=0.9, val_metrics={"loss": 0.9}))
    h.record(HistoryEntry(step=1, train_loss=1.1))  # val_loss=None, val_metrics={}
    assert h.entries[0].val_loss == 0.9
    assert h.entries[0].val_metrics == {"loss": 0.9}


def test_different_steps_still_append() -> None:
    h = History()
    h.record(HistoryEntry(step=1, train_loss=1.0))
    h.record(HistoryEntry(step=2, train_loss=0.9))
    assert [e.step for e in h] == [1, 2]


def test_append_stays_unmerged() -> None:
    """``append`` is the raw primitive; only ``record`` merges."""
    h = History()
    h.append(HistoryEntry(step=1, train_loss=1.0))
    h.append(HistoryEntry(step=1, val_loss=2.0))
    assert len(h) == 2


def test_to_dict_omits_what_was_not_measured() -> None:
    d = HistoryEntry(step=3, train_loss=1.0).to_dict()
    assert d == {"step": 3, "train_loss": 1.0}
    assert "val_loss" not in d and "test_metrics" not in d and "custom" not in d


def test_round_trip_survives_the_omissions() -> None:
    e = HistoryEntry(step=7, train_loss=1.0, val_metrics={"f1": 0.5}, wall_time=2.0)
    back = HistoryEntry.from_dict(e.to_dict())
    assert back == e


def test_zero_is_kept_but_none_is_dropped() -> None:
    """0.0 is a measurement; None is the absence of one."""
    d = HistoryEntry(step=1, train_loss=0.0, val_loss=None).to_dict()
    assert d["train_loss"] == 0.0
    assert "val_loss" not in d

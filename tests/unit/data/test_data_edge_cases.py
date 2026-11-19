"""Regression tests for data-layer edge cases.

- m3: ``label_counts`` warns when labels fall outside ``[0, num_classes)`` or
  are non-integer (both were silently dropped / truncated).
- m5: span→token conversion rejects negative/inverted spans (silent no-label).
- m14: ``infer_task`` raises a clear error on negative class labels instead of
  a misleading "num_classes >= 2, got 0" or a later CE index crash.
"""

from __future__ import annotations

import logging

import pytest

from auto_chasm import Dataset
from auto_chasm.data import label_counts, span_labels_to_tokens


class _CharTok:
    """One token id per character (so span→token offsets are exact)."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


def test_m3_label_counts_warns_on_out_of_range_labels(caplog) -> None:  # noqa: ANN001
    """A label >= num_classes is not counted -- and now warns (m3)."""
    samples = [{"tokens": [1, 2, 3], "labels": [0, 1, 5]}]  # 5 is outside [0, 2)
    with caplog.at_level(logging.WARNING, "auto_chasm"):
        counts = label_counts(samples, num_classes=2)
    assert counts == [1, 1]  # the 5 is dropped (unchanged behavior)
    assert "outside [0, 2)" in caplog.text


def test_m3_label_counts_warns_on_fractional_labels(caplog) -> None:  # noqa: ANN001
    """Non-integer labels truncate toward zero when counted -- and now warn (m3)."""
    samples = [{"tokens": [1, 2], "labels": [0.0, 1.7]}]
    with caplog.at_level(logging.WARNING, "auto_chasm"):
        label_counts(samples, num_classes=2)
    assert "non-integer labels" in caplog.text


def test_m5_span_labels_warn_on_inverted_span(caplog) -> None:  # noqa: ANN001
    """An inverted span (start > end) warns (was silent) but still labels nothing (m5)."""
    with caplog.at_level(logging.WARNING, "auto_chasm"):
        labels = span_labels_to_tokens("abcdef", _CharTok(), [{"start": 4, "end": 2, "label": 1}])
    assert all(v == -100 for v in labels)  # inverted span labels nothing (unchanged)
    assert "Malformed label span" in caplog.text


def test_m5_span_labels_warn_on_negative_span(caplog) -> None:  # noqa: ANN001
    """A negative start warns (was silent) but still clamps and labels (m5)."""
    with caplog.at_level(logging.WARNING, "auto_chasm"):
        labels = span_labels_to_tokens("abcdef", _CharTok(), [{"start": -1, "end": 3, "label": 1}])
    assert labels[0] == 1  # negative start clamps to 0 (unchanged)
    assert "Malformed label span" in caplog.text


def test_m5_zero_supervision_warns_when_spans_cover_nothing(caplog) -> None:  # noqa: ANN001
    """A token-site sample whose spans cover no token (warmup>=len) warns (m5)."""
    # warmup at char 6 over a 5-char text -> the span [5,5) covers no token, and
    # uncovered tokens are masked -> zero supervision. (default_label=None => -100.)
    with caplog.at_level(logging.WARNING, "auto_chasm"):
        labels = span_labels_to_tokens("abcde", _CharTok(), [{"start": 5, "end": 5, "label": 1}])
    assert all(v == -100 for v in labels)
    assert "zero supervision" in caplog.text


def test_m5_zero_supervision_silent_with_default_label(caplog) -> None:  # noqa: ANN001
    """With default_label set, uncovered tokens carry a real class -> no false warning (m5)."""
    with caplog.at_level(logging.WARNING, "auto_chasm"):
        labels = span_labels_to_tokens(
            "abcde", _CharTok(), [{"start": 5, "end": 5, "label": 1}], default_label=0
        )
    assert all(v == 0 for v in labels)  # every token is the default class (real supervision)
    assert "zero supervision" not in caplog.text


def test_m14_infer_task_rejects_all_negative_labels() -> None:
    """All-(-1) labels raise a clear negative-label error, not 'num_classes >= 2, got 0' (m14)."""
    ds = Dataset([{"tokens": [1, 2], "labels": [-1, -1]}])
    with pytest.raises(ValueError, match="negative class label"):
        ds.infer_task()


def test_m14_infer_task_rejects_signed_binary_labels() -> None:
    """{-1, +1} labels raise (would otherwise become multiclass(2) that dies at CE) (m14)."""
    ds = Dataset([{"tokens": [1, 2], "labels": [-1, 1]}])
    with pytest.raises(ValueError, match="negative class label"):
        ds.infer_task()


def test_m14_infer_task_regression_allows_negative_targets() -> None:
    """Signed *float* targets are valid regression -- the negative check must not fire (m14)."""
    ds = Dataset([{"tokens": [1, 2], "labels": [-1.5, 2.5]}])
    task = ds.infer_task(kind="regression")
    assert task.kind == "regression"


def test_from_texts_preserves_float_regression_labels() -> None:
    """from_texts keeps a fractional label as a float (regression), not int-truncated."""
    # 1.7 must survive as 1.7 (was truncated to 1, silently breaking regression targets).
    ds = Dataset.from_texts(
        ["hello world"],
        [1.7],
        _CharTok(),
        probe_name="score",
        label_site="response",
        append_eos=True,
    )
    labeled = [v for v in ds[0]["labels"] if v != -100]
    assert labeled == [1.7]
    # A whole-number label stays an int class index (classification unaffected).
    ds2 = Dataset.from_texts(
        ["hi"], [2], _CharTok(), probe_name="c", label_site="response", append_eos=True
    )
    kept = [v for v in ds2[0]["labels"] if v != -100]
    assert kept == [2] and isinstance(kept[0], int)

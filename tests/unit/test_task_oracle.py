"""Oracle tests for data-driven task inference (`Dataset.infer_task` + `Task`).

Pins the Phase-4 `task="auto"` surface: the label dtype/range → task-kind rules,
the consistent (out_features, loss, metrics) that follow from a `Task`, and the
explicit-override path (modeling integer ordinal labels as regression).  These
are pure-numpy/label-shape checks — no model, no backend compute — so they run on
any machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from auto_chasm import Dataset, JointLoss, Task


def _ds(label_lists: list[list], probe: str | None = None) -> Dataset:
    """Build a Dataset of ``{"tokens", "labels"}`` samples from raw label lists."""
    samples = []
    for labs in label_lists:
        tokens = list(range(len(labs)))
        labels = {probe: labs} if probe is not None else labs
        samples.append({"tokens": tokens, "labels": labels})
    return Dataset(samples)


# --------------------------------------------------------------------------- #
# Task value object: constructors, derived knobs, validation.                 #
# --------------------------------------------------------------------------- #


def test_task_derived_knobs() -> None:
    """out_features / loss_spec / is_classification follow from the kind."""
    binary = Task.binary()
    assert (binary.kind, binary.num_classes, binary.out_features, binary.loss_spec) == (
        "binary",
        2,
        1,
        "bce",
    )
    assert binary.is_classification

    multi = Task.multiclass(6)
    assert (multi.out_features, multi.loss_spec, multi.is_classification) == (6, "ce", True)

    reg = Task.regression(6)
    assert (reg.out_features, reg.loss_spec, reg.is_classification) == (1, "mse", False)


def test_task_classification_factory_picks_binary_at_two() -> None:
    """Task.classification(2) is binary; classification(C>2) is multiclass."""
    assert Task.classification(2) == Task.binary()
    assert Task.classification(5) == Task.multiclass(5)


def test_task_validation_rejects_bad_num_classes() -> None:
    """binary/multiclass need num_classes >= 2; regression bins must be >= 2 or None."""
    with pytest.raises(ValueError, match="needs num_classes >= 2"):
        Task("multiclass", 1)
    with pytest.raises(ValueError, match="needs num_classes >= 2"):
        Task("binary", None)
    with pytest.raises(ValueError, match="ordinal bins"):
        Task("regression", 1)
    assert Task("regression", None).num_classes is None  # continuous is allowed


# --------------------------------------------------------------------------- #
# infer_task: dtype + range -> kind.                                          #
# --------------------------------------------------------------------------- #


def test_infer_auto_multiclass_from_int_range() -> None:
    """Integer labels spanning 0..5 (with -100 ignores) → multiclass, num_classes=6."""
    ds = _ds([[0, 1, 2, -100, 3], [4, 5, 0, 1, 2]])
    assert ds.infer_task() == Task.multiclass(6)


def test_infer_auto_binary_from_zero_one() -> None:
    """Integer labels ⊆ {0,1} → binary (single-logit bce head)."""
    ds = _ds([[0, 1, 1, -100, 0], [1, 0, 1, 1, 0]])
    task = ds.infer_task()
    assert task == Task.binary()
    assert task.out_features == 1  # single logit, NOT 2


def test_infer_auto_regression_from_float_dtype() -> None:
    """Floating-point labels → regression (the dtype is the intent signal)."""
    ds = _ds([[0.1, 0.9, -100.0, 0.4], [0.7, 0.2, 0.5, -100.0]])
    task = ds.infer_task()
    assert task.kind == "regression"
    assert task.out_features == 1 and task.loss_spec == "mse"


def test_infer_regression_override_on_integer_ordinal_labels() -> None:
    """kind='regression' models integer ordinal labels (0..5) as scalar MSE.

    This is the german-cefr regression variant: the SAME integer CEFR labels that
    auto-infer as multiclass(6) are, on request, a scalar regression whose ordinal
    bin count (for the discretized accuracy metrics) is still read from the labels.
    """
    ds = _ds([[0, 1, 2, -100, 3], [4, 5, 0, 1, 2]])
    assert ds.infer_task() == Task.multiclass(6)  # auto default
    assert ds.infer_task(kind="regression") == Task.regression(6)  # explicit override


def test_infer_classification_selector_forces_class_over_float() -> None:
    """kind='classification' treats whole-valued float labels as classes, not regression."""
    ds = _ds([[0.0, 1.0, 2.0, -100.0], [2.0, 1.0, 0.0, 1.0]])
    # auto would call this regression (float dtype); classification overrides.
    assert ds.infer_task().kind == "regression"
    assert ds.infer_task(kind="classification") == Task.multiclass(3)


def test_infer_explicit_binary_and_multiclass_selectors() -> None:
    """Explicit 'binary'/'multiclass' selectors bypass the range heuristic."""
    ds = _ds([[0, 1, 2, 3]])
    assert ds.infer_task(kind="binary") == Task.binary()
    assert ds.infer_task(kind="multiclass") == Task.multiclass(4)


def test_infer_dict_labels_per_probe() -> None:
    """Per-probe (dict) labels infer each head independently by probe_name."""
    ds = _ds([[0, 1, 2], [3, 2, 1]], probe="cefr")
    assert ds.infer_task("cefr") == Task.multiclass(4)


def test_infer_dict_labels_requires_probe_name() -> None:
    """A dict-labelled dataset with no probe_name is ambiguous → raises."""
    ds = _ds([[0, 1]], probe="cefr")
    with pytest.raises(ValueError, match="per-probe"):
        ds.infer_task()


def test_infer_empty_labels_raises() -> None:
    """A fully-ignored (all -100) dataset cannot yield a task → clear error."""
    ds = _ds([[-100, -100], [-100, -100]])
    with pytest.raises(ValueError, match="no labeled"):
        ds.infer_task()


def test_infer_unknown_kind_raises() -> None:
    """An unknown kind selector is a typo error, not a silent fallback."""
    ds = _ds([[0, 1, 2]])
    with pytest.raises(ValueError, match="Unknown task kind selector"):
        ds.infer_task(kind="ordinal")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Task -> ready-to-use loss + metrics.                                        #
# --------------------------------------------------------------------------- #


def test_build_loss_shapes_a_single_probe_jointloss() -> None:
    """build_loss produces JointLoss(weights={'lm_head': lm_weight}, losses={probe: spec})."""
    task = Task.multiclass(6)
    loss = task.build_loss("cefr", lm_weight=0.0)
    assert isinstance(loss, JointLoss)
    # Pure-probe: lm_head weighted 0; the probe carries the task's loss spec.
    assert loss._weights == {"lm_head": 0.0}
    assert loss._losses == {"cefr": "ce"}


def test_build_loss_rejects_reserved_name_and_bad_class_weights() -> None:
    """lm_head probe name and class_weights on a non-CE task are rejected."""
    with pytest.raises(ValueError, match="reserved"):
        Task.multiclass(3).build_loss("lm_head")
    with pytest.raises(ValueError, match="class_weights apply only to a multi-class"):
        Task.binary().build_loss("p", class_weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="class_weights apply only to a multi-class"):
        Task.regression(4).build_loss("p", class_weights=[1.0, 1.0, 1.0, 1.0])
    # multiclass accepts class weights.
    ok = Task.multiclass(3).build_loss("p", class_weights=[1.0, 2.0, 1.0])
    assert isinstance(ok, JointLoss)


def test_build_metrics_matches_task_and_reports_expected_keys() -> None:
    """Classification → *_macro_f1 keys; regression → *_mse/_mae keys; both via a fake head."""

    class _FakeMLX:
        """Minimal stand-in exposing get_probe(name) -> (hidden -> logits)."""

        def __init__(self, logits: dict[str, np.ndarray]) -> None:
            self._logits = logits

        def get_probe(self, name: str):  # noqa: ANN202
            return lambda _hidden: self._logits[name]

    # Multi-class head: [B=1, T=2, C=3]; targets match -> acc 1.0.
    cls_fn = Task.multiclass(3).build_metrics(ordinal_tol=1)
    cls_logits = {"p": np.array([[[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])}
    out = cls_fn(_FakeMLX(cls_logits), {"p": None}, np.array([[0, 2]]), np.array([[1, 1]]))
    assert set(out) == {"p_acc", "p_adj", "p_macro_f1"}
    assert out["p_acc"] == pytest.approx(1.0)

    # Regression head: [B=1, T=2, 1]; scalar preds -> mse/mae + discretized acc.
    reg_fn = Task.regression(6).build_metrics()
    reg_logits = {"p": np.array([[[1.9], [4.1]]])}  # rounds to 2, 4
    out_r = reg_fn(_FakeMLX(reg_logits), {"p": None}, np.array([[2, 4]]), np.array([[1, 1]]))
    assert {"p_mse", "p_mae", "p_acc", "p_adj"} <= set(out_r)
    assert out_r["p_acc"] == pytest.approx(1.0)  # rounded preds hit the targets


def test_continuous_regression_metrics_needs_bins() -> None:
    """A continuous regression Task (num_classes None) can't discretize → clear error."""
    with pytest.raises(ValueError, match="need num_classes"):
        Task.regression(None).build_metrics()

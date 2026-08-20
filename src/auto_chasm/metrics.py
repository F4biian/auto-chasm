"""Backend-agnostic classification metrics and probe-eval helpers.

These let you score probe heads without importing ``mlx``/``torch`` or writing
tensor-to-numpy glue.  Every numeric step runs in NumPy after a single
type-dispatched conversion, so the same code runs on a torch-only box and an
MLX Mac alike.

The headline entry point is :func:`classification_metrics`, a factory that
returns a ready-to-use ``eval_metrics_fn`` for ``Trainer(eval_metrics_fn=...)``
reporting per-probe accuracy, adjacent (ordinal) accuracy, and macro-F1.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def to_numpy(x: Any) -> np.ndarray:
    """Convert an MLX / PyTorch / NumPy tensor to a float32 NumPy array.

    Dispatches on the value's module (never on which framework is importable, so
    a torch tensor is never routed through MLX on a machine with both), and casts
    via float32 first so bf16 tensors — which NumPy cannot represent — survive.

    Args:
        x: An ``mlx.core.array``, ``torch.Tensor``, NumPy array, or array-like.

    Returns:
        A float32 NumPy array with the same shape as ``x``.
    """
    module = type(x).__module__
    if module.startswith("torch"):
        return x.detach().to("cpu").float().numpy()
    if module.startswith("mlx"):
        import mlx.core as mx

        return np.array(x.astype(mx.float32))
    return np.asarray(x, dtype=np.float32)


def run_probe(train_model: Any, name: str, hidden: Any) -> Any:
    """Run a probe head by name on a hidden state, on either backend.

    Prefers the ``Probe.forward`` path on **both** backends (MLX
    ``_TrainableModel`` exposes the ``Probe`` objects as ``_probe_captures``;
    torch ``_TorchProbeWrapper`` as ``_probes``) so the probe's pooling /
    granularity is applied identically — otherwise a response/sentence head would
    return unpooled ``[B, T, C]`` on MLX but pooled ``[B, C]`` on torch.  Falls
    back to a bare module via ``get_probe`` for minimal stand-ins that expose
    only that.

    Args:
        train_model: The trainable wrapper passed to an ``eval_metrics_fn``.
        name: Probe name.
        hidden: The captured hidden state ``[B, T, H]`` for that probe.

    Returns:
        The probe's logits — ``[B, T, C]`` for token granularity, ``[B, C]`` for
        a pooled (response/sentence) head — identical on both backends.
    """
    probes = getattr(train_model, "_probe_captures", None)
    if probes is None:
        probes = getattr(train_model, "_probes", None)
    if probes is not None and name in probes:
        # ``hidden`` is the list of ALL captured layer states for a multi-layer probe
        # (a bare tensor for a one-layer probe); forward every layer so aggregation is
        # applied correctly (concat/mean/max over all layers, not just the last).
        states = hidden if isinstance(hidden, list) else [hidden]
        return probes[name].forward(states)
    return train_model.get_probe(name)(hidden)


def _prep(preds: Any, targets: Any, mask: Any) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``preds``/``targets`` to the valid (masked, non ``-100``) positions.

    Args:
        preds: Predicted class indices (any backend or numpy), shape ``[B, T]``.
        targets: Target class indices, shape ``[B, T]``; ``-100`` is ignored.
        mask: Boolean validity mask, shape ``[B, T]``.

    Returns:
        Tuple ``(preds_kept, targets_kept)`` as 1-D int64 arrays over the kept
        positions.
    """
    preds_np = to_numpy(preds).astype(np.int64)
    targets_np = to_numpy(targets).astype(np.int64)
    keep = to_numpy(mask).astype(bool) & (targets_np != -100)
    return preds_np[keep], targets_np[keep]


def accuracy(preds: Any, targets: Any, mask: Any) -> float:
    """Masked, ``-100``-aware classification accuracy.

    Args:
        preds: Predicted class indices ``[B, T]`` (argmaxed logits).
        targets: Target class indices ``[B, T]``; ``-100`` positions are excluded.
        mask: Boolean validity mask ``[B, T]``.

    Returns:
        Fraction of kept positions where ``pred == target``; ``0.0`` if no
        position is kept (never ``NaN``).
    """
    p, t = _prep(preds, targets, mask)
    return float((p == t).mean()) if p.size else 0.0


def ordinal_accuracy(preds: Any, targets: Any, mask: Any, tol: int = 1) -> float:
    """Adjacent / off-by-one accuracy for ordinal labels.

    A prediction counts as correct when ``abs(pred - target) <= tol``.  Because
    plain integer distance is used (no wraparound), the lowest and highest
    classes are never treated as neighbors.

    Args:
        preds: Predicted class indices ``[B, T]``.
        targets: Target class indices ``[B, T]``; ``-100`` positions are excluded.
        mask: Boolean validity mask ``[B, T]``.
        tol: Maximum class distance still counted correct (``1`` = neighbors OK).

    Returns:
        Fraction of kept positions within ``tol`` classes; ``0.0`` if none kept.
    """
    p, t = _prep(preds, targets, mask)
    return float((np.abs(p - t) <= tol).mean()) if p.size else 0.0


def macro_f1(preds: Any, targets: Any, mask: Any, num_classes: int) -> float:
    """Macro-averaged F1 over ``num_classes`` (masked, ``-100``-aware).

    Args:
        preds: Predicted class indices ``[B, T]``.
        targets: Target class indices ``[B, T]``; ``-100`` positions are excluded.
        mask: Boolean validity mask ``[B, T]``.
        num_classes: Number of classes to average F1 over.

    Returns:
        The unweighted mean of per-class F1 scores; ``0.0`` if no position kept.
    """
    p, t = _prep(preds, targets, mask)
    if not p.size:
        return 0.0
    f1s: list[float] = []
    for c in range(num_classes):
        tp = int(np.sum((p == c) & (t == c)))
        fp = int(np.sum((p == c) & (t != c)))
        fn = int(np.sum((p != c) & (t == c)))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _prep_float(preds: Any, targets: Any, mask: Any) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`_prep` but keeps floats — for continuous regression outputs.

    Args:
        preds: Continuous predictions ``[B, T]`` (any backend or numpy).
        targets: Target values ``[B, T]``; ``-100`` is ignored.
        mask: Boolean validity mask ``[B, T]``.

    Returns:
        Tuple ``(preds_kept, targets_kept)`` as 1-D float64 arrays over the kept
        (masked, non ``-100``) positions.
    """
    preds_np = to_numpy(preds).astype(np.float64)
    targets_np = to_numpy(targets).astype(np.float64)
    keep = to_numpy(mask).astype(bool) & (targets_np != -100)
    return preds_np[keep], targets_np[keep]


def _pool_targets_to_preds(preds: Any, targets: Any, mask: Any) -> tuple[Any, Any]:
    """Collapse per-token targets/mask to per-sequence when the probe is pooled.

    A ``response``/``sentence``-granularity head returns ONE prediction per
    sequence (``preds`` shape ``[B]``), but the trainer always hands the metric
    per-token targets and mask (``[B, T]``).  Reduce each row to the label at its
    first valid (masked, non ``-100``) position — every valid position of a pooled
    sequence carries that sequence's single label — so the metric aligns with the
    pooled prediction instead of index-erroring on ``preds[mask]``.  Token-
    granularity preds (``[B, T]``) are returned unchanged.

    Args:
        preds: The probe's predictions (``[B]`` pooled, or ``[B, T]`` per token).
        targets: Per-token targets ``[B, T]`` (or a scalar-selected array).
        mask: Per-token validity mask ``[B, T]``.

    Returns:
        ``(targets, mask)`` — collapsed to ``[B]`` when ``preds`` is pooled, else
        the originals.
    """
    p = to_numpy(preds)
    t = to_numpy(targets)
    m = to_numpy(mask)
    if p.ndim >= m.ndim:
        return targets, mask  # token granularity: shapes already align
    valid = m.astype(bool) & (t != -100)
    rows = np.arange(t.shape[0])
    seq_targets = t[rows, valid.argmax(axis=1)]  # first valid label per row
    seq_mask = valid.any(axis=1)  # False for a fully-masked (unlabeled) sequence
    return seq_targets, seq_mask


def mse(preds: Any, targets: Any, mask: Any) -> float:
    """Masked, ``-100``-aware mean squared error.

    Args:
        preds: Continuous predictions ``[B, T]``.
        targets: Target values ``[B, T]``; ``-100`` positions are excluded.
        mask: Boolean validity mask ``[B, T]``.

    Returns:
        Mean of ``(pred - target)**2`` over kept positions; ``0.0`` if none kept.
    """
    p, t = _prep_float(preds, targets, mask)
    return float(((p - t) ** 2).mean()) if p.size else 0.0


def mae(preds: Any, targets: Any, mask: Any) -> float:
    """Masked, ``-100``-aware mean absolute error.

    Args:
        preds: Continuous predictions ``[B, T]``.
        targets: Target values ``[B, T]``; ``-100`` positions are excluded.
        mask: Boolean validity mask ``[B, T]``.

    Returns:
        Mean of ``abs(pred - target)`` over kept positions; ``0.0`` if none kept.
    """
    p, t = _prep_float(preds, targets, mask)
    return float(np.abs(p - t).mean()) if p.size else 0.0


def discretize(preds: Any, num_classes: int) -> np.ndarray:
    """Round a continuous ordinal prediction to the nearest class, clipped to range.

    Maps a regression output on the ordinal scale ``0..num_classes-1`` back to a
    class index by rounding to the nearest integer and clamping into the valid
    range (so an out-of-range prediction snaps to the boundary class).

    Args:
        preds: Continuous predictions (any backend or numpy).
        num_classes: Number of ordinal classes.

    Returns:
        A NumPy array of class indices (same shape as ``preds``).
    """
    return np.clip(np.rint(to_numpy(preds)), 0, num_classes - 1)


def regression_metrics(
    num_classes: int,
    ordinal_tol: int = 1,
) -> Callable[..., dict[str, float]]:
    """Build an ``eval_metrics_fn`` for an ordinal **regression** probe head.

    The head outputs one continuous value per position (``out_features=1``) on
    the ordinal scale ``0..num_classes-1``.  For each probe this reports the
    regression error (``{name}_mse``, ``{name}_mae``) AND — by rounding the
    prediction back to the nearest class (:func:`discretize`) — the same
    classification numbers the classification metrics report (``{name}_acc``,
    ``{name}_adj``), so a regressor is directly comparable to a classifier.

    Works on both backends and on per-probe ``{name: array}`` dict targets.

    Args:
        num_classes: Number of ordinal classes (sets the discretization range).
        ordinal_tol: Tolerance for the adjacent (group) accuracy on the
            discretized predictions.

    Returns:
        An ``eval_metrics_fn`` callable with the trainer's metric signature
        ``(train_model, captured, targets, mask) -> dict[str, float]``.
    """

    def _fn(
        train_model: Any, captured: dict[str, Any], targets: Any, mask: Any
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, hidden in captured.items():
            raw = to_numpy(run_probe(train_model, name, hidden))
            # Scalar head: [B, T, 1] -> [B, T] (pooled [B, 1] -> [B]).
            preds = raw[..., 0] if raw.ndim >= 2 and raw.shape[-1] == 1 else raw
            tgt = targets[name] if isinstance(targets, dict) else targets
            tgt, msk = _pool_targets_to_preds(preds, tgt, mask)  # align pooled heads
            out[f"{name}_mse"] = mse(preds, tgt, msk)
            out[f"{name}_mae"] = mae(preds, tgt, msk)
            disc = discretize(preds, num_classes)
            out[f"{name}_acc"] = accuracy(disc, tgt, msk)
            out[f"{name}_adj"] = ordinal_accuracy(disc, tgt, msk, tol=ordinal_tol)
        return out

    return _fn


def auroc(scores: Any, targets: Any, mask: Any) -> float:
    """Masked, ``-100``-aware AUROC for a binary head, from raw SCORES.

    Rank-based (Mann-Whitney U) with ties averaged, so no sklearn dependency and
    no threshold: unlike accuracy or F1 it reads the head's continuous output, and
    it is invariant to any positive rescaling or shift of that output. That is
    what makes it the right metric for a probe whose direction is frozen and whose
    scale/bias are free.

    Args:
        scores: Continuous head output ``[B, T]`` (a logit, not a class index).
        targets: Binary targets ``[B, T]``; ``-100`` positions are excluded.
        mask: Boolean validity mask ``[B, T]``.

    Returns:
        AUROC in ``[0, 1]``, or ``nan`` when the selection holds only one class
        (undefined then -- the caller should OMIT the key rather than record a
        number, so the eval loop averages over the batches where it existed).
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    t = np.asarray(targets).reshape(-1)
    m = np.asarray(mask).reshape(-1).astype(bool) & (t != -100)
    s, t = s[m], t[m].astype(np.int64)
    pos = t == 1
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # Average ranks inside tie groups, or a head that outputs constants would
    # score 0 or 1 instead of the correct 0.5.
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt), dtype=np.float64)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def classification_metrics(
    num_classes: int | None = None,
    ordinal_tol: int = 1,
) -> Callable[..., dict[str, float]]:
    """Build an ``eval_metrics_fn`` reporting per-probe classification metrics.

    The returned callable has the trainer's metric signature
    ``(train_model, captured, targets, mask) -> dict[str, float]``.  For each
    probe in ``captured`` it turns the head's logits into class predictions — a
    multi-logit head is ``argmax``-ed, and a single-logit (binary) head is
    thresholded at ``sigmoid(logit) > 0.5`` (i.e. ``logit > 0``), never
    ``argmax``-ed over its size-1 axis — then emits ``{name}_acc`` (exact
    accuracy), ``{name}_adj`` (adjacent / ordinal accuracy), and
    ``{name}_macro_f1``.  It works on both backends and on per-probe ``{name:
    array}`` dict targets (selecting each head's own target).

    Args:
        num_classes: Class count for macro-F1 on a multi-logit head.  ``None``
            infers it from the logit dimension.  A single-logit (binary) head is
            always scored over 2 classes and ignores this argument (so a caller
            like ``LayerSweep`` that passes ``num_classes=out_features=1`` still
            gets a correct two-class macro-F1).
        ordinal_tol: Tolerance for the adjacent-accuracy metric.

    Returns:
        An ``eval_metrics_fn`` callable.
    """

    def _fn(
        train_model: Any, captured: dict[str, Any], targets: Any, mask: Any
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, hidden in captured.items():
            logits = to_numpy(run_probe(train_model, name, hidden))
            if logits.shape[-1] == 1:
                # Binary (single-logit) head: threshold the sigmoid at 0.5
                # (equivalently, logit > 0). ``argmax`` over the size-1 class axis
                # would always return class 0, collapsing every prediction to the
                # negative class and scoring the head at its base rate. A
                # single-logit head is definitionally 2-class, so macro-F1 is
                # scored over 2 classes regardless of a caller-supplied
                # ``num_classes`` (LayerSweep defaults num_classes=out_features=1,
                # which would otherwise drop the positive class from macro-F1).
                preds = (logits[..., 0] > 0.0).astype(np.int64)
                n_cls = 2
            else:
                preds = logits.argmax(-1)
                n_cls = num_classes if num_classes is not None else int(logits.shape[-1])
            tgt = targets[name] if isinstance(targets, dict) else targets
            tgt, msk = _pool_targets_to_preds(preds, tgt, mask)  # align pooled heads
            out[f"{name}_acc"] = accuracy(preds, tgt, msk)
            out[f"{name}_adj"] = ordinal_accuracy(preds, tgt, msk, tol=ordinal_tol)
            out[f"{name}_macro_f1"] = macro_f1(preds, tgt, msk, n_cls)
            if logits.shape[-1] == 1:
                # Threshold-free, and the metric separability studies report. OMIT
                # the key on a single-class batch rather than emitting nan: the
                # eval loop averages each key over the batches where it appeared,
                # so a nan would poison the whole average.
                score = auroc(logits[..., 0], tgt, msk)
                if not np.isnan(score):
                    out[f"{name}_auroc"] = score
        return out

    return _fn

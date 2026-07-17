"""The class-weighted cross-entropy formula, expressed once (backend-agnostic).

Kept out of ``loss.py`` so that file stays under the line cap.  :func:`weighted_ce`
computes a weighted, masked-mean CE — fp32 accumulation, a per-class weight gather,
and a weighted denominator — through the :mod:`auto_chasm.ops` facade, so a single
code path runs identically on MLX and torch and an all-(``-100``) window returns a
finite ``0``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from auto_chasm.utils import tensor_backend

_SEQ_CW_MSG = (
    "class_weights is only supported for token-level CE; this probe is "
    "sequence-level (granularity='response'/'sentence'), whose pooled target is "
    "a float average that cannot index a per-class weight."
)


def _is_weightable_loss(spec: Any) -> bool:
    """Whether ``class_weights`` apply to this loss spec (built-in ``ce``/``bce``).

    A custom callable or a numeric (``mse``/``mae``) loss has no class structure to
    weight, so it is not weightable. The loss router lower-cases string specs, so
    ``"CE"``/``"BCE"`` match too.
    """
    return isinstance(spec, str) and spec.strip().lower() in ("ce", "bce")


def check_class_weights_applicable(probe_name: str, spec: Any, class_weights: Any) -> None:
    """Raise if ``class_weights`` is set on a probe whose loss cannot use it.

    Setting class weights on an ``mse``/``mae``/custom probe would be a SILENT no-op,
    so fail loudly at compute time — where the resolved per-probe loss is known —
    naming the probe.

    Args:
        probe_name: The probe the weights were resolved for.
        spec: The probe's resolved loss spec (a built-in name or a callable).
        class_weights: The per-probe class weights, or ``None``.

    Raises:
        ValueError: If ``class_weights`` is set and ``spec`` is not ``ce``/``bce``.
    """
    if class_weights is not None and not _is_weightable_loss(spec):
        kind = "a custom callable" if callable(spec) else f"'{spec}'"
        raise ValueError(
            f"class_weights was set for probe {probe_name!r}, but its loss is {kind}; "
            "class weights only apply to the built-in 'ce'/'bce' losses. Pass a "
            "{probe: weights} dict to weight only the ce/bce probes."
        )


def _require_resolved_class_weights(class_weights: Any) -> None:
    """Raise if class weights are still the unresolved ``"balanced"`` sentinel."""
    if isinstance(class_weights, str):
        raise NotImplementedError(
            "class_weights='balanced' must be resolved from the training data. "
            "Train via Trainer.train(dataset, ...) (which resolves it), or pass "
            "an explicit list, e.g. dataset.class_weights(num_classes)."
        )


def _validate_class_weights(
    class_weights: Any,
    default_loss: str | Callable[..., Any],
    probe_losses: dict[str, str | Callable[..., Any]],
) -> Any:
    """Validate ``class_weights`` and return its normalized form (or ``None``).

    Args:
        class_weights: A sequence, a ``{probe: sequence}`` dict, ``"balanced"``,
            or ``None``.
        default_loss: The default probe-loss spec.
        probe_losses: Per-probe loss overrides.

    Returns:
        ``None``, ``"balanced"``, a ``list[float]``, or a ``{probe: value}`` dict.

    Raises:
        ValueError: If no probe uses a built-in ``"ce"``/``"bce"`` loss, an entry
            is negative, or an unknown string is given.
    """
    if class_weights is None:
        return None

    # Class weights apply to both CE (a per-class vector) and BCE (a [w_neg, w_pos]
    # pair); the precise per-probe applicability is enforced at compute time by
    # ``check_class_weights_applicable`` (in JointLoss._probe_term).
    weightable = _is_weightable_loss(default_loss) or any(
        _is_weightable_loss(v) for v in probe_losses.values()
    )
    if not weightable:
        raise ValueError(
            "class_weights only affects the built-in 'ce'/'bce' probe losses, but "
            "no probe uses probe_loss='ce' or 'bce'. Set one of those (globally or "
            "per-probe), or drop class_weights."
        )

    def _norm(value: Any) -> Any:
        if isinstance(value, str):
            if value != "balanced":
                raise ValueError(
                    f"Unknown class_weights {value!r}. Use a list of floats, a "
                    "{probe: list} dict, or 'balanced'."
                )
            return value
        seq = [float(x) for x in value]
        if any(x < 0 for x in seq):
            raise ValueError("class_weights entries must be non-negative.")
        return seq

    if isinstance(class_weights, dict):
        return {k: _norm(v) for k, v in class_weights.items()}
    return _norm(class_weights)


def _check_weight_len(weights: Sequence[float], num_classes: int) -> None:
    """Raise if the per-class weight vector length does not match the class count.

    Mirrors ``_check_class_indices`` on the label side: an ill-sized vector is a
    user error that must fail identically on both backends, not silently zero
    out-of-range classes (MLX's gather is bounds-unsafe) or raise an opaque
    ``IndexError`` (torch).

    Args:
        weights: The per-class weights.
        num_classes: The logits' last dimension.

    Raises:
        ValueError: If ``len(weights) != num_classes``.
    """
    if len(weights) != num_classes:
        raise ValueError(
            f"class_weights has {len(weights)} entries but the probe head has "
            f"{num_classes} classes; provide exactly one weight per class."
        )


def _to_float32(x: Any) -> Any:
    """Cast ``x`` to float32 on its own backend (branch on the tensor type)."""
    if tensor_backend(x) == "torch":
        import torch

        return x.to(torch.float32)
    import mlx.core as mx

    return x.astype(mx.float32)


def _onehot(indices: Any, num_classes: int, ref: Any) -> Any:
    """Float32 one-hot of integer ``indices`` over ``num_classes`` (backend-agnostic).

    Built by comparing ``indices[..., None]`` against ``arange(num_classes)`` so no
    backend-specific gather/scatter is needed; the result matches ``ref``'s backend
    and device via :func:`auto_chasm.ops.arange`.

    Args:
        indices: Integer class indices ``[...]`` (all in ``[0, num_classes)``).
        num_classes: The class count ``C``.
        ref: A tensor whose backend/device to match.

    Returns:
        A float32 one-hot tensor ``[..., C]``.
    """
    from auto_chasm import ops

    classes = ops.arange(num_classes, like=ref)
    return _to_float32(indices[..., None] == classes)


def weighted_ce(
    probe_logits: Any,
    shifted: Any,
    label_valid: Any,
    probe_mask: Any,
    weights: Sequence[float],
) -> Any:
    """Class-weighted token-level cross-entropy (backend-agnostic).

    One code path for MLX and torch: fp32 accumulation, a one-hot per-class weight
    gather, and a weighted denominator, so both backends agree numerically and an
    all-(``-100``) window returns a finite ``0``.

    Args:
        probe_logits: Per-token logits ``[B, T, C]``.
        shifted: Shifted class-index labels ``[B, T]`` (``-100`` at ignored).
        label_valid: Boolean ``[B, T]`` mask of non-``-100`` labels.
        probe_mask: Boolean ``[B, T]`` validity mask (length window AND label).
        weights: Per-class weights of length ``C``.

    Returns:
        Scalar weighted-mean CE.
    """
    from auto_chasm import ops

    logits = _to_float32(probe_logits)
    num_classes = int(logits.shape[-1])
    _check_weight_len(weights, num_classes)
    zero = ops.zeros_like(shifted)
    safe_t = ops.where(label_valid, shifted, zero)
    onehot = _onehot(safe_t, num_classes, logits)
    ce_each = ops.logsumexp(logits, axis=-1) - ops.sum(logits * onehot, axis=-1)
    w = _class_weight_vector(weights, logits)
    w_gathered = ops.sum(w * onehot, axis=-1)
    ww = w_gathered * _to_float32(probe_mask)
    # Weighted mean CE = sum(w*ce) / sum(w). Guard ONLY the all-masked 0/0 case with a
    # tiny epsilon (numerator is 0 there too). Do NOT clamp the denominator to 1.0 —
    # that silently rescaled the loss whenever the summed class weights were < 1
    # (fractional / normalized weights), making the loss scale batch-content-dependent.
    return ops.sum(ce_each * ww) / ops.clamp(ops.sum(ww), lo=1e-8)


def _class_weight_vector(weights: Sequence[float], ref: Any) -> Any:
    """A float32 ``[C]`` weight vector on ``ref``'s backend/device.

    Args:
        weights: Per-class weights.
        ref: A tensor whose backend/device to match.

    Returns:
        A float32 tensor of the weights.
    """
    if tensor_backend(ref) == "torch":
        import torch

        return torch.as_tensor(list(weights), dtype=torch.float32, device=ref.device)
    import mlx.core as mx

    return mx.array(list(weights)).astype(mx.float32)


# Floor on ``1 - p`` inside the unlikelihood term, matching the reference
# implementation of Welleck et al. 2019 (arXiv:1908.04319). Caps a single
# token's unlikelihood loss at -log(1e-5) ~= 11.5 instead of letting it run to
# infinity for a token the model currently assigns probability ~1.
_UL_EPS = 1e-5


def weighted_lm_ce(
    lm_logits: Any,
    targets: Any,
    mask: Any,
    token_weights: Any,
) -> Any:
    """Per-token-weighted language-model loss: likelihood + unlikelihood.

    Each target token's contribution is selected and scaled by its weight from
    the data's reserved ``labels["lm_head"]`` channel:

    - ``w > 0`` (``1.0`` is the default) — normal training: ``w * -log p``,
      the usual cross-entropy, pushing the token's probability UP.
    - ``0.0`` — masked: the token contributes nothing (like ``-100`` labels).
    - ``w < 0`` (e.g. ``-1.0``) — **unlikelihood training** (Welleck et al.
      2019, arXiv:1908.04319): ``|w| * -log(1 - p)``, pushing the token's
      probability DOWN. ``|w|`` is exactly the paper's ``alpha`` mixing
      coefficient, so magnitudes other than 1 stay meaningful (``-5.0`` is
      five times the unlikelihood pressure, not a different objective).
    - ``-100`` (the ignore/padding sentinel) — treated as the DEFAULT ``1.0``:
      it marks "weight not specified" (padding, or a sample without an
      ``lm_head`` channel batched next to samples with one), never "mask".
      Positions outside the valid window are excluded by ``mask`` regardless.

    The result is a weighted mean: ``sum(per_token * mask) /
    max(sum(|w| * mask), eps)`` — with all-``1.0`` weights this reduces exactly
    to the unweighted ``JointOutputs.lm_ce``. The ``|w|`` denominator measures
    the amount of supervision signal (a masked token contributes none; an
    unlearned token contributes as much as a trained one).

    **Why unlikelihood and not negated CE.** Simply negating a token's CE is
    gradient ascent, which is unbounded below: pushing an already-impossible
    token further down keeps "improving" the loss forever, so the term
    eventually dominates and the model collapses. ``-log(1 - p)`` instead
    decays to 0 as ``p -> 0`` — once the token is unlikely, there is nothing
    left to gain. Every term here is therefore ``>= 0`` and the total loss is
    bounded below by 0, like any ordinary loss.

    ``p`` is recovered as ``exp(-ce)`` rather than by gathering from a
    ``softmax``: they are mathematically identical (``ce`` is already
    ``-log_softmax`` at the target), but this avoids materializing a
    ``[B, T, V]`` probability tensor over a ~100k-token vocabulary.
    ``1 - p`` is floored at :data:`_UL_EPS` exactly as in the paper's
    reference implementation, capping the loss (and its gradient) for a token
    the model is already certain of.

    Args:
        lm_logits: LM logits ``[B, T, V]`` (already next-token aligned).
        targets: Target token ids ``[B, T]``.
        mask: Boolean valid-window mask ``[B, T]``.
        token_weights: Per-target-token weights ``[B, T]`` (the shifted
            ``labels["lm_head"][:, 1:]`` channel; ``-100`` = unspecified).

    Returns:
        Scalar weighted-mean loss (fp32 accumulation), ``>= 0``.
    """
    from auto_chasm import ops

    # Per-token CE via the framework kernel (an ops one-hot over the ~100k-token
    # vocabulary would materialize a [B, T, V] intermediate — far too large).
    if tensor_backend(lm_logits) == "torch":
        import torch
        from torch.nn import functional

        ce_each = functional.cross_entropy(
            lm_logits.reshape(-1, lm_logits.shape[-1]),
            targets.reshape(-1).long(),
            reduction="none",
        ).reshape(targets.shape)
        w_raw = token_weights.to(torch.float32)
    else:
        import mlx.core as mx
        import mlx.nn as nn

        ce_each = nn.losses.cross_entropy(lm_logits, targets)
        w_raw = token_weights.astype(mx.float32)

    ones = _to_float32(w_raw * 0.0 + 1.0)  # ones_like, backend-agnostic
    w = ops.where(w_raw == -100.0, ones, w_raw)  # -100 = "unspecified" -> default 1.0
    m = _to_float32(mask)
    ce = _to_float32(ce_each)

    # Unlikelihood term for the negative-weight positions: -log(1 - p), with
    # p = exp(-ce) the target token's own probability. Both branches are
    # evaluated (ops.where selects afterwards), so both must stay finite for
    # every input or the unselected branch would poison the gradient -- the
    # _UL_EPS floor is what guarantees that here.
    p = ops.exp(-ce)
    ul = -ops.log(ops.clamp(ones - p, lo=_UL_EPS))

    per_token = ops.where(w >= 0.0, w * ce, ops.abs(w) * ul)
    num = ops.sum(per_token * m)
    denom = ops.clamp(ops.sum(ops.abs(w) * m), lo=1e-8)
    return num / denom


def _seq_target_and_mask(labels_shifted: Any, probe_mask: Any) -> tuple[Any, Any]:
    """Derive a per-sequence target and validity mask (backend-agnostic).

    Reduces the per-token shifted labels to one target per row by averaging over the
    valid (non-padding, non ``-100``) region, mirroring the masked response pooling
    on the probe side.  A single :mod:`auto_chasm.ops` code path replaces the former
    MLX/torch pair.

    Args:
        labels_shifted: Shifted labels ``[B, T-1]`` (``labels[:, 1:]``).
        probe_mask: Boolean validity mask ``[B, T-1]``.

    Returns:
        Tuple of ``(per_sequence_target [B], per_sequence_valid [B])``.
    """
    from auto_chasm import ops

    m = _to_float32(probe_mask)
    valid_counts = ops.sum(m, axis=1)
    safe_counts = ops.clamp(valid_counts, lo=1.0)
    seq_target = ops.sum(_to_float32(labels_shifted) * m, axis=1) / safe_counts
    seq_valid = valid_counts > 0
    return seq_target, seq_valid

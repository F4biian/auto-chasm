"""Backend-agnostic math ops for writing custom (joint) losses.

A custom loss should read the same on MLX and PyTorch.  These helpers
dispatch on the tensor's type (never on which framework is installed) so a
one-line custom joint loss like ``l_ce * lam * ops.exp(l_bce)`` runs
identically on both backends.

Example::

    from auto_chasm import ops
    from auto_chasm.outputs import JointOutputs

    def my_loss(model, batch, labels, lengths):
        lm_logits, probes = model(batch[:, :-1])
        o = JointOutputs(lm_logits, probes, batch[:, 1:], lengths)
        l_ce = o.lm_ce
        l_bce = o.probes["digit"].bce(labels[:, 1:], mask=o.mask)
        total = l_ce * 0.5 * ops.exp(l_bce)
        return total, o.ntoks, {"ce": l_ce, "bce": l_bce}
"""

from __future__ import annotations

from typing import Any

from auto_chasm.utils import tensor_backend

__all__ = [
    "exp",
    "log",
    "sqrt",
    "abs",
    "clamp",
    "pow",
    "sigmoid",
    "softplus",
    "where",
    "arange",
    "sum",
    "mean",
    "max",
    "argmax",
    "logsumexp",
    "softmax",
    "log_softmax",
    "zeros_like",
    "masked_mean",
]


def _mod(x: Any) -> Any:
    """Return the array module (``torch`` or ``mlx.core``) for ``x``."""
    if tensor_backend(x) == "torch":
        import torch

        return torch
    import mlx.core as mx

    return mx


def exp(x: Any) -> Any:
    """Element-wise exponential, backend-agnostic."""
    return _mod(x).exp(x)


def log(x: Any) -> Any:
    """Element-wise natural log, backend-agnostic."""
    return _mod(x).log(x)


def sqrt(x: Any) -> Any:
    """Element-wise square root, backend-agnostic."""
    return _mod(x).sqrt(x)


def abs(x: Any) -> Any:  # noqa: A001  (intentionally shadows builtin for symmetry)
    """Element-wise absolute value, backend-agnostic."""
    return _mod(x).abs(x)


def pow(x: Any, p: float) -> Any:  # noqa: A001
    """Element-wise power ``x ** p``, backend-agnostic."""
    return x**p


def clamp(x: Any, lo: float | None = None, hi: float | None = None) -> Any:
    """Clamp ``x`` to ``[lo, hi]`` (either bound optional), backend-agnostic.

    With both bounds ``None`` this is a no-op that returns ``x`` unchanged on
    *both* backends (torch's ``clamp`` would otherwise raise, so we early-return
    to match MLX).
    """
    if lo is None and hi is None:
        return x
    if tensor_backend(x) == "torch":
        return x.clamp(min=lo, max=hi)
    import mlx.core as mx

    if lo is not None:
        x = mx.maximum(x, lo)
    if hi is not None:
        x = mx.minimum(x, hi)
    return x


def sigmoid(x: Any) -> Any:
    """Element-wise logistic sigmoid, backend-agnostic."""
    if tensor_backend(x) == "torch":
        import torch

        return torch.sigmoid(x)
    import mlx.core as mx

    return mx.sigmoid(x)


def softplus(x: Any) -> Any:
    """Element-wise softplus ``log(1 + exp(x))``, backend-agnostic.

    Uses the overflow-safe identity ``softplus(x) = max(x, 0) + log1p(exp(-|x|))``
    so that large positive ``x`` (e.g. ``x >= 89`` in float32) returns the correct
    finite value with a finite gradient instead of ``inf``/``NaN`` from a naive
    ``log1p(exp(x))``.

    Args:
        x: Input tensor.

    Returns:
        Element-wise softplus of ``x``.
    """
    m = _mod(x)
    return clamp(x, lo=0.0) + m.log1p(m.exp(-m.abs(x)))


def where(cond: Any, a: Any, b: Any) -> Any:
    """Element-wise select, backend-agnostic."""
    return _mod(cond).where(cond, a, b)


def arange(n: int, like: Any, start: int = 0) -> Any:
    """Integer range ``[start, start + n)`` on ``like``'s backend (and device).

    Useful for building time-step masks inside a custom loss without writing
    backend-specific code.

    Args:
        n: Number of steps.
        like: A tensor whose backend/device to match.
        start: First value (default ``0``).

    Returns:
        A 1-D tensor ``[start, start+1, …, start+n-1]``.
    """
    if tensor_backend(like) == "torch":
        import torch

        return torch.arange(start, start + n, device=like.device)
    import mlx.core as mx

    return mx.arange(start, start + n)


def sum(  # noqa: A001  (intentionally shadows builtin for symmetry)
    x: Any, axis: int | None = None, keepdims: bool = False
) -> Any:
    """Sum-reduce ``x``, backend-agnostic.

    Args:
        x: Input tensor.
        axis: Axis to reduce, or ``None`` for a full reduction to a scalar.
        keepdims: Keep the reduced axis with size 1.  Ignored when ``axis is
            None`` (a full reduction always yields a scalar).

    Returns:
        The reduced tensor (a scalar when ``axis is None``).
    """
    if tensor_backend(x) == "torch":
        import torch

        if axis is None:
            return torch.sum(x)
        return torch.sum(x, dim=axis, keepdim=keepdims)
    import mlx.core as mx

    if axis is None:
        return mx.sum(x)
    return mx.sum(x, axis=axis, keepdims=keepdims)


def mean(x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
    """Mean-reduce ``x``, backend-agnostic.

    Pass float inputs: torch's ``mean`` raises on integer tensors (MLX would
    silently promote to float), so the two backends only agree on floats.

    Args:
        x: Input tensor.
        axis: Axis to reduce, or ``None`` for a full reduction to a scalar.
        keepdims: Keep the reduced axis with size 1.  Ignored when ``axis is
            None`` (a full reduction always yields a scalar).

    Returns:
        The reduced tensor (a scalar when ``axis is None``).
    """
    if tensor_backend(x) == "torch":
        import torch

        if axis is None:
            return torch.mean(x)
        return torch.mean(x, dim=axis, keepdim=keepdims)
    import mlx.core as mx

    if axis is None:
        return mx.mean(x)
    return mx.mean(x, axis=axis, keepdims=keepdims)


def max(  # noqa: A001  (intentionally shadows builtin for symmetry)
    x: Any, axis: int | None = None, keepdims: bool = False
) -> Any:
    """Max-reduce ``x``, backend-agnostic (values only).

    ``torch.max(x, dim=...)`` returns a ``(values, indices)`` namedtuple; this
    unwraps ``.values`` so both backends return only the maximum values.

    Args:
        x: Input tensor.
        axis: Axis to reduce, or ``None`` for a full reduction to a scalar.
        keepdims: Keep the reduced axis with size 1.  Ignored when ``axis is
            None`` (a full reduction always yields a scalar).

    Returns:
        The reduced maximum values (a scalar when ``axis is None``).
    """
    if tensor_backend(x) == "torch":
        import torch

        if axis is None:
            return torch.max(x)
        return torch.max(x, dim=axis, keepdim=keepdims).values
    import mlx.core as mx

    if axis is None:
        return mx.max(x)
    return mx.max(x, axis=axis, keepdims=keepdims)


def argmax(x: Any, axis: int = -1) -> Any:
    """Index of the maximum along ``axis``, backend-agnostic.

    The index dtype is backend-specific (MLX ``uint32``, torch ``int64``); cast
    with ``int(...)`` if you need a common integer dtype for downstream indexing.

    Args:
        x: Input tensor.
        axis: Axis to reduce (default ``-1``).

    Returns:
        Integer indices of the maxima along ``axis``.
    """
    if tensor_backend(x) == "torch":
        import torch

        return torch.argmax(x, dim=axis)
    import mlx.core as mx

    return mx.argmax(x, axis=axis)


def logsumexp(x: Any, axis: int = -1, keepdims: bool = False) -> Any:
    """Numerically-stable ``log(sum(exp(x)))`` along ``axis``, backend-agnostic.

    Args:
        x: Input tensor.
        axis: Axis to reduce (default ``-1``).
        keepdims: Keep the reduced axis with size 1.

    Returns:
        The log-sum-exp reduced along ``axis``.
    """
    if tensor_backend(x) == "torch":
        import torch

        return torch.logsumexp(x, dim=axis, keepdim=keepdims)
    import mlx.core as mx

    return mx.logsumexp(x, axis=axis, keepdims=keepdims)


def softmax(x: Any, axis: int = -1) -> Any:
    """Softmax along ``axis``, backend-agnostic.

    Args:
        x: Input tensor.
        axis: Axis over which to normalize (default ``-1``).

    Returns:
        The softmax of ``x`` along ``axis``.
    """
    if tensor_backend(x) == "torch":
        import torch

        return torch.softmax(x, dim=axis)
    import mlx.core as mx

    return mx.softmax(x, axis=axis)


def log_softmax(x: Any, axis: int = -1) -> Any:
    """Log-softmax along ``axis``, backend-agnostic.

    Implemented composably as ``x - logsumexp(x, axis, keepdims=True)`` on top of
    the facade's own :func:`logsumexp`, so no native ``log_softmax`` is required
    and the facade is proven to compose.

    Args:
        x: Input tensor.
        axis: Axis over which to normalize (default ``-1``).

    Returns:
        The log-softmax of ``x`` along ``axis``.
    """
    return x - logsumexp(x, axis=axis, keepdims=True)


def zeros_like(x: Any) -> Any:
    """A zero tensor with the same shape/dtype/backend as ``x``.

    Args:
        x: A tensor whose shape/dtype/backend to match.

    Returns:
        A tensor of zeros like ``x``.
    """
    if tensor_backend(x) == "torch":
        import torch

        return torch.zeros_like(x)
    import mlx.core as mx

    return mx.zeros_like(x)


def masked_mean(x: Any, mask: Any) -> Any:
    """Scalar mean of ``x`` over the elements selected by ``mask``, backend-agnostic.

    Computes ``sum(x * m) / max(sum(m), 1)`` as a full reduction to a scalar,
    where ``m`` is ``mask`` **broadcast to** ``x``'s shape first — so the numerator
    and denominator count the same elements even when ``mask`` is broadcastable but
    not the same shape as ``x`` (e.g. a ``[B, 1]`` per-row mask over ``[B, T]``;
    summing the un-broadcast mask in the denominator would divide by the wrong
    count).  The branchless ``lo=1`` floor makes an all-``False`` mask return ``0``
    rather than ``NaN``.

    This is the loss-normalization idiom (divide by the valid-element count) and is
    a **scalar** reduction — not the axis-wise, keepdims ``masked_mean_over_time``
    pooling used inside probe aggregation.

    Args:
        x: Input tensor.
        mask: A boolean or float tensor broadcastable to ``x``; nonzero entries
            select elements to average.

    Returns:
        The masked mean as a scalar.
    """
    if tensor_backend(x) == "torch":
        import torch

        m = torch.broadcast_to(mask, x.shape)
    else:
        import mlx.core as mx

        m = mx.broadcast_to(mask, x.shape)
    return sum(x * m) / clamp(sum(m), lo=1)

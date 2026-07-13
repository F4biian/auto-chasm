"""Output containers for multi-head training.

Provides structured loss and model outputs so that training loops
and logging code never need to unpack raw tuples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mlx.core as mx
    import torch


def _cast_like(x: Any, ref: Any) -> Any:
    """Cast tensor ``x`` to ``ref``'s dtype (backend-agnostic).

    Lets ``.bce()``/``.mse()`` accept integer labels without the caller
    writing a backend-specific cast.

    Args:
        x: Tensor to cast.
        ref: Tensor whose dtype to match.

    Returns:
        ``x`` as ``ref``'s dtype.
    """
    if hasattr(ref, "device"):  # torch
        return x.to(ref.dtype)
    return x.astype(ref.dtype)


def _check_class_indices(targets: Any, num_classes: int) -> None:
    """Validate cross-entropy class indices, ignoring the ``-100`` sentinel.

    MLX's ``cross_entropy`` gathers scores with no bounds check, so an
    out-of-range class index is silently wrong; torch raises ``IndexError``.
    This guard makes both backends raise an identical clear ``ValueError``.

    The check reads a scalar (``.item()``), which forces an evaluation. Inside an
    MLX function transformation (``value_and_grad`` / ``mx.compile`` — i.e. the
    training loop) that is disallowed and raises ``[eval] ...``. In that traced
    context the check is silently skipped: it cannot raise there anyway, and it
    still runs on every eager call (``ProbeOutput.ce``, the torch training path,
    direct loss calls), where mislabeled data is what actually needs catching.

    Args:
        targets: Class-index tensor (any backend).
        num_classes: Number of classes (the logits' last dimension).

    Raises:
        ValueError: If any non-``-100`` index lies outside ``[0, num_classes)``.
    """
    if hasattr(targets, "device"):  # torch
        valid = targets != -100
        in_range = (targets >= 0) & (targets < num_classes)
        ok = bool((in_range | ~valid).all().item())
    else:
        import mlx.core as mx

        valid = targets != -100
        in_range = (targets >= 0) & (targets < num_classes)
        try:
            ok = bool(mx.all(mx.logical_or(in_range, mx.logical_not(valid))).item())
        except ValueError:
            return  # traced (value_and_grad/compile): cannot eval, skip the check
    if not ok:
        raise ValueError(
            f"Class index out of range: every target must be -100 or in "
            f"[0, {num_classes}) for {num_classes}-class logits."
        )


def _masked_mean(values: Any, mask: Any | None) -> Any:
    """Mean of ``values`` over valid positions, branchlessly (graph-safe).

    Uses a clamped denominator instead of a Python ``if denom == 0`` check,
    which would evaluate a traced array and crash under ``value_and_grad``.

    Args:
        values: Per-element loss tensor.
        mask: Optional boolean mask (same shape); ``None`` means all valid.

    Returns:
        Scalar mean over the valid (masked) elements.
    """
    if mask is not None:
        if hasattr(values, "device"):  # torch
            import torch

            bmask = torch.broadcast_to(mask, values.shape)
            values = values * bmask
            denom = bmask.sum().clamp(min=1)
        else:
            import mlx.core as mx

            bmask = mx.broadcast_to(mask, values.shape)
            values = values * bmask
            denom = mx.maximum(bmask.sum(), 1)
        return values.sum() / denom
    n = values.numel() if hasattr(values, "numel") else values.size
    return values.sum() / max(n, 1)


def _combine_ignore_mask(targets: Any, mask: Any | None) -> Any:
    """Combine a user mask with a ``targets != -100`` ignore mask (any backend).

    ``-100`` is the universal ignore sentinel (matching ``JointLoss`` and the data
    pipeline): positions labeled ``-100`` are excluded from the loss mean. Without
    this, ``bce`` computes ``softplus(z) + 100*z`` at those positions and the loss
    explodes, and ``mse``/``mae`` square/abs a ``-100`` target — so the documented
    ``o.probes[name].bce(labels, mask=o.mask)`` recipe was silently wrong whenever
    the labels contained ``-100`` (special tokens, padding inside the length).

    Args:
        targets: The target tensor (any backend); ``-100`` marks ignored positions.
        mask: The caller's validity mask (float or bool), or ``None``.

    Returns:
        A boolean mask = ``(mask != 0) & (targets != -100)`` (or just the latter
        when ``mask`` is ``None``).
    """
    valid = targets != -100
    if mask is None:
        return valid
    if hasattr(targets, "device"):  # torch
        return valid & (mask != 0)
    import mlx.core as mx

    return mx.logical_and(valid, mask != 0)


def _to_f32(x: Any) -> Any:
    """Cast ``x`` to float32 on its own backend (branch on the tensor type)."""
    if hasattr(x, "device"):  # torch
        import torch

        return x.to(torch.float32)
    import mlx.core as mx

    return x.astype(mx.float32)


def _weighted_masked_mean_bce(per_elem: Any, targets: Any, mask: Any, weights: Any) -> Any:
    """Weighted masked mean of per-element BCE (the binary class-weight formula).

    ``weights[0]`` scales the negative (0) class and ``weights[1]`` the positive
    (1) class; a soft target ``t in [0, 1]`` gets the interpolated per-position
    weight ``w_neg + (w_pos - w_neg)*t``.  Divides by the SUMMED weight (a true
    weighted mean), guarding only the all-masked ``0/0`` with a tiny epsilon —
    never clamping the denominator to ``1.0`` (which would rescale the loss when
    the summed weights are ``< 1``).  A single :mod:`auto_chasm.ops` path, so MLX
    and torch agree numerically.

    Args:
        per_elem: Per-element BCE (``reduction="none"``), any shape.
        targets: Float targets in ``[0, 1]`` (``-100`` positions are excluded by
            ``mask``), same shape as ``per_elem``.
        mask: Boolean validity mask, same shape as ``per_elem``.
        weights: ``[w_neg, w_pos]`` — exactly two entries (binary).

    Returns:
        Scalar weighted-mean BCE.

    Raises:
        ValueError: If ``len(weights) != 2``.
    """
    from auto_chasm import ops

    if len(weights) != 2:
        raise ValueError(
            f"class_weights for a binary ('bce') probe needs exactly 2 entries "
            f"[w_neg, w_pos]; got {len(weights)}."
        )
    w_neg, w_pos = float(weights[0]), float(weights[1])
    # Clamp the target to [0, 1] so a masked -100 label cannot inflate the weight.
    t01 = ops.clamp(_to_f32(targets), lo=0.0, hi=1.0)
    w_each = w_neg + (w_pos - w_neg) * t01
    ww = w_each * _to_f32(mask)
    return ops.sum(per_elem * ww) / ops.clamp(ops.sum(ww), lo=1e-8)


@dataclass
class ProbeLossInfo:
    """Per-probe loss breakdown.

    Attributes:
        total: Weighted total loss for this probe.
        components: Named loss components (e.g., ``{"probe_bce": ..., "probe_mse": ...}``).
    """

    total: mx.array | torch.Tensor
    components: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossOutputs:
    """Structured loss container returned by the training step.

    The ``total`` field is the scalar that is differentiated.
    All other fields are for logging / debugging.

    Attributes:
        lm_ce: Language-modeling cross-entropy loss.
        probes: Per-probe loss breakdown.
        total: Weighted sum of all loss terms (the differentiable scalar).
    """

    lm_ce: mx.array | torch.Tensor | None = None
    probes: dict[str, ProbeLossInfo] = field(default_factory=dict)
    total: mx.array | torch.Tensor | None = None

    @property
    def all_components(self) -> dict[str, Any]:
        """Return a flat dict of all scalar loss components for logging."""
        result: dict[str, Any] = {}
        if self.lm_ce is not None:
            result["lm_ce"] = self.lm_ce
        for name, info in self.probes.items():
            result[name] = info.total
            for comp_name, comp_val in info.components.items():
                result[f"{name}_{comp_name}"] = comp_val
        if self.total is not None:
            result["total"] = self.total
        return result


@dataclass
class ProbeOutput:
    """Output from a single probe head.

    Attributes:
        logits: Raw probe logits.
        hidden_states: Captured hidden states (before probe module).
        weights: Attention weights or importance scores (optional).
        aggregated: Whether multi-layer aggregation was applied.
        mask: Optional bound validity mask so a custom loss can call
            ``probe.reduce(values)`` without re-passing the mask.
    """

    logits: mx.array | torch.Tensor | None = None
    hidden_states: Any = None
    weights: mx.array | torch.Tensor | None = None
    aggregated: bool = False
    mask: Any = None

    @property
    def n_classes(self) -> int:
        """Number of classes — the size of the logits' last axis.

        Returns:
            ``logits.shape[-1]`` as an ``int``.
        """
        assert self.logits is not None
        return int(self.logits.shape[-1])

    def softmax(self, axis: int = -1) -> mx.array | torch.Tensor:
        """Softmax of the probe logits along ``axis`` (backend-agnostic).

        Args:
            axis: Axis over which to normalize (default ``-1``).

        Returns:
            The softmax of ``logits`` along ``axis``.
        """
        from auto_chasm import ops

        assert self.logits is not None
        return ops.softmax(self.logits, axis)

    def log_softmax(self, axis: int = -1) -> mx.array | torch.Tensor:
        """Log-softmax of the probe logits along ``axis`` (backend-agnostic).

        Args:
            axis: Axis over which to normalize (default ``-1``).

        Returns:
            The log-softmax of ``logits`` along ``axis``.
        """
        from auto_chasm import ops

        assert self.logits is not None
        return ops.log_softmax(self.logits, axis)

    def reduce(
        self,
        values: mx.array | torch.Tensor,
        mask: mx.array | torch.Tensor | None = None,
    ) -> mx.array | torch.Tensor:
        """Masked mean of ``values``, using ``mask`` or the bound :attr:`mask`.

        The effective mask is ``mask`` when given, else the bound :attr:`mask`.
        When both are ``None`` this is a plain (unmasked) mean, so a custom loss
        can reduce a per-element loss to a scalar the same way on both backends.

        Args:
            values: Per-element values to reduce (any backend).
            mask: Optional validity mask; falls back to the bound :attr:`mask`.

        Returns:
            The scalar (masked) mean of ``values``.
        """
        from auto_chasm import ops

        eff_mask = mask if mask is not None else self.mask
        if eff_mask is None:
            return ops.mean(values)
        return ops.masked_mean(values, eff_mask)

    def bce(
        self,
        targets: mx.array | torch.Tensor,
        mask: mx.array | torch.Tensor | None = None,
        weights: Any = None,
    ) -> mx.array | torch.Tensor:
        """Binary cross-entropy loss with logits.

        Args:
            targets: Target labels matching logits shape. ``-100`` marks ignored
                positions and is excluded from the mean (matches ``JointLoss``).
            mask: Optional boolean mask (same shape); falls back to the bound
                :attr:`mask` (so a 2-arg custom loss respects padding automatically).
            weights: Optional binary class weights ``[w_neg, w_pos]``.  When given,
                each position's loss is scaled by ``w_neg`` (0-class) / ``w_pos``
                (1-class) — the BCE counterpart of ``weighted_ce`` — and the mean is
                a true weighted mean.  ``None`` (default) is the plain masked mean.

        Returns:
            Scalar BCE loss.
        """
        assert self.logits is not None
        mask = mask if mask is not None else self.mask
        targets = _cast_like(targets, self.logits)
        if hasattr(self.logits, "device"):
            from torch.nn import functional

            bce = functional.binary_cross_entropy_with_logits(
                self.logits, targets, reduction="none"
            )
        else:
            import mlx.nn as nn

            bce = nn.losses.binary_cross_entropy(
                self.logits, targets, reduction="none", with_logits=True
            )
        eff_mask = _combine_ignore_mask(targets, mask)
        if weights is None:
            return _masked_mean(bce, eff_mask)
        return _weighted_masked_mean_bce(bce, targets, eff_mask, weights)

    def mse(
        self,
        targets: mx.array | torch.Tensor,
        mask: mx.array | torch.Tensor | None = None,
    ) -> mx.array | torch.Tensor:
        """Mean squared error loss.

        Args:
            targets: Target values matching logits shape.
            mask: Optional boolean mask; falls back to the bound :attr:`mask`.

        Returns:
            Scalar MSE loss.

        Raises:
            ValueError: If ``targets.shape`` does not match ``logits.shape``, so
                torch does not silently broadcast where MLX would raise.
        """
        assert self.logits is not None
        mask = mask if mask is not None else self.mask
        targets = _cast_like(targets, self.logits)
        if tuple(targets.shape) != tuple(self.logits.shape):
            raise ValueError(
                f"mse targets shape {tuple(targets.shape)} does not match "
                f"logits shape {tuple(self.logits.shape)}."
            )
        if hasattr(self.logits, "device"):
            from torch.nn import functional

            mse = functional.mse_loss(self.logits, targets, reduction="none")
        else:
            import mlx.nn as nn

            mse = nn.losses.mse_loss(self.logits, targets, reduction="none")
        return _masked_mean(mse, _combine_ignore_mask(targets, mask))

    def mae(
        self,
        targets: mx.array | torch.Tensor,
        mask: mx.array | torch.Tensor | None = None,
    ) -> mx.array | torch.Tensor:
        """Mean absolute error loss.

        Mirrors :meth:`mse` but uses the absolute (L1) error, matching the
        first-class ``"mae"`` ``probe_loss`` available in ``JointLoss``.

        Args:
            targets: Target values matching logits shape.
            mask: Optional boolean mask; falls back to the bound :attr:`mask`.

        Returns:
            Scalar MAE loss.

        Raises:
            ValueError: If ``targets.shape`` does not match ``logits.shape``, so
                torch does not silently broadcast where MLX would raise.
        """
        assert self.logits is not None
        mask = mask if mask is not None else self.mask
        targets = _cast_like(targets, self.logits)
        if tuple(targets.shape) != tuple(self.logits.shape):
            raise ValueError(
                f"mae targets shape {tuple(targets.shape)} does not match "
                f"logits shape {tuple(self.logits.shape)}."
            )
        if hasattr(self.logits, "device"):
            diff = (self.logits - targets).abs()
        else:
            import mlx.core as mx

            diff = mx.abs(self.logits - targets)
        return _masked_mean(diff, _combine_ignore_mask(targets, mask))

    def ce(
        self,
        targets: mx.array | torch.Tensor,
        mask: mx.array | torch.Tensor | None = None,
    ) -> mx.array | torch.Tensor:
        """Multi-class cross-entropy loss.

        Args:
            targets: Class indices (long) matching ``logits.shape[:-1]``.
            mask: Optional boolean mask; falls back to the bound :attr:`mask`.

        Returns:
            Scalar CE loss.

        Raises:
            ValueError: If any class index in ``targets`` is out of range
                ``[0, num_classes)`` (ignoring the ``-100`` sentinel), so both
                backends fail identically instead of MLX silently wrapping.

        Notes:
            The ``-100`` ignore sentinel is folded into the loss mask on **both**
            backends (and the masked index is clamped before the gather so MLX
            does not gather an out-of-bounds class), so the result is the mean
            over the valid positions only — identical on MLX and PyTorch. An
            empty (zero-token) window returns a finite ``0.0`` rather than
            crashing MLX's ``logsumexp``.
        """
        assert self.logits is not None
        mask = mask if mask is not None else self.mask
        num_classes = self.logits.shape[-1]
        _check_class_indices(targets, num_classes)
        if hasattr(self.logits, "device"):
            import torch
            from torch.nn import functional

            valid = targets != -100
            safe = torch.where(valid, targets, torch.zeros_like(targets)).reshape(-1).long()
            flat = self.logits.reshape(-1, num_classes)
            if flat.shape[0] == 0:
                return flat.sum() * 0.0  # empty window -> finite 0.0
            ce = functional.cross_entropy(flat, safe, reduction="none").reshape(targets.shape)
            # `mask` may be float (0/1) or bool; `!= 0` normalizes it to bool so
            # `&` works regardless of the caller's mask dtype.
            combined = valid if mask is None else (valid & (mask != 0))
        else:
            import mlx.core as mx
            import mlx.nn as nn

            if self.logits.size == 0:
                return mx.array(0.0)  # empty window -> finite 0.0 (no logsumexp crash)
            valid = targets != -100
            safe = mx.where(valid, targets, mx.zeros_like(targets))
            ce = nn.losses.cross_entropy(self.logits, safe, reduction="none")
            combined = valid if mask is None else mx.logical_and(valid, mask != 0)
        return _masked_mean(ce, combined)


@dataclass
class ModelOutputs:
    """Output from a full model forward pass.

    Attributes:
        lm_logits: Language-model logits.
        probes: Per-probe outputs keyed by probe name.
        loss: Structured loss outputs (populated during training).
    """

    lm_logits: mx.array | torch.Tensor | None = None
    probes: dict[str, ProbeOutput] = field(default_factory=dict)
    loss: LossOutputs | None = None


@dataclass
class JointOutputs:
    """Structured outputs from a model forward pass for custom losses.

    Usage in a custom loss function::

        def my_loss(model, batch, labels, lengths):
            lm_logits, probe_dict = model(batch[:, :-1])
            outputs = JointOutputs(lm_logits, probe_dict, batch[:, 1:], lengths)

            total = outputs.lm_ce
            if "digit" in outputs.probes:
                total = total + outputs.probes["digit"].bce(
                    labels[:, 1:], mask=outputs.mask
                )
            return total, outputs.ntoks, {}

    Attributes:
        lm_logits: Language-model logits.
        probes: Per-probe outputs keyed by probe name.
        targets: Target tokens shifted by 1 (``batch[:, 1:]``).
        lengths: Per-sequence token ranges.
    """

    lm_logits: mx.array | torch.Tensor
    probes: dict[str, Any]
    targets: mx.array | torch.Tensor
    lengths: mx.array | torch.Tensor

    def __post_init__(self) -> None:
        """Wrap raw probe-logit tensors into ``ProbeOutput`` for ``.bce()`` etc.

        The trainable model returns ``(lm_logits, {name: raw_logits})``, so a
        custom loss can write ``outputs.probes["x"].bce(...)`` directly — raw
        arrays and already-wrapped ``ProbeOutput`` are both accepted.
        """
        self.probes = {
            name: val if isinstance(val, ProbeOutput) else ProbeOutput(logits=val)
            for name, val in self.probes.items()
        }

    @property
    def mask(self) -> mx.array | torch.Tensor:
        """Boolean mask for valid (non-padding) tokens."""
        if hasattr(self.targets, "device"):
            import torch

            steps = torch.arange(1, self.targets.shape[1] + 1, device=self.targets.device)
            return (steps >= self.lengths[:, 0:1]) & (steps < self.lengths[:, 1:])
        else:
            import mlx.core as mx

            steps = mx.arange(1, self.targets.shape[1] + 1)
            return mx.logical_and(
                steps >= self.lengths[:, 0:1],
                steps < self.lengths[:, 1:],
            )

    @property
    def ntoks(self) -> mx.array | torch.Tensor:
        """Number of valid (non-padding) tokens."""
        return self.mask.sum()

    @property
    def lm_ce(self) -> mx.array | torch.Tensor:
        """Language-model cross-entropy loss (guarded against zero tokens)."""
        if hasattr(self.lm_logits, "device"):
            from torch.nn import functional

            ce = (
                functional.cross_entropy(
                    self.lm_logits.reshape(-1, self.lm_logits.shape[-1]),
                    self.targets.reshape(-1).long(),  # torch CE requires int64 targets
                    reduction="none",
                ).reshape(self.targets.shape)
                * self.mask
            )
            denom = self.ntoks.clamp(min=1) if hasattr(self.ntoks, "clamp") else max(self.ntoks, 1)
            return ce.float().sum() / denom
        else:
            import mlx.core as mx
            import mlx.nn as nn

            ce = (
                nn.losses.cross_entropy(self.lm_logits, self.targets).astype(mx.float32) * self.mask
            )
            denom = mx.maximum(self.ntoks, 1)
            return ce.sum() / denom


@dataclass
class GenerationStep:
    """One step of probe-aware generation.

    Yielded by ``Model.generate_with_probes()`` at each autoregressive
    step.  Gives researchers full access to the probe's prediction and
    the raw next-token logits without touching backend internals.

    Attributes:
        token_id: The sampled token ID.
        token_str: The decoded token string.
        probes: Per-probe outputs at this step (logits, hidden states).
        next_logits: Full vocabulary logits for the next token (1-D).
    """

    token_id: int
    token_str: str
    probes: dict[str, ProbeOutput] = field(default_factory=dict)
    next_logits: mx.array | torch.Tensor | None = None

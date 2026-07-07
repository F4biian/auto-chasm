"""Multi-layer aggregation for probes (backend-agnostic).

Split out of ``probe.py`` to keep that module under the file-length limit.
Both functions dispatch on the backend by name (``"mlx"`` vs torch), never by
import availability, so a mixed-install machine stays correct.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def infer_callable_agg_dim(
    aggregation: Callable[..., Any],
    n_layers: int,
    hidden_dim: int,
    backend_name: str,
) -> int:
    """Infer the feature width a custom aggregation callable produces.

    A user ``aggregation`` callable may concatenate (width
    ``hidden * n_layers``), reduce (width ``hidden``), or emit any other width.
    Rather than assume concat width, probe it once with dummy per-layer states of
    shape ``[1, 1, hidden]`` and read its output's last dimension, so the probe
    head is sized to whatever it returns.

    Args:
        aggregation: The user aggregation callable (receives the per-layer list).
        n_layers: Number of layers the probe reads.
        hidden_dim: The model hidden size.
        backend_name: ``"mlx"`` or ``"torch"``.

    Returns:
        The feature width the aggregation callable outputs.

    Raises:
        ValueError: If the callable cannot be probed; pass
            ``module_config={'in_features': N}`` to size the head explicitly.
    """
    try:
        if backend_name == "mlx":
            import mlx.core as mx

            dummy: list[Any] = [mx.zeros((1, 1, hidden_dim)) for _ in range(n_layers)]
        else:
            import torch

            dummy = [torch.zeros((1, 1, hidden_dim)) for _ in range(n_layers)]
        return int(aggregation(dummy).shape[-1])
    except Exception as exc:
        raise ValueError(
            "Could not infer the output width of the custom aggregation "
            f"callable ({exc!r}). Pass module_config={{'in_features': N}} to "
            "size the probe head explicitly."
        ) from exc


def masked_mean_over_time(logits: Any, mask: Any, backend_name: str) -> Any:
    """Mean-pool ``[B, T, out]`` over time, ignoring padding when masked.

    Args:
        logits: Probe logits ``[B, T, out_dim]``.
        mask: Optional boolean ``[B, T]`` mask; ``None`` treats all positions as
            valid (correct for a single, unpadded sequence).
        backend_name: ``"mlx"`` or ``"torch"``.

    Returns:
        Pooled logits ``[B, out_dim]``.
    """
    if backend_name == "mlx":
        import mlx.core as mx

        if mask is None:
            return logits.mean(axis=1)
        m = mx.expand_dims(mask.astype(logits.dtype), -1)
        denom = mx.maximum(mx.sum(m, axis=1), 1e-9)
        return mx.sum(logits * m, axis=1) / denom

    if mask is None:
        return logits.mean(dim=1)
    m = mask.to(logits.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp(min=1e-9)
    return (logits * m).sum(dim=1) / denom


def call_custom_pooling(pooling: Callable[..., Any], logits: Any, mask: Any) -> Any:
    """Invoke a custom pooling callable, passing the mask when it accepts one.

    If the callable takes a second positional arg or a ``mask=`` keyword, the
    padding mask is forwarded; otherwise it is called with logits only, so
    single-arg poolers stay compatible.

    Args:
        pooling: User-supplied pooling callable.
        logits: Probe logits ``[B, T, out_dim]``.
        mask: Optional boolean ``[B, T]`` valid-position mask.

    Returns:
        The pooled logits returned by the callable.
    """
    import inspect

    try:
        sig = inspect.signature(pooling)
    except (TypeError, ValueError):
        return pooling(logits)

    params = list(sig.parameters.values())
    pos_kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    positional = [p for p in params if p.kind in pos_kinds]
    has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

    mask_param = sig.parameters.get("mask")
    accepts_mask_kw = mask_param is not None and mask_param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    if accepts_mask_kw:
        return pooling(logits, mask=mask)
    if len(positional) >= 2 or has_var_pos:
        return pooling(logits, mask)
    return pooling(logits)


def apply_pooling(
    config: Any,
    name: str,
    backend_name: str,
    logits: Any,
    mask: Any,
    input_ids: Any,
) -> Any:
    """Apply ``config.granularity`` pooling to per-token probe ``logits``.

    Args:
        config: The ``ProbeConfig`` (granularity, pooling, module_config).
        name: Probe name (for error messages).
        backend_name: ``"mlx"`` or ``"torch"``.
        logits: Raw per-token logits ``[B, T, out]``.
        mask: Optional boolean ``[B, T]`` valid-position mask.
        input_ids: Token ids ``[B, T]`` (needed for ``sentence``).

    Returns:
        Pooled logits; ``[B, T, out]`` for token/sentence, ``[B, out]`` for
        response, or whatever a custom pooler returns.

    Raises:
        ValueError: If ``granularity="sentence"`` but ``input_ids`` is absent.
    """
    g = config.granularity
    if g == "response":
        return masked_mean_over_time(logits, mask, backend_name)
    if g == "sentence":
        if input_ids is None:
            raise ValueError(
                f"Probe {name!r}: granularity='sentence' needs input_ids to find "
                "sentence boundaries, but none were provided to forward()."
            )
        delims = config.module_config["sentence_delimiters"]
        return sentence_pool(logits, input_ids, delims, mask, backend_name)
    if g == "custom" and config.pooling is not None:
        return call_custom_pooling(config.pooling, logits, mask)
    return logits  # "token", or "custom" with no pooling callable


def sentence_pool(
    logits: Any,
    input_ids: Any,
    delimiters: list[int],
    mask: Any,
    backend_name: str,
) -> Any:
    """Mean-pool per-token logits within each sentence and broadcast back.

    A sentence ends at (and includes) a delimiter token; the next token starts a
    new one. Each token's output becomes the mean of its sentence's logits over
    the valid ``mask`` (so padding never leaks in), broadcast back to every token
    of that sentence. The output keeps the ``[B, T, out]`` shape, so per-token
    labels still align.

    Args:
        logits: Per-token probe logits ``[B, T, out]``.
        input_ids: Token ids ``[B, T]`` used to locate sentence boundaries.
        delimiters: Token ids that END a sentence (e.g. ids of ``.``/``!``/``?``).
        mask: Optional boolean ``[B, T]`` valid-position mask (excludes padding).
        backend_name: ``"mlx"`` or ``"torch"``.

    Returns:
        Sentence-pooled logits ``[B, T, out]``.
    """
    if backend_name == "mlx":
        import mlx.core as mx

        is_delim = mx.zeros(input_ids.shape, dtype=mx.float32)
        for d in delimiters:
            is_delim = mx.maximum(is_delim, (input_ids == d).astype(mx.float32))
        seg = (mx.cumsum(is_delim, axis=1) - is_delim).astype(mx.int32)  # [B, T]
        n_seg = int(seg.max()) + 1
        onehot = (seg[..., None] == mx.arange(n_seg)[None, None, :]).astype(mx.float32)
        m = (mask.astype(mx.float32) if mask is not None else mx.ones(seg.shape))[..., None]
        masked = onehot * m  # [B, T, S]
        counts = masked.sum(axis=1)  # [B, S]
        seg_sum = mx.matmul(masked.transpose(0, 2, 1), logits)  # [B, S, out]
        seg_mean = seg_sum / mx.maximum(counts, 1)[..., None]
        return mx.matmul(onehot, seg_mean)  # [B, T, out]

    import torch

    is_delim = torch.zeros_like(input_ids, dtype=torch.float32)
    for d in delimiters:
        is_delim = torch.maximum(is_delim, (input_ids == d).float())
    seg = (torch.cumsum(is_delim, dim=1) - is_delim).long()  # [B, T]
    n_seg = int(seg.max().item()) + 1
    rng = torch.arange(n_seg, device=seg.device)
    onehot = (seg.unsqueeze(-1) == rng).float()  # [B, T, S]
    valid = mask.float() if mask is not None else torch.ones_like(seg, dtype=torch.float32)
    m = valid.unsqueeze(-1)
    masked = onehot * m
    counts = masked.sum(dim=1)  # [B, S]
    seg_sum = torch.bmm(masked.transpose(1, 2), logits)  # [B, S, out]
    seg_mean = seg_sum / counts.clamp(min=1).unsqueeze(-1)
    return torch.bmm(onehot, seg_mean)  # [B, T, out]


def unwrap_submodule_captures(layers: Any) -> None:
    """Restore any attention/mlp submodule replaced by a capture wrapper.

    Sub-block sources (``attention``/``mlp``) wrap a *submodule* of a block; the
    block-level ``restore_original_layers`` does not touch those. This walks each
    block's known submodule attributes and swaps a capture wrapper back for the
    original module it holds under ``.layer``.

    Args:
        layers: The model's transformer block list.
    """
    from auto_chasm.probe import _SUBMODULE_ATTRS

    capture_types = ("_MLXLayerCapture", "_TorchLayerCapture")
    for block in layers:
        for names in _SUBMODULE_ATTRS.values():
            for name in names:
                child = getattr(block, name, None)
                if child is not None and type(child).__name__ in capture_types:
                    setattr(block, name, child.layer)


def aggregate(
    hidden_states: list[Any],
    strategy: str | Callable[..., Any],
    backend_name: str,
) -> Any:
    """Aggregate multi-layer hidden states by ``strategy``.

    Args:
        hidden_states: List of per-layer hidden-state tensors (config order).
        strategy: ``"concat"``, ``"mean"``, ``"max"``, ``"last"``, or a callable
            that receives the per-layer list and returns the aggregated tensor.
        backend_name: ``"mlx"`` or ``"torch"``.

    Returns:
        The aggregated tensor.

    Raises:
        ValueError: If ``strategy`` is an unknown string.
    """
    if callable(strategy) and not isinstance(strategy, str):
        return strategy(hidden_states)

    is_mlx = backend_name == "mlx"

    if strategy == "concat":
        if is_mlx:
            import mlx.core as mx

            return mx.concatenate(hidden_states, axis=-1)
        import torch

        return torch.cat(hidden_states, dim=-1)
    if strategy == "mean":
        if is_mlx:
            import mlx.core as mx

            return mx.mean(mx.stack(hidden_states, axis=0), axis=0)
        import torch

        return torch.mean(torch.stack(hidden_states, dim=0), dim=0)
    if strategy == "max":
        if is_mlx:
            import mlx.core as mx

            return mx.max(mx.stack(hidden_states, axis=0), axis=0)
        import torch

        return torch.max(torch.stack(hidden_states, dim=0), dim=0)[0]
    if strategy == "last":
        return hidden_states[-1]
    raise ValueError(f"Unknown aggregation strategy: {strategy!r}")

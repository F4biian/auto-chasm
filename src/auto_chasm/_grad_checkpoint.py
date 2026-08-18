"""Gradient (activation) checkpointing — trade recompute for activation memory.

Without it, every intermediate a transformer block produces is retained until the
backward pass, so peak memory grows LINEARLY with sequence length and batch size.
Measured on Qwen3.5-0.8B (MLX, LoRA r=8, batch 1): peak was ``~42 MB per token``
-- 7.9 GB at 131 tokens, 28.4 GB at 623, 55.2 GB at 623x2. A 0.8B model on a
64 GB machine ran out of memory at ordinary sequence lengths, and neither the
loss nor gradient accumulation was responsible (measured: unlikelihood weights
cost 0.00 GB extra, and accumulating 4 micro-batches cost 0.6 GB).

Checkpointing keeps only each block's INPUT and recomputes the interior during
backward, which turns that linear term over ~24 blocks into roughly one block's
worth. The cost is one extra forward pass (~30% slower per step).

Both backends patch at the block level, but differently:

* **MLX** wraps ``type(block).__call__`` in ``mx.checkpoint``. MLX resolves
  ``__call__`` on the CLASS, so this necessarily affects every instance of that
  block type in the process -- see :func:`enable`'s note.
* **PyTorch** delegates to transformers' own
  ``gradient_checkpointing_enable``, plus ``enable_input_require_grads`` which
  is REQUIRED with a frozen LoRA base: checkpointing needs at least one input
  with ``requires_grad``, and a fully frozen base has none, which otherwise
  fails with "none of the output has requires_grad=True".
"""

from __future__ import annotations

from typing import Any

from auto_chasm.logger import get_logger

logger = get_logger(__name__)

#: Set on a patched block CLASS so a second enable() is a no-op rather than
#: wrapping the wrapper (which would recompute twice and cost more, not less).
_FLAG = "_auto_chasm_checkpointed"
#: Where the pre-patch ``__call__`` is stashed so :func:`disable` can restore it.
_ORIG = "_auto_chasm_original_call"


def _blocks(model: Any) -> Any:
    """The transformer block list, or raise with a usable message."""
    from auto_chasm.probe import _find_layers

    layers = _find_layers(model)
    if layers is None or len(layers) == 0:
        raise RuntimeError(
            "Could not locate the transformer block list on this model, so gradient "
            "checkpointing cannot be applied. This is the same block list probes attach "
            "to; if probes work on this model, please report it as a bug."
        )
    return layers


def _enable_mlx(model: Any) -> int:
    """Wrap each distinct block type's ``__call__`` in ``mx.checkpoint``."""
    import mlx.core as mx

    patched = 0
    for block in _blocks(model):
        cls = type(block)
        if getattr(cls, _FLAG, False):
            continue

        def _wrap(fn: Any) -> Any:
            # Factory, not a closure over the loop variable: with several block
            # types (hybrid attention stacks have more than one) a late-binding
            # closure would give every type the LAST type's forward.
            def checkpointed(self: Any, *args: Any, **kwargs: Any) -> Any:
                def inner(params: Any, *a: Any, **kw: Any) -> Any:
                    self.update(params)
                    return fn(self, *a, **kw)

                return mx.checkpoint(inner)(self.trainable_parameters(), *args, **kwargs)

            return checkpointed

        setattr(cls, _ORIG, cls.__call__)
        cls.__call__ = _wrap(cls.__call__)
        setattr(cls, _FLAG, True)
        patched += 1
    return patched


def _disable_mlx(model: Any) -> int:
    """Restore the original ``__call__`` on each patched block type."""
    restored = 0
    for block in _blocks(model):
        cls = type(block)
        if not getattr(cls, _FLAG, False):
            continue
        cls.__call__ = getattr(cls, _ORIG)
        delattr(cls, _ORIG)
        delattr(cls, _FLAG)
        restored += 1
    return restored


def _torch_target(model: Any) -> Any:
    """The object carrying transformers' checkpointing API (past PEFT wrappers)."""
    for candidate in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        if candidate is not None and hasattr(candidate, "gradient_checkpointing_enable"):
            return candidate
    raise RuntimeError(
        "This torch model does not expose gradient_checkpointing_enable(); it is provided "
        "by transformers' PreTrainedModel. Gradient checkpointing is unavailable for a "
        "custom nn.Module base."
    )


def _enable_torch(model: Any) -> int:
    """Turn on transformers checkpointing, and make it work with a frozen base."""
    target = _torch_target(model)
    # use_reentrant=False is the non-deprecated implementation and the one that
    # composes with frozen parameters and hooks.
    target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # REQUIRED with LoRA: the base is frozen, so without this no input to a
    # checkpointed block requires grad and autograd silently returns no
    # gradient for the block's interior.
    if hasattr(target, "enable_input_require_grads"):
        target.enable_input_require_grads()
    return 1


def _disable_torch(model: Any) -> int:
    """Turn transformers checkpointing back off."""
    target = _torch_target(model)
    if hasattr(target, "gradient_checkpointing_disable"):
        target.gradient_checkpointing_disable()
        return 1
    return 0


def enable(model: Any, backend_name: str) -> int:
    """Enable gradient checkpointing on ``model``'s transformer blocks.

    Args:
        model: The wrapped base language model.
        backend_name: ``"mlx"`` or ``"torch"``.

    Returns:
        Number of block types patched (MLX) or ``1`` (torch).

    Note:
        On MLX this patches the block CLASS, because MLX looks ``__call__`` up on
        the type rather than the instance. Every model of that block type in the
        process therefore gets checkpointing, not just this one. That is also how
        ``mlx_lm.tuner.trainer.grad_checkpoint`` behaves; call
        :func:`disable` to undo it.
    """
    n = _enable_mlx(model) if backend_name == "mlx" else _enable_torch(model)
    logger.info("Gradient checkpointing enabled (%s, %d block type(s) patched).", backend_name, n)
    return n


def disable(model: Any, backend_name: str) -> int:
    """Undo :func:`enable`. Returns how many block types were restored."""
    n = _disable_mlx(model) if backend_name == "mlx" else _disable_torch(model)
    logger.info("Gradient checkpointing disabled (%s, %d restored).", backend_name, n)
    return n


# --- diagnosing architectures that unroll a recurrence during training -------

#: Block attributes that mark a linear-attention / state-space mixer. Such layers
#: keep a per-head STATE matrix; when the differentiable path is an unrolled loop
#: over timesteps, one state per timestep is retained for backward.
_RECURRENT_ATTRS = ("linear_attn", "mamba", "ssm", "gated_delta", "recurrent")


def unrolled_recurrence_layers(model: Any) -> int:
    """How many blocks carry a linear-attention / state-space mixer (0 if none).

    Detected structurally rather than by model name, so it holds for any
    architecture that mixes such layers in.
    """
    from auto_chasm.probe import _find_layers

    layers = _find_layers(model)
    if layers is None:
        return 0
    return sum(1 for b in layers if any(hasattr(b, a) for a in _RECURRENT_ATTRS))


def memory_warning(model: Any) -> str | None:
    """A warning about training-time activation memory, or ``None`` if not applicable.

    Linear-attention blocks are usually implemented with a fused kernel that has
    no vjp, so the DIFFERENTIABLE path is a loop over timesteps -- mlx-lm's
    ``gated_delta_update`` selects it with ``use_kernel=not self.training``. Every
    timestep's ``[B, heads, Dv, Dk]`` state is then retained for backward, so peak
    memory grows linearly with sequence length at a rate set by the state size,
    not by the parameter count. Measured on Qwen3.5-0.8B: 1.0 MB per timestep per
    layer x 18 such layers = 18 MB/token of state, ~48 MB/token in total -- 64 GB
    exhausted at ~1300 tokens by a 0.8B model.

    The same unrolled loop also multiplies the number of distinct buffers in one
    graph, which is what surfaces as
    ``[metal::malloc] Resource limit (499000) exceeded`` at longer sequences or
    higher ``grad_accum_steps``.

    Returns:
        The warning text, or ``None`` when the model has no such layers.
    """
    n = unrolled_recurrence_layers(model)
    if n == 0:
        return None
    already = False
    from auto_chasm.probe import _find_layers

    layers = _find_layers(model)
    if layers is not None and len(layers) > 0:
        already = any(getattr(type(b), _FLAG, False) for b in layers)
    hint = (
        "Gradient checkpointing is ON, which is the main mitigation available here."
        if already
        else "Call model.enable_gradient_checkpointing() -- measured 3-4x lower peak "
        "on this architecture, at ~15-30% slower steps."
    )
    return (
        f"{n} block(s) use linear-attention / state-space mixing. During TRAINING these "
        f"fall back to an unrolled per-timestep loop (the fused kernel has no backward), "
        f"so activation memory grows linearly with sequence length at tens of MB per "
        f"token -- far above what the parameter count suggests. {hint} Shorter "
        f"max_seq_length and batch_size=1 also help; grad_accum_steps does NOT (measured: "
        f"+0.6 GB from 1 to 4), but high values can trip the Metal buffer-count limit."
    )

"""Torch per-step training escape hatch (``Trainer.step`` on the torch backend).

Mirrors the MLX ``JointTrainer.step`` escape hatch: a persistent optimizer +
LR scheduler built once and advanced one optimizer update per call.  The
``_TorchProbeWrapper`` shares parameter tensors with the ``Model``, so updates
made here are visible to ``evaluate()`` (which reads the same ``Model``).

Kept in its own module so ``trainer.py`` stays under the project file-length
limit and the torch-specific helpers live next to the other torch trainer
helpers (``_metrics.py``, ``wrappers.py``).
"""

from __future__ import annotations

from typing import Any
from weakref import WeakKeyDictionary

from auto_chasm.trainers.wrappers import _TorchProbeWrapper

# Original base dtype per model, recorded before the first mixed-precision cast so a
# later fp32/fp16 run on the SAME model restores it instead of silently keeping bf16.
_ORIG_BASE_DTYPE: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()


def apply_base_precision(raw_model: Any, mixed_precision: str) -> None:
    """Cast the base to the dtype implied by ``mixed_precision``, idempotently.

    ``bf16`` casts the (frozen) base to bfloat16; ``fp32``/``fp16`` keep fp32 weights
    (fp16 runs the forward under autocast). The cast mutates the base in place and
    persists, so the original dtype is recorded once and RESTORED for a non-bf16 run —
    otherwise a second trainer built with fp32/fp16 on a reused ``Model`` would silently
    train on the stale bf16 base (a reproducibility hazard for precision comparisons).

    Args:
        raw_model: The base torch model (``Model.model``).
        mixed_precision: ``"fp32"``, ``"bf16"``, or ``"fp16"``.
    """
    import torch

    orig = _ORIG_BASE_DTYPE.setdefault(raw_model, next(raw_model.parameters()).dtype)
    raw_model.to(torch.bfloat16 if mixed_precision == "bf16" else orig)


def build_torch_optim_sched(
    model: Any,
    learning_rate: float,
    weight_decay: float,
    lr_schedule: str,
    warmup_ratio: float,
    n_updates: int,
) -> tuple[Any, Any]:
    """Build the AdamW optimizer and LR scheduler for the torch backend.

    Shared by ``Trainer.train()`` and the ``step()`` escape hatch so the two
    paths cannot drift apart.  ``n_updates`` is the scheduler horizon: the
    number of optimizer *updates* the caller will perform (one per accumulation
    group in ``train()``, one per ``step()`` call in the manual loop).

    Implements ``warmup_ratio`` to match the MLX schedule: a linear ramp
    ``0 -> peak_lr`` over ``int(n_updates * warmup_ratio)`` updates, followed by
    cosine/linear/constant decay over the remaining updates.  With
    ``warmup_ratio == 0`` this reduces exactly to the previous
    cosine/linear/constant schedulers (``eta_min=0``).

    Args:
        model: The ``_TorchProbeWrapper`` whose ``requires_grad`` parameters
            are optimized.
        learning_rate: Peak learning rate (AdamW base lr).
        weight_decay: AdamW weight decay.
        lr_schedule: One of ``"cosine"``, ``"linear"``, ``"constant"``.
        warmup_ratio: Fraction of ``n_updates`` spent on the linear warmup ramp.
        n_updates: Total optimizer updates (scheduler horizon).

    Returns:
        Tuple of ``(optimizer, scheduler)``.
    """
    import math

    import torch

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    warmup_steps = int(n_updates * warmup_ratio)
    # Clamp so warmup never swallows the whole run (mirrors build_lr_schedule):
    # otherwise the peak LR is never reached and the decay is silently dropped.
    if warmup_steps >= n_updates and n_updates >= 1:
        warmup_steps = n_updates - 1
    main_steps = max(n_updates - warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        """Return the LR multiplier in ``[0, 1]`` for optimizer update ``step``.

        ``step`` is the 0-indexed optimizer-update count: ``LambdaLR`` sets the
        LR for update ``it`` (1-indexed) to ``base * lr_lambda(it - 1)``,
        matching MLX where that update uses ``schedule(it - 1)`` — so the first
        update sits at the bottom of the warmup ramp (LR 0).
        """
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = min(step - warmup_steps, main_steps) / main_steps
        if lr_schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if lr_schedule == "linear":
            return 1.0 - progress
        return 1.0  # constant

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def get_torch_step_state(trainer: Any) -> dict[str, Any]:
    """Build (once) and cache the persistent state for ``trainer.step()`` on torch.

    Mirrors ``Trainer._train_torch``'s setup: a ``_TorchProbeWrapper`` over the
    same model + probe modules, plus an AdamW optimizer and LR scheduler.  One
    ``step()`` performs one optimizer update, so the scheduler horizon is
    ``num_iters`` (not ``num_iters // accum``).

    Args:
        trainer: The ``Trainer`` whose ``_torch_step_state`` is populated.

    Returns:
        Dict with ``"model"``, ``"optimizer"``, ``"scheduler"`` keys.

    Raises:
        RuntimeError: If no parameters require gradients (nothing to train).
    """
    if trainer._torch_step_state is not None:
        return trainer._torch_step_state  # type: ignore[no-any-return]

    import torch

    raw_model = trainer.model.model
    model = _TorchProbeWrapper(raw_model, trainer.model._probes)
    model.train()

    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError(
            "Trainer.step(): no trainable parameters. Unfreeze the probes "
            "and/or model (e.g. model.unfreeze_all_probes()) before stepping."
        )

    # Mixed precision (see _torch_loop): bf16 casts the frozen base; fp16 keeps
    # weights fp32 and uses autocast + a GradScaler in torch_step.
    mp = getattr(trainer, "_mixed_precision", "fp32")
    device_type = next(raw_model.parameters()).device.type
    apply_base_precision(raw_model, mp)

    optimizer, scheduler = build_torch_optim_sched(
        model,
        trainer.learning_rate,
        trainer.weight_decay,
        trainer.lr_schedule,
        trainer.warmup_ratio,
        max(1, trainer.num_iters),
    )
    trainer._torch_step_state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "amp_dtype": torch.float16 if mp == "fp16" else None,
        "device_type": device_type,
        "scaler": torch.amp.GradScaler(device_type, enabled=(mp == "fp16")),
    }
    return trainer._torch_step_state  # type: ignore[no-any-return]


def torch_step(trainer: Any, batch: tuple[Any, Any, Any]) -> dict[str, Any]:
    """Run one torch training step (forward + backward + optimizer update).

    Args:
        trainer: The ``Trainer`` driving the step (holds loss_fn, state, config).
        batch: ``(tokens, labels, lengths)`` tuple of numpy arrays.

    Returns:
        Dict with keys ``"loss"`` (float), ``"ntoks"`` (int), ``"components"``
        (dict of str → float) — matching the MLX escape-hatch contract.
    """
    from contextlib import nullcontext

    import torch

    from auto_chasm.trainers.data_utils import labels_to_torch

    state = get_torch_step_state(trainer)
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    scaler = state["scaler"]
    amp_dtype = state["amp_dtype"]
    device_type = state["device_type"]

    # evaluate() flips the shared base model to eval mode; restore train mode
    # each step so dropout/batchnorm behave as during training.
    model.train()

    tokens, labels, lengths = batch
    device = model.device if hasattr(model, "device") else "cpu"
    tokens = torch.from_numpy(tokens).to(device)
    labels = labels_to_torch(labels, device)
    lengths = torch.from_numpy(lengths).to(device)

    optimizer.zero_grad()
    with torch.autocast(device_type, dtype=amp_dtype) if amp_dtype else nullcontext():
        total, ntoks, components = trainer.loss_fn(model, tokens, labels, lengths)
    scaler.scale(total).backward()
    if trainer.grad_clip_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            trainer.grad_clip_norm,
        )
    prev_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    # Advance the LR schedule only when the optimizer actually stepped (fp16 skips the
    # step on grad overflow and backs the scale off — see _torch_loop for the rationale).
    if scaler.get_scale() >= prev_scale:
        scheduler.step()

    return {
        "loss": float(total.item()),
        "ntoks": int(ntoks.item()) if hasattr(ntoks, "item") else int(ntoks),
        "components": {
            k: float(v.item() if hasattr(v, "item") else v) for k, v in components.items()
        },
    }

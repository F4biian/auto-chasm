"""Extracted PyTorch training loop.

Holds the body of ``Trainer._train_torch`` so ``trainer.py`` stays under the
project file-length limit.  ``train_torch`` receives the live ``Trainer`` and
reads its config / calls its ``_evaluate_torch`` / ``_fire_callback`` methods,
so behavior is byte-identical to the inline loop it replaced.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

from auto_chasm.history import History, HistoryEntry
from auto_chasm.trainers.wrappers import _TorchProbeWrapper


def train_torch(
    trainer: Any,
    train_data: Any,
    val_data: Any | None,
    test_data: Any | None,
) -> dict[str, Any]:
    """Run the PyTorch training loop with validation, early stopping, checkpointing.

    Args:
        trainer: The ``Trainer`` driving the run (holds loss_fn, config, and the
            ``_evaluate_torch`` / ``_fire_callback`` methods).
        train_data: Training dataset.
        val_data: Validation dataset (optional, enables early stopping).
        test_data: Test dataset (optional, evaluated after training).

    Returns:
        Dict with keys ``"history"``, ``"test_metrics"``, ``"output_dir"``.
    """
    import torch

    from auto_chasm.trainers._torch_step import apply_base_precision, build_torch_optim_sched
    from auto_chasm.trainers.data_utils import iterate_batches, labels_to_torch

    raw_model = trainer.model.model
    trainer.output_dir.mkdir(parents=True, exist_ok=True)

    # Wrap the raw model + probes so the loss function receives
    # (lm_logits, probe_logits_dict) — same contract as MLX's
    # _TrainableModel.__call__.
    model = _TorchProbeWrapper(raw_model, trainer.model._probes)
    model.train()

    # Mixed precision: bf16 casts the frozen base to bfloat16 (probes/optimizer stay
    # fp32; bf16 shares fp32's range so no scaling is needed). fp16 keeps weights fp32
    # but runs the forward under autocast + a GradScaler (fp16's narrow range needs
    # loss scaling). fp32 -> both are no-ops.
    mp = getattr(trainer, "_mixed_precision", "fp32")
    device_type = next(raw_model.parameters()).device.type
    apply_base_precision(raw_model, mp)  # bf16 casts the base; fp32/fp16 restore fp32 weights
    amp_dtype = torch.float16 if mp == "fp16" else None
    scaler = torch.amp.GradScaler(device_type, enabled=(mp == "fp16"))

    def autocast_ctx() -> Any:
        """Return an autocast context for fp16, else a no-op (bf16/fp32)."""
        return torch.autocast(device_type, dtype=amp_dtype) if amp_dtype else nullcontext()

    # The scheduler advances once per optimizer *update*, not once per
    # micro-batch, so its horizon is the number of accumulation groups.
    accum = max(1, trainer.grad_accum_steps)
    n_updates = max(1, trainer.num_iters // accum)
    optimizer, scheduler = build_torch_optim_sched(
        model,
        trainer.learning_rate,
        trainer.weight_decay,
        trainer.lr_schedule,
        trainer.warmup_ratio,
        n_updates,
    )

    history = History()
    trainer._torch_history = history

    best_metric = float("-inf") if trainer.early_stopping_higher_is_better else float("inf")
    patience_counter = 0
    best_iter = 0
    best_state: dict[str, Any] | None = None
    from auto_chasm import _grad_checkpoint

    _warning = _grad_checkpoint.memory_warning(getattr(trainer.model, "model", trainer.model))
    if _warning and trainer.verbose:
        print(f"  [memory] {_warning}")
    es_active = trainer.early_stopping_patience > 0 and val_data is not None
    eval_steps = trainer.eval_steps if trainer.eval_steps is not None else trainer.save_steps
    wall_start = time.perf_counter()

    for global_step, batch in zip(
        range(1, trainer.num_iters + 1),
        iterate_batches(
            train_data, trainer.batch_size, trainer.max_seq_length, loop=True, seed=trainer.seed
        ),
        strict=False,
    ):
        tokens, labels, lengths = batch
        device = model.device if hasattr(model, "device") else "cpu"
        tokens = torch.from_numpy(tokens).to(device)
        labels = labels_to_torch(labels, device)
        lengths = torch.from_numpy(lengths).to(device)

        with autocast_ctx():
            total, _, components = trainer.loss_fn(model, tokens, labels, lengths)
        # Scale so accumulated gradients average (not sum) across the group; the
        # GradScaler (fp16) is a no-op for bf16/fp32.
        scaler.scale(total / accum).backward()

        is_boundary = (global_step % accum == 0) or (global_step == trainer.num_iters)
        if is_boundary:
            if trainer.grad_clip_norm > 0:
                scaler.unscale_(optimizer)  # unscale before clipping (no-op when disabled)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    trainer.grad_clip_norm,
                )
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only advance the LR schedule when the optimizer actually stepped. On
            # fp16 grad overflow (normal while the scaler calibrates) scaler.step is a
            # no-op and update() backs the scale off, so a lower scale means the step
            # was skipped — advancing the schedule anyway would skip the peak LR.
            if scaler.get_scale() >= prev_scale:
                scheduler.step()
            optimizer.zero_grad()

        loss_val = total.item()

        # --- Logging ---
        if global_step % trainer.logging_steps == 0:
            lr = optimizer.param_groups[0]["lr"]
            comp_parts = ", ".join(
                f"{k} {v.item() if hasattr(v, 'item') else v:.4f}" for k, v in components.items()
            )
            if trainer.verbose:
                print(f"Step {global_step}: loss={loss_val:.4f} ({comp_parts}), LR {lr:.3e}")

            history.record(
                HistoryEntry(
                    step=global_step,
                    train_loss=loss_val,
                    loss_components={
                        k: v.item() if hasattr(v, "item") else float(v)
                        for k, v in components.items()
                    },
                    learning_rate=lr,
                    wall_time=time.perf_counter() - wall_start,
                )
            )

        trainer._fire_callback(
            "on_step_end",
            step=global_step,
            loss=loss_val,
            components=components,
        )

        # --- Validation + early stopping ---
        should_eval = (
            val_data is not None
            and eval_steps > 0
            and (global_step % eval_steps == 0 or global_step >= trainer.num_iters)
        )
        if should_eval:
            val_metrics = trainer._evaluate_torch(val_data)
            val_loss = val_metrics["loss"]

            # Record val metrics on a dedicated entry AT this eval step (mirrors the
            # MLX loop). Mutating history[-1] mis-attributed them to the last *logging*
            # entry — a different step when eval_steps != logging_steps — and dropped
            # them entirely when no logging entry existed yet (e.g. first eval).
            history.record(
                HistoryEntry(
                    step=global_step,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    wall_time=time.perf_counter() - wall_start,
                )
            )

            model.train()

            if trainer.verbose:
                print(f"  Step {global_step}: val_loss={val_loss:.4f}")

            # Track the best-val checkpoint whenever we evaluate -- NOT only when
            # early stopping is armed. Gating this on ``es_active`` (patience > 0)
            # meant a run with patience=0 + val_data kept the last-step weights,
            # while MLX restored the best-val weights: a silent cross-backend
            # divergence. Only the patience counter / stop-early break stays gated.
            from auto_chasm.trainers._metrics import resolve_early_stopping_metric

            current = resolve_early_stopping_metric(trainer.early_stopping_metric, val_metrics)
            higher_is_better = trainer.early_stopping_higher_is_better
            improved = (
                current > best_metric + trainer.min_delta
                if higher_is_better
                else current < best_metric - trainer.min_delta
            )

            if improved:
                best_metric = current
                best_iter = global_step
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if trainer.verbose:
                    print(f"    -> New best {trainer.early_stopping_metric}={current:.4f}")
                if es_active:
                    patience_counter = 0
            elif es_active:
                patience_counter += 1
                if trainer.verbose:
                    print(
                        f"    -> No improvement "
                        f"({patience_counter}/{trainer.early_stopping_patience})"
                    )
                if patience_counter >= trainer.early_stopping_patience:
                    if trainer.verbose:
                        print(f"Early stopping at step {global_step}. Best was {best_iter}.")
                    break

        # --- Periodic checkpoint ---
        if trainer.save_steps > 0 and global_step % trainer.save_steps == 0:
            path = trainer.output_dir / f"{global_step:07d}_adapters.pt"
            torch.save(model.state_dict(), path)

    # --- Restore best (OPT-IN, and it must stay in lockstep with the MLX loop) ---
    # Best-val tracking above is deliberately ungated so ``best_iter`` is always
    # reported; the ROLLBACK is what the caller opts into. Defaulting this to on
    # silently rewound fixed-budget runs to whichever eval happened to score best.
    if best_state is not None and trainer.restore_best_weights:
        model.load_state_dict(best_state, _strict=False)
        if trainer.verbose:
            print(f"Restored best checkpoint from step {best_iter}.")
    elif best_state is not None and trainer.verbose:
        print(f"Kept final-step weights (best was step {best_iter}; "
              f"pass restore_best_weights=True to roll back to it).")

    # --- Save best checkpoint ---
    from auto_chasm.checkpoint import save_checkpoint

    best_path = str(trainer.output_dir / "final")
    save_checkpoint(trainer.model, best_path)

    # --- Free GPU memory from training ---
    del best_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Manifest, history, and checkpoint cleanup ---
    from auto_chasm.trainers._metrics import finalize_torch_run, torch_manifest

    manifest = torch_manifest(
        base_model=getattr(trainer.model, "_base_model_name", None),
        best_iter=best_iter,
        best_metric=best_metric,
        best_metric_name=trainer.early_stopping_metric,
        num_iters=trainer.num_iters,
        early_stopping_patience=trainer.early_stopping_patience,
        min_delta=trainer.min_delta,
        keep_best_only=trainer.keep_best_only,
        restore_best_weights=trainer.restore_best_weights,
    )
    finalize_torch_run(
        output_dir=trainer.output_dir,
        history=history,
        manifest=manifest,
        save_history=trainer.save_history,
        keep_best_only=trainer.keep_best_only,
        verbose=trainer.verbose,
    )

    trainer._fire_callback("on_epoch_end", epoch=1, history=history)

    test_metrics: dict[str, float] | None = None
    if test_data is not None:
        test_metrics = trainer._evaluate_torch(test_data)
        # Same as the MLX path: put it in the HISTORY, not only the return value,
        # so training_history.json stops showing test_metrics as empty when the
        # test set was in fact evaluated.
        trainer._record_test_metrics(history, test_metrics)

    return {
        "history": history,
        "test_metrics": test_metrics,
        "output_dir": str(trainer.output_dir),
    }

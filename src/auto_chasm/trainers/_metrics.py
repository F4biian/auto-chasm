"""Shared trainer metric and schedule helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from auto_chasm.logger import get_logger

logger = get_logger(__name__)


def finalize_torch_run(
    output_dir: Path,
    history: Any,
    manifest: dict[str, Any],
    save_history: bool,
    keep_best_only: bool,
    verbose: bool,
) -> None:
    """Write the manifest/history and clean up checkpoints after a torch run.

    Args:
        output_dir: The trainer output directory.
        history: The ``History`` object to persist.
        manifest: The training manifest dict to write.
        save_history: Whether to write ``training_history.json``.
        keep_best_only: Whether to delete periodic ``.pt`` checkpoints.
        verbose: Whether to print progress.
    """
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    with open(final_dir / "training_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if save_history:
        history_path = final_dir / "training_history.json"
        history.save_json(history_path)
        if verbose:
            print(f"Training history saved to {history_path}")

    for stale in ("adapters.pt", "training_manifest.json", "training_history.json"):
        stale_path = output_dir / stale
        if stale_path.exists():
            stale_path.unlink()

    if keep_best_only:
        count = 0
        for path in output_dir.iterdir():
            # Periodic torch checkpoints are ``{step:07d}_adapters.pt``; match the
            # zero-padded step prefix, not any leading digit (which would also delete
            # a digit-leading probe name's saved head file).
            head = path.name.split("_", 1)[0]
            if path.is_file() and head.isdigit() and len(head) >= 7 and path.name.endswith(".pt"):
                path.unlink()
                count += 1
        if count > 0 and verbose:
            print(f"Cleaned up {count} periodic checkpoints (keep_best_only=True).")


def torch_manifest(
    base_model: str | None,
    best_iter: int,
    best_metric: float,
    best_metric_name: str,
    num_iters: int,
    early_stopping_patience: int,
    min_delta: float,
    keep_best_only: bool,
) -> dict[str, Any]:
    """Build the torch training manifest dict.

    Args:
        base_model: Base model name, if known.
        best_iter: Iteration of the best checkpoint (``0`` if none).
        best_metric: Best metric value.
        best_metric_name: Name of the monitored metric.
        num_iters: Total training iterations.
        early_stopping_patience: Early-stopping patience.
        min_delta: Early-stopping minimum delta.
        keep_best_only: Whether periodic checkpoints were removed.

    Returns:
        The manifest dict (JSON-safe; non-finite metrics become ``None``).
    """
    metric_for_json: float | None = None
    if best_iter > 0 and math.isfinite(best_metric):
        metric_for_json = best_metric
    return {
        "base_model": base_model,
        "backend": "torch",
        "best_iter": best_iter,
        "best_metric": metric_for_json,
        "best_metric_name": best_metric_name,
        "num_iters": num_iters,
        "early_stopping_patience": early_stopping_patience,
        "min_delta": min_delta,
        "keep_best_only": keep_best_only,
    }


def validate_loss_weight_keys(loss_fn: Any, probe_names: Any) -> None:
    """Eagerly validate a ``JointLoss``'s weight keys against the model's probes.

    Call this BEFORE the ``value_and_grad`` / ``mx.compile`` trace: an unknown-key
    ``ValueError`` raised inside the trace poisons ``mx.random.state`` and bricks
    every subsequent MLX training in the process.  A no-op for loss functions that do
    not expose ``_validate_weight_keys`` (e.g. a plain callable).

    Args:
        loss_fn: The loss function (a ``JointLoss`` or any callable).
        probe_names: The attached probes' names.
    """
    validate = getattr(loss_fn, "_validate_weight_keys", None)
    if callable(validate):
        validate(list(probe_names))


def require_trainable_params(has_trainable: bool) -> None:
    """Raise a clear error if a trainer has no trainable parameters.

    Without trainable parameters MLX's ``nn.value_and_grad`` raises an opaque
    ``[grad] Must specify at least one argument``; name the real
    misconfiguration (everything frozen) instead.

    Args:
        has_trainable: Whether any trainable parameters exist.

    Raises:
        ValueError: If ``has_trainable`` is ``False``.
    """
    if not has_trainable:
        raise ValueError(
            "No trainable parameters — did you freeze the base and all probes? "
            "Unfreeze a probe or call prepare_for_joint_training() before training."
        )


def build_lr_schedule(
    lr_schedule: str, learning_rate: float, num_iters: int, warmup_steps: int
) -> Any:
    """Build an MLX learning rate schedule (cosine/linear/constant + warmup).

    Args:
        lr_schedule: One of ``"cosine"``, ``"linear"``, ``"constant"``.
        learning_rate: Peak learning rate.
        num_iters: Total training iterations.
        warmup_steps: Linear warmup length (``0`` for none).

    Returns:
        An MLX schedule callable consumable by an optimizer.

    Raises:
        ValueError: If ``lr_schedule`` is not a known schedule type.
    """
    import mlx.core as mx
    import mlx.optimizers as optim

    # Clamp warmup so it never swallows the whole run.  If warmup_steps exceeds
    # num_iters the warmup ramp alone would govern training and the configured
    # peak LR would never be attained (the ramp tops out at
    # peak * num_iters / warmup_steps), silently discarding the decay schedule.
    # Cap at num_iters - 1 so the peak is reached and the decay still applies.
    # warmup_steps == num_iters is left intact: a pure warmup ramp that reaches
    # peak exactly at the final step (warmup_ratio=1.0) is a valid choice.
    if warmup_steps > num_iters and num_iters >= 1:
        clamped = num_iters - 1
        logger.warning(
            "warmup_steps (%d) >= num_iters (%d); clamping warmup to %d so the "
            "configured peak LR (%.2e) is actually reached.",
            warmup_steps,
            num_iters,
            clamped,
            learning_rate,
        )
        warmup_steps = clamped

    warmup = optim.linear_schedule(0.0, learning_rate, warmup_steps) if warmup_steps > 0 else None

    if lr_schedule == "cosine":
        main_steps = max(num_iters - warmup_steps, 1)
        main_schedule = optim.cosine_decay(learning_rate, main_steps)
    elif lr_schedule == "linear":
        main_steps = max(num_iters - warmup_steps, 1)

        def _linear(it: Any) -> Any:
            # Clamp progress to [0, 1] so the LR floors at 0 past the horizon
            # (matches cosine_decay and torch LinearLR); without this the
            # multiplier (1 - progress) goes negative.  ``it`` is an mx.array
            # under the optimizer, so use mx.minimum.
            progress = mx.minimum(it / main_steps, 1.0)
            return learning_rate * (1.0 - progress)

        main_schedule = _linear
    elif lr_schedule == "constant":

        def _constant(_: Any) -> Any:
            # Return an mx.array (like the other schedules): the MLX optimizer calls
            # ``.astype`` on the schedule's output, which a Python float lacks.
            return mx.array(learning_rate)

        main_schedule = _constant
    else:
        raise ValueError(
            f"Unknown lr_schedule {lr_schedule!r}. Expected one of: 'cosine', 'linear', 'constant'."
        )

    if warmup is not None and warmup_steps < num_iters:
        return optim.join_schedules([warmup, main_schedule], [warmup_steps])
    if warmup is not None:
        return warmup
    return main_schedule


def evaluate_torch_model(
    model_wrapper: Any,
    dataset: Any,
    batch_size: int,
    max_seq_length: int,
    loss_fn: Any,
    eval_metrics_fn: Any | None,
) -> dict[str, float]:
    """Evaluate a torch model, including optional custom metrics.

    Args:
        model_wrapper: The ``Model`` whose ``.model`` and ``._probes`` are used.
        dataset: The evaluation dataset.
        batch_size: Batch size.
        max_seq_length: Maximum sequence length.
        loss_fn: Loss function returning ``(total, ntoks, components)``.
        eval_metrics_fn: Optional ``(wrapper, captured, targets, mask) ->
            dict[str, float]`` producing custom metrics (e.g. F1).

    Returns:
        Dict of evaluation metrics (loss, components, and any custom metrics).
    """
    import torch

    from auto_chasm.trainers.data_utils import iterate_batches, labels_to_torch
    from auto_chasm.trainers.wrappers import _TorchProbeWrapper

    raw_model = model_wrapper.model
    raw_model.eval()
    model = _TorchProbeWrapper(raw_model, model_wrapper._probes)

    total_loss = 0.0
    total_components: dict[str, float] = {}
    total_ntoks = 0.0
    metric_accum: dict[str, float] = {}
    metric_weight = 0.0  # sum of per-batch valid-token counts (metric denominator)

    with torch.no_grad():
        for batch in iterate_batches(dataset, batch_size, max_seq_length, loop=False):
            tokens, labels, lengths = batch
            device = model.device if hasattr(model, "device") else "cpu"
            tokens = torch.from_numpy(tokens).to(device)
            labels = labels_to_torch(labels, device)
            lengths = torch.from_numpy(lengths).to(device)

            total, ntoks, components = loss_fn(model, tokens, labels, lengths)
            # Token-weight the averages (like the MLX evaluate_joint_model): a plain
            # per-batch mean over-weights a short final batch and diverges from MLX
            # whenever batches hold different token counts.
            n = float(ntoks.item() if hasattr(ntoks, "item") else ntoks)
            total_loss += total.item() * n
            for key, val in components.items():
                base = total_components.get(key, 0.0)
                v = val.item() if hasattr(val, "item") else float(val)
                total_components[key] = base + v * n
            total_ntoks += n

            if eval_metrics_fn is not None:
                # Pass ALL captured layers (matching the MLX path, which stores the
                # full list): run_probe forwards every layer, so a multi-layer probe
                # is scored identically on both backends. Truncating to states[-1]
                # crashed a concat head and silently dropped layers under mean/max.
                captured = {
                    name: states
                    for name, probe in model_wrapper._probes.items()
                    if (states := probe.get_captured_states())
                }
                if captured:
                    # Per-probe ``labels`` dict → a matching dict of shifted
                    # targets; a single array → a single shifted target.
                    if isinstance(labels, dict):
                        b_targets: Any = {k: v[:, 1:].float() for k, v in labels.items()}
                    else:
                        b_targets = labels[:, 1:].float()
                    n_time = tokens.shape[1] - 1
                    steps = torch.arange(1, n_time + 1, device=device)
                    m = (steps >= lengths[:, 0:1]) & (steps < lengths[:, 1:])
                    batch_metrics = eval_metrics_fn(model, captured, b_targets, m)
                    # Token-weight by valid-position count -> corpus metric, not a
                    # per-batch mean (see the MLX evaluate_joint_model for the note on
                    # macro-F1 becoming a token-weighted mean across eval batches).
                    w = float(m.sum().item())
                    metric_weight += w
                    for k, v in batch_metrics.items():
                        metric_accum[k] = metric_accum.get(k, 0.0) + float(v) * w

    result: dict[str, float] = {
        "loss": total_loss / total_ntoks if total_ntoks > 0 else float("inf")
    }
    for key, acc in total_components.items():
        result[key] = acc / total_ntoks if total_ntoks > 0 else 0.0
    for key, acc in metric_accum.items():
        result[key] = acc / metric_weight if metric_weight > 0 else 0.0
    return result


def resolve_early_stopping_metric(metric_name: str, val_metrics: dict[str, Any]) -> float:
    """Resolve the early-stopping metric value, failing loudly if absent.

    There is deliberately **no** silent fallback to the loss: falling back
    would let a misconfigured ``early_stopping_metric`` (e.g. ``"val_f1"``
    when no F1 is computed) compare loss values in the wrong direction and
    select the worst model without any warning.

    Args:
        metric_name: e.g. ``"val_loss"`` or ``"val_f1"``.
        val_metrics: Validation metrics dict.

    Returns:
        The metric value as a float.

    Raises:
        KeyError: If the requested metric is not present.
    """
    if metric_name in ("val_loss", "loss"):
        return float(val_metrics["loss"])
    key = metric_name.removeprefix("val_")  # strip the PREFIX only (a probe named
    if key in val_metrics:  # 'retrieval' contains 'val_' mid-string)
        return float(val_metrics[key])
    raise KeyError(
        f"early_stopping_metric={metric_name!r} is not available in validation "
        f"metrics {sorted(val_metrics)}. The trainer computes loss/perplexity/"
        f"loss-components by default; supply a metrics function that produces "
        f"{key!r}, or use early_stopping_metric='val_loss'."
    )

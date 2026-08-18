"""SFT trainer — supervised fine-tuning with joint probe losses."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from auto_chasm.config import TrainingConfig
from auto_chasm.logger import get_logger
from auto_chasm.model import Model
from auto_chasm.trainers.base import _UNSET, JointTrainer

logger = get_logger(__name__)


def _require_mlx_backend(model: Model, trainer_name: str) -> None:
    """Raise a clear error when a trainer is used on a non-MLX backend.

    The SFT/RL training path wraps the model in the MLX-only
    ``_TrainableModel`` (which calls ``module.unfreeze()`` — a method torch
    ``nn.Module`` does not have).  Fail loudly at construction instead of
    crashing deep in the loop with ``'Linear' has no attribute 'unfreeze'``.

    Args:
        model: The model to check.
        trainer_name: Name of the trainer (for the error message).

    Raises:
        ValueError: If ``model.backend.name`` is not ``"mlx"``.
    """
    backend = getattr(getattr(model, "backend", None), "name", None)
    if backend != "mlx":
        raise ValueError(
            f"{trainer_name} supports only the MLX backend "
            f"(got backend={backend!r}). The supervised training path wraps the "
            f"model in the MLX-only _TrainableModel. For PyTorch, use the "
            f"backend-agnostic Trainer with a JointLoss instead."
        )


class SFTTrainer:
    """Supervised fine-tuning trainer with joint probe losses.

    Wraps ``JointTrainer`` with a default loss function that
    combines LM cross-entropy with a probe loss term.

    Args:
        model: The ``Model`` instance to train.
        lm_weight: Weight for the LM cross-entropy term.
            Set to ``0`` for pure classifier mode.
        probe_weight: Weight for the probe loss term.
        probe_loss: Probe loss — ``"bce"``, ``"mse"``, or a callable.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        grad_clip_norm: Gradient clipping max norm.
        num_iters: Total training iterations.
        batch_size: Per-step batch size.
        max_seq_length: Maximum sequence length.
        grad_accum_steps: Gradient accumulation steps.
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.
        early_stopping_patience: Stop after N eval rounds without improvement.
            0 (the default) disables early stopping.
        restore_best_weights: If ``True``, reload the best-scoring checkpoint at
            the end of ``train()``. Default ``False`` — the FINAL-step weights
            are what you get, which is what a fixed-budget run almost always
            wants. Best-val tracking still happens (``best_iter`` is reported in
            the manifest) so the diagnostic survives; only the rollback is opt-in.
        early_stopping_metric: Metric for early stopping.
        early_stopping_higher_is_better: If ``True``, maximize the metric.
        eval_steps: Evaluate every N steps.  ``None`` defaults to ``save_steps``.
        lr_schedule: LR schedule (``"cosine"``, ``"linear"``, ``"constant"``).
        warmup_ratio: Fraction of total steps for linear warmup.
        output_dir: Directory for checkpoints.
        verbose: Whether to print training progress to stdout.
        loss_fn: Override loss function.
        eval_metrics_fn: Optional custom validation metrics callable.
        config: Optional ``TrainingConfig`` providing defaults for
            hyperparameters.  Forwarded to the underlying ``JointTrainer`` so
            ``lr_schedule``, ``warmup_ratio``, ``eval_steps`` and early-stopping
            direction are all honored.  Individual keyword arguments override
            config values when explicitly set.

    Raises:
        ValueError: If ``model`` is not on the MLX backend (the SFT path wraps
            the model in the MLX-only ``_TrainableModel``).
    """

    def __init__(
        self,
        model: Model,
        lm_weight: float = _UNSET,
        probe_weight: float = _UNSET,
        probe_loss: str | Callable[..., Any] = "bce",
        learning_rate: float = _UNSET,
        weight_decay: float = _UNSET,
        grad_clip_norm: float = _UNSET,
        num_iters: int = 500,
        batch_size: int = _UNSET,
        max_seq_length: int = 256,
        grad_accum_steps: int = _UNSET,
        logging_steps: int = _UNSET,
        save_steps: int = _UNSET,
        eval_steps: int | None = None,
        early_stopping_patience: int = 0,
        restore_best_weights: bool = False,
        early_stopping_metric: str = "val_loss",
        early_stopping_higher_is_better: bool = False,
        lr_schedule: str = _UNSET,
        warmup_ratio: float = _UNSET,
        output_dir: str = _UNSET,
        verbose: bool = True,
        loss_fn: Callable[..., Any] | None = None,
        eval_metrics_fn: Callable[..., dict[str, float]] | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        """Initialize the SFT trainer.

        If ``config`` is provided, it is forwarded to the underlying
        ``JointTrainer`` (so all of its fields are honored).  Individual
        keyword arguments override config values when explicitly provided.
        """
        _require_mlx_backend(model, "SFTTrainer")
        self.model = model

        # SFT-specific loss weights: an explicit value wins over config (via _UNSET),
        # else config, else 1.0. Every trainer hyperparameter below is forwarded as its
        # _UNSET sentinel so JointTrainer's config precedence resolves it (nothing dropped).
        if lm_weight is _UNSET:
            lm_weight = config.lm_weight if config is not None else 1.0
        if probe_weight is _UNSET:
            probe_weight = config.probe_weight if config is not None else 1.0

        if loss_fn is None:
            # Phase 3b: adapt the pre-3b (lm_weight, probe_weight, probe_loss) knobs
            # onto the new JointLoss API (Phase 3b-2 migrates this call site's tests).
            from auto_chasm.trainers.loss import _joint_loss_from_legacy

            loss_fn = _joint_loss_from_legacy(
                lm_weight=lm_weight,
                probe_weight=probe_weight,
                probe_loss=probe_loss,
            )

        self._trainer = JointTrainer(
            model=model,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            num_iters=num_iters,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            grad_accum_steps=grad_accum_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            early_stopping_patience=early_stopping_patience,
            restore_best_weights=restore_best_weights,
            early_stopping_metric=early_stopping_metric,
            early_stopping_higher_is_better=early_stopping_higher_is_better,
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            output_dir=output_dir,
            verbose=verbose,
            eval_metrics_fn=eval_metrics_fn,
            config=config,
        )

    def train(
        self,
        train_data: Any,
        val_data: Any | None = None,
    ) -> dict[str, Any]:
        """Run SFT training.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset (optional).

        Returns:
            Dict with keys ``"history"`` (``History``) and ``"output_dir"``
            (the unified trainer return contract; ``run()`` remains available
            as the lower-level ``History``-returning API).
        """
        history = self._trainer.run(train_data, val_data)
        return {"history": history, "output_dir": str(self._trainer.output_dir)}

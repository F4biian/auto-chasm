"""Backend-agnostic trainer facade.

Provides a clean ``Trainer`` class that hides all backend-specific
training logic.  Users pass a ``Model`` and a loss function; the
trainer handles freeze/unfreeze, optimizer setup, gradient clipping,
checkpointing, and callbacks automatically.

For MLX the trainer delegates to ``JointTrainer`` (proven
``mx.compile`` + ``nn.value_and_grad`` loop).  For PyTorch a
standard autograd loop is used.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from auto_chasm.config import TrainingConfig
from auto_chasm.history import History
from auto_chasm.logger import get_logger
from auto_chasm.model import Model
from auto_chasm.trainers.wrappers import (
    TrainerCallback,
)

if TYPE_CHECKING:
    from auto_chasm.trainers.base import JointTrainer

logger = get_logger(__name__)

# Sentinel for "argument not explicitly provided". Typed ``Any`` so it can default
# a typed parameter without widening its annotation, while still being distinct from
# every real value — so an explicit kwarg that happens to equal a library default is
# NOT mistaken for "unset" and silently overridden by ``config``.
_UNSET: Any = object()


class Trainer:
    """Backend-agnostic trainer for joint LM + probe fine-tuning.

    Wraps the proven ``JointTrainer`` (MLX) or a standard PyTorch
    loop.  Handles freeze/unfreeze, optimizer creation, gradient
    clipping, and checkpointing automatically.

    Args:
        model: The ``Model`` instance to train.
        loss_fn: Loss function with signature
            ``(model, batch, labels, lengths) -> (total, ntoks, components)``
            where ``components`` is a ``dict[str, tensor]`` of named
            loss terms (backend scalar tensors).  Use ``JointLoss`` for a
            backend-agnostic default, or write your own using
            framework-specific ops.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        grad_clip_norm: Gradient clipping max norm (0 to disable).
        num_iters: Total training iterations.
        batch_size: Per-step batch size.
        max_seq_length: Maximum sequence length.
        grad_accum_steps: Gradient accumulation steps.
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.  0 disables periodic saves.
        eval_steps: Evaluate every N steps.  0 disables mid-training eval.
            Defaults to ``save_steps`` if ``None``.
        early_stopping_patience: Stop after N eval rounds without improvement.
            0 (the default) disables early stopping.
        compile_step: Whether to ``mx.compile`` the training step (MLX).
            ``None`` (default) compiles EXCEPT on models that unroll a
            per-timestep recurrence, where one compiled graph per input shape
            exhausts Metal's buffer limit a few hundred iterations in. ``True``
            forces compilation, ``False`` disables it.
        restore_best_weights: If ``True``, reload the best-scoring checkpoint at
            the end of ``train()``. Default ``False`` — the FINAL-step weights
            are what you get, which is what a fixed-budget run almost always
            wants. Best-val tracking still happens (``best_iter`` is reported in
            the manifest) so the diagnostic survives; only the rollback is opt-in.
        early_stopping_metric: Metric to monitor (``"val_loss"``, ``"val_f1"``, etc.).
        early_stopping_higher_is_better: If ``True``, maximize the metric (for F1, accuracy);
            if ``False``, minimize (for loss, perplexity).  Default ``False``.
        min_delta: Minimum improvement to count as an improvement.
        keep_best_only: If ``True``, delete periodic checkpoints after training.
        save_history: If ``True``, save training history as JSON to the
            output directory after training.  Default ``True``.
        output_dir: Directory for checkpoints and logs.
        callbacks: List of ``TrainerCallback`` instances.
        custom_train_fn: If set, replaces the default training loop entirely.
            Called as ``custom_train_fn(model, trainer)``.
        verbose: Whether to print training progress to stdout.
        seed: Seeds the training batch ORDER (``iterate_batches`` shuffles with
            ``np.random.default_rng(seed)``, which draws OS entropy when this is
            ``None`` and deliberately ignores the global numpy seed — so batch
            order is reproducible only if you pass this). ``None`` (default)
            keeps the previous unseeded behavior. Seed the frameworks yourself
            for weight init.
        eval_metrics_fn: Optional callable ``(train_model, captured, targets,
            mask) -> dict[str, float]`` producing custom validation metrics
            (e.g. F1, accuracy), enabling non-loss ``early_stopping_metric``
            values such as ``"val_f1"``.
        config: Optional ``TrainingConfig`` providing defaults for
            hyperparameters.  Individual keyword arguments override
            config values when explicitly set.
    """

    def __init__(
        self,
        model: Model,
        loss_fn: Callable[..., Any],
        learning_rate: float = _UNSET,
        weight_decay: float = _UNSET,
        grad_clip_norm: float = _UNSET,
        num_iters: int = 500,
        batch_size: int = _UNSET,
        max_seq_length: int = 256,
        grad_accum_steps: int = _UNSET,
        logging_steps: int = _UNSET,
        save_steps: int = _UNSET,
        eval_steps: int | None = _UNSET,
        early_stopping_patience: int = 0,
        restore_best_weights: bool = False,
        compile_step: bool | None = None,
        early_stopping_metric: str = "val_loss",
        early_stopping_higher_is_better: bool = False,
        min_delta: float = 1e-4,
        keep_best_only: bool = False,
        save_history: bool = True,
        history_save_frequency: str = "val",
        output_dir: str = _UNSET,
        callbacks: list[TrainerCallback] | None = None,
        custom_train_fn: Callable[..., Any] | None = None,
        verbose: bool = True,
        lr_schedule: str = _UNSET,
        warmup_ratio: float = _UNSET,
        eval_metrics_fn: Callable[..., dict[str, float]] | None = None,
        seed: int | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        """Initialize the trainer.

        If ``config`` is provided, its values are used as defaults for
        mapped hyperparameters.  Individual keyword arguments override
        config values when explicitly provided.
        """
        if config is not None:
            import numpy as np

            np.random.seed(config.seed)
            try:
                import mlx.core as mx

                mx.random.seed(config.seed)
            except ImportError:
                pass

        self.model = model
        self.loss_fn = loss_fn
        self.eval_metrics_fn = eval_metrics_fn
        self.seed = seed
        self.num_iters = num_iters
        self.max_seq_length = max_seq_length
        self.early_stopping_patience = early_stopping_patience
        self.restore_best_weights = restore_best_weights
        self.compile_step = compile_step
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_higher_is_better = early_stopping_higher_is_better
        self.min_delta = min_delta
        self.keep_best_only = keep_best_only
        self.save_history = save_history
        self.history_save_frequency = history_save_frequency
        self.callbacks = callbacks or []
        self.custom_train_fn = custom_train_fn
        self.verbose = verbose
        self._joint_trainer: JointTrainer | None = None
        self._torch_history: History = History()
        # Lazily-built persistent state for the torch per-step escape hatch
        # (Trainer.step()): wrapper + optimizer + scheduler, created on first
        # step() call and reused across the manual loop.
        self._torch_step_state: dict[str, Any] | None = None

        # Mapped hyperparameters: an EXPLICIT kwarg wins (even when it equals a
        # library default); otherwise ``config`` supplies the value; otherwise the
        # documented default. Using a sentinel (not "== default") is what lets a user
        # force, say, learning_rate=2e-4 alongside a config that sets something else.
        def _pick(explicit: Any, cfg_attr: str, default: Any) -> Any:
            if explicit is not _UNSET:
                return explicit
            return getattr(config, cfg_attr) if config is not None else default

        self.learning_rate = _pick(learning_rate, "learning_rate", 2e-4)
        self.weight_decay = _pick(weight_decay, "weight_decay", 0.0)
        self.grad_clip_norm = _pick(grad_clip_norm, "max_grad_norm", 1.0)
        self.batch_size = _pick(batch_size, "batch_size", 8)
        self.grad_accum_steps = _pick(grad_accum_steps, "gradient_accumulation_steps", 1)
        self.logging_steps = _pick(logging_steps, "logging_steps", 25)
        self.save_steps = _pick(save_steps, "save_steps", 100)
        self.output_dir = Path(_pick(output_dir, "output_dir", "./checkpoints"))
        self.eval_steps = _pick(eval_steps, "eval_steps", None)
        self.lr_schedule: str = _pick(lr_schedule, "lr_schedule", "cosine")
        self.warmup_ratio: float = _pick(warmup_ratio, "warmup_ratio", 0.0)

        # Config-only fields (no dedicated kwarg to override them).
        self._probe_weights: dict[str, float] = {}
        self._config_lm_weight: float | None = None
        self._config_probe_weight: float | None = None
        self._mixed_precision: str = "fp32"

        if config is not None:
            self._probe_weights = config.probe_weights
            # Apply non-default global weights (default 1.0 leaves the loss's own
            # weights untouched, so a pure-probe JointLoss(weights={"lm_head":0}) is
            # not clobbered by the config default).
            if config.lm_weight != 1.0:
                self._config_lm_weight = config.lm_weight
            if config.probe_weight != 1.0:
                self._config_probe_weight = config.probe_weight
            self._mixed_precision = config.mixed_precision

        self._apply_probe_weights()

        self._backend_name = model.backend.name
        logger.info("Trainer initialized (backend=%s).", self._backend_name)

    def train(
        self,
        train_data: Any,
        val_data: Any | None = None,
        test_data: Any | None = None,
    ) -> dict[str, Any]:
        """Run training.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset (optional, enables early stopping).
            test_data: Test dataset (optional, evaluated after training).

        Returns:
            Dict with keys ``"history"`` (``History``), ``"test_metrics"``,
            ``"output_dir"``.
        """
        self._fire_callback("on_train_begin")
        self._resolve_class_weights(train_data)

        if self.custom_train_fn is not None:
            result: dict[str, Any] = self.custom_train_fn(self.model, self)
            self._fire_callback("on_train_end")
            return result

        if self._backend_name == "mlx":
            result = self._train_mlx(train_data, val_data, test_data)
        elif self._backend_name == "torch":
            result = self._train_torch(train_data, val_data, test_data)
        else:
            raise RuntimeError(f"Unsupported backend: {self._backend_name}")

        self._fire_callback("on_train_end")
        return result

    # ------------------------------------------------------------------
    # Escape-hatch API (delegates to JointTrainer)
    # ------------------------------------------------------------------

    def iterate(self, train_data: Any) -> Iterator[tuple[Any, Any, Any]]:
        """Iterate over training batches (works on both backends).

        Args:
            train_data: Training dataset.

        Yields:
            Tuple of ``(tokens, labels, lengths)`` as numpy arrays.
        """
        if self._backend_name == "mlx":
            return self._get_joint().iterate(train_data)  # type: ignore[no-any-return]
        from auto_chasm.trainers.data_utils import iterate_batches

        return iterate_batches(
            train_data, self.batch_size, self.max_seq_length, loop=True, seed=self.seed
        )

    def step(self, batch: tuple[Any, Any, Any]) -> dict[str, float]:
        """Run one training step (works on both backends).

        The per-step escape hatch for manual training loops. On first call it
        lazily builds a persistent optimizer/scheduler; each call performs one
        optimizer update. Because it mutates the same parameters ``evaluate()``
        reads, interleaving ``step()`` and ``evaluate()`` works on both backends.

        On torch the scheduler horizon is ``num_iters`` (one update per call);
        gradient accumulation is *not* applied here (one ``step()`` = one
        update), matching the MLX escape hatch.

        Args:
            batch: ``(tokens, labels, lengths)`` tuple.

        Returns:
            Dict with keys ``"loss"`` (float), ``"ntoks"`` (int),
            ``"components"`` (dict of str → float).
        """
        if self._backend_name == "mlx":
            return self._get_joint().step(batch)  # type: ignore[no-any-return]
        from auto_chasm.trainers._torch_step import torch_step

        return torch_step(self, batch)

    def evaluate(self, val_data: Any, num_batches: int = -1) -> dict[str, float]:
        """Evaluate on a validation dataset (works on both backends).

        Args:
            val_data: Validation dataset.
            num_batches: Max batches (-1 = all).  Honored on MLX; the torch
                path evaluates the full dataset.

        Returns:
            Dict of evaluation metrics.
        """
        if self._backend_name == "mlx":
            return self._get_joint().evaluate(val_data, num_batches)  # type: ignore[no-any-return]
        return self._evaluate_torch(val_data)

    def save_checkpoint(self, path: str | None = None) -> str:
        """Save the current model state (works on both backends).

        Args:
            path: Directory path to save to.  Defaults to
                ``{output_dir}/checkpoint``.

        Returns:
            The path the checkpoint was saved to.
        """
        if self._backend_name == "mlx":
            return self._get_joint().save_checkpoint(path)  # type: ignore[no-any-return]
        from auto_chasm.checkpoint import save_checkpoint

        target = path if path is not None else str(self.output_dir / "checkpoint")
        save_checkpoint(self.model, target)
        return target

    def restore_checkpoint(self, path: str) -> None:
        """Restore model weights from a checkpoint (MLX only).

        Args:
            path: Path to a checkpoint directory.

        Raises:
            NotImplementedError: On the torch backend.  ``load_checkpoint``
                builds a fresh ``Model`` rather than restoring weights into an
                existing one in place; use ``auto_chasm.checkpoint.load_checkpoint``
                to obtain a restored model instead.
        """
        if self._backend_name == "mlx":
            self._get_joint().restore_checkpoint(path)
            return
        raise NotImplementedError(
            "Trainer.restore_checkpoint() is implemented only on the MLX backend. "
            "On torch, load a fresh restored model with "
            "auto_chasm.checkpoint.load_checkpoint(path, backend_name='torch')."
        )

    def get_history(self) -> History:
        """Return the current training history (works on both backends).

        Returns:
            ``History`` object with all logged metrics.
        """
        if self._backend_name == "mlx":
            return self._get_joint().get_history()  # type: ignore[no-any-return]
        return self._torch_history

    def _get_joint(self) -> Any:
        """Get or create the ``JointTrainer`` instance.

        Returns:
            The ``JointTrainer``.

        Raises:
            RuntimeError: If no supported backend is available and no
                custom train function was set.
        """
        if self._joint_trainer is not None:
            return self._joint_trainer

        if self._backend_name == "mlx":
            from auto_chasm.trainers.base import JointTrainer

            self._joint_trainer = JointTrainer(
                model=self.model,
                loss_fn=self.loss_fn,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                grad_clip_norm=self.grad_clip_norm,
                num_iters=self.num_iters,
                batch_size=self.batch_size,
                max_seq_length=self.max_seq_length,
                grad_accum_steps=self.grad_accum_steps,
                logging_steps=self.logging_steps,
                save_steps=self.save_steps,
                eval_steps=self.eval_steps,
                early_stopping_patience=self.early_stopping_patience,
                restore_best_weights=self.restore_best_weights,
                compile_step=self.compile_step,
                early_stopping_metric=self.early_stopping_metric,
                early_stopping_higher_is_better=self.early_stopping_higher_is_better,
                min_delta=self.min_delta,
                keep_best_only=self.keep_best_only,
                save_history=self.save_history,
                history_save_frequency=self.history_save_frequency,
                output_dir=str(self.output_dir),
                verbose=self.verbose,
                lr_schedule=self.lr_schedule,
                warmup_ratio=self.warmup_ratio,
                eval_metrics_fn=self.eval_metrics_fn,
                mixed_precision=self._mixed_precision,
            )
            return self._joint_trainer

        raise RuntimeError(
            f"Escape-hatch API requires an MLX backend. Got backend: {self._backend_name}"
        )

    # ------------------------------------------------------------------
    # MLX path — delegates to JointTrainer
    # ------------------------------------------------------------------

    @property
    def stop_requested(self) -> bool:
        """Whether a callback has asked the running loop to finish early.

        Forwards to the live ``JointTrainer``: a callback is handed this facade,
        but the loop that must actually break lives on the inner trainer, so
        setting the flag here otherwise did nothing at all.
        """
        return bool(getattr(self._joint_trainer, "stop_requested", False))

    @stop_requested.setter
    def stop_requested(self, value: bool) -> None:
        """Ask the running loop to stop (ignored when no loop is running)."""
        if self._joint_trainer is not None:
            self._joint_trainer.stop_requested = value

    def _train_mlx(
        self,
        train_data: Any,
        val_data: Any | None,
        test_data: Any | None,
    ) -> dict[str, Any]:
        """MLX training via JointTrainer."""
        from auto_chasm.trainers.base import JointTrainer

        joint = JointTrainer(
            seed=self.seed,
            model=self.model,
            loss_fn=self.loss_fn,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
            num_iters=self.num_iters,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            grad_accum_steps=self.grad_accum_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            early_stopping_patience=self.early_stopping_patience,
            restore_best_weights=self.restore_best_weights,
            compile_step=self.compile_step,
            early_stopping_metric=self.early_stopping_metric,
            early_stopping_higher_is_better=self.early_stopping_higher_is_better,
            min_delta=self.min_delta,
            keep_best_only=self.keep_best_only,
            save_history=self.save_history,
            history_save_frequency=self.history_save_frequency,
            output_dir=str(self.output_dir),
            verbose=self.verbose,
            lr_schedule=self.lr_schedule,
            warmup_ratio=self.warmup_ratio,
            eval_metrics_fn=self.eval_metrics_fn,
            mixed_precision=self._mixed_precision,
        )
        self._joint_trainer = joint

        def _step_callback(step: int, loss: float, components: Any) -> None:
            self._fire_callback("on_step_end", step=step, loss=loss, components=components)

        history = joint.run(train_data, val_data, step_callback=_step_callback)

        self._fire_callback("on_epoch_end", epoch=1, history=history)

        test_metrics: dict[str, float] | None = None
        if test_data is not None:
            test_metrics = self._evaluate_mlx(test_data)
            # Record it in the HISTORY too, not only in the returned dict.
            # HistoryEntry has always advertised test_loss/test_metrics, and the
            # MLX path never filled them -- so training_history.json showed them
            # as null even when test_data WAS evaluated, which reads as "the test
            # set was ignored". Merged into the final step's entry.
            self._record_test_metrics(history, test_metrics)

        return {
            "history": history,
            "test_metrics": test_metrics,
            "output_dir": str(self.output_dir),
        }

    def _record_test_metrics(self, history: Any, test_metrics: dict[str, float]) -> None:
        """Merge test metrics into the history's final step and RE-SAVE the file.

        The history JSON is written during training, before the test pass runs, so
        without this the on-disk file never sees the test numbers even though the
        returned History object would.
        """
        from auto_chasm.history import HistoryEntry

        step = history.entries[-1].step if len(history) else 0
        history.record(
            HistoryEntry(
                step=step, test_loss=test_metrics.get("loss"), test_metrics=dict(test_metrics)
            )
        )
        if self.save_history:
            history.save_json(Path(self.output_dir) / "training_history.json")

    def _evaluate_mlx(self, dataset: Any) -> dict[str, float]:
        """Evaluate on a dataset (MLX)."""
        from auto_chasm.trainers.trainable import evaluate_joint_model

        if self._joint_trainer is None:
            raise RuntimeError("Call train() before evaluate().")

        return evaluate_joint_model(
            train_model=self._joint_trainer._train_model,
            dataset=dataset,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            loss_fn=self.loss_fn,
            eval_metrics_fn=self.eval_metrics_fn,
        )

    # ------------------------------------------------------------------
    # PyTorch path
    # ------------------------------------------------------------------

    def _train_torch(
        self,
        train_data: Any,
        val_data: Any | None,
        test_data: Any | None,
    ) -> dict[str, Any]:
        """PyTorch training loop with validation, early stopping, checkpointing."""
        from auto_chasm.trainers._torch_loop import train_torch

        return train_torch(self, train_data, val_data, test_data)

    def _evaluate_torch(self, dataset: Any) -> dict[str, float]:
        """Evaluate on a dataset (PyTorch).

        Args:
            dataset: The evaluation dataset.

        Returns:
            Dict of evaluation metrics (loss, components, custom metrics).
        """
        from auto_chasm.trainers._metrics import evaluate_torch_model

        return evaluate_torch_model(
            model_wrapper=self.model,
            dataset=dataset,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            loss_fn=self.loss_fn,
            eval_metrics_fn=self.eval_metrics_fn,
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _apply_probe_weights(self) -> None:
        """Apply ``TrainingConfig`` weight fields to the loss function.

        Injects ``probe_weights`` (per-probe overrides), plus a non-default
        ``lm_weight`` (onto the ``lm_head`` term) and ``probe_weight`` (the default
        probe weight), from the config into a ``JointLoss``.  No-op if the loss is
        not a ``JointLoss``.
        """
        from auto_chasm.config import LM_HEAD
        from auto_chasm.trainers.loss import JointLoss

        if not isinstance(self.loss_fn, JointLoss):
            return
        # In combine= mode the per-term weights are never consulted (the combine
        # callable composes the terms itself), so injecting config weights would be
        # a silent no-op. Warn instead of pretending they took effect.
        if self.loss_fn._combine is not None:
            if (
                self._probe_weights
                or self._config_lm_weight is not None
                or self._config_probe_weight is not None
            ):
                logger.warning(
                    "TrainingConfig probe_weights/lm_weight/probe_weight are ignored: "
                    "the loss uses combine=, which composes the terms itself. Fold the "
                    "weights into the combine callable instead."
                )
            return
        if self._probe_weights:
            self.loss_fn._probe_weights.update(self._probe_weights)
        if self._config_lm_weight is not None:
            self.loss_fn._weights[LM_HEAD] = self._config_lm_weight
        if self._config_probe_weight is not None:
            self.loss_fn._default_weight = self._config_probe_weight

    def _resolve_class_weights(self, train_data: Any) -> None:
        """Resolve any ``class_weights="balanced"`` on a ``JointLoss`` from data.

        Computes inverse-frequency weights from ``train_data`` (per probe for a
        dict spec, else over the shared labels), so a user can pass
        ``class_weights="balanced"`` and have the trainer fill in the vector.
        No-op for an explicit list, a non-``JointLoss``, or ``None``.

        Args:
            train_data: The training dataset the run is about to start on.
        """
        from auto_chasm.trainers.loss import JointLoss

        loss = self.loss_fn
        if not isinstance(loss, JointLoss):
            return
        spec = loss._class_weights
        if spec is None:
            return
        from auto_chasm.data import balanced_class_weights

        if isinstance(spec, dict):
            loss._class_weights = {
                name: (
                    balanced_class_weights(train_data, None, probe_name=name)
                    if value == "balanced"
                    else value
                )
                for name, value in spec.items()
            }
        elif spec == "balanced":
            loss._class_weights = balanced_class_weights(train_data, None, probe_name=None)

    # ------------------------------------------------------------------
    # Callback dispatch
    # ------------------------------------------------------------------

    def _fire_callback(self, event: str, **kwargs: Any) -> None:
        """Dispatch an event to callbacks; a raising callback is re-raised.

        Exceptions are logged at WARNING and re-raised (not swallowed) so a
        broken callback fails loudly instead of invisibly mid-training.
        """
        for cb in self.callbacks:
            method = getattr(cb, event, None)
            if method is not None:
                try:
                    method(**kwargs)
                except Exception:
                    logger.warning(
                        "Callback %s.%s raised an exception; re-raising.",
                        type(cb).__name__,
                        event,
                        exc_info=True,
                    )
                    raise

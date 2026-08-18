"""Joint trainer — the core training loop for auxiliary probes + LM loss.

Uses ``mx.compile`` with state tracking, ``nn.value_and_grad``,
gradient accumulation, and functional ``optimizer.update``.

For researchers who need custom training loops (RL, curriculum
learning, custom schedulers), ``JointTrainer`` also exposes a
step-level escape-hatch API::

    trainer = JointTrainer(model=model, loss=loss_fn, ...)
    for batch in trainer.iterate(train_data):
        metrics = trainer.step(batch)
        if should_log:
            trainer.log(metrics)
        if should_eval:
            val_metrics = trainer.evaluate(val_data)
        if should_checkpoint:
            trainer.save_checkpoint()
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

from auto_chasm.config import TrainingConfig
from auto_chasm.history import History, HistoryEntry
from auto_chasm.logger import get_logger
from auto_chasm.model import Model
from auto_chasm.trainers._metrics import (
    require_trainable_params,
    resolve_early_stopping_metric,
    validate_loss_weight_keys,
)
from auto_chasm.trainers.trainable import (
    LossFn,
    _TrainableModel,
    clip_grad_norm,
    evaluate_joint_model,
)

logger = get_logger(__name__)

# Sentinel for "argument not explicitly provided" (typed Any so it can default a typed
# parameter): an explicit kwarg equal to a library default must still win over config.
_UNSET: Any = object()


class JointTrainer:
    """Core training loop with configurable LR schedule, early stopping, checkpointing.

    Also exposes a step-level escape-hatch API for custom training loops.

    Args:
        model: The ``Model`` instance to train.
        loss_fn: Loss function returning ``(total, ntoks, components)``.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        grad_clip_norm: Gradient clipping max norm.
        num_iters: Total training iterations.
        batch_size: Per-step batch size.
        max_seq_length: Maximum sequence length.
        grad_accum_steps: Gradient accumulation steps.
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.  0 disables periodic saves.
        eval_steps: Evaluate every N steps.  0 disables mid-training eval.
            If ``None``, defaults to ``save_steps``.
        early_stopping_patience: Stop after N eval rounds without improvement.
            0 (the default) disables early stopping.
        restore_best_weights: If ``True``, reload the best-scoring checkpoint at
            the end of ``train()``. Default ``False`` — the FINAL-step weights
            are what you get, which is what a fixed-budget run almost always
            wants. Best-val tracking still happens (``best_iter`` is reported in
            the manifest) so the diagnostic survives; only the rollback is opt-in.
        early_stopping_metric: Metric to monitor: ``"val_loss"`` (default) or a metric
            your ``eval_metrics_fn`` emits, keyed ``"val_<probe>_<name>"`` (e.g.
            ``"val_digit_macro_f1"``).
        early_stopping_higher_is_better: If ``True``, maximize the metric (for F1, accuracy);
            if ``False``, minimize (for loss, perplexity).  Default ``False``.
        min_delta: Minimum improvement to count as an improvement.
        keep_best_only: If ``True``, delete periodic checkpoints after training,
            keeping only the best.
        save_history: If ``True``, save training history as JSON to the
            output directory after training and incrementally.  Default ``True``.
        history_save_frequency: When to save history incrementally.
            ``"val"`` — after each validation step (default).  ``"log"`` — after
            each logging step.  ``"never"`` — only at end of training.
        output_dir: Directory for checkpoints.
        verbose: Whether to print training progress to stdout.
        lr_schedule: LR schedule type (``"cosine"``, ``"linear"``, ``"constant"``).
        warmup_ratio: Fraction of total steps for linear warmup (``0.0`` = no warmup).
        eval_metrics_fn: Optional callable ``(train_model, captured, targets,
            mask) -> dict[str, float]`` producing custom validation metrics
            (e.g. F1, accuracy).  Threaded into every evaluation so its metrics
            (keyed ``"<probe>_<name>"``) become reachable for ``early_stopping_metric``.
        config: Optional ``TrainingConfig`` providing defaults for
            hyperparameters.  Individual keyword arguments override
            config values when explicitly set.
    """

    def __init__(
        self,
        model: Model,
        loss_fn: LossFn,
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
        min_delta: float = 1e-4,
        keep_best_only: bool = False,
        save_history: bool = True,
        history_save_frequency: str = "val",
        output_dir: str = _UNSET,
        verbose: bool = True,
        lr_schedule: str = _UNSET,
        warmup_ratio: float = _UNSET,
        eval_metrics_fn: Callable[..., dict[str, float]] | None = None,
        seed: int | None = None,
        mixed_precision: str = _UNSET,
        config: TrainingConfig | None = None,
    ) -> None:
        """Initialize the trainer.

        If ``config`` is provided, its values are used as defaults for
        mapped hyperparameters.  Individual keyword arguments override
        config values when explicitly provided.
        """
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten

        self.wrapper = model
        self.loss_fn = loss_fn
        self.eval_metrics_fn = eval_metrics_fn
        self.seed = seed
        self.num_iters = num_iters
        self.max_seq_length = max_seq_length
        self.early_stopping_patience = early_stopping_patience
        self.restore_best_weights = restore_best_weights
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_higher_is_better = early_stopping_higher_is_better
        self.min_delta = min_delta
        self.keep_best_only = keep_best_only
        self.save_history = save_history
        self.history_save_frequency = history_save_frequency
        self.verbose = verbose

        # An EXPLICIT kwarg wins over config even when it equals a default (unlike ==-tests).
        def _pick(explicit: Any, cfg_attr: str, default: Any) -> Any:
            if explicit is not _UNSET:
                return explicit
            return getattr(config, cfg_attr) if config is not None else default

        learning_rate = _pick(learning_rate, "learning_rate", 2e-4)
        weight_decay = _pick(weight_decay, "weight_decay", 0.0)
        grad_clip_norm = _pick(grad_clip_norm, "max_grad_norm", 1.0)
        batch_size = _pick(batch_size, "batch_size", 8)
        grad_accum_steps = _pick(grad_accum_steps, "gradient_accumulation_steps", 1)
        logging_steps = _pick(logging_steps, "logging_steps", 25)
        save_steps = _pick(save_steps, "save_steps", 100)
        output_dir = _pick(output_dir, "output_dir", "./checkpoints")
        # eval_steps: None is its unset value ("fall back to save_steps"); config fills it.
        eval_steps = config.eval_steps if eval_steps is None and config is not None else eval_steps
        lr_schedule = _pick(lr_schedule, "lr_schedule", "cosine")
        warmup_ratio = _pick(warmup_ratio, "warmup_ratio", 0.0)
        mixed_precision = _pick(mixed_precision, "mixed_precision", "fp32")
        if config is not None:
            try:
                import numpy as np

                np.random.seed(config.seed)
            except ImportError:
                pass
            mx.random.seed(config.seed)

        self.grad_clip_norm = grad_clip_norm
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps if eval_steps is not None else save_steps
        self.output_dir = Path(output_dir)

        self._train_model = _TrainableModel(model.model, model._probes)

        # bf16 mixed precision: frozen base in bf16, trainable params/optimizer fp32.
        self.mixed_precision = mixed_precision
        if mixed_precision == "bf16":
            self._train_model.base.set_dtype(mx.bfloat16)
        elif mixed_precision == "fp16":
            raise NotImplementedError(
                "mixed_precision='fp16' is torch-only (needs a GradScaler MLX lacks); "
                "use 'bf16' on MLX (numerically stable on Apple Silicon)."
            )

        self._base_lr = learning_rate
        # The schedule's horizon must be counted in OPTIMIZER UPDATES, not
        # micro-iterations: mlx.optimizers advances a schedule once per
        # optimizer.update() call, which fires only on gradient-accumulation
        # boundaries (plus the final partial flush). Sizing it by num_iters
        # stretched the schedule by grad_accum_steps — with 500 iters and
        # accum 8, warmup (10%) covered 50 of the 63 real updates (~80% of
        # training) and the cosine tail never ran. The torch loop already
        # counts updates (see _torch_loop); this makes MLX match it.
        n_updates = num_iters // grad_accum_steps + (1 if num_iters % grad_accum_steps else 0)
        warmup_steps = int(n_updates * warmup_ratio)
        self.lr_schedule = self._build_lr_schedule(
            lr_schedule, learning_rate, n_updates, warmup_steps
        )
        self.optimizer = optim.AdamW(
            learning_rate=self.lr_schedule,
            weight_decay=weight_decay,
        )

        # Escape-hatch state
        self._global_step: int = 0
        self._history = History()
        self._loss_value_and_grad: Any = None

        trainable_params = list(tree_flatten(self._train_model.trainable_parameters()))
        self._has_trainable_params = bool(trainable_params)
        logger.debug(
            "JointTrainer: trainable_groups=%d, lr=%.2e, num_iters=%d, batch=%d",
            len(trainable_params),
            learning_rate,
            num_iters,
            batch_size,
        )

    @staticmethod
    def _build_lr_schedule(
        lr_schedule: str, learning_rate: float, num_iters: int, warmup_steps: int
    ) -> Any:
        """Build an MLX LR schedule (cosine/linear/constant; delegates to _metrics)."""
        from auto_chasm.trainers._metrics import build_lr_schedule

        return build_lr_schedule(lr_schedule, learning_rate, num_iters, warmup_steps)

    def _log(self, msg: str) -> None:
        """Print and log a message if verbose."""
        logger.info(msg)
        if self.verbose:
            print(msg, flush=True)

    def run(
        self,
        train_data: Any,
        val_data: Any | None = None,
        step_callback: Callable[..., None] | None = None,
    ) -> History:
        """Run the training loop.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset (optional).
            step_callback: Optional ``step_callback(step, loss, components)`` invoked
                after every step (the ``Trainer`` facade uses it for ``on_step_end``).

        Returns:
            ``History`` object with all logged metrics.
        """
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_map

        from auto_chasm.trainers.data_utils import iterate_batches, labels_to_mlx

        require_trainable_params(self._has_trainable_params)
        model = self._train_model
        loss_fn = self.loss_fn
        # Eager weight-key validation BEFORE the trace (in-trace error poisons mx.random.state).
        validate_loss_weight_keys(loss_fn, getattr(model, "_probe_names", ()))
        loss_value_and_grad = nn.value_and_grad(model, loss_fn)

        grad_accum_steps = self.grad_accum_steps
        state = [model.state, self.optimizer.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(
            batch: Any,
            labels: Any,
            lengths: Any,
            prev_grad: Any,
            do_update: bool,
        ) -> tuple[Any, Any, dict[str, Any], Any]:
            """Compute ``(loss, ntoks, components, grad)``; accumulate/update per flags."""
            (lvalue, toks, components), grad = loss_value_and_grad(model, batch, labels, lengths)
            if prev_grad is not None:
                grad = tree_map(lambda x, y: x + y, grad, prev_grad)

            if do_update:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
                if self.grad_clip_norm > 0:
                    grad = clip_grad_norm(grad, self.grad_clip_norm)
                self.optimizer.update(model, grad)
                grad = None

            return lvalue, toks, components, grad

        model.train()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = str(self.output_dir / "adapters.safetensors")

        losses = mx.array(0.0)
        n_tokens = mx.array(0.0)
        component_accum: dict[str, Any] = {}
        steps = 0
        train_time = 0.0
        wall_start = time.perf_counter()
        grad_accum = None

        best_metric = -float("inf") if self.early_stopping_higher_is_better else float("inf")
        patience_counter = 0
        best_iter = 0
        saved_best = False  # did THIS run save a best? (else never restore a stale file)
        es_active = self.early_stopping_patience > 0 and val_data is not None

        history = History()
        self._history = history

        self._log(
            f"Starting training: {self.num_iters} iters, "
            f"batch_size={self.batch_size}, lr={self._base_lr:.2e}"
        )
        if es_active:
            self._log(
                f"  Early stopping: patience={self.early_stopping_patience}, "
                f"metric={self.early_stopping_metric}, min_delta={self.min_delta}"
            )

        batches = iterate_batches(
            train_data, self.batch_size, self.max_seq_length, loop=True, seed=self.seed
        )
        for it, batch in zip(range(1, self.num_iters + 1), batches, strict=False):
            tic = time.perf_counter()

            # Take the optimizer step FIRST, then evaluate — so an eval (and any best
            # checkpoint) measures the model AFTER this update (else the untrained init
            # became a "best" and the final step's state went unevaluated).
            tokens, labels, lengths = batch
            tokens = mx.array(tokens)
            labels = labels_to_mlx(labels)
            lengths = mx.array(lengths)
            # Flush the last partial accumulation group on the final iteration (torch parity).
            do_update = (it % grad_accum_steps == 0) or (it == self.num_iters)
            lvalue, toks, components, grad_accum = step(
                tokens, labels, lengths, grad_accum, do_update
            )

            should_eval = (
                val_data is not None
                and self.eval_steps > 0
                and (it % self.eval_steps == 0 or it == self.num_iters)
            )
            if should_eval:
                val_metrics = evaluate_joint_model(
                    train_model=model,
                    dataset=val_data,
                    batch_size=self.batch_size,
                    max_seq_length=self.max_seq_length,
                    loss_fn=loss_fn,
                    eval_metrics_fn=self.eval_metrics_fn,
                )
                model.train()

                val_parts = [f"Val loss {val_metrics['loss']:.4f}"]
                if "perplexity" in val_metrics:
                    val_parts.append(f"PPL {val_metrics['perplexity']:.2f}")
                # Probe metrics only when computed (an eval_metrics_fn was given);
                # never a placeholder "F1 0.0000" that looks like a real result.
                f1s = (f"{k} {v:.4f}" for k, v in val_metrics.items() if k.endswith("_f1"))
                val_parts.extend(f1s)
                self._log(f"Iter {it}: {', '.join(val_parts)}")

                # Record val metrics in history
                history.append(
                    HistoryEntry(
                        step=it,
                        val_loss=val_metrics["loss"],
                        val_metrics=val_metrics,
                        wall_time=time.perf_counter() - wall_start,
                    )
                )
                if self.save_history and self.history_save_frequency in ("val", "log"):
                    history_path = self.output_dir / "training_history.json"
                    history.save_json(history_path)

                current_metric = resolve_early_stopping_metric(
                    self.early_stopping_metric, val_metrics
                )
                improved = (
                    current_metric > best_metric + self.min_delta
                    if self.early_stopping_higher_is_better
                    else current_metric < best_metric - self.min_delta
                )

                if improved:
                    best_metric = current_metric
                    best_iter = it
                    saved_best = True
                    self._save_best_weights(adapter_file)
                    self._log(
                        f"  -> New best {self.early_stopping_metric}="
                        f"{current_metric:.4f}, saved checkpoint."
                    )
                    if es_active:
                        patience_counter = 0
                elif es_active:
                    patience_counter += 1
                    self._log(
                        f"  -> No improvement ({patience_counter}/{self.early_stopping_patience})"
                    )
                    if patience_counter >= self.early_stopping_patience:
                        self._log(f"Early stopping at iter {it}. Best was {best_iter}.")
                        break

            losses = losses + lvalue
            n_tokens = n_tokens + toks
            for key, val in components.items():
                if key not in component_accum:
                    component_accum[key] = mx.array(0.0)
                component_accum[key] = component_accum[key] + val
            steps += 1

            if step_callback is not None:
                step_callback(step=it, loss=float(lvalue), components=components)

            eval_args = [state, losses, n_tokens, grad_accum]
            eval_args.extend(component_accum.values())
            mx.eval(*eval_args)
            train_time += time.perf_counter() - tic

            if it <= 3 or it % 50 == 0:
                comp_str = ", ".join(f"{k}={float(v):.4f}" for k, v in components.items())
                logger.debug(
                    "step %d: lvalue=%.4f, %s, toks=%s, grad_accum=%s",
                    it,
                    float(lvalue),
                    comp_str,
                    int(toks) if hasattr(toks, "item") else toks,
                    "present" if grad_accum is not None else "None",
                )

            if it % self.logging_steps == 0 or it == self.num_iters:
                train_loss = losses.item() / steps
                avg_components = {k: v.item() / steps for k, v in component_accum.items()}
                if callable(self.lr_schedule):
                    # The schedule advances once per OPTIMIZER UPDATE (its
                    # horizon is n_updates, not num_iters), so index it by the
                    # number of updates completed by iteration ``it`` — indexing
                    # by micro-iteration would read the curve 8x too far in and
                    # log a fictitious LR. max(..., 0) covers logging before the
                    # first accumulation boundary.
                    updates_done = it // self.grad_accum_steps + (
                        1 if it == self.num_iters and it % self.grad_accum_steps else 0
                    )
                    raw_lr = self.lr_schedule(max(updates_done - 1, 0))
                    lr = raw_lr.item() if hasattr(raw_lr, "item") else float(raw_lr)
                else:
                    lr = self._base_lr
                it_sec = self.logging_steps / train_time if train_time > 0 else 0
                tokens_sec = float(n_tokens.item()) / train_time if train_time > 0 else 0

                comp_parts = ", ".join(f"{k} {v:.4f}" for k, v in avg_components.items())
                self._log(
                    f"Iter {it}: Train loss {train_loss:.4f} "
                    f"({comp_parts}), "
                    f"LR {lr:.3e}, It/sec {it_sec:.3f}, Tokens/sec {tokens_sec:.1f}"
                )

                history.append(
                    HistoryEntry(
                        step=it,
                        train_loss=train_loss,
                        loss_components=avg_components,
                        learning_rate=lr,
                        it_sec=it_sec,
                        tokens_sec=tokens_sec,
                        wall_time=time.perf_counter() - wall_start,
                    )
                )
                if self.save_history and self.history_save_frequency == "log":
                    history_path = self.output_dir / "training_history.json"
                    history.save_json(history_path)

                losses = mx.array(0.0)
                n_tokens = mx.array(0.0)
                component_accum = {}
                steps = 0
                train_time = 0.0

            if self.save_steps > 0 and it % self.save_steps == 0:
                self._save_checkpoint(adapter_file, it)

        self._train_model.restore_capture_fns()
        # OPT-IN rollback. The final-step weights stand unless the caller asked for
        # the best-scoring checkpoint: a fixed-budget run that merely logs a val
        # curve used to be silently rewound to whichever eval scored best, which is
        # not what "train for N iters" means and is actively wrong for an unlearning
        # run (val loss there is not monotone by construction). `saved_best` still
        # guards it, so a leftover best-file from a previous run is never loaded
        # into (e.g.) a LayerSweep pass that saved no best of its own.
        if saved_best and self.restore_best_weights:
            self._restore_best(adapter_file)
            self._log(f"Training complete. Restored best {self.early_stopping_metric} "
                      f"from iter {best_iter}.")
        elif saved_best:
            self._log(f"Training complete. Kept final-step weights "
                      f"(best {self.early_stopping_metric} was at iter {best_iter}; "
                      f"pass restore_best_weights=True to roll back to it).")
        else:
            self._log("Training complete.")

        self.wrapper.save_checkpoint(str(self.output_dir / "final"))

        self._save_training_manifest(best_iter, best_metric)
        if self.keep_best_only:
            self._cleanup_periodic_checkpoints()

        if self.save_history:
            history_path = self.output_dir / "training_history.json"
            history.save_json(history_path)
            self._log(f"  Training history saved to {history_path}")

        return history

    def train(
        self,
        train_data: Any,
        val_data: Any | None = None,
    ) -> dict[str, Any]:
        """Run training and return the unified trainer result.

        Thin wrapper over :meth:`run` providing the same ``{"history",
        "output_dir"}`` contract exposed by ``Trainer``, ``SFTTrainer`` and
        ``RLTrainer``.  Use :meth:`run` directly for the lower-level
        ``History``-returning API.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset (optional).

        Returns:
            Dict with keys ``"history"`` (``History``) and ``"output_dir"``.
        """
        history = self.run(train_data, val_data)
        return {"history": history, "output_dir": str(self.output_dir)}

    def iterate(self, train_data: Any) -> Iterator[tuple[Any, Any, Any]]:
        """Iterate over training batches.

        Simple wrapper around ``iterate_batches()`` with ``loop=True``.
        Researchers use this with ``trainer.step()`` to write custom
        training loops.

        Args:
            train_data: Training dataset.

        Yields:
            Tuple of ``(tokens, labels, lengths)`` as numpy arrays.
        """
        from auto_chasm.trainers.data_utils import iterate_batches

        return iterate_batches(
            train_data, self.batch_size, self.max_seq_length, loop=True, seed=self.seed
        )

    def step(self, batch: tuple[Any, Any, Any]) -> dict[str, Any]:
        """Run one training step (forward + backward + optimizer update).

        Builds the ``nn.value_and_grad`` function lazily on the first
        call.  No ``mx.compile`` or gradient accumulation — this is
        the simple one-step-at-a-time implementation for researchers
        who prioritise flexibility over performance.

        Args:
            batch: ``(tokens, labels, lengths)`` tuple.

        Returns:
            Dict with keys ``"loss"`` (float), ``"ntoks"`` (int),
            ``"components"`` (dict of str → float).
        """
        import mlx.core as mx
        import mlx.nn as nn

        from auto_chasm.trainers.data_utils import labels_to_mlx
        from auto_chasm.trainers.trainable import clip_grad_norm

        require_trainable_params(self._has_trainable_params)
        if self._loss_value_and_grad is None:
            self._loss_value_and_grad = nn.value_and_grad(self._train_model, self.loss_fn)

        tokens, labels, lengths = batch
        tokens = mx.array(tokens)
        labels = labels_to_mlx(labels)
        lengths = mx.array(lengths)

        (lvalue, toks, components), grad = self._loss_value_and_grad(
            self._train_model, tokens, labels, lengths
        )

        if self.grad_clip_norm > 0:
            grad = clip_grad_norm(grad, self.grad_clip_norm)

        self.optimizer.update(self._train_model, grad)
        mx.eval(self._train_model.state, self.optimizer.state)
        self._global_step += 1

        return {
            "loss": float(lvalue),
            "ntoks": int(toks) if hasattr(toks, "item") else int(toks),
            "components": {k: float(v) for k, v in components.items()},
        }

    def evaluate(self, val_data: Any, num_batches: int = -1) -> dict[str, float]:
        """Evaluate on a validation dataset.

        Args:
            val_data: Validation dataset.
            num_batches: Max batches (-1 = all).

        Returns:
            Dict of evaluation metrics (loss, perplexity, etc.).
        """
        return evaluate_joint_model(
            train_model=self._train_model,
            dataset=val_data,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            loss_fn=self.loss_fn,
            num_batches=num_batches,
            eval_metrics_fn=self.eval_metrics_fn,
        )

    def save_checkpoint(self, path: str | None = None) -> str:
        """Save the current model state (probes + adapters).

        If ``path`` is ``None``, saves to
        ``{output_dir}/checkpoint_{global_step}``.

        Args:
            path: Directory path to save to.  Created if it does not exist.

        Returns:
            The path the checkpoint was saved to.
        """
        if path is None:
            path = str(self.output_dir / f"checkpoint_{self._global_step}")
        self.wrapper.save_checkpoint(path)
        return path

    def restore_checkpoint(self, path: str) -> None:
        """Restore model weights from a checkpoint directory.

        Args:
            path: Path to a checkpoint directory previously saved with
                ``save_checkpoint()`` or ``Model.save_checkpoint()``.
        """
        from auto_chasm.checkpoint import ADAPTERS_NAME, PROBES_DIR

        p = Path(path)

        adapter_path = p / ADAPTERS_NAME
        if adapter_path.exists():
            self._train_model.load_weights(str(adapter_path), strict=False)
            logger.info("Restored adapter weights from %s", adapter_path)

        probes_path = p / PROBES_DIR
        if probes_path.exists():
            for name in self._train_model._probe_names:
                head_path = probes_path / f"{name}.safetensors"
                if head_path.exists():
                    # load_weights un-flattens dotted safetensors keys before
                    # update(); update() alone rejects flat keys (submodule heads).
                    self._train_model.get_probe(name).load_weights(str(head_path), strict=False)
                    logger.info("Restored probe '%s' from %s", name, head_path)

    def get_history(self) -> History:
        """Return the current training history.

        Returns:
            ``History`` object with all logged metrics.
        """
        return self._history

    def _save_best_weights(self, adapter_file: str) -> None:
        """Save the best checkpoint (adapters + probe heads)."""
        import mlx.core as mx
        from mlx.utils import tree_flatten

        model = self._train_model
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(adapter_file, adapter_weights)

        for name in model._probe_names:
            head_path = self.output_dir / f"{name}_head.safetensors"
            head_weights = dict(tree_flatten(model.get_probe(name).parameters()))
            mx.save_safetensors(str(head_path), head_weights)

    def _save_checkpoint(self, _adapter_file: str, it: int) -> None:
        """Save a timestamped checkpoint for the given iteration."""
        import mlx.core as mx
        from mlx.utils import tree_flatten

        model = self._train_model
        checkpoint = self.output_dir / f"{it:07d}_adapters.safetensors"
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(checkpoint), adapter_weights)

        for name in model._probe_names:
            head_path = self.output_dir / f"{it:07d}_{name}_head.safetensors"
            head_weights = dict(tree_flatten(model.get_probe(name).parameters()))
            mx.save_safetensors(str(head_path), head_weights)

        self._log(f"  Saved checkpoint at iter {it}")

    def _restore_best(self, adapter_file: str) -> None:
        """Restore the best checkpoint after training."""
        model = self._train_model
        best_path = Path(adapter_file)
        if best_path.exists():
            model.load_weights(str(best_path), strict=False)
            self._log("Restored best adapter checkpoint.")

        for name in model._probe_names:
            head_path = self.output_dir / f"{name}_head.safetensors"
            if head_path.exists():
                # load_weights un-flattens the dotted safetensors keys before update;
                # update() alone expects a nested tree and rejects flat keys.
                model.get_probe(name).load_weights(str(head_path), strict=False)
                self._log(f"Restored best head for probe '{name}'.")

    def _save_training_manifest(self, best_iter: int, best_metric: float) -> None:
        """Write a training manifest with best-checkpoint metadata."""
        from auto_chasm.trainers._metrics import write_training_manifest

        path = write_training_manifest(self, best_iter, best_metric)
        self._log(f"  Training manifest saved to {path}")

    def _cleanup_periodic_checkpoints(self) -> None:
        """Delete periodic checkpoint files, keeping only the best."""
        count = 0
        for path in self.output_dir.iterdir():
            name = path.name
            if (
                path.is_file()
                and name[:7].isdigit()  # the {step:07d}_ prefix only (spares 3d_head.* etc.)
                and ("_adapters.safetensors" in name or "_head.safetensors" in name)
            ):
                path.unlink()
                count += 1
        if count > 0:
            self._log(f"  Cleaned up {count} periodic checkpoint files (keep_best_only=True).")

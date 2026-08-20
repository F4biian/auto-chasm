"""Layer sweep: train one probe head per layer and report the best per layer.

:class:`LayerSweep` attaches one head to every transformer layer, trains them
together in a single frozen-base pass, and — on its own eval cadence — snapshots
each head's weights at *its own* best-validation step.  After training it
restores every head to its best snapshot (so the live model keeps the trained
probes) and evaluates the test set once on those restored heads.  The result is
each layer's accuracy and adjacent (ordinal) accuracy at its own best checkpoint.

Backend-agnostic: it drives ``Trainer.train`` + ``Trainer.evaluate`` (both run on
MLX and torch) and never uses the MLX-only ``Trainer.step``.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

from auto_chasm.config import ProbeConfig
from auto_chasm.metrics import classification_metrics
from auto_chasm.task import Task
from auto_chasm.trainers.wrappers import TrainerCallback

if TYPE_CHECKING:
    from auto_chasm.dataset import Dataset
    from auto_chasm.model import Model
    from auto_chasm.modules import ModuleSpec
    from auto_chasm.trainers.loss import LossFn

    # What the sweep accepts as a dataset: a Dataset, or a raw list of samples.
    DatasetLike: TypeAlias = Dataset | Sequence[dict[str, Any]]
    # A probe head spec: a built-in name, a ModuleSpec, or a builder callable.
    HeadSpec: TypeAlias = str | ModuleSpec | Callable[..., Any]


def _snapshot(module: Any, backend: str) -> Any:
    """Clone a probe module's parameters (detached) for later restoration."""
    if backend == "torch":
        return {k: v.detach().clone() for k, v in module.state_dict().items()}
    import mlx.core as mx
    from mlx.utils import tree_flatten

    return [(k, mx.array(v)) for k, v in tree_flatten(module.parameters())]


def _restore(module: Any, snapshot: Any, backend: str) -> None:
    """Restore a probe module's parameters from a :func:`_snapshot`."""
    if backend == "torch":
        module.load_state_dict(snapshot)
        return
    from mlx.utils import tree_unflatten

    module.update(tree_unflatten(snapshot))


def _set_train_mode(model: Any, trainer: Any) -> None:
    """Return the trained module to train mode after a callback evaluation."""
    if model.backend.name == "torch":
        raw = getattr(model, "model", None)
        if raw is not None and hasattr(raw, "train"):
            raw.train()
        return
    joint = getattr(trainer, "_joint_trainer", None)
    train_model = getattr(joint, "_train_model", None)
    if train_model is not None:
        train_model.train()


def _layer_loss(metrics: dict[str, float], name: str) -> float:
    """Pull a layer's per-probe loss component out of an evaluate() dict.

    ``JointLoss`` keys each probe's loss component by its BARE probe name (e.g.
    ``L3``) — so a per-layer sweep must rank each layer on ITS OWN component, never
    the combined LM+probe total.  The bare name is preferred; older ``{loss}:{probe}``
    / ``probe_*`` component keys are honored as a fallback; the total ``loss`` is a
    last resort only.
    """
    if name in metrics:
        return metrics[name]
    for key, value in metrics.items():
        if key.endswith(f":{name}"):
            return value
    for key, value in metrics.items():
        if key.startswith("probe_") and ":" not in key:
            return value
    return metrics.get("loss", float("inf"))


class _BestPerLayerCallback(TrainerCallback):
    """Evaluate val on a cadence and snapshot each head at its own best-val step."""

    def __init__(
        self,
        model: Model,
        val_data: DatasetLike,
        names: list[str],
        score_metric: str,
        higher_is_better: bool,
        eval_every: int,
        num_iters: int,
        patience: int = 0,
        min_delta: float = 0.0,
    ) -> None:
        """Initialize the per-layer best tracker."""
        self.model = model
        self.val_data = val_data
        self.names = names
        self.score_metric = score_metric
        self.higher_is_better = higher_is_better
        self.eval_every = eval_every
        self.num_iters = num_iters
        self.patience = patience
        self.min_delta = min_delta
        self.trainer: Any = None
        self.best: dict[str, dict[str, Any]] = {}
        #: evals since this layer last improved, and the step it gave up at.
        self.stale: dict[str, int] = dict.fromkeys(names, 0)
        self.stopped: dict[str, int] = {}

    def _score(self, metrics: dict[str, float], name: str) -> float:
        """Resolve the scalar this layer is ranked on at the current eval.

        Raises loudly when ``score_metric`` names a metric this eval does not
        produce, instead of silently scoring ``0.0`` (which tied every layer and
        disabled per-layer best selection).
        """
        if self.score_metric in ("val_loss", "loss"):
            return _layer_loss(metrics, name)
        key = self.score_metric.removeprefix("val_")  # strip the prefix, not mid-string
        if key == "f1":  # classification_metrics emits "<probe>_macro_f1"
            key = "macro_f1"
        full = f"{name}_{key}"
        if full not in metrics:
            avail = sorted(k[len(name) + 1 :] for k in metrics if k.startswith(f"{name}_"))
            options = ", ".join(avail) if avail else "loss only (pass task= or eval_metrics_fn)"
            raise ValueError(
                f"score_metric={self.score_metric!r} needs metric {full!r}, which this "
                f"eval does not produce (layer {name!r} has: {options}). Use 'val_loss', "
                "'val_acc', 'val_adj', or 'val_macro_f1'."
            )
        return metrics[full]

    def _layer_val(self, metrics: dict[str, float], name: str) -> dict[str, float]:
        """Collect EVERY validation number this layer produced at the current eval.

        Hardcoding acc/adj meant a custom ``eval_metrics_fn`` could rank layers on
        its metric (``score_metric="val_auroc"`` resolves fine) yet never see that
        metric again -- it reached neither the result rows nor the CSV.
        """
        out = {"val_loss": _layer_loss(metrics, name)}
        prefix = f"{name}_"
        for key, value in metrics.items():
            if key.startswith(prefix):
                out[f"val_{key[len(prefix):]}"] = value
        out.setdefault("val_acc", 0.0)
        out.setdefault("val_adj", 0.0)
        return out

    def on_step_end(self, **kwargs: Any) -> None:
        """On the eval cadence, evaluate val and snapshot each improved head."""
        step = int(kwargs.get("step", 0))
        if step % self.eval_every != 0 and step != self.num_iters:
            return
        metrics = self.trainer.evaluate(self.val_data)
        for name in self.names:
            if name in self.stopped:
                continue  # plateaued: its best snapshot is already kept
            score = self._score(metrics, name)
            current = self.best.get(name)
            # min_delta is a MARGIN, so noise around a plateau does not read as
            # progress and reset the counter forever.
            improved = current is None or (
                score > current["score"] + self.min_delta
                if self.higher_is_better
                else score < current["score"] - self.min_delta
            )
            if improved:
                self.stale[name] = 0
                self.best[name] = {
                    "iter": step,
                    "score": score,
                    "val": self._layer_val(metrics, name),
                    "snapshot": _snapshot(self.model.probes[name].module, self.model.backend.name),
                }
            elif self.patience > 0:
                self.stale[name] += 1
                if self.stale[name] >= self.patience:
                    self.stopped[name] = step
        _set_train_mode(self.model, self.trainer)

        if self.patience > 0 and len(self.stopped) == len(self.names) and self.trainer is not None:
            # Nothing left to learn anywhere: heads share ONE forward pass, so a
            # layer cannot be stopped on its own -- but once all of them have
            # plateaued the remaining iterations are pure waste.
            self.trainer.stop_requested = True

    def restore_best(self) -> None:
        """Restore every head to its best-validation snapshot (in place)."""
        backend = self.model.backend.name
        for name in self.names:
            record = self.best.get(name)
            if record is not None and record["snapshot"] is not None:
                _restore(self.model.probes[name].module, record["snapshot"], backend)


@dataclass
class SweepResult:
    """Per-layer best-validation results, with CSV and plot writers.

    Attributes:
        best: ``{layer_index: {"iter", "val_loss", "val_acc", "val_adj",
            "test_loss", "test_acc", "test_adj"}}``.
        model: The model whose probe heads now hold each layer's best-val weights.
    """

    best: dict[int, dict[str, float]] = field(default_factory=dict)
    model: Model | None = None

    def to_csv(self, path: str) -> None:
        """Write one row per layer, with EVERY metric the run produced.

        Columns are derived from the rows rather than fixed, so a custom
        ``eval_metrics_fn`` (an AUROC, say) reaches the file. The historical
        columns keep their names -- ``val_adj``/``test_adj`` are written as
        ``val_group_acc``/``test_group_acc`` -- so existing readers still work.
        """
        alias = {"val_adj": "val_group_acc", "test_adj": "test_group_acc"}
        lead = ["val_loss", "val_acc", "val_adj", "test_loss", "test_acc", "test_adj"]
        seen: list[str] = []
        for row in self.best.values():
            seen.extend(k for k in row if k not in seen and k != "iter")
        ordered = [k for k in lead if k in seen] + sorted(k for k in seen if k not in lead)

        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["layer", "iter", *(alias.get(k, k) for k in ordered)])
            for i in sorted(self.best):
                row = self.best[i]
                writer.writerow(
                    [i, row["iter"], *(row.get(k, float("nan")) for k in ordered)]
                )

    def plot(self, path: str, title: str = "Test Performance") -> None:
        """Plot per-layer test accuracy and group (adjacent) accuracy to ``path``.

        Args:
            path: Output image path.
            title: Plot title.

        Raises:
            ImportError: If matplotlib is not installed (it is not a core dep).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "SweepResult.plot() needs matplotlib. Install it with "
                "`pip install matplotlib` (it is not a core auto-chasm dependency)."
            ) from exc

        xs = sorted(self.best)
        acc = [self.best[i]["test_acc"] * 100 for i in xs]
        grp = [self.best[i]["test_adj"] * 100 for i in xs]
        plt.figure()
        plt.plot(xs, acc, color="tab:blue", label="Accuracy (%)")
        plt.plot(xs, grp, color="tab:orange", label="Group Accuracy (%)")
        plt.xlabel("Layer")
        plt.title(title)
        plt.ylim(0, 100)
        plt.legend()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def best_layer(self, metric: str = "test_adj") -> int:
        """Return the layer index that maximizes ``metric`` (e.g. ``"test_acc"``)."""
        return max(self.best, key=lambda i: self.best[i][metric])


class LayerSweep:
    """Train one probe head per layer and report each layer's best-val result.

    The head width, class count, and metrics can be supplied EITHER explicitly
    (``out_features=`` [+ ``num_classes=``/``eval_metrics_fn=``]) OR — the DX-first
    way — by handing over a :class:`~auto_chasm.Task` inferred from the data
    (``task=full.infer_task("probe")``), which derives all three consistently so a
    classification↔regression switch is a one-argument change.

    Args:
        model: A fresh ``Model`` (no probes attached yet).
        task: A :class:`~auto_chasm.Task` (e.g. from ``Dataset.infer_task``) that
            supplies ``out_features``, ``num_classes``, and the matching
            ``eval_metrics_fn`` (classification vs regression).  Mutually exclusive
            with ``out_features=``/``num_classes=``.
        out_features: Output width of every head (number of classes).  Required
            unless ``task=`` is given.
        module_spec: A :class:`~auto_chasm.ModuleSpec` (or callable) for the head.
            ``None`` uses a built-in single ``Linear``.
        layers: Layer indices to sweep.  ``None`` sweeps all layers.
        num_classes: Class count for the metrics.  ``None`` uses ``out_features``.
        ordinal_tol: Tolerance for the adjacent-accuracy metric.
        score_metric: Which validation metric selects each layer's best checkpoint
            (``"val_loss"``, ``"val_acc"``, or ``"val_adj"``).
        higher_is_better: Whether ``score_metric`` is maximized.
        eval_metrics_fn: The per-probe metric callable (trainer metric signature).
            ``None`` uses :func:`~auto_chasm.classification_metrics` (or the task's
            metrics when ``task=`` is given); pass
            :func:`~auto_chasm.regression_metrics` for an ``out_features=1``
            regression sweep (it still emits ``{name}_acc``/``{name}_adj`` from the
            discretized prediction, so the CSV/plot columns are unchanged).

    Raises:
        ValueError: If both ``task=`` and ``out_features=``/``num_classes=`` are
            given, or if neither ``task=`` nor ``out_features=`` is provided.
    """

    def __init__(
        self,
        model: Model,
        *,
        task: Task | None = None,
        out_features: int | None = None,
        module_spec: HeadSpec | None = None,
        layers: Sequence[int] | None = None,
        num_classes: int | None = None,
        ordinal_tol: int = 1,
        score_metric: str = "val_loss",
        higher_is_better: bool = False,
        early_stopping_patience: int = 0,
        min_delta: float = 0.0,
        eval_metrics_fn: Callable[..., dict[str, float]] | None = None,
    ) -> None:
        """Configure the sweep."""
        if task is not None:
            if out_features is not None or num_classes is not None:
                raise ValueError(
                    "Pass either task= (which derives out_features + num_classes + "
                    "metrics) or out_features=/num_classes= explicitly, not both."
                )
            out_features = task.out_features
            num_classes = task.num_classes
            if eval_metrics_fn is None:
                eval_metrics_fn = task.build_metrics(ordinal_tol=ordinal_tol)
        if out_features is None:
            raise ValueError(
                "LayerSweep needs the head width: pass out_features= (or a task= to "
                "derive it from the data)."
            )
        self.task = task
        self.model = model
        self.out_features = out_features
        self.module_spec = module_spec
        self.layers = list(layers) if layers is not None else list(range(model.num_layers))
        self.num_classes = num_classes if num_classes is not None else out_features
        self.ordinal_tol = ordinal_tol
        self.score_metric = score_metric
        self.higher_is_better = higher_is_better
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.eval_metrics_fn = eval_metrics_fn

    def _attach_heads(self) -> list[str]:
        """Attach one head per layer; return the probe names."""
        names = [f"L{i}" for i in self.layers]
        configs = []
        for i, name in zip(self.layers, names, strict=True):
            kwargs: dict[str, Any] = {"module_config": {"out_features": self.out_features}}
            if self.module_spec is not None:
                kwargs["module_type"] = self.module_spec
            configs.append(ProbeConfig(name=name, layers=[i], **kwargs))
        self.model.add_probes(configs)
        self.model.freeze_model()
        self.model.unfreeze_all_probes()
        return names

    def run(
        self,
        train_data: DatasetLike,
        val_data: DatasetLike,
        test_data: DatasetLike | None = None,
        *,
        loss_fn: LossFn,
        num_iters: int,
        eval_every: int,
        **trainer_kwargs: Any,
    ) -> SweepResult:
        """Run the sweep and return per-layer best-validation results.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset (drives per-layer checkpoint selection).
            test_data: Test dataset, evaluated once on the restored best heads.
                ``None`` SKIPS that pass and leaves the ``test_*`` columns empty —
                use it when the test numbers come from ``model.probe_scores``
                instead, which needs its own pass anyway and additionally yields
                confidence intervals. Scoring the same set twice is pure waste.
            loss_fn: The loss (e.g. pure-probe ``JointLoss(weights={"lm_head": 0.0})``,
                or ``JointLoss(weights={"lm_head": 0.0}, losses={"<probe>": "ce"})``).
            num_iters: Total training iterations.
            eval_every: Validate (and snapshot best heads) every N steps.
            **trainer_kwargs: Forwarded to ``Trainer`` (batch_size, learning_rate,
                max_seq_length, warmup_ratio, weight_decay, verbose, ...).

        Returns:
            A :class:`SweepResult`; ``self.model`` now holds each layer's best
            head (save it with ``model.save_checkpoint`` to keep the probes).
        """
        from auto_chasm.trainers.trainer import Trainer

        names = self._attach_heads()
        metric_fn = self.eval_metrics_fn or classification_metrics(
            self.num_classes, self.ordinal_tol
        )
        callback = _BestPerLayerCallback(
            self.model,
            val_data,
            names,
            self.score_metric,
            self.higher_is_better,
            eval_every,
            num_iters,
            self.early_stopping_patience,
            self.min_delta,
        )
        # These three are OWNED by the sweep: it runs its own per-layer validation
        # and keeps each head at its own best step (see _BestPerLayerCallback), so
        # the trainer's global eval/save/early-stop would either duplicate that or
        # stop EVERY layer when one plateaus. Passing them through **trainer_kwargs
        # used to surface as "got multiple values for keyword argument", which does
        # not say which argument or why.
        reserved = {"eval_steps", "save_steps", "early_stopping_patience"} & set(trainer_kwargs)
        if reserved:
            raise TypeError(
                f"LayerSweep.run() manages {sorted(reserved)} itself; remove them from the "
                "call. Per-layer checkpoint selection replaces them: use eval_every= for the "
                "validation cadence, and score_metric=/higher_is_better= on LayerSweep(...) "
                "to choose what each layer's 'best' means."
            )
        trainer = Trainer(
            model=self.model,
            loss_fn=loss_fn,
            num_iters=num_iters,
            eval_steps=0,
            save_steps=0,
            early_stopping_patience=0,
            eval_metrics_fn=metric_fn,
            callbacks=[callback],
            **trainer_kwargs,
        )
        callback.trainer = trainer
        trainer.train(train_data)

        callback.restore_best()
        # Skipping this saves a FULL pass over the test set. The heads are already
        # restored, so model.probe_scores(test_data) reads exactly these weights.
        test_metrics: dict[str, float] = {} if test_data is None else trainer.evaluate(test_data)

        best: dict[int, dict[str, float]] = {}
        for i, name in zip(self.layers, names, strict=True):
            record = callback.best.get(name, {})
            val = record.get("val", {})
            row = {
                "iter": float(record.get("iter", 0)),
                # The step this layer plateaued at (nan = still improving at the end).
                "stopped_at": float(callback.stopped.get(name, float("nan"))),
                "test_loss": _layer_loss(test_metrics, name) if test_metrics else float("nan"),
                "test_acc": test_metrics.get(f"{name}_acc", float("nan")),
                "test_adj": test_metrics.get(f"{name}_adj", float("nan")),
            }
            row.update(val)
            row.setdefault("val_loss", float("nan"))
            # Every custom test metric too, so eval_metrics_fn output survives to
            # the CSV rather than stopping at the ranking step.
            prefix = f"{name}_"
            for key, value in test_metrics.items():
                if key.startswith(prefix):
                    row[f"test_{key[len(prefix):]}"] = value
            best[i] = row
        return SweepResult(best=best, model=self.model)

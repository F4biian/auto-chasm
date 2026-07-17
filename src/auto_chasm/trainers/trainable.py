"""Trainable model wrapper and training utilities for MLX.

Provides ``_TrainableModel`` (wraps base model + probes for
``nn.value_and_grad``), the MLX-specific ``make_joint_loss``,
``clip_grad_norm``, and ``evaluate_joint_model``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from auto_chasm.config import LM_HEAD
from auto_chasm.logger import get_logger

logger = get_logger(__name__)

LossFn = Callable[[Any, Any, Any, Any], tuple[Any, Any, dict[str, Any]]]

try:
    import mlx.nn as nn

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False
    nn = None

_BaseModule: type = nn.Module if _MLX_AVAILABLE else object


class _TrainableModel(_BaseModule):  # type: ignore[misc]
    """Wraps base model + probes so ``optimizer.update`` sees all weights.

    Mirrors the proven ``JointModel`` from ``test_joint_sft.py``:
    ``__call__`` returns ``(lm_logits, probe_logits)`` so that
    ``nn.value_and_grad`` captures gradients through the probe.

    **Key design:** Probe heads are stored with a non-underscore prefix
    (``self.probe_digit``) because MLX's ``nn.Module`` silently drops
    ``_``-prefixed attributes from the parameter tree — making them
    invisible to ``trainable_parameters()`` and ``nn.value_and_grad``.
    Captured hidden states use ``self._captured_hidden`` (underscore is
    fine here — it's transient state, not a trainable parameter).
    """

    def __init__(self, base: Any, probes: dict[str, Any]) -> None:
        """Initialize the trainable model wrapper."""
        if not _MLX_AVAILABLE:
            raise ImportError("MLX is required for _TrainableModel. Install with: uv add mlx")
        from mlx.utils import tree_flatten

        super().__init__()
        self.base = base
        self._probe_names_list: list[str] = []
        self._captured_hidden: dict[str, Any] = {}
        self._orig_capture_fns: list[tuple[Any, Any]] = []

        for name, probe in probes.items():
            attr = f"probe_{name}"
            # Respect the user's freeze choice: only unfreeze probes that were
            # trainable before wrapping. Unconditionally unfreezing here would
            # silently override Model.freeze_probe(name).
            was_trainable = len(tree_flatten(probe.module.trainable_parameters())) > 0
            setattr(self, attr, probe.module)
            if was_trainable:
                getattr(self, attr).unfreeze()
            self._probe_names_list.append(name)

        for probe_name, probe in probes.items():
            for lc in probe.layer_captures:
                original_fn = lc.capture_fn

                def _make_capture(model_ref: _TrainableModel, p_name: str, orig_fn: Any) -> Any:
                    def _capture(h: Any) -> None:
                        orig_fn(h)
                        model_ref._captured_hidden[p_name] = h

                    return _capture

                self._orig_capture_fns.append((lc, original_fn))
                lc.capture_fn = _make_capture(self, probe_name, original_fn)

        self._probe_captures = dict(probes.items())

        base_trainable = len(list(tree_flatten(self.base.trainable_parameters())))
        total_trainable = len(list(tree_flatten(self.trainable_parameters())))
        logger.debug(
            "_TrainableModel.__init__: probes=%s, base_trainable=%d, total_trainable=%d",
            self._probe_names_list,
            base_trainable,
            total_trainable,
        )

    def __call__(self, inputs: Any, mask: Any | None = None) -> tuple[Any, dict[str, Any]]:
        """Forward pass returning (lm_logits, probe_logits_dict).

        Runs the base model (triggering LayerCapture wrappers which set
        per-probe captured hidden states), then runs each probe
        on its captured hidden state.  Returns a dict mapping probe
        name to logits tensor.

        Args:
            inputs: Tokenized input batch of shape ``[B, T-1]``.
            mask: Optional boolean ``[B, T-1]`` mask of valid positions,
                threaded into ``probe.forward`` so ``granularity="response"``
                pooling ignores padding (mirrors ``Model.forward``).
        """
        for probe in self._probe_captures.values():
            probe.clear_captured()
        self._captured_hidden.clear()

        lm_logits = self.base(inputs)

        probe_logits: dict[str, Any] = {}
        for name in self._probe_names_list:
            probe = self._probe_captures.get(name)
            if probe is not None:
                captured = probe.get_captured_states()
                if captured:
                    # Track the hidden state for eval_metrics_fn directly here, so
                    # it survives `restore_capture_fns()` (called at the end of
                    # training). Otherwise a standalone `trainer.evaluate()` AFTER
                    # training silently dropped all probe metrics — the wrapped
                    # capture_fn that used to populate this was already torn down.
                    # Store ALL captured layer states (not just the last): a
                    # multi-layer probe's eval metrics must forward every layer, or
                    # aggregation="concat" crashes and "mean"/"max" score the wrong
                    # (last-only) input. run_probe forwards the full list.
                    self._captured_hidden[name] = captured
                    logits = probe.forward(captured, mask=mask, input_ids=inputs)
                    # Squeeze only a trailing single-logit dim (binary/regression);
                    # multi-class heads keep their [B, T, C] shape.
                    if logits.ndim > 2 and logits.shape[-1] == 1:
                        logits = logits.squeeze(-1)
                    probe_logits[name] = logits

        return lm_logits, probe_logits

    def get_probe(self, name: str) -> Any:
        """Get a probe module by name."""
        return getattr(self, f"probe_{name}")

    @property
    def _probe_names(self) -> list[str]:
        """List of probe names."""
        return self._probe_names_list

    def restore_capture_fns(self) -> None:
        """Restore original capture functions on all layer captures."""
        for lc, orig_fn in self._orig_capture_fns:
            lc.capture_fn = orig_fn
        self._orig_capture_fns.clear()


def make_joint_loss(
    lm_weight: float = 1.0,
    probe_weight: float = 1.0,
    probe_loss: str | Callable[..., Any] = "bce",
    probe_weights: dict[str, float] | None = None,
    probe_losses: dict[str, str | Callable[..., Any]] | None = None,
    class_weights: Any = None,
) -> LossFn:
    """Build a loss function with signature ``loss(model, batch, labels, lengths)``.

    The model must be a ``_TrainableModel`` that returns
    ``(lm_logits, probe_logits_dict)`` from its ``__call__``.

    Args:
        lm_weight: Weight for the LM cross-entropy term.
            Set to ``0`` for pure classifier mode (no next-token prediction).
        probe_weight: Default weight for probes not in ``probe_weights``.
        probe_loss: Default probe loss — ``"bce"``, ``"mse"``, or a callable.
        probe_weights: Per-probe weight overrides.
        probe_losses: Per-probe loss overrides.
        class_weights: Per-class weights for the built-in ``"ce"`` loss (see
            ``JointLoss``); ``None`` for unweighted.

    Returns:
        A loss function compatible with ``nn.value_and_grad``.
    """
    from auto_chasm.trainers.loss import _joint_loss_from_legacy

    # Phase 3b: JointLoss is now a single backend-agnostic ``__call__`` path; the old
    # ``_compute_mlx`` entry point is gone.  ``_joint_loss_from_legacy`` adapts these
    # pre-3b keyword arguments onto the new ``weights=``/``losses=`` API (Phase 3b-2
    # migrates this call site's remaining tests).
    loss_obj = _joint_loss_from_legacy(
        lm_weight=lm_weight,
        probe_weight=probe_weight,
        probe_loss=probe_loss,
        probe_weights=probe_weights,
        probe_losses=probe_losses,
        class_weights=class_weights,
    )

    def joint_loss(
        train_model: Any,
        batch: Any,
        labels: Any,
        lengths: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Compute the joint loss via the single backend-agnostic ``JointLoss``.

        Args:
            train_model: The wrapped ``_TrainableModel``.
            batch: Tokenized input batch of shape ``[B, T]``.
            labels: Probe label tensor aligned with ``batch``.
            lengths: Per-sequence token ranges.

        Returns:
            ``(total_loss, ntoks, components)``.
        """
        return loss_obj(train_model, batch, labels, lengths)

    return joint_loss


def clip_grad_norm(grad: Any, max_norm: float) -> Any:
    """Global L2 gradient clipping.

    Args:
        grad: Gradient tree.
        max_norm: Maximum gradient norm.

    Returns:
        Clipped gradient tree.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    norm_sq = sum(mx.sum(g * g) for _, g in tree_flatten(grad))
    norm = mx.sqrt(norm_sq)
    factor = mx.minimum(max_norm / (norm + 1e-6), 1.0)
    return tree_map(lambda g: g * factor, grad)


def default_binary_metrics(  # pragma: no cover — user-facing convenience
    train_model: Any,
    captured: dict[str, Any],
    targets: Any,
    mask: Any,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Pre-written binary classification metrics (backend-agnostic).

    Computes accuracy, precision, recall, and F1 per probe using the given
    threshold.  Pass it to ``Trainer`` / ``evaluate_joint_model`` as
    ``eval_metrics_fn`` if you want binary metrics.

    Works on **both** the MLX and PyTorch backends: tensors are converted to
    NumPy for the metric arithmetic, so neither framework's ops are required.
    MLX is imported lazily only when an MLX array is actually seen, so the
    torch backend never touches it (and this runs on a torch-only Linux box).

    Args:
        train_model: The trainable wrapper — MLX ``_TrainableModel`` (exposes
            ``get_probe``) or torch ``_TorchProbeWrapper`` (exposes ``_probes``).
        captured: Dict ``{probe_name: hidden_states}``.
        targets: Target labels ``[B, T]`` (or ``{probe_name: [B, T]}`` for
            per-probe labels).
        mask: Token mask of shape ``[B, T]``.
        threshold: Decision threshold (default 0.5).

    Returns:
        Dict with ``{probe_name}_accuracy``, ``_precision``, ``_recall``,
        ``_f1`` keys.
    """
    import numpy as np

    from auto_chasm.metrics import run_probe, to_numpy

    result: dict[str, float] = {}
    mask_np = to_numpy(mask).astype(np.int32)
    for probe_name, hidden in captured.items():
        logits = to_numpy(run_probe(train_model, probe_name, hidden))
        if logits.ndim > 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        preds = (1.0 / (1.0 + np.exp(-logits)) > threshold).astype(np.int32)

        tgt = targets[probe_name] if isinstance(targets, dict) else targets
        targets_int = to_numpy(tgt).astype(np.int32)
        # Exclude ignored (-100) labels even INSIDE the length window: an unlabeled
        # in-window token is not a prediction target, so counting it in the accuracy
        # denominator (as a guaranteed "wrong") silently deflated the score.
        eff = mask_np * (targets_int != -100).astype(np.int32)

        corr = float(((preds == targets_int) * eff).sum())
        tp = float((((preds == 1) & (targets_int == 1)) * eff).sum())
        fp = float((((preds == 1) & (targets_int == 0)) * eff).sum())
        fn = float((((preds == 0) & (targets_int == 1)) * eff).sum())
        masked = float(eff.sum())
        result[f"{probe_name}_accuracy"] = corr / max(masked, 1e-8)
        result[f"{probe_name}_precision"] = tp / max(tp + fp, 1e-8)
        result[f"{probe_name}_recall"] = tp / max(tp + fn, 1e-8)
        pc = result[f"{probe_name}_precision"]
        rc = result[f"{probe_name}_recall"]
        result[f"{probe_name}_f1"] = 2 * pc * rc / max(pc + rc, 1e-8)
    return result


def evaluate_joint_model(  # pragma: no cover — tested via integration
    train_model: Any,
    dataset: Any,
    batch_size: int,
    max_seq_length: int,
    loss_fn: LossFn,
    num_batches: int = -1,
    eval_metrics_fn: Callable[..., dict[str, float]] | None = None,
) -> dict[str, float]:
    """Evaluate the model on a dataset.

    Args:
        train_model: The ``_TrainableModel``.
        dataset: The evaluation dataset.
        batch_size: Batch size.
        max_seq_length: Maximum sequence length.
        loss_fn: Loss function.
        num_batches: Max batches to evaluate (-1 = all).
        eval_metrics_fn: Optional callable ``(train_model, logits_dict,
            targets, mask) -> dict[str, float]`` that returns custom
            metrics.  If ``None``, only loss and perplexity are reported.

    Returns:
        Dict of evaluation metrics.
    """
    import mlx.core as mx

    from auto_chasm.trainers.data_utils import iterate_batches, labels_to_mlx

    train_model.eval()

    total_loss = mx.array(0.0)
    total_components: dict[str, Any] = {}
    total_ntoks = mx.array(0.0)
    metric_accum: dict[str, float] = {}
    # PER-KEY weight sums: a metric fn may legitimately omit a key for some
    # batches (e.g. AUROC is undefined for a single-class batch). Each key must
    # be averaged over the weight of the batches where it was PRESENT — dividing
    # by the total weight (the old single-scalar denominator) silently deflated
    # any sometimes-missing metric by the weight fraction of the batches that
    # omitted it.
    metric_weights: dict[str, float] = {}

    if len(dataset) == 0:
        return {"loss": float("inf"), "perplexity": float("inf"), "ntokens": 0}
    for i, batch in enumerate(iterate_batches(dataset, batch_size, max_seq_length, loop=False)):
        if num_batches >= 0 and i >= num_batches:
            break

        tokens, batch_labels, lengths = batch
        tokens = mx.array(tokens)
        # Per-probe (multi-head) targets arrive as a {probe_name: array} dict;
        # mx.array(dict) would raise. labels_to_mlx handles array AND dict, mirroring
        # the training path so eval supports the same datasets training does.
        batch_labels = labels_to_mlx(batch_labels)
        lengths = mx.array(lengths)
        lvalue, ntoks, components = loss_fn(train_model, tokens, batch_labels, lengths)

        total_loss = total_loss + lvalue * ntoks
        total_ntoks = total_ntoks + ntoks

        for key, val in components.items():
            if key not in total_components:
                total_components[key] = mx.array(0.0)
            total_components[key] = total_components[key] + val * ntoks

        # User-provided metrics. Per-probe ({name: array}) dict labels are
        # supported (mirrors the torch evaluate path): each head's target is
        # shifted independently and the metric fn selects its own — previously
        # this was silently skipped on MLX, so per-layer metrics were unreachable
        # for multi-head dict-label datasets.
        if eval_metrics_fn is not None and train_model._captured_hidden:
            n_time = tokens.shape[1] - 1
            steps = mx.arange(1, n_time + 1)
            m = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
            b_targets: Any
            if isinstance(batch_labels, dict):
                b_targets = {k: v[:, 1:].astype(mx.float32) for k, v in batch_labels.items()}
            else:
                b_targets = batch_labels[:, 1:].astype(mx.float32)
            batch_metrics = eval_metrics_fn(train_model, train_model._captured_hidden, b_targets, m)
            # Token-weight by the batch's valid-position count so metrics are the
            # corpus value, not a per-batch mean (which over-weights a short final
            # batch). Exact for accuracy/precision/recall/mse/mae; macro-F1 becomes a
            # token-weighted mean (an approximation of corpus F1 across eval batches).
            w = float(m.sum())
            for k, v in batch_metrics.items():
                metric_accum[k] = metric_accum.get(k, 0.0) + float(v) * w
                metric_weights[k] = metric_weights.get(k, 0.0) + w

        eval_args = [total_loss, total_ntoks]
        eval_args.extend(total_components.values())
        mx.eval(*eval_args)

    # Divide each key by ITS OWN summed weight (token-weighted mean over the
    # batches where the key was present — see the metric_weights note above).
    for k in metric_accum:
        if metric_weights.get(k, 0.0) > 0:
            metric_accum[k] /= metric_weights[k]

    ntoks_val = float(total_ntoks)
    avg_loss = (total_loss / total_ntoks).item() if ntoks_val > 0 else float("inf")

    result: dict[str, float] = {
        "loss": avg_loss,
        "ntokens": int(ntoks_val),
        **metric_accum,
    }

    for key, acc in total_components.items():
        result[key] = (acc / total_ntoks).item() if ntoks_val > 0 else 0.0

    # JointLoss keys the language-model cross-entropy component "lm_head" (pure-probe
    # runs with lm weight 0 skip it, so perplexity is absent there — as intended).
    if LM_HEAD in result:
        lm_ce = result[LM_HEAD]
        result["perplexity"] = math.exp(lm_ce) if lm_ce < 100 else float("inf")

    return result

"""Regression: eval metrics on a pooled (response) probe must not crash.

A response/sentence-granularity head returns one prediction per sequence ([B]),
but the trainer hands the metric per-token targets/mask ([B,T]); the metric used
to index the 1-D preds with the 2-D mask and raise IndexError — so evaluating the
flagship whole-text (response) probe crashed after training. The metric now
collapses the per-token targets to per-sequence for pooled heads.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from auto_chasm import JointLoss, Model, ProbeConfig
from auto_chasm.metrics import (
    _pool_targets_to_preds,
    classification_metrics,
    regression_metrics,
)
from auto_chasm.trainers.trainable import _TrainableModel, evaluate_joint_model


def test_pool_targets_to_preds_collapses_pooled() -> None:
    """Pooled preds [B] collapse per-token targets/mask to per-sequence (first valid)."""
    preds = np.array([1, 0])  # [B] pooled
    targets = np.array([[-100, 2, 2, 2], [1, 1, -100, -100]])  # [B, T]
    mask = np.array([[0, 1, 1, 1], [1, 1, 0, 0]], dtype=bool)
    tgt, msk = _pool_targets_to_preds(preds, targets, mask)
    np.testing.assert_array_equal(np.asarray(tgt), [2, 1])  # first VALID label per row
    np.testing.assert_array_equal(np.asarray(msk), [True, True])


def test_pool_targets_to_preds_leaves_token_granularity() -> None:
    """Token preds [B,T] are returned unchanged (shapes already align)."""
    preds = np.array([[1, 0], [0, 1]])
    targets = np.array([[1, 0], [0, 1]])
    mask = np.array([[1, 1], [1, 1]], dtype=bool)
    tgt, msk = _pool_targets_to_preds(preds, targets, mask)
    assert tgt is targets and msk is mask


class _TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = [nn.Linear(8, 8) for _ in range(2)]
        self.output_proj = nn.Linear(8, 16)

    def __call__(self, x: mx.array, **k: object) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = 8
    num_hidden_layers = 2


def _evaluate(granularity: str, out_features: int, loss: str, metric) -> dict:  # noqa: ANN001
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    kw: dict = {
        "name": "p",
        "layers": [1],
        "source": "hidden",
        "module_config": {"out_features": out_features},
    }
    if granularity == "response":
        kw["granularity"] = "response"
        kw["aggregation"] = "mean"
    m.attach_probe(ProbeConfig(**kw))
    tm = _TrainableModel(m.model, m._probes)
    data = [{"tokens": [1, 2, 3, 4, 5], "labels": [0, 2, 2, 2, 2]} for _ in range(4)]
    return evaluate_joint_model(
        tm,
        data,
        batch_size=2,
        max_seq_length=8,
        loss_fn=JointLoss(weights={"lm_head": 0.0, "p": 1.0}, losses={"p": loss}),
        eval_metrics_fn=metric,
    )


def test_response_classification_probe_evaluates() -> None:
    """A response-granularity classification probe + classification_metrics no longer crashes."""
    res = _evaluate("response", 3, "ce", classification_metrics(num_classes=3))
    for key in ("p_acc", "p_adj", "p_macro_f1"):
        assert key in res and 0.0 <= res[key] <= 1.0


def test_response_regression_probe_evaluates() -> None:
    """A response-granularity regression probe + regression_metrics no longer crashes."""
    res = _evaluate("response", 1, "mse", regression_metrics(num_classes=3))
    for key in ("p_mse", "p_mae", "p_acc", "p_adj"):
        assert key in res
    assert 0.0 <= res["p_acc"] <= 1.0


def test_token_classification_still_works() -> None:
    """Token granularity (the common case) is unaffected by the pooled-alignment fix."""
    res = _evaluate("token", 3, "ce", classification_metrics(num_classes=3))
    for key in ("p_acc", "p_adj", "p_macro_f1"):
        assert key in res and 0.0 <= res[key] <= 1.0

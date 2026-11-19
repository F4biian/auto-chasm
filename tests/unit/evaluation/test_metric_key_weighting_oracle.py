"""Oracle: sometimes-missing eval-metric keys are averaged per-key, not deflated.

An ``eval_metrics_fn`` may legitimately omit a key for some batches — the
canonical case is AUROC, which is undefined for a batch containing only one
class. The evaluators token-weight metrics across batches; the regression this
pins is that a key absent from some batches was still divided by the TOTAL
weight, silently scaling the metric down by the weight-fraction of the batches
that omitted it (an AUROC of ~0.5 over 40% mixed-class sequences reported as
~0.2). Correct behavior: each key is averaged over the weight of the batches
where it was PRESENT.

Both backends are pinned with the same construction: two single-sequence
batches with EQUAL valid-token counts; the metric fn emits ``always=1.0`` for
both and ``flaky=1.0`` only for the batch whose labels contain class 1. The
correct weighted mean of ``flaky`` is 1.0; the old bug reported 0.5.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from auto_chasm import JointLoss, Model, ProbeConfig
from auto_chasm.metrics import to_numpy
from auto_chasm.trainers.trainable import _TrainableModel, evaluate_joint_model


class _TinyMlp(nn.Module):
    def __init__(self, h: int = 16, v: int = 32, layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(v, h)
        self.layers = [nn.Linear(h, h) for _ in range(layers)]
        self.output_proj = nn.Linear(h, v)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = 16
    num_hidden_layers = 2


def _flaky_metric_fn(train_model: Any, captured: dict[str, Any], targets: Any, mask: Any):
    """Emit ``always`` for every batch, ``flaky`` only when class 1 is present."""
    tgt = targets["p"] if isinstance(targets, dict) else targets
    tgt_np = to_numpy(tgt)
    out = {"always": 1.0}
    if (tgt_np == 1).any():
        out["flaky"] = 1.0
    return out


# Two sequences with the SAME token count (=> equal weights): one carries a
# class-1 label, one does not.
_DATA = [
    {"tokens": [1, 2, 3, 4, 5], "labels": [0, 1, 0, 0, 0]},  # mixed -> flaky present
    {"tokens": [6, 7, 8, 9, 10], "labels": [0, 0, 0, 0, 0]},  # single-class -> flaky absent
]


def test_mlx_sometimes_missing_key_not_deflated() -> None:
    """MLX evaluate_joint_model: flaky == 1.0 (per-key weight), not 0.5 (total weight)."""
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(
        ProbeConfig(
            name="p",
            layers=[0],
            aggregation="last",
            granularity="token",
            module_config={"out_features": 1},
        )
    )
    tm = _TrainableModel(m.model, m._probes)
    metrics = evaluate_joint_model(
        train_model=tm,
        dataset=_DATA,
        batch_size=1,
        max_seq_length=8,
        loss_fn=JointLoss(weights={"lm_head": 0.0}),
        eval_metrics_fn=_flaky_metric_fn,
    )
    assert metrics["always"] == 1.0
    assert metrics["flaky"] == 1.0, (
        f"flaky={metrics['flaky']}: a key omitted for single-class batches was "
        "divided by the TOTAL weight instead of its own per-key weight."
    )


def test_torch_sometimes_missing_key_not_deflated() -> None:
    """Torch evaluate_torch_model: same oracle as the MLX path."""
    import pytest

    torch = pytest.importorskip("torch")
    import torch.nn as tnn

    from auto_chasm.trainers._metrics import evaluate_torch_model

    class _TinyTorch(tnn.Module):
        def __init__(self, h: int = 16, v: int = 32) -> None:
            super().__init__()
            self.embedding = tnn.Embedding(v, h)
            self.layers = tnn.ModuleList([tnn.Linear(h, h) for _ in range(2)])
            self.output_proj = tnn.Linear(h, v)
            self.config = _Cfg()

        def forward(self, x: Any) -> Any:
            h = self.embedding(x)
            for layer in self.layers:
                h = torch.nn.functional.gelu(layer(h))
            return self.output_proj(h)

    m = Model(_TinyTorch(), None, "torch")
    m.attach_probe(
        ProbeConfig(
            name="p",
            layers=[0],
            aggregation="last",
            granularity="token",
            module_config={"out_features": 1},
        )
    )
    metrics = evaluate_torch_model(
        model_wrapper=m,
        dataset=_DATA,
        batch_size=1,
        max_seq_length=8,
        loss_fn=JointLoss(weights={"lm_head": 0.0}),
        eval_metrics_fn=_flaky_metric_fn,
    )
    assert metrics["always"] == 1.0
    assert metrics["flaky"] == 1.0, (
        f"flaky={metrics['flaky']}: a key omitted for single-class batches was "
        "divided by the TOTAL weight instead of its own per-key weight."
    )

"""Regression tests over the DX-refactor surface.

Each test asserts the CORRECT behavior and failed before its fix:
1. run_probe must apply the probe's pooling on BOTH backends (response/sentence
   heads) — MLX used to return the bare unpooled module output.
2. sweep._layer_loss must return a single-probe's own CE component, not the
   total joint loss (which differs once lm_weight > 0).
3. class_weights of the wrong length must raise on BOTH backends (MLX used to
   silently zero out-of-range classes; torch raised IndexError).
4. The MLX and torch eval paths must hand eval_metrics_fn the SAME captured
   layer for a multi-layer probe.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm import JointLoss, Model, ProbeConfig
from auto_chasm.metrics import run_probe
from auto_chasm.trainers._loss_ce import weighted_ce
from auto_chasm.trainers.trainable import _TrainableModel


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


# --- Bug 1: run_probe must pool on both backends ---------------------------------


def test_run_probe_pools_response_head_mlx() -> None:
    """run_probe on a response probe returns the POOLED [B, C], not [B, T, C]."""
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(
        ProbeConfig(
            name="r",
            layers=[0],
            aggregation="last",
            granularity="response",
            module_config={"out_features": 3},
        )
    )
    tm = _TrainableModel(m.model, m._probes)
    tm(mx.array([[1, 2, 3, 4, 5]]))  # populate _captured_hidden
    out = run_probe(tm, "r", tm._captured_hidden["r"])
    assert out.ndim == 2  # pooled to one prediction per sequence
    assert tuple(out.shape) == (1, 3)


# --- Bug 2: sweep._layer_loss must report the probe's own component --------------


def test_layer_loss_prefers_single_probe_component_over_total() -> None:
    """A one-probe component is keyed bare ('probe_ce'); _layer_loss must use it."""
    from auto_chasm.sweep import _layer_loss

    metrics = {"loss": 2.0, "lm_ce": 1.5, "probe_ce": 0.5, "L0_acc": 0.3}
    assert _layer_loss(metrics, "L0") == 0.5  # not the total 2.0


def test_layer_loss_multi_probe_no_suffix_collision() -> None:
    """':L1' must not match ':L10'/':L11' (the colon prevents prefix collisions)."""
    from auto_chasm.sweep import _layer_loss

    metrics = {"loss": 9.0, "probe_ce:L1": 0.1, "probe_ce:L10": 0.9, "probe_ce:L11": 0.7}
    assert _layer_loss(metrics, "L1") == 0.1
    assert _layer_loss(metrics, "L10") == 0.9
    assert _layer_loss(metrics, "L11") == 0.7


# --- Bug 3: class_weights length must be validated identically -------------------


def _model(out_features: int) -> Model:
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(
        ProbeConfig(
            name="p", layers=[0], aggregation="last", module_config={"out_features": out_features}
        )
    )
    return m


def test_class_weights_wrong_length_raises_mlx() -> None:
    """A weight vector shorter than num_classes raises (no silent OOB-zeroing)."""
    m = _model(3)
    tm = _TrainableModel(m.model, m._probes)
    loss = JointLoss(
        weights={"lm_head": 0.0}, losses={"p": "ce"}, class_weights=[1.0, 2.0]
    )  # len 2 for 3 classes
    with pytest.raises(ValueError, match="class_weights"):
        loss(tm, mx.array([[1, 2, 3, 4, 5]]), mx.array([[0, 1, 2, 1, 0]]), mx.array([[0, 4]]))


def test_weighted_ce_torch_wrong_length_raises() -> None:
    """The torch weighted-CE raises a clear ValueError (not IndexError) on bad length."""
    torch = pytest.importorskip("torch")

    logits = torch.zeros((1, 2, 3))
    shifted = torch.tensor([[0, 1]])
    lv = shifted != -100
    with pytest.raises(ValueError, match="class_weights"):
        weighted_ce(logits, shifted, lv, lv, [1.0, 2.0])  # len 2 for 3 classes


def test_weighted_ce_mlx_wrong_length_raises() -> None:
    """The MLX weighted-CE raises on a wrong-length weight vector (parity with torch)."""
    logits = mx.zeros((1, 2, 3))
    shifted = mx.array([[0, 1]])
    lv = shifted != -100
    with pytest.raises(ValueError, match="class_weights"):
        weighted_ce(logits, shifted, lv, lv, [1.0, 2.0])


# --- Bug 4: eval picks the SAME captured layer on both backends ------------------


def test_eval_uses_last_captured_layer_torch() -> None:
    """For a 2-layer probe, the torch eval hands the metric ALL captured layers.

    Matches the MLX path (`_captured_hidden[name]` stores the full list and
    ``run_probe`` forwards every layer); torch previously passed only ``states[-1]``,
    which crashed a concat head and silently dropped layers under mean/max.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as tnn

    from auto_chasm.trainers._metrics import evaluate_torch_model
    from auto_chasm.trainers.wrappers import _TorchProbeWrapper

    class _TorchTiny(tnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = tnn.Embedding(32, 16)
            self.layers = tnn.ModuleList([tnn.Linear(16, 16) for _ in range(2)])
            self.output_proj = tnn.Linear(16, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.embedding(x)
            for layer in self.layers:
                h = torch.nn.functional.gelu(layer(h))
            return self.output_proj(h)

    raw = _TorchTiny()
    raw.config = _Cfg()
    m = Model(raw, None, "torch")
    # mean aggregation keeps the head input at hidden width, so it runs on a single layer.
    m.attach_probe(
        ProbeConfig(name="p", layers=[0, 1], aggregation="mean", module_config={"out_features": 3})
    )
    data = [{"tokens": [1, 2, 3, 4, 5], "labels": [0, 1, 2, 1, 0]}]
    # max_seq_length == token count => no padding, so the reference forward below
    # sees the exact same inputs[:, :-1] that evaluate_torch_model uses internally.
    max_len = 5

    # Reference: the probe's two captured layers from one deterministic forward.
    raw.eval()
    wrapper = _TorchProbeWrapper(raw, m._probes)
    wrapper(torch.tensor([data[0]["tokens"]])[:, :-1])
    states = m._probes["p"].get_captured_states()
    first_layer = states[0].detach().cpu().numpy()
    last_layer = states[-1].detach().cpu().numpy()
    assert not np.allclose(first_layer, last_layer)  # the two layers genuinely differ

    seen: dict = {}

    def recorder(model, captured, targets, mask):  # noqa: ANN001, ANN202
        # captured["p"] is the FULL list of per-layer states (both backends).
        seen["hidden"] = [h.detach().cpu().numpy() for h in captured["p"]]
        return {"x": 0.0}

    evaluate_torch_model(
        m, data, 1, max_len, JointLoss(weights={"lm_head": 0.0}, losses={"p": "ce"}), recorder
    )
    assert len(seen["hidden"]) == 2  # both captured layers are handed to the metric
    np.testing.assert_allclose(seen["hidden"][0], first_layer, atol=1e-5)
    np.testing.assert_allclose(seen["hidden"][-1], last_layer, atol=1e-5)

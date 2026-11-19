"""Oracle + integration tests for the ordinal-regression probe path.

Covers the pieces that make ``out_features=1`` regression work end to end:

- the loss fix: token-level ``out_features=1`` ``mse``/``mae`` align ``[B, T, 1]``
  predictions with ``[B, T]`` targets instead of broadcasting (MLX raised, torch
  silently expanded to ``[B, T, T]``);
- ``regression_metrics``: reports ``mse``/``mae`` AND the discretized
  ``acc``/``adj`` so a regressor is comparable to a classifier;
- ``discretize`` round-and-clip;
- a real ``evaluate`` and a ``LayerSweep`` driven by ``regression_metrics``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm import JointLoss, LayerSweep, Model, ModuleSpec, ProbeConfig, regression_metrics
from auto_chasm.metrics import discretize

# --------------------------------------------------------------------------- #
# Loss: token-level out_features=1 mse/mae no longer broadcast.                #
# --------------------------------------------------------------------------- #


class _FixedMlx:
    """MLX model returning a fixed ``(lm_logits, {"p": [B, T, 1]})``."""

    def __init__(self, lm: np.ndarray, pr: np.ndarray) -> None:
        self.lm = mx.array(lm)
        self.pr = {"p": mx.array(pr)}

    def __call__(self, inputs: object, mask: object = None) -> tuple:
        return self.lm, self.pr


class _FixedTorch:
    """Torch counterpart of :class:`_FixedMlx`."""

    def __init__(self, lm: np.ndarray, pr: np.ndarray) -> None:
        import torch

        self.lm = torch.tensor(lm)
        self.pr = {"p": torch.tensor(pr)}

    def __call__(self, inputs: object, mask: object = None) -> tuple:
        return self.lm, self.pr


def _run_scalar_loss(backend: str, probe_loss: str) -> float:
    """Run JointLoss on a fixed [B,T,1] scalar head; return the probe component.

    One labeled (EOS-style) position: pred 0.5 vs target 2 => squared err 2.25,
    abs err 1.5. A broadcasting bug would change the value (or raise).
    """
    lm = np.zeros((1, 3, 5), dtype=np.float32)
    pr = np.array([[[1.0], [1.0], [0.5]]], dtype=np.float32)  # [1, 3, 1]
    labels_np = np.array([[-100, -100, -100, 2]], dtype=np.int64)  # shifted -> [-100,-100,2]
    batch_np = np.array([[1, 2, 3, 4]], dtype=np.int64)
    lengths_np = np.array([[0, 4]], dtype=np.int64)

    loss = JointLoss(weights={"lm_head": 0.0}, losses={"p": probe_loss})
    if backend == "mlx":
        model = _FixedMlx(lm, pr)
        total, _, comp = loss(model, mx.array(batch_np), mx.array(labels_np), mx.array(lengths_np))
    else:
        import torch

        model = _FixedTorch(lm, pr)
        total, _, comp = loss(
            model,
            torch.tensor(batch_np),
            torch.tensor(labels_np),
            torch.tensor(lengths_np),
        )
    return float(comp["p"])


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_token_scalar_mse_matches_hand_value(backend: str) -> None:
    """out=1 token MSE at the single labeled position == (0.5 - 2)**2 == 2.25."""
    if backend == "torch":
        pytest.importorskip("torch")
    assert _run_scalar_loss(backend, "mse") == pytest.approx(2.25, rel=1e-5)


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_token_scalar_mae_matches_hand_value(backend: str) -> None:
    """out=1 token MAE at the single labeled position == abs(0.5 - 2) == 1.5."""
    if backend == "torch":
        pytest.importorskip("torch")
    assert _run_scalar_loss(backend, "mae") == pytest.approx(1.5, rel=1e-5)


def test_mlx_torch_scalar_mse_parity() -> None:
    """The scalar-head MSE value agrees across backends."""
    pytest.importorskip("torch")
    assert _run_scalar_loss("mlx", "mse") == pytest.approx(_run_scalar_loss("torch", "mse"))


# --------------------------------------------------------------------------- #
# discretize + regression_metrics.                                            #
# --------------------------------------------------------------------------- #


def test_discretize_rounds_and_clips() -> None:
    """Round to nearest class, clamp into [0, num_classes-1]."""
    out = discretize(np.array([-0.4, 0.6, 2.5, 5.9, 100.0]), num_classes=6)
    # rint: -0.4->0 (clip), 0.6->1, 2.5->2 (banker's), 5.9->6->5 (clip), 100->5 (clip)
    np.testing.assert_array_equal(out, [0.0, 1.0, 2.0, 5.0, 5.0])


class _FakeMLX:
    """Stands in for the MLX trainable wrapper (exposes ``get_probe``)."""

    def __init__(self, logits: dict[str, np.ndarray]) -> None:
        self._logits = logits

    def get_probe(self, name: str):  # noqa: ANN201
        logits = self._logits[name]
        return lambda _hidden: logits


class _FakeTorchProbe:
    def __init__(self, logits: np.ndarray) -> None:
        self._logits = logits

    def forward(self, _hiddens):  # noqa: ANN201
        return self._logits


class _FakeTorch:
    """Stands in for the torch wrapper (exposes ``_probes``)."""

    def __init__(self, logits: dict[str, np.ndarray]) -> None:
        self._probes = {n: _FakeTorchProbe(v) for n, v in logits.items()}


@pytest.mark.parametrize("fake_cls", [_FakeMLX, _FakeTorch])
def test_regression_metrics_values_and_dispatch(fake_cls: type) -> None:
    """[B,T,1] continuous head -> mse/mae plus discretized acc/adj, both wrappers."""
    logits = {"p": np.array([[[2.4], [0.4], [5.9]]])}  # [1, 3, 1]
    fn = regression_metrics(num_classes=6, ordinal_tol=1)
    targets = np.array([[2, 0, 5]])
    mask = np.array([[1, 1, 1]])
    out = fn(fake_cls(logits), {"p": None}, targets, mask)
    assert set(out) == {"p_mse", "p_mae", "p_acc", "p_adj"}
    # mse = ((2.4-2)^2 + (0.4-0)^2 + (5.9-5)^2)/3 = (0.16+0.16+0.81)/3
    assert out["p_mse"] == pytest.approx((0.16 + 0.16 + 0.81) / 3, rel=1e-5)
    assert out["p_mae"] == pytest.approx((0.4 + 0.4 + 0.9) / 3, rel=1e-5)
    # discretized preds [2,0,6->5] == targets [2,0,5] -> perfect acc and adj.
    assert out["p_acc"] == pytest.approx(1.0)
    assert out["p_adj"] == pytest.approx(1.0)


def test_regression_metrics_dict_targets() -> None:
    """A per-probe ``{name: array}`` targets dict selects each head's own target."""
    logits = {"p": np.array([[[1.1], [3.0]]])}
    fn = regression_metrics(num_classes=6)
    out = fn(_FakeMLX(logits), {"p": None}, {"p": np.array([[1, 3]])}, np.array([[1, 1]]))
    assert out["p_acc"] == pytest.approx(1.0)
    assert out["p_mae"] == pytest.approx((0.1 + 0.0) / 2, rel=1e-5)


# --------------------------------------------------------------------------- #
# End-to-end integration (real eval + sweep).                                 #
# --------------------------------------------------------------------------- #


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
    def __init__(self, layers: int = 2) -> None:
        self.hidden_size = 16
        self.num_hidden_layers = layers


def test_regression_end_to_end_evaluate() -> None:
    """A real ``evaluate`` of an out=1 MSE probe yields finite mse/mae/acc/adj."""
    from auto_chasm.trainers.trainable import _TrainableModel, evaluate_joint_model

    m = Model(_TinyMlp(layers=2), None, "mlx")
    m.model.config = _Cfg(layers=2)
    m.add_probes([ProbeConfig(name="cefr", layers=[0], module_config={"out_features": 1})])
    data = [
        {"tokens": [1, 2, 3, 4, 5], "labels": [-100, -100, -100, -100, 3]},
        {"tokens": [6, 7, 8, 9, 10], "labels": [-100, -100, -100, -100, 1]},
    ]
    tm = _TrainableModel(m.model, m._probes)
    result = evaluate_joint_model(
        train_model=tm,
        dataset=data,
        batch_size=2,
        max_seq_length=16,
        loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"cefr": "mse"}),
        eval_metrics_fn=regression_metrics(num_classes=6, ordinal_tol=1),
    )
    for key in ("cefr_mse", "cefr_mae", "cefr_acc", "cefr_adj"):
        assert key in result, f"missing {key}"
        assert math.isfinite(result[key])


def test_layer_sweep_regression_end_to_end_mlx(tmp_path) -> None:  # noqa: ANN001
    """A regression sweep runs with regression_metrics and fills acc/adj columns."""

    def _data(n: int) -> list[dict]:
        return [
            {
                "tokens": [(i + k) % 30 + 1 for k in range(5)],
                "labels": [-100, -100, -100, -100, i % 6],
            }
            for i in range(n)
        ]

    m = Model(_TinyMlp(layers=2), None, "mlx")
    m.model.config = _Cfg(layers=2)
    sweep = LayerSweep(
        m,
        out_features=1,
        num_classes=6,
        module_spec=ModuleSpec.linear(out_features=1),
        layers=[0, 1],
        ordinal_tol=1,
        eval_metrics_fn=regression_metrics(num_classes=6, ordinal_tol=1),
    )
    result = sweep.run(
        _data(6),
        _data(2),
        _data(2),
        # The sweep attaches one head per layer, named ``L<i>``; mse for each.
        loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"L0": "mse", "L1": "mse"}),
        num_iters=20,
        eval_every=10,
        batch_size=2,
        max_seq_length=16,
        learning_rate=1e-2,
        verbose=False,
        output_dir=str(tmp_path),
    )
    assert set(result.best) == {0, 1}
    for i in (0, 1):
        row = result.best[i]
        for key in ("val_acc", "val_adj", "test_acc", "test_adj"):
            assert key in row

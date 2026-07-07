"""Oracle + parity tests for the public ``metrics`` module.

Covers the helpers that retire the hand-written ``_np``/``_head``/metric glue:
``to_numpy``, ``run_probe``, ``accuracy``, ``ordinal_accuracy``, ``macro_f1``,
and the ``classification_metrics`` factory.  Also pins the MLX ``evaluate``
dict-label fix (per-probe metrics were previously skipped on MLX).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from auto_chasm import classification_metrics
from auto_chasm.metrics import accuracy, macro_f1, ordinal_accuracy, to_numpy


def test_accuracy_and_ordinal_oracle() -> None:
    """Hand-built arrays: exact accuracy 1/3, adjacent (±1) accuracy 2/3."""
    preds = np.array([[0, 2, 5]])
    targets = np.array([[1, 2, 3]])
    mask = np.array([[1, 1, 1]])
    assert accuracy(preds, targets, mask) == pytest.approx(1 / 3)
    # |0-1|=1 ok, |2-2|=0 ok, |5-3|=2 not ok  -> 2/3
    assert ordinal_accuracy(preds, targets, mask, tol=1) == pytest.approx(2 / 3)


def test_accuracy_excludes_ignore_and_mask() -> None:
    """``-100`` targets and masked-out positions are dropped before the mean."""
    preds = np.array([[1, 0, 3]])
    targets = np.array([[1, -100, 3]])  # middle ignored
    assert accuracy(preds, targets, np.array([[1, 1, 1]])) == pytest.approx(1.0)
    # Same but middle masked out instead of -100.
    preds2 = np.array([[1, 9, 3]])
    targets2 = np.array([[1, 2, 3]])
    assert accuracy(preds2, targets2, np.array([[1, 0, 1]])) == pytest.approx(1.0)


def test_empty_keep_is_zero_not_nan() -> None:
    """No kept position returns a finite 0.0, never NaN."""
    preds = np.array([[0, 1]])
    targets = np.array([[0, 1]])
    mask = np.array([[0, 0]])
    assert accuracy(preds, targets, mask) == 0.0
    assert ordinal_accuracy(preds, targets, mask) == 0.0
    assert macro_f1(preds, targets, mask, num_classes=2) == 0.0


def test_macro_f1_oracle() -> None:
    """Two-class confusion (each class P=R=0.5) → macro-F1 0.5."""
    preds = np.array([[0, 0, 1, 1]])
    targets = np.array([[0, 1, 0, 1]])
    mask = np.array([[1, 1, 1, 1]])
    assert macro_f1(preds, targets, mask, num_classes=2) == pytest.approx(0.5)


def test_to_numpy_mlx_bf16() -> None:
    """An MLX bf16 array converts to the correct float32 NumPy array."""
    mx = pytest.importorskip("mlx.core")
    arr = mx.array([1.0, 2.0, 3.0]).astype(mx.bfloat16)
    out = to_numpy(arr)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_to_numpy_torch_bf16() -> None:
    """A torch bf16 tensor converts to the correct float32 NumPy array."""
    torch = pytest.importorskip("torch")
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    out = to_numpy(t)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


class _FakeMLX:
    """Stands in for the MLX ``_TrainableModel`` (exposes ``get_probe``)."""

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
    """Stands in for the torch ``_TorchProbeWrapper`` (exposes ``_probes``)."""

    def __init__(self, logits: dict[str, np.ndarray]) -> None:
        self._probes = {n: _FakeTorchProbe(v) for n, v in logits.items()}


@pytest.mark.parametrize("fake_cls", [_FakeMLX, _FakeTorch])
def test_classification_metrics_values_and_dispatch(fake_cls: type) -> None:
    """Both wrapper dispatch paths produce identical, correct metric values."""
    # pos0 -> class 0, pos1 -> class 2; targets match exactly.
    logits = {"p": np.array([[[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])}
    fn = classification_metrics(num_classes=3, ordinal_tol=1)
    captured = {"p": None}
    targets = np.array([[0, 2]])
    mask = np.array([[1, 1]])
    out = fn(fake_cls(logits), captured, targets, mask)
    assert set(out) == {"p_acc", "p_adj", "p_macro_f1"}
    assert out["p_acc"] == pytest.approx(1.0)
    assert out["p_adj"] == pytest.approx(1.0)
    # class 0 and class 2 perfect (F1=1), class 1 absent (F1=0) -> mean 2/3.
    assert out["p_macro_f1"] == pytest.approx(2 / 3)


def test_classification_metrics_dict_targets() -> None:
    """A per-probe ``{name: array}`` targets dict selects each head's own target."""
    logits = {"p": np.array([[[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])}
    fn = classification_metrics(num_classes=3)
    out = fn(_FakeMLX(logits), {"p": None}, {"p": np.array([[0, 2]])}, np.array([[1, 1]]))
    assert out["p_acc"] == pytest.approx(1.0)


@pytest.mark.parametrize("fake_cls", [_FakeMLX, _FakeTorch])
def test_classification_metrics_binary_head_uses_sigmoid_not_argmax(fake_cls: type) -> None:
    """A single-logit (binary) head is scored by sigmoid-threshold, not argmax.

    A binary probe emits one logit per position (``[B, T, 1]``); ``argmax(-1)`` over
    that size-1 axis is ALWAYS class 0, collapsing every prediction to the negative
    class and scoring the head at the base rate. The oracle alternates the logit's
    sign so the correct predictions ``[1, 0, 1, 0]`` equal the targets (accuracy
    1.0, macro-F1 1.0); the argmax bug would instead predict all-zeros (accuracy
    0.5 = base rate).
    """
    # [B=1, T=4, out_features=1]: +logit -> class 1, -logit -> class 0.
    logits = {"h": np.array([[[3.0], [-3.0], [3.0], [-3.0]]])}
    fn = classification_metrics()  # num_classes inferred; a binary head is 2 classes
    targets = np.array([[1, 0, 1, 0]])
    mask = np.array([[1, 1, 1, 1]])
    out = fn(fake_cls(logits), {"h": None}, targets, mask)
    assert out["h_acc"] == pytest.approx(1.0)
    assert out["h_macro_f1"] == pytest.approx(1.0)


def test_classification_metrics_binary_scores_two_classes_even_if_num_classes_1() -> None:
    """A binary head must score macro-F1 over BOTH classes even when ``num_classes=1``.

    ``LayerSweep`` passes ``num_classes=out_features``, which is ``1`` for the
    default single-logit head — so ``classification_metrics(1)`` must still treat a
    single-logit head as 2-class, otherwise macro-F1 silently drops the positive
    class. Oracle: predictions ``[1, 1, 1, 0]`` (one false positive) vs targets
    ``[1, 0, 1, 0]`` → class-0 F1 = 2/3, class-1 F1 = 0.8, so macro-F1 = (2/3 +
    0.8)/2 ≈ 0.7333 (NOT 2/3 = 0.6667, which is what averaging over class 0 alone
    would give).
    """
    logits = {"h": np.array([[[3.0], [3.0], [3.0], [-3.0]]])}  # preds [1, 1, 1, 0]
    fn = classification_metrics(num_classes=1)  # the LayerSweep binary default
    targets = np.array([[1, 0, 1, 0]])
    mask = np.array([[1, 1, 1, 1]])
    out = fn(_FakeMLX(logits), {"h": None}, targets, mask)
    assert out["h_acc"] == pytest.approx(0.75)  # 3/4 correct
    assert out["h_macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_evaluate_joint_model_dict_labels_metrics() -> None:
    """MLX ``evaluate_joint_model`` now computes per-probe metrics for dict labels.

    Regression for the fix: before, dict labels were silently skipped, so a
    multi-head sweep got no per-layer metrics on MLX.
    """
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from auto_chasm import JointLoss, Model, ProbeConfig
    from auto_chasm.trainers.trainable import _TrainableModel, evaluate_joint_model

    class _TinyMlp(nn.Module):
        def __init__(self, h: int = 16, v: int = 32, layers: int = 3) -> None:
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
        num_hidden_layers = 3

    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.add_probes(
        [
            ProbeConfig(name="a", layers=[0], module_config={"out_features": 3}),
            ProbeConfig(name="b", layers=[1], module_config={"out_features": 3}),
        ]
    )
    data = [
        {"tokens": [1, 2, 3, 4, 5], "labels": {"a": [0, 1, 2, 1, 0], "b": [2, 2, 1, 0, 0]}},
        {"tokens": [6, 7, 8, 9, 10], "labels": {"a": [1, 1, 0, 2, 2], "b": [0, 1, 1, 2, 0]}},
    ]
    tm = _TrainableModel(m.model, m._probes)
    result = evaluate_joint_model(
        train_model=tm,
        dataset=data,
        batch_size=2,
        max_seq_length=16,
        loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"a": "ce", "b": "ce"}),
        eval_metrics_fn=classification_metrics(num_classes=3),
    )
    # Both heads' metrics are present — previously the dict-label branch was skipped.
    for key in ("a_acc", "a_adj", "a_macro_f1", "b_acc", "b_adj", "b_macro_f1"):
        assert key in result, f"missing {key}"

    # A joint eval (lm weight > 0) reports perplexity from the "lm_head" component —
    # regression for the Phase-3b key rename that silently dropped it (was keyed
    # "lm_ce"). Pure-probe (weight 0) omits the lm term, so perplexity is absent.
    joint = evaluate_joint_model(
        train_model=tm,
        dataset=data,
        batch_size=2,
        max_seq_length=16,
        loss_fn=JointLoss(weights={"lm_head": 1.0}, losses={"a": "ce", "b": "ce"}),
        eval_metrics_fn=classification_metrics(num_classes=3),
    )
    assert "lm_head" in joint and "perplexity" in joint
    assert joint["perplexity"] == pytest.approx(math.exp(joint["lm_head"]), rel=1e-5)
    assert "perplexity" not in result  # pure-probe run has no lm term


def test_multilayer_concat_probe_eval_metrics_use_all_layers() -> None:
    """A multi-layer concat probe's eval metrics forward ALL layers (was last-only).

    Regression: eval stored only the last captured layer, so aggregation="concat"
    crashed at eval (dim mismatch) and "mean"/"max" scored the wrong input.
    """
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from auto_chasm import JointLoss, Model, ProbeConfig
    from auto_chasm.trainers.trainable import _TrainableModel, evaluate_joint_model

    class _TinyMlp(nn.Module):
        def __init__(self, h: int = 16, v: int = 32, layers: int = 3) -> None:
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
        num_hidden_layers = 3

    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    # A concat probe over TWO layers: in-dim = 2*hidden. Its eval forward must feed
    # both captured layers, not just the last (which would be dim 16, not 32).
    m.add_probes(
        [
            ProbeConfig(
                name="c", layers=[0, 1], aggregation="concat", module_config={"out_features": 3}
            )
        ]
    )
    data = [
        {"tokens": [1, 2, 3, 4, 5], "labels": [0, 1, 2, 1, 0]},
        {"tokens": [6, 7, 8, 9, 10], "labels": [1, 1, 0, 2, 2]},
    ]
    tm = _TrainableModel(m.model, m._probes)
    result = evaluate_joint_model(
        train_model=tm,
        dataset=data,
        batch_size=2,
        max_seq_length=16,
        loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"c": "ce"}),
        eval_metrics_fn=classification_metrics(num_classes=3),
    )
    for key in ("c_acc", "c_adj", "c_macro_f1"):
        assert key in result, f"missing {key} (concat eval used to crash)"

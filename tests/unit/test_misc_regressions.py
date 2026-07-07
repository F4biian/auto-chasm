"""Regression tests for edge cases.

- m1: the truncation warning is specific about label loss (not just "Truncating").
- m2: a grouped split that empties the train set warns (was silent).
- m4: stratify + val_fraction=0 does not emit spurious per-class warnings.
- m18: a single-token batch returns empty loss components (no fabricated lm_head).
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from auto_chasm import Dataset


def test_m1_truncation_warning_mentions_label_loss(caplog) -> None:  # noqa: ANN001
    """The truncation warning explains that labels past the cutoff are dropped (m1)."""
    from auto_chasm.trainers.data_utils import iterate_batches

    data = [{"tokens": list(range(1, 60)), "labels": [0] * 59}]  # 59 > max_seq_length
    with caplog.at_level(logging.WARNING, "auto_chasm"):
        next(iterate_batches(data, batch_size=1, max_seq_length=16, loop=False))
    assert "labels on tokens past the cutoff are dropped" in caplog.text


def test_m2_grouped_split_emptying_train_warns(caplog) -> None:  # noqa: ANN001
    """One oversized atomic group taking every sample into val warns (m2)."""
    ds = Dataset([{"tokens": [i], "labels": [0]} for i in range(6)])
    # A single group over ALL samples is atomic: it goes entirely to val, so train
    # is emptied -- which must warn rather than silently returning an empty train.
    with caplog.at_level(logging.WARNING, "auto_chasm"):
        train, _ = ds.split(0.5, seed=0, groups=["g"] * 6)
    assert len(train) == 0
    assert "train EMPTY" in caplog.text


def test_m4_stratify_zero_val_fraction_is_silent(caplog) -> None:  # noqa: ANN001
    """stratify with val_fraction=0 must not warn once per class (m4)."""
    ds = Dataset([{"tokens": [i], "labels": [i % 3]} for i in range(9)])
    with caplog.at_level(logging.WARNING, "auto_chasm"):
        train, val = ds.split(0.0, seed=0, stratify="label")
    assert len(val) == 0
    assert "rounds to 0" not in caplog.text  # no spurious per-class warnings


def test_m18_single_token_batch_has_empty_components() -> None:
    """A single-token batch supervises nothing -> empty components, not {lm_head: 0} (m18)."""
    from auto_chasm.model import Model
    from auto_chasm.trainers.loss import JointLoss
    from auto_chasm.trainers.trainable import _TrainableModel

    class _TinyMlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.layers = [nn.Linear(8, 8)]
            self.output_proj = nn.Linear(8, 16)

        def __call__(self, x: mx.array, **kwargs: object) -> mx.array:
            h = self.embedding(x)
            for layer in self.layers:
                h = nn.gelu(layer(h))
            return self.output_proj(h)

    class _Cfg:
        hidden_size = 8
        num_hidden_layers = 1

    base = _TinyMlp()
    base.config = _Cfg()
    m = Model(base, None, "mlx")
    m.attach_probe(__import__("auto_chasm").ProbeConfig(name="p", layers=[0], source="hidden"))
    tm = _TrainableModel(m.model, m._probes)

    total, ntoks, components = JointLoss(weights={"lm_head": 0.0})(
        tm, mx.array([[3]]), mx.array([[1]]), mx.array([[0, 1]])
    )
    import math

    assert math.isfinite(float(total))
    assert components == {}  # nothing to supervise -> no terms (was {"lm_head": 0})
    assert float(ntoks) == 0.0

"""Regression tests for critical correctness issues.

Each test pins a specific silent-corruption bug:
- the -100 ignore sentinel must be masked in the RL probe penalty (like JointLoss);
- float regression labels must survive build_dataset + iterate_batches (no int cast);
- Model.freeze_probe must be respected by the MLX trainer wrapper;
- the built-in MLP head must use the same param keys on both backends.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from auto_chasm import JointLoss, Model, ProbeConfig
from auto_chasm.config import RLConfig
from auto_chasm.trainers.rl import RLTrainer
from auto_chasm.trainers.trainable import _TrainableModel


class _TinyMlp(nn.Module):
    def __init__(self, h: int = 16, v: int = 32, layers: int = 4) -> None:
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
    num_hidden_layers = 4


def _model(out_features: int = 1) -> Model:
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(
        ProbeConfig(
            name="p",
            layers=[0],
            aggregation="last",
            module_config={"out_features": out_features},
        )
    )
    # amplify the near-zero head so probe logits are non-trivial — this is what
    # makes the -100 masking bug observable.
    m._probes["p"].module.weight = m._probes["p"].module.weight + 0.7
    return m


def test_rl_penalty_masks_minus100_like_jointloss():
    """The RL probe penalty must equal JointLoss's bce (both ignore -100)."""
    m = _model()
    tm = _TrainableModel(m.model, m._probes)
    batch = mx.array([[1, 2, 3, 4, 5, 6]])
    labels = mx.array([[-100, 0, 1, -100, 1, -100]])  # half ignored
    lengths = mx.array([[0, 5]])

    trainer = RLTrainer(model=m, rl_config=RLConfig(algorithm="sft", beta=1.0), num_iters=1)
    _, _, comp = trainer._sft_probe_loss(tm, batch, labels, lengths)

    jl = JointLoss(weights={"lm_head": 0.0}, losses={"p": "bce"})
    _, _, jcomp = jl(tm, batch, labels, lengths)

    assert float(comp["probe_penalty"]) == np.float32(float(jcomp["p"]))
    # and it is a sane value, not the hundreds the bug produced
    assert 0.0 <= float(comp["probe_penalty"]) < 10.0


def test_float_regression_labels_survive_build_and_batch():
    """Float targets must not be truncated to int anywhere in the data path."""
    from auto_chasm.trainers.data_utils import iterate_batches

    data = [{"tokens": [1, 2, 3, 4, 5], "labels": [0.1, 0.8, 0.3, 0.5, 0.9]} for _ in range(4)]
    _, labels, _ = next(iterate_batches(data, batch_size=2, max_seq_length=16, loop=False))
    assert labels.dtype.kind == "f"
    # a known float value is preserved (would be 0 after int truncation)
    assert 0.7 < float(labels[labels > 0.5].max()) <= 0.9


def test_freeze_probe_respected_by_trainable_wrapper():
    """A frozen probe must stay frozen after _TrainableModel wraps it."""
    from mlx.utils import tree_flatten

    m = _model()
    m.freeze_probe("p")
    tm = _TrainableModel(m.model, m._probes)
    trainable_keys = [k for k, _ in tree_flatten(tm.trainable_parameters())]
    probe_keys = [k for k in trainable_keys if "probe_p" in k]
    assert probe_keys == [], f"frozen probe leaked into trainable set: {probe_keys}"


def test_builtin_mlp_head_keys_match_across_backends():
    """The torch and MLX MLP heads must share parameter key names (fc1/fc2)."""
    import torch

    cfg = ProbeConfig(name="m", layers=[-1], module_type="mlp", module_config={"out_features": 1})
    from auto_chasm.probe import Probe

    mlx_keys = set(
        dict(
            __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(
                Probe(cfg, hidden_dim=16, backend_name="mlx").module.parameters()
            )
        ).keys()
    )
    torch_keys = set(Probe(cfg, hidden_dim=16, backend_name="torch").module.state_dict().keys())
    assert {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"} <= mlx_keys
    assert {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"} <= torch_keys
    del torch

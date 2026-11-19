"""Regression tests for loss edge cases.

Each pins one silent-failure fix in the loss layer:

- m16: ``class_weights`` validation is case-insensitive (``"CE"`` counts as CE).
- m17: a ``(probe, target, opt=default)`` custom loss is the modern API (gets a
  ``ProbeOutput``), not the legacy 3-arg ``(logits, target, mask)`` path.
- m13: a labels dict whose keys match no probe warns (was a silent zero loss).
- m15: config weight fields on a ``combine=`` loss warn (were silently ignored).
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from auto_chasm.config import ProbeConfig, TrainingConfig
from auto_chasm.model import Model
from auto_chasm.trainers.loss import JointLoss
from auto_chasm.trainers.trainable import _TrainableModel
from auto_chasm.trainers.trainer import Trainer


class _TinyMlp(nn.Module):
    """Minimal MLX language model."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = [nn.Linear(8, 8) for _ in range(2)]
        self.output_proj = nn.Linear(8, 16)

    def __call__(self, x: mx.array, **kwargs: object) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Tok:
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int], **kwargs: object) -> str:
        return "x"


class _Cfg:
    hidden_size = 8
    num_hidden_layers = 2


def _model() -> Model:
    base = _TinyMlp()
    base.config = _Cfg()
    return Model(base, _Tok(), "mlx")


def test_m16_class_weights_accepts_uppercase_ce() -> None:
    """class_weights validates against a case-insensitive 'CE' spec (m16)."""
    # Must NOT raise "no probe uses probe_loss='ce'": the router lower-cases specs,
    # so "CE" is a valid CE loss and class_weights is reachable.
    loss = JointLoss(losses={"p": "CE"}, class_weights=[1.0, 1.0])
    assert loss is not None


def test_m17_custom_loss_with_defaulted_param_is_modern_api() -> None:
    """A (probe, target, opt=default) custom loss gets a ProbeOutput, not raw logits (m17)."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden"))
    tm = _TrainableModel(m.model, m._probes)
    seen: dict[str, str] = {}

    def custom(probe: object, target: object, scale: float = 1.0) -> object:
        seen["type"] = type(probe).__name__
        return probe.bce(target)  # only works if probe is a ProbeOutput

    batch = mx.array([[1, 2, 3, 4, 5]])
    labels = mx.array([[0, 1, 0, 1, 0]])
    lengths = mx.array([[0, 4]])
    _, _, comp = JointLoss(weights={"lm_head": 0.0}, losses={"p": custom})(
        tm, batch, labels, lengths
    )
    assert "p" in comp
    assert seen["type"] == "ProbeOutput"  # third param has a default -> still 2-arg API


def test_m13_labels_dict_matching_no_probe_warns(caplog) -> None:  # noqa: ANN001
    """A labels dict whose keys match no probe warns instead of silently zeroing (m13)."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden"))
    tm = _TrainableModel(m.model, m._probes)
    batch = mx.array([[1, 2, 3, 4, 5]])
    lengths = mx.array([[0, 4]])
    labels = {"typo": mx.array([[0, 1, 0, 1, 0]])}  # key is not the probe name
    with caplog.at_level(logging.WARNING, "auto_chasm"):
        JointLoss(weights={"lm_head": 0.0}, losses={"p": "bce"})(tm, batch, labels, lengths)
    assert "match none of the attached probes" in caplog.text


def test_m15_combine_mode_ignores_config_weights_warns(caplog) -> None:  # noqa: ANN001
    """Config probe_weights on a combine= loss warn (silently ignored otherwise) (m15)."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden"))
    loss = JointLoss(combine=lambda terms: terms.lm_head)
    with caplog.at_level(logging.WARNING, "auto_chasm"):
        Trainer(
            model=m,
            loss_fn=loss,
            config=TrainingConfig(probe_weights={"p": 2.0}),
            num_iters=1,
            verbose=False,
        )
    assert "combine=" in caplog.text

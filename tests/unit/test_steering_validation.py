"""Steering regressions: reject silent-noop / ambiguous configs.

STEER-A method='custom' with no steer_fn silently fell through to boundary steering.
STEER-B multi-layer probes are ambiguous to steer (which layer?).
STEER-C config.layer was documented but ignored (steering used the probe's layer).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import Model, ProbeConfig, SteeringConfig


class _TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = [nn.Linear(8, 8) for _ in range(3)]
        self.output_proj = nn.Linear(8, 16)

    def __call__(self, x: mx.array, **k: object) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = 8
    num_hidden_layers = 3


def _model() -> Model:
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    return m


def _noop(hidden, head, logits):  # noqa: ANN001, ANN202
    return hidden


# --- STEER-A: method='custom' requires a steer_fn --------------------------------


def test_steer_a_custom_without_fn_raises() -> None:
    """method='custom' with no steer_fn raises (was a silent fall-through to boundary)."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden"))
    with pytest.raises(ValueError, match="requires a steer_fn"):
        m.enable_steering("p", config=SteeringConfig(method="custom"))


def test_steer_a_custom_with_fn_ok() -> None:
    """method='custom' with a steer_fn enables cleanly."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden"))
    m.enable_steering("p", config=SteeringConfig(method="custom"), steer_fn=_noop)
    assert m.steering_hooks["p"].enabled


# --- STEER-B: multi-layer probes cannot be steered -------------------------------


def test_steer_b_multilayer_probe_raises() -> None:
    """Steering a probe that spans >1 layer is rejected as ambiguous."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[0, 1], source="hidden", aggregation="mean"))
    with pytest.raises(ValueError, match="single-layer"):
        m.enable_steering("p", config=SteeringConfig(method="custom"), steer_fn=_noop)


# --- STEER-C: config.layer must match the probe's layer (or be None) --------------


def test_steer_c_mismatched_layer_raises() -> None:
    """config.layer that disagrees with the probe's layer raises (was ignored)."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden"))
    with pytest.raises(ValueError, match="does not match"):
        m.enable_steering("p", config=SteeringConfig(layer=0, method="custom"), steer_fn=_noop)


def test_steer_c_matching_layer_ok() -> None:
    """config.layer equal to the probe's layer enables cleanly."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden"))
    m.enable_steering("p", config=SteeringConfig(layer=1, method="custom"), steer_fn=_noop)
    assert m.steering_hooks["p"].enabled


def test_steer_c_none_layer_ok() -> None:
    """config.layer left None (the default) is fine — steering uses the probe's layer."""
    m = _model()
    m.attach_probe(ProbeConfig(name="p", layers=[1], source="hidden"))
    m.enable_steering("p", config=SteeringConfig(method="custom"), steer_fn=_noop)
    assert m.steering_hooks["p"].enabled

"""Oracle tests for multi-layer probe aggregation (concat/mean/max/last/callable).

These pin the *value* of the tensor a multi-layer probe feeds to its head, not
merely that a forward pass runs.  A probe reads two **distinct** layers, and we
recompute each aggregation independently from the captured per-layer states:

- ``concat`` → ``concatenate`` along the feature axis (head ``in_features ==
  hidden * len(layers)``),
- ``mean``   → element-wise mean over the layers,
- ``max``    → element-wise max over the layers,
- ``last``   → exactly the **last listed** layer's state (checked with a
  reversed layer order so "last listed" != "last executed"),
- ``callable`` → the custom fn applied to the per-layer state **list** (a
  weighted sum with a known closed form).

Two layers (``[0, 2]``) are used so the per-layer states genuinely differ; a
single-layer model would make every aggregation collapse to the same tensor and
the oracle would be vacuous.

A multi-layer *callable* aggregation must size the head's ``in_features`` to the
callable's actual output width, not to ``hidden * len(layers)``.  A natural
reduction callable (weighted sum, output width ``hidden``) is exercised
end-to-end (see ``TestCallableAggregation.test_reduction_callable_head_sizing``)
alongside the per-aggregation math oracles.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from auto_chasm import Model, ProbeConfig
from auto_chasm.trainers.trainable import _TrainableModel

_HIDDEN = 16
_LAYERS = [0, 2]  # two *distinct*, non-adjacent layers so states differ


class _TinyMlp(nn.Module):
    def __init__(self, h: int = _HIDDEN, v: int = 32, layers: int = 4) -> None:
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
    hidden_size = _HIDDEN
    num_hidden_layers = 4


def _attach(aggregation, layers=None, out_features: int = 2):
    """Build a model with one multi-layer probe; return (model, probe)."""
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    probe = m.attach_probe(
        ProbeConfig(
            name="p",
            layers=list(layers if layers is not None else _LAYERS),
            aggregation=aggregation,
            module_config={"out_features": out_features},
        )
    )
    return m, probe


def _per_layer_states(model: Model, probe, tokens: mx.array):
    """Run the base model once and return captured per-layer states (config order).

    The list mirrors exactly what ``Probe.forward`` aggregates: states ordered
    by ``config.layers`` via ``get_captured_states``.
    """
    probe.clear_captured()
    model.model(tokens)
    states = probe.get_captured_states()
    mx.eval(*states)
    return states


_TOKENS = mx.array([[1, 2, 3, 4, 5]])


class TestConcatAggregation:
    """concat feeds the per-layer states concatenated along the feature axis."""

    def test_aggregated_tensor_is_feature_concat(self) -> None:
        m, probe = _attach("concat")
        s0, s1 = _per_layer_states(m, probe, _TOKENS)
        # Sanity: the two layers really produced different states.
        assert not bool(mx.allclose(s0, s1))

        expected = mx.concatenate([s0, s1], axis=-1)
        assert bool(mx.array_equal(probe._aggregate([s0, s1]), expected))

        # End-to-end: the head output equals the head applied to the oracle agg.
        tm = _TrainableModel(m.model, m._probes)
        _, probes = tm(_TOKENS)
        mx.eval(probes["p"])
        assert bool(mx.allclose(probes["p"], probe.module(expected)))

    def test_head_in_features_is_hidden_times_layers(self) -> None:
        _m, probe = _attach("concat")
        # MLX nn.Linear weight is [out_features, in_features].
        assert probe.module.weight.shape[1] == _HIDDEN * len(_LAYERS)


class TestMeanAggregation:
    """mean feeds the element-wise mean over the captured layers."""

    def test_aggregated_tensor_is_elementwise_mean(self) -> None:
        m, probe = _attach("mean")
        s0, s1 = _per_layer_states(m, probe, _TOKENS)
        expected = (s0 + s1) / 2.0
        assert bool(mx.allclose(probe._aggregate([s0, s1]), expected))

        tm = _TrainableModel(m.model, m._probes)
        _, probes = tm(_TOKENS)
        mx.eval(probes["p"])
        assert bool(mx.allclose(probes["p"], probe.module(expected)))

    def test_head_in_features_is_hidden(self) -> None:
        _m, probe = _attach("mean")
        assert probe.module.weight.shape[1] == _HIDDEN


class TestMaxAggregation:
    """max feeds the element-wise maximum over the captured layers."""

    def test_aggregated_tensor_is_elementwise_max(self) -> None:
        m, probe = _attach("max")
        s0, s1 = _per_layer_states(m, probe, _TOKENS)
        expected = mx.maximum(s0, s1)
        assert bool(mx.array_equal(probe._aggregate([s0, s1]), expected))

        tm = _TrainableModel(m.model, m._probes)
        _, probes = tm(_TOKENS)
        mx.eval(probes["p"])
        assert bool(mx.allclose(probes["p"], probe.module(expected)))

    def test_head_in_features_is_hidden(self) -> None:
        _m, probe = _attach("max")
        assert probe.module.weight.shape[1] == _HIDDEN


class TestLastAggregation:
    """last feeds exactly the LAST LISTED layer's state (config order, not exec)."""

    def test_aggregated_tensor_is_last_listed_layer(self) -> None:
        # Reverse the order so the last *listed* layer (0) is the first *executed*
        # one — distinguishing "last in config" from "last computed".
        m, probe = _attach("last", layers=[2, 0])
        states = _per_layer_states(m, probe, _TOKENS)
        assert probe._resolved_layers == [2, 0]
        expected = states[-1]  # last in config order == layer 0's state
        assert bool(mx.array_equal(probe._aggregate(states), expected))
        # And it is genuinely layer 0, not layer 2 (states[0] is layer 2 here).
        assert not bool(mx.allclose(states[0], states[-1]))

        tm = _TrainableModel(m.model, m._probes)
        _, probes = tm(_TOKENS)
        mx.eval(probes["p"])
        assert bool(mx.allclose(probes["p"], probe.module(expected)))

    def test_head_in_features_is_hidden(self) -> None:
        _m, probe = _attach("last")
        assert probe.module.weight.shape[1] == _HIDDEN


class TestCallableAggregation:
    """A custom callable receives the per-layer state LIST; its math is recomputed."""

    @staticmethod
    def _weighted(states):
        """Closed-form weighted sum over the per-layer states."""
        return 0.25 * states[0] + 0.75 * states[1]

    def test_callable_receives_state_list_and_math_matches(self) -> None:
        m, probe = _attach(self._weighted)
        s0, s1 = _per_layer_states(m, probe, _TOKENS)
        expected = 0.25 * s0 + 0.75 * s1
        # The callable's aggregation math is correct: it reduces the layers to a
        # single hidden-width tensor exactly as the closed form predicts.
        agg = probe._aggregate([s0, s1])
        assert agg.shape == (1, 5, _HIDDEN)
        assert bool(mx.allclose(agg, expected))

    def test_reduction_callable_head_sizing(self) -> None:
        """BUG: a multi-layer callable's head is hardcoded to hidden*len(layers).

        The weighted-sum callable outputs width ``hidden`` (16), but the head is
        built with ``in_features == hidden * len(layers)`` (32).  The correct
        behavior is for the head to match the aggregation it consumes, so the
        end-to-end forward must succeed.  It currently raises a shape error —
        this assertion pins the CORRECT behavior and FAILS, exposing the bug.
        """
        m, probe = _attach(self._weighted)
        s0, s1 = _per_layer_states(m, probe, _TOKENS)
        expected_agg = 0.25 * s0 + 0.75 * s1

        # Correct wiring: the head consumes the aggregation, so its in_features
        # must equal the aggregation's feature width.
        assert probe.module.weight.shape[1] == expected_agg.shape[-1]

        tm = _TrainableModel(m.model, m._probes)
        _, probes = tm(_TOKENS)
        mx.eval(probes["p"])
        # Head output equals the head applied to the (correct) weighted sum.
        assert bool(mx.allclose(probes["p"], probe.module(expected_agg)))

"""Tests for steering module — boundary method, custom steering, serialization."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from auto_chasm.config import SteeringConfig
from auto_chasm.steering import SteeringHook


class TestSteeringBoundary:
    """Tests for the boundary steering method."""

    def test_boundary_method_mlx(self) -> None:
        config = SteeringConfig(method="boundary")
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 16)
        hook._mean_1 = mx.array([2.0] * 16)
        hook._direction = hook._mean_1 - hook._mean_0
        hook._head_norm = 5.0
        hook.enabled = True

        hidden = mx.random.normal((2, 3, 16))
        head = nn.Linear(16, 1)
        logits = head(hidden).squeeze(-1)

        result = hook.steer(hidden, head, logits)
        assert result.shape == (2, 3, 16)

    def test_boundary_with_no_direction(self) -> None:
        config = SteeringConfig(method="boundary")
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 16)
        hook._mean_1 = mx.array([2.0] * 16)
        hook._direction = None  # explicitly no direction
        hook._head_norm = 5.0
        hook.enabled = True

        hidden = mx.random.normal((2, 3, 16))
        head = nn.Linear(16, 1)
        logits = head(hidden).squeeze(-1)

        result = hook.steer(hidden, head, logits)
        assert result is hidden  # unchanged


class TestSteeringCustom:
    """Tests for custom steering functions."""

    def test_set_custom_function(self) -> None:
        config = SteeringConfig()
        hook = SteeringHook("test", config)

        def my_steer(hidden: Any, head: Any, logits: Any) -> Any:
            return hidden * 2.0

        hook.set_custom(my_steer)
        hook.enable()

        hidden = mx.array([[1.0, 2.0]])
        head = nn.Linear(2, 1)
        logits = mx.array([[0.5]])

        result = hook.steer(hidden, head, logits)
        mx.eval(result)
        assert float(result[0, 0].item()) == 2.0
        assert float(result[0, 1].item()) == 4.0

    def test_custom_overrides_builtin(self) -> None:
        config = SteeringConfig(method="nullify")
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 4)
        hook._mean_1 = mx.array([1.0] * 4)
        hook._direction = hook._mean_1 - hook._mean_0
        hook._head_norm = 1.0

        def identity(_h: Any, _head: Any, _logits: Any) -> Any:
            return _h

        hook.set_custom(identity)
        hook.enable()

        hidden = mx.array([[1.0, 2.0, 3.0, 4.0]])
        head = nn.Linear(4, 1)
        logits = mx.array([[0.5]])

        result = hook.steer(hidden, head, logits)
        assert float(mx.sum(result - hidden).item()) == 0.0

    def test_disabled_returns_unchanged(self) -> None:
        config = SteeringConfig()
        hook = SteeringHook("test", config)
        # not enabled

        hidden = mx.array([[1.0, 2.0, 3.0]])
        head = nn.Linear(3, 1)
        logits = mx.array([[0.5]])

        result = hook.steer(hidden, head, logits)
        assert result is hidden


class TestSteeringSerialization:
    """Tests for serialization roundtrips."""

    def test_to_from_dict_roundtrip(self) -> None:
        config = SteeringConfig(method="nullify", scale=1.5)
        hook = SteeringHook("my_probe", config)
        hook._mean_0 = mx.array([1.0, 2.0, 3.0])
        hook._mean_1 = mx.array([4.0, 5.0, 6.0])
        hook._head_norm = 2.5
        hook._head_bias = mx.array([0.1])

        data = hook.to_dict()
        restored = SteeringHook.from_dict(data)

        assert restored.probe_name == "my_probe"
        assert restored.has_geometry
        assert restored._head_norm == 2.5

    def test_to_list_from_list(self) -> None:
        t = mx.array([1.0, 2.0, 3.0])
        lst = SteeringHook._to_list(t)
        assert isinstance(lst, list)
        assert len(lst) == 3

        restored = SteeringHook._from_list([4.0, 5.0, 6.0])
        assert restored.shape == (3,)


class TestSteeringNorm:
    """Tests for the _norm helper."""

    def test_norm_mlx(self) -> None:
        config = SteeringConfig()
        hook = SteeringHook("test", config)
        t = mx.array([3.0, 4.0])
        norm = hook._norm(t)
        assert abs(norm - 5.0) < 1e-6

    def test_norm_zero(self) -> None:
        config = SteeringConfig()
        hook = SteeringHook("test", config)
        t = mx.array([0.0, 0.0, 0.0])
        norm = hook._norm(t)
        assert norm == 0.0


class TestComputeGeometry:
    """Tests for compute_geometry."""

    def test_computes_direction(self) -> None:
        config = SteeringConfig()
        hook = SteeringHook("test", config)
        hidden_by_class = {
            0: [mx.array([1.0, 0.0, 0.0])],
            1: [mx.array([0.0, 1.0, 0.0])],
        }
        head = nn.Linear(3, 1)
        hook.compute_geometry(hidden_by_class, head.weight, head.bias)
        assert hook.has_geometry
        assert hook._direction is not None

    def test_compute_geometry_missing_class(self) -> None:
        config = SteeringConfig()
        hook = SteeringHook("test", config)
        hidden_by_class = {
            0: [mx.array([1.0, 0.0])],
        }
        head = nn.Linear(2, 1)
        hook.compute_geometry(hidden_by_class, head.weight, head.bias)
        assert not hook.has_geometry


class TestSteerWithDirectionOverride:
    """Tests for direction override in config."""

    def test_override_direction(self) -> None:
        config = SteeringConfig(method="nullify", direction=mx.array([1.0] * 8))
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 8)
        hook._mean_1 = mx.array([2.0] * 8)
        hook._direction = hook._mean_1 - hook._mean_0
        hook._head_norm = 1.0
        hook.enabled = True

        hidden = mx.random.normal((1, 5, 8))
        head = nn.Linear(8, 1)
        logits = head(hidden).squeeze(-1)

        result = hook.steer(hidden, head, logits)
        assert result.shape == (1, 5, 8)

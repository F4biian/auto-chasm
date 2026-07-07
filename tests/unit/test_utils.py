"""Tests for utils module — class means, param counting, freeze/unfreeze."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.utils import (
    clean_adapter_keys,
    compute_class_means,
    count_parameters,
    freeze_module,
    unfreeze_module,
)


class TestCleanAdapterKeys:
    """Tests for clean_adapter_keys."""

    def test_cleans_standard_pattern(self) -> None:
        state = {
            "model.layer.3.self_attn.q_proj.lora_a": "a",
            "model.layer.7.self_attn.v_proj.lora_b": "b",
        }
        cleaned = clean_adapter_keys(state)
        assert "model.3.self_attn.q_proj.lora_a" in cleaned

    def test_preserves_unaffected_keys(self) -> None:
        state = {"model.embed_tokens.weight": "x", "model.norm.weight": "y"}
        cleaned = clean_adapter_keys(state)
        assert cleaned == state

    def test_empty_dict(self) -> None:
        assert clean_adapter_keys({}) == {}


class TestComputeClassMeans:
    """Tests for compute_class_means."""

    def test_mlx(self) -> None:
        h = {0: [mx.array([1.0, 2.0]), mx.array([3.0, 4.0])], 1: [mx.array([5.0, 6.0])]}
        means = compute_class_means(h)
        assert 0 in means
        assert float(means[0][0]) == pytest.approx(2.0)
        assert float(means[1][0]) == pytest.approx(5.0)

    def test_empty_class(self) -> None:
        means = compute_class_means({0: [], 1: [mx.array([1.0])]})
        assert 0 not in means
        assert 1 in means

    def test_single_element(self) -> None:
        h = {0: [mx.array([7.0, 8.0])]}
        means = compute_class_means(h)
        assert float(means[0][0]) == 7.0


class TestCountParameters:
    """Tests for count_parameters."""

    def test_mlx_linear(self) -> None:
        mod = nn.Linear(4, 2)
        n = count_parameters(mod)
        assert n == 4 * 2 + 2  # weights + bias

    def test_empty_module(self) -> None:
        class Empty(nn.Module):
            """Test helper."""

            pass

        n = count_parameters(Empty())
        assert n == 0


class TestFreezeUnfreeze:
    """Tests for freeze_module and unfreeze_module."""

    def test_freeze_unfreeze_mlx(self) -> None:
        mod = nn.Linear(4, 2)
        freeze_module(mod)
        unfreeze_module(mod)

    def test_double_freeze(self) -> None:
        mod = nn.Linear(4, 2)
        freeze_module(mod)
        freeze_module(mod)  # should be idempotent


class TestUtilsTorchFallback:
    """Torch fallback paths in utils.py."""

    def test_freeze_module_torch(self) -> None:
        """freeze_module on torch nn.Module disables requires_grad."""
        pytest.importorskip("torch")
        import torch.nn as tnn

        mod = tnn.Linear(4, 2)
        assert all(p.requires_grad for p in mod.parameters())
        freeze_module(mod)
        assert all(not p.requires_grad for p in mod.parameters())

    def test_unfreeze_module_torch(self) -> None:
        """unfreeze_module on torch nn.Module enables requires_grad."""
        pytest.importorskip("torch")
        import torch.nn as tnn

        mod = tnn.Linear(4, 2)
        for p in mod.parameters():
            p.requires_grad = False
        assert not any(p.requires_grad for p in mod.parameters())
        unfreeze_module(mod)
        assert all(p.requires_grad for p in mod.parameters())

    def test_count_parameters_torch(self) -> None:
        """count_parameters works on torch modules."""
        pytest.importorskip("torch")
        import torch.nn as tnn

        mod = tnn.Linear(4, 2)
        n = count_parameters(mod)
        assert n == 4 * 2 + 2  # weights + bias

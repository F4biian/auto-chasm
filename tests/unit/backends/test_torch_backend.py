"""Tests for the PyTorch backend — mirrors test_backend.py (MLX).

Every method in TorchTensorOps, TorchModuleOps, TorchOptimOps, and
TorchModelWrapping is tested.  Seeds are fixed for determinism.
"""

from __future__ import annotations

import numpy as np
import torch

from auto_chasm.backends import Backend
from auto_chasm.backends.torch_backend import (
    TorchModelWrapping,
    TorchModuleOps,
    TorchOptimOps,
    TorchTensorOps,
)

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestTorchBackendSelection:
    """Tests for torch backend instantiation."""

    def test_force_torch(self) -> None:
        backend = Backend(force="torch")
        assert backend.name == "torch"

    def test_has_all_interfaces(self) -> None:
        backend = Backend(force="torch")
        assert hasattr(backend, "tensor")
        assert hasattr(backend, "module")
        assert hasattr(backend, "optim")
        assert hasattr(backend, "wrapping")

    def test_tensor_ops_type(self) -> None:
        backend = Backend(force="torch")
        assert isinstance(backend.tensor, TorchTensorOps)

    def test_module_ops_type(self) -> None:
        backend = Backend(force="torch")
        assert isinstance(backend.module, TorchModuleOps)

    def test_optim_ops_type(self) -> None:
        backend = Backend(force="torch")
        assert isinstance(backend.optim, TorchOptimOps)

    def test_wrapping_type(self) -> None:
        backend = Backend(force="torch")
        assert isinstance(backend.wrapping, TorchModelWrapping)


# ---------------------------------------------------------------------------
# TorchTensorOps
# ---------------------------------------------------------------------------


class TestTorchTensorOps:
    """Tests for TorchTensorOps methods."""

    def setup_method(self) -> None:
        torch.manual_seed(42)
        self.ops = TorchTensorOps()

    def test_tensor_from_list(self) -> None:
        t = self.ops.tensor([1.0, 2.0, 3.0])
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3,)
        assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0]))

    def test_tensor_from_tensor(self) -> None:
        original = torch.tensor([1.0, 2.0])
        result = self.ops.tensor(original)
        assert result is original

    def test_tensor_dtype_cast(self) -> None:
        original = torch.tensor([1, 2, 3])
        result = self.ops.tensor(original, dtype=torch.float32)
        assert result.dtype == torch.float32

    def test_tensor_with_dtype(self) -> None:
        t = self.ops.tensor([1, 2, 3], dtype=torch.float64)
        assert t.dtype == torch.float64

    def test_zeros(self) -> None:
        t = self.ops.zeros((3, 4))
        assert t.shape == (3, 4)
        assert t.dtype == torch.float32
        assert torch.all(t == 0)

    def test_zeros_dtype(self) -> None:
        t = self.ops.zeros((2,), dtype=torch.float64)
        assert t.dtype == torch.float64

    def test_ones(self) -> None:
        t = self.ops.ones((2, 3))
        assert t.shape == (2, 3)
        assert torch.all(t == 1)

    def test_float32(self) -> None:
        assert self.ops.float32() == torch.float32

    def test_to_numpy(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        result = self.ops.to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_to_numpy_no_detach_needed(self) -> None:
        t = torch.tensor([42.0])
        result = self.ops.to_numpy(t)
        assert result[0] == 42.0

    def test_stack(self) -> None:
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        stacked = self.ops.stack([a, b], axis=0)
        assert stacked.shape == (2, 2)

    def test_stack_axis1(self) -> None:
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        stacked = self.ops.stack([a, b], axis=1)
        assert stacked.shape == (2, 2)

    def test_concatenate(self) -> None:
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        cat = self.ops.concatenate([a, b], axis=0)
        assert cat.shape == (2, 2)

    def test_concatenate_axis1(self) -> None:
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        cat = self.ops.concatenate([a, b], axis=1)
        assert cat.shape == (1, 4)

    def test_mean_no_axis(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        m = self.ops.mean(t)
        assert abs(float(m) - 2.5) < 1e-6

    def test_mean_with_axis(self) -> None:
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        m = self.ops.mean(t, axis=0)
        assert m.shape == (2,)
        assert abs(float(m[0]) - 2.0) < 1e-6
        assert abs(float(m[1]) - 3.0) < 1e-6

    def test_matmul(self) -> None:
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0], [6.0]])
        result = self.ops.matmul(a, b)
        assert result.shape == (2, 1)
        assert abs(float(result[0, 0]) - 17.0) < 1e-6
        assert abs(float(result[1, 0]) - 39.0) < 1e-6

    def test_sample_greedy(self) -> None:
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
        token = self.ops.sample(logits, temperature=0.0)
        assert token == 1

    def test_sample_stochastic_deterministic_with_seed(self) -> None:
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.manual_seed(99)
        token1 = self.ops.sample(logits, temperature=1.0)
        torch.manual_seed(99)
        token2 = self.ops.sample(logits, temperature=1.0)
        assert token1 == token2


# ---------------------------------------------------------------------------
# TorchModuleOps
# ---------------------------------------------------------------------------


class TestTorchModuleOps:
    """Tests for TorchModuleOps methods."""

    def setup_method(self) -> None:
        self.ops = TorchModuleOps()
        torch.manual_seed(42)

    def test_linear_parameters(self) -> None:
        linear = torch.nn.Linear(4, 2)
        params = self.ops.parameters(linear)
        assert len(params) == 2  # weight + bias

    def test_parameters_shapes(self) -> None:
        linear = torch.nn.Linear(4, 2)
        params = self.ops.parameters(linear)
        shapes = [p.shape for p in params]
        assert torch.Size([2, 4]) in shapes
        assert torch.Size([2]) in shapes

    def test_trainable_parameters(self) -> None:
        linear = torch.nn.Linear(4, 2)
        params = self.ops.trainable_parameters(linear)
        assert len(params) == 2
        for p in params:
            assert p.requires_grad

    def test_trainable_parameters_after_freeze(self) -> None:
        linear = torch.nn.Linear(4, 2)
        self.ops.freeze(linear)
        trainable = self.ops.trainable_parameters(linear)
        assert len(trainable) == 0

    def test_named_parameters(self) -> None:
        linear = torch.nn.Linear(4, 2)
        named = self.ops.named_parameters(linear)
        assert "weight" in named
        assert "bias" in named

    def test_freeze(self) -> None:
        linear = torch.nn.Linear(4, 2)
        self.ops.freeze(linear)
        for p in linear.parameters():
            assert not p.requires_grad

    def test_unfreeze(self) -> None:
        linear = torch.nn.Linear(4, 2)
        self.ops.freeze(linear)
        self.ops.unfreeze(linear)
        for p in linear.parameters():
            assert p.requires_grad

    def test_freeze_unfreeze_roundtrip(self) -> None:
        linear = torch.nn.Linear(4, 2)
        self.ops.freeze(linear)
        self.ops.unfreeze(linear)
        params = self.ops.trainable_parameters(linear)
        assert len(params) == 2

    def test_eval(self) -> None:
        linear = torch.nn.Linear(4, 2)
        linear.train()
        self.ops.eval(linear)
        assert not linear.training

    def test_train(self) -> None:
        linear = torch.nn.Linear(4, 2)
        linear.eval()
        self.ops.train(linear)
        assert linear.training

    def test_forward(self) -> None:
        linear = torch.nn.Linear(4, 2)
        x = torch.randn(1, 4)
        out = self.ops.forward(linear, x)
        assert out.shape == (1, 2)

    def test_module_list_parameters(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )
        params = self.ops.parameters(model)
        assert len(params) == 4  # 2 weights + 2 biases


# ---------------------------------------------------------------------------
# TorchOptimOps
# ---------------------------------------------------------------------------


class TestTorchOptimOps:
    """Tests for TorchOptimOps methods."""

    def setup_method(self) -> None:
        self.ops = TorchOptimOps()

    def test_create_adamw(self) -> None:
        optim = self.ops.create_adamw(learning_rate=1e-3)
        assert isinstance(optim, torch.optim.AdamW)
        assert optim.param_groups[0]["lr"] == 1e-3

    def test_create_adamw_weight_decay(self) -> None:
        optim = self.ops.create_adamw(learning_rate=1e-3, weight_decay=0.01)
        assert optim.param_groups[0]["weight_decay"] == 0.01

    def test_create_sgd(self) -> None:
        optim = self.ops.create_sgd(learning_rate=1e-3)
        assert isinstance(optim, torch.optim.SGD)
        assert optim.param_groups[0]["lr"] == 1e-3

    def test_create_sgd_momentum(self) -> None:
        optim = self.ops.create_sgd(learning_rate=1e-3, momentum=0.9)
        assert optim.param_groups[0]["momentum"] == 0.9

    def test_zero_grad(self) -> None:
        linear = torch.nn.Linear(4, 2)
        optim = torch.optim.AdamW(linear.parameters(), lr=1e-3)
        x = torch.randn(1, 4)
        loss = linear(x).sum()
        loss.backward()
        assert linear.weight.grad is not None
        self.ops.zero_grad(optim)
        assert linear.weight.grad is None

    def test_scale_lr(self) -> None:
        optim = torch.optim.AdamW([torch.zeros(1)], lr=1e-3)
        self.ops.scale_lr(optim, 0.5)
        assert abs(optim.param_groups[0]["lr"] - 5e-4) < 1e-10

    def test_step_updates_weights(self) -> None:
        torch.manual_seed(42)
        linear = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-2)
        x = torch.randn(2, 4)
        old_weight = linear.weight.clone()
        loss = linear(x).sum()
        loss.backward()
        self.ops.step(optimizer, linear, None)
        assert not torch.allclose(linear.weight, old_weight)

    def test_step_clips_gradients(self) -> None:
        linear = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-3)
        x = torch.randn(1, 4) * 100
        loss = linear(x).sum()
        loss.backward()
        self.ops.step(optimizer, linear, None, max_norm=1.0)
        assert linear.weight.grad is None

    def test_step_returns_optimizer_and_model(self) -> None:
        linear = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-3)
        x = torch.randn(1, 4)
        loss = linear(x).sum()
        loss.backward()
        result = self.ops.step(optimizer, linear, None)
        assert len(result) == 2

    def test_step_lazy_param_injection(self) -> None:
        torch.manual_seed(42)
        linear = torch.nn.Linear(4, 2)
        optimizer = self.ops.create_adamw(learning_rate=1e-2)
        x = torch.randn(2, 4)
        old_weight = linear.weight.clone()
        loss = linear(x).sum()
        loss.backward()
        self.ops.step(optimizer, linear, None)
        assert not torch.allclose(linear.weight, old_weight)
        assert len(optimizer.param_groups[0]["params"]) > 0


# ---------------------------------------------------------------------------
# TorchModelWrapping
# ---------------------------------------------------------------------------


class TestTorchModelWrapping:
    """Tests for TorchModelWrapping methods."""

    def setup_method(self) -> None:
        self.wrapping = TorchModelWrapping()
        torch.manual_seed(42)

    def test_save_load_class_means_roundtrip(self, tmp_path):  # type: ignore[no-untyped-def]
        path = str(tmp_path / "means.pt")
        means = {
            "probe_a": {"mean_0": torch.tensor([1.0, 2.0]), "mean_1": torch.tensor([3.0, 4.0])},
        }
        self.wrapping.save_class_means(means, path)
        loaded = self.wrapping.load_class_means(path)
        assert "probe_a" in loaded
        assert torch.allclose(loaded["probe_a"]["mean_0"], torch.tensor([1.0, 2.0]))
        assert torch.allclose(loaded["probe_a"]["mean_1"], torch.tensor([3.0, 4.0]))

    def test_save_load_class_means_flat(self, tmp_path):  # type: ignore[no-untyped-def]
        path = str(tmp_path / "means_flat.pt")
        means = {"mean_0": torch.tensor([1.0]), "mean_1": torch.tensor([2.0])}
        self.wrapping.save_class_means(means, path)
        loaded = self.wrapping.load_class_means(path)
        assert torch.allclose(loaded["mean_0"], torch.tensor([1.0]))
        assert torch.allclose(loaded["mean_1"], torch.tensor([2.0]))

    def test_save_load_class_means_multiple_probes(self, tmp_path):  # type: ignore[no-untyped-def]
        path = str(tmp_path / "means_multi.pt")
        means = {
            "probe_a": {"mean_0": torch.tensor([1.0]), "mean_1": torch.tensor([2.0])},
            "probe_b": {"mean_0": torch.tensor([3.0]), "mean_1": torch.tensor([4.0])},
        }
        self.wrapping.save_class_means(means, path)
        loaded = self.wrapping.load_class_means(path)
        assert "probe_a" in loaded
        assert "probe_b" in loaded
        assert torch.allclose(loaded["probe_a"]["mean_0"], torch.tensor([1.0]))
        assert torch.allclose(loaded["probe_b"]["mean_1"], torch.tensor([4.0]))

    def test_get_trainable_params(self) -> None:
        linear = torch.nn.Linear(4, 2)
        params = self.wrapping.get_trainable_params(linear)
        assert len(params) == 2
        for p in params:
            assert p.requires_grad

    def test_get_trainable_params_frozen(self) -> None:
        linear = torch.nn.Linear(4, 2)
        for p in linear.parameters():
            p.requires_grad = False
        params = self.wrapping.get_trainable_params(linear)
        assert len(params) == 0

    def test_save_load_adapters_roundtrip(self, tmp_path):  # type: ignore[no-untyped-def]
        path = str(tmp_path / "adapters.pt")
        model = torch.nn.Linear(4, 2)
        # Manually add lora_ keys to state dict for testing
        state = model.state_dict()
        lora_state = {f"lora_{k}": v for k, v in state.items()}
        torch.save(lora_state, path)
        loaded = torch.load(path, map_location="cpu")
        assert any("lora_" in k for k in loaded)

    def test_save_adapters_no_lora_keys(self, tmp_path):  # type: ignore[no-untyped-def]
        path = str(tmp_path / "adapters_empty.pt")
        linear = torch.nn.Linear(4, 2)
        self.wrapping.save_adapters(linear, path)
        assert not (tmp_path / "adapters_empty.pt").exists()


# ---------------------------------------------------------------------------
# Cross-backend interface conformance
# ---------------------------------------------------------------------------


class TestTorchProtocolConformance:
    """Verify TorchTensorOps satisfies the Protocol interfaces."""

    def test_tensor_ops_conforms(self) -> None:
        from auto_chasm.backends.base import TensorOps

        ops = TorchTensorOps()
        assert isinstance(ops, TensorOps)

    def test_module_ops_conforms(self) -> None:
        from auto_chasm.backends.base import ModuleOps

        ops = TorchModuleOps()
        assert isinstance(ops, ModuleOps)

    def test_optim_ops_conforms(self) -> None:
        from auto_chasm.backends.base import OptimOps

        ops = TorchOptimOps()
        assert isinstance(ops, OptimOps)

    def test_wrapping_conforms(self) -> None:
        from auto_chasm.backends.base import ModelWrapping

        ops = TorchModelWrapping()
        assert isinstance(ops, ModelWrapping)

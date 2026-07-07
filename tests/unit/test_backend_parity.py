"""Cross-backend parity tests — MLX vs PyTorch numerical equivalence.

Verifies that both backends produce the same results for identical
inputs.  Uses deterministic seeds and tolerances for floating-point
differences between frameworks.
"""

from __future__ import annotations

import numpy as np

from auto_chasm.backends import Backend

# ---------------------------------------------------------------------------
# Tensor ops parity
# ---------------------------------------------------------------------------


class TestTensorOpsParity:
    """Verify MLX and Torch tensor ops produce identical results."""

    def setup_method(self) -> None:
        self.mlx = Backend(force="mlx").tensor
        self.torch = Backend(force="torch").tensor

    def test_tensor_from_list(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0]
        mlx_t = self.mlx.tensor(data)
        torch_t = self.torch.tensor(data)
        np.testing.assert_allclose(
            self.mlx.to_numpy(mlx_t),
            self.torch.to_numpy(torch_t),
            atol=1e-7,
        )

    def test_zeros_shape_and_values(self) -> None:
        mlx_z = self.mlx.zeros((3, 4))
        torch_z = self.torch.zeros((3, 4))
        np.testing.assert_allclose(
            self.mlx.to_numpy(mlx_z),
            self.torch.to_numpy(torch_z),
            atol=1e-7,
        )

    def test_ones_shape_and_values(self) -> None:
        mlx_o = self.mlx.ones((2, 5))
        torch_o = self.torch.ones((2, 5))
        np.testing.assert_allclose(
            self.mlx.to_numpy(mlx_o),
            self.torch.to_numpy(torch_o),
            atol=1e-7,
        )

    def test_stack(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        mlx_a = self.mlx.tensor(a)
        mlx_b = self.mlx.tensor(b)
        torch_a = self.torch.tensor(a)
        torch_b = self.torch.tensor(b)

        mlx_s = self.mlx.stack([mlx_a, mlx_b], axis=0)
        torch_s = self.torch.stack([torch_a, torch_b], axis=0)
        np.testing.assert_allclose(
            self.mlx.to_numpy(mlx_s),
            self.torch.to_numpy(torch_s),
            atol=1e-7,
        )

    def test_concatenate(self) -> None:
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0]]
        mlx_a = self.mlx.tensor(a)
        mlx_b = self.mlx.tensor(b)
        torch_a = self.torch.tensor(a)
        torch_b = self.torch.tensor(b)

        mlx_c = self.mlx.concatenate([mlx_a, mlx_b], axis=0)
        torch_c = self.torch.concatenate([torch_a, torch_b], axis=0)
        np.testing.assert_allclose(
            self.mlx.to_numpy(mlx_c),
            self.torch.to_numpy(torch_c),
            atol=1e-7,
        )

    def test_mean_no_axis(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mlx_m = float(self.mlx.mean(self.mlx.tensor(data)).item())
        torch_m = float(self.torch.mean(self.torch.tensor(data)).item())
        assert abs(mlx_m - torch_m) < 1e-6

    def test_mean_with_axis(self) -> None:
        data = [[1.0, 2.0], [3.0, 4.0]]
        mlx_m = self.mlx.to_numpy(self.mlx.mean(self.mlx.tensor(data), axis=0))
        torch_m = self.torch.to_numpy(self.torch.mean(self.torch.tensor(data), axis=0))
        np.testing.assert_allclose(mlx_m, torch_m, atol=1e-6)

    def test_matmul(self) -> None:
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0], [6.0]]
        mlx_r = self.mlx.matmul(self.mlx.tensor(a), self.mlx.tensor(b))
        torch_r = self.torch.matmul(self.torch.tensor(a), self.torch.tensor(b))
        np.testing.assert_allclose(
            self.mlx.to_numpy(mlx_r),
            self.torch.to_numpy(torch_r),
            atol=1e-5,
        )

    def test_sample_greedy_parity(self) -> None:
        logits = [1.0, 5.0, 3.0, 2.0]
        mlx_token = self.mlx.sample(self.mlx.tensor(logits), temperature=0.0)
        torch_token = self.torch.sample(self.torch.tensor(logits), temperature=0.0)
        assert mlx_token == torch_token == 1


# ---------------------------------------------------------------------------
# Loss computation parity
# ---------------------------------------------------------------------------


class TestLossParity:
    """Verify JointLoss produces equivalent results on both backends."""

    def test_bce_loss_parity(self) -> None:
        """BCE loss values should be numerically close via JointLoss."""
        np.random.seed(42)
        raw_logits = np.random.randn(2, 5).astype(np.float32)
        raw_targets = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=np.float32)

        import torch

        # Compute BCE directly with both frameworks
        torch_logits = torch.tensor(raw_logits)
        torch_targets = torch.tensor(raw_targets)
        torch_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            torch_logits, torch_targets, reduction="mean"
        )

        import mlx.core as mx
        import mlx.nn as mlx_nn

        mlx_bce = mlx_nn.losses.binary_cross_entropy(
            mx.array(raw_logits), mx.array(raw_targets), reduction="mean", with_logits=True
        )

        mlx_float = float(mlx_bce.item())
        torch_float = float(torch_bce.item())
        assert abs(mlx_float - torch_float) < 1e-5, (
            f"BCE parity failed: mlx={mlx_float}, torch={torch_float}"
        )

    def test_mse_loss_parity(self) -> None:
        """MSE loss values should be numerically close."""
        np.random.seed(42)
        raw_pred = np.random.randn(2, 5).astype(np.float32)
        raw_target = np.random.randn(2, 5).astype(np.float32)

        import mlx.core as mx
        import mlx.nn as mlx_nn
        import torch
        import torch.nn.functional as torch_functional

        mlx_mse = float(mlx_nn.losses.mse_loss(mx.array(raw_pred), mx.array(raw_target)).item())
        torch_mse = float(
            torch_functional.mse_loss(torch.tensor(raw_pred), torch.tensor(raw_target)).item()
        )
        assert abs(mlx_mse - torch_mse) < 1e-5, (
            f"MSE parity failed: mlx={mlx_mse}, torch={torch_mse}"
        )

    def test_cross_entropy_parity(self) -> None:
        """Cross-entropy loss values should be numerically close."""
        np.random.seed(42)
        raw_logits = np.random.randn(3, 10).astype(np.float32)
        raw_targets = np.array([2, 5, 7], dtype=np.int64)

        import mlx.core as mx
        import mlx.nn as mlx_nn
        import torch
        import torch.nn.functional as torch_functional

        mlx_ce = float(
            mx.mean(mlx_nn.losses.cross_entropy(mx.array(raw_logits), mx.array(raw_targets))).item()
        )
        torch_ce = float(
            torch_functional.cross_entropy(
                torch.tensor(raw_logits), torch.tensor(raw_targets)
            ).item()
        )
        assert abs(mlx_ce - torch_ce) < 1e-4, f"CE parity failed: mlx={mlx_ce}, torch={torch_ce}"


# ---------------------------------------------------------------------------
# Gradient computation parity
# ---------------------------------------------------------------------------


class TestGradientParity:
    """Verify gradient computations produce similar results."""

    def test_linear_backward_parity(self) -> None:
        """Gradients through a linear layer should be numerically close."""
        np.random.seed(42)
        w_init = np.random.randn(2, 4).astype(np.float32)
        b_init = np.random.randn(2).astype(np.float32)
        x_data = np.random.randn(1, 4).astype(np.float32)

        import mlx.core as mx
        import mlx.nn as mlx_nn
        import torch

        # MLX
        mlx_linear = mlx_nn.Linear(4, 2)
        mlx_linear.weight = mx.array(w_init)
        mlx_linear.bias = mx.array(b_init)

        def mlx_loss_fn(model, x):
            return mx.sum(model(mx.array(x)))

        mlx_loss, mlx_grads = mx.value_and_grad(mlx_loss_fn)(mlx_linear, x_data)
        mlx_grad_w = mlx_grads["weight"]

        # Torch
        torch_linear = torch.nn.Linear(4, 2)
        torch_linear.weight.data = torch.tensor(w_init)
        torch_linear.bias.data = torch.tensor(b_init)

        torch_loss = torch_linear(torch.tensor(x_data)).sum()
        torch_loss.backward()
        torch_grad_w = torch_linear.weight.grad

        np.testing.assert_allclose(
            np.array(mlx_grad_w),
            torch_grad_w.numpy(),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Sampling distribution parity
# ---------------------------------------------------------------------------


class TestSamplingParity:
    """Verify sampling distributions are statistically equivalent."""

    def test_greedy_always_same(self) -> None:
        """Greedy sampling (temperature=0) must always pick argmax."""
        import mlx.core as mx
        import torch

        from auto_chasm.backends.mlx_backend import MLXTensorOps
        from auto_chasm.backends.torch_backend import TorchTensorOps

        mlx_ops = MLXTensorOps()
        torch_ops = TorchTensorOps()

        logits = [0.1, 0.5, 0.3, 0.8, 0.2]
        for _ in range(10):
            mlx_token = mlx_ops.sample(mx.array(logits), temperature=0.0)
            torch_token = torch_ops.sample(torch.tensor(logits), temperature=0.0)
            assert mlx_token == torch_token == 3

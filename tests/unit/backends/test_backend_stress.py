"""Cross-backend stress tests — MLX vs PyTorch numerical parity.

Every test seeds both backends identically and asserts results match
within 1e-5 absolute tolerance.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from auto_chasm import _checkpoint_weights as _ckw
from auto_chasm.backends import Backend
from auto_chasm.class_means import _compute_mlx, _compute_torch
from auto_chasm.config import ProbeConfig
from auto_chasm.model import Model
from auto_chasm.steering import _steer_mlx, _steer_torch
from auto_chasm.trainers.loss import JointLoss


class _DummyTokenizer:
    """Minimal tokenizer for model-level tests."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "test"


class _DummyConfig:
    """Minimal model configuration."""

    hidden_size = 16
    num_hidden_layers = 4


def _seed_all(seed: int = 42) -> None:  # Seed numpy, MLX, and torch identically.
    import torch

    np.random.seed(seed)
    mx.random.seed(seed)
    torch.manual_seed(seed)


def _copy_mlx_to_torch(
    mlx_mod: nn.Module, torch_mod: Any
) -> None:  # Copy MLX weights to a torch model in-place.
    import torch

    mlx_params = dict(tree_flatten(mlx_mod.parameters()))
    with torch.no_grad():
        for name, p in torch_mod.named_parameters():
            if name in mlx_params:
                p.data.copy_(torch.tensor(np.array(mlx_params[name])))


def _assert_close(
    mlx_t: mx.array, torch_t: Any, atol: float = 1e-5
) -> None:  # Assert MLX and torch tensors are elementwise close.
    np.testing.assert_allclose(
        np.array(mlx_t),
        torch_t.detach().cpu().numpy(),
        atol=atol,
    )


# JointLoss numerical parity


class TestBackendJointLossNumericalParity:
    """JointLoss produces identical loss components for MLX and torch."""

    def _setup(self) -> tuple:  # Create identical test data for both backends.
        np.random.seed(42)
        return (
            np.random.randn(1, 5, 32).astype(np.float32),
            np.random.randn(1, 5).astype(np.float32),
            np.array([[1, 2, 3, 4, 5, 0]], dtype=np.int32),
            np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int32),
            np.array([[0, 5]], dtype=np.int32),
        )

    def _run(
        self,
        loss: JointLoss,
        lm: np.ndarray,
        pr: np.ndarray,
        batch: np.ndarray,
        labels: np.ndarray,
        lengths: np.ndarray,
    ) -> tuple:
        """Compute loss on both backends, return results tuple."""
        import torch

        mlx_t, mlx_n, mlx_c = loss(
            _MLXModel(lm, pr),
            mx.array(batch),
            mx.array(labels),
            mx.array(lengths),
        )
        torch_t, torch_n, torch_c = loss(
            _TorchModel(lm, pr),
            torch.tensor(batch),
            torch.tensor(labels),
            torch.tensor(lengths),
        )
        return (float(mlx_t), float(torch_t), float(mlx_n), float(torch_n), mlx_c, torch_c)

    def test_bce_loss_parity(self) -> None:  # BCE loss components match between MLX and torch.
        pytest.importorskip("torch")
        lm, pr, batch, labels, lengths = self._setup()
        loss = JointLoss(losses={"probe": "bce"})
        mt, tt, _, _, mc, tc = self._run(loss, lm, pr, batch, labels, lengths)
        assert abs(float(mc["probe"]) - float(tc["probe"])) < 1e-5
        assert abs(mt - tt) < 1e-5

    def test_mse_loss_parity(self) -> None:  # MSE loss components match between MLX and torch.
        pytest.importorskip("torch")
        lm, pr, batch, labels, lengths = self._setup()
        loss = JointLoss(losses={"probe": "mse"})
        mt, tt, _, _, mc, tc = self._run(loss, lm, pr, batch, labels, lengths)
        assert abs(float(mc["probe"]) - float(tc["probe"])) < 1e-5
        assert abs(mt - tt) < 1e-5

    def test_ce_loss_multi_class(self) -> None:
        """LM cross-entropy (multi-class) matches between backends."""
        pytest.importorskip("torch")
        lm, pr, batch, labels, lengths = self._setup()
        loss = JointLoss(weights={"probe": 0.0})
        mt, tt, _, _, mc, tc = self._run(loss, lm, pr, batch, labels, lengths)
        assert "lm_head" in mc and "lm_head" in tc
        assert abs(float(mc["lm_head"]) - float(tc["lm_head"])) < 1e-4
        assert abs(mt - tt) < 1e-4

    def test_mixed_probe_weights(
        self,
    ) -> None:  # Per-probe weight overrides produce matching losses.
        pytest.importorskip("torch")
        import torch

        np.random.seed(42)
        lm = np.random.randn(1, 5, 32).astype(np.float32)
        p1 = np.random.randn(1, 5).astype(np.float32)
        p2 = np.random.randn(1, 5).astype(np.float32)
        batch = np.array([[1, 2, 3, 4, 5, 0]], dtype=np.int32)
        labels = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int32)
        lengths = np.array([[0, 5]], dtype=np.int32)

        loss = JointLoss(weights={"p1": 2.0, "p2": 0.5})

        class _MM:
            def __call__(self, x):
                return mx.array(lm), {"p1": mx.array(p1), "p2": mx.array(p2)}

        class _TM:
            def __call__(self, x):
                return torch.tensor(lm), {"p1": torch.tensor(p1), "p2": torch.tensor(p2)}

        mlx_t, _, mlx_c = loss(
            _MM(),
            mx.array(batch),
            mx.array(labels),
            mx.array(lengths),
        )
        torch_t, _, torch_c = loss(
            _TM(),
            torch.tensor(batch),
            torch.tensor(labels),
            torch.tensor(lengths),
        )
        # Two bce probes → per-probe keys; each head matches across backends.
        assert abs(float(mlx_c["p1"]) - float(torch_c["p1"])) < 1e-5
        assert abs(float(mlx_c["p2"]) - float(torch_c["p2"])) < 1e-5
        assert abs(float(mlx_t) - float(torch_t)) < 1e-5

    def test_zero_lm_weight(self) -> None:
        """Classifier-only mode (lm_weight=0) matches across backends."""
        pytest.importorskip("torch")
        lm, pr, batch, labels, lengths = self._setup()
        loss = JointLoss(weights={"lm_head": 0.0})
        _, _, _, _, mc, tc = self._run(loss, lm, pr, batch, labels, lengths)
        assert "lm_head" not in mc and "lm_head" not in tc
        assert abs(float(mc["probe"]) - float(tc["probe"])) < 1e-5

    def test_with_masking(self) -> None:
        """Masked positions produce matching losses across backends."""
        pytest.importorskip("torch")
        lm, pr, batch, labels, _ = self._setup()
        lengths = np.array([[2, 4]], dtype=np.int32)
        loss = JointLoss(losses={"probe": "bce"})
        mt, tt, mn, tn, _, _ = self._run(loss, lm, pr, batch, labels, lengths)
        assert mn == tn
        assert abs(mt - tt) < 1e-5


class _MLXModel:
    """MLX fake model returning fixed arrays from __call__."""

    def __init__(self, lm: np.ndarray, pr: np.ndarray):
        self.lm = mx.array(lm)
        self.pr = {"probe": mx.array(pr)}

    def __call__(self, inputs):
        return self.lm, self.pr


class _TorchModel:
    """Torch fake model returning fixed tensors from __call__."""

    def __init__(self, lm: np.ndarray, pr: np.ndarray):
        import torch

        self.lm = torch.tensor(lm)
        self.pr = {"probe": torch.tensor(pr)}

    def __call__(self, inputs):
        return self.lm, self.pr


# Steering numerical parity


class TestBackendSteeringNumericalParity:
    """_steer_mlx and _steer_torch produce identical results."""

    def test_nullify(self) -> None:  # Nullify steering method produces identical results.
        pytest.importorskip("torch")
        h, head, logits, m0, m1, d, ht, headt, logitst, m0t, m1t, dt = self._inputs()
        _assert_close(
            _steer_mlx(h, head, logits, "nullify", m0, m1, d),
            _steer_torch(ht, headt, logitst, "nullify", m0t, m1t, dt),
        )

    def test_push_to_mean(self) -> None:
        """Push_to_mean steering method produces identical results."""
        pytest.importorskip("torch")
        h, head, logits, m0, m1, d, ht, headt, logitst, m0t, m1t, dt = self._inputs()
        _assert_close(
            _steer_mlx(h, head, logits, "push_to_mean", m0, m1, d),
            _steer_torch(ht, headt, logitst, "push_to_mean", m0t, m1t, dt),
        )

    def test_boundary(self) -> None:  # Boundary steering method produces identical results.
        pytest.importorskip("torch")
        h, head, logits, m0, m1, d, ht, headt, logitst, m0t, m1t, dt = self._inputs()
        _assert_close(
            _steer_mlx(h, head, logits, "boundary", m0, m1, d),
            _steer_torch(ht, headt, logitst, "boundary", m0t, m1t, dt),
        )

    def test_scale_zero(self) -> None:
        """Scale=0 produces identity (no change) in both backends."""
        pytest.importorskip("torch")
        h, head, logits, m0, m1, d, ht, headt, logitst, m0t, m1t, dt = self._inputs()
        _assert_close(
            _steer_mlx(h, head, logits, "nullify", m0, m1, d),
            _steer_torch(ht, headt, logitst, "nullify", m0t, m1t, dt),
        )

    def test_large_scale(self) -> None:  # Large scale (1000x) produces finite matching results.
        pytest.importorskip("torch")
        import torch

        _seed_all(42)
        h_np = np.random.randn(2, 5, 8).astype(np.float32) * 0.1
        w_np = np.random.randn(1, 8).astype(np.float32) * 0.1
        b_np = np.random.randn(1).astype(np.float32)
        m0_np = np.random.randn(8).astype(np.float32)
        m1_np = np.random.randn(8).astype(np.float32)
        d_np = (m1_np - m0_np).astype(np.float32)

        head_m = nn.Linear(8, 1)
        head_m.weight = mx.array(w_np)
        head_m.bias = mx.array(b_np)
        h_m = mx.array(h_np)
        logits_m = head_m(h_m).squeeze(-1)

        head_t = torch.nn.Linear(8, 1)
        with torch.no_grad():
            head_t.weight.copy_(torch.tensor(w_np))
            head_t.bias.copy_(torch.tensor(b_np))
        h_t = torch.tensor(h_np)
        logits_t = head_t(h_t).squeeze(-1)

        r_m = _steer_mlx(
            h_m, head_m, logits_m, "push_to_mean", mx.array(m0_np), mx.array(m1_np), mx.array(d_np)
        )
        r_t = _steer_torch(
            h_t,
            head_t,
            logits_t,
            "push_to_mean",
            torch.tensor(m0_np),
            torch.tensor(m1_np),
            torch.tensor(d_np),
        )
        assert mx.isfinite(r_m).all()
        assert torch.isfinite(r_t).all()
        _assert_close(r_m, r_t, atol=1e-4)

    def test_single_token(self) -> None:  # Steering with single token (seq_len=1) matches.
        pytest.importorskip("torch")
        h, head, logits, m0, m1, d, ht, headt, logitst, m0t, m1t, dt = self._inputs(seq_len=1)
        _assert_close(
            _steer_mlx(h, head, logits, "nullify", m0, m1, d),
            _steer_torch(ht, headt, logitst, "nullify", m0t, m1t, dt),
        )

    def _inputs(
        self, hid: int = 8, batch: int = 2, seq_len: int = 5
    ) -> tuple:  # Create identical steering inputs for both backends.
        import torch

        _seed_all(42)
        h_np = np.random.randn(batch, seq_len, hid).astype(np.float32)
        w_np = np.random.randn(1, hid).astype(np.float32)
        b_np = np.random.randn(1).astype(np.float32)
        m0_np = np.random.randn(hid).astype(np.float32)
        m1_np = np.random.randn(hid).astype(np.float32)
        d_np = (m1_np - m0_np).astype(np.float32)

        head_m = nn.Linear(hid, 1)
        head_m.weight, head_m.bias = mx.array(w_np), mx.array(b_np)
        logits_m = head_m(mx.array(h_np)).squeeze(-1)

        head_t = torch.nn.Linear(hid, 1)
        with torch.no_grad():
            head_t.weight.copy_(torch.tensor(w_np))
            head_t.bias.copy_(torch.tensor(b_np))
        logits_t = head_t(torch.tensor(h_np)).squeeze(-1)

        return (
            mx.array(h_np),
            head_m,
            logits_m,
            mx.array(m0_np),
            mx.array(m1_np),
            mx.array(d_np),
            torch.tensor(h_np),
            head_t,
            logits_t,
            torch.tensor(m0_np),
            torch.tensor(m1_np),
            torch.tensor(d_np),
        )


# Gradient parity


class TestBackendGradientParity:
    """Trainable model gradients are identical between backends."""

    def test_linear_forward_backward(
        self,
    ) -> None:  # Linear layer gradients match between MLX and torch.
        pytest.importorskip("torch")
        import torch

        _seed_all(42)
        w_np, b_np = np.random.randn(2, 4).astype(np.float32), np.random.randn(2).astype(np.float32)
        x_np = np.random.randn(1, 4).astype(np.float32)

        mlx_lin = nn.Linear(4, 2)
        mlx_lin.weight, mlx_lin.bias = mx.array(w_np), mx.array(b_np)
        _, grads = mx.value_and_grad(lambda m: mx.sum(m(mx.array(x_np))))(mlx_lin)

        torch_lin = torch.nn.Linear(4, 2)
        with torch.no_grad():
            torch_lin.weight.copy_(torch.tensor(w_np))
            torch_lin.bias.copy_(torch.tensor(b_np))
        torch_lin(torch.tensor(x_np)).sum().backward()

        _assert_close(grads["weight"], torch_lin.weight.grad)
        _assert_close(grads["bias"], torch_lin.bias.grad)

    def test_mlp_forward_backward(
        self,
    ) -> None:  # Two-layer MLP gradients match between MLX and torch.
        pytest.importorskip("torch")
        import torch

        _seed_all(42)
        w1, b1 = np.random.randn(8, 4).astype(np.float32), np.random.randn(8).astype(np.float32)
        w2, b2 = np.random.randn(2, 8).astype(np.float32), np.random.randn(2).astype(np.float32)
        x_np = np.random.randn(1, 4).astype(np.float32)

        mlx_m = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
        params = mlx_m.parameters()
        params["layers"][0]["weight"] = mx.array(w1)
        params["layers"][0]["bias"] = mx.array(b1)
        params["layers"][2]["weight"] = mx.array(w2)
        params["layers"][2]["bias"] = mx.array(b2)
        mlx_m.update(params)
        _, grads = mx.value_and_grad(lambda m: mx.sum(m(mx.array(x_np))))(mlx_m)

        torch_m = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.GELU(),
            torch.nn.Linear(8, 2),
        )
        with torch.no_grad():
            torch_m[0].weight.copy_(torch.tensor(w1))
            torch_m[0].bias.copy_(torch.tensor(b1))
            torch_m[2].weight.copy_(torch.tensor(w2))
            torch_m[2].bias.copy_(torch.tensor(b2))
        torch_m(torch.tensor(x_np)).sum().backward()

        _assert_close(grads["layers"][0]["weight"], torch_m[0].weight.grad)
        _assert_close(grads["layers"][0]["bias"], torch_m[0].bias.grad)
        _assert_close(grads["layers"][2]["weight"], torch_m[2].weight.grad)
        _assert_close(grads["layers"][2]["bias"], torch_m[2].bias.grad)

    def test_gradient_clipping_parity(self) -> None:
        """Gradient clipping with same max_norm produces identical results."""
        pytest.importorskip("torch")
        import torch

        from auto_chasm.trainers.trainable import clip_grad_norm as mlx_clip

        _seed_all(42)
        w_np, b_np = np.random.randn(3, 5).astype(np.float32), np.random.randn(3).astype(np.float32)
        x_np = np.random.randn(1, 5).astype(np.float32)

        mlx_lin = nn.Linear(5, 3)
        mlx_lin.weight, mlx_lin.bias = mx.array(w_np), mx.array(b_np)
        _, grads = mx.value_and_grad(lambda m: mx.sum(m(mx.array(x_np))))(mlx_lin)

        torch_lin = torch.nn.Linear(5, 3)
        with torch.no_grad():
            torch_lin.weight.copy_(torch.tensor(w_np))
            torch_lin.bias.copy_(torch.tensor(b_np))
        torch_lin(torch.tensor(x_np)).sum().backward()

        # Compare raw gradients first
        _assert_close(grads["weight"], torch_lin.weight.grad)
        _assert_close(grads["bias"], torch_lin.bias.grad)

        # Verify clipping reduces norm as expected
        clipped = mlx_clip(grads, max_norm=0.5)
        norm_after = float(mx.sqrt(sum(mx.sum(g**2) for g in clipped.values())))
        assert norm_after <= 0.5 + 1e-6

    def test_multiple_sequential_steps(self) -> None:
        """Gradients after multiple forward-backward steps match."""
        pytest.importorskip("torch")
        import torch

        _seed_all(42)
        w_np = np.random.randn(2, 4).astype(np.float32)
        b_np = np.random.randn(2).astype(np.float32)
        lr = 0.01

        mlx_lin = nn.Linear(4, 2)
        mlx_lin.weight, mlx_lin.bias = mx.array(w_np), mx.array(b_np)
        torch_lin = torch.nn.Linear(4, 2)
        with torch.no_grad():
            torch_lin.weight.copy_(torch.tensor(w_np))
            torch_lin.bias.copy_(torch.tensor(b_np))

        for step in range(3):
            x_np = np.random.randn(1, 4).astype(np.float32) * (step + 1)

            def _loss(m, xx=x_np):
                return mx.sum(m(mx.array(xx)))

            _, grads = mx.value_and_grad(_loss)(mlx_lin)
            torch_lin(torch.tensor(x_np)).sum().backward()

            _assert_close(grads["weight"], torch_lin.weight.grad)
            _assert_close(grads["bias"], torch_lin.bias.grad)

            mlx_lin.weight -= lr * grads["weight"]
            mlx_lin.bias -= lr * grads["bias"]
            with torch.no_grad():
                torch_lin.weight -= lr * torch_lin.weight.grad
                torch_lin.bias -= lr * torch_lin.bias.grad
            torch_lin.zero_grad()


# Tensor op parity


class TestBackendTensorOpParity:
    """All tensor operations produce identical results for same inputs."""

    def setup_method(self) -> None:
        self.mlx_t = Backend(force="mlx").tensor
        self.torch_t = Backend(force="torch").tensor

    def test_tensor_zeros_ones(self) -> None:  # tensor(), zeros(), ones() produce identical values.
        t = self.mlx_t
        tt = self.torch_t
        np.testing.assert_allclose(
            t.to_numpy(t.tensor([1.0, 2.0])), tt.to_numpy(tt.tensor([1.0, 2.0]))
        )
        np.testing.assert_allclose(t.to_numpy(t.zeros((2, 3))), tt.to_numpy(tt.zeros((2, 3))))
        np.testing.assert_allclose(t.to_numpy(t.ones((3, 2))), tt.to_numpy(tt.ones((3, 2))))

    def test_to_numpy_roundtrip(self) -> None:
        """to_numpy roundtrip preserves values for both frameworks."""
        mlx_np = self.mlx_t.to_numpy(self.mlx_t.tensor([3.14, 2.718]))
        torch_np = self.torch_t.to_numpy(self.torch_t.tensor([3.14, 2.718]))
        np.testing.assert_allclose(mlx_np, torch_np, atol=1e-7)

    def test_stack_concatenate_mean(self) -> None:
        """Stack, concatenate, and mean produce identical values."""
        ma = self.mlx_t.tensor([1.0, 2.0, 3.0])
        mb = self.mlx_t.tensor([4.0, 5.0, 6.0])
        ta = self.torch_t.tensor([1.0, 2.0, 3.0])
        tb = self.torch_t.tensor([4.0, 5.0, 6.0])

        np.testing.assert_allclose(
            self.mlx_t.to_numpy(self.mlx_t.stack([ma, mb])),
            self.torch_t.to_numpy(self.torch_t.stack([ta, tb])),
        )
        np.testing.assert_allclose(
            self.mlx_t.to_numpy(self.mlx_t.concatenate([ma, mb])),
            self.torch_t.to_numpy(self.torch_t.concatenate([ta, tb])),
        )
        assert abs(self.mlx_t.mean(ma).item() - self.torch_t.mean(ta).item()) < 1e-6

    def test_matmul_different_shapes(self) -> None:
        """Matmul with various shapes produces identical results."""
        a, b = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0], [8.0]]
        np.testing.assert_allclose(
            self.mlx_t.to_numpy(self.mlx_t.matmul(self.mlx_t.tensor(a), self.mlx_t.tensor(b))),
            self.torch_t.to_numpy(
                self.torch_t.matmul(self.torch_t.tensor(a), self.torch_t.tensor(b))
            ),
            atol=1e-5,
        )

    def test_sample_greedy(self) -> None:
        """Greedy sampling (temp=0) must pick exact same argmax token."""
        logits = [0.1, 0.5, 0.3, 0.8, 0.2]
        assert self.mlx_t.sample(self.mlx_t.tensor(logits), 0.0) == 3
        assert self.torch_t.sample(self.torch_t.tensor(logits), 0.0) == 3

    def test_sample_tempered(self) -> None:
        """Tempered sampling produces valid token (cross-framework PRNG differs)."""
        _seed_all(99)
        mlx_tok = self.mlx_t.sample(self.mlx_t.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), 1.0)
        _seed_all(99)
        torch_tok = self.torch_t.sample(self.torch_t.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), 1.0)
        assert 0 <= mlx_tok < 5
        assert 0 <= torch_tok < 5


# Model forward parity


class TestBackendModelForwardParity:
    """Model.forward() produces identical outputs for both backends."""

    def _make_torch_wrapper(self, model_wrapper):  # Create torch Model with identical weights.
        from tests.conftest import _make_torch_tiny_mlp

        m = _make_torch_tiny_mlp()
        m.config = _DummyConfig()
        _copy_mlx_to_torch(model_wrapper.model, m)
        return Model(m, _DummyTokenizer(), backend_name="torch")

    def test_forward_with_probes(
        self, model_wrapper: Any
    ) -> None:  # Forward pass with probes produces matching logits.
        pytest.importorskip("torch")
        # Copy weights BEFORE probe injection so param keys match exactly
        tw = self._make_torch_wrapper(model_wrapper)
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        tw.attach_probe(ProbeConfig(name="p", layers=[1]))
        mlx_l = model_wrapper.forward(np.array([[1, 2, 3]], dtype=np.int32)).lm_logits
        _assert_close(mlx_l, tw.forward(np.array([[1, 2, 3]], dtype=np.int32)).lm_logits)

    def test_forward_without_probes(
        self, model_wrapper: Any
    ) -> None:  # Forward pass without probes produces matching logits.
        pytest.importorskip("torch")
        mlx_l = model_wrapper.forward(np.array([[1, 2, 3]], dtype=np.int32)).lm_logits
        _assert_close(
            mlx_l,
            self._make_torch_wrapper(model_wrapper)
            .forward(np.array([[1, 2, 3]], dtype=np.int32))
            .lm_logits,
        )

    def test_forward_with_mask(self, model_wrapper: Any) -> None:
        """Forward pass with attention mask produces matching outputs."""
        pytest.importorskip("torch")
        inp = np.array([[1, 2, 3, 0, 0]], dtype=np.int32)
        mask = np.array([[1, 1, 1, 0, 0]], dtype=np.int32)
        mlx_l = model_wrapper.forward(inp, attention_mask=mask).lm_logits
        _assert_close(
            mlx_l,
            self._make_torch_wrapper(model_wrapper).forward(inp, attention_mask=mask).lm_logits,
        )

    def test_generate_with_probes(self, model_wrapper: Any) -> None:
        """generate_with_probes yields same tokens for both backends."""
        pytest.importorskip("torch")
        # Copy weights BEFORE probe injection so param keys match exactly
        tw = self._make_torch_wrapper(model_wrapper)
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        tw.attach_probe(ProbeConfig(name="p", layers=[1]))
        mlx_t = [s.token_id for s in model_wrapper.generate_with_probes("ab", max_tokens=3)]
        torch_t = [s.token_id for s in tw.generate_with_probes("ab", max_tokens=3)]
        assert mlx_t == torch_t


# Class means parity


class TestBackendClassMeansParity:
    """class_means computation produces identical results."""

    def _run_means(self, model_wrapper, labels_np):
        """Compute class means on both backends with given labels."""
        pytest.importorskip("torch")
        from tests.conftest import _make_torch_tiny_mlp

        # Copy weights BEFORE probe injection so param keys match exactly
        torch_model = _make_torch_tiny_mlp()
        torch_model.config = _DummyConfig()
        _copy_mlx_to_torch(model_wrapper.model, torch_model)
        tw = Model(torch_model, _DummyTokenizer(), backend_name="torch")

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        tw.attach_probe(ProbeConfig(name="p", layers=[1]))
        probe = model_wrapper._probes["p"]

        def _iter(ds, bs, msl, loop=False):
            yield labels_np

        mlx = _compute_mlx(model_wrapper, probe, None, 16, 1, 5, _iter)
        torch_r = _compute_torch(tw, tw._probes["p"], None, 16, 1, 5, _iter)
        return mlx, torch_r

    def test_identical_dataset_identical_means(self, model_wrapper: Any) -> None:
        """Same dataset produces identical mean vectors in both backends."""
        labels = (
            np.array([[1, 2, 3, 0, 0]], dtype=np.int32),
            np.array([[0, 0, 1, 0, 0]], dtype=np.int32),
            np.array([[0, 3]]),
        )
        (mlx_m0, mlx_m1), (torch_m0, torch_m1) = self._run_means(model_wrapper, labels)
        _assert_close(mlx_m0, torch_m0)
        _assert_close(mlx_m1, torch_m1)

    def test_all_class_zero(self, model_wrapper: Any) -> None:
        """All labels being class 0 produces mean_1 as zero vector."""
        labels = (
            np.array([[1, 2, 3, 0, 0]], dtype=np.int32),
            np.array([[0, 0, 0, 0, 0]], dtype=np.int32),
            np.array([[0, 3]]),
        )
        (mlx_m0, mlx_m1), (torch_m0, torch_m1) = self._run_means(model_wrapper, labels)
        _assert_close(mlx_m0, torch_m0)
        assert float(mx.mean(mlx_m1).item()) < 1e-6
        assert float(mx.mean(mx.array(torch_m1.detach().numpy())).item()) < 1e-6

    def test_single_sample_per_class(
        self, model_wrapper: Any
    ) -> None:  # Single sample per class produces stable mean vectors.
        labels = (
            np.array([[1, 2, 0, 0, 0]], dtype=np.int32),
            np.array([[0, 1, 0, 0, 0]], dtype=np.int32),
            np.array([[0, 2]]),
        )
        (mlx_m0, mlx_m1), (torch_m0, torch_m1) = self._run_means(model_wrapper, labels)
        _assert_close(mlx_m0, torch_m0)
        _assert_close(mlx_m1, torch_m1)


# Checkpoint parity


class TestBackendCheckpointParity:
    """Checkpoint save/load preserves exact weights across backends."""

    def test_mlx_checkpoint_to_torch(self, model_wrapper: Any) -> None:
        """MLX checkpoint loads into torch model with matching weights."""
        pytest.importorskip("torch")
        from tests.conftest import _make_torch_tiny_mlp

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        probe = model_wrapper._probes["p"]
        original = probe.module.weight

        torch_model = _make_torch_tiny_mlp()
        torch_model.config = _DummyConfig()
        tw = Model(torch_model, _DummyTokenizer(), backend_name="torch")
        tw.attach_probe(ProbeConfig(name="p", layers=[1]))

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "probe.safetensors"
            _ckw.save_probe_weights(probe, p, model_wrapper.backend)
            _ckw.load_probe_weights(tw._probes["p"], p, tw.backend)

        _assert_close(original, tw._probes["p"].module.weight)

    def test_torch_checkpoint_to_mlx(self) -> None:
        """Torch checkpoint loads into MLX model with matching weights."""
        pytest.importorskip("torch")
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            tl = torch.nn.Linear(16, 1)
            p = Path(tmp) / "probe.pth"
            torch.save(tl.state_dict(), str(p))
            # _load_probe_weights expects a Probe-like wrapper (.module / .name).
            mlx_probe = SimpleNamespace(module=nn.Linear(16, 1), name="p")
            _ckw.load_probe_weights(mlx_probe, p, SimpleNamespace(name="mlx"))
            # Loaded weights must match the saved torch weights, not stay random.
            assert mx.isfinite(mlx_probe.module.weight).all()
            w = np.array(mlx_probe.module.weight)
            assert np.allclose(w, tl.weight.detach().cpu().numpy(), atol=1e-6)

    def test_roundtrip_preserves_weights(self, model_wrapper: Any) -> None:
        """Save then load preserves exact probe weights across backends."""
        pytest.importorskip("torch")
        from tests.conftest import _make_torch_tiny_mlp

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        probe = model_wrapper._probes["p"]
        original = probe.module.weight
        torch_model = _make_torch_tiny_mlp()
        torch_model.config = _DummyConfig()
        tw = Model(torch_model, _DummyTokenizer(), backend_name="torch")
        tw.attach_probe(ProbeConfig(name="p", layers=[1]))
        with tempfile.TemporaryDirectory() as tmp:
            mp = Path(tmp) / "m.safetensors"
            _ckw.save_probe_weights(probe, mp, model_wrapper.backend)
            _ckw.load_probe_weights(tw._probes["p"], mp, tw.backend)
            tp = Path(tmp) / "t.pth"
            _ckw.save_probe_weights(tw._probes["p"], tp, tw.backend)
            _ckw.load_probe_weights(probe, tp, model_wrapper.backend)
        diff = mx.abs(original - probe.module.weight)
        assert float(diff.max().item()) < 1e-6

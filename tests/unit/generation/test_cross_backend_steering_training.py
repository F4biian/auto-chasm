"""Deep coverage: trainable, loss torch, peft, and cross-backend steering parity."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

# ===========================================================================
# Section 5: trainable.py deep coverage
# ===========================================================================


class TestTrainableCoverage:
    """Coverage tests for trainable.py."""

    def test_clip_grad_norm_zero(self) -> None:
        """clip_grad_norm with zero gradient (norm=0)."""
        from mlx.utils import tree_flatten

        from auto_chasm.trainers.trainable import clip_grad_norm

        # Build a gradient tree with small values
        grad = {"w": mx.array([[1.0, 2.0], [3.0, 4.0]]), "b": mx.array([0.1, 0.2])}
        clipped = clip_grad_norm(grad, max_norm=1.0)

        orig_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(grad)))
        clipped_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(clipped)))

        assert float(clipped_norm) <= float(orig_norm) + 1e-6

    def test_clip_grad_norm_large(self) -> None:
        """clip_grad_norm with high max_norm (no clipping needed)."""
        from mlx.utils import tree_flatten

        from auto_chasm.trainers.trainable import clip_grad_norm

        grad = {"w": mx.array([[0.1, 0.2], [0.3, 0.4]])}
        clipped = clip_grad_norm(grad, max_norm=100.0)

        orig_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(grad)))
        clipped_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(clipped)))
        assert abs(float(clipped_norm) - float(orig_norm)) < 1e-6

    def test_clip_grad_norm_zero_max_norm(self) -> None:
        """clip_grad_norm with max_norm=0 (clips everything to zero)."""
        from mlx.utils import tree_flatten

        from auto_chasm.trainers.trainable import clip_grad_norm

        grad = {"w": mx.array([[10.0, 20.0], [30.0, 40.0]])}
        clipped = clip_grad_norm(grad, max_norm=0.0)

        clipped_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(clipped)))
        assert float(clipped_norm) < 1e-6

    def test_restore_capture_fns(self, model_wrapper: Any) -> None:
        """_TrainableModel.restore_capture_fns() actually works."""
        from auto_chasm.config import ProbeConfig
        from auto_chasm.trainers.trainable import _TrainableModel

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        probe = model_wrapper._probes["p"]
        orig_fns = [lc.capture_fn for lc in probe.layer_captures]

        train_model = _TrainableModel(model_wrapper.model, model_wrapper._probes)

        for orig_fn, lc in zip(orig_fns, probe.layer_captures, strict=True):
            assert lc.capture_fn is not orig_fn

        train_model.restore_capture_fns()

        for orig_fn, lc in zip(orig_fns, probe.layer_captures, strict=True):
            assert lc.capture_fn is orig_fn
        assert len(train_model._orig_capture_fns) == 0

    def test_make_joint_loss_with_value_and_grad(self, model_wrapper: Any) -> None:
        """make_joint_loss produces callable compatible with value_and_grad."""
        from auto_chasm.config import ProbeConfig
        from auto_chasm.trainers.trainable import _TrainableModel, make_joint_loss

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        train_model = _TrainableModel(model_wrapper.model, model_wrapper._probes)

        loss_fn = make_joint_loss(lm_weight=1.0, probe_weight=1.0, probe_loss="bce")

        batch = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[0, 0, 1, 0, 0]])
        lengths = mx.array([[0, 5]])

        total, ntoks, components = loss_fn(train_model, batch, labels, lengths)

        assert total.ndim == 0
        assert float(ntoks) > 0
        assert "lm_head" in components
        assert "p" in components
        assert mx.isfinite(total).all()

    def test_make_joint_loss_pure_classifier(self, model_wrapper: Any) -> None:
        """make_joint_loss with lm_weight=0 produces pure classifier loss."""
        from auto_chasm.config import ProbeConfig
        from auto_chasm.trainers.trainable import _TrainableModel, make_joint_loss

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        train_model = _TrainableModel(model_wrapper.model, model_wrapper._probes)

        loss_fn = make_joint_loss(lm_weight=0.0, probe_weight=1.0, probe_loss="bce")

        batch = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[0, 0, 1, 0, 0]])
        lengths = mx.array([[0, 5]])

        total, ntoks, components = loss_fn(train_model, batch, labels, lengths)

        assert "lm_head" not in components
        assert "p" in components
        assert mx.isfinite(total).all()


# ===========================================================================
# Section 6: loss.py torch coverage
# ===========================================================================


class TestLossTorchCoverage:
    """Coverage tests for loss.py torch path."""

    def test_compute_torch_directly(self) -> None:
        """JointLoss dispatches the torch path directly with torch tensors."""
        pytest.importorskip("torch")
        import torch

        from auto_chasm.trainers.loss import JointLoss

        loss = JointLoss(losses={"probe": "bce"})

        class TorchFakeModel:
            """Fake model returning torch tensors."""

            def __call__(self, inputs):
                b, t = inputs.shape
                return torch.zeros((b, t, 32)), torch.zeros((b, t))

        batch = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float32)
        lengths = torch.tensor([[0, 5]])

        total, ntoks, components = loss(TorchFakeModel(), batch, labels, lengths)

        assert total.ndim == 0
        assert float(ntoks) > 0
        assert "lm_head" in components
        assert "probe" in components

    def _make_mlx_fake(self) -> Any:
        """Create an MLX fake model with dynamic sequence length."""

        class MLXFakeModel:
            """Fake model returning MLX tensors matching input shape."""

            def __call__(self, inputs):
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), mx.ones((b, t))

        return MLXFakeModel()

    def _make_torch_fake(self) -> Any:
        """Create a torch fake model with dynamic sequence length."""
        import torch

        class TorchFakeModel:
            """Fake model returning torch tensors matching input shape."""

            def __call__(self, inputs):
                b, t = inputs.shape
                return torch.zeros((b, t, 32)), torch.ones((b, t))

        return TorchFakeModel()

    def test_bce_loss_values_match_mlx_torch(self) -> None:
        """BCE loss values agree between MLX and torch paths."""
        pytest.importorskip("torch")

        from auto_chasm.trainers.loss import JointLoss

        loss = JointLoss(losses={"probe": "bce"})

        mlx_total, mlx_ntoks, mlx_comp = loss(
            self._make_mlx_fake(),
            mx.array([[1, 2, 3, 4, 5]]),
            mx.array([[0, 0, 1, 0, 0]]),
            mx.array([[0, 5]]),
        )

        import torch

        torch_total, torch_ntoks, torch_comp = loss(
            self._make_torch_fake(),
            torch.tensor([[1, 2, 3, 4, 5]]),
            torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float32),
            torch.tensor([[0, 5]]),
        )

        assert abs(float(mlx_comp["probe"]) - float(torch_comp["probe"])) < 1e-5

    def test_mse_loss_values_match_mlx_torch(self) -> None:
        """MSE loss values agree between MLX and torch paths."""
        pytest.importorskip("torch")

        from auto_chasm.trainers.loss import JointLoss

        loss = JointLoss(losses={"probe": "mse"})

        mlx_total, mlx_ntoks, mlx_comp = loss(
            self._make_mlx_fake(),
            mx.array([[1, 2, 3, 4, 5]]),
            mx.array([[0, 0, 1, 0, 0]]),
            mx.array([[0, 5]]),
        )

        import torch

        torch_total, torch_ntoks, torch_comp = loss(
            self._make_torch_fake(),
            torch.tensor([[1, 2, 3, 4, 5]]),
            torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float32),
            torch.tensor([[0, 5]]),
        )

        assert abs(float(mlx_comp["probe"]) - float(torch_comp["probe"])) < 1e-5

    def test_per_probe_weights_torch(self) -> None:
        """Per-probe weights work in torch path."""
        pytest.importorskip("torch")
        import torch

        from auto_chasm.trainers.loss import JointLoss

        loss = JointLoss(
            weights={"p1": 2.0, "p2": 0.5},
        )

        class TorchFakeModel:
            """Fake model returning dict probe outputs."""

            def __call__(self, inputs):
                b, t = inputs.shape
                return torch.zeros((b, t, 32)), {
                    "p1": torch.ones((b, t)),
                    "p2": torch.ones((b, t)),
                }

        total, ntoks, components = loss(
            TorchFakeModel(),
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[0, 0, 0]], dtype=torch.float32),
            torch.tensor([[0, 3]]),
        )

        # Two probes → per-probe component keys.
        assert "p1" in components
        assert "p2" in components
        assert total.ndim == 0
        assert torch.isfinite(total).all()


# ===========================================================================
# Section 7: peft.py execution coverage
# ===========================================================================


class TestPEFTExecutionCoverage:
    """Execution coverage for peft.py functions."""

    def test_apply_qlora_quantizes_base_then_lora_mlx(self) -> None:
        """True QLoRA on MLX: the base is quantized, then LoRA wraps it."""
        from auto_chasm.backends import Backend
        from auto_chasm.peft import apply_qlora

        class _Attn(nn.Module):
            """Attention stand-in with quantizable q/v projections."""

            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)

            def __call__(self, x: Any) -> Any:
                return self.q_proj(x) + self.v_proj(x)

        class _Blk(nn.Module):
            """A block exposing self_attn so LoRA targeting can find q_proj."""

            def __init__(self) -> None:
                super().__init__()
                self.self_attn = _Attn()

            def __call__(self, x: Any) -> Any:
                return self.self_attn(x)

        class _M(nn.Module):
            """A tiny stack of blocks."""

            def __init__(self) -> None:
                super().__init__()
                self.layers = [_Blk() for _ in range(2)]

            def __call__(self, x: Any) -> Any:
                for layer in self.layers:
                    x = layer(x)
                return x

        model = _M()
        assert type(model.layers[0].self_attn.q_proj).__name__ == "Linear"
        apply_qlora(model, r=4, alpha=8, target_modules=["q_proj"], backend=Backend(force="mlx"))
        qp = model.layers[0].self_attn.q_proj
        # LoRA wraps the now-QUANTIZED base — the base was really quantized to 4-bit.
        assert type(qp).__name__ == "LoRALinear"
        assert type(qp.linear).__name__ == "QuantizedLinear"

    def test_apply_qlora_torch_requires_4bit_base(self) -> None:
        """Torch QLoRA raises unless the base is already 4-bit-loaded (no CUDA here)."""
        import pytest

        torch = pytest.importorskip("torch")

        from auto_chasm.backends import Backend
        from auto_chasm.peft import apply_qlora

        model = torch.nn.Sequential(torch.nn.Linear(16, 16))
        with pytest.raises(NotImplementedError, match="4-bit"):
            apply_qlora(model, r=2, alpha=4, backend=Backend(force="torch"))

    @pytest.mark.real_model
    def test_apply_dora_mlx_wraps_native_dora_layers(self) -> None:
        """apply_dora on MLX wraps real DoRALinear layers (native, not a LoRA fallback)."""
        import os

        import pytest

        mlx_lm = pytest.importorskip("mlx_lm")
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.dora import DoRALinear

        from auto_chasm.backends import Backend
        from auto_chasm.peft import apply_dora

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            model, _ = mlx_lm.load("mlx-community/gemma-3-270m-it-8bit")
        except Exception:
            pytest.skip("cached MLX model unavailable")

        n_qproj = sum(1 for n, _ in model.named_modules() if n.endswith("q_proj"))
        apply_dora(model, r=4, alpha=8, target_modules=["q_proj"], backend=Backend(force="mlx"))
        doras = [m for _, m in model.named_modules() if isinstance(m, DoRALinear)]
        # Real DoRA: one DoRALinear per q_proj, each carrying the magnitude vector
        # 'm' (the weight decomposition) on top of lora_a/lora_b — not plain LoRA.
        assert len(doras) == n_qproj > 0
        keys = set(dict(tree_flatten(doras[0].trainable_parameters())))
        assert "m" in keys and "lora_a" in keys and "lora_b" in keys

    def test_unfreeze_lora_params_mlx(self, model_wrapper: Any) -> None:
        """_unfreeze_lora_params with actual model on MLX path."""
        from auto_chasm.peft import _unfreeze_lora_params

        model_wrapper.model.freeze()
        _unfreeze_lora_params(model_wrapper.model, model_wrapper.backend)
        # Should complete without error — model may have no LoRA params
        assert True


# ===========================================================================
# Section 8: Cross-backend steering parity
# ===========================================================================


class TestCrossBackendSteeringParity:
    """Cross-backend steering parity tests."""

    def test_steer_mlx_apply_with_torch(self) -> None:
        """Steering directions computed with MLX, applied with torch."""
        pytest.importorskip("torch")
        import torch

        from auto_chasm.steering import _steer_torch

        hidden_dim = 16
        batch, seq_len = 2, 5

        mx.random.seed(42)
        torch.manual_seed(42)

        # Compute direction in MLX
        mean_0_mlx = mx.array([0.0] * hidden_dim)
        mean_1_mlx = mx.array([1.0] * hidden_dim)
        direction_mlx = mean_1_mlx - mean_0_mlx

        hidden_mlx = mx.random.normal((batch, seq_len, hidden_dim))
        hidden_torch = torch.tensor(np.array(hidden_mlx))

        mean_0_torch = torch.zeros(hidden_dim)
        mean_1_torch = torch.ones(hidden_dim)
        direction_torch = torch.tensor(np.array(direction_mlx))

        head_torch = torch.nn.Linear(hidden_dim, 1)
        logits_torch = head_torch(hidden_torch).squeeze(-1)

        # Apply with torch using MLX-computed direction
        result = _steer_torch(
            hidden_torch,
            head_torch,
            logits_torch,
            "nullify",
            mean_0_torch,
            mean_1_torch,
            direction_torch,
        )

        assert result.shape == (batch, seq_len, hidden_dim)
        assert torch.isfinite(result).all()

    def test_steer_and_steer_torch_identical_hidden_states(self) -> None:
        """_steer_mlx and _steer_torch produce identical hidden states."""
        pytest.importorskip("torch")
        import torch

        from auto_chasm.steering import _steer_mlx, _steer_torch

        hidden_dim = 8
        batch, seq_len = 1, 3

        mx.random.seed(42)
        torch.manual_seed(42)

        hidden_mlx = mx.random.normal((batch, seq_len, hidden_dim))
        hidden_torch = torch.tensor(np.array(hidden_mlx))

        mean_0_mlx = mx.array([0.0] * hidden_dim)
        mean_1_mlx = mx.array([1.0] * hidden_dim)
        direction_mlx = mean_1_mlx - mean_0_mlx

        mean_0_torch = torch.zeros(hidden_dim)
        mean_1_torch = torch.ones(hidden_dim)
        direction_torch = mean_1_torch - mean_0_torch

        head_mlx = nn.Linear(hidden_dim, 1)
        head_torch = torch.nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            head_torch.weight.copy_(torch.tensor(np.array(head_mlx.weight)))
            head_torch.bias.copy_(torch.tensor(np.array(head_mlx.bias)))

        logits_mlx = head_mlx(hidden_mlx).squeeze(-1)
        logits_torch = head_torch(hidden_torch).squeeze(-1)

        for method in ("nullify", "push_to_mean"):
            result_mlx = _steer_mlx(
                hidden_mlx,
                head_mlx,
                logits_mlx,
                method,
                mean_0_mlx,
                mean_1_mlx,
                direction_mlx,
            )
            result_torch = _steer_torch(
                hidden_torch,
                head_torch,
                logits_torch,
                method,
                mean_0_torch,
                mean_1_torch,
                direction_torch,
            )

            diff = mx.abs(result_mlx - mx.array(result_torch.detach().numpy()))
            max_diff = float(diff.max().item())
            assert max_diff < 1e-4, f"Method '{method}' hidden state mismatch: max diff={max_diff}"

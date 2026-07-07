"""Tests for the MLX backend."""

from __future__ import annotations

import mlx.core as mx
import pytest

from auto_chasm.backends import Backend


class TestBackendAutoDetect:
    """Tests for backend auto-detection."""

    def test_auto_detect_mlx(self) -> None:
        backend = Backend()
        assert backend.name == "mlx"

    def test_force_mlx(self) -> None:
        backend = Backend(force="mlx")
        assert backend.name == "mlx"

    def test_has_all_interfaces(self) -> None:
        backend = Backend(force="mlx")
        assert hasattr(backend, "tensor")
        assert hasattr(backend, "module")
        assert hasattr(backend, "optim")
        assert hasattr(backend, "wrapping")


class TestMLXTensorOps:
    """Tests for MLX tensor operations."""

    def setup_method(self) -> None:
        self.backend = Backend(force="mlx")

    def test_tensor_creation(self) -> None:
        t = self.backend.tensor.tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)

    def test_tensor_dtype(self) -> None:
        t = self.backend.tensor.tensor([1, 2, 3], dtype=mx.float32)
        assert t.dtype == mx.float32

    def test_zeros(self) -> None:
        t = self.backend.tensor.zeros((3, 4))
        assert t.shape == (3, 4)
        assert float(mx.sum(t).item()) == 0.0

    def test_ones(self) -> None:
        t = self.backend.tensor.ones((2, 3))
        assert t.shape == (2, 3)

    def test_float32(self) -> None:
        dtype = self.backend.tensor.float32()
        assert dtype == mx.float32

    def test_stack(self) -> None:
        a = mx.array([1.0, 2.0])
        b = mx.array([3.0, 4.0])
        stacked = self.backend.tensor.stack([a, b], axis=0)
        assert stacked.shape == (2, 2)

    def test_concatenate(self) -> None:
        a = mx.array([[1.0, 2.0]])
        b = mx.array([[3.0, 4.0]])
        cat = self.backend.tensor.concatenate([a, b], axis=0)
        assert cat.shape == (2, 2)

    def test_mean(self) -> None:
        t = mx.array([1.0, 2.0, 3.0, 4.0])
        m = self.backend.tensor.mean(t)
        assert abs(float(m.item()) - 2.5) < 1e-6

    def test_matmul(self) -> None:
        a = mx.array([[1.0, 2.0], [3.0, 4.0]])
        b = mx.array([[5.0], [6.0]])
        result = self.backend.tensor.matmul(a, b)
        assert result.shape == (2, 1)


class TestMLXModuleOps:
    """Tests for MLX module operations."""

    def setup_method(self) -> None:
        self.backend = Backend(force="mlx")

    def test_linear_parameters(self) -> None:
        import mlx.nn as nn

        linear = nn.Linear(4, 2)
        params = self.backend.module.parameters(linear)
        assert len(params) > 0

    def test_named_parameters(self) -> None:
        import mlx.nn as nn

        linear = nn.Linear(4, 2)
        named = self.backend.module.named_parameters(linear)
        assert len(named) > 0

    def test_freeze_unfreeze(self) -> None:
        import mlx.nn as nn

        linear = nn.Linear(4, 2)
        self.backend.module.freeze(linear)
        self.backend.module.unfreeze(linear)

    def test_eval_train(self) -> None:
        import mlx.nn as nn

        linear = nn.Linear(4, 2)
        self.backend.module.eval(linear)
        self.backend.module.train(linear)


class TestMLXOptimOps:
    """Tests for MLX optimizer operations."""

    def setup_method(self) -> None:
        self.backend = Backend(force="mlx")

    def test_create_adamw(self) -> None:
        optim = self.backend.optim.create_adamw(learning_rate=1e-3)
        assert optim is not None

    def test_create_sgd(self) -> None:
        optim = self.backend.optim.create_sgd(learning_rate=1e-3)
        assert optim is not None

    def test_zero_grad_noop(self) -> None:
        optim = self.backend.optim.create_adamw(learning_rate=1e-3)
        self.backend.optim.zero_grad(optim)

    def test_scale_lr(self) -> None:
        optim = self.backend.optim.create_adamw(learning_rate=1e-3)
        self.backend.optim.scale_lr(optim, 0.5)


class TestNoUnconditionalMLXImports:
    """Ensure trainers don't import MLX unconditionally at module level.

    Note: Conditional ``try: import mlx.nn as nn`` patterns (as in
    ``trainable.py`` and ``probe.py``) are acceptable, mirroring the
    existing pattern.  This test verifies files that previously had
    completely unconditional ``import mlx`` at the top level.
    """

    def test_no_top_level_mlx_in_trainers(self) -> None:
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        files_to_check = [
            root / "src" / "auto_chasm" / "trainers" / "base.py",
            root / "src" / "auto_chasm" / "trainers" / "trainable.py",
        ]
        for filepath in files_to_check:
            tree = ast.parse(filepath.read_text())
            # Only check top-level nodes (not inside functions/classes).
            # try/except blocks with conditional imports are acceptable.
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Try):
                    continue
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("mlx"), (
                            f"Unconditional MLX import in {filepath}: import {alias.name}"
                        )
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and node.module.startswith("mlx")
                ):
                    raise AssertionError(
                        f"Unconditional MLX import in {filepath}: from {node.module} import ..."
                    )


def test_resolve_backend_name_validates_before_load() -> None:
    """A bad backend_name raises a clear ValueError (was a misleading post-load error)."""
    from auto_chasm.backends.loaders import resolve_backend_name

    for bad in ("pytorch", "MLX", "Torch", "cuda", ""):
        with pytest.raises(ValueError, match="Unknown backend_name"):
            resolve_backend_name(bad)
    assert resolve_backend_name("mlx") == "mlx"
    assert resolve_backend_name("torch") == "torch"

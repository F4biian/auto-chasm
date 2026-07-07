"""Backend protocol — the single place where MLX and PyTorch differ.

Every other module in auto-chasm talks to the framework through this
protocol.  Only four interfaces: tensor ops, module ops, optimizer ops,
and model wrapping (LoRA / adapters).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TensorOps(Protocol):
    """Low-level tensor creation and manipulation."""

    def tensor(self, data: Any, dtype: Any = None) -> Any:
        """Create a tensor from data (list, numpy array, etc.)."""
        ...

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create a zero tensor."""
        ...

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create a ones tensor."""
        ...

    def float32(self) -> Any:
        """Return the float32 dtype for this backend."""
        ...

    def to_numpy(self, tensor: Any) -> Any:
        """Convert tensor to numpy array."""
        ...

    def stack(self, tensors: list[Any], axis: int = 0) -> Any:
        """Stack tensors along a new axis."""
        ...

    def concatenate(self, tensors: list[Any], axis: int = 0) -> Any:
        """Concatenate tensors along an existing axis."""
        ...

    def mean(self, tensor: Any, axis: int | None = None) -> Any:
        """Compute mean along axis."""
        ...

    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiply."""
        ...

    def sample(self, logits: Any, temperature: float) -> int:
        """Sample a token index from logits.

        Args:
            logits: 1-D logit tensor.
            temperature: Sampling temperature (0 = greedy).

        Returns:
            Sampled token index.
        """
        ...


@runtime_checkable
class ModuleOps(Protocol):
    """Module-level operations (forward pass, parameter access)."""

    def forward(self, module: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a forward pass."""
        ...

    def parameters(self, module: Any) -> list[Any]:
        """Return all parameter tensors (flat list)."""
        ...

    def trainable_parameters(self, module: Any) -> list[Any]:
        """Return only trainable parameter tensors."""
        ...

    def named_parameters(self, module: Any) -> dict[str, Any]:
        """Return ``{name: tensor}`` for all parameters."""
        ...

    def freeze(self, module: Any) -> None:
        """Freeze all parameters (set requires_grad=False / zero grad filter)."""
        ...

    def unfreeze(self, module: Any) -> None:
        """Unfreeze all parameters."""
        ...

    def eval(self, module: Any) -> None:
        """Set module to evaluation mode."""
        ...

    def train(self, module: Any) -> None:
        """Set module to training mode."""
        ...


@runtime_checkable
class OptimOps(Protocol):
    """Optimizer creation, stepping, and gradient management."""

    def create_adamw(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
    ) -> Any:
        """Create an AdamW optimizer."""
        ...

    def create_sgd(self, learning_rate: float, momentum: float = 0.0) -> Any:
        """Create an SGD optimizer."""
        ...

    def step(
        self,
        optimizer: Any,
        model: Any,
        gradients: Any,
        trainable_params: list[Any] | None = None,
    ) -> Any:
        """Apply gradients and return updated (optimizer, model).

        In MLX this is ``optimizer.update(model, grads)``.
        In PyTorch this is ``optimizer.step(); optimizer.zero_grad()``.
        """
        ...

    def zero_grad(self, optimizer: Any) -> None:
        """Zero optimizer gradients (no-op in MLX)."""
        ...

    def scale_lr(self, optimizer: Any, scale: float) -> Any:
        """Scale the optimizer's learning rate by *scale*."""
        ...


@runtime_checkable
class ModelWrapping(Protocol):
    """Adapter (LoRA / QLoRA / DoRA) integration."""

    def apply_adapters(
        self,
        model: Any,
        adapter_config: dict[str, Any],
        target_modules: list[str],
        method: str = "lora",
    ) -> Any:
        """Wrap a model with parameter-efficient adapters.

        Args:
            model: The base model.
            adapter_config: PEFT config dict.
            target_modules: Module names to target.
            method: PEFT variant (``"lora"``, ``"qlora"``, ``"dora"``).
        """
        ...

    def save_adapters(self, model: Any, path: str) -> None:
        """Save adapter weights to *path*."""
        ...

    def load_adapters(self, model: Any, path: str) -> Any:
        """Load adapter weights from *path*."""
        ...

    def get_trainable_params(self, model: Any) -> list[Any]:
        """Return trainable parameter tensors after adapter wrapping."""
        ...

    def save_class_means(self, class_means: dict[str, Any], path: str) -> None:
        """Save class-mean vectors to a file."""
        ...

    def load_class_means(self, path: str) -> dict[str, Any]:
        """Load class-mean vectors from a file."""
        ...


class Backend:
    """Container for all backend operations.

    Users never instantiate this directly; use ``Backend()`` which
    auto-detects the available framework, or pass ``force="mlx"``
    / ``force="torch"`` to override.

    Attributes:
        tensor: Tensor creation and manipulation ops.
        module: Module-level ops (forward, params, freeze).
        optim: Optimizer ops.
        wrapping: Adapter integration ops.
        name: ``"mlx"`` or ``"torch"``.
    """

    tensor: TensorOps
    module: ModuleOps
    optim: OptimOps
    wrapping: ModelWrapping
    name: str

    def __init__(self, force: str | None = None) -> None:
        """Initialize backend ops by selecting MLX or PyTorch."""
        if force == "mlx" or (force is None and _mlx_available()):
            from auto_chasm.backends.mlx_backend import (
                MLXModelWrapping,
                MLXModuleOps,
                MLXOptimOps,
                MLXTensorOps,
            )

            self.tensor = MLXTensorOps()
            self.module = MLXModuleOps()
            self.optim = MLXOptimOps()
            self.wrapping = MLXModelWrapping()
            self.name = "mlx"
        elif force == "torch" or (force is None and _torch_available()):
            from auto_chasm.backends.torch_backend import (
                TorchModelWrapping,
                TorchModuleOps,
                TorchOptimOps,
                TorchTensorOps,
            )

            self.tensor = TorchTensorOps()
            self.module = TorchModuleOps()
            self.optim = TorchOptimOps()
            self.wrapping = TorchModelWrapping()
            self.name = "torch"
        else:
            raise RuntimeError("No supported backend found. Install 'mlx' (macOS) or 'torch'.")


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False

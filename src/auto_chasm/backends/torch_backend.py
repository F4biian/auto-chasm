"""PyTorch backend — implements all four backend interfaces on stock PyTorch.

``torch`` is imported lazily inside methods, so ``import auto_chasm`` never requires
PyTorch (the backend is instantiated only when a torch model is used).
"""

from __future__ import annotations

from typing import Any

from auto_chasm.logger import get_logger

logger = get_logger(__name__)


class TorchTensorOps:
    """Tensor creation and manipulation for PyTorch."""

    def tensor(self, data: Any, dtype: Any = None) -> Any:
        """Create a PyTorch tensor."""
        import torch

        if isinstance(data, torch.Tensor):
            if dtype is not None:
                return data.to(dtype=dtype)
            return data
        kwargs: dict[str, Any] = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch.tensor(data, **kwargs)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create a zero tensor."""
        import torch

        return torch.zeros(shape, dtype=dtype or torch.float32)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create a ones tensor."""
        import torch

        return torch.ones(shape, dtype=dtype or torch.float32)

    def float32(self) -> Any:
        """Return torch.float32."""
        import torch

        return torch.float32

    def to_numpy(self, tensor: Any) -> Any:
        """Convert tensor to numpy."""
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy()
        return tensor

    def stack(self, tensors: list[Any], axis: int = 0) -> Any:
        """Stack tensors."""
        import torch

        return torch.stack(tensors, dim=axis)

    def concatenate(self, tensors: list[Any], axis: int = 0) -> Any:
        """Concatenate tensors."""
        import torch

        return torch.cat(tensors, dim=axis)

    def mean(self, tensor: Any, axis: int | None = None) -> Any:
        """Compute mean.

        Integer/boolean inputs are upcast to float32 before reducing so the
        result is a well-defined float mean on both backends (stock torch
        raises ``mean(): could not infer output dtype`` for integer tensors,
        whereas MLX silently upcasts).

        Args:
            tensor: Input tensor.
            axis: Axis to reduce, or ``None`` to reduce over all elements.

        Returns:
            The (float) mean.
        """
        import torch

        if not torch.is_floating_point(tensor):
            tensor = tensor.to(torch.float32)
        if axis is None:
            return tensor.mean()
        return tensor.mean(dim=axis)

    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiply."""
        return a @ b

    def sample(self, logits: Any, temperature: float) -> int:
        """Sample a token index from logits.

        Args:
            logits: 1-D logit tensor.
            temperature: Sampling temperature (0 = greedy).

        Returns:
            Sampled token index.
        """
        import torch

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            return int(torch.multinomial(probs, 1).item())
        return int(torch.argmax(logits).item())


class TorchModuleOps:
    """Module operations for PyTorch."""

    def forward(self, module: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a forward pass."""
        return module(*args, **kwargs)

    def parameters(self, module: Any) -> list[Any]:
        """Return all parameters."""
        return list(module.parameters())

    def trainable_parameters(self, module: Any) -> list[Any]:
        """Return trainable parameters (requires_grad=True)."""
        return [p for p in module.parameters() if p.requires_grad]

    def named_parameters(self, module: Any) -> dict[str, Any]:
        """Return ``{name: tensor}`` for all parameters."""
        return dict(module.named_parameters())

    def freeze(self, module: Any) -> None:
        """Freeze all parameters."""
        for p in module.parameters():
            p.requires_grad = False

    def unfreeze(self, module: Any) -> None:
        """Unfreeze all parameters."""
        for p in module.parameters():
            p.requires_grad = True

    def eval(self, module: Any) -> None:
        """Set module to eval mode."""
        module.eval()

    def train(self, module: Any) -> None:
        """Set module to train mode."""
        module.train()


class TorchOptimOps:
    """Optimizer operations for PyTorch."""

    def create_adamw(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
    ) -> Any:
        """Create an AdamW optimizer.

        Starts with a dummy parameter; real model parameters are injected
        lazily by ``step()`` on the first training step.

        Args:
            learning_rate: Peak learning rate.
            weight_decay: Weight decay coefficient.

        Returns:
            A ``torch.optim.AdamW`` optimizer.
        """
        import torch

        dummy = torch.zeros(1, requires_grad=False)
        return torch.optim.AdamW([dummy], lr=learning_rate, weight_decay=weight_decay, fused=False)

    def create_sgd(self, learning_rate: float, momentum: float = 0.0) -> Any:
        """Create an SGD optimizer.

        Starts with a dummy parameter; real model parameters are injected
        lazily by ``step()`` on the first training step.

        Args:
            learning_rate: Learning rate.
            momentum: Momentum factor.

        Returns:
            A ``torch.optim.SGD`` optimizer.
        """
        import torch

        dummy = torch.zeros(1, requires_grad=False)
        return torch.optim.SGD([dummy], lr=learning_rate, momentum=momentum)

    def step(
        self,
        optimizer: Any,
        model: Any,
        _gradients: Any,
        _trainable_params: list[Any] | None = None,
        max_norm: float = 1.0,
    ) -> tuple[Any, Any]:
        """Apply gradients via optimizer.step().

        In PyTorch, gradients are already on the tensors via autograd,
        so ``optimizer.step()`` is sufficient.  If the optimizer was
        created with ``create_adamw`` (empty param list), model
        parameters are added lazily on the first call.

        Args:
            optimizer: The PyTorch optimizer.
            model: The model (returned unchanged).
            gradients: Unused (PyTorch stores grads on tensors).
            trainable_params: Unused.
            max_norm: Maximum gradient norm for clipping.

        Returns:
            Tuple of ``(optimizer, model)`` — both unchanged.
        """
        import torch

        # Lazy param injection: create_adamw/create_sgd start with a dummy.
        params = optimizer.param_groups[0]["params"]
        if len(params) == 1 and not params[0].requires_grad:
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer.param_groups[0]["params"] = trainable

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=max_norm
        )
        optimizer.step()
        optimizer.zero_grad()
        return optimizer, model

    def zero_grad(self, optimizer: Any) -> None:
        """Zero optimizer gradients."""
        optimizer.zero_grad()

    def scale_lr(self, optimizer: Any, scale: float) -> Any:
        """Scale the optimizer's learning rate.

        Args:
            optimizer: The PyTorch optimizer.
            scale: Multiplicative factor.

        Returns:
            The optimizer (modified in-place).
        """
        for pg in optimizer.param_groups:
            pg["lr"] = pg["lr"] * scale
        return optimizer


class TorchModelWrapping:
    """LoRA / adapter integration for PyTorch (via PEFT)."""

    def apply_adapters(
        self,
        model: Any,
        adapter_config: dict[str, Any],
        target_modules: list[str],
        method: str = "lora",
    ) -> Any:
        """Apply LoRA adapters via PEFT.

        Args:
            model: The PyTorch model.
            adapter_config: LoRA config (``r``, ``alpha``, ``dropout``, etc.).
            target_modules: Module names to apply LoRA to.
            method: PEFT variant (``"lora"``, ``"qlora"``, ``"dora"``).
                Set ``use_dora=True`` in the PEFT config when ``method="dora"``.

        Returns:
            PEFT-wrapped model.

        Raises:
            ValueError: If the model already has PEFT adapters applied.
        """
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        if isinstance(model, PeftModel) or any(
            "lora_" in name for name, _ in model.named_parameters()
        ):
            raise ValueError(
                "model already has adapters applied; remove them first "
                "(e.g. merge_and_unload() or reload the base) before applying LoRA/DoRA again."
            )

        use_dora = method == "dora"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=adapter_config.get("r", 8),
            lora_alpha=adapter_config.get("alpha", 16),
            lora_dropout=adapter_config.get("dropout", 0.0),
            target_modules=target_modules,
            bias="none",
            use_dora=use_dora,
        )
        return get_peft_model(model, peft_config)

    def save_adapters(self, model: Any, path: str) -> None:
        """Save LoRA adapter weights as a single file.

        Saves only LoRA-specific parameters (keys containing ``lora_``)
        so the format is consistent with the MLX backend's single-file
        approach.

        Args:
            model: The PEFT model.
            path: File path to save to.
        """
        import torch

        state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
        if not state:
            # No adapter params -> the model has no LoRA. Warn instead of silently
            # writing nothing (a later "adapter file missing" would be confusing).
            logger.warning(
                "save_adapters: no LoRA parameters found on the model; nothing to save. "
                "Was apply_lora()/attach_lora() called?"
            )
            return
        torch.save(state, path)

    def load_adapters(self, model: Any, path: str) -> Any:
        """Load adapter weights from a single file.

        Loads LoRA parameters with ``strict=False`` so that only the
        adapter keys are updated.

        Args:
            model: The base model (must already have LoRA applied).
            path: File path to load from.

        Returns:
            Model with loaded adapter weights.
        """
        import torch

        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model

    def get_trainable_params(self, model: Any) -> list[Any]:
        """Return trainable parameter tensors.

        Args:
            model: The PyTorch model.

        Returns:
            List of trainable parameter tensors.
        """
        return [p for p in model.parameters() if p.requires_grad]

    def save_class_means(self, class_means: dict[str, Any], path: str) -> None:
        """Save class-mean vectors.

        Args:
            class_means: ``{probe_name: {"mean_0": tensor, "mean_1": tensor}}``
                or flat ``{"mean_0": tensor, "mean_1": tensor}``.
            path: File path.
        """
        import torch

        flat: dict[str, Any] = {}
        for key, val in class_means.items():
            if isinstance(val, dict) and "mean_0" in val:
                flat[f"{key}_mean_0"] = val["mean_0"].cpu()
                flat[f"{key}_mean_1"] = val["mean_1"].cpu()
            else:
                flat[key] = val.cpu() if hasattr(val, "cpu") else val
        torch.save(flat, path)

    def load_class_means(self, path: str) -> dict[str, Any]:
        """Load class-mean vectors.

        Reconstructs nested ``{probe_name: {"mean_0": ..., "mean_1": ...}}``
        structure from the flat saved format, matching MLX behavior.

        Args:
            path: File path.

        Returns:
            Dict with nested or flat mean vectors.
        """
        import torch

        loaded = torch.load(path, map_location="cpu")
        if not isinstance(loaded, dict):
            return {"data": loaded}
        result: dict[str, Any] = {}
        for key, val in loaded.items():
            if key.endswith("_mean_0"):
                probe_name = key[: -len("_mean_0")]
                result.setdefault(probe_name, {})["mean_0"] = val
            elif key.endswith("_mean_1"):
                probe_name = key[: -len("_mean_1")]
                result.setdefault(probe_name, {})["mean_1"] = val
            else:
                result[key] = val
        return result

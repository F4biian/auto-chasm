"""MLX backend — implements all four backend interfaces for Apple Silicon.

Uses ``mlx.core``, ``mlx.nn``, ``mlx.optimizers``, and ``mlx_lm.tuner``
for LoRA integration.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from auto_chasm._mlx_compat import ensure_mlx_lm_compat
from auto_chasm.logger import get_logger

logger = get_logger(__name__)

#: Default ceiling for MLX's buffer cache, in bytes. MLX retains freed buffers of
#: every distinct shape it has seen so it can reuse them; with variable-length
#: sequences that set keeps growing, and NOTHING in this library ever released it.
#: A real consequence: a 0.5B-parameter MLX run drove a 64 GB Mac to 0.4 GB free
#: and had to be killed, while `ps` RSS reported 2.7 GB -- Metal's unified memory
#: is invisible to RSS, so neither the process nor any RSS-based watchdog can see
#: it coming. Capping the cache bounds the growth; measured peak with a 1 GiB cap
#: was ~5 GB for that same run.
#:
#: Override with AUTO_CHASM_MLX_CACHE_LIMIT_GB (0 disables the cap entirely).
_DEFAULT_CACHE_LIMIT_GB = 4.0


def configure_mlx_memory() -> None:
    """Bound MLX's buffer cache. Idempotent; safe to call more than once.

    Reads AUTO_CHASM_MLX_CACHE_LIMIT_GB (float GiB; ``0`` = no cap) and
    AUTO_CHASM_MLX_MEMORY_LIMIT_GB (float GiB; unset = no hard limit).
    """
    import os

    try:
        cache_gb = float(os.environ.get("AUTO_CHASM_MLX_CACHE_LIMIT_GB", _DEFAULT_CACHE_LIMIT_GB))
        if cache_gb > 0:
            mx.set_cache_limit(int(cache_gb * 1024**3))
        mem_gb = os.environ.get("AUTO_CHASM_MLX_MEMORY_LIMIT_GB")
        if mem_gb:
            mx.set_memory_limit(int(float(mem_gb) * 1024**3))
    except Exception as exc:  # noqa: BLE001 -- a memory hint must never break a run
        logger.warning("Could not configure MLX memory limits: %s", exc)


# Leaf parameter names that belong to a LoRA/DoRA adapter (never the wrapped base):
# the low-rank factors and DoRA's magnitude vector.
_ADAPTER_LEAVES = frozenset({"lora_a", "lora_b", "m"})


def _lora_module_types() -> tuple[type, ...]:
    """Return the mlx_lm adapter module classes available in this install."""
    ensure_mlx_lm_compat()
    types: list[type] = []
    try:
        from mlx_lm.tuner.lora import LoRALinear, LoRASwitchLinear

        types += [LoRALinear, LoRASwitchLinear]
    except ImportError:
        pass
    try:
        from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear

        types += [DoRALinear, DoRAEmbedding]
    except ImportError:
        pass
    return tuple(types)


class MLXTensorOps:
    """Tensor creation and manipulation for MLX."""

    def tensor(self, data: Any, dtype: Any = None) -> mx.array:
        """Create an MLX array from data."""
        arr = mx.array(data)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> mx.array:
        """Create a zero array."""
        return mx.zeros(shape, dtype=dtype or mx.float32)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> mx.array:
        """Create a ones array."""
        return mx.ones(shape, dtype=dtype or mx.float32)

    def float32(self) -> Any:
        """Return MLX float32 dtype."""
        return mx.float32

    def to_numpy(self, tensor: mx.array) -> Any:
        """Convert MLX array to numpy."""
        import numpy as np

        return np.array(tensor)

    def stack(self, tensors: list[mx.array], axis: int = 0) -> mx.array:
        """Stack arrays along a new axis."""
        return mx.stack(tensors, axis=axis)

    def concatenate(self, tensors: list[mx.array], axis: int = 0) -> mx.array:
        """Concatenate arrays along an existing axis."""
        return mx.concatenate(tensors, axis=axis)

    def mean(self, tensor: mx.array, axis: int | None = None) -> mx.array:
        """Compute mean along axis.

        Integer inputs are upcast to float32 before reducing so the result is a
        well-defined float mean on both backends (torch refuses to infer an
        output dtype for integer ``mean``).

        Args:
            tensor: Input array.
            axis: Axis to reduce, or ``None`` to reduce over all elements.

        Returns:
            The (float) mean.
        """
        if tensor.dtype in (
            mx.int8,
            mx.int16,
            mx.int32,
            mx.int64,
            mx.uint8,
            mx.uint16,
            mx.uint32,
            mx.uint64,
            mx.bool_,
        ):
            tensor = tensor.astype(mx.float32)
        return mx.mean(tensor, axis=axis)

    def matmul(self, a: mx.array, b: mx.array) -> mx.array:
        """Matrix multiply."""
        return mx.matmul(a, b)

    def sample(self, logits: mx.array, temperature: float) -> int:
        """Sample a token index from logits.

        Args:
            logits: 1-D logit tensor.
            temperature: Sampling temperature (0 = greedy).

        Returns:
            Sampled token index.
        """
        if temperature > 0:
            return int(mx.random.categorical(logits / temperature).item())
        return int(mx.argmax(logits).item())


class MLXModuleOps:
    """Module operations for MLX (forward, parameters, freeze)."""

    def forward(self, module: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a forward pass on an MLX nn.Module."""
        return module(*args, **kwargs)

    def parameters(self, module: nn.Module) -> list[mx.array]:
        """Return all parameter arrays as a flat list."""
        return [v for _, v in tree_flatten(module.parameters())]

    def trainable_parameters(self, module: nn.Module) -> list[mx.array]:
        """Return trainable parameter arrays, respecting MLX's frozen state.

        Uses MLX's native ``module.trainable_parameters()`` so that frozen
        parameters (e.g. after ``freeze()``) are excluded — returning
        ``parameters()`` here would report a frozen base as fully trainable.

        Args:
            module: The MLX module.

        Returns:
            List of trainable parameter arrays.
        """
        return [v for _, v in tree_flatten(module.trainable_parameters())]

    def named_parameters(self, module: nn.Module) -> dict[str, mx.array]:
        """Return ``{name: array}`` for all parameters."""
        flat = tree_flatten(module.parameters())
        return dict(flat)

    def freeze(self, module: nn.Module) -> None:
        """Freeze all parameters by stopping gradients."""
        module.freeze()

    def unfreeze(self, module: nn.Module) -> None:
        """Unfreeze all parameters."""
        module.unfreeze()

    def eval(self, module: nn.Module) -> None:
        """Set module to evaluation mode (no-op in MLX by default)."""
        module.eval()

    def train(self, module: nn.Module) -> None:
        """Set module to training mode."""
        module.train()


class MLXOptimOps:
    """Optimizer operations for MLX."""

    def create_adamw(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
    ) -> optim.Adam:
        """Create an AdamW optimizer.

        Args:
            learning_rate: Peak learning rate.
            weight_decay: Weight decay coefficient.

        Returns:
            An ``mlx.optimizers.AdamW`` with the given settings.
        """
        return optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    def create_sgd(self, learning_rate: float, momentum: float = 0.0) -> optim.SGD:
        """Create an SGD optimizer.

        Args:
            learning_rate: Learning rate.
            momentum: Momentum factor.

        Returns:
            An ``mlx.optimizers.SGD`` with the given settings.
        """
        return optim.SGD(learning_rate=learning_rate, momentum=momentum)

    def step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        gradients: Any,
        _trainable_params: list[Any] | None = None,
    ) -> tuple[optim.Optimizer, nn.Module]:
        """Apply gradients via functional optimizer update.

        Args:
            optimizer: The MLX optimizer.
            model: The MLX model (functional update returns new state).
            gradients: Gradient tree (same structure as model parameters).
            trainable_params: Unused in MLX (gradient tree already aligned).

        Returns:
            Tuple of ``(updated_optimizer, updated_model)``.
        """
        new_optimizer, new_model = optimizer.update(model, gradients)
        return new_optimizer, new_model

    def zero_grad(self, _optimizer: optim.Optimizer) -> None:
        """No-op — MLX has no persistent gradient buffers."""
        return None

    def scale_lr(self, optimizer: optim.Optimizer, scale: float) -> optim.Optimizer:
        """Scale the optimizer's learning rate.

        Args:
            optimizer: The MLX optimizer.
            scale: Multiplicative factor.

        Returns:
            The optimizer with scaled learning rate.
        """
        optimizer.learning_rate = optimizer.learning_rate * scale
        return optimizer


class MLXModelWrapping:
    """LoRA / adapter integration for MLX via mlx_lm."""

    def apply_adapters(
        self,
        model: Any,
        adapter_config: dict[str, Any],
        target_modules: list[str],
        method: str = "lora",
    ) -> Any:
        """Apply LoRA or DoRA adapters to an MLX model.

        Args:
            model: The MLX model to wrap.
            adapter_config: LoRA config (``r``, ``alpha``, ``dropout``, etc.).
            target_modules: Module names to apply LoRA to.
            method: PEFT variant. ``"dora"`` uses mlx_lm's native weight-decomposed
                DoRA layers (``use_dora=True``); ``"lora"``/``"qlora"`` use LoRA.

        Returns:
            The model with LoRA/DoRA layers applied in-place.

        Raises:
            ValueError: If the model already has LoRA/DoRA adapters applied.
        """
        ensure_mlx_lm_compat()
        from mlx_lm.tuner.utils import linear_to_lora_layers

        # Cover every adapter module kind (linear, switch, embedding; LoRA and DoRA) so
        # a re-apply on any of them raises this clean error, not a muddier downstream one.
        if any(isinstance(m, _lora_module_types()) for _, m in model.named_modules()):
            raise ValueError(
                "model already has LoRA/DoRA adapters applied; remove them first "
                "before applying adapters again."
            )

        r = adapter_config.get("r", 8)
        alpha = adapter_config.get("alpha", 16)
        num_layers = adapter_config.get("num_layers", -1)
        lora_config = {
            "rank": r,
            "scale": alpha / r,
            "dropout": adapter_config.get("dropout", 0.0),
            "keys": target_modules,
        }
        # Freeze the base BEFORE wrapping so only the adapter params are trainable,
        # matching torch's get_peft_model (and mlx_lm's own tuner flow). Without this
        # the MLX base stayed trainable: LoRA training also updated the base, and
        # get_trainable_params/save_adapters saw base weights. linear_to_lora_layers
        # keeps the frozen base linear inside each adapter and adds fresh (trainable)
        # low-rank factors, so the adapters remain trainable.
        model.freeze()
        linear_to_lora_layers(model, num_layers, lora_config, use_dora=method == "dora")
        return model

    def save_adapters(self, model: nn.Module, path: str) -> None:
        """Save only the LoRA/DoRA adapter weights (matching the torch backend).

        ``model.trainable_parameters()`` also includes the base weights whenever the
        base is left unfrozen (nothing freezes it after ``apply_adapters`` on MLX), so
        it is NOT an "adapters only" filter — unlike torch's ``"lora_" in k``. That
        divergence meant an MLX adapter file silently carried the whole base. Collect
        the adapter params straight from the adapter modules instead: the LoRA factors
        (``lora_a``/``lora_b``) plus DoRA's magnitude (``m``), never the wrapped base.

        Args:
            model: The model with adapters applied.
            path: File path to save to.
        """
        from mlx.utils import tree_flatten

        adapter_types = _lora_module_types()
        adapter_dict: dict[str, mx.array] = {}
        for name, module in model.named_modules():
            if not isinstance(module, adapter_types):
                continue
            mod: Any = module
            prefix = f"{name}." if name else ""
            for key, value in tree_flatten(mod.parameters()):
                if key.rsplit(".", 1)[-1] in _ADAPTER_LEAVES:
                    adapter_dict[f"{prefix}{key}"] = value
        if not adapter_dict:
            # No adapter params -> the model has no LoRA (matches the torch warning).
            logger.warning(
                "save_adapters: no LoRA/DoRA parameters found on the model; nothing to "
                "save. Was apply_lora()/attach_lora() called?"
            )
        mx.save_safetensors(path, adapter_dict)

    def load_adapters(self, model: nn.Module, path: str) -> nn.Module:
        """Load adapter weights into a model.

        Uses ``Module.load_weights(..., strict=False)`` — the adapter file holds
        only the LoRA params (a subset of the model tree), and the flat dotted
        keys (``model.layers.9.self_attn.v_proj.lora_a``) must be matched against
        the nested module tree. The previous ``model.update(mx.load(path))`` fed
        a flat dict to ``update`` (which expects a nested tree) and silently
        failed to apply the adapters.

        Args:
            model: The model to load adapters into (LoRA already applied).
            path: Path to the adapter safetensors file.

        Returns:
            The model with loaded adapter weights.

        Raises:
            ValueError: If none of the adapter file's keys exist in the model
                (a wrong base/LoRA config) — so a mismatch is loud, not a no-op.
        """
        from mlx.utils import tree_flatten

        file_keys = set(mx.load(path).keys())
        model_keys = {k for k, _ in tree_flatten(model.parameters())}
        if file_keys and file_keys.isdisjoint(model_keys):
            raise ValueError(
                f"Adapter file {path!r} has no parameters matching the model "
                f"(e.g. {next(iter(file_keys))!r}). The LoRA config or base model "
                "does not match what produced these adapters."
            )
        model.load_weights(path, strict=False)
        return model

    def get_trainable_params(self, model: nn.Module) -> list[mx.array]:
        """Return trainable parameter arrays.

        Args:
            model: The MLX model.

        Returns:
            List of trainable parameter arrays.
        """
        return [v for _, v in tree_flatten(model.trainable_parameters())]

    def save_class_means(self, class_means: dict[str, Any], path: str) -> None:
        """Save class-mean vectors as safetensors.

        Args:
            class_means: ``{probe_name: {"mean_0": arr, "mean_1": arr}}``
                or flat ``{"mean_0": arr, "mean_1": arr}``.
            path: File path.
        """
        flat: dict[str, mx.array] = {}
        for key, val in class_means.items():
            if isinstance(val, dict) and "mean_0" in val:
                flat[f"{key}_mean_0"] = val["mean_0"]
                flat[f"{key}_mean_1"] = val["mean_1"]
            else:
                flat[key] = val
        mx.save_safetensors(path, flat)

    def load_class_means(self, path: str) -> dict[str, Any]:
        """Load class-mean vectors from safetensors.

        Handles both flat (``{"probe_mean_0": ...}``) and nested
        (``{"probe": {"mean_0": ...}}``) formats.

        Args:
            path: File path.

        Returns:
            Nested dict ``{probe_name: {"mean_0": arr, "mean_1": arr}}``
            or flat ``{"mean_0": arr, "mean_1": arr}``.
        """
        loaded = mx.load(path)
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

        if len(result) == 2 and "mean_0" in result and "mean_1" in result:
            return result
        return result

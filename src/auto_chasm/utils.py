"""Shared utilities — key cleaning, class means, parameter helpers.

These are the small but critical functions extracted from playground code.
"""

from __future__ import annotations

from typing import Any

from auto_chasm.logger import get_logger

logger = get_logger(__name__)


def tensor_backend(x: Any) -> str:
    """Return the backend of a tensor by its concrete type.

    Dispatch must depend on what kind of tensor ``x`` actually is, never
    on whether a framework happens to be importable.  On a machine with
    both MLX and PyTorch installed, ``try: import mlx ... except ImportError``
    always takes the MLX branch and silently mishandles torch tensors —
    the recurring cross-backend bug this helper exists to prevent.

    Args:
        x: A tensor (``mlx.core.array`` or ``torch.Tensor``).

    Returns:
        ``"torch"`` or ``"mlx"``.
    """
    module = type(x).__module__
    if module.startswith("torch"):
        return "torch"
    if module.startswith("mlx"):
        return "mlx"
    # Fallback heuristic: only torch tensors carry a ``.device`` attribute.
    return "torch" if hasattr(x, "device") else "mlx"


def clean_adapter_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip the ``layers.`` path segment from adapter state-dict keys.

    Some adapter savers (mlx-tune) emit keys like
    ``model.layers.3.self_attn.q_proj.lora_a`` while a consumer expects
    ``model.3.self_attn.q_proj.lora_a``.  This normalizes both the
    ``model.layers.`` and the legacy ``model.layer.`` spellings to
    ``model.``.

    Args:
        state_dict: Raw state dict from checkpoint.

    Returns:
        State dict with the ``layers.``/``layer.`` segment removed.
    """
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.layers.", "model.")
        new_key = new_key.replace("model.layer.", "model.")
        cleaned[new_key] = value
    return cleaned


def compute_class_means(
    hidden_states_by_class: dict[int, list[Any]],
) -> dict[int, Any]:
    """Compute mean hidden-state vectors per class.

    Used by steering methods (``push_to_mean``, ``boundary``, ``nullify``)
    to find the class-mean direction in activation space.

    Args:
        hidden_states_by_class: ``{class_label: [hidden_state_arrays]}``.

    Returns:
        ``{class_label: mean_vector}`` for each class.
    """
    means: dict[int, Any] = {}
    for cls, states in hidden_states_by_class.items():
        if not states:
            continue
        if tensor_backend(states[0]) == "torch":
            import torch

            stacked = torch.stack(states, dim=0)
            means[cls] = torch.mean(stacked, dim=0)
        else:
            import mlx.core as mx

            stacked = mx.stack(states, axis=0)
            means[cls] = mx.mean(stacked, axis=0)
    return means


def count_parameters(module: Any) -> int:
    """Count total parameters in a module.

    Args:
        module: An ``nn.Module`` (MLX or PyTorch).

    Returns:
        Total parameter count.
    """
    params = module.parameters()
    # MLX ``nn.Module.parameters()`` returns a nested dict; torch a generator/list of
    # tensors. Flatten to a tensor list, then count per-tensor by its ACTUAL backend
    # (never by which framework is importable — that mishandles torch when MLX is also
    # installed; ``tensor_backend`` exists to prevent exactly that).
    if isinstance(params, dict):
        from mlx.utils import tree_flatten

        arrays = [arr for _, arr in tree_flatten(params)]
    else:
        arrays = list(params)
    return sum(int(a.numel()) if tensor_backend(a) == "torch" else int(a.size) for a in arrays)


def freeze_module(module: Any) -> None:
    """Freeze all parameters in a module.

    Args:
        module: The module to freeze.
    """
    try:
        module.freeze()
    except AttributeError:
        for p in module.parameters():
            p.requires_grad = False


def unfreeze_module(module: Any) -> None:
    """Unfreeze all parameters in a module.

    Args:
        module: The module to unfreeze.
    """
    try:
        module.unfreeze()
    except AttributeError:
        for p in module.parameters():
            p.requires_grad = True

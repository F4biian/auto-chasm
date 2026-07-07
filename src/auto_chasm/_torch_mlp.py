"""PyTorch probe-head modules.

Imported lazily, only when a probe head is built on the torch backend, so
``import auto_chasm`` never requires PyTorch.  Mirrors :mod:`auto_chasm._mlx_mlp`:
the configurable ``TorchSequentialHead`` and the ``_TorchNormThenModule``
LayerNorm wrapper used by :mod:`auto_chasm.modules`, plus the built-in
``"linear"``/``"mlp"`` head builders (kept byte-identical for checkpoint
compatibility, with the ``fc1``/``fc2`` key names matching the MLX heads).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


def _resolve_torch_activation(activation: str | Any) -> Any:
    """Resolve an activation name (or pass through a callable) for torch.

    Args:
        activation: One of ``"relu"``, ``"gelu"``, ``"tanh"``, ``"silu"``,
            ``"identity"``, or a callable ``(tensor) -> tensor``.

    Returns:
        A callable activation function.

    Raises:
        ValueError: If ``activation`` is an unknown name.
    """
    if callable(activation) and not isinstance(activation, str):
        return activation
    table = {
        "relu": torch.relu,
        "gelu": torch.nn.functional.gelu,
        "tanh": torch.tanh,
        "silu": torch.nn.functional.silu,
        "identity": lambda x: x,
    }
    if activation in table:
        return table[activation]
    raise ValueError(
        f"Unknown activation {activation!r}. Use one of {sorted(table)} or pass a callable."
    )


class TorchSequentialHead(nn.Module):  # type: ignore[misc]
    """A configurable linear/MLP head for torch.

    Args:
        dims: Layer dimensions ``[in, *hidden, out]`` (one ``Linear`` per
            consecutive pair).
        activation: Activation callable applied between hidden layers.
        dropout: Dropout probability applied after each activation.
        input_layer_norm: If ``True``, a ``LayerNorm(dims[0])`` runs first.
        bias: Whether the linear layers use a bias.
    """

    def __init__(
        self,
        dims: list[int],
        activation: Any,
        dropout: float = 0.0,
        input_layer_norm: bool = False,
        bias: bool = True,
    ) -> None:
        """Build the linear stack, optional input norm, and dropout."""
        super().__init__()
        self.norm = nn.LayerNorm(dims[0]) if input_layer_norm else None
        self.linears = nn.ModuleList(
            [nn.Linear(a, b, bias=bias) for a, b in zip(dims[:-1], dims[1:], strict=True)]
        )
        self.dropout = nn.Dropout(dropout)
        self._activation = activation

    def forward(self, x: Any) -> Any:
        """Forward pass: optional norm, then linear/act/dropout per hidden layer."""
        if self.norm is not None:
            x = self.norm(x)
        last = len(self.linears) - 1
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < last:
                x = self.dropout(self._activation(x))
        return x


class _TorchNormThenModule(nn.Module):  # type: ignore[misc]
    """Run a ``LayerNorm`` before an inner head (torch).

    Args:
        inner: The wrapped head module.
        in_features: Feature dimension the LayerNorm normalizes over.
    """

    def __init__(self, inner: Any, in_features: int) -> None:
        """Build the input LayerNorm and store the inner module."""
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.inner = inner

    def forward(self, x: Any) -> Any:
        """Normalize the input, then run the inner head."""
        return self.inner(self.norm(x))


def build_torch_head(
    dims: list[int],
    activation: str | Any,
    dropout: float,
    input_layer_norm: bool,
    bias: bool,
) -> Any:
    """Build a :class:`TorchSequentialHead` (resolving the activation)."""
    return TorchSequentialHead(
        dims, _resolve_torch_activation(activation), dropout, input_layer_norm, bias
    )


def build_torch_linear(in_features: int, out_features: int) -> Any:
    """Build the built-in ``"linear"`` head on torch (near-zero init when width 1)."""
    module = nn.Linear(in_features, out_features)
    if out_features == 1:
        with torch.no_grad():
            module.weight.mul_(1e-4)
            module.bias.zero_()
    return module


def build_torch_builtin_mlp(
    in_features: int, out_features: int, hidden_dim: int, dropout: float
) -> Any:
    """Build the built-in ``"mlp"`` head on torch (``fc1``/``fc2`` matching MLX)."""
    return nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(in_features, hidden_dim)),
                ("act", nn.GELU()),
                ("dropout", nn.Dropout(dropout)),
                ("fc2", nn.Linear(hidden_dim, out_features)),
            ]
        )
    )


def wrap_torch_layer_norm(inner: Any, in_features: int) -> Any:
    """Wrap ``inner`` so a ``LayerNorm(in_features)`` runs before it (torch)."""
    return _TorchNormThenModule(inner, in_features)

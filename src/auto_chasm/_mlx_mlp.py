"""MLX probe-head modules (``mlx.nn`` has no ``Sequential``).

Imported lazily, only when a probe head is built on the MLX backend, so
``import auto_chasm`` never requires MLX.  Holds the legacy two-layer ``MLXMlp``
(kept byte-identical for checkpoint compatibility) plus the configurable
``MLXSequential`` and the ``_NormThenModule`` LayerNorm wrapper used by
:mod:`auto_chasm.modules`.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


class MLXMlp(nn.Module):  # type: ignore[misc]
    """A small two-layer GELU MLP for MLX probe heads.

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output dimension.
        dropout: Dropout probability.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        """Build the two linear layers and dropout."""
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: Any) -> Any:
        """Forward pass through the MLP."""
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.dropout(x)
        return self.fc2(x)


def _resolve_mlx_activation(activation: str | Any) -> Any:
    """Resolve an activation name (or pass through a callable) for MLX.

    Args:
        activation: One of ``"relu"``, ``"gelu"``, ``"tanh"``, ``"silu"``,
            ``"identity"``, or a callable ``(array) -> array``.

    Returns:
        A callable activation function.

    Raises:
        ValueError: If ``activation`` is an unknown name.
    """
    if callable(activation) and not isinstance(activation, str):
        return activation
    table = {
        "relu": nn.relu,
        "gelu": nn.gelu,
        "tanh": mx.tanh,
        "silu": nn.silu,
        "identity": lambda x: x,
    }
    if activation in table:
        return table[activation]
    raise ValueError(
        f"Unknown activation {activation!r}. Use one of {sorted(table)} or pass a callable."
    )


class MLXSequential(nn.Module):  # type: ignore[misc]
    """A configurable linear/MLP head for MLX.

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
        if input_layer_norm:
            self.norm = nn.LayerNorm(dims[0])
        self.linears = [
            nn.Linear(a, b, bias=bias) for a, b in zip(dims[:-1], dims[1:], strict=True)
        ]
        self.dropout = nn.Dropout(dropout)
        # Underscore: a plain callable, intentionally excluded from the MLX
        # parameter tree (it is not a trainable parameter).
        self._activation = activation

    def __call__(self, x: Any) -> Any:
        """Forward pass: optional norm, then linear/act/dropout per hidden layer."""
        norm = getattr(self, "norm", None)
        if norm is not None:
            x = norm(x)
        last = len(self.linears) - 1
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < last:
                x = self.dropout(self._activation(x))
        return x


class _NormThenModule(nn.Module):  # type: ignore[misc]
    """Run a ``LayerNorm`` before an inner head (MLX).

    Args:
        inner: The wrapped head module.
        in_features: Feature dimension the LayerNorm normalizes over.
    """

    def __init__(self, inner: Any, in_features: int) -> None:
        """Build the input LayerNorm and store the inner module."""
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.inner = inner

    def __call__(self, x: Any) -> Any:
        """Normalize the input, then run the inner head."""
        return self.inner(self.norm(x))


def build_mlx_head(
    dims: list[int],
    activation: str | Any,
    dropout: float,
    input_layer_norm: bool,
    bias: bool,
) -> Any:
    """Build an :class:`MLXSequential` head (resolving the activation)."""
    return MLXSequential(dims, _resolve_mlx_activation(activation), dropout, input_layer_norm, bias)


def build_mlx_linear(in_features: int, out_features: int) -> Any:
    """Build the built-in ``"linear"`` head on MLX (near-zero init when width 1)."""
    module = nn.Linear(in_features, out_features)
    if out_features == 1:
        module.weight = module.weight * 1e-4
        module.bias = module.bias * 0.0
    return module


def build_mlx_builtin_mlp(
    in_features: int, out_features: int, hidden_dim: int, dropout: float
) -> Any:
    """Build the built-in ``"mlp"`` head on MLX (the legacy two-layer ``MLXMlp``)."""
    return MLXMlp(in_features, hidden_dim, out_features, dropout)


def wrap_mlx_layer_norm(inner: Any, in_features: int) -> Any:
    """Wrap ``inner`` so a ``LayerNorm(in_features)`` runs before it (MLX)."""
    return _NormThenModule(inner, in_features)

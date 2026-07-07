"""Backend-agnostic probe-head construction.

:class:`ModuleSpec` is a declarative, callable head specification that slots
straight into ``ProbeConfig.module_type``.  You describe a head — depth, widths,
activation, dropout, input LayerNorm — and the library builds the concrete
``mlx``/``torch`` module.  The user never imports a framework and never branches
on the backend; the backend is threaded in for you.  Power users can still pass
any ``(in_features, cfg) -> module`` callable for a fully custom architecture,
so nothing here caps what is expressible.

This module imports no framework at top level (``import auto_chasm`` works with
neither installed); all framework construction is delegated lazily to
:mod:`auto_chasm._mlx_mlp` / :mod:`auto_chasm._torch_mlp`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


def build_module(
    in_features: int,
    out_features: int,
    *,
    hidden_dims: Sequence[int] = (),
    activation: str | Callable[..., Any] = "relu",
    dropout: float = 0.0,
    input_layer_norm: bool = False,
    bias: bool = True,
    backend: str = "mlx",
) -> Any:
    """Build a concrete linear/MLP head for the given backend.

    Args:
        in_features: Input feature dimension.
        out_features: Output dimension (e.g. number of classes).
        hidden_dims: Hidden layer widths; ``()`` builds a single ``Linear``.
        activation: Activation name (``"relu"``/``"gelu"``/``"tanh"``/``"silu"``/
            ``"identity"``) or a callable applied between hidden layers.
        dropout: Dropout probability after each activation.
        input_layer_norm: If ``True``, a ``LayerNorm(in_features)`` runs first.
        bias: Whether the linear layers carry a bias.
        backend: ``"mlx"`` or ``"torch"``.

    Returns:
        A framework ``nn.Module`` head.
    """
    dims = [in_features, *hidden_dims, out_features]
    if backend == "torch":
        from auto_chasm._torch_mlp import build_torch_head

        return build_torch_head(dims, activation, dropout, input_layer_norm, bias)
    from auto_chasm._mlx_mlp import build_mlx_head

    return build_mlx_head(dims, activation, dropout, input_layer_norm, bias)


def wrap_with_layer_norm(inner: Any, in_features: int, backend: str) -> Any:
    """Wrap ``inner`` so a ``LayerNorm(in_features)`` runs before it.

    Args:
        inner: The head module to wrap.
        in_features: Feature dimension the LayerNorm normalizes over.
        backend: ``"mlx"`` or ``"torch"``.

    Returns:
        A module that applies the LayerNorm and then ``inner``.
    """
    if backend == "torch":
        from auto_chasm._torch_mlp import wrap_torch_layer_norm

        return wrap_torch_layer_norm(inner, in_features)
    from auto_chasm._mlx_mlp import wrap_mlx_layer_norm

    return wrap_mlx_layer_norm(inner, in_features)


def _build_linear(in_features: int, out_features: int, backend: str) -> Any:
    """Build the built-in ``"linear"`` head on the given backend."""
    if backend == "torch":
        from auto_chasm._torch_mlp import build_torch_linear

        return build_torch_linear(in_features, out_features)
    from auto_chasm._mlx_mlp import build_mlx_linear

    return build_mlx_linear(in_features, out_features)


def _build_builtin_mlp(in_features: int, cfg: dict[str, Any], backend: str) -> Any:
    """Build the built-in ``"mlp"`` head (``hidden_dim``/``dropout`` from ``cfg``)."""
    out_features = cfg.get("out_features", 1)
    hidden_dim = cfg.get("hidden_dim", 256)
    dropout = cfg.get("dropout", 0.0)
    if backend == "torch":
        from auto_chasm._torch_mlp import build_torch_builtin_mlp

        return build_torch_builtin_mlp(in_features, out_features, hidden_dim, dropout)
    from auto_chasm._mlx_mlp import build_mlx_builtin_mlp

    return build_mlx_builtin_mlp(in_features, out_features, hidden_dim, dropout)


def build_probe_module(
    config: Any, in_features: int, cfg: dict[str, Any], backend_name: str
) -> Any:
    """Build a probe's trainable head from its :class:`ProbeConfig`.

    Handles the built-in ``"linear"``/``"mlp"`` strings, a user callable
    ``(in_features, cfg) -> module`` (a :class:`ModuleSpec` or any callable),
    and the ``ProbeConfig.layer_norm`` input-normalization wrap.

    Args:
        config: The probe configuration (``module_type``, ``layer_norm``).
        in_features: Resolved input dimension for the head.
        cfg: A mutable copy of ``module_config`` (``in_features`` already popped).
        backend_name: ``"mlx"`` or ``"torch"``.

    Returns:
        The constructed head module.

    Raises:
        ValueError: If ``module_type`` is an unknown string.
    """
    module_type = config.module_type
    if callable(module_type) and not isinstance(module_type, str):
        # Inject the backend so a ModuleSpec (or any callable) can build the
        # right framework module without the user ever naming a backend.  The
        # key is only ADDED — legacy callables that read cfg are unaffected.
        inner = module_type(in_features, {**cfg, "_backend_name": backend_name})
    elif module_type == "linear":
        inner = _build_linear(in_features, cfg.get("out_features", 1), backend_name)
    elif module_type == "mlp":
        inner = _build_builtin_mlp(in_features, cfg, backend_name)
    else:
        raise ValueError(
            f"Unknown module_type {module_type!r}. "
            "Use 'linear', 'mlp', a ModuleSpec, or a callable."
        )

    if config.layer_norm:
        inner = wrap_with_layer_norm(inner, in_features, backend_name)
    return inner


@dataclass(frozen=True)
class ModuleSpec:
    """A declarative, backend-agnostic probe-head specification.

    Pass an instance as ``ProbeConfig(module_type=ModuleSpec.mlp(...))``.  When
    the probe is attached, the library calls the spec with the resolved input
    width and builds the concrete head on the model's backend — the user never
    imports ``mlx``/``torch`` or branches on the backend.

    Attributes:
        hidden_dims: Hidden layer widths; ``()`` is a single ``Linear``.
        out_features: Default output width (overridden by
            ``ProbeConfig(module_config={"out_features": N})`` if set there).
        activation: Activation name or a callable applied between hidden layers.
        dropout: Dropout probability after each activation.
        input_layer_norm: If ``True``, a ``LayerNorm`` normalizes the head input.
        bias: Whether the linear layers carry a bias.
    """

    hidden_dims: tuple[int, ...] = ()
    out_features: int = 1
    activation: str | Callable[..., Any] = "relu"
    dropout: float = 0.0
    input_layer_norm: bool = False
    bias: bool = True

    def __call__(self, in_features: int, cfg: dict[str, Any]) -> Any:
        """Build the concrete head for this spec.

        Args:
            in_features: Resolved input dimension supplied by the probe.
            cfg: The probe's ``module_config`` plus the injected
                ``"_backend_name"`` (read here, never required from the user).

        Returns:
            A framework ``nn.Module`` head.
        """
        out_features = cfg.get("out_features", self.out_features)
        backend = cfg.get("_backend_name", "mlx")
        return build_module(
            in_features,
            out_features,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            input_layer_norm=self.input_layer_norm,
            bias=self.bias,
            backend=backend,
        )

    @classmethod
    def mlp(
        cls,
        hidden_dims: Sequence[int],
        out_features: int = 1,
        *,
        activation: str | Callable[..., Any] = "relu",
        dropout: float = 0.0,
        input_layer_norm: bool = False,
        bias: bool = True,
    ) -> ModuleSpec:
        """Build a multi-layer MLP head spec.

        Args:
            hidden_dims: Hidden layer widths (e.g. ``(256, 128)``).
            out_features: Output width (e.g. number of classes).
            activation: Activation name or callable between hidden layers.
            dropout: Dropout probability after each activation.
            input_layer_norm: If ``True``, normalize the head input.
            bias: Whether the linear layers carry a bias.

        Returns:
            A :class:`ModuleSpec`.
        """
        return cls(
            hidden_dims=tuple(hidden_dims),
            out_features=out_features,
            activation=activation,
            dropout=dropout,
            input_layer_norm=input_layer_norm,
            bias=bias,
        )

    @classmethod
    def linear(
        cls,
        out_features: int = 1,
        *,
        input_layer_norm: bool = False,
        bias: bool = True,
    ) -> ModuleSpec:
        """Build a single-``Linear`` head spec.

        Args:
            out_features: Output width.
            input_layer_norm: If ``True``, normalize the head input.
            bias: Whether the linear layer carries a bias.

        Returns:
            A :class:`ModuleSpec`.
        """
        return cls(
            hidden_dims=(), out_features=out_features, input_layer_norm=input_layer_norm, bias=bias
        )

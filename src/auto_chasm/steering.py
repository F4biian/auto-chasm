"""Steering hooks — modify hidden states during the forward pass.

Steering intercepts captured hidden states *inside* the forward pass
(before they flow to subsequent layers) and modifies them using
pre-computed geometry (class means, head direction, etc.).

Built-in methods (closed-form, applied by ``build_auto_steer_fn`` during
generation; all honor ``config.scale`` and only edit the last position):
    - ``nullify``: Drive the head logit toward 0 (only when currently
      positive), i.e. remove the head-aligned component.
    - ``push_to_mean``: Drive the head logit toward the negative-class
      (``mean_0``) logit — the suppression direction — only when the probe
      currently predicts positive.  This is the "push to the safe mean"
      method from the original experiments, not a push toward ``mean_1``.
    - ``boundary``: Push the activation's projection across the midpoint
      between the two class means.

Custom steering functions receive ``(hidden, head, logits)`` and must
return a modified hidden tensor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from auto_chasm.config import SteeringConfig
from auto_chasm.logger import get_logger
from auto_chasm.utils import tensor_backend

if TYPE_CHECKING:
    import mlx.core as mx
    import torch

logger = get_logger(__name__)

SteerFn = Callable[[Any, Any, Any], Any]


class SteeringHook:
    """Runtime steering hook for a single probe.

    Computes and stores steering geometry (class means, head direction,
    Fisher statistics) and applies steering during the forward pass.

    Args:
        probe_name: Name of the probe this hook is attached to.
        config: Steering configuration.
    """

    def __init__(self, probe_name: str, config: SteeringConfig) -> None:
        """Initialize the steering hook."""
        self.probe_name = probe_name
        self.config = config
        self.enabled = False

        self._mean_0: Any = None
        self._mean_1: Any = None
        self._direction: Any = None
        self._head_weight: Any = None
        self._head_bias: Any = None
        self._head_norm: float = 0.0
        self._fisher_along: float = 0.0
        self._custom_fn: SteerFn | None = None

    @property
    def has_geometry(self) -> bool:
        """Whether steering geometry has been computed."""
        return self._mean_0 is not None and self._mean_1 is not None

    def compute_geometry(
        self,
        hidden_by_class: dict[int, list[Any]],
        head_weight: mx.array | torch.Tensor,
        head_bias: mx.array | torch.Tensor,
    ) -> None:
        """Compute steering geometry from class-separated hidden states.

        Args:
            hidden_by_class: ``{class_label: [hidden_state_tensors]}``.
            head_weight: Probe head weight tensor (``[1, hidden_dim]``).
            head_bias: Probe head bias tensor (``[1]``).
        """
        from auto_chasm.utils import compute_class_means

        means = compute_class_means(hidden_by_class)
        self._mean_0 = means.get(0)
        self._mean_1 = means.get(1)

        if self._mean_0 is not None and self._mean_1 is not None:
            self._direction = self._mean_1 - self._mean_0

        self._head_weight = head_weight
        self._head_bias = head_bias

        try:
            self._head_norm = self._norm(head_weight)
        except Exception:
            self._head_norm = 0.0

        logger.info(
            "Steering '%s': geometry computed (||w||=%.2f, ||delta_mean||=%.2f)",
            self.probe_name,
            self._head_norm,
            self._norm(self._direction) if self._direction is not None else 0.0,
        )

    def set_custom(self, fn: SteerFn) -> None:
        """Set a custom steering function.

        Args:
            fn: ``(hidden, head_module, logits) -> modified_hidden``.
        """
        self._custom_fn = fn

    def enable(self) -> None:
        """Activate steering."""
        self.enabled = True
        logger.info("Steering '%s': enabled (%s)", self.probe_name, self.config.method)

    def disable(self) -> None:
        """Deactivate steering."""
        self.enabled = False

    def steer(
        self,
        hidden: mx.array | torch.Tensor,
        head: Any,
        logits: mx.array | torch.Tensor,
    ) -> mx.array | torch.Tensor:
        """Apply steering to a hidden-state tensor.

        If steering is not enabled or geometry is missing, returns
        the hidden state unchanged.

        Args:
            hidden: Hidden-state tensor ``[batch, seq, hidden_dim]``.
            head: The probe head module.
            logits: Pre-steering probe logits.

        Returns:
            Modified hidden-state tensor.
        """
        if not self.enabled:
            return hidden

        if self._custom_fn is not None:
            return self._custom_fn(hidden, head, logits)

        if not self.has_geometry:
            logger.warning(
                "Steering '%s': no geometry computed; returning unchanged.",
                self.probe_name,
            )
            return hidden

        method = self.config.method
        scale = self.config.scale
        direction = self.config.direction if self.config.direction is not None else self._direction

        if direction is None:
            logger.warning("Steering '%s': no direction available.", self.probe_name)
            return hidden

        if method == "nullify":
            return self._nullify(hidden, direction, scale)
        elif method == "push_to_mean":
            return self._push_to_mean(hidden, direction, scale)
        elif method == "boundary":
            return self._boundary(hidden, direction, logits, scale)
        else:
            return hidden

    def _nullify(
        self,
        hidden: mx.array | torch.Tensor,
        direction: mx.array | torch.Tensor,
        scale: float,
    ) -> mx.array | torch.Tensor:
        """Remove the direction-aligned component from hidden states.

        Projects out the component of ``hidden`` along ``direction``.
        This is the least-invasive steering method.

        Args:
            hidden: Hidden-state tensor.
            direction: Direction to project out.
            scale: Steering intensity.

        Returns:
            Modified hidden states.
        """
        if tensor_backend(hidden) == "torch":
            import torch

            dir_norm = direction / (torch.norm(direction) + 1e-8)
            proj = torch.sum(hidden * dir_norm, dim=-1, keepdim=True)
            return hidden - scale * proj * dir_norm
        import mlx.core as mx

        dir_norm = direction / (mx.linalg.norm(direction) + 1e-8)
        proj = mx.sum(hidden * dir_norm, axis=-1, keepdims=True)
        return hidden - scale * proj * dir_norm

    def _push_to_mean(
        self,
        hidden: mx.array | torch.Tensor,
        direction: mx.array | torch.Tensor,
        scale: float,
    ) -> mx.array | torch.Tensor:
        """Additively push hidden states along the class-mean axis.

        A raw primitive: moves activations by ``scale * direction`` where
        ``direction`` defaults to ``mean_1 - mean_0`` (toward the positive
        class).  This is the direct, standalone ``SteeringHook.steer()``
        API.  The generation path instead uses the closed-form
        ``build_auto_steer_fn`` whose ``push_to_mean`` drives the head
        logit toward the negative class (suppression); see the module
        docstring.

        Args:
            hidden: Hidden-state tensor.
            direction: Direction vector (defaults to ``mean_1 - mean_0``).
            scale: Steering intensity.

        Returns:
            Modified hidden states.
        """
        return hidden + scale * direction

    def _boundary(
        self,
        hidden: mx.array | torch.Tensor,
        direction: mx.array | torch.Tensor,
        logits: mx.array | torch.Tensor,
        scale: float,
    ) -> mx.array | torch.Tensor:
        """Push activations across the decision boundary.

        Uses the head weight to find the distance to the boundary
        and pushes by ``scale * distance`` along the direction.

        Args:
            hidden: Hidden-state tensor.
            direction: Direction vector.
            logits: Pre-steering probe logits.
            scale: Steering intensity.

        Returns:
            Modified hidden states.
        """
        if tensor_backend(hidden) == "torch":
            import torch

            dir_norm = direction / (torch.norm(direction) + 1e-8)
            boundary_dist = torch.abs(logits) / (self._head_norm + 1e-8)
            shift = scale * boundary_dist.unsqueeze(-1) * dir_norm
            return hidden + shift
        import mlx.core as mx

        dir_norm = direction / (mx.linalg.norm(direction) + 1e-8)
        boundary_dist = mx.abs(logits) / (self._head_norm + 1e-8)
        shift = scale * mx.expand_dims(boundary_dist, -1) * dir_norm
        return hidden + shift

    def _norm(self, tensor: mx.array | torch.Tensor) -> float:
        """Compute L2 norm of a tensor."""
        if tensor_backend(tensor) == "torch":
            import torch

            return float(torch.norm(tensor).item())
        import mlx.core as mx

        return float(mx.linalg.norm(tensor).item())

    def to_dict(self) -> dict[str, Any]:
        """Serialize steering geometry to a JSON-compatible dict.

        Returns:
            Dict with class means, head geometry, and config.
        """
        result: dict[str, Any] = {
            "probe_name": self.probe_name,
            "method": self.config.method,
            "scale": self.config.scale,
            "head_norm": self._head_norm,
            "fisher_along": self._fisher_along,
        }
        if self.config.layer is not None:
            result["config_layer"] = self.config.layer
        if self.config.direction is not None:
            # Persist the override direction so a reloaded model steers the same
            # way; without this the override is silently dropped on reload.
            result["config_direction"] = self._to_list(self.config.direction)
        if self._mean_0 is not None:
            result["mean_0"] = self._to_list(self._mean_0)
        if self._mean_1 is not None:
            result["mean_1"] = self._to_list(self._mean_1)
        if self._head_bias is not None:
            result["head_bias"] = self._to_list(self._head_bias)
        if self._head_weight is not None:
            result["head_weight"] = self._to_list(self._head_weight)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any], backend: str = "mlx") -> SteeringHook:
        """Restore a hook from a serialized dict.

        Args:
            data: Dict produced by ``to_dict()``.
            backend: ``"mlx"`` or ``"torch"`` — the tensor type to restore
                geometry as.  Must match the model the hook will steer, or
                the forward pass would mix tensor backends.

        Returns:
            A ``SteeringHook`` with geometry loaded.
        """
        config = SteeringConfig(
            method=data.get("method", "nullify"),
            scale=data.get("scale", 1.0),
            layer=data.get("config_layer"),
        )
        if "config_direction" in data:
            config.direction = cls._from_list(data["config_direction"], backend)
        hook = cls(data["probe_name"], config)
        hook._head_norm = data.get("head_norm", 0.0)
        hook._fisher_along = data.get("fisher_along", 0.0)
        if "mean_0" in data:
            hook._mean_0 = cls._from_list(data["mean_0"], backend)
        if "mean_1" in data:
            hook._mean_1 = cls._from_list(data["mean_1"], backend)
        if hook._mean_0 is not None and hook._mean_1 is not None:
            hook._direction = hook._mean_1 - hook._mean_0
        if "head_bias" in data:
            hook._head_bias = cls._from_list(data["head_bias"], backend)
        if "head_weight" in data:
            hook._head_weight = cls._from_list(data["head_weight"], backend)
        return hook

    @staticmethod
    def _to_list(tensor: Any) -> list[float]:
        """Convert a tensor to a flat list for JSON serialization."""
        try:
            return tensor.tolist()  # type: ignore[no-any-return]
        except AttributeError:
            return list(tensor)

    @staticmethod
    def _from_list(data: list[float], backend: str = "mlx") -> Any:
        """Restore a tensor from a flat list in the requested backend."""
        if backend == "torch":
            import torch

            return torch.tensor(data)
        import mlx.core as mx

        return mx.array(data)


def validate_steering(hook: SteeringHook, probe: Any, model: Any) -> None:
    """Reject steering configurations that would silently misbehave.

    Guards three silent-failure cases:

    - A **multi-layer probe** is ambiguous (which layer's hidden state should be
      modified?), so steering it is rejected.
    - ``method="custom"`` **without a custom ``steer_fn``** would fall through to the
      closed-form path and be treated as ``boundary`` steering (the ``else`` branch of
      the method dispatch), silently applying a different method than requested.
    - ``config.layer``, when set, must match the probe's (single) layer — steering acts
      at the probe's layer, so a divergent ``config.layer`` would otherwise be ignored.

    Args:
        hook: The steering hook (its config and any custom fn are inspected).
        probe: The probe being steered.
        model: The underlying model, used to resolve negative layer indices.

    Raises:
        ValueError: If steering cannot be applied cleanly.
    """
    layers = list(probe.config.layers)
    if len(layers) > 1:
        raise ValueError(
            f"Cannot steer probe '{probe.name}': it spans layers {layers}. Steering "
            f"supports single-layer probes only (which layer to modify is ambiguous)."
        )
    if hook.config.method == "custom" and hook._custom_fn is None:
        raise ValueError(
            f"Steering method='custom' for probe '{probe.name}' requires a steer_fn. "
            f"Pass steer_fn=... to enable_steering; refusing a silent fallback to "
            f"boundary steering."
        )
    if hook.config.layer is not None and layers:
        n = getattr(getattr(model, "config", None), "num_hidden_layers", None)

        def _resolve(idx: int) -> int:
            return idx + n if idx < 0 and n is not None else idx

        if _resolve(hook.config.layer) != _resolve(layers[0]):
            raise ValueError(
                f"SteeringConfig.layer={hook.config.layer} does not match the layer of "
                f"probe '{probe.name}' ({layers[0]}). Steering acts at the probe's layer; "
                f"set config.layer to it or leave it None."
            )


def build_auto_steer_fn(hook: SteeringHook) -> SteerFn | None:
    """Build a closed-form steering function from a hook's geometry.

    Implements the ``auto_steer`` algorithm from ``test_joint_sft.py``:
    finds the direction that pushes the head logit toward the method's
    target, using the head weight and class-mean geometry.  The
    ``config.scale`` knob multiplies the computed shift (``scale=1.0`` is
    the closed-form intensity, ``0.0`` disables, ``>1.0`` overshoots), and
    ``config.direction`` overrides the class-mean axis when provided.

    Only steers the **last** token position (the token about to be
    generated).

    Args:
        hook: Steering hook with computed geometry (or a custom function).

    Returns:
        A ``steer_fn(hidden, head, logits) -> modified_hidden``,
        or ``None`` if neither a custom function nor geometry is present.
    """
    if hook._custom_fn is not None:
        return hook._custom_fn

    if not hook.has_geometry:
        logger.warning("No steering geometry; steering will be a no-op.")
        return None

    method = hook.config.method
    scale = hook.config.scale
    mean_0 = hook._mean_0
    mean_1 = hook._mean_1
    direction = hook.config.direction if hook.config.direction is not None else hook._direction

    def auto_steer(hidden: Any, head: Any, logits: Any) -> Any:
        """Closed-form steering — dispatched by tensor type."""
        if tensor_backend(hidden) == "torch":
            return _steer_torch(hidden, head, logits, method, mean_0, mean_1, direction, scale)
        return _steer_mlx(hidden, head, logits, method, mean_0, mean_1, direction, scale)

    return auto_steer


def _steer_mlx(
    hidden: mx.array,
    head: Any,
    logits: mx.array,
    method: str,
    mean_0: mx.array,
    mean_1: mx.array,
    direction: mx.array,
    scale: float = 1.0,
) -> mx.array:
    """MLX steering implementation."""
    import mlx.core as mx

    w = head.weight
    b = head.bias
    w_norm = mx.linalg.norm(w)

    if float(w_norm) < 1e-6:
        return hidden

    unit_dir = direction / (mx.linalg.norm(direction) + 1e-8)
    w_dot_d = mx.sum(w * unit_dir)
    alignment = mx.abs(w_dot_d) / (w_norm + 1e-8)

    if alignment < 0.1:
        d_vec = -w / w_norm
        w_dot_d_eff = -w_norm
    else:
        if w_dot_d > 0:
            d_vec = -unit_dir
            w_dot_d_eff = -w_dot_d
        else:
            d_vec = unit_dir
            w_dot_d_eff = w_dot_d

    if d_vec.ndim == 1:
        d_vec = mx.expand_dims(d_vec, axis=0)

    last_logits = logits[:, -1:]

    if method == "nullify":
        target = 0.0
        alpha = (target - last_logits) / w_dot_d_eff
        alpha = mx.where(last_logits > target, alpha, mx.array(0.0))
    elif method == "push_to_mean":
        # Drive the logit toward the negative-class (mean_0) logit — the
        # suppression target — and only intervene when currently positive.
        target = mx.sum(w * mean_0) + b
        alpha = (target - last_logits) / w_dot_d_eff
        alpha = mx.where(last_logits > 0.0, alpha, mx.array(0.0))
    else:
        unit_exp = mx.expand_dims(unit_dir, (0, 1))
        proj = mx.sum(hidden[:, -1:, :] * unit_exp, axis=-1)
        proj_0 = mx.sum(mean_0 * unit_dir)
        proj_1 = mx.sum(mean_1 * unit_dir)
        boundary_proj = (proj_0 + proj_1) / 2.0
        alpha = (boundary_proj - proj) / 1.0
        alpha = mx.where(proj < boundary_proj, alpha, mx.array(0.0))
        # Boundary is a geometric displacement along +unit_dir (toward the
        # midpoint) regardless of the head-weight sign, so it must NOT reuse
        # the logit-space d_vec chosen above.
        d_vec = mx.expand_dims(unit_dir, axis=0)

    delta = scale * mx.expand_dims(alpha, axis=-1) * mx.expand_dims(d_vec, axis=0)
    seq_len = hidden.shape[1]
    if seq_len <= 1:
        return hidden + delta
    return mx.concatenate([hidden[:, :-1, :], hidden[:, -1:, :] + delta], axis=1)


def _steer_torch(
    hidden: torch.Tensor,
    head: Any,
    logits: torch.Tensor,
    method: str,
    mean_0: torch.Tensor,
    mean_1: torch.Tensor,
    direction: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """PyTorch steering implementation."""
    import torch

    # Steering geometry (class means / direction) is loaded onto CPU by
    # load_class_means, while the hidden states and probe head live on the
    # model's device (e.g. cuda). Align the geometry to the hidden tensor's
    # device+dtype so cross-device ops don't raise "expected all tensors on the
    # same device" — which the capture catches and silently skips, leaving
    # steering a no-op (steered output == unsteered).
    device, dtype = hidden.device, hidden.dtype
    if mean_0 is not None:
        mean_0 = mean_0.to(device=device, dtype=dtype)
    if mean_1 is not None:
        mean_1 = mean_1.to(device=device, dtype=dtype)
    if direction is not None:
        direction = direction.to(device=device, dtype=dtype)

    w = head.weight
    b = head.bias
    w_norm = torch.norm(w)

    if float(w_norm.detach()) < 1e-6:
        return hidden

    unit_dir = direction / (torch.norm(direction) + 1e-8)
    w_dot_d = torch.sum(w * unit_dir)
    alignment = torch.abs(w_dot_d) / (w_norm + 1e-8)

    if alignment < 0.1:
        d_vec = -w / w_norm
        w_dot_d_eff = -w_norm
    else:
        if w_dot_d > 0:
            d_vec = -unit_dir
            w_dot_d_eff = -w_dot_d
        else:
            d_vec = unit_dir
            w_dot_d_eff = w_dot_d

    if d_vec.dim() == 1:
        d_vec = d_vec.unsqueeze(0)

    last_logits = logits[:, -1:]

    if method == "nullify":
        target = 0.0
        alpha = (target - last_logits) / w_dot_d_eff
        alpha = torch.where(last_logits > target, alpha, torch.zeros_like(alpha))
    elif method == "push_to_mean":
        target = torch.sum(w * mean_0) + b
        alpha = (target - last_logits) / w_dot_d_eff
        alpha = torch.where(last_logits > 0.0, alpha, torch.zeros_like(alpha))
    else:
        unit_exp = unit_dir.unsqueeze(0).unsqueeze(0)
        proj = torch.sum(hidden[:, -1:, :] * unit_exp, dim=-1)
        proj_0 = torch.sum(mean_0 * unit_dir)
        proj_1 = torch.sum(mean_1 * unit_dir)
        boundary_proj = (proj_0 + proj_1) / 2.0
        alpha = (boundary_proj - proj) / 1.0
        alpha = torch.where(proj < boundary_proj, alpha, torch.zeros_like(alpha))
        # Boundary is a geometric displacement along +unit_dir (toward the
        # midpoint) regardless of the head-weight sign, so it must NOT reuse
        # the logit-space d_vec chosen above.
        d_vec = unit_dir.unsqueeze(0)

    delta = scale * alpha.unsqueeze(-1) * d_vec.unsqueeze(0)
    seq_len = hidden.shape[1]
    if seq_len <= 1:
        return hidden + delta
    return torch.cat([hidden[:, :-1, :], hidden[:, -1:, :] + delta], dim=1)

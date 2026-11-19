"""Build a ``JointLoss`` from the flat keyword arguments used by the trainers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from auto_chasm.config import LM_HEAD

if TYPE_CHECKING:
    from auto_chasm.trainers.loss import JointLoss


def _joint_loss_from_keywords(
    *,
    lm_weight: float = 1.0,
    probe_weight: float = 1.0,
    probe_loss: str | Callable[..., Any] = "bce",
    probe_weights: dict[str, float] | None = None,
    probe_losses: dict[str, str | Callable[..., Any]] | None = None,
    class_weights: Any = None,
) -> JointLoss:
    """Build a :class:`JointLoss` from flat keyword arguments.

    ``TrainingConfig``, ``SFTTrainer`` and ``make_joint_loss`` describe the loss with
    a global ``probe_loss``/``probe_weight`` plus per-probe overrides. This maps
    those keywords onto the per-term ``weights``/``losses`` of ``JointLoss``.

    Args:
        lm_weight: Weight for the LM cross-entropy term.
        probe_weight: Default weight for probes not listed in ``probe_weights``.
        probe_loss: Default loss for probes not listed in ``probe_losses``.
        probe_weights: Per-probe weight overrides.
        probe_losses: Per-probe loss overrides.
        class_weights: Per-class weights for the built-in ``"ce"`` loss.

    Returns:
        The configured ``JointLoss``.
    """
    from auto_chasm.trainers.loss import JointLoss  # lazy: avoids circular import

    weights: dict[str, float] = {LM_HEAD: float(lm_weight)}
    weights.update(probe_weights or {})
    # Class weights are applied after the default loss is set, so they validate
    # against ``probe_loss`` (e.g. "ce") rather than the "bce" constructor default.
    jl = JointLoss(weights=weights, losses=probe_losses)
    jl._default_loss = probe_loss
    jl._default_weight = float(probe_weight)
    jl.set_class_weights(class_weights)
    return jl

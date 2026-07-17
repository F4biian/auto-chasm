"""The pre-Phase-3b ``JointLoss`` keyword adapter (a shim; see ``loss.py``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from auto_chasm.config import LM_HEAD

if TYPE_CHECKING:
    from auto_chasm.trainers.loss import JointLoss


def _joint_loss_from_legacy(
    *,
    lm_weight: float = 1.0,
    probe_weight: float = 1.0,
    probe_loss: str | Callable[..., Any] = "bce",
    probe_weights: dict[str, float] | None = None,
    probe_losses: dict[str, str | Callable[..., Any]] | None = None,
    class_weights: Any = None,
) -> JointLoss:
    """Build a :class:`JointLoss` from the pre-Phase-3b keyword arguments (a shim).

    Phase 3b-2 wires ``trainer.py``/``model.py``/``sft.py`` onto the new
    ``weights=``/``losses=``/``combine=`` API and migrates the remaining tests; until
    then this adapter lets ``make_joint_loss`` and ``SFTTrainer`` keep constructing a
    ``JointLoss`` without the removed ``lm_weight``/``probe_weight``/``probe_loss``
    constructor.  The old per-probe ``probe_weights``/``probe_losses`` dicts map
    directly onto the new per-term ``weights``/``losses`` dicts.

    Args:
        lm_weight: Weight for the LM cross-entropy term.
        probe_weight: The default weight for probes not listed in ``probe_weights``
            (carried onto the loss's internal ``_default_weight``).
        probe_loss: The default loss for probes not listed in ``probe_losses``
            (carried onto the loss's internal ``_default_loss``).
        probe_weights: Per-probe weight overrides.
        probe_losses: Per-probe loss overrides.
        class_weights: Per-class weights for the built-in ``"ce"`` loss.

    Returns:
        A ``JointLoss`` configured for the equivalent behavior.
    """
    from auto_chasm.trainers.loss import JointLoss  # lazy: avoids circular import

    weights: dict[str, float] = {LM_HEAD: float(lm_weight)}
    weights.update(probe_weights or {})
    # Build WITHOUT class_weights, set the old global defaults, THEN apply class
    # weights — so they validate against the real default loss (probe_loss), not the
    # "bce" JointLoss default (else make_joint_loss(probe_loss="ce", class_weights=...)
    # wrongly raised "no probe uses 'ce'").
    jl = JointLoss(weights=weights, losses=probe_losses)
    jl._default_loss = probe_loss
    jl._default_weight = float(probe_weight)
    jl.set_class_weights(class_weights)
    return jl

"""Tests for the new OOP ``JointLoss`` surface (Phase 3b-1).

These pin the behaviors introduced by the single backend-agnostic path that the
migrated oracles do not exercise directly:

- the reserved ``"lm_head"`` name (an actual probe named ``lm_head`` raises);
- ``weights``-key typo protection;
- ``combine=`` composing exactly like the equivalent weighted sum;
- ``weights``/``combine`` mutual exclusion;
- a 2-arg ``(probe, target)`` custom loss and a 3-arg ``(logits, target, mask)``
  custom loss producing IDENTICAL values (both signatures are supported).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from auto_chasm import JointLoss, ops
from auto_chasm.metrics import to_numpy

# Shared batch: B=2, T=5 -> 4 target tokens; valid window steps 1..3.
_BATCH = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]], dtype=np.int64)
_LENGTHS = np.array([[1, 4], [1, 4]], dtype=np.int64)
_RNG = np.random.default_rng(0)
_LM = _RNG.standard_normal((2, 4, 6)).astype(np.float32)
_BIN = _RNG.standard_normal((2, 4)).astype(np.float32)
_BIN_LAB = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=np.float32)


class _FakeModel:
    """Returns fixed ``(lm_logits, probe_dict)``; accepts the trainer's mask kwarg."""

    def __init__(self, lm: Any, probes: dict[str, Any]) -> None:
        self._lm = lm
        self._probes = probes

    def __call__(self, inputs: Any, mask: Any = None) -> tuple[Any, dict[str, Any]]:
        """Ignore inputs; return the fixed logits."""
        return self._lm, self._probes


def _model(probe_names: list[str]) -> _FakeModel:
    """A fake model emitting the same ``_BIN`` head under every requested name."""
    return _FakeModel(mx.array(_LM), {name: mx.array(_BIN) for name in probe_names})


def _run(jl: JointLoss, model: _FakeModel, labels: Any = None) -> tuple[float, dict[str, float]]:
    """Run ``jl`` and return ``(total, {component: value})`` as Python floats."""
    lab = mx.array(_BIN_LAB) if labels is None else labels
    total, _n, comp = jl(model, mx.array(_BATCH), lab, mx.array(_LENGTHS))
    return float(to_numpy(total)), {k: float(to_numpy(v)) for k, v in comp.items()}


# --------------------------------------------------------------------------- #
# Reserved name + typo protection.                                            #
# --------------------------------------------------------------------------- #


def test_lm_head_probe_name_is_reserved() -> None:
    """A probe actually named ``lm_head`` raises a clear ValueError at compute time."""
    jl = JointLoss()
    with pytest.raises(ValueError, match="reserved for the language-model head"):
        _run(jl, _model(["lm_head"]))


def test_unknown_weights_key_raises() -> None:
    """A ``weights`` key that is neither 'lm_head' nor a probe name is a typo error."""
    jl = JointLoss(weights={"lm_head": 0.0, "typo": 1.0})
    with pytest.raises(ValueError, match="Unknown weights key"):
        _run(jl, _model(["p"]), labels={"p": mx.array(_BIN_LAB)})


def test_weights_and_combine_mutually_exclusive() -> None:
    """Passing both ``weights`` and ``combine`` raises at construction."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        JointLoss(weights={"lm_head": 1.0}, combine=lambda L: L.lm_head)


# --------------------------------------------------------------------------- #
# combine == the equivalent weighted sum.                                     #
# --------------------------------------------------------------------------- #


def test_combine_equals_weighted_sum() -> None:
    """``combine=lambda L: L.lm_head + 0.5*L.p`` equals ``weights={lm_head:1, p:0.5}``."""
    labels = {"p": mx.array(_BIN_LAB)}
    weighted, wcomp = _run(JointLoss(weights={"lm_head": 1.0, "p": 0.5}), _model(["p"]), labels)
    combined, ccomp = _run(
        JointLoss(combine=lambda L: L.lm_head + 0.5 * L.p), _model(["p"]), labels
    )
    assert combined == pytest.approx(weighted, rel=1e-6)
    # Both report the same per-term components (the combine path computes all terms).
    assert set(wcomp) == set(ccomp) == {"lm_head", "p"}
    for key in wcomp:
        assert ccomp[key] == pytest.approx(wcomp[key], rel=1e-6)


def test_combine_computes_all_terms_even_when_unreferenced() -> None:
    """A ``combine`` that ignores a term still reports it in the components dict."""
    labels = {"p": mx.array(_BIN_LAB)}
    _total, comp = _run(JointLoss(combine=lambda L: L.p), _model(["p"]), labels)
    assert set(comp) == {"lm_head", "p"}  # lm_head is computed though unused


# --------------------------------------------------------------------------- #
# 2-arg and 3-arg custom losses agree.                                        #
# --------------------------------------------------------------------------- #


def _custom_3arg(logits: Any, targets: Any, mask: Any) -> Any:
    """Legacy 3-param custom loss: masked BCE-with-logits via ``ops``."""
    bce = ops.softplus(logits) - logits * targets
    return ops.masked_mean(bce, mask)


def _custom_2arg(probe: Any, targets: Any) -> Any:
    """New 2-param custom loss: the ProbeOutput's own ``bce`` over its bound mask."""
    return probe.bce(targets, mask=probe.mask)


def test_two_arg_and_three_arg_custom_losses_match() -> None:
    """The 2-arg ``(probe, target)`` and 3-arg ``(logits, target, mask)`` forms agree.

    Both compute the same masked BCE (the 2-arg form via ``probe.bce`` folds in the
    ``-100`` sentinel; here the labels have none, so the values are identical).
    """
    labels = {"p": mx.array(_BIN_LAB)}
    _t3, c3 = _run(
        JointLoss(weights={"lm_head": 0.0}, losses={"p": _custom_3arg}), _model(["p"]), labels
    )
    _t2, c2 = _run(
        JointLoss(weights={"lm_head": 0.0}, losses={"p": _custom_2arg}), _model(["p"]), labels
    )
    assert c2["p"] == pytest.approx(c3["p"], rel=1e-5)


def test_two_arg_custom_loss_can_reduce_over_bound_mask() -> None:
    """A 2-arg custom loss may call ``probe.reduce`` with the bound validity mask."""

    def _l1(probe: Any, targets: Any) -> Any:
        return probe.reduce(ops.abs(probe.logits - targets))

    labels = {"p": mx.array(_BIN_LAB)}
    _t, comp = _run(JointLoss(weights={"lm_head": 0.0}, losses={"p": _l1}), _model(["p"]), labels)
    # Independent recompute: masked-mean |logit - label| over steps 1..3.
    steps = np.arange(1, 5)
    mask = (steps >= 1) & (steps < 4)
    expected = float((np.abs(_BIN - _BIN_LAB[:, 1:]) * mask).sum() / (mask.sum() * 2))
    assert comp["p"] == pytest.approx(expected, rel=1e-5)


def test_lm_head_loss_override_by_callable() -> None:
    """``losses={'lm_head': fn}`` overrides the default token cross-entropy."""

    def _zero_lm(outputs: Any, targets: Any) -> Any:
        return ops.zeros_like(outputs.lm_ce)

    labels = {"p": mx.array(_BIN_LAB)}
    total, comp = _run(JointLoss(losses={"lm_head": _zero_lm}), _model(["p"]), labels)
    assert comp["lm_head"] == pytest.approx(0.0, abs=1e-7)
    # Total is then the probe term alone (bce, weight 1.0).
    assert total == pytest.approx(comp["p"], rel=1e-6)


def test_lm_head_loss_string_spec_raises_at_construction() -> None:
    """A string ``losses={'lm_head': 'ce'}`` is rejected at construction, not compute.

    The LM head defaults to token cross-entropy; only a callable can override it.
    A string loss name (bce/ce/mse/mae) is a probe-head spec and is meaningless for
    the LM head, so it must fail fast at ``__init__`` (good DX) rather than deep in
    ``_lm_term`` on the first training step.
    """
    with pytest.raises(ValueError, match=r"losses\['lm_head'\] must be a callable"):
        JointLoss(losses={"lm_head": "ce"})
    # A callable override is accepted.
    JointLoss(losses={"lm_head": lambda outputs, target: outputs.lm_ce})


def test_two_arg_custom_loss_calling_probe_ce_on_int_target_does_not_abort() -> None:
    """A 2-param custom loss may call probe.ce(target) on INT labels without crashing.

    Regression: the 2-param branch received the FLOAT-cast target, so probe.ce (which
    gathers with the target as class indices) triggered an uncatchable MLX C++ abort
    on float indices. The branch now passes the raw int labels; a finite value proves
    the abort is gone.
    """
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((2, 4, 3)).astype(np.float32)
    labels = np.array([[0, 1, 2, 0, 1], [2, 1, 0, 1, 2]], dtype=np.int64)

    def _ce_via_probe(probe: Any, target: Any) -> Any:
        return probe.ce(target, mask=probe.mask)

    model = _FakeModel(mx.array(_LM), {"p": mx.array(logits)})
    _t, comp = _run(
        JointLoss(weights={"lm_head": 0.0}, losses={"p": _ce_via_probe}), model, mx.array(labels)
    )
    assert np.isfinite(comp["p"]) and comp["p"] > 0

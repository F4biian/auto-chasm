"""Regression tests for outputs.py / ops.py.

Covers five confirmed bugs, each with a correctness oracle (hand-computed
value or reference behavior) and MLX <-> PyTorch parity where applicable:

1. ``ProbeOutput.ce`` raises a clear ``ValueError`` on an out-of-range class
   index on BOTH backends (finding #50).
2. ``_masked_mean`` divides by the broadcast mask, so a per-sequence ``[B, 1]``
   mask gives the true masked mean (finding #52).
3. ``ProbeOutput.mse`` raises a clear ``ValueError`` on a shape-mismatched
   target on BOTH backends (finding #62).
4. ``ProbeOutput.mae`` exists and equals a hand-computed absolute error
   (finding #66).
5. ``ops.softplus`` is finite with a finite gradient for large ``x`` (finding
   #51 / #67).
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest
import torch

from auto_chasm import ops
from auto_chasm.outputs import ProbeOutput, _masked_mean

# --------------------------------------------------------------------------- #
# Bug 1 — ProbeOutput.ce out-of-range class index                              #
# --------------------------------------------------------------------------- #


def test_ce_out_of_range_raises_mlx() -> None:
    """MLX ``ce`` must raise ValueError on an OOB class index (was silent)."""
    po = ProbeOutput(logits=mx.zeros((1, 2, 3)))  # C = 3, valid indices 0..2
    with pytest.raises(ValueError, match="out of range"):
        po.ce(mx.array([[0, 5]]))


def test_ce_out_of_range_raises_torch() -> None:
    """Torch ``ce`` must raise the same ValueError on an OOB class index."""
    po = ProbeOutput(logits=torch.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="out of range"):
        po.ce(torch.tensor([[0, 5]]))


def test_ce_negative_out_of_range_raises_both() -> None:
    """A negative (non -100) index is out of range on both backends."""
    with pytest.raises(ValueError, match="out of range"):
        ProbeOutput(logits=mx.zeros((1, 2, 3))).ce(mx.array([[0, -1]]))
    with pytest.raises(ValueError, match="out of range"):
        ProbeOutput(logits=torch.zeros((1, 2, 3))).ce(torch.tensor([[0, -1]]))


def test_ce_ignore_sentinel_is_exempt_from_bounds_check() -> None:
    """The -100 sentinel must NOT trip the OOB bounds check on either backend.

    The bounds check only rejects genuine out-of-range indices; -100 (the ignore
    sentinel) is masked out by the caller, so ``ce`` must return a finite value
    rather than raising. (Value parity for -100 is the caller's job via ``mask``,
    so this asserts only that neither backend raises and both stay finite.)
    """
    logits_mx = mx.array([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]])
    logits_t = torch.tensor([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]])
    ce_mx = float(ProbeOutput(logits=logits_mx).ce(mx.array([[0, -100]])))
    ce_t = float(ProbeOutput(logits=logits_t).ce(torch.tensor([[0, -100]])))
    assert math.isfinite(ce_mx)
    assert math.isfinite(ce_t)


def test_ce_masked_sentinel_value_and_parity() -> None:
    """With the -100 position masked out, ce is correct and backends agree.

    Oracle: only the valid position (uniform logits) contributes => CE == ln(C).
    """
    logits_mx = mx.array([[[0.0, 0.0], [0.0, 0.0]]])  # C = 2
    logits_t = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    mask_mx = mx.array([[1.0, 0.0]])
    mask_t = torch.tensor([[1.0, 0.0]])
    ce_mx = float(ProbeOutput(logits=logits_mx).ce(mx.array([[1, -100]]), mask=mask_mx))
    ce_t = float(ProbeOutput(logits=logits_t).ce(torch.tensor([[1, -100]]), mask=mask_t))
    assert ce_mx == pytest.approx(math.log(2.0), rel=1e-5)
    assert ce_t == pytest.approx(math.log(2.0), rel=1e-5)


def test_ce_in_range_value_and_parity() -> None:
    """Oracle: uniform logits give CE == ln(C); MLX and torch agree."""
    logits_mx = mx.array([[[0.0, 0.0], [0.0, 0.0]]])  # C = 2
    logits_t = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    ce_mx = float(ProbeOutput(logits=logits_mx).ce(mx.array([[0, 1]])))
    ce_t = float(ProbeOutput(logits=logits_t).ce(torch.tensor([[0, 1]])))
    assert ce_mx == pytest.approx(math.log(2.0), rel=1e-5)
    assert ce_t == pytest.approx(math.log(2.0), rel=1e-5)


# --------------------------------------------------------------------------- #
# Bug 2 — _masked_mean broadcasts the mask before dividing                     #
# --------------------------------------------------------------------------- #


def test_masked_mean_b1_mask_mlx() -> None:
    """Oracle: a [B,1] mask counts every valid token, not one per row.

    values = [[1,2,3],[4,5,6]], mask = [[1],[0]] -> mean over row 0's three
    valid positions = (1+2+3)/3 = 2.0.
    """
    values = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = mx.array([[1.0], [0.0]])
    assert float(_masked_mean(values, mask)) == pytest.approx(2.0, rel=1e-5)


def test_masked_mean_b1_mask_torch() -> None:
    """Same oracle as the MLX case, torch backend."""
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0], [0.0]])
    assert float(_masked_mean(values, mask)) == pytest.approx(2.0, rel=1e-5)


def test_masked_mean_b1_parity() -> None:
    """MLX and torch agree on the [B,1]-masked mean (regression for factor-T bug)."""
    vals = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    m = [[1.0], [0.0]]
    out_mx = float(_masked_mean(mx.array(vals), mx.array(m)))
    out_t = float(_masked_mean(torch.tensor(vals), torch.tensor(m)))
    assert out_mx == pytest.approx(out_t, rel=1e-5)


def test_masked_mean_full_mask_equals_mean() -> None:
    """A full [B,T] mask of ones reduces to the plain mean (sanity oracle)."""
    values = mx.array([[1.0, 2.0, 3.0]])
    mask = mx.ones((1, 3))
    assert float(_masked_mean(values, mask)) == pytest.approx(2.0, rel=1e-5)


def test_probe_mse_b1_mask_value_and_parity() -> None:
    """End-to-end through ProbeOutput.mse with a [B,1] mask.

    logits = [[1,2,3],[4,5,6]], targets = 0, mask = [[1],[0]] -> masked MSE over
    row 0 = (1 + 4 + 9)/3 = 14/3.
    """
    logits = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    po_mx = ProbeOutput(logits=mx.array(logits))
    po_t = ProbeOutput(logits=torch.tensor(logits))
    out_mx = float(po_mx.mse(mx.zeros((2, 3)), mask=mx.array([[1.0], [0.0]])))
    out_t = float(po_t.mse(torch.zeros((2, 3)), mask=torch.tensor([[1.0], [0.0]])))
    assert out_mx == pytest.approx(14.0 / 3.0, rel=1e-5)
    assert out_t == pytest.approx(14.0 / 3.0, rel=1e-5)
    assert out_mx == pytest.approx(out_t, rel=1e-5)


# --------------------------------------------------------------------------- #
# Bug 3 — ProbeOutput.mse shape guard                                          #
# --------------------------------------------------------------------------- #


def test_mse_shape_mismatch_raises_mlx() -> None:
    """MLX already raised; assert it stays a clear ValueError."""
    po = ProbeOutput(logits=mx.zeros((2, 3)))
    with pytest.raises(ValueError, match="does not match"):
        po.mse(mx.array([1.0, 0.0, 1.0]))


def test_mse_shape_mismatch_raises_torch() -> None:
    """Torch must now raise instead of silently broadcasting a 1-D target."""
    po = ProbeOutput(logits=torch.zeros((2, 3)))
    with pytest.raises(ValueError, match="does not match"):
        po.mse(torch.tensor([1.0, 0.0, 1.0]))


def test_mse_matching_shape_value_and_parity() -> None:
    """Oracle: a correctly shaped target still gives the right MSE on both."""
    logits = [[1.0, 2.0, 3.0]]
    targets = [[0.0, 0.0, 0.0]]
    out_mx = float(ProbeOutput(logits=mx.array(logits)).mse(mx.array(targets)))
    out_t = float(ProbeOutput(logits=torch.tensor(logits)).mse(torch.tensor(targets)))
    assert out_mx == pytest.approx((1 + 4 + 9) / 3, rel=1e-5)
    assert out_t == pytest.approx((1 + 4 + 9) / 3, rel=1e-5)


# --------------------------------------------------------------------------- #
# Bug 4 — ProbeOutput.mae helper                                              #
# --------------------------------------------------------------------------- #


def test_mae_exists() -> None:
    """ProbeOutput must expose a callable .mae() on both backends."""
    assert callable(getattr(ProbeOutput(logits=mx.zeros((1, 3))), "mae", None))
    assert callable(getattr(ProbeOutput(logits=torch.zeros((1, 3))), "mae", None))


def test_mae_value_and_parity() -> None:
    """Oracle: mae([[1,-2,3]], 0) = mean(|1|,|2|,|3|) = 2.0 on both backends."""
    logits = [[1.0, -2.0, 3.0]]
    out_mx = float(ProbeOutput(logits=mx.array(logits)).mae(mx.zeros((1, 3))))
    out_t = float(ProbeOutput(logits=torch.tensor(logits)).mae(torch.zeros((1, 3))))
    assert out_mx == pytest.approx(2.0, rel=1e-5)
    assert out_t == pytest.approx(2.0, rel=1e-5)


def test_mae_masked_value() -> None:
    """Oracle: a [B,1] mask makes mae count every valid token.

    logits = [[1,-2,3],[9,9,9]], target = 0, mask = [[1],[0]] -> mean over row 0
    of |1|,|2|,|3| = 2.0 (row 1 fully masked out).
    """
    logits = mx.array([[1.0, -2.0, 3.0], [9.0, 9.0, 9.0]])
    out = ProbeOutput(logits=logits).mae(mx.zeros((2, 3)), mask=mx.array([[1.0], [0.0]]))
    assert float(out) == pytest.approx(2.0, rel=1e-5)


def test_mae_shape_mismatch_raises_both() -> None:
    """MAE guards shape on both backends, mirroring mse."""
    with pytest.raises(ValueError, match="does not match"):
        ProbeOutput(logits=mx.zeros((2, 3))).mae(mx.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="does not match"):
        ProbeOutput(logits=torch.zeros((2, 3))).mae(torch.tensor([1.0, 0.0, 1.0]))


# --------------------------------------------------------------------------- #
# Bug 5 — ops.softplus numerical stability                                     #
# --------------------------------------------------------------------------- #


def test_softplus_small_value_parity() -> None:
    """Oracle: softplus(0) == ln(2) on both backends (unchanged regression)."""
    assert float(ops.softplus(mx.array([0.0]))) == pytest.approx(math.log(2.0), rel=1e-5)
    assert float(ops.softplus(torch.tensor([0.0]))) == pytest.approx(math.log(2.0), rel=1e-5)


def test_softplus_large_value_finite_and_approx_identity() -> None:
    """Oracle: softplus(100) is finite and ~= 100 on both backends (was inf)."""
    out_mx = float(ops.softplus(mx.array([100.0])))
    out_t = float(ops.softplus(torch.tensor([100.0])))
    assert math.isfinite(out_mx)
    assert math.isfinite(out_t)
    assert out_mx == pytest.approx(100.0, rel=1e-5)
    assert out_t == pytest.approx(100.0, rel=1e-5)


def test_softplus_large_value_finite_gradient_torch() -> None:
    """Oracle: d/dx softplus(100) == sigmoid(100) == 1.0, finite (was NaN)."""
    x = torch.tensor([100.0], requires_grad=True)
    ops.softplus(x).backward()
    assert x.grad is not None
    g = float(x.grad)
    assert math.isfinite(g)
    assert g == pytest.approx(1.0, abs=1e-6)


def test_softplus_large_value_finite_gradient_mlx() -> None:
    """Oracle: MLX grad of softplus(100) == 1.0 and is finite (was NaN)."""

    def f(x: mx.array) -> mx.array:
        return ops.softplus(x).sum()

    g = float(mx.grad(f)(mx.array([100.0]))[0])
    assert math.isfinite(g)
    assert g == pytest.approx(1.0, abs=1e-6)


def test_softplus_matches_reference_across_range() -> None:
    """ops.softplus tracks the framework references over a wide x range."""
    xs = [-100.0, -5.0, 0.0, 5.0, 50.0, 89.0, 100.0]
    for x in xs:
        ref_t = float(torch.nn.functional.softplus(torch.tensor([x])))
        out_t = float(ops.softplus(torch.tensor([x])))
        out_mx = float(ops.softplus(mx.array([x])))
        assert math.isfinite(out_t) and math.isfinite(out_mx)
        # Reference may also be ~0 for very negative x; compare with abs tol there.
        assert out_t == pytest.approx(ref_t, rel=1e-4, abs=1e-6)
        assert out_mx == pytest.approx(ref_t, rel=1e-4, abs=1e-6)

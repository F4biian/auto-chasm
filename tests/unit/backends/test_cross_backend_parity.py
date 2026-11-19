"""Cross-backend parity + numeric/utility/config tests.

Covers ``ops.py``, ``utils.py``, ``backends/{base,mlx,torch}.py``,
``config.py`` and the shared loss math in ``outputs.py`` (the ``ProbeOutput`` /
``JointOutputs`` ops a custom loss actually calls).

The central correctness claim under test: **MLX and PyTorch give the same
answers**, and the shared math/utility primitives behave consistently and
validate their inputs.

Tests named ``test_BUG_*`` are regression tests for specific past defects; the
rest are general regression coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import mlx.core as mx  # noqa: E402

from auto_chasm import ops  # noqa: E402
from auto_chasm.config import (  # noqa: E402
    GenerationConfig,
    LoraConfig,
    ProbeConfig,
    RLConfig,
)
from auto_chasm.outputs import JointOutputs, ProbeOutput  # noqa: E402

TOL = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _t(arr: np.ndarray):
    return torch.tensor(arr)


def _m(arr: np.ndarray):
    return mx.array(arr)


def _np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.array(x)


def _both_close(torch_val, mlx_val, tol: float = TOL) -> None:
    np.testing.assert_allclose(_np(torch_val), _np(mlx_val), rtol=tol, atol=tol)


# ===========================================================================
# 1. ops.py element-wise parity (regression — these should PASS)
# ===========================================================================

EXTREME_INPUTS = np.array(
    [-1000.0, -89.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 50.0, 89.0, 100.0, 1000.0],
    dtype=np.float32,
)


@pytest.mark.parametrize("name", ["exp", "log", "sqrt", "abs", "sigmoid", "softplus"])
def test_unary_ops_parity(name: str) -> None:
    """Each unary op gives the same result on torch and MLX (incl. extremes)."""
    fn = getattr(ops, name)
    x = EXTREME_INPUTS.copy()
    tv = _np(fn(_t(x)))
    mv = _np(fn(_m(x)))
    # NaN/inf must appear in the SAME positions on both backends.
    np.testing.assert_array_equal(np.isnan(tv), np.isnan(mv))
    np.testing.assert_array_equal(np.isinf(tv), np.isinf(mv))
    finite = np.isfinite(tv) & np.isfinite(mv)
    np.testing.assert_allclose(tv[finite], mv[finite], rtol=1e-4, atol=1e-4)


def test_softplus_overflow_safe_both_backends() -> None:
    """softplus(large x) stays finite and ~= x on both backends (log-sum-exp trick)."""
    x = np.array([89.0, 100.0, 1000.0], dtype=np.float32)
    tv = _np(ops.softplus(_t(x)))
    mv = _np(ops.softplus(_m(x)))
    assert np.all(np.isfinite(tv)) and np.all(np.isfinite(mv))
    np.testing.assert_allclose(tv, x, rtol=1e-4)
    np.testing.assert_allclose(mv, x, rtol=1e-4)


@pytest.mark.parametrize("bounds", [(-2.0, None), (None, 2.0), (-2.0, 2.0)])
def test_clamp_parity_with_bounds(bounds) -> None:
    """Clamp with at least one bound matches across backends."""
    lo, hi = bounds
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
    _both_close(ops.clamp(_t(x), lo, hi), ops.clamp(_m(x), lo, hi))


def test_where_arange_parity() -> None:
    """Where + arange behave identically on both backends."""
    cond = np.array([True, False, True])
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    _both_close(ops.where(_t(cond), _t(a), _t(b)), ops.where(_m(cond), _m(a), _m(b)))
    _both_close(
        ops.arange(5, _t(np.array([0.0], dtype=np.float32)), start=2),
        ops.arange(5, _m(np.array([0.0], dtype=np.float32)), start=2),
    )


def test_BUG_clamp_no_bounds_diverges_across_backends() -> None:
    """clamp(x) with NO bounds must behave the same on both backends.

    BUG: ``ops.clamp(x)`` (both bounds ``None``) is a no-op on MLX (returns
    ``x`` unchanged) but RAISES ``RuntimeError`` on torch
    (``torch.clamp: At least one of 'min' or 'max' must not be None``).  Same
    call, opposite outcome — a cross-backend divergence in a documented op.
    Desired: identical behavior (both no-op, returning ``x``).
    """
    x = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    mlx_out = _np(ops.clamp(_m(x)))  # MLX: no-op, returns x
    # The torch path should mirror MLX (no-op) instead of raising.
    torch_out = _np(ops.clamp(_t(x)))  # currently raises RuntimeError -> test FAILS
    np.testing.assert_allclose(torch_out, mlx_out)
    np.testing.assert_allclose(torch_out, x)


# ===========================================================================
# 2. backend tensor-ops parity / silent upcast
# ===========================================================================


def test_BUG_mean_op_int_silently_upcasts_on_mlx_only() -> None:
    """backend.mean on an INTEGER tensor must behave the same on both backends.

    BUG: ``TorchTensorOps.mean`` raises on a Long tensor
    (``mean(): could not infer output dtype``) while ``MLXTensorOps.mean``
    silently upcasts to float and returns ``2.5``.  A user who computes a mean
    over integer data gets a number on MLX and a crash on torch — exactly the
    "silently upcast on one backend but not the other" footgun.  Both backends
    should agree (either both upcast, or both raise the same clear error).
    """
    from auto_chasm.backends.mlx_backend import MLXTensorOps
    from auto_chasm.backends.torch_backend import TorchTensorOps

    xi = np.array([1, 2, 3, 4], dtype=np.int64)
    mlx_val = float(MLXTensorOps().mean(_m(xi)))  # MLX: 2.5
    torch_val = float(TorchTensorOps().mean(_t(xi)))  # torch: raises -> test FAILS
    assert mlx_val == pytest.approx(torch_val)


def test_mean_op_float_parity() -> None:
    """backend.mean on float tensors matches across backends (regression)."""
    from auto_chasm.backends.mlx_backend import MLXTensorOps
    from auto_chasm.backends.torch_backend import TorchTensorOps

    x = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    _both_close(TorchTensorOps().mean(_t(x)), MLXTensorOps().mean(_m(x)))
    _both_close(TorchTensorOps().mean(_t(x), 0), MLXTensorOps().mean(_m(x), 0))


# ===========================================================================
# 3. Shared loss math parity (ProbeOutput / JointOutputs)
# ===========================================================================


def test_bce_mse_mae_ce_parity_with_mask() -> None:
    """bce/mse/mae/ce agree across backends with and without a mask (regression)."""
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((2, 3)).astype(np.float32)
    targets = rng.integers(0, 2, (2, 3)).astype(np.float32)
    mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=bool)
    pt, pm = ProbeOutput(logits=_t(logits)), ProbeOutput(logits=_m(logits))
    for m_arr in (None, mask):
        mt = None if m_arr is None else _t(m_arr)
        mm = None if m_arr is None else _m(m_arr)
        _both_close(pt.bce(_t(targets), mt), pm.bce(_m(targets), mm))
        _both_close(pt.mse(_t(targets), mt), pm.mse(_m(targets), mm))
        _both_close(pt.mae(_t(targets), mt), pm.mae(_m(targets), mm))
    # multi-class CE
    cl = rng.standard_normal((2, 3, 4)).astype(np.float32)
    ti = rng.integers(0, 4, (2, 3)).astype(np.int64)
    _both_close(
        ProbeOutput(logits=_t(cl)).ce(_t(ti)),
        ProbeOutput(logits=_m(cl)).ce(_m(ti)),
    )


def test_BUG_probe_ce_ignores_minus_100_inconsistently() -> None:
    """ProbeOutput.ce must ignore the -100 sentinel (and agree across backends).

    The -100 sentinel is the documented ignore index everywhere in the library
    (class-means, RL penalty, lm masking).  But ``ProbeOutput.ce`` with -100
    targets and no explicit mask is wrong on BOTH backends AND they disagree:

    * torch counts the -100 slot in the denominator (divides by N, not N_valid)
      -> too small.
    * MLX gathers an out-of-bounds class score for the -100 index (no bounds
      check) and adds a garbage non-zero loss at the masked position -> too big.

    Correct value = mean CE over the non-(-100) positions (``ignore_index=-100``).
    """
    cl = np.random.default_rng(1).standard_normal((1, 4, 3)).astype(np.float32)
    ti = np.array([[0, 1, -100, 2]], dtype=np.int64)

    val_t = float(ProbeOutput(logits=_t(cl)).ce(_t(ti)))
    val_m = float(ProbeOutput(logits=_m(cl)).ce(_m(ti)))

    ref = float(
        torch.nn.functional.cross_entropy(
            _t(cl).reshape(-1, 3), _t(ti).reshape(-1), ignore_index=-100
        )
    )
    # Each backend should equal the ignore_index reference (and thus each other).
    assert val_t == pytest.approx(ref, abs=1e-5), f"torch ce {val_t} != ref {ref}"
    assert val_m == pytest.approx(ref, abs=1e-5), f"mlx ce {val_m} != ref {ref}"


def test_BUG_probe_ce_empty_window_crashes_on_mlx_only() -> None:
    """ProbeOutput.ce on an all-padding (zero-token) window must not crash.

    The documented behaviour is "empty-window batches -> finite 0 (no NaN)".  That
    holds for ``lm_ce`` and for bce/mse/mae, but ``ProbeOutput.ce`` on an empty
    token window returns 0.0 on torch and RAISES on MLX
    (``ValueError: [logsumexp] Received empty array``) — MLX's cross_entropy
    blows up before the zero-token-safe ``_masked_mean`` is reached.
    Desired: both backends return a finite 0.0.
    """
    cl = np.zeros((1, 0, 3), dtype=np.float32)
    ti = np.zeros((1, 0), dtype=np.int64)
    val_t = float(ProbeOutput(logits=_t(cl)).ce(_t(ti)))
    assert val_t == 0.0
    val_m = float(ProbeOutput(logits=_m(cl)).ce(_m(ti)))  # raises today -> FAILS
    assert val_m == 0.0


def test_bce_mse_mae_empty_window_parity() -> None:
    """bce/mse/mae on a zero-token window return finite 0 on both backends."""
    le = np.zeros((1, 0), dtype=np.float32)
    for method in ("bce", "mse", "mae"):
        vt = float(getattr(ProbeOutput(logits=_t(le)), method)(_t(le)))
        vm = float(getattr(ProbeOutput(logits=_m(le)), method)(_m(le)))
        assert vt == 0.0 and vm == 0.0


def test_lm_ce_empty_window_parity() -> None:
    """JointOutputs.lm_ce with no valid tokens returns 0 on both backends."""
    ll = np.random.default_rng(0).standard_normal((1, 3, 5)).astype(np.float32)
    tgt = np.array([[1, 2, 3]], dtype=np.int64)
    lengths = np.array([[5, 5]], dtype=np.int64)  # window empty for steps 1..3
    jt = JointOutputs(_t(ll), {}, _t(tgt), _t(lengths))
    jm = JointOutputs(_m(ll), {}, _m(tgt), _m(lengths))
    assert float(jt.lm_ce) == 0.0 and float(jm.lm_ce) == 0.0


def test_bce_all_masked_parity() -> None:
    """Bce with an all-False mask returns 0 on both backends (regression)."""
    logits = np.random.default_rng(2).standard_normal((2, 3)).astype(np.float32)
    targets = np.zeros((2, 3), dtype=np.float32)
    mask = np.zeros((2, 3), dtype=bool)
    vt = float(ProbeOutput(logits=_t(logits)).bce(_t(targets), _t(mask)))
    vm = float(ProbeOutput(logits=_m(logits)).bce(_m(targets), _m(mask)))
    assert vt == 0.0 and vm == 0.0


def test_int_target_bce_parity() -> None:
    """Integer labels to bce upcast identically on both backends (regression)."""
    lg = np.array([[0.5, -0.5]], dtype=np.float32)
    ti = np.array([[1, 0]], dtype=np.int64)
    _both_close(
        ProbeOutput(logits=_t(lg)).bce(_t(ti)),
        ProbeOutput(logits=_m(lg)).bce(_m(ti)),
    )


# ===========================================================================
# 4. config validation — every config must reject bad values clearly,
#    not silently accept / silently do the wrong thing.
# ===========================================================================


def test_BUG_probe_config_invalid_source_silently_accepted() -> None:
    """ProbeConfig.source is a Literal but an invalid value is accepted.

    BUG: ``ProbeConfig.__post_init__`` validates ``aggregation`` (raises on an
    unknown string) but NOT the sibling ``Literal`` field ``source``.  A typo
    like ``source="banana"`` constructs fine and is later silently treated as
    ``"embedding"`` in ``Model.attach_probe`` (the ``else`` branch) — so the
    probe captures the embedding instead of the intended site, with no error.
    A typed enum field with a typo must raise a clear ValueError.
    """
    with pytest.raises((ValueError, TypeError)):
        ProbeConfig(name="p", layers=[0], source="banana")


def test_BUG_probe_config_invalid_granularity_silently_accepted() -> None:
    """ProbeConfig.granularity Literal typo is accepted then silently no-ops.

    BUG: ``granularity="paragraph"`` constructs without error and downstream is
    silently treated as ``"token"`` (per-token output) — the requested
    granularity is ignored rather than rejected.  Must raise a clear ValueError
    listing the valid granularities, like ``aggregation`` already does.
    """
    with pytest.raises((ValueError, TypeError)):
        ProbeConfig(name="p", layers=[0], granularity="paragraph")


def test_probe_config_invalid_aggregation_raises() -> None:
    """Sanity: aggregation IS validated (the inconsistency reference point)."""
    with pytest.raises(ValueError):
        ProbeConfig(name="p", layers=[0], aggregation="nope")


def test_BUG_lora_config_rejects_nonpositive_rank() -> None:
    """LoraConfig must reject rank <= 0.

    BUG: ``LoraConfig`` has no ``__post_init__`` at all, so ``rank=0`` (a
    divide-by-zero waiting to happen: effective scale is ``alpha/rank`` and MLX
    uses ``scale = alpha / r``) and ``rank=-1`` are silently accepted.  A
    non-positive rank is never valid and should raise at construction, not
    blow up cryptically inside ``linear_to_lora_layers``.
    """
    with pytest.raises((ValueError, ZeroDivisionError)):
        LoraConfig(rank=0)
    with pytest.raises(ValueError):
        LoraConfig(rank=-1)


def test_BUG_lora_config_rejects_unknown_peft_method() -> None:
    """LoraConfig.peft_method Literal typo must be rejected.

    BUG: ``peft_method="banana"`` is accepted (no validation).  It will only
    surface much later inside adapter application, as an opaque failure.
    A typed enum field should validate at construction.
    """
    with pytest.raises((ValueError, TypeError)):
        LoraConfig(peft_method="banana")


def test_BUG_generation_config_rejects_negative_temperature() -> None:
    """GenerationConfig must reject a negative temperature.

    BUG: ``GenerationConfig`` performs no validation, so ``temperature=-1``,
    ``top_p=5.0``, ``top_k=-3`` and ``max_tokens=-10`` are all silently
    accepted.  A negative temperature divides the logits by a negative number
    (flipping the distribution) — silently wrong sampling, not an error.
    """
    with pytest.raises(ValueError):
        GenerationConfig(temperature=-1.0)


def test_BUG_generation_config_rejects_out_of_range_top_p() -> None:
    """GenerationConfig.top_p outside (0, 1] must be rejected (no validation today)."""
    with pytest.raises(ValueError):
        GenerationConfig(top_p=5.0)


def test_BUG_rl_config_rejects_negative_beta() -> None:
    """RLConfig must reject a negative penalty strength.

    BUG: ``RLConfig`` has no validation; ``beta=-1`` is accepted.  For
    ``algorithm="sft"`` the total is ``ce + beta*penalty`` — a negative beta
    silently *rewards* the probe-penalty term (anti-training), a research
    footgun that should be rejected.
    """
    with pytest.raises(ValueError):
        RLConfig(beta=-1.0)


def test_BUG_rl_config_rejects_unknown_algorithm() -> None:
    """RLConfig.algorithm Literal typo should be rejected at construction.

    BUG: ``algorithm="banana"`` constructs fine; the unknown-algorithm
    ``ValueError`` only fires deep inside the trainer's ``_select_loss`` at
    train time.  ``ppo``/``grpo`` are *intentional* deferred raises (see
    documented), but an outright-unknown name should be caught early.
    """
    with pytest.raises((ValueError, TypeError)):
        RLConfig(algorithm="banana")


# ===========================================================================
# 5. config defaults vs documentation
# ===========================================================================


def test_config_defaults_match_docs() -> None:
    """Documented defaults match the dataclass defaults (regression)."""
    assert RLConfig().algorithm == "sft"
    assert RLConfig().beta == 0.1
    assert LoraConfig().rank == 8
    assert LoraConfig().alpha == 16
    assert GenerationConfig().temperature == 0.0
    assert GenerationConfig().max_tokens == 256
    assert ProbeConfig(name="p", layers=[0]).source == "hidden"
    assert ProbeConfig(name="p", layers=[0]).granularity == "token"
    assert ProbeConfig(name="p", layers=[0]).aggregation == "concat"


# ===========================================================================
# 6. utils.tensor_backend dispatch (the central anti-bug helper)
# ===========================================================================


def test_tensor_backend_dispatch() -> None:
    """tensor_backend identifies torch vs mlx by concrete type (regression)."""
    from auto_chasm.utils import tensor_backend

    assert tensor_backend(torch.zeros(2)) == "torch"
    assert tensor_backend(mx.zeros((2,))) == "mlx"

"""Parity oracle tests for the backend-agnostic tensor facade ``ops``.

Phase-2 correctness bar: for every new ``ops`` primitive, the MLX-CPU result,
the torch-CPU result, and an independent NumPy reference must all agree.  Both
frameworks are exercised on the *same* NumPy input; conftest forces MLX onto the
CPU device so the comparison is on equal footing.

torch is guarded with ``pytest.importorskip`` (MLX is always present).  These
tests load no model and are therefore not marked ``real_model``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from auto_chasm import ops
from auto_chasm.metrics import to_numpy

torch = pytest.importorskip("torch")

RTOL = 1e-5
ATOL = 1e-6


def _mx(np_x: np.ndarray) -> mx.array:
    """Build an MLX array from NumPy data."""
    return mx.array(np_x)


def _torch(np_x: np.ndarray) -> Any:
    """Build a torch tensor from NumPy data."""
    return torch.tensor(np_x)


def _both_equal(np_x: np.ndarray, fn: object, expected: np.ndarray, **kw: object) -> None:
    """Assert ``ops.<fn>`` matches across MLX, torch, and a NumPy reference.

    Args:
        np_x: The shared NumPy input.
        fn: The ``ops`` callable under test.
        expected: The independent NumPy reference result.
        **kw: Keyword arguments forwarded to ``fn``.
    """
    mlx_out = to_numpy(fn(_mx(np_x), **kw))  # type: ignore[operator]
    torch_out = to_numpy(fn(_torch(np_x), **kw))  # type: ignore[operator]
    np.testing.assert_allclose(mlx_out, torch_out, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(mlx_out, expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(torch_out, expected, rtol=RTOL, atol=ATOL)


def _np_logsumexp(x: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    """SciPy-free NumPy reference for log-sum-exp."""
    m = x.max(axis=axis, keepdims=True)
    out = np.log(np.exp(x - m).sum(axis=axis, keepdims=True)) + m
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _np_softmax(x: np.ndarray, axis: int) -> np.ndarray:
    """NumPy reference for softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# sum / mean
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
@pytest.mark.parametrize("axis", [0, 1, -1, None])
@pytest.mark.parametrize("keepdims", [True, False])
def test_sum_parity(shape: tuple[int, ...], axis: int | None, keepdims: bool) -> None:
    """``ops.sum`` matches NumPy across axes, keepdims, and full reduction."""
    np_x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = np.sum(np_x) if axis is None else np.sum(np_x, axis=axis, keepdims=keepdims)
    _both_equal(np_x, ops.sum, expected, axis=axis, keepdims=keepdims)


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
@pytest.mark.parametrize("axis", [0, 1, -1, None])
@pytest.mark.parametrize("keepdims", [True, False])
def test_mean_parity(shape: tuple[int, ...], axis: int | None, keepdims: bool) -> None:
    """``ops.mean`` matches NumPy across axes, keepdims, and full reduction."""
    rng = np.random.default_rng(0)
    np_x = rng.standard_normal(shape).astype(np.float32)
    expected = np.mean(np_x) if axis is None else np.mean(np_x, axis=axis, keepdims=keepdims)
    _both_equal(np_x, ops.mean, expected, axis=axis, keepdims=keepdims)


# ---------------------------------------------------------------------------
# max
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
@pytest.mark.parametrize("axis", [0, 1, -1, None])
@pytest.mark.parametrize("keepdims", [True, False])
def test_max_parity(shape: tuple[int, ...], axis: int | None, keepdims: bool) -> None:
    """``ops.max`` matches NumPy, incl. torch's ``dim``->``.values`` unwrap."""
    rng = np.random.default_rng(1)
    np_x = rng.standard_normal(shape).astype(np.float32)
    expected = np.max(np_x) if axis is None else np.max(np_x, axis=axis, keepdims=keepdims)
    _both_equal(np_x, ops.max, expected, axis=axis, keepdims=keepdims)


# ---------------------------------------------------------------------------
# argmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_argmax_parity(axis: int) -> None:
    """``ops.argmax`` matches NumPy on a tie-free input with known argmax."""
    np_x = np.array(
        [[0.1, 0.9, 0.2, 0.3], [0.5, 0.4, 0.8, 0.1], [0.2, 0.3, 0.1, 0.7]],
        dtype=np.float32,
    )
    expected = np.argmax(np_x, axis=axis).astype(np.float32)
    _both_equal(np_x, ops.argmax, expected, axis=axis)


def test_argmax_known_row() -> None:
    """``ops.argmax`` returns the hand-known index for a single tie-free row."""
    np_x = np.array([[0.1, 0.2, 0.9, 0.3]], dtype=np.float32)
    mlx_out = to_numpy(ops.argmax(_mx(np_x), axis=-1))
    torch_out = to_numpy(ops.argmax(_torch(np_x), axis=-1))
    np.testing.assert_array_equal(mlx_out, np.array([2.0], dtype=np.float32))
    np.testing.assert_array_equal(torch_out, np.array([2.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# logsumexp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("keepdims", [True, False])
def test_logsumexp_parity(shape: tuple[int, ...], axis: int, keepdims: bool) -> None:
    """``ops.logsumexp`` matches a stable NumPy reference (incl. negatives)."""
    rng = np.random.default_rng(2)
    np_x = (rng.standard_normal(shape) * 3.0 - 1.0).astype(np.float32)
    expected = _np_logsumexp(np_x, axis=axis, keepdims=keepdims)
    _both_equal(np_x, ops.logsumexp, expected, axis=axis, keepdims=keepdims)


# ---------------------------------------------------------------------------
# softmax / log_softmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_softmax_parity(shape: tuple[int, ...], axis: int) -> None:
    """``ops.softmax`` matches ``e / e.sum`` NumPy reference (incl. negatives)."""
    rng = np.random.default_rng(3)
    np_x = (rng.standard_normal(shape) * 2.0 - 0.5).astype(np.float32)
    expected = _np_softmax(np_x, axis=axis)
    _both_equal(np_x, ops.softmax, expected, axis=axis)


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_log_softmax_parity(shape: tuple[int, ...], axis: int) -> None:
    """``ops.log_softmax`` (facade-composed) matches ``log(softmax)`` NumPy."""
    rng = np.random.default_rng(4)
    np_x = (rng.standard_normal(shape) * 2.0 - 0.5).astype(np.float32)
    expected = np.log(_np_softmax(np_x, axis=axis))
    _both_equal(np_x, ops.log_softmax, expected, axis=axis)


# ---------------------------------------------------------------------------
# zeros_like
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_zeros_like_parity(shape: tuple[int, ...]) -> None:
    """``ops.zeros_like`` matches a NumPy zeros reference of the same shape."""
    rng = np.random.default_rng(5)
    np_x = rng.standard_normal(shape).astype(np.float32)
    expected = np.zeros(shape, dtype=np.float32)
    _both_equal(np_x, ops.zeros_like, expected)


# ---------------------------------------------------------------------------
# masked_mean
# ---------------------------------------------------------------------------


def test_masked_mean_hand_computed() -> None:
    """``ops.masked_mean`` matches a hand-computed value: mean of [1,3] is 2."""
    np_x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np_mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    mlx_out = to_numpy(ops.masked_mean(_mx(np_x), _mx(np_mask)))
    torch_out = to_numpy(ops.masked_mean(_torch(np_x), _torch(np_mask)))
    np.testing.assert_allclose(mlx_out, 2.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(torch_out, 2.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(mlx_out, torch_out, rtol=RTOL, atol=ATOL)


def test_masked_mean_all_false_returns_zero() -> None:
    """An all-``False`` mask yields ``0`` (branchless guard), not ``NaN``."""
    np_x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np_mask = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mlx_out = to_numpy(ops.masked_mean(_mx(np_x), _mx(np_mask)))
    torch_out = to_numpy(ops.masked_mean(_torch(np_x), _torch(np_mask)))
    np.testing.assert_allclose(mlx_out, 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(torch_out, 0.0, rtol=RTOL, atol=ATOL)
    assert not np.isnan(mlx_out).any()
    assert not np.isnan(torch_out).any()


def test_masked_mean_broadcast_row_mask() -> None:
    """A ``[B,1]`` row mask must broadcast before BOTH sums: mean of row 0 = 2.0.

    Regression pin for the broadcast bug: broadcasting the mask only in the
    numerator (sum=6) while summing the un-broadcast ``[[1],[0]]`` mask in the
    denominator (sum=1) gave 6.0 on both backends; the correct answer is 2.0.
    A parity-only (same-shape-mask) test cannot catch this.
    """
    np_x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np_mask = np.array([[1.0], [0.0]], dtype=np.float32)  # keep row 0, drop row 1
    mlx_out = to_numpy(ops.masked_mean(_mx(np_x), _mx(np_mask)))
    torch_out = to_numpy(ops.masked_mean(_torch(np_x), _torch(np_mask)))
    np.testing.assert_allclose(mlx_out, 2.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(torch_out, 2.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(mlx_out, torch_out, rtol=RTOL, atol=ATOL)


def test_masked_mean_1d_mask_on_2d() -> None:
    """A 1-D column mask broadcasts across rows: mean over columns 0 and 2 = 4.0."""
    np_x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    np_mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)  # cols 0,2 across both rows
    # kept: 1,3 (row 0) + 5,7 (row 1) -> mean = 16 / 4 = 4.0
    mlx_out = to_numpy(ops.masked_mean(_mx(np_x), _mx(np_mask)))
    torch_out = to_numpy(ops.masked_mean(_torch(np_x), _torch(np_mask)))
    np.testing.assert_allclose(mlx_out, 4.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(torch_out, 4.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(mlx_out, torch_out, rtol=RTOL, atol=ATOL)


def test_masked_mean_bool_mask_parity() -> None:
    """A boolean mask matches the equivalent float mask on both backends."""
    rng = np.random.default_rng(6)
    np_x = rng.standard_normal((3, 4)).astype(np.float32)
    np_mask = np.array(
        [[True, False, True, True], [False, False, True, False], [True, True, False, True]]
    )
    expected = np_x[np_mask].sum() / np_mask.sum()
    mlx_out = to_numpy(ops.masked_mean(_mx(np_x), _mx(np_mask)))
    torch_out = to_numpy(ops.masked_mean(_torch(np_x), _torch(np_mask)))
    np.testing.assert_allclose(mlx_out, expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(torch_out, expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(mlx_out, torch_out, rtol=RTOL, atol=ATOL)

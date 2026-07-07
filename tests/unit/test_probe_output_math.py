"""Oracle tests for the ``ProbeOutput`` math extensions.

Checks ``softmax``, ``log_softmax``, ``n_classes``, and ``reduce`` (incl. the
masked and bound-``self.mask`` cases) against an independent numpy computation
and for MLX↔torch parity, so custom losses read identically on both backends.
"""

from __future__ import annotations

import numpy as np
import pytest

from auto_chasm.metrics import to_numpy
from auto_chasm.outputs import ProbeOutput

_LOGITS = [[2.0, 1.0, 0.1], [-1.0, 0.5, 3.0]]


def _mlx_probe(logits, mask=None):
    """Build a ``ProbeOutput`` with MLX logits (and optional MLX mask)."""
    import mlx.core as mx

    m = None if mask is None else mx.array(mask)
    return ProbeOutput(logits=mx.array(logits), mask=m)


def _torch_probe(logits, mask=None):
    """Build a ``ProbeOutput`` with torch logits (and optional torch mask)."""
    import torch

    m = None if mask is None else torch.tensor(mask)
    return ProbeOutput(logits=torch.tensor(logits), mask=m)


BUILDERS = [
    pytest.param(_mlx_probe, id="mlx"),
    pytest.param(_torch_probe, id="torch"),
]


def _np_softmax(x: np.ndarray) -> np.ndarray:
    """Independent numpy softmax along the last axis."""
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("build", BUILDERS)
def test_softmax_matches_numpy(build):
    """``ProbeOutput.softmax`` matches an independent numpy softmax."""
    pytest.importorskip("torch")
    out = to_numpy(build(_LOGITS).softmax())
    np.testing.assert_allclose(out, _np_softmax(np.array(_LOGITS)), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("build", BUILDERS)
def test_log_softmax_matches_numpy(build):
    """``ProbeOutput.log_softmax`` matches ``log`` of the numpy softmax."""
    pytest.importorskip("torch")
    out = to_numpy(build(_LOGITS).log_softmax())
    np.testing.assert_allclose(out, np.log(_np_softmax(np.array(_LOGITS))), rtol=1e-5, atol=1e-6)


def test_softmax_backend_parity():
    """MLX and torch ``softmax`` agree to numerical tolerance."""
    pytest.importorskip("torch")
    mlx_out = to_numpy(_mlx_probe(_LOGITS).softmax())
    torch_out = to_numpy(_torch_probe(_LOGITS).softmax())
    np.testing.assert_allclose(mlx_out, torch_out, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("build", BUILDERS)
def test_n_classes(build):
    """``n_classes`` is the last-axis size of the logits."""
    pytest.importorskip("torch")
    assert build(_LOGITS).n_classes == 3


@pytest.mark.parametrize("build", BUILDERS)
def test_reduce_plain_mean_when_no_mask(build):
    """``reduce`` with no mask anywhere is a plain mean."""
    pytest.importorskip("torch")
    probe = build(_LOGITS)  # bound mask is None
    values = [1.0, 2.0, 3.0, 4.0]
    got = float(to_numpy(probe.reduce(_as_tensor(probe, values))).reshape(()))
    assert got == pytest.approx(2.5)


@pytest.mark.parametrize("build", BUILDERS)
def test_reduce_with_explicit_mask(build):
    """Masked ``reduce`` averages only the selected elements ([1,3] -> 2.0)."""
    pytest.importorskip("torch")
    probe = build(_LOGITS)
    values = _as_tensor(probe, [1.0, 2.0, 3.0, 4.0])
    mask = _as_tensor(probe, [1.0, 0.0, 1.0, 0.0])
    got = float(to_numpy(probe.reduce(values, mask=mask)).reshape(()))
    assert got == pytest.approx(2.0)  # (1 + 3) / 2


@pytest.mark.parametrize("build", BUILDERS)
def test_reduce_uses_bound_mask(build):
    """``reduce`` falls back to the bound ``self.mask`` when none is passed."""
    pytest.importorskip("torch")
    probe = build(_LOGITS, mask=[1.0, 0.0, 1.0, 0.0])
    values = _as_tensor(probe, [1.0, 2.0, 3.0, 4.0])
    got = float(to_numpy(probe.reduce(values)).reshape(()))
    assert got == pytest.approx(2.0)  # (1 + 3) / 2


def test_reduce_backend_parity():
    """Masked ``reduce`` agrees across MLX and torch."""
    pytest.importorskip("torch")
    mlx_probe = _mlx_probe(_LOGITS)
    torch_probe = _torch_probe(_LOGITS)
    vals = [0.5, 1.5, 2.5, 3.5]
    mask = [1.0, 1.0, 0.0, 1.0]
    mlx_got = float(
        to_numpy(
            mlx_probe.reduce(_as_tensor(mlx_probe, vals), mask=_as_tensor(mlx_probe, mask))
        ).reshape(())
    )
    torch_got = float(
        to_numpy(
            torch_probe.reduce(_as_tensor(torch_probe, vals), mask=_as_tensor(torch_probe, mask))
        ).reshape(())
    )
    assert mlx_got == pytest.approx(torch_got)
    assert mlx_got == pytest.approx((0.5 + 1.5 + 3.5) / 3)


def _as_tensor(probe: ProbeOutput, data):
    """Build a tensor on the same backend as ``probe.logits``."""
    if hasattr(probe.logits, "device"):
        import torch

        return torch.tensor(data)
    import mlx.core as mx

    return mx.array(data)

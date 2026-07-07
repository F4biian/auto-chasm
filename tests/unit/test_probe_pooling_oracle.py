"""Oracle tests for response-granularity pooling.

The bug these guard against: ``granularity="response"`` used to mean-pool
over *every* time step including padding, so a per-sequence probe score
depended on how much padding happened to be in the batch.  The masked mean
must equal the mean over only the valid positions, and must differ from the
naive (padding-contaminated) mean when padding carries different values.
"""

from __future__ import annotations

import numpy as np

from auto_chasm.config import ProbeConfig
from auto_chasm.probe import Probe

# Valid positions [0, 1] hold 1 and 3 (mean 2.0); padded positions [2, 3]
# hold 100 so a naive mean (51.0) is obviously wrong.
LOGITS = np.array([[[1.0], [3.0], [100.0], [100.0]]], dtype=np.float32)  # [1, 4, 1]
MASK = np.array([[1, 1, 0, 0]], dtype=np.float32)  # [1, 4]
EXPECTED_MASKED = 2.0
NAIVE = 51.0


def _probe(backend: str) -> Probe:
    cfg = ProbeConfig(name="r", layers=[-1], granularity="response")
    return Probe(cfg, hidden_dim=8, backend_name=backend)


def test_masked_mean_ignores_padding_mlx():
    import mlx.core as mx

    probe = _probe("mlx")
    out = probe._masked_mean_over_time(mx.array(LOGITS), mx.array(MASK))
    out = float(np.array(out)[0, 0])
    assert abs(out - EXPECTED_MASKED) < 1e-5
    assert abs(out - NAIVE) > 1.0  # padding really was excluded


def test_masked_mean_ignores_padding_torch():
    import torch

    probe = _probe("torch")
    out = probe._masked_mean_over_time(torch.tensor(LOGITS), torch.tensor(MASK))
    out = float(out[0, 0])
    assert abs(out - EXPECTED_MASKED) < 1e-5
    assert abs(out - NAIVE) > 1.0


def test_no_mask_is_plain_mean_mlx():
    import mlx.core as mx

    probe = _probe("mlx")
    out = float(np.array(probe._masked_mean_over_time(mx.array(LOGITS), None))[0, 0])
    assert abs(out - NAIVE) < 1e-4


def test_mlx_torch_pooling_parity():
    import mlx.core as mx
    import torch

    out_mlx = float(
        np.array(_probe("mlx")._masked_mean_over_time(mx.array(LOGITS), mx.array(MASK)))[0, 0]
    )
    out_torch = float(
        _probe("torch")._masked_mean_over_time(torch.tensor(LOGITS), torch.tensor(MASK))[0, 0]
    )
    assert abs(out_mlx - out_torch) < 1e-5

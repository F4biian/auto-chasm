"""Oracle tests for closed-form steering.

These assert *correct results* against independent ground truth, not just
"runs without crashing":

* scale linearity — the closed-form shift must scale linearly with
  ``config.scale`` (and vanish at ``scale=0``);
* backend parity — the MLX and PyTorch implementations must produce
  numerically identical shifts for identical inputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from auto_chasm.steering import _steer_mlx, _steer_torch

H = 4
W = np.array([[0.5, -0.3, 0.2, 0.1]], dtype=np.float32)  # head weight [1, H]
B = np.array([0.0], dtype=np.float32)
HIDDEN = np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, -0.1, 0.2, 0.0]]], dtype=np.float32)
# The closed-form contract assumes the probe logit equals w.h + b at each
# position, so derive it from the head rather than inventing a value.
LOGITS = (HIDDEN @ W.reshape(-1, 1)).reshape(1, -1) + B  # => [[0.09, 0.32]]
MEAN0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
MEAN1 = np.array([1.0, -0.5, 0.5, 0.2], dtype=np.float32)
DIRECTION = MEAN1 - MEAN0


def _mlx_inputs():
    import mlx.core as mx

    head = SimpleNamespace(weight=mx.array(W), bias=mx.array(B))
    return (
        mx.array(HIDDEN),
        head,
        mx.array(LOGITS),
        mx.array(MEAN0),
        mx.array(MEAN1),
        mx.array(DIRECTION),
    )


def _torch_inputs():
    import torch

    head = SimpleNamespace(weight=torch.tensor(W), bias=torch.tensor(B))
    return (
        torch.tensor(HIDDEN),
        head,
        torch.tensor(LOGITS),
        torch.tensor(MEAN0),
        torch.tensor(MEAN1),
        torch.tensor(DIRECTION),
    )


def test_scale_zero_is_identity_mlx():
    hidden, head, logits, m0, m1, d = _mlx_inputs()
    out = _steer_mlx(hidden, head, logits, "nullify", m0, m1, d, scale=0.0)
    assert np.allclose(np.array(out), HIDDEN, atol=1e-6)


def test_scale_is_linear_mlx():
    """delta(scale=2) must be exactly 2x delta(scale=1)."""
    hidden, head, logits, m0, m1, d = _mlx_inputs()
    out1 = np.array(_steer_mlx(hidden, head, logits, "nullify", m0, m1, d, scale=1.0))
    out2 = np.array(_steer_mlx(hidden, head, logits, "nullify", m0, m1, d, scale=2.0))
    delta1 = out1 - HIDDEN
    delta2 = out2 - HIDDEN
    assert np.allclose(delta2, 2.0 * delta1, atol=1e-5)
    # the shift must be non-trivial (guards against a silent no-op)
    assert np.abs(delta1).max() > 1e-4


def test_scale_is_linear_torch():
    hidden, head, logits, m0, m1, d = _torch_inputs()
    out1 = _steer_torch(hidden, head, logits, "nullify", m0, m1, d, scale=1.0).numpy()
    out2 = _steer_torch(hidden, head, logits, "nullify", m0, m1, d, scale=2.0).numpy()
    delta1 = out1 - HIDDEN
    delta2 = out2 - HIDDEN
    assert np.allclose(delta2, 2.0 * delta1, atol=1e-5)
    assert np.abs(delta1).max() > 1e-4


def test_nullify_drives_logit_to_zero_mlx():
    """After nullify, the head logit at the steered (last) position is ~0."""
    import mlx.core as mx

    hidden, head, logits, m0, m1, d = _mlx_inputs()
    out = _steer_mlx(hidden, head, logits, "nullify", m0, m1, d, scale=1.0)
    last = out[:, -1, :]
    new_logit = float(mx.sum(last * head.weight) + head.bias)
    assert abs(new_logit) < 1e-3


def test_mlx_torch_parity_all_methods():
    """MLX and PyTorch steering must agree numerically for every method."""
    for method in ("nullify", "push_to_mean", "boundary"):
        m_hidden, m_head, m_logits, m_m0, m_m1, m_d = _mlx_inputs()
        t_hidden, t_head, t_logits, t_m0, t_m1, t_d = _torch_inputs()
        out_mlx = np.array(
            _steer_mlx(m_hidden, m_head, m_logits, method, m_m0, m_m1, m_d, scale=1.3)
        )
        out_torch = _steer_torch(
            t_hidden, t_head, t_logits, method, t_m0, t_m1, t_d, scale=1.3
        ).numpy()
        assert np.allclose(out_mlx, out_torch, atol=1e-5), f"backend mismatch for {method}"

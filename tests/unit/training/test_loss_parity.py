"""MLX↔torch parity oracles for ``JointLoss`` (Phase 3b).

The joint LM + probe loss is now computed by a SINGLE backend-agnostic
``__call__`` path (no ``_compute_mlx`` / ``_compute_torch``). These tests feed the
SAME fixed logits and labels to both backends and assert the returned
``(total, ntoks, components)`` are numerically identical — the parity guarantee
proving the one path behaves the same on MLX and torch. Every probe-loss branch
is exercised. No model is loaded: the fake model returns fixed logits, so both
backends see identical inputs.

Component keys are the new term-name scheme: ``"lm_head"`` for the LM term and the
probe's own name for each head.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from auto_chasm import JointLoss, ops
from auto_chasm.metrics import to_numpy

torch = pytest.importorskip("torch")

# Shared batch: B=2, T=5 -> T-1=4 target tokens; valid window = steps 1..3 (3/seq).
_BATCH = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]], dtype=np.int64)
_LENGTHS = np.array([[1, 4], [1, 4]], dtype=np.int64)
_RNG = np.random.default_rng(0)
_LM = _RNG.standard_normal((2, 4, 6)).astype(np.float32)  # lm logits [B, T-1, vocab]

# Per-probe head logits (shapes the loss expects):
_BIN = _RNG.standard_normal((2, 4)).astype(np.float32)  # token binary (bce): [B, T]
_SCALAR = _RNG.standard_normal((2, 4, 1)).astype(np.float32)  # token scalar (mse/mae): [B, T, 1]
_MC = _RNG.standard_normal((2, 4, 3)).astype(np.float32)  # token multi-class (ce): [B, T, C]
_SEQ1 = _RNG.standard_normal((2, 1)).astype(np.float32)  # response scalar (seq): [B, 1]
_SEQ3 = _RNG.standard_normal((2, 3)).astype(np.float32)  # response multi-class (seq): [B, C]

_BIN_LAB = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=np.float32)  # 0/1 labels
_MC_LAB = np.array([[0, 1, 2, 0, 1], [2, 1, 0, 2, 0]], dtype=np.int64)  # class indices
_REG_LAB = np.array([[0, 1, 2, 3, 4], [5, 4, 3, 2, 1]], dtype=np.float32)  # continuous


class _FakeModel:
    """Returns fixed ``(lm_logits, probe_dict)``; accepts the trainer's mask kwarg."""

    def __init__(self, lm: Any, probes: dict[str, Any]) -> None:
        self._lm = lm
        self._probes = probes

    def __call__(self, inputs: Any, mask: Any = None) -> tuple[Any, dict[str, Any]]:
        """Ignore inputs; return the fixed logits."""
        return self._lm, self._probes


def _to(backend: str, arr: np.ndarray) -> Any:
    """Materialize a NumPy array as an MLX array or a torch tensor."""
    return mx.array(arr) if backend == "mlx" else torch.tensor(arr)


def _run(
    backend: str,
    jl: JointLoss,
    probes_np: dict[str, np.ndarray],
    labels_np: Any,
    lm_np: np.ndarray = _LM,
    batch_np: np.ndarray = _BATCH,
    lengths_np: np.ndarray = _LENGTHS,
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    """Run ``jl`` on one backend and return numpy ``(total, ntoks, components)``."""
    lm = _to(backend, lm_np)
    probes = {k: _to(backend, v) for k, v in probes_np.items()}
    batch = _to(backend, batch_np)
    labels: Any = (
        {k: _to(backend, v) for k, v in labels_np.items()}
        if isinstance(labels_np, dict)
        else _to(backend, labels_np)
    )
    lengths = _to(backend, lengths_np)
    total, ntoks, comps = jl(_FakeModel(lm, probes), batch, labels, lengths)
    return to_numpy(total), float(to_numpy(ntoks)), {k: to_numpy(v) for k, v in comps.items()}


def _assert_parity(
    jl: JointLoss, probes_np: dict[str, np.ndarray], labels_np: Any, **kw: Any
) -> tuple[Any, Any]:
    """Assert MLX and torch return identical total / ntoks / component values."""
    m = _run("mlx", jl, probes_np, labels_np, **kw)
    t = _run("torch", jl, probes_np, labels_np, **kw)
    np.testing.assert_allclose(m[0], t[0], rtol=1e-5, atol=1e-6)  # total scalar
    assert m[1] == pytest.approx(t[1])  # ntoks
    assert set(m[2]) == set(t[2]), (set(m[2]), set(t[2]))  # identical component keys
    for k in m[2]:
        np.testing.assert_allclose(m[2][k], t[2][k], rtol=1e-5, atol=1e-6, err_msg=k)
    return m, t


# ``weights={"lm_head": 0.0}`` reproduces the old ``lm_weight=0.0`` (pure-probe);
# an un-listed probe defaults to weight 1.0 and loss ``"bce"``.
def _probe_only(loss: str | None = None) -> JointLoss:
    """A pure-probe ``JointLoss`` (no LM term) for the single ``"p"`` head."""
    losses = {"p": loss} if loss is not None else None
    return JointLoss(weights={"lm_head": 0.0}, losses=losses)


def test_parity_token_bce_probe_only() -> None:
    """Token BCE, no LM term: MLX == torch."""
    _assert_parity(_probe_only("bce"), {"p": _BIN}, _BIN_LAB)


def test_parity_token_bce_with_lm() -> None:
    """Token BCE + LM cross-entropy: both terms match across backends."""
    m, _ = _assert_parity(JointLoss(), {"p": _BIN}, _BIN_LAB)
    assert "lm_head" in m[2]


def test_parity_token_ce_unweighted() -> None:
    """Token multi-class CE: MLX == torch."""
    _assert_parity(_probe_only("ce"), {"p": _MC}, _MC_LAB)


def test_parity_token_ce_weighted() -> None:
    """Class-weighted token CE (exercises weighted_ce on both backends)."""
    jl = JointLoss(weights={"lm_head": 0.0}, losses={"p": "ce"}, class_weights=[1.0, 2.0, 0.5])
    _assert_parity(jl, {"p": _MC}, _MC_LAB)


def test_parity_token_mse() -> None:
    """Token MSE on a scalar [B,T,1] head (exercises the scalar-head squeeze)."""
    _assert_parity(_probe_only("mse"), {"p": _SCALAR}, _REG_LAB)


def test_parity_token_mae() -> None:
    """Token MAE on a scalar head: MLX == torch."""
    _assert_parity(_probe_only("mae"), {"p": _SCALAR}, _REG_LAB)


def test_parity_seq_bce() -> None:
    """Response-level (pooled) BCE: MLX == torch (seq target reduction)."""
    _assert_parity(_probe_only("bce"), {"p": _SEQ1}, _BIN_LAB)


def test_parity_seq_mse() -> None:
    """Response-level MSE: MLX == torch."""
    _assert_parity(_probe_only("mse"), {"p": _SEQ1}, _REG_LAB)


def test_parity_seq_ce() -> None:
    """Response-level multi-class CE: MLX == torch (rounded pooled target)."""
    _assert_parity(_probe_only("ce"), {"p": _SEQ3}, _MC_LAB)


def test_parity_seq_mae() -> None:
    """Response-level MAE: MLX == torch (covers the seq-mae branch)."""
    _assert_parity(_probe_only("mae"), {"p": _SEQ1}, _REG_LAB)


def _seq_custom(logits: Any, targets: Any, mask: Any) -> Any:
    """Backend-agnostic masked-L1 for a response-level ``[B]`` logit."""
    return ops.masked_mean(ops.abs(logits - targets), mask)


def test_parity_seq_custom() -> None:
    """Response-level custom loss: MLX == torch (covers the seq-custom branch)."""
    _assert_parity(_probe_only(_seq_custom), {"p": _SEQ1}, _REG_LAB)


def _custom(logits: Any, targets: Any, mask: Any) -> Any:
    """A backend-agnostic custom probe loss (masked L1) written with ``ops``."""
    return ops.masked_mean(ops.abs(logits[..., 0] - targets), mask)


def test_parity_custom_fn() -> None:
    """A user custom loss written with ``ops`` is backend-identical."""
    _assert_parity(_probe_only(_custom), {"p": _SCALAR}, _REG_LAB)


def test_parity_multi_probe() -> None:
    """Two heads with different losses + LM: component keys and values all match."""
    jl = JointLoss(losses={"a": "bce", "b": "ce"})
    m, _ = _assert_parity(jl, {"a": _BIN, "b": _MC}, {"a": _BIN_LAB, "b": _MC_LAB})
    # New term-name component keys: the LM head plus each probe's own name.
    assert {"lm_head", "a", "b"} == set(m[2])


def test_parity_single_token_batch() -> None:
    """A single-token batch (empty targets) returns a finite 0 on both backends."""
    jl = JointLoss()
    batch1 = np.array([[3], [4]], dtype=np.int64)
    lengths1 = np.array([[0, 1], [0, 1]], dtype=np.int64)
    m = _run("mlx", jl, {"p": _BIN}, _BIN_LAB, batch_np=batch1, lengths_np=lengths1)
    t = _run("torch", jl, {"p": _BIN}, _BIN_LAB, batch_np=batch1, lengths_np=lengths1)
    assert m[1] == 0.0 and t[1] == 0.0
    # No next-token targets -> nothing supervised -> empty components on BOTH
    # backends (parity holds); the old {"lm_head": 0} was a fabricated term (m18).
    assert set(m[2]) == set(t[2]) == set()
    np.testing.assert_allclose(m[0], t[0], atol=1e-6)

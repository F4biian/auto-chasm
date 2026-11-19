"""Tests for steering and class means.

Covers:
  * ``src/auto_chasm/steering.py`` — ``SteeringHook`` (direct ``steer`` API:
    ``_nullify`` / ``_push_to_mean`` / ``_boundary``), ``scale`` handling,
    enable/disable lifecycle, and the closed-form ``build_auto_steer_fn`` /
    ``_steer_mlx`` / ``_steer_torch`` used during generation.
  * ``src/auto_chasm/class_means.py`` — ``compute_class_means`` (model-level,
    with ``-100`` masking and length gating).
  * ``src/auto_chasm/utils.py`` — ``compute_class_means`` (the per-class
    averaging helper the hook uses via ``compute_geometry``).
  * ``SteeringConfig`` validation in ``src/auto_chasm/config.py``.

Every assertion here is against an independent, hand-computed oracle so a
silently-wrong number fails loudly (e.g. a ``scale`` that is silently ignored,
making steering a no-op).  Tests named ``test_BUG_*`` are regression tests for
specific past defects; the rest are general regression coverage.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm import Model
from auto_chasm.config import ProbeConfig, SteeringConfig
from auto_chasm.steering import SteeringHook, _steer_mlx, _steer_torch, build_auto_steer_fn
from auto_chasm.utils import compute_class_means as _mean_helper

# ===========================================================================
# Helpers — tiny deterministic models, no network.
# ===========================================================================


class _TinyMlx(nn.Module):
    """Tiny MLX transformer-shaped model: embedding -> linear layers -> head."""

    def __init__(self, h: int = 4, v: int = 8, layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(v, h)
        self.layers = [nn.Linear(h, h) for _ in range(layers)]
        self.output_proj = nn.Linear(h, v)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h)


class _Cfg:
    """Minimal config exposing what the probe engine reads."""

    def __init__(self, h: int, v: int, layers: int) -> None:
        self.hidden_size = h
        self.num_hidden_layers = layers
        self.vocab_size = v


def _mlx_model(h: int = 4, v: int = 8, layers: int = 2) -> Model:
    base = _TinyMlx(h, v, layers)
    m = Model(base, None, "mlx")
    m.model.config = _Cfg(h, v, layers)
    return m


def _torch_model(h: int = 4, v: int = 8, layers: int = 2) -> Model:
    import torch.nn as tnn

    class _TinyTorch(tnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = tnn.Embedding(v, h)
            self.layers = tnn.ModuleList([tnn.Linear(h, h) for _ in range(layers)])
            self.output_proj = tnn.Linear(h, v)

        def forward(self, x: Any) -> Any:
            out = self.embedding(x)
            for layer in self.layers:
                out = layer(out)
            return self.output_proj(out)

    m = Model(_TinyTorch(), None, "torch")
    m.model.config = _Cfg(h, v, layers)
    return m


def _np(t: Any) -> np.ndarray:
    if t is None:
        raise AssertionError("expected a tensor, got None")
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    mx.eval(t)
    return np.array(t)


# Shared closed-form fixtures (probe logit == w.h + b by construction).
_W = np.array([[0.5, -0.3, 0.2, 0.1]], dtype=np.float32)
_B = np.array([0.2], dtype=np.float32)
_HIDDEN = np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, -0.1, 0.2, 0.0]]], dtype=np.float32)
_LOGITS = (_HIDDEN @ _W.reshape(-1, 1)).reshape(1, -1) + _B
_MEAN0 = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)
_MEAN1 = np.array([1.0, -0.5, 0.5, 0.2], dtype=np.float32)
_DIR = _MEAN1 - _MEAN0


def _mlx_cf_inputs():
    head = SimpleNamespace(weight=mx.array(_W), bias=mx.array(_B))
    return (
        mx.array(_HIDDEN),
        head,
        mx.array(_LOGITS),
        mx.array(_MEAN0),
        mx.array(_MEAN1),
        mx.array(_DIR),
    )


def _torch_cf_inputs():
    import torch

    head = SimpleNamespace(weight=torch.tensor(_W), bias=torch.tensor(_B))
    return (
        torch.tensor(_HIDDEN),
        head,
        torch.tensor(_LOGITS),
        torch.tensor(_MEAN0),
        torch.tensor(_MEAN1),
        torch.tensor(_DIR),
    )


# ===========================================================================
# 1. scale actually applies (direct SteeringHook.steer API)
# ===========================================================================


def _direct_hook(method: str, scale: float) -> SteeringHook:
    """Build a direct-API hook with mean0=origin, mean1=e0, head along e0."""
    mean0 = mx.array([0.0, 0.0, 0.0, 0.0])
    mean1 = mx.array([1.0, 0.0, 0.0, 0.0])
    hw = mx.array([[2.0, 0.0, 0.0, 0.0]])
    hb = mx.array([0.0])
    h = SteeringHook("p", SteeringConfig(method=method, scale=scale))
    h.compute_geometry({0: [mean0], 1: [mean1]}, hw, hb)
    h.enable()
    return h


@pytest.mark.parametrize("method", ["nullify", "push_to_mean", "boundary"])
def test_direct_scale_zero_is_identity(method: str) -> None:
    """scale=0 must leave the hidden state exactly unchanged for every method."""
    h = _direct_hook(method, 0.0)
    hidden = mx.array([[[0.5, 1.0, 2.0, 3.0]]])
    logits = mx.array([[1.0]])
    out = _np(h.steer(hidden, None, logits))
    assert np.allclose(out, _np(hidden), atol=1e-6), f"{method}: scale=0 changed hidden"


@pytest.mark.parametrize("method", ["nullify", "push_to_mean", "boundary"])
def test_direct_scale_is_linear(method: str) -> None:
    """delta(scale=2) == 2*delta(scale=1) and the edit is non-trivial (no silent no-op)."""
    hidden = mx.array([[[0.5, 1.0, 2.0, 3.0]]])
    logits = mx.array([[1.0]])
    base = _np(hidden)
    d1 = _np(_direct_hook(method, 1.0).steer(hidden, None, logits)) - base
    d2 = _np(_direct_hook(method, 2.0).steer(hidden, None, logits)) - base
    assert np.abs(d1).max() > 1e-4, f"{method}: scale=1 produced a silent no-op"
    assert np.allclose(d2, 2.0 * d1, atol=1e-5), f"{method}: not linear in scale"


@pytest.mark.parametrize("method", ["nullify", "push_to_mean", "boundary"])
def test_direct_negative_scale_flips_direction(method: str) -> None:
    """delta(scale=-1) == -delta(scale=1): a negative scale reverses the edit."""
    hidden = mx.array([[[0.5, 1.0, 2.0, 3.0]]])
    logits = mx.array([[1.0]])
    base = _np(hidden)
    d1 = _np(_direct_hook(method, 1.0).steer(hidden, None, logits)) - base
    dn = _np(_direct_hook(method, -1.0).steer(hidden, None, logits)) - base
    assert np.allclose(dn, -d1, atol=1e-5), f"{method}: negative scale did not flip"


# ===========================================================================
# 2. nullify — projection onto the direction is ~0 afterward (hand math)
# ===========================================================================


def test_direct_nullify_removes_direction_component() -> None:
    """After nullify (scale=1) the projection of hidden onto the unit direction is ~0."""
    h = _direct_hook("nullify", 1.0)
    hidden = mx.array([[[0.5, 1.0, 2.0, 3.0]]])
    out = _np(h.steer(hidden, None, mx.array([[1.0]])))
    d = _np(h._direction)
    dn = d / np.linalg.norm(d)
    proj_before = float((_np(hidden)[0, 0] * dn).sum())
    proj_after = float((out[0, 0] * dn).sum())
    assert abs(proj_before) > 1e-3, "test setup: projection should start non-zero"
    assert abs(proj_after) < 1e-5, "nullify did not remove the direction component"


# ===========================================================================
# 3. push_to_mean — direction & magnitude vs hand math (direct API)
# ===========================================================================


def test_direct_push_to_mean_adds_scale_times_direction() -> None:
    """Direct push_to_mean = hidden + scale*direction (exact hand math)."""
    h = _direct_hook("push_to_mean", 1.5)
    hidden = mx.array([[[0.5, 1.0, 2.0, 3.0]]])
    out = _np(h.steer(hidden, None, mx.array([[1.0]])))
    expected = _np(hidden) + 1.5 * _np(h._direction)
    assert np.allclose(out, expected, atol=1e-5)


# ===========================================================================
# 4. boundary — geometry vs hand math (direct API)
# ===========================================================================


def test_direct_boundary_shift_matches_hand_math() -> None:
    """Direct boundary shift = scale*(|logit|/||w||)*unit_dir (exact)."""
    mean0 = mx.array([0.0, 0.0])
    mean1 = mx.array([3.0, 0.0])  # unit_dir = [1, 0]
    hw = mx.array([[2.0, 0.0]])  # ||w|| = 2
    h = SteeringHook("p", SteeringConfig(method="boundary", scale=1.0))
    h.compute_geometry({0: [mean0], 1: [mean1]}, hw, mx.array([0.0]))
    h.enable()
    hidden = mx.array([[[1.0, 5.0]]])
    out = _np(h.steer(hidden, None, mx.array([[4.0]])))
    # shift = 1*(|4|/2)*[1,0] = [2,0]; new = [3,5]
    assert np.allclose(out[0, 0], [3.0, 5.0], atol=1e-4)


# ===========================================================================
# 5. Closed-form steering oracles (generation path) — all methods
# ===========================================================================


def test_cf_nullify_drives_logit_to_zero_mlx() -> None:
    hidden, head, logits, m0, m1, d = _mlx_cf_inputs()
    out = _steer_mlx(hidden, head, logits, "nullify", m0, m1, d, scale=1.0)
    last = _np(out)[0, -1]
    new_logit = float((last * _W[0]).sum() + _B[0])
    assert abs(new_logit) < 1e-3


def test_cf_push_to_mean_drives_logit_to_mean0_logit_mlx() -> None:
    """push_to_mean drives the steered logit to the class-0 (suppression) logit."""
    hidden, head, logits, m0, m1, d = _mlx_cf_inputs()
    out = _steer_mlx(hidden, head, logits, "push_to_mean", m0, m1, d, scale=1.0)
    last = _np(out)[0, -1]
    new_logit = float((last * _W[0]).sum() + _B[0])
    target = float((_W[0] * _MEAN0).sum() + _B[0])
    assert abs(new_logit - target) < 1e-3


def test_cf_boundary_lands_on_midpoint_projection_mlx() -> None:
    """Boundary pushes the projection onto the class axis to the class-mean midpoint."""
    hidden, head, logits, m0, m1, d = _mlx_cf_inputs()
    out = _steer_mlx(hidden, head, logits, "boundary", m0, m1, d, scale=1.0)
    last = _np(out)[0, -1]
    dn = _DIR / np.linalg.norm(_DIR)
    proj_new = float((last * dn).sum())
    midpoint = float(((_MEAN0 + _MEAN1) / 2.0 * dn).sum())
    assert abs(proj_new - midpoint) < 1e-3


def test_cf_scale_zero_identity_and_linear_all_methods_mlx() -> None:
    for method in ("nullify", "push_to_mean", "boundary"):
        hidden, head, logits, m0, m1, d = _mlx_cf_inputs()
        o0 = _np(_steer_mlx(hidden, head, logits, method, m0, m1, d, scale=0.0))
        o1 = _np(_steer_mlx(hidden, head, logits, method, m0, m1, d, scale=1.0))
        o2 = _np(_steer_mlx(hidden, head, logits, method, m0, m1, d, scale=2.0))
        assert np.allclose(o0, _HIDDEN, atol=1e-6), f"{method}: scale=0 not identity"
        d1, d2 = o1 - _HIDDEN, o2 - _HIDDEN
        assert np.abs(d1).max() > 1e-4, f"{method}: silent no-op at scale=1"
        assert np.allclose(d2, 2.0 * d1, atol=1e-5), f"{method}: not linear in scale"


def test_cf_only_last_position_is_edited_mlx() -> None:
    """The closed-form path must edit ONLY the last position (the next-token slot)."""
    hidden, head, logits, m0, m1, d = _mlx_cf_inputs()
    out = _np(_steer_mlx(hidden, head, logits, "nullify", m0, m1, d, scale=1.0))
    assert np.allclose(out[:, :-1, :], _HIDDEN[:, :-1, :], atol=1e-6), "non-last edited"
    assert not np.allclose(out[:, -1, :], _HIDDEN[:, -1, :], atol=1e-4), "last unchanged"


def test_cf_mlx_torch_parity_all_methods() -> None:
    pytest.importorskip("torch")
    for method in ("nullify", "push_to_mean", "boundary"):
        m = _mlx_cf_inputs()
        t = _torch_cf_inputs()
        out_mlx = _np(_steer_mlx(m[0], m[1], m[2], method, m[3], m[4], m[5], scale=1.3))
        out_torch = _np(_steer_torch(t[0], t[1], t[2], method, t[3], t[4], t[5], scale=1.3))
        assert np.allclose(out_mlx, out_torch, atol=1e-5), f"backend mismatch: {method}"


# ===========================================================================
# 6. No-op refusal & degenerate geometry
# ===========================================================================


def test_enable_steering_without_class_means_raises() -> None:
    """Enabling steering with no geometry and no custom fn must RAISE (intended)."""
    m = _mlx_model()
    m.attach_probe(
        ProbeConfig(name="p0", layers=[0], aggregation="last", module_config={"out_features": 1})
    )
    with pytest.raises(ValueError, match="[Rr]efusing a no-op"):
        m.enable_steering("p0", config=SteeringConfig(method="nullify"))


def test_build_auto_steer_fn_returns_none_without_geometry() -> None:
    """The builder yields None (not a silent identity fn) when geometry is missing."""
    hook = SteeringHook("p", SteeringConfig(method="nullify"))
    assert build_auto_steer_fn(hook) is None


def test_degenerate_identical_means_no_nan() -> None:
    """Identical class means (zero direction) must not produce NaN/inf hidden states."""
    m = _mlx_model()
    m.attach_probe(
        ProbeConfig(name="p0", layers=[0], aggregation="last", module_config={"out_features": 1})
    )
    same = mx.array([1.0, 1.0, 1.0, 1.0])
    m.enable_steering(
        "p0", config=SteeringConfig(method="nullify"), class_means={"mean_0": same, "mean_1": same}
    )
    out = m.forward(mx.array([[1, 2, 3]]))
    assert bool(mx.all(mx.isfinite(out.lm_logits)).item()), "degenerate geometry produced NaN/inf"


def test_steer_without_enable_returns_unchanged() -> None:
    """A hook with geometry but not enabled is a strict no-op."""
    h = _direct_hook("push_to_mean", 5.0)
    h.disable()
    hidden = mx.array([[[0.5, 1.0, 2.0, 3.0]]])
    out = _np(h.steer(hidden, None, mx.array([[1.0]])))
    assert np.allclose(out, _np(hidden), atol=1e-6)


# ===========================================================================
# 7. compute_class_means correctness (model-level, class_means.py)
# ===========================================================================

# Probe captures hidden from model.forward(tokens[:, :-1]); labels[1:] align to
# captures. Labels: idx1,2 -> class 0,0; idx3 -> -100 (ignored); idx4 -> class 1.
_TOKENS = [1, 2, 3, 4, 5]
_LABELS = [0, 0, 1, -100, 1]
_TOKENS_FLIP_IGNORED = [1, 2, 6, 4, 5]  # differ only at the -100 position's source token


def _attach(m: Model) -> Any:
    return m.attach_probe(
        ProbeConfig(name="p", layers=[0], aggregation="last", module_config={"out_features": 1})
    )


def _gt_means(probe: Any, m: Model, tokens, labels):
    probe.clear_captured()
    if m.backend.name == "torch":
        import torch

        with torch.no_grad():
            m.forward(torch.tensor([tokens])[:, :-1])
        h = probe.get_captured_states()[0].float().detach().cpu().numpy()[0]
    else:
        m.forward(mx.array([tokens])[:, :-1])
        cap = probe.get_captured_states()[0]
        mx.eval(cap)
        h = np.array(cap)[0]
    bl = np.array(labels[1:])
    return h[bl == 0].mean(axis=0), h[bl == 1].mean(axis=0)


def test_class_means_equal_hand_computed_mlx() -> None:
    m = _mlx_model()
    p = _attach(m)
    gt0, gt1 = _gt_means(p, m, _TOKENS, _LABELS)
    res = m.compute_class_means([(_TOKENS, _LABELS)])
    assert np.allclose(_np(res["p"]["mean_0"]), gt0, atol=1e-5)
    assert np.allclose(_np(res["p"]["mean_1"]), gt1, atol=1e-5)


def test_class_means_exclude_masked_tokens_mlx() -> None:
    """Flipping the token at a -100 position must NOT change the means (it is excluded)."""
    m = _mlx_model()
    _attach(m)
    base = m.compute_class_means([(_TOKENS, _LABELS)])
    flip = m.compute_class_means([(_TOKENS_FLIP_IGNORED, _LABELS)])
    assert np.allclose(_np(base["p"]["mean_0"]), _np(flip["p"]["mean_0"]), atol=1e-6)
    assert np.allclose(_np(base["p"]["mean_1"]), _np(flip["p"]["mean_1"]), atol=1e-6)


def test_class_means_zero_examples_is_zero_not_nan_mlx() -> None:
    """A class with zero examples yields a finite (zero) vector, never NaN."""
    m = _mlx_model()
    _attach(m)
    res = m.compute_class_means([(_TOKENS, [1, 1, 1, 1, 1])])  # no class-0 tokens
    mean0 = _np(res["p"]["mean_0"])
    assert np.all(np.isfinite(mean0)), "empty class produced non-finite mean"
    assert np.allclose(mean0, 0.0), "empty class should be the zero vector"


def test_class_means_all_one_class_mlx() -> None:
    """When every token is class 1, mean_1 equals the unmasked mean and mean_0 is zero."""
    m = _mlx_model()
    p = _attach(m)
    labels = [1, 1, 1, 1, 1]
    p.clear_captured()
    m.forward(mx.array([_TOKENS])[:, :-1])
    cap = p.get_captured_states()[0]
    mx.eval(cap)
    h = np.array(cap)[0]
    gt1 = h.mean(axis=0)  # all 4 captured positions are class 1
    res = m.compute_class_means([(_TOKENS, labels)])
    assert np.allclose(_np(res["p"]["mean_1"]), gt1, atol=1e-5)
    assert np.allclose(_np(res["p"]["mean_0"]), 0.0, atol=1e-6)


def test_class_means_cross_backend_parity() -> None:
    """MLX and Torch class means agree (identical weights) with -100 masking."""
    import torch

    mx.random.seed(0)
    torch.manual_seed(0)
    mm = _mlx_model()
    tm = _torch_model()
    # Copy MLX weights into the torch model so the two forwards are identical.
    base_m, base_t = mm.model, tm.model
    base_t.embedding.weight.data = torch.tensor(_np(base_m.embedding.weight))
    for i in range(len(base_m.layers)):
        base_t.layers[i].weight.data = torch.tensor(_np(base_m.layers[i].weight))
        base_t.layers[i].bias.data = torch.tensor(_np(base_m.layers[i].bias))
    base_t.output_proj.weight.data = torch.tensor(_np(base_m.output_proj.weight))
    base_t.output_proj.bias.data = torch.tensor(_np(base_m.output_proj.bias))
    _attach(mm)
    _attach(tm)
    rM = mm.compute_class_means([(_TOKENS, _LABELS)])
    rT = tm.compute_class_means([(_TOKENS, _LABELS)])
    assert np.allclose(_np(rM["p"]["mean_0"]), _np(rT["p"]["mean_0"]), atol=1e-4)
    assert np.allclose(_np(rM["p"]["mean_1"]), _np(rT["p"]["mean_1"]), atol=1e-4)


def test_class_means_multibatch_accumulation_mlx() -> None:
    """Means must be the global per-class average across MULTIPLE batches.

    Variable-length samples (sorted + padded into different batches) stress the
    running sum/count accumulation; a per-batch reset would silently corrupt the
    axis.  Oracle: hand-accumulate over every unmasked token across all samples.
    """
    m = _mlx_model()
    p = _attach(m)
    data = [
        ([1, 2, 3, 4, 5], [0, 0, 1, 1, 0]),
        ([6, 7, 3], [1, 0, 1]),
        ([2, 4, 6, 1, 3, 5, 7], [0, 1, 1, 0, 1, 0, 1]),
    ]
    s0 = np.zeros(4)
    s1 = np.zeros(4)
    c0 = c1 = 0
    for toks, labs in data:
        p.clear_captured()
        m.forward(mx.array([toks])[:, :-1])
        cap = p.get_captured_states()[0]
        mx.eval(cap)
        h = np.array(cap)[0]
        bl = np.array(labs[1:])
        for j in range(len(bl)):
            if bl[j] == 0:
                s0 += h[j]
                c0 += 1
            elif bl[j] == 1:
                s1 += h[j]
                c1 += 1
    gt0, gt1 = s0 / c0, s1 / c1
    # batch_size=1 forces three separate batches.
    res = m.compute_class_means(data, batch_size=1)
    assert np.allclose(_np(res["p"]["mean_0"]), gt0, atol=1e-4), "multi-batch mean_0 corrupted"
    assert np.allclose(_np(res["p"]["mean_1"]), gt1, atol=1e-4), "multi-batch mean_1 corrupted"


# ===========================================================================
# 8. compute_class_means averaging helper (utils.py)
# ===========================================================================


def test_mean_helper_averages_correctly() -> None:
    res = _mean_helper({0: [mx.array([0.0, 0.0]), mx.array([2.0, 4.0])]})
    assert np.allclose(_np(res[0]), [1.0, 2.0])


def test_mean_helper_skips_empty_class() -> None:
    res = _mean_helper({0: [mx.array([1.0, 2.0])], 1: []})
    assert set(res.keys()) == {0}, "empty class must be skipped, not crash or fabricate"


def test_mean_helper_single_class() -> None:
    res = _mean_helper({1: [mx.array([1.0, 2.0]), mx.array([3.0, 4.0])]})
    assert np.allclose(_np(res[1]), [2.0, 3.0])


# ===========================================================================
# 9. Lifecycle — enable/disable restore, multiple probes, idempotence
# ===========================================================================

# Closed-form ``nullify``/``push_to_mean`` are GATED to fire only when the
# probe's last-position logit is > 0 (documented suppression semantics).  A
# freshly-attached probe head is initialised near zero, so we pin a real head
# (strong weight + large positive bias) to guarantee a positive last-position
# logit; otherwise these end-to-end tests would assert against a *correct* gated
# no-op and report a phantom bug.
_STEER_W = mx.array([[3.0, 0.0, 0.0, 0.0]])
_STEER_B = mx.array([5.0])
_STEER_MEANS = {"mean_0": mx.array([0.0, 0.0, 0.0, 0.0]), "mean_1": mx.array([1.0, 0.0, 0.0, 0.0])}


def _attach_steerable(m: Model, name: str, layer: int = 0) -> Any:
    """Attach a binary probe whose head guarantees a positive last-position logit."""
    p = m.attach_probe(
        ProbeConfig(
            name=name, layers=[layer], aggregation="last", module_config={"out_features": 1}
        )
    )
    p.module.weight = mx.array(_STEER_W)
    p.module.bias = mx.array(_STEER_B)
    return p


def _enable_geo(m: Model, name: str, method: str = "nullify", scale: float = 5.0) -> None:
    m.enable_steering(
        name,
        config=SteeringConfig(method=method, scale=scale),
        class_means={k: mx.array(v) for k, v in _STEER_MEANS.items()},
    )


def test_lifecycle_disable_fully_restores_logits() -> None:
    """Enable -> disable must restore the LM logits bit-for-bit to the pre-enable state."""
    m = _mlx_model()
    _attach_steerable(m, "p0")
    x = mx.array([[1, 2, 3]])
    base = _np(m.forward(x).lm_logits)
    _enable_geo(m, "p0")
    steered = _np(m.forward(x).lm_logits)
    m.disable_steering("p0")
    restored = _np(m.forward(x).lm_logits)
    assert not np.allclose(base, steered, atol=1e-5), "steering was a silent no-op"
    assert np.allclose(base, restored, atol=1e-6), "disable did not restore the forward"


def test_lifecycle_disable_when_not_enabled_is_harmless() -> None:
    m = _mlx_model()
    _attach_steerable(m, "p0")
    m.disable_steering("p0")  # must not raise
    m.disable_steering("p0")  # twice, still fine


def test_lifecycle_enable_twice_is_idempotent() -> None:
    """Enabling twice reuses the hook and yields the same steered output."""
    m = _mlx_model()
    _attach_steerable(m, "p0")
    x = mx.array([[1, 2, 3]])
    _enable_geo(m, "p0")
    once = _np(m.forward(x).lm_logits)
    _enable_geo(m, "p0")
    twice = _np(m.forward(x).lm_logits)
    assert np.allclose(once, twice, atol=1e-6)
    assert len(m.steering_hooks) == 1, "enabling twice should not duplicate the hook"


def test_lifecycle_multiple_probes_steered_independently() -> None:
    """Two probes steered at once; disabling each peels off its effect cleanly.

    Uses ``push_to_mean`` (a strictly additive shift) so each probe has a
    guaranteed non-trivial, independent contribution at its own layer.
    """
    m = _mlx_model()
    _attach_steerable(m, "p0", layer=0)
    _attach_steerable(m, "p1", layer=1)
    x = mx.array([[1, 2, 3]])
    base = _np(m.forward(x).lm_logits)

    _enable_geo(m, "p0", method="push_to_mean", scale=5.0)
    only_p0 = _np(m.forward(x).lm_logits)
    assert not np.allclose(base, only_p0, atol=1e-5), "p0 steering was a no-op"

    _enable_geo(m, "p1", method="push_to_mean", scale=5.0)
    both = _np(m.forward(x).lm_logits)
    assert not np.allclose(only_p0, both, atol=1e-5), "p1 added no independent effect"

    # Disabling p1 must peel off exactly its contribution -> back to only_p0.
    m.disable_steering("p1")
    assert np.allclose(_np(m.forward(x).lm_logits), only_p0, atol=1e-6), (
        "disabling p1 did not restore the p0-only state"
    )

    # Disabling p0 too -> back to the unsteered baseline.
    m.disable_steering("p0")
    assert np.allclose(base, _np(m.forward(x).lm_logits), atol=1e-6)


def test_enable_steering_unknown_probe_raises_keyerror() -> None:
    m = _mlx_model()
    with pytest.raises(KeyError):
        _enable_geo(m, "does_not_exist")


# ===========================================================================
# 10. enable_steering must not mutate the caller's config / class_means
# ===========================================================================


def test_enable_steering_does_not_mutate_caller_config() -> None:
    """The SteeringConfig the caller passes must be unchanged after enabling."""
    m = _mlx_model()
    m.attach_probe(
        ProbeConfig(name="p0", layers=[0], aggregation="last", module_config={"out_features": 1})
    )
    cfg = SteeringConfig(method="nullify", scale=2.0)
    _means = {"mean_0": mx.array([0.0, 0.0, 0.0, 0.0]), "mean_1": mx.array([1.0, 0.0, 0.0, 0.0])}
    m.enable_steering("p0", config=cfg, class_means=_means)
    assert cfg.method == "nullify"
    assert cfg.scale == 2.0
    assert cfg.direction is None, "enable_steering mutated the caller's config.direction"


# ===========================================================================
# 11. Generation interaction — steering changes output, not a silent no-op
# ===========================================================================


def test_generation_changes_under_strong_steering() -> None:
    """Greedy generation must differ between unsteered and strongly-steered runs.

    The tiny model has no tokenizer, so we run a manual greedy loop over
    ``Model.forward`` — which is exactly the steered forward pass generation
    drives — and compare the produced token sequences.
    """
    m = _mlx_model(h=4, v=8, layers=2)
    _attach_steerable(m, "p0")

    def greedy(prompt: list[int], n: int) -> list[int]:
        seq = list(prompt)
        for _ in range(n):
            logits = m.forward(mx.array([seq])).lm_logits
            nxt = int(mx.argmax(logits[0, -1]).item())
            seq.append(nxt)
        return seq[len(prompt) :]

    unsteered = greedy([1, 2, 3], 8)
    _enable_geo(m, "p0", method="push_to_mean", scale=50.0)
    steered = greedy([1, 2, 3], 8)
    assert unsteered != steered, "strong steering left the generated tokens unchanged (no-op)"


# ===========================================================================
# 12. SteeringConfig validation
# ===========================================================================


def test_steeringconfig_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown steering method"):
        SteeringConfig(method="bogus")  # type: ignore[arg-type]


def test_steeringconfig_accepts_known_methods() -> None:
    for method in ("nullify", "push_to_mean", "boundary", "custom"):
        SteeringConfig(method=method)  # must not raise

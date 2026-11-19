"""Regression tests for probe / steering / checkpoint.

Each test asserts a *correct result* against an independent ground truth, not
merely "runs without crashing". The bugs covered:

1. Multi-layer aggregation ignored ``ProbeConfig.layers`` order (concat columns
   / mean weighting used model-execution order instead).
2. A custom pooling callable never received the padding mask.
3. ``SteeringConfig.direction`` / ``layer`` overrides were dropped on
   ``to_dict``/``from_dict`` round-trip.
4. The ``boundary`` steering method moved the projection AWAY from the midpoint.
5. Reloading a checkpoint of a callable ``module_type``/``aggregation`` (or
   ``granularity="custom"``) degraded silently instead of failing loudly.
6. A probe-only MLX checkpoint always wrote ``adapters.safetensors``, so reload
   injected a phantom LoRA the user never asked for.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm.checkpoint import load_checkpoint, save_checkpoint
from auto_chasm.config import ProbeConfig, SteeringConfig
from auto_chasm.model import Model
from auto_chasm.steering import SteeringHook, _steer_mlx, _steer_torch


class _TinyMlp(nn.Module):
    """A tiny 4-layer MLP whose per-layer outputs are distinguishable."""

    def __init__(self, hidden_dim: int = 8, vocab_size: int = 16, num_layers: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array, **kwargs: Any) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _DummyTokenizer:
    """Minimal tokenizer stub."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 16 for c in text[:5]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids if i > 0)


class _Config:
    """Model config stub."""

    hidden_size = 8
    num_hidden_layers = 4
    vocab_size = 16


def _make_mlx_model() -> Model:
    mx.random.seed(7)
    base = _TinyMlp()
    base.config = _Config()
    return Model(base, _DummyTokenizer(), "mlx")


# ---------------------------------------------------------------------------
# Bug 1 — aggregation respects ProbeConfig.layers order
# ---------------------------------------------------------------------------


def _captured_for_single_layer(layer: int, tokens: mx.array) -> mx.array:
    """Capture the hidden state at exactly one layer, as an oracle reference."""
    model = _make_mlx_model()
    probe = model.attach_probe(ProbeConfig(name="ref", layers=[layer]))
    probe.clear_captured()
    model.forward(tokens)
    return np.array(probe.get_captured_states()[0])


def test_aggregation_respects_layers_order_concat_columns_mlx() -> None:
    """Concat columns must follow config.layers order, not execution order."""
    tokens = mx.array([[1, 2, 3, 4, 5]])

    # Oracle: hidden states of the *same* fresh model at layers 2 and 0.
    ref_layer2 = _captured_for_single_layer(2, tokens)
    ref_layer0 = _captured_for_single_layer(0, tokens)

    # Probe with layers OUT of execution order: [2, 0].
    model = _make_mlx_model()
    probe = model.attach_probe(ProbeConfig(name="agg", layers=[2, 0], aggregation="concat"))
    probe.clear_captured()
    model.forward(tokens)
    states = [np.array(h) for h in probe.get_captured_states()]

    assert len(states) == 2
    # states[0] must be layer 2 (first in config.layers), states[1] layer 0.
    assert np.allclose(states[0], ref_layer2, atol=1e-5)
    assert np.allclose(states[1], ref_layer0, atol=1e-5)

    # The concat output column blocks must mirror that order.
    out = np.array(probe._aggregate([mx.array(s) for s in states]))
    hidden = ref_layer2.shape[-1]
    assert np.allclose(out[..., :hidden], ref_layer2, atol=1e-5)
    assert np.allclose(out[..., hidden:], ref_layer0, atol=1e-5)


def test_aggregation_last_picks_config_last_not_exec_last_mlx() -> None:
    """aggregation='last' must return the layer LISTED last, not executed last."""
    tokens = mx.array([[1, 2, 3, 4, 5]])
    ref_layer0 = _captured_for_single_layer(0, tokens)

    model = _make_mlx_model()
    probe = model.attach_probe(ProbeConfig(name="lp", layers=[2, 0], aggregation="last"))
    probe.clear_captured()
    model.forward(tokens)
    states = probe.get_captured_states()
    out = np.array(probe._aggregate(states))
    # config.layers[-1] == 0, so "last" must equal layer 0's hidden state.
    assert np.allclose(out, ref_layer0, atol=1e-5)


def test_aggregation_respects_layers_order_torch() -> None:
    """The torch backend must reorder captures to config.layers order too."""
    import torch

    torch.manual_seed(7)

    import torch.nn as tnn

    class _TorchTiny(tnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = tnn.Embedding(16, 8)
            self.layers = tnn.ModuleList([tnn.Linear(8, 8) for _ in range(4)])
            self.output_proj = tnn.Linear(8, 16)

        def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            h = self.embedding(x)
            for layer in self.layers:
                h = torch.nn.functional.gelu(layer(h))
            return self.output_proj(h)

    def _ref(layer: int) -> np.ndarray:
        torch.manual_seed(7)
        base = _TorchTiny()
        base.config = _Config()
        m = Model(base, _DummyTokenizer(), "torch")
        p = m.attach_probe(ProbeConfig(name="ref", layers=[layer]))
        p.clear_captured()
        m.forward(torch.tensor([[1, 2, 3, 4, 5]]))
        return p.get_captured_states()[0].detach().numpy()

    ref_layer3 = _ref(3)
    ref_layer1 = _ref(1)

    torch.manual_seed(7)
    base = _TorchTiny()
    base.config = _Config()
    model = Model(base, _DummyTokenizer(), "torch")
    probe = model.attach_probe(ProbeConfig(name="agg", layers=[3, 1], aggregation="concat"))
    probe.clear_captured()
    model.forward(torch.tensor([[1, 2, 3, 4, 5]]))
    states = [h.detach().numpy() for h in probe.get_captured_states()]
    assert np.allclose(states[0], ref_layer3, atol=1e-5)
    assert np.allclose(states[1], ref_layer1, atol=1e-5)


# ---------------------------------------------------------------------------
# Bug 2 — custom pooling receives the padding mask
# ---------------------------------------------------------------------------


def test_custom_pooling_receives_mask_kwarg() -> None:
    """A pooling callable with a mask= kwarg must be given the real mask."""
    received: dict[str, Any] = {}

    def pool_with_mask(logits: Any, mask: Any = None) -> Any:
        received["mask"] = mask
        return logits.mean(axis=1)

    model = _make_mlx_model()
    probe = model.attach_probe(
        ProbeConfig(name="cp", layers=[1], granularity="custom", pooling=pool_with_mask)
    )
    logits = mx.ones((1, 4, 1))
    mask = mx.array([[1.0, 1.0, 0.0, 0.0]])
    probe._apply_pooling(logits, mask)
    assert received["mask"] is not None
    assert np.allclose(np.array(received["mask"]), np.array(mask))


def test_custom_pooling_mask_changes_result() -> None:
    """A mask-aware pooler must ignore padding instead of averaging over it."""

    def masked_mean(logits: Any, mask: Any = None) -> Any:
        if mask is None:
            return logits.mean(axis=1)
        m = mx.expand_dims(mask.astype(logits.dtype), -1)
        return mx.sum(logits * m, axis=1) / mx.maximum(mx.sum(m, axis=1), 1e-9)

    model = _make_mlx_model()
    probe = model.attach_probe(
        ProbeConfig(name="mm", layers=[1], granularity="custom", pooling=masked_mean)
    )
    # logits: two real positions valued 4.0, two padding positions valued 0.0.
    logits = mx.array([[[4.0], [4.0], [0.0], [0.0]]])
    mask = mx.array([[1.0, 1.0, 0.0, 0.0]])
    pooled = float(np.array(probe._apply_pooling(logits, mask)).reshape(-1)[0])
    # Oracle: mean over the two REAL positions == 4.0 (not 2.0 over all four).
    assert abs(pooled - 4.0) < 1e-5


def test_custom_pooling_single_arg_still_works() -> None:
    """A legacy single-arg pooler must still be called with logits only."""

    def pool_no_mask(logits: Any) -> Any:
        return logits.sum(axis=1)

    model = _make_mlx_model()
    probe = model.attach_probe(
        ProbeConfig(name="legacy", layers=[1], granularity="custom", pooling=pool_no_mask)
    )
    logits = mx.array([[[1.0], [2.0], [3.0]]])
    out = float(np.array(probe._apply_pooling(logits, mx.array([[1.0, 1.0, 1.0]]))).reshape(-1)[0])
    assert abs(out - 6.0) < 1e-5


# ---------------------------------------------------------------------------
# Bug 3 — direction / layer survive a to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_steering_config_direction_and_layer_round_trip_mlx() -> None:
    """config.direction and config.layer must survive to_dict/from_dict."""
    direction = mx.array([0.5, -1.0, 2.0])
    cfg = SteeringConfig(method="push_to_mean", scale=1.5, layer=3, direction=direction)
    hook = SteeringHook("p", cfg)
    hook._mean_0 = mx.array([0.0, 0.0, 0.0])
    hook._mean_1 = mx.array([1.0, 1.0, 1.0])

    restored = SteeringHook.from_dict(hook.to_dict(), backend="mlx")
    assert restored.config.layer == 3
    assert restored.config.direction is not None
    assert np.allclose(np.array(restored.config.direction), np.array(direction))
    assert restored.config.method == "push_to_mean"
    assert restored.config.scale == 1.5


def test_steering_config_direction_round_trip_torch_backend() -> None:
    """The override direction must restore as a torch tensor when asked."""
    from auto_chasm.utils import tensor_backend

    direction = mx.array([1.0, 2.0, 3.0])
    cfg = SteeringConfig(method="boundary", scale=2.0, layer=5, direction=direction)
    hook = SteeringHook("p", cfg)
    restored = SteeringHook.from_dict(hook.to_dict(), backend="torch")
    assert restored.config.layer == 5
    assert restored.config.direction is not None
    assert tensor_backend(restored.config.direction) == "torch"
    assert np.allclose(restored.config.direction.numpy(), np.array(direction))


def test_steering_config_no_override_round_trips_clean() -> None:
    """When no override is set, the restored config must keep direction=None."""
    cfg = SteeringConfig(method="nullify", scale=1.0)
    hook = SteeringHook("p", cfg)
    restored = SteeringHook.from_dict(hook.to_dict(), backend="mlx")
    assert restored.config.direction is None
    assert restored.config.layer is None


# ---------------------------------------------------------------------------
# Bug 4 — boundary moves the last-token projection TOWARD the midpoint
# ---------------------------------------------------------------------------

# Geometry from the finding: W=+x, mean_0=0, mean_1=2x => midpoint proj = 1.0.
_BH = 4
_BW = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
_BB = np.array([0.0], dtype=np.float32)
_BHIDDEN = np.array([[[0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]]], dtype=np.float32)
_BMEAN0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_BMEAN1 = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
_BDIR = _BMEAN1 - _BMEAN0
_BLOGITS = (_BHIDDEN @ _BW.reshape(-1, 1)).reshape(1, -1) + _BB  # [[0.0, 0.2]]
_MIDPOINT = 1.0
_PROJ_BEFORE = 0.2  # last-token projection onto unit_dir (+x)


def test_boundary_moves_toward_midpoint_mlx() -> None:
    """The last-token projection must move toward (not away from) the midpoint."""
    from types import SimpleNamespace

    head = SimpleNamespace(weight=mx.array(_BW), bias=mx.array(_BB))
    out = _steer_mlx(
        mx.array(_BHIDDEN),
        head,
        mx.array(_BLOGITS),
        "boundary",
        mx.array(_BMEAN0),
        mx.array(_BMEAN1),
        mx.array(_BDIR),
        scale=0.5,
    )
    unit = _BDIR / np.linalg.norm(_BDIR)
    proj_after = float(np.array(out)[0, -1, :] @ unit)
    # Closed-form: 0.2 + 0.5*(1.0-0.2) = 0.6 — strictly closer to the midpoint.
    assert abs(proj_after - 0.6) < 1e-4
    assert abs(proj_after - _MIDPOINT) < abs(_PROJ_BEFORE - _MIDPOINT)


def test_boundary_scale_one_reaches_midpoint_mlx() -> None:
    """At scale=1.0 the projection should land on the midpoint."""
    from types import SimpleNamespace

    head = SimpleNamespace(weight=mx.array(_BW), bias=mx.array(_BB))
    out = _steer_mlx(
        mx.array(_BHIDDEN),
        head,
        mx.array(_BLOGITS),
        "boundary",
        mx.array(_BMEAN0),
        mx.array(_BMEAN1),
        mx.array(_BDIR),
        scale=1.0,
    )
    unit = _BDIR / np.linalg.norm(_BDIR)
    proj_after = float(np.array(out)[0, -1, :] @ unit)
    assert abs(proj_after - _MIDPOINT) < 1e-4


def test_boundary_moves_toward_midpoint_torch() -> None:
    """Same boundary oracle on the torch backend (parity)."""
    from types import SimpleNamespace

    import torch

    head = SimpleNamespace(weight=torch.tensor(_BW), bias=torch.tensor(_BB))
    out = _steer_torch(
        torch.tensor(_BHIDDEN),
        head,
        torch.tensor(_BLOGITS),
        "boundary",
        torch.tensor(_BMEAN0),
        torch.tensor(_BMEAN1),
        torch.tensor(_BDIR),
        scale=0.5,
    ).numpy()
    unit = _BDIR / np.linalg.norm(_BDIR)
    proj_after = float(out[0, -1, :] @ unit)
    assert abs(proj_after - 0.6) < 1e-4
    assert abs(proj_after - _MIDPOINT) < abs(_PROJ_BEFORE - _MIDPOINT)


# ---------------------------------------------------------------------------
# Bugs 5 & 6 — checkpoint reload behaviour
# ---------------------------------------------------------------------------


def _patch_from_pretrained(monkeypatch: pytest.MonkeyPatch, factory: Any) -> None:
    """Make Model.from_pretrained return a fresh in-memory tiny model."""

    def fake(model_name: str, backend_name: str | None = None, **kwargs: Any) -> Model:
        m = factory()
        m._base_model_name = model_name
        return m

    monkeypatch.setattr(Model, "from_pretrained", staticmethod(fake))


def test_callable_module_type_checkpoint_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reloading a callable-module_type probe must fail loudly, not degrade."""

    def make_head(in_features: int, cfg: dict[str, Any]) -> nn.Module:
        return nn.Linear(in_features, 1)

    model = _make_mlx_model()
    model._base_model_name = "tiny"
    model.attach_probe(ProbeConfig(name="cust", layers=[1], module_type=make_head))

    _patch_from_pretrained(monkeypatch, _make_mlx_model)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt"
        save_checkpoint(model, str(ckpt))
        # Manifest must record the un-reconstructable sentinel.
        manifest = json.loads((ckpt / "manifest.json").read_text())
        assert manifest["probes"]["cust"]["module_type"] == "__callable__"
        with pytest.raises(ValueError, match="callable module_type"):
            load_checkpoint(str(ckpt), backend_name="mlx")


def test_callable_aggregation_checkpoint_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reloading a callable-aggregation probe must fail loudly."""

    def agg(states: list[Any]) -> Any:
        return states[0]

    model = _make_mlx_model()
    model._base_model_name = "tiny"
    model.attach_probe(ProbeConfig(name="agg", layers=[0, 1], aggregation=agg))

    _patch_from_pretrained(monkeypatch, _make_mlx_model)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt"
        save_checkpoint(model, str(ckpt))
        with pytest.raises(ValueError, match="callable aggregation"):
            load_checkpoint(str(ckpt), backend_name="mlx")


def test_custom_granularity_checkpoint_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A granularity='custom' probe cannot reload its pooling — must fail loudly."""

    def pool(logits: Any) -> Any:
        return logits.mean(axis=1)

    model = _make_mlx_model()
    model._base_model_name = "tiny"
    model.attach_probe(ProbeConfig(name="cg", layers=[1], granularity="custom", pooling=pool))

    _patch_from_pretrained(monkeypatch, _make_mlx_model)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt"
        save_checkpoint(model, str(ckpt))
        with pytest.raises(ValueError, match="granularity='custom'"):
            load_checkpoint(str(ckpt), backend_name="mlx")


def test_plain_linear_probe_checkpoint_reloads_fine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain linear probe must still reload (regression guard for bug 5)."""
    model = _make_mlx_model()
    model._base_model_name = "tiny"
    probe = model.attach_probe(ProbeConfig(name="lin", layers=[1]))
    # Give the head a recognizable weight so we can confirm it restores.
    probe.module.weight = probe.module.weight * 0.0 + 0.123

    _patch_from_pretrained(monkeypatch, _make_mlx_model)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt"
        save_checkpoint(model, str(ckpt))
        reloaded = load_checkpoint(str(ckpt), backend_name="mlx")
    assert "lin" in reloaded.probes
    w = np.array(reloaded.probes["lin"].module.weight)
    assert np.allclose(w, 0.123, atol=1e-5)


def test_probe_only_checkpoint_has_no_phantom_lora(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe-only checkpoint must round-trip with NO LoRA injected."""
    model = _make_mlx_model()
    model._base_model_name = "tiny"
    assert model.lora_config is None
    model.attach_probe(ProbeConfig(name="p", layers=[1]))

    _patch_from_pretrained(monkeypatch, _make_mlx_model)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt"
        save_checkpoint(model, str(ckpt))
        # No adapters file should be written for a probe-only model.
        assert not (ckpt / "adapters.safetensors").exists()
        # And the manifest must not record a lora block.
        manifest = json.loads((ckpt / "manifest.json").read_text())
        assert "lora" not in manifest
        reloaded = load_checkpoint(str(ckpt), backend_name="mlx")
    # The reloaded model must NOT have a phantom LoRA config.
    assert reloaded.lora_config is None

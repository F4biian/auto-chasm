"""Tests: checkpoints, special tokens, model wrapper, loaders.

Covers ``checkpoint.py``, ``special_tokens.py``, ``model.py``,
``backends/loaders.py``, ``backends/base.py``, ``utils.py``.

Tests named ``test_BUG_*`` are regression tests for specific past defects; the
rest are general regression coverage.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm.config import ProbeConfig, SteeringConfig
from auto_chasm.model import Model

# ---------------------------------------------------------------------------
# Shared tiny MLX model / tokenizer
# ---------------------------------------------------------------------------


class _TinyMlp(nn.Module):
    """A tiny deterministic MLP used as a stand-in language model."""

    def __init__(self, h: int = 16, v: int = 32, layers: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(v, h)
        self.layers = [nn.Linear(h, h) for _ in range(layers)]
        self.output_proj = nn.Linear(h, v)

    def __call__(self, x: mx.array, **_: object) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    """Minimal HF-style config."""

    hidden_size = 16
    num_hidden_layers = 4
    vocab_size = 32


class _Tok:
    """Minimal tokenizer that supports add_tokens + len()."""

    eos_token_id = 0

    def __init__(self) -> None:
        self.vocab = 32
        self.extra: list[str] = []

    def add_tokens(self, toks: list[str], special_tokens: bool = False) -> int:
        new = [t for t in toks if t not in self.extra]
        self.extra += new
        self.vocab += len(new)
        return len(new)

    def __len__(self) -> int:
        return self.vocab

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "x"


def _build_model(seed: int = 0) -> Model:
    """Build a fresh, deterministically-initialised Model wrapper."""
    mx.random.seed(seed)
    base = _TinyMlp()
    base.config = _Cfg()
    m = Model(base, _Tok(), "mlx")
    m._base_model_name = None  # type: ignore[attr-defined]
    return m


def _flat(params: object) -> dict:
    from mlx.utils import tree_flatten

    return dict(tree_flatten(params))


# ===========================================================================
# add_special_tokens — losslessness (regression coverage; these PASS)
# ===========================================================================


def test_special_tokens_existing_rows_byte_identical_and_logits_unchanged():
    """Plain MLX: existing embedding rows byte-identical, existing logits unchanged."""
    m = _build_model()
    ids = mx.array([[1, 2, 3, 4, 5]])
    before_logits = m.model(ids)
    emb_before = mx.array(m.model.embedding.weight[:32])

    added = m.add_special_tokens(["<a>", "<b>"])
    assert added == 2
    assert m.model.embedding.weight.shape[0] == 34
    assert m.model.output_proj.weight.shape[0] == 34
    assert bool(mx.array_equal(m.model.embedding.weight[:32], emb_before))

    after_logits = m.model(ids)
    assert bool(mx.array_equal(before_logits, after_logits[:, :, :32]))
    # New ids look up and produce logits over the grown vocab.
    grown = m.model(mx.array([[32, 33, 1]]))
    assert grown.shape == (1, 3, 34)


def test_special_tokens_noop_and_duplicates():
    """Add 0 tokens, then add a duplicate — both are no-ops (return 0)."""
    m = _build_model()
    assert m.add_special_tokens([]) == 0
    m.add_special_tokens(["<x>"])
    assert m.add_special_tokens(["<x>"]) == 0
    # Mixed: one new, one duplicate -> only 1 added.
    assert m.add_special_tokens(["<x>", "<new>"]) == 1


def test_special_tokens_quantized_rows_lossless():
    """Quantized MLX embedding: existing packed rows stay byte-identical."""

    class QMlp(nn.Module):
        """Quantized embedding model (dim divisible by group size)."""

        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.QuantizedEmbedding(32, 64, group_size=64, bits=4)

        def __call__(self, x: mx.array) -> mx.array:
            return self.embed_tokens(x)

    m = Model(QMlp(), _Tok(), "mlx")
    qe = m.model.embed_tokens
    ow, os_, ob = mx.array(qe.weight), mx.array(qe.scales), mx.array(qe.biases)
    assert m.add_special_tokens(["<x>", "<y>"]) == 2
    assert qe.num_embeddings == 34
    assert bool(mx.array_equal(qe.weight[:32], ow))
    assert bool(mx.array_equal(qe.scales[:32], os_))
    assert bool(mx.array_equal(qe.biases[:32], ob))


def test_special_tokens_untied_head_bias_grows_correctly():
    """Untied head with a bias grows its weight AND bias; existing bias preserved."""

    class M(nn.Module):
        """Tiny model used to exercise checkpoint save/load."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(32, 8)
            self.output_proj = nn.Linear(8, 32, bias=True)

        def __call__(self, x: mx.array) -> mx.array:
            return self.output_proj(self.embedding(x))

    m = Model(M(), _Tok(), "mlx")
    bias_before = mx.array(m.model.output_proj.bias)
    m.add_special_tokens(["<a>", "<b>"])
    assert m.model.output_proj.weight.shape[0] == 34
    assert m.model.output_proj.bias.shape[0] == 34
    assert bool(mx.array_equal(m.model.output_proj.bias[:32], bias_before))
    assert bool(mx.array_equal(m.model.output_proj.bias[32:], mx.zeros((2,))))


# ===========================================================================
# Checkpoint roundtrip — losslessness oracle (MLX) (regression; PASSES)
# ===========================================================================


def test_checkpoint_probe_roundtrip_is_lossless_mlx():
    """Save -> load a probe; reloaded probe must produce IDENTICAL outputs."""
    from auto_chasm._checkpoint_weights import (
        load_probe_weights as _load_probe_weights,
    )
    from auto_chasm._checkpoint_weights import (
        save_probe_weights as _save_probe_weights,
    )

    src = _build_model()
    p = src.attach_probe(
        ProbeConfig(name="p1", layers=[2], source="hidden", module_config={"out_features": 1})
    )
    ids = mx.array([[1, 2, 3, 4, 5]])
    out0 = src.forward(ids)
    ref = np.array(out0.probes["p1"].logits)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p1.safetensors"
        _save_probe_weights(p, path, src.backend)

        fresh = _build_model()
        p2 = fresh.attach_probe(
            ProbeConfig(name="p1", layers=[2], source="hidden", module_config={"out_features": 1})
        )
        _load_probe_weights(p2, path, fresh.backend)
        got = np.array(fresh.forward(ids).probes["p1"].logits)

    assert np.array_equal(ref, got), "probe roundtrip is lossy"


# ===========================================================================
# BUG: load_checkpoint restores steering geometry but never re-activates it.
# A model saved with steering ENABLED reloads producing UNSTEERED logits.
# ===========================================================================


def test_BUG_steering_active_after_checkpoint_load():
    """A checkpoint saved with steering enabled must steer after reload.

    ``load_checkpoint`` repopulates ``model._steering_hooks`` but never wires the
    hook into the probe's ``layer_captures`` nor calls ``hook.enable()``, so a
    restored model silently produces the *unsteered* baseline — a silent
    behavioural divergence between save-time and load-time.

    FAILS now (steering is inert after load).
    """
    from auto_chasm.steering import SteeringHook

    src = _build_model()
    p = src.attach_probe(
        ProbeConfig(name="p1", layers=[2], source="hidden", module_config={"out_features": 1})
    )
    src.enable_steering(
        "p1",
        SteeringConfig(method="nullify", scale=1.0),
        class_means={"mean_0": mx.zeros((16,)), "mean_1": mx.ones((16,))},
    )
    ids = mx.array([[1, 2, 3, 4, 5]])
    steered = np.array(src.forward(ids).lm_logits)

    with tempfile.TemporaryDirectory() as tmp:
        ck = os.path.join(tmp, "ck")
        src.save_checkpoint(ck)
        with open(os.path.join(ck, "steering", "p1.json")) as _f:
            steer_json = json.load(_f)

    # Reproduce exactly what load_checkpoint does for the steering branch:
    from auto_chasm.checkpoint import _reactivate_steering

    dst = _build_model()
    p2 = dst.attach_probe(
        ProbeConfig(name="p1", layers=[2], source="hidden", module_config={"out_features": 1})
    )
    p2.module.update(_flat(p.module.parameters()))  # identical base+probe
    hook = SteeringHook.from_dict(steer_json, backend="mlx")
    dst._steering_hooks["p1"] = hook
    # The fixed load_checkpoint re-wires the restored hook into the probe's
    # captures and enables it (instead of leaving it inert).
    _reactivate_steering(dst, "p1", hook)

    loaded = np.array(dst.forward(ids).lm_logits)

    # Sanity: steering DID change logits at save time.
    base = _build_model()
    pp = base.attach_probe(
        ProbeConfig(name="p1", layers=[2], source="hidden", module_config={"out_features": 1})
    )
    pp.module.update(_flat(p.module.parameters()))
    baseline = np.array(base.forward(ids).lm_logits)
    assert not np.allclose(steered, baseline), "test setup: steering had no effect at save time"

    # The restored model must reproduce the steered logits — not the unsteered baseline.
    assert np.allclose(loaded, steered), (
        "steering is inert after checkpoint load: restored model produced unsteered logits"
    )


# ===========================================================================
# BUG: silent shape mismatch on probe weight load.
#   MLX  -> module.update() silently RESHAPES the probe (no validation).
#   torch-> load_state_dict raises, but the bare except swallows it.
# Either way a config/weights mismatch must be a CLEAR error, not silent.
# ===========================================================================


def test_BUG_probe_weight_shape_mismatch_mlx_must_not_silently_reshape():
    """Loading weights with a different out_features must error, not silently reshape.

    The probe is configured ``out_features=1`` but the on-disk weights are
    ``out_features=4``.  ``probe.module.update()`` replaces arrays without any
    shape check, so the probe silently becomes a 4-class head — contradicting its
    own config and the checkpoint manifest.

    FAILS now (no error; probe is silently reshaped to (4, 16)).
    """
    from auto_chasm._checkpoint_weights import (
        load_probe_weights as _load_probe_weights,
    )
    from auto_chasm._checkpoint_weights import (
        save_probe_weights as _save_probe_weights,
    )

    src = _build_model()
    big = src.attach_probe(
        ProbeConfig(name="p1", layers=[2], source="hidden", module_config={"out_features": 4})
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p1.safetensors"
        _save_probe_weights(big, path, src.backend)

        dst = _build_model()
        small = dst.attach_probe(
            ProbeConfig(name="p1", layers=[2], source="hidden", module_config={"out_features": 1})
        )
        assert small.module.weight.shape == (1, 16)
        with pytest.raises(Exception):  # noqa: B017 - any clear error is acceptable
            _load_probe_weights(small, path, dst.backend)
        # And the probe must NOT have been silently mutated to the wrong shape.
        assert small.module.weight.shape == (1, 16), (
            "probe was silently reshaped to match incompatible on-disk weights"
        )


def test_BUG_probe_weight_load_failure_is_not_swallowed():
    """A missing/failed probe-weight load must surface, not return an untrained probe.

    ``_load_probe_weights`` wraps everything in ``except Exception`` and only logs a
    warning.  A checkpoint with a missing probe file therefore "loads" into a model
    whose probe still has random, untrained weights — a research-poisoning footgun
    (the model looks restored but the probe is garbage).

    FAILS now (no exception; probe silently keeps its untrained weights).
    """
    from auto_chasm._checkpoint_weights import load_probe_weights as _load_probe_weights

    dst = _build_model()
    probe = dst.attach_probe(
        ProbeConfig(name="p1", layers=[2], source="hidden", module_config={"out_features": 1})
    )
    untrained = np.array(probe.module.weight)
    missing = Path(tempfile.gettempdir()) / "auto_chasm_nope_does_not_exist.safetensors"
    if missing.exists():
        missing.unlink()

    with pytest.raises(Exception):  # noqa: B017 - clear error expected on a missing probe file
        _load_probe_weights(probe, missing, dst.backend)
    # The probe must not be quietly left in its untrained state with no signal.
    assert np.array_equal(np.array(probe.module.weight), untrained)


# ===========================================================================
# BUG: import_checkpoint returns a non-checkpoint path when output dir is non-empty.
# ===========================================================================


def test_BUG_import_checkpoint_returns_usable_path_into_nonempty_dir():
    """import_checkpoint must return a directory that actually holds the manifest.

    The result heuristic ``len(extracted) == 1 and extracted[0].is_dir()`` is fooled
    by any pre-existing entry in ``output_dir``: it falls back to returning the
    parent dir, which has no manifest, so the returned path is not loadable.

    FAILS now (returned path has no manifest.json).
    """
    from auto_chasm.checkpoint import export_checkpoint, import_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(tmp) / "mychk"
        ck.mkdir()
        (ck / "manifest.json").write_text(
            json.dumps({"auto_chasm_version": "0.1.0", "backend": "mlx"})
        )
        arc = Path(tmp) / "out.auto_chasm"
        export_checkpoint(str(ck), str(arc))

        out = Path(tmp) / "extract"
        out.mkdir()
        (out / "preexisting").mkdir()  # non-empty target dir
        result = import_checkpoint(str(arc), str(out))

        assert (Path(result) / "manifest.json").exists(), (
            f"import_checkpoint returned {result!r}, which is not a valid checkpoint dir"
        )


def test_import_checkpoint_clean_roundtrip_is_loadable():
    """Regression: clean export -> import yields a manifest-bearing directory."""
    from auto_chasm.checkpoint import export_checkpoint, import_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        ck = Path(tmp) / "mychk"
        ck.mkdir()
        (ck / "manifest.json").write_text(
            json.dumps({"auto_chasm_version": "0.1.0", "backend": "mlx"})
        )
        arc = Path(tmp) / "out.auto_chasm"
        export_checkpoint(str(ck), str(arc))
        result = import_checkpoint(str(arc), str(Path(tmp) / "fresh"))
        assert (Path(result) / "manifest.json").exists()


# ===========================================================================
# BUG: duplicate probe name silently overwrites AND leaves an orphan capture
#      wrapper injected in the model (runs on every forward; untracked).
# ===========================================================================


def test_BUG_duplicate_probe_name_does_not_silently_orphan_a_capture():
    """Re-using a probe name must error (or cleanly replace), not orphan a wrapper.

    ``attach_probe`` does ``self._probes[name] = probe`` unconditionally.  A second
    probe with the same name at a different layer replaces the dict entry, but the
    FIRST probe's capture wrapper stays injected on its layer — running on every
    forward, capturing into a now-unreachable probe.  No warning is raised.

    FAILS now (no error; layer 0 keeps an orphaned capture wrapper).
    """
    m = _build_model()
    layer0_original_type = type(m.model.layers[0]).__name__

    m.attach_probe(
        ProbeConfig(name="p", layers=[0], source="hidden", module_config={"out_features": 1})
    )
    wrapped_type = type(m.model.layers[0]).__name__
    assert wrapped_type != layer0_original_type  # confirm it got wrapped

    with pytest.raises(Exception):  # noqa: B017 - a clear duplicate-name error is expected
        m.attach_probe(
            ProbeConfig(name="p", layers=[2], source="hidden", module_config={"out_features": 1})
        )


def test_BUG_add_probes_rejects_duplicate_names_in_one_call():
    """add_probes with two same-named configs must not silently drop one.

    FAILS now (the second silently overwrites the first; only one probe survives,
    and the first probe's capture wrapper is orphaned on its layer).
    """
    m = _build_model()
    with pytest.raises(Exception):  # noqa: B017 - duplicate names should be rejected
        m.add_probes(
            [
                ProbeConfig(
                    name="dup", layers=[0], source="hidden", module_config={"out_features": 1}
                ),
                ProbeConfig(
                    name="dup", layers=[2], source="hidden", module_config={"out_features": 1}
                ),
            ]
        )


# ===========================================================================
# Model wrapper: freeze/unfreeze state must be queryable AND correct.
# freeze_model / freeze_probe set the native MLX frozen state correctly, but
# Backend.module.trainable_parameters() reports frozen params as trainable.
# ===========================================================================


def test_freeze_model_sets_native_frozen_state():
    """Regression: freeze_model() makes the native MLX trainable set empty."""
    from mlx.utils import tree_flatten

    m = _build_model()
    m.freeze_model()
    native_trainable = tree_flatten(m.model.trainable_parameters())
    assert len(native_trainable) == 0


def test_BUG_backend_trainable_parameters_respects_freeze():
    """Backend.module.trainable_parameters must reflect freeze state.

    ``MLXModuleOps.trainable_parameters`` returns ``module.parameters()`` (ALL of
    them), ignoring MLX's frozen flag.  After ``freeze_model()`` the native
    ``trainable_parameters()`` is empty, but the backend helper still reports every
    parameter as trainable — so anything querying trainability through the backend
    (the documented freeze surface) gets the wrong answer.

    FAILS now (backend reports frozen params as trainable).
    """
    m = _build_model()
    m.freeze_model()
    backend_trainable = m.backend.module.trainable_parameters(m.model)
    assert len(backend_trainable) == 0, (
        "backend.trainable_parameters ignores the frozen state set by freeze_model()"
    )


def test_freeze_unfreeze_probe_keyerror_on_missing():
    """Regression: freeze/unfreeze_probe raise a clear KeyError for unknown names."""
    m = _build_model()
    m.attach_probe(
        ProbeConfig(name="p", layers=[0], source="hidden", module_config={"out_features": 1})
    )
    with pytest.raises(KeyError):
        m.freeze_probe("nope")
    with pytest.raises(KeyError):
        m.unfreeze_probe("nope")


def test_freeze_probe_sets_native_frozen_state():
    """Regression: freeze_probe('p') empties that probe's native trainable set."""
    from mlx.utils import tree_flatten

    m = _build_model()
    p = m.attach_probe(
        ProbeConfig(name="p", layers=[0], source="hidden", module_config={"out_features": 1})
    )
    m.freeze_probe("p")
    assert len(tree_flatten(p.module.trainable_parameters())) == 0


# ===========================================================================
# Loader / config-shape detection (regression coverage; PASS)
# ===========================================================================


def test_hidden_dim_and_vocab_detection_across_config_shapes():
    """hidden_size/d_model/n_embd/dim and vocab_size/n_vocab are all detected."""
    from auto_chasm.probe import _get_hidden_dim, _get_vocab_size

    base = _TinyMlp()

    class C1:
        """Config stub using `d_model`/`n_vocab` attribute names."""

        d_model = 64
        n_vocab = 999

    base.config = C1()
    assert _get_hidden_dim(base) == 64
    assert _get_vocab_size(base) == 999

    class C2:
        """Config stub using the `n_embd` attribute name."""

        n_embd = 48

    base.config = C2()
    assert _get_hidden_dim(base) == 48

    class C3:
        """Config stub exposing hidden size under the `dim` attribute."""

        dim = 20

    base.config = C3()
    assert _get_hidden_dim(base) == 20

    # args-based fallback (mlx-lm style)
    base2 = _TinyMlp()

    class Args:
        """Config stub exposing dims under a non-standard `args` attribute."""

        hidden_size = 128
        vocab_size = 77

    base2.args = Args()
    if hasattr(base2, "config"):
        del base2.config
    assert _get_hidden_dim(base2) == 128
    assert _get_vocab_size(base2) == 77


def test_num_layers_detection():
    """Model.num_layers / get_num_layers find the block list."""
    m = _build_model()
    assert m.num_layers == 4


# ===========================================================================
# Cross-backend checkpoint parity (torch <-> mlx probe weights).
# ===========================================================================


def test_cross_backend_probe_weights_load_with_parity():
    """Probe weights saved on MLX load into a torch probe with numerical parity."""
    pytest.importorskip("torch")
    import torch.nn as tnn

    from auto_chasm._checkpoint_weights import (
        load_probe_weights as _load_probe_weights,
    )
    from auto_chasm._checkpoint_weights import (
        save_probe_weights as _save_probe_weights,
    )

    src = _build_model()
    pmx = src.attach_probe(
        ProbeConfig(name="p1", layers=[2], source="hidden", module_config={"out_features": 1})
    )
    mlx_w = np.array(pmx.module.weight)
    mlx_b = np.array(pmx.module.bias)

    class _TorchMlp(tnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = tnn.Embedding(32, 16)
            self.layers = tnn.ModuleList([tnn.Linear(16, 16) for _ in range(4)])
            self.output_proj = tnn.Linear(16, 32)

        def forward(self, x: object, **_: object) -> object:
            h = self.embedding(x)
            for layer in self.layers:
                h = tnn.functional.gelu(layer(h))
            return self.output_proj(h)

    tbase = _TorchMlp()
    tbase.config = _Cfg()
    mt = Model(tbase, _Tok(), "torch")
    pt = mt.attach_probe(
        ProbeConfig(name="p1", layers=[2], source="hidden", module_config={"out_features": 1})
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p1.safetensors"
        _save_probe_weights(pmx, path, src.backend)
        _load_probe_weights(pt, path, mt.backend)

    got_w = pt.module.weight.detach().cpu().numpy()
    got_b = pt.module.bias.detach().cpu().numpy()
    assert np.allclose(got_w, mlx_w, atol=1e-6)
    assert np.allclose(got_b, mlx_b, atol=1e-6)


# ===========================================================================
# Low: torch probe save downcasts fp64 -> fp32 (lossy for double probes).
# ===========================================================================


def test_BUG_torch_probe_save_preserves_fp64_precision():
    """Saving a float64 probe must not silently lose precision via a .float() cast.

    ``_save_probe_weights`` does ``v.float()`` (fp32) on the torch path; a probe held
    in fp64 round-trips with ~1e-8 error.  Low severity, but it is a silent, lossy
    cast on a "lossless" save path.

    FAILS now (fp64 precision lost).
    """
    torch = pytest.importorskip("torch")
    from auto_chasm._checkpoint_weights import (
        load_probe_weights as _load_probe_weights,
    )
    from auto_chasm._checkpoint_weights import (
        save_probe_weights as _save_probe_weights,
    )

    class _P:
        def __init__(self, module: object, name: str) -> None:
            self.module = module
            self.name = name

    from auto_chasm.backends import Backend

    backend = Backend(force="torch")
    mod = torch.nn.Linear(2, 1).double()
    with torch.no_grad():
        mod.weight.copy_(
            torch.tensor([[0.12345678901234567, 0.98765432109876543]], dtype=torch.float64)
        )
    orig = mod.weight.clone()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p.safetensors"
        _save_probe_weights(_P(mod, "p"), path, backend)
        mod2 = torch.nn.Linear(2, 1).double()
        _load_probe_weights(_P(mod2, "p"), path, backend)

    assert torch.equal(orig, mod2.weight), "fp64 probe weights lost precision on save"

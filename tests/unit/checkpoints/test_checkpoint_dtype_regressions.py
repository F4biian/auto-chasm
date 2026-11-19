"""Checkpoint regressions: bf16 probes, unpersisted-base warnings, orphans."""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from auto_chasm import Model, ProbeConfig
from auto_chasm._checkpoint_weights import (
    load_probe_weights,
    read_probe_weights_numpy,
    save_probe_weights,
)
from auto_chasm.checkpoint import save_checkpoint


class _TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = [nn.Linear(8, 8) for _ in range(2)]
        self.output_proj = nn.Linear(8, 16)

    def __call__(self, x: mx.array, **k: object) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = 8
    num_hidden_layers = 2


def _model_with_probe(name: str = "p") -> Model:
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(ProbeConfig(name=name, layers=[1], source="hidden"))
    return m


# --- CK-BUG1: a bf16 probe checkpoint must be loadable (numpy has no bf16) --------


def test_ckbug1_bf16_probe_weights_readable(tmp_path) -> None:  # noqa: ANN001
    """read_probe_weights_numpy handles a bf16 safetensors file (was unloadable)."""
    m = _model_with_probe()
    probe = m._probes["p"]
    probe.module.set_dtype(mx.bfloat16)
    path = tmp_path / "p.safetensors"
    save_probe_weights(probe, path, m.backend)
    weights = read_probe_weights_numpy(path)  # before the fix this raised
    assert weights  # non-empty
    assert all(np.asarray(v).dtype == np.float32 for v in weights.values())  # bf16 upcast


def test_ckbug1_bf16_probe_roundtrip_preserves_values_and_dtype(tmp_path) -> None:  # noqa: ANN001
    """A bf16 probe round-trips: values match (bf16 precision) and dtype stays bf16."""
    src = _model_with_probe()
    src_probe = src._probes["p"]
    src_probe.module.set_dtype(mx.bfloat16)
    path = tmp_path / "p.safetensors"
    save_probe_weights(src_probe, path, src.backend)

    dst = _model_with_probe()
    dst_probe = dst._probes["p"]
    dst_probe.module.set_dtype(mx.bfloat16)
    load_probe_weights(dst_probe, path, dst.backend)

    from mlx.utils import tree_flatten

    src_flat = dict(tree_flatten(src_probe.module.parameters()))
    dst_flat = dict(tree_flatten(dst_probe.module.parameters()))
    assert set(src_flat) == set(dst_flat)
    for k, v in src_flat.items():
        assert dst_flat[k].dtype == mx.bfloat16  # dtype preserved on load
        # numpy has no bf16, so compare as float32 (a lossless widening of both).
        got = np.array(dst_flat[k].astype(mx.float32))
        want = np.array(v.astype(mx.float32))
        np.testing.assert_array_equal(got, want)  # exact bf16 match


# --- CK-BUG2: warn on trainable base weights that the checkpoint drops ------------


def test_ckbug2_full_finetune_warns(tmp_path, caplog) -> None:  # noqa: ANN001
    """An unfrozen base with no LoRA warns that base weights are not persisted."""
    m = _model_with_probe()  # base left unfrozen == a full fine-tune
    with caplog.at_level(logging.WARNING, logger="auto_chasm.checkpoint"):
        save_checkpoint(m, str(tmp_path / "ckpt"))
    assert any("does not persist base-model weights" in r.message for r in caplog.records)


def test_ckbug2_frozen_base_does_not_warn(tmp_path, caplog) -> None:  # noqa: ANN001
    """A frozen base (only the probe trains) does not trigger the base-weights warning."""
    m = _model_with_probe()
    m.backend.module.freeze(m.model)
    with caplog.at_level(logging.WARNING, logger="auto_chasm.checkpoint"):
        save_checkpoint(m, str(tmp_path / "ckpt"))
    assert not any("does not persist base-model weights" in r.message for r in caplog.records)


def test_ckbug2_lora_branch_still_warns_on_unfrozen_base(caplog) -> None:  # noqa: ANN001
    """The LoRA branch subtracts saved adapters but still flags an unfrozen base."""
    from auto_chasm.checkpoint import _warn_unpersisted_base

    m = _model_with_probe()  # 7 trainable base tensors, 2 probe tensors
    # Simulate LoRA having persisted 1 adapter tensor; the base is still unfrozen.
    with caplog.at_level(logging.WARNING, logger="auto_chasm.checkpoint"):
        _warn_unpersisted_base(m, n_saved_adapter_tensors=1)
    assert any("does not persist base-model weights" in r.message for r in caplog.records)


# --- ST-F2: warn that added special tokens are not persisted ----------------------


def test_stf2_added_special_tokens_warns(tmp_path, caplog) -> None:  # noqa: ANN001
    """Saving a model whose vocab was grown warns the tokens are not persisted."""
    m = _model_with_probe()
    m.backend.module.freeze(m.model)  # isolate the special-token warning
    m._n_added_special_tokens = 3
    with caplog.at_level(logging.WARNING, logger="auto_chasm.checkpoint"):
        save_checkpoint(m, str(tmp_path / "ckpt"))
    assert any("added special tokens" in r.message for r in caplog.records)


# --- CK-ISSUE3: re-saving prunes orphaned probe/adapter files ---------------------


def test_ckissue3_removed_probe_file_is_pruned(tmp_path) -> None:  # noqa: ANN001
    """Re-saving after dropping a probe deletes its stale weights file."""
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(ProbeConfig(name="a", layers=[1], source="hidden"))
    m.attach_probe(ProbeConfig(name="b", layers=[1], source="hidden"))
    ckpt = tmp_path / "ckpt"
    save_checkpoint(m, str(ckpt))
    assert (ckpt / "probes" / "b.safetensors").exists()

    del m._probes["b"]
    save_checkpoint(m, str(ckpt))
    assert (ckpt / "probes" / "a.safetensors").exists()
    assert not (ckpt / "probes" / "b.safetensors").exists()  # orphan pruned


def test_ckissue3_orphan_adapters_removed_when_no_lora(tmp_path) -> None:  # noqa: ANN001
    """A stale adapters file is removed when re-saving a model without LoRA."""
    m = _model_with_probe()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    stale = ckpt / "adapters.safetensors"
    stale.write_bytes(b"stale")  # left over from a prior LoRA save
    save_checkpoint(m, str(ckpt))
    assert not stale.exists()  # would otherwise resurrect a phantom LoRA on reload

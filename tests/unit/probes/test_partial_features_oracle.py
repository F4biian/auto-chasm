"""Oracle tests closing the last two ``partial`` ledger rows.

* ``granularity="custom"`` — a user pooling callable is applied, receives the
  padding mask, and its result is what the probe emits (value-checked).
* Single-file checkpoint export/import — a directory survives a tar round-trip
  byte-for-byte, and exporting a directory without a manifest raises.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from auto_chasm._probe_agg import call_custom_pooling
from auto_chasm.checkpoint import MANIFEST_NAME, export_checkpoint, import_checkpoint


class TestCustomPooling:
    """granularity='custom' delegates to the user callable (mask-aware)."""

    def test_mask_aware_pooling_value(self) -> None:
        # Sum over valid (non-padding) time positions only.
        def pool(logits: mx.array, mask: mx.array) -> mx.array:
            m = mask[..., None].astype(logits.dtype)
            return (logits * m).sum(axis=1)

        logits = mx.array([[[1.0], [2.0], [3.0]]])  # [1, 3, 1]
        mask = mx.array([[True, False, True]])  # drop position 1
        out = call_custom_pooling(pool, logits, mask)
        assert out.tolist() == [[4.0]]  # 1 + 3, position 1 excluded

    def test_single_arg_pooling_still_works(self) -> None:
        # A pooler that takes only logits is called without the mask.
        def pool(logits: mx.array) -> mx.array:
            return logits.mean(axis=1)

        logits = mx.array([[[2.0], [4.0]]])
        out = call_custom_pooling(pool, logits, mx.array([[True, True]]))
        assert out.tolist() == [[3.0]]

    def test_probe_routes_to_custom_pooling(self) -> None:
        import mlx.nn as nn

        from auto_chasm import Model, ProbeConfig
        from auto_chasm.trainers.trainable import _TrainableModel

        class _TinyMlp(nn.Module):
            """Embedding -> linear -> output projection."""

            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(32, 16)
                self.layers = [nn.Linear(16, 16)]
                self.output_proj = nn.Linear(16, 32)

            def __call__(self, x: mx.array) -> mx.array:
                return self.output_proj(nn.gelu(self.layers[0](self.embedding(x))))

        class _Cfg:
            """Minimal model config."""

            hidden_size = 16
            num_hidden_layers = 1
            vocab_size = 32

        called = {"hit": False}

        def pool(logits: mx.array) -> mx.array:
            called["hit"] = True
            return logits.sum(axis=1)

        m = Model(_TinyMlp(), None, "mlx")
        m.model.config = _Cfg()
        m.attach_probe(ProbeConfig(name="p", layers=[0], granularity="custom", pooling=pool))
        _TrainableModel(m.model, m._probes)(mx.array([[1, 2, 3]]))
        out = m._probes["p"].forward()
        assert called["hit"]  # the custom pooler ran
        assert out.shape == (1, 1)  # pooled over time (3 -> 1)


class TestCheckpointSingleFileRoundTrip:
    """export_checkpoint -> import_checkpoint preserves the directory exactly."""

    def test_roundtrip_is_byte_identical(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / MANIFEST_NAME).write_text('{"version": 1, "base_model": "x"}')
        (ckpt / "weights.safetensors").write_bytes(b"\x00\x01\x02weights")

        archive = tmp_path / "bundle.auto_chasm"
        export_checkpoint(str(ckpt), str(archive))
        assert archive.exists()

        restored = Path(import_checkpoint(str(archive), str(tmp_path / "restored")))
        assert (restored / MANIFEST_NAME).read_text() == '{"version": 1, "base_model": "x"}'
        assert (restored / "weights.safetensors").read_bytes() == b"\x00\x01\x02weights"

    def test_export_without_manifest_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "no_manifest"
        d.mkdir()
        (d / "weights.bin").write_bytes(b"data")
        with pytest.raises(ValueError, match="manifest"):
            export_checkpoint(str(d), str(tmp_path / "out.auto_chasm"))

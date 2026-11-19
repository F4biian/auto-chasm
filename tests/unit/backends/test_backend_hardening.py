"""Hardening: rep-penalty bounds, MLX guard breadth, empty-save warn."""

from __future__ import annotations

import logging
import tempfile

import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm._generation_utils import filter_logits
from auto_chasm.backends import Backend
from auto_chasm.backends.mlx_backend import _lora_module_types


def test_filter_logits_ignores_out_of_range_ids() -> None:
    """A stray out-of-range token id in the context is ignored, not wrapped/IndexError'd."""
    out = filter_logits(np.array([1.0, 2.0, 3.0]), 0.0, None, None, 2.0, [0, 5, -1])
    # Only the valid id 0 is penalised (1.0 / 2); 5 and -1 are dropped, so id 2 (the
    # numpy wraparound target of -1) is untouched.
    np.testing.assert_allclose(out, [0.5, 2.0, 3.0])


def test_lora_module_types_covers_all_adapter_kinds() -> None:
    """The MLX guard's type set includes switch and embedding adapters, not just linear."""
    names = {t.__name__ for t in _lora_module_types()}
    assert {"LoRALinear", "LoRASwitchLinear", "DoRALinear", "DoRAEmbedding"} <= names


class _AttnModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        blk = nn.Module()
        blk.self_attn = nn.Module()
        blk.self_attn.q_proj = nn.Linear(8, 8)
        self.layers = [blk]
        self.output_proj = nn.Linear(8, 16)


def test_double_apply_still_raises_clean_error() -> None:
    """Re-applying adapters raises the clean guard error (now via the full type set)."""
    backend = Backend(force="mlx")
    m = _AttnModel()
    backend.wrapping.apply_adapters(m, {"r": 4, "alpha": 8}, ["self_attn.q_proj"])
    with pytest.raises(ValueError, match="already has LoRA/DoRA adapters"):
        backend.wrapping.apply_adapters(m, {"r": 4, "alpha": 8}, ["self_attn.q_proj"])


def test_mlx_save_adapters_warns_on_no_adapter_model(caplog) -> None:  # noqa: ANN001
    """MLX save_adapters warns when the model has no adapters (parity with torch)."""
    backend = Backend(force="mlx")
    m = _AttnModel()  # never adapted
    with caplog.at_level(logging.WARNING, logger="auto_chasm.backends.mlx_backend"):
        backend.wrapping.save_adapters(m, tempfile.mktemp(suffix=".safetensors"))
    assert any("nothing to" in r.message for r in caplog.records)

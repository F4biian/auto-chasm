"""The torch backend must write REAL safetensors into ``adapters.safetensors``.

It used to write a ``torch.save`` pickle under that name. torch >= 2.6 dispatches
``torch.load`` on the ``.safetensors`` extension to safetensors, so the pickle
became unreadable by the very loader meant to read it: a finished training run
died at checkpoint time with

    SafetensorError: Error while deserializing header: header too large

raised from ``_count_tensor_file`` -- a call whose only purpose is deciding
whether to log a warning.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from auto_chasm.backends.torch_backend import TorchModelWrapping  # noqa: E402
from auto_chasm.checkpoint import _count_tensor_file  # noqa: E402


class _Fake(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.Linear(4, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, 4, bias=False)
        self.base = torch.nn.Linear(4, 4, bias=False)


def _adapters(tmp_path: Path) -> tuple[TorchModelWrapping, _Fake, Path]:
    wrapping = TorchModelWrapping()
    model = _Fake()
    path = tmp_path / "adapters.safetensors"
    wrapping.save_adapters(model, str(path))
    return wrapping, model, path


def test_save_adapters_writes_real_safetensors(tmp_path: Path) -> None:
    """The bytes on disk must parse as a safetensors header, not as a pickle."""
    _, _, path = _adapters(tmp_path)
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    assert sorted(k for k in header if k != "__metadata__") == [
        "lora_A.weight",
        "lora_B.weight",
    ]


def test_count_tensor_file_reads_the_saved_adapters(tmp_path: Path) -> None:
    """The exact call that killed the cluster job."""
    _, _, path = _adapters(tmp_path)
    assert _count_tensor_file(path) == 2


def test_adapters_round_trip_and_leave_base_alone(tmp_path: Path) -> None:
    wrapping, model, path = _adapters(tmp_path)
    fresh = _Fake()
    with torch.no_grad():
        for param in fresh.parameters():
            param.zero_()
    wrapping.load_adapters(fresh, str(path))
    assert torch.equal(fresh.lora_A.weight, model.lora_A.weight)
    assert torch.equal(fresh.lora_B.weight, model.lora_B.weight)
    # only "lora_" keys are saved, so the base must still be the zeros we set
    assert torch.equal(fresh.base.weight, torch.zeros(4, 4))


def test_legacy_pickle_checkpoints_still_load(tmp_path: Path) -> None:
    """Checkpoints written before the fix must not become unreadable."""
    wrapping = TorchModelWrapping()
    model = _Fake()
    path = tmp_path / "legacy.safetensors"
    torch.save({k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}, path)

    fresh = _Fake()
    wrapping.load_adapters(fresh, str(path))
    assert torch.equal(fresh.lora_A.weight, model.lora_A.weight)
    assert _count_tensor_file(path) == 2


def test_unreadable_file_warns_instead_of_raising(tmp_path: Path) -> None:
    """A warning heuristic must never take down a finished training run."""
    path = tmp_path / "corrupt.safetensors"
    path.write_bytes(b"\x00" * 32)
    assert _count_tensor_file(path) == 0

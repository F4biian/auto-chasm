"""Oracle test for PyTorch gradient accumulation.

The bug: the torch training loop stepped the optimizer every micro-batch
and ignored ``grad_accum_steps`` entirely, so the effective batch size was
silently ``grad_accum_steps`` times smaller than requested.  This counts
the actual optimizer updates and asserts accumulation reduces them exactly.
"""

from __future__ import annotations

import torch

from auto_chasm import Model
from auto_chasm.trainers.loss import JointLoss
from auto_chasm.trainers.trainer import Trainer

# Capture the genuine optimizer step before any test patches it.
_ORIG_ADAMW_STEP = torch.optim.AdamW.step


class _Tok:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3, 4, 5]

    def decode(self, ids: list[int], **kwargs: object) -> str:
        return "x"


def _wrapper() -> Model:
    from tests.conftest import _make_torch_tiny_mlp

    torch.manual_seed(0)
    m = _make_torch_tiny_mlp(hidden_dim=16, vocab_size=32, num_layers=4)

    class Config:
        """Minimal model config for the tiny torch model."""

        hidden_size = 16
        num_hidden_layers = 4
        vocab_size = 32

    m.config = Config()
    return Model(m, _Tok(), backend_name="torch")


def _count_updates(monkeypatch, tmp_path, accum: int, num_iters: int) -> int:
    counter = {"n": 0}

    def counting_step(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        return _ORIG_ADAMW_STEP(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)

    trainer = Trainer(
        model=_wrapper(),
        loss_fn=JointLoss(),
        num_iters=num_iters,
        batch_size=4,
        grad_accum_steps=accum,
        logging_steps=100,
        save_steps=0,
        early_stopping_patience=0,
        verbose=False,
        output_dir=str(tmp_path / f"out_{accum}"),
    )
    data = [{"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 0, 0]} for _ in range(16)]
    trainer.train(data)
    return counter["n"]


def test_grad_accum_reduces_optimizer_steps(tmp_path, monkeypatch):
    """6 iters with accum=1 => 6 updates; with accum=2 => 3 updates."""
    n_no_accum = _count_updates(monkeypatch, tmp_path, accum=1, num_iters=6)
    n_accum2 = _count_updates(monkeypatch, tmp_path, accum=2, num_iters=6)
    assert n_no_accum == 6
    assert n_accum2 == 3

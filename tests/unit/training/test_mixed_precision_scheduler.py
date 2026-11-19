"""Mixed-precision regressions: fp16 scheduler sync + bf16 cast reversion."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from auto_chasm import Model, ProbeConfig  # noqa: E402
from auto_chasm.config import TrainingConfig  # noqa: E402
from auto_chasm.trainers.loss import JointLoss  # noqa: E402
from auto_chasm.trainers.trainer import Trainer  # noqa: E402


def _model() -> Model:
    from tests.conftest import DummyTokenizer, _make_torch_tiny_mlp

    torch.manual_seed(0)
    base = _make_torch_tiny_mlp(hidden_dim=16, vocab_size=32, num_layers=2)

    class _C:
        hidden_size = 16
        num_hidden_layers = 2

    base.config = _C()
    m = Model(base, DummyTokenizer(), backend_name="torch")
    m.attach_probe(ProbeConfig(name="p", layers=[0]))
    return m


def _data(n: int = 12) -> list[dict[str, list[int]]]:
    return [{"tokens": [1, 2, 3, 4, 5], "labels": [1, 1, 1, 1, 1]} for _ in range(n)]


def _trainer(m: Model, mp: str, loss=None) -> Trainer:  # noqa: ANN001
    return Trainer(
        model=m,
        loss_fn=loss or JointLoss(weights={"lm_head": 0.0}, losses={"p": "bce"}),
        config=TrainingConfig(mixed_precision=mp, batch_size=4),
        num_iters=10,
        learning_rate=5e-2,
        verbose=False,
    )


class _InfOnce:
    """Wrap a loss so the first call yields an inf loss (forces an fp16 scaler skip)."""

    def __init__(self, real: JointLoss) -> None:
        self.real = real
        self.calls = 0

    def __call__(self, model, tokens, labels, lengths):  # noqa: ANN001, ANN204
        total, ntoks, components = self.real(model, tokens, labels, lengths)
        self.calls += 1
        if self.calls == 1:
            total = total * float("inf")  # inf grads -> GradScaler skips the step
        return total, ntoks, components


def test_fp16_skipped_step_does_not_advance_scheduler() -> None:
    """On an fp16 grad overflow the optimizer step is skipped; the LR schedule must not
    advance for it (else it skips the peak LR and floors early)."""
    m = _model()
    loss = _InfOnce(JointLoss(weights={"lm_head": 0.0}, losses={"p": "bce"}))
    tr = _trainer(m, "fp16", loss=loss)
    batches = tr.iterate(_data())
    for _ in range(3):
        tr.step(next(batches))  # step 1 overflows (skipped), steps 2 & 3 real
    scheduler = tr._torch_step_state["scheduler"]
    assert scheduler.last_epoch == 2  # exactly the two applied steps, not 3


def test_fp16_normal_run_keeps_scheduler_in_sync() -> None:
    """Without overflow the guard is a no-op: the schedule advances once per step."""
    m = _model()
    tr = _trainer(m, "fp16")
    batches = tr.iterate(_data())
    for _ in range(4):
        tr.step(next(batches))
    assert tr._torch_step_state["scheduler"].last_epoch == 4


def test_bf16_cast_reverts_on_reused_model() -> None:
    """A bf16 run casts the base; a later fp32 run on the SAME Model restores fp32
    weights instead of silently keeping the stale bf16 base."""
    m = _model()
    assert next(m.model.parameters()).dtype == torch.float32
    tr_bf16 = _trainer(m, "bf16")
    tr_bf16.step(next(tr_bf16.iterate(_data())))
    assert next(m.model.parameters()).dtype == torch.bfloat16  # cast happened

    tr_fp32 = _trainer(m, "fp32")  # a fresh trainer on the reused Model
    tr_fp32.step(next(tr_fp32.iterate(_data())))
    assert next(m.model.parameters()).dtype == torch.float32  # restored, not stale bf16


def test_bf16_reused_stays_bf16() -> None:
    """A second bf16 run on the same model keeps bf16 (restore only kicks in for fp32/fp16)."""
    m = _model()
    tr1 = _trainer(m, "bf16")
    tr1.step(next(tr1.iterate(_data())))
    tr2 = _trainer(m, "bf16")
    tr2.step(next(tr2.iterate(_data())))
    assert next(m.model.parameters()).dtype == torch.bfloat16

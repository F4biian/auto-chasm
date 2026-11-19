"""Early stopping and best-checkpoint rollback are OPT-IN.

Both used to be on: ``early_stopping_patience`` defaulted to 15, and the rollback
was not gated on it at all -- any run given ``val_data`` was rewound at the end of
``train()`` to whichever eval scored best. A fixed-budget run that merely wanted a
val curve silently got weights from some earlier step, and an unlikelihood
("unlearning") run got the worst of it: its val loss rises by construction, so the
"best" checkpoint is an early one and most of the unlearning was thrown away.

These pin the defaults and the wiring so neither can drift back.
"""

from __future__ import annotations

import inspect

from auto_chasm.trainers.base import JointTrainer
from auto_chasm.trainers.sft import SFTTrainer
from auto_chasm.trainers.trainer import Trainer


def _default(fn, name: str):
    return inspect.signature(fn).parameters[name].default


def test_early_stopping_off_by_default() -> None:
    for cls in (Trainer, JointTrainer, SFTTrainer):
        assert _default(cls.__init__, "early_stopping_patience") == 0, cls.__name__


def test_restore_best_off_by_default() -> None:
    for cls in (Trainer, JointTrainer, SFTTrainer):
        assert _default(cls.__init__, "restore_best_weights") is False, cls.__name__


def test_trainer_forwards_restore_best_to_the_joint_trainer() -> None:
    """The facade stores it and hands it to every JointTrainer it builds."""
    src = inspect.getsource(Trainer)
    assert "self.restore_best_weights = restore_best_weights" in src
    # One forward per JointTrainer construction site, or the flag is silently lost.
    assert src.count("restore_best_weights=self.restore_best_weights") == src.count(
        "early_stopping_patience=self.early_stopping_patience"
    )


def test_sft_forwards_restore_best() -> None:
    assert "restore_best_weights=restore_best_weights" in inspect.getsource(SFTTrainer)


def test_both_backends_gate_the_rollback() -> None:
    """MLX and torch must agree -- an ungated rollback on one side is a silent
    cross-backend divergence in what ``train()`` returns."""
    from auto_chasm.trainers import _torch_loop

    assert "and self.restore_best_weights" in inspect.getsource(JointTrainer)
    assert "and trainer.restore_best_weights" in inspect.getsource(_torch_loop)


def test_manifest_records_which_weights_it_describes() -> None:
    from auto_chasm.trainers._metrics import torch_manifest

    m = torch_manifest(
        base_model="m", best_iter=3, best_metric=0.1, best_metric_name="val_loss",
        num_iters=10, early_stopping_patience=0, min_delta=1e-4, keep_best_only=False,
    )
    assert m["restore_best_weights"] is False
    assert m["best_iter"] == 3  # tracking still reported even with the rollback off

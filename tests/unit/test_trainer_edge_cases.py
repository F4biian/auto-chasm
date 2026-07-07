"""Regression tests for trainer edge cases.

- m8: eval accuracy excludes ignored (-100) labels even inside the length window
  (they no longer count as wrong predictions).
- m9: torch eval loss is token-weighted (matches MLX), and val metrics land on a
  dedicated history entry at the eval step (never dropped / mis-attributed).
- m10: an explicit Trainer kwarg wins over a TrainingConfig even when it equals
  the library default; keep_best_only keeps a digit-leading probe's head file.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


class _TinyMlp(nn.Module):
    """Minimal MLX language model."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = [nn.Linear(8, 8) for _ in range(2)]
        self.output_proj = nn.Linear(8, 16)

    def __call__(self, x: mx.array, **kwargs: object) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Tok:
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int], **kwargs: object) -> str:
        return "x"


class _Cfg:
    hidden_size = 8
    num_hidden_layers = 2


def _mlx_model():  # type: ignore[no-untyped-def]
    from auto_chasm.model import Model

    base = _TinyMlp()
    base.config = _Cfg()
    return Model(base, _Tok(), "mlx")


def test_m8_accuracy_excludes_in_window_ignore_labels(monkeypatch) -> None:  # noqa: ANN001
    """A -100 label inside the length window is not counted as a wrong prediction (m8)."""
    from auto_chasm.trainers import trainable

    # Predictions from fixed logits: sigmoid(+10)>0.5 -> 1, sigmoid(-10) -> 0.
    monkeypatch.setattr(
        "auto_chasm.metrics.run_probe",
        lambda _tm, _name, _hidden: mx.array([[10.0, 10.0, -10.0, 10.0]]),
    )
    captured = {"p": mx.array([[0.0]])}  # ignored by the patched run_probe
    targets = mx.array([[1, 1, 0, -100]])  # last position is in-window but ignored
    mask = mx.array([[1, 1, 1, 1]])  # every position is inside the length window

    result = trainable.default_binary_metrics(object(), captured, targets, mask)
    # preds = [1, 1, 0, 1]; positions 0..2 all correct, position 3 is -100 (excluded).
    # Old code counted position 3 in the denominator -> 3/4 = 0.75. Now -> 3/3 = 1.0.
    assert result["p_accuracy"] == pytest.approx(1.0)


def test_m9_torch_eval_loss_is_token_weighted(monkeypatch) -> None:  # noqa: ANN001
    """evaluate_torch_model token-weights the loss like MLX, not a per-batch mean (m9)."""
    pytest.importorskip("torch")
    import torch

    from auto_chasm.trainers import _metrics

    # Two single-sample batches with different token counts.
    def fake_iterate(dataset, batch_size, max_seq_length, loop=False):  # noqa: ANN001, ANN202
        for tokens in ([[1, 2, 3]], [[4, 5, 6]]):
            yield (
                np.array(tokens, dtype=np.int32),
                np.array(tokens, dtype=np.int32),
                np.array([[0, 2]], dtype=np.int32),
            )

    calls = {"i": 0}

    def fake_loss(_model, _tokens, _labels, _lengths):  # noqa: ANN001, ANN202
        # batch 0: loss 1.0 over 10 tokens; batch 1: loss 3.0 over 30 tokens.
        loss, ntoks = ((1.0, 10.0), (3.0, 30.0))[calls["i"]]
        calls["i"] += 1
        return torch.tensor(loss), torch.tensor(ntoks), {}

    monkeypatch.setattr(_metrics, "iterate_batches", fake_iterate, raising=False)
    # iterate_batches is imported inside evaluate_torch_model from data_utils:
    monkeypatch.setattr("auto_chasm.trainers.data_utils.iterate_batches", fake_iterate)

    class _Wrapper:
        model = torch.nn.Linear(2, 2)
        _probes: dict = {}

    result = _metrics.evaluate_torch_model(
        _Wrapper(),
        dataset=[0, 1],
        batch_size=1,
        max_seq_length=8,
        loss_fn=fake_loss,
        eval_metrics_fn=None,
    )
    # token-weighted: (1*10 + 3*30) / (10+30) = 100/40 = 2.5  (per-batch mean would be 2.0)
    assert result["loss"] == pytest.approx(2.5)


def test_m9_torch_val_metrics_recorded_at_eval_step(tmp_path) -> None:
    """Torch val metrics land on their own history entry, never dropped (m9)."""
    pytest.importorskip("torch")

    from auto_chasm import Model
    from auto_chasm.config import ProbeConfig
    from auto_chasm.trainers.loss import JointLoss
    from auto_chasm.trainers.trainer import Trainer
    from tests.conftest import DummyTokenizer, _make_torch_tiny_mlp

    torch_model = _make_torch_tiny_mlp(hidden_dim=4, vocab_size=8, num_layers=2)

    class _Cfg4:
        hidden_size = 4
        num_hidden_layers = 2

    torch_model.config = _Cfg4()
    wrapper = Model(torch_model, DummyTokenizer(), backend_name="torch")
    wrapper.attach_probe(ProbeConfig(name="p", layers=[-1]))
    wrapper.prepare_for_joint_training()

    data = [{"tokens": [1, 2, 3], "labels": [0, 0, 1]} for _ in range(2)]
    # logging_steps=5 never fires in 2 iters, so there is NO logging entry to graft
    # onto: the old code dropped every eval's val metrics.
    trainer = Trainer(
        model=wrapper,
        loss_fn=JointLoss(),
        num_iters=2,
        batch_size=1,
        max_seq_length=8,
        logging_steps=5,
        eval_steps=1,
        early_stopping_patience=0,
        output_dir=str(tmp_path / "out"),
        verbose=False,
    )
    result = trainer.train(data, val_data=data)
    history = result["history"]
    assert any(getattr(e, "val_metrics", None) for e in history), (
        "m9: torch val metrics were dropped (no logging entry to attach to)."
    )


def test_m10_explicit_kwarg_overrides_config_even_when_equal_to_default() -> None:
    """An explicit kwarg equal to the library default still wins over config (m10)."""
    from auto_chasm.config import TrainingConfig
    from auto_chasm.trainers.loss import JointLoss
    from auto_chasm.trainers.trainer import Trainer

    m = _mlx_model()
    # config asks for lr=0.01; the user explicitly passes the library default 2e-4.
    trainer = Trainer(
        model=m,
        loss_fn=JointLoss(),
        learning_rate=2e-4,  # explicit, and equal to the default
        config=TrainingConfig(learning_rate=0.01),
        num_iters=1,
        verbose=False,
    )
    assert trainer.learning_rate == pytest.approx(2e-4)  # explicit wins, not 0.01


def test_m10_config_still_fills_unset_kwargs() -> None:
    """With no explicit kwarg, config supplies the value (m10 -- no regression)."""
    from auto_chasm.config import TrainingConfig
    from auto_chasm.trainers.loss import JointLoss
    from auto_chasm.trainers.trainer import Trainer

    m = _mlx_model()
    trainer = Trainer(
        model=m,
        loss_fn=JointLoss(),
        config=TrainingConfig(learning_rate=0.01),
        num_iters=1,
        verbose=False,
    )
    assert trainer.learning_rate == pytest.approx(0.01)  # config fills the unset kwarg

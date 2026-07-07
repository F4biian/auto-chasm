"""Regression tests: sweep scoring, config precedence, torch eval cadence.

- F1: LayerSweep._score raises on a metric the eval does not produce (was a silent
  0.0 that tied every layer and disabled per-layer best selection); f1 -> macro_f1.
- F3: JointTrainer/SFTTrainer honor an explicit kwarg over config even when it equals
  a library default (the _UNSET sentinel), and SFTTrainer forwards config fields.
- F5: the torch loop no longer evaluates at step 1 (parity with the MLX loop).
"""

from __future__ import annotations

import mlx.nn as nn
import pytest

from auto_chasm import JointLoss, Model
from auto_chasm.config import TrainingConfig
from auto_chasm.sweep import _BestPerLayerCallback
from auto_chasm.trainers.base import JointTrainer
from auto_chasm.trainers.sft import SFTTrainer


class _TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = [nn.Linear(8, 8)]
        self.output_proj = nn.Linear(8, 16)

    def __call__(self, x, **k):  # noqa: ANN001, ANN003, ANN204
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = 8
    num_hidden_layers = 1


def _mlx_model() -> Model:
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    return m


def test_f1_sweep_score_raises_on_missing_metric_and_aliases_f1() -> None:
    """_score fails loudly on an absent metric (was silent 0.0) and maps f1->macro_f1."""
    cb = _BestPerLayerCallback.__new__(_BestPerLayerCallback)
    cb.score_metric = "val_f1"
    assert cb._score({"L0_macro_f1": 0.8, "L0_loss": 0.3}, "L0") == 0.8  # f1 alias
    with pytest.raises(ValueError, match="needs metric 'L0_macro_f1'"):
        cb._score({"L0_loss": 0.3}, "L0")  # no macro_f1 this eval -> loud, not 0.0


def test_f3_jointtrainer_explicit_beats_config_even_when_equal_to_default() -> None:
    """An explicit JointTrainer kwarg equal to the default still wins over config."""
    m = _mlx_model()
    t = JointTrainer(
        model=m,
        loss_fn=JointLoss(weights={"lm_head": 1.0}),
        learning_rate=2e-4,  # explicit, equal to the library default
        batch_size=8,  # explicit, equal to the default
        config=TrainingConfig(learning_rate=0.01, batch_size=16),
        num_iters=1,
    )
    assert t._base_lr == pytest.approx(2e-4) and t.batch_size == 8
    # With no explicit kwargs, config fills them.
    t2 = JointTrainer(
        model=_mlx_model(),
        loss_fn=JointLoss(weights={"lm_head": 1.0}),
        config=TrainingConfig(learning_rate=0.01, batch_size=16),
        num_iters=1,
    )
    assert t2._base_lr == pytest.approx(0.01) and t2.batch_size == 16


def test_f3_sfttrainer_forwards_config_fields() -> None:
    """SFTTrainer forwards config (lr_schedule/eval_steps/lm_weight) to JointTrainer."""
    m = _mlx_model()
    trainer = SFTTrainer(
        model=m, num_iters=40, config=TrainingConfig(lr_schedule="linear", eval_steps=7)
    )
    jt = trainer._trainer
    assert jt.eval_steps == 7  # config's eval_steps reached JointTrainer (was 100)
    # A "linear" warmup schedule floors at 0 past the horizon (cosine would not).
    import mlx.core as mx

    assert float(jt.lr_schedule(mx.array(40.0))) == pytest.approx(0.0, abs=1e-9)


def test_f5_torch_loop_does_not_eval_at_step_one(tmp_path) -> None:
    """The torch loop evaluates on cadence / at the end, not at step 1 (MLX parity)."""
    pytest.importorskip("torch")
    from auto_chasm.config import ProbeConfig
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
    # eval_steps=5 never divides steps 1..3, so the only eval is the final-step one.
    result = Trainer(
        model=wrapper,
        loss_fn=JointLoss(),
        num_iters=3,
        batch_size=1,
        eval_steps=5,
        early_stopping_patience=0,
        output_dir=str(tmp_path / "out"),
        verbose=False,
    ).train(data, val_data=data)
    val_steps = [e.step for e in result["history"] if getattr(e, "val_metrics", None)]
    assert 1 not in val_steps  # step 1 is no longer an eval point
    assert val_steps == [3]  # only the final step

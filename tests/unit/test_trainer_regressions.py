"""Regression tests for trainer.

Each test pins one confirmed bug fix against an independent ground truth, not
merely "runs without crashing":

1.  granularity="response" probes train without crashing AND ignore padding.
2.  evaluate_joint_model averages custom metrics by the true batch count.
3.  MLX "linear" LR schedule never returns a negative learning rate.
4.  MLX gradient accumulation flushes the final partial group (update count).
5.  on_step_end callbacks fire N times on MLX (parity with torch).
6.  eval_metrics_fn makes a non-loss metric ("val_f1") reachable for early stop.
7.  SFTTrainer forwards a TrainingConfig's lr_schedule/warmup/eval_steps.
8.  The torch Trainer's evaluate/get_history/save_checkpoint/iterate work;
    step/restore raise a clear NotImplementedError.
9.  All four trainers' .train() return the same {"history", "output_dir"} keys.
10. SFTTrainer/RLTrainer raise a clear error on a non-MLX backend.
11. (docstring) loss components are tensors — asserted by type, not prose.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import ProbeConfig, RLConfig, TrainingConfig
from auto_chasm.model import Model
from auto_chasm.trainers.base import JointTrainer
from auto_chasm.trainers.loss import JointLoss
from auto_chasm.trainers.rl import RLTrainer
from auto_chasm.trainers.sft import SFTTrainer
from auto_chasm.trainers.trainable import _TrainableModel, evaluate_joint_model
from auto_chasm.trainers.trainer import Trainer
from auto_chasm.trainers.wrappers import TrainerCallback


class _TinyMlp(nn.Module):
    """Minimal MLX language model for runnable trainer tests."""

    def __init__(self, hidden_dim: int = 8, vocab_size: int = 16, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array, **kwargs: Any) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Tok:
    """Minimal tokenizer for trainer tests."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return "x"


class _Cfg:
    """Minimal model config."""

    hidden_size = 8
    num_hidden_layers = 2


def _mlx_model() -> Model:
    """Build a tiny MLX-backed Model."""
    base = _TinyMlp()
    base.config = _Cfg()
    return Model(base, _Tok(), "mlx")


def _torch_model() -> Model:
    """Build a tiny torch-backed Model mirroring the MLX TinyMlp."""
    import torch
    import torch.nn as tnn

    class TorchTinyMlp(tnn.Module):
        """Tiny torch MLP."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = tnn.Embedding(16, 8)
            self.layers = tnn.ModuleList([tnn.Linear(8, 8) for _ in range(2)])
            self.output_proj = tnn.Linear(8, 16)

        def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
            h = self.embedding(x)
            for layer in self.layers:
                h = torch.nn.functional.gelu(layer(h))
            return self.output_proj(h)

    torch.manual_seed(0)
    base = TorchTinyMlp()
    base.config = _Cfg()
    return Model(base, _Tok(), "torch")


def _data(n: int = 8) -> list[dict]:
    """A small token-label dataset."""
    return [{"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 0, 0]} for _ in range(n)]


# ---------------------------------------------------------------------------
# Bug 1 — response granularity trains; padding is ignored
# ---------------------------------------------------------------------------


class TestBug1ResponseGranularity:
    """granularity='response' must train without crashing and ignore padding."""

    def test_response_loss_does_not_crash(self) -> None:
        m = _mlx_model()
        m.attach_probe(ProbeConfig(name="d", layers=[1], source="hidden", granularity="response"))
        tm = _TrainableModel(m.model, m._probes)
        batch = mx.array([[1, 2, 3, 4, 5, 6]])
        labels = mx.array([[-100, 1, 1, 1, 1, -100]])
        lengths = mx.array([[0, 5]])
        total, _, comp = JointLoss(weights={"lm_head": 0.0}, losses={"d": "bce"})(
            tm, batch, labels, lengths
        )
        assert "d" in comp
        assert float(total) == pytest.approx(float(comp["d"]), rel=1e-6)

    def test_response_pooling_ignores_padding(self) -> None:
        """Oracle: appending padding beyond ``lengths`` must not change the loss."""
        mx.random.seed(0)
        m = _mlx_model()
        m.attach_probe(ProbeConfig(name="d", layers=[1], source="hidden", granularity="response"))
        m._probes["d"].module.weight = m._probes["d"].module.weight + 0.5
        tm = _TrainableModel(m.model, m._probes)
        jl = JointLoss(weights={"lm_head": 0.0}, losses={"d": "bce"})

        batch_a = mx.array([[1, 2, 3, 4, 5]])
        labels_a = mx.array([[-100, 1, 1, 1, -100]])
        lengths = mx.array([[1, 4]])
        loss_a, _, _ = jl(tm, batch_a, labels_a, lengths)

        batch_b = mx.array([[1, 2, 3, 4, 5, 9, 9, 9]])
        labels_b = mx.array([[-100, 1, 1, 1, -100, -100, -100, -100]])
        loss_b, _, _ = jl(tm, batch_b, labels_b, lengths)

        assert float(loss_a) == pytest.approx(float(loss_b), abs=1e-6)

    def test_response_probe_trains_end_to_end(self) -> None:
        """A full Trainer.train run with a response probe completes and steps."""
        m = _mlx_model()
        m.attach_probe(ProbeConfig(name="d", layers=[1], source="hidden", granularity="response"))
        trainer = Trainer(
            model=m,
            loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"d": "bce"}),
            num_iters=4,
            batch_size=2,
            logging_steps=1,
            save_steps=0,
            early_stopping_patience=0,
            verbose=False,
            output_dir="/tmp/ach_regr_resp",
        )
        result = trainer.train(_data())
        assert len(result["history"].train_losses) >= 1


# ---------------------------------------------------------------------------
# Bug 2 — evaluate_joint_model divides by the true batch count
# ---------------------------------------------------------------------------


class TestBug2EvalMetricBatchCount:
    """A constant metric must average to its constant, not be deflated."""

    def test_constant_metric_not_deflated(self) -> None:
        m = _mlx_model()
        m.attach_probe(ProbeConfig(name="p", layers=[0], aggregation="last"))
        tm = _TrainableModel(m.model, m._probes)

        def const_metric(_tm: Any, _cap: Any, _tgt: Any, _mask: Any) -> dict[str, float]:
            return {"const": 1.0}

        # 16 samples, batch 4 -> 4 batches; the old `len//bs + 1 == 5` divisor
        # deflated 1.0 to 0.8.
        result = evaluate_joint_model(
            train_model=tm,
            dataset=_data(16),
            batch_size=4,
            max_seq_length=16,
            loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"p": "bce"}),
            eval_metrics_fn=const_metric,
        )
        assert result["const"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Bug 3 — MLX linear LR schedule never negative
# ---------------------------------------------------------------------------


class TestBug3LinearLrFloor:
    """The linear schedule must floor at 0, like cosine and torch LinearLR."""

    def test_linear_lr_never_negative(self) -> None:
        sched = JointTrainer._build_lr_schedule("linear", 1e-3, num_iters=5, warmup_steps=0)
        for i in range(12):
            lr = float(sched(mx.array(float(i))))
            assert lr >= 0.0, f"negative LR {lr} at step {i}"
        # Oracle: at and past the horizon the LR is exactly 0.
        assert float(sched(mx.array(5.0))) == pytest.approx(0.0, abs=1e-12)
        assert float(sched(mx.array(9.0))) == pytest.approx(0.0, abs=1e-12)
        # Sanity: mid-horizon it is the expected linear value.
        assert float(sched(mx.array(2.0))) == pytest.approx(1e-3 * (1 - 2 / 5), rel=1e-6)


# ---------------------------------------------------------------------------
# Bug 4 — MLX grad-accum flushes the final partial group
# ---------------------------------------------------------------------------


class TestBug4GradAccumTailFlush:
    """The optimizer-update count must include the tail flush."""

    def _count_updates(self, num_iters: int, accum: int) -> int:
        m = _mlx_model()
        m.attach_probe(ProbeConfig(name="p", layers=[0], aggregation="last"))
        trainer = JointTrainer(
            model=m,
            loss_fn=JointLoss(losses={"p": "bce"}),
            num_iters=num_iters,
            batch_size=2,
            grad_accum_steps=accum,
            logging_steps=100,
            save_steps=0,
            early_stopping_patience=0,
            verbose=False,
            output_dir=f"/tmp/ach_regr_accum_{num_iters}_{accum}",
        )
        counter = {"n": 0}
        orig = trainer.optimizer.update

        def counting_update(model: Any, grad: Any) -> Any:
            counter["n"] += 1
            return orig(model, grad)

        trainer.optimizer.update = counting_update  # type: ignore[method-assign]
        trainer.run(_data(8))
        return counter["n"]

    def test_partial_group_is_flushed(self) -> None:
        # 3 iters, accum 2 -> 1 boundary update + 1 tail flush = 2 (was 1).
        assert self._count_updates(num_iters=3, accum=2) == 2
        # 5 iters, accum 2 -> updates at 2,4 + tail at 5 = 3 (was 2).
        assert self._count_updates(num_iters=5, accum=2) == 3
        # Exact multiple: no extra flush (last iter is already a boundary).
        assert self._count_updates(num_iters=4, accum=2) == 2


# ---------------------------------------------------------------------------
# Bug 5 — MLX on_step_end callbacks fire N times
# ---------------------------------------------------------------------------


class _CountingCallback(TrainerCallback):
    """Counts on_step_end invocations."""

    def __init__(self) -> None:
        self.steps = 0

    def on_step_end(self, **kwargs: Any) -> None:
        self.steps += 1


class TestBug5MlxCallbacksFire:
    """on_step_end must fire once per iteration on MLX (parity with torch)."""

    def test_callback_fires_per_step(self) -> None:
        m = _mlx_model()
        m.attach_probe(ProbeConfig(name="p", layers=[0], aggregation="last"))
        cb = _CountingCallback()
        trainer = Trainer(
            model=m,
            loss_fn=JointLoss(losses={"p": "bce"}),
            num_iters=5,
            batch_size=2,
            logging_steps=100,
            save_steps=0,
            early_stopping_patience=0,
            callbacks=[cb],
            verbose=False,
            output_dir="/tmp/ach_regr_cb",
        )
        trainer.train(_data())
        assert cb.steps == 5


# ---------------------------------------------------------------------------
# Bug 6 — eval_metrics_fn enables F1 early stopping
# ---------------------------------------------------------------------------


class TestBug6EvalMetricsEarlyStopping:
    """early_stopping_metric='val_f1' must be reachable via eval_metrics_fn."""

    def test_val_f1_resolvable(self) -> None:
        from auto_chasm.trainers.trainable import default_binary_metrics

        m = _mlx_model()
        m.attach_probe(ProbeConfig(name="p", layers=[0], aggregation="last"))

        def metrics_fn(tm: Any, captured: Any, targets: Any, mask: Any) -> dict[str, float]:
            raw = default_binary_metrics(tm, captured, targets, mask)
            # Collapse the per-probe key to the bare 'f1' the resolver looks for.
            return {"f1": raw["p_f1"]}

        trainer = Trainer(
            model=m,
            loss_fn=JointLoss(losses={"p": "bce"}),
            num_iters=4,
            batch_size=2,
            eval_steps=2,
            logging_steps=100,
            save_steps=0,
            early_stopping_patience=2,
            early_stopping_metric="val_f1",
            early_stopping_higher_is_better=True,
            eval_metrics_fn=metrics_fn,
            verbose=False,
            output_dir="/tmp/ach_regr_f1",
        )
        # Without the eval_metrics_fn plumbing this raised KeyError on 'val_f1'.
        result = trainer.train(_data(8), val_data=_data(8))
        # The recorded val metrics must contain f1 (proving the path is live).
        val_entries = [e for e in result["history"] if e.val_metrics]
        assert val_entries, "no validation metrics recorded"
        assert any("f1" in e.val_metrics for e in val_entries)


# ---------------------------------------------------------------------------
# Bug 7 — SFTTrainer honors a TrainingConfig's schedule fields
# ---------------------------------------------------------------------------


class TestBug7SftHonorsConfig:
    """SFTTrainer must forward lr_schedule/warmup/eval_steps from the config."""

    def test_lr_schedule_forwarded(self) -> None:
        cfg = TrainingConfig(
            lr_schedule="linear",
            warmup_ratio=0.25,
            eval_steps=7,
        )
        m = _mlx_model()
        trainer = SFTTrainer(model=m, num_iters=40, config=cfg)
        # The schedule honoring is observable: a "linear" schedule with warmup
        # floors at 0 at the horizon, whereas the default cosine would not have
        # been built from the config at all.
        jt = trainer._trainer
        assert jt.eval_steps == 7
        sched = jt.lr_schedule
        # warmup_steps = int(40 * 0.25) = 10; past num_iters the linear tail is 0.
        assert float(sched(mx.array(40.0))) == pytest.approx(0.0, abs=1e-9)
        # At the warmup peak (step 10) the LR equals the base LR.
        assert float(sched(mx.array(10.0))) == pytest.approx(jt._base_lr, rel=1e-5)

    def test_early_stopping_direction_forwarded(self) -> None:
        cfg = TrainingConfig()
        m = _mlx_model()
        trainer = SFTTrainer(
            model=m, num_iters=10, early_stopping_higher_is_better=True, config=cfg
        )
        assert trainer._trainer.early_stopping_higher_is_better is True


# ---------------------------------------------------------------------------
# Bug 8 — torch Trainer escape-hatch consistency
# ---------------------------------------------------------------------------


class TestBug8TorchEscapeHatch:
    """evaluate/get_history/save_checkpoint/iterate/step all work on torch."""

    def _trainer(self) -> Trainer:
        m = _torch_model()
        m.attach_probe(ProbeConfig(name="p", layers=[0], aggregation="last"))
        return Trainer(
            model=m,
            loss_fn=JointLoss(losses={"p": "bce"}),
            num_iters=2,
            batch_size=2,
            logging_steps=100,
            save_steps=0,
            early_stopping_patience=0,
            verbose=False,
            output_dir="/tmp/ach_regr_torch",
        )

    def test_evaluate_works_on_torch(self) -> None:
        trainer = self._trainer()
        metrics = trainer.evaluate(_data(4))
        assert "loss" in metrics and metrics["loss"] >= 0.0

    def test_iterate_works_on_torch(self) -> None:
        trainer = self._trainer()
        it = trainer.iterate(_data(4))
        tokens, labels, lengths = next(iter(it))
        assert tokens.shape[0] == 2

    def test_get_history_and_save_checkpoint_on_torch(self, tmp_path: Any) -> None:
        m = _torch_model()
        m.attach_probe(ProbeConfig(name="p", layers=[0], aggregation="last"))
        trainer = Trainer(
            model=m,
            loss_fn=JointLoss(losses={"p": "bce"}),
            num_iters=2,
            batch_size=2,
            logging_steps=1,
            save_steps=0,
            early_stopping_patience=0,
            verbose=False,
            output_dir=str(tmp_path / "out"),
        )
        trainer.train(_data(4))
        from auto_chasm.history import History

        assert isinstance(trainer.get_history(), History)
        path = trainer.save_checkpoint(str(tmp_path / "ckpt"))
        from pathlib import Path

        assert Path(path).exists()

    def test_step_works_on_torch(self) -> None:
        trainer = self._trainer()
        batch = next(iter(trainer.iterate(_data(4))))
        out = trainer.step(batch)
        assert set(out) == {"loss", "ntoks", "components"}
        assert out["loss"] >= 0.0

    def test_restore_checkpoint_raises_clear_error_on_torch(self) -> None:
        trainer = self._trainer()
        with pytest.raises(NotImplementedError, match="MLX backend"):
            trainer.restore_checkpoint("/tmp/nope")


# ---------------------------------------------------------------------------
# Bug 9 — unified .train() return contract across all four trainers
# ---------------------------------------------------------------------------


class TestBug9UnifiedTrainReturn:
    """Trainer/SFTTrainer/RLTrainer/JointTrainer .train() return the same keys."""

    _REQUIRED = {"history", "output_dir"}

    def test_trainer_train_returns_dict(self) -> None:
        m = _mlx_model()
        m.attach_probe(ProbeConfig(name="p", layers=[0], aggregation="last"))
        result = Trainer(
            model=m,
            loss_fn=JointLoss(losses={"p": "bce"}),
            num_iters=2,
            batch_size=2,
            logging_steps=100,
            save_steps=0,
            early_stopping_patience=0,
            verbose=False,
            output_dir="/tmp/ach_regr_u_trainer",
        ).train(_data())
        assert set(result) >= self._REQUIRED

    def test_joint_trainer_train_returns_dict(self) -> None:
        m = _mlx_model()
        m.attach_probe(ProbeConfig(name="p", layers=[0], aggregation="last"))
        result = JointTrainer(
            model=m,
            loss_fn=JointLoss(losses={"p": "bce"}),
            num_iters=2,
            batch_size=2,
            logging_steps=100,
            save_steps=0,
            early_stopping_patience=0,
            verbose=False,
            output_dir="/tmp/ach_regr_u_joint",
        ).train(_data())
        assert set(result) >= self._REQUIRED

    def test_sft_trainer_train_returns_dict(self) -> None:
        m = _mlx_model()
        m.attach_probe(ProbeConfig(name="p", layers=[0], aggregation="last"))
        result = SFTTrainer(
            model=m,
            num_iters=2,
            batch_size=2,
            logging_steps=100,
            save_steps=0,
            early_stopping_patience=0,
            verbose=False,
            output_dir="/tmp/ach_regr_u_sft",
        ).train(_data())
        assert set(result) >= self._REQUIRED

    def test_rl_trainer_train_returns_dict(self) -> None:
        m = _mlx_model()
        m.attach_probe(ProbeConfig(name="p", layers=[0], aggregation="last"))
        result = RLTrainer(
            model=m,
            rl_config=RLConfig(algorithm="sft", beta=0.1),
            num_iters=2,
            batch_size=2,
            output_dir="/tmp/ach_regr_u_rl",
        ).train(_data())
        assert set(result) >= self._REQUIRED


# ---------------------------------------------------------------------------
# Bug 10 — SFT/RL raise a clear error on non-MLX backends
# ---------------------------------------------------------------------------


class TestBug10NonMlxBackendRaises:
    """SFTTrainer/RLTrainer must fail loudly on torch, not deep in the loop."""

    def test_sft_trainer_rejects_torch(self) -> None:
        m = _torch_model()
        with pytest.raises(ValueError, match="MLX backend"):
            SFTTrainer(model=m, num_iters=2)

    def test_rl_trainer_rejects_torch(self) -> None:
        m = _torch_model()
        with pytest.raises(ValueError, match="MLX backend"):
            RLTrainer(model=m, rl_config=RLConfig(algorithm="sft"), num_iters=2)


# ---------------------------------------------------------------------------
# Bug 11 — loss components are tensors (docstring correctness)
# ---------------------------------------------------------------------------


class TestBug11ComponentsAreTensors:
    """JointLoss components are backend tensors, not Python floats."""

    def test_components_are_mlx_arrays(self) -> None:
        m = _mlx_model()
        m.attach_probe(ProbeConfig(name="p", layers=[0], aggregation="last"))
        tm = _TrainableModel(m.model, m._probes)
        batch = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[0, 0, 1, 0, 0]])
        lengths = mx.array([[0, 4]])
        _, _, comp = JointLoss(losses={"p": "bce"})(tm, batch, labels, lengths)
        assert comp, "expected at least one component"
        for value in comp.values():
            assert isinstance(value, mx.array)
            assert not isinstance(value, float)


# ---------------------------------------------------------------------------
# M21 — response multi-class probe routing (metadata, not tensor shape)
# ---------------------------------------------------------------------------


class TestM21ResponseMulticlassRouting:
    """M21: a response multi-class probe ``[B, C]`` must route by granularity.

    Disambiguating per-token vs sequence-level from tensor shape alone misroutes a
    response multi-class probe ``[B, C]`` whenever a batch pads so ``T-1 == C`` —
    an intermittent, batch-shape-dependent crash. Routing from the probe's declared
    granularity removes the ambiguity.
    """

    def test_sequence_level_prefers_granularity_at_shape_collision(self) -> None:
        """At ``T-1 == C`` the shape heuristic misroutes; granularity fixes it."""
        import numpy as np

        from auto_chasm.trainers._loss_routing import _sequence_level

        logits = np.zeros((2, 5), dtype=np.float32)  # response multi-class [B=2, C=5]
        n_time = 5  # a batch padded so T-1 == C
        # Shape alone can't tell [B, C] (pooled) from [B, T-1] (per-token) here:
        assert _sequence_level(logits, n_time, None) is False  # the misroute
        # The declared granularity resolves it unambiguously:
        assert _sequence_level(logits, n_time, "response") is True
        assert _sequence_level(logits, n_time, "token") is False

    def test_response_multiclass_loss_routes_correctly_at_collision(self) -> None:
        """JointLoss on a response CE probe with ``T-1 == C`` completes via the seq path."""
        import math

        classes = 5
        m = _mlx_model()
        m.attach_probe(
            ProbeConfig(
                name="p",
                layers=[1],
                source="hidden",
                granularity="response",
                module_config={"out_features": classes},
            )
        )
        tm = _TrainableModel(m.model, m._probes)
        # T = C + 1 tokens => T-1 == C: the shape collision that misrouted before.
        batch = mx.array([[1, 2, 3, 4, 5, 6]])
        labels = mx.array([[-100, 3, 3, 3, 3, 3]])  # response class 3 (< C)
        lengths = mx.array([[0, 5]])
        total, _, comp = JointLoss(weights={"lm_head": 0.0}, losses={"p": "ce"})(
            tm, batch, labels, lengths
        )
        assert "p" in comp
        assert math.isfinite(float(total))

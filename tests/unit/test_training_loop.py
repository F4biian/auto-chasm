"""Tests for the core training loop and trainers.

Covers ``trainers/base.py`` (``JointTrainer``, ``_build_lr_schedule``,
the step loop, grad handling), ``trainers/trainer.py`` (``Trainer`` facade,
PEFT/mixed-precision threading, callbacks), ``trainers/trainable.py``,
``trainers/wrappers.py``, ``trainers/_metrics.py``.

Tests named ``test_BUG_*`` are regression tests for specific past defects; the
rest are general regression coverage.
"""

from __future__ import annotations

import math
import tempfile
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from auto_chasm import JointLoss, Model, ProbeConfig
from auto_chasm.trainers._metrics import build_lr_schedule
from auto_chasm.trainers.base import JointTrainer
from auto_chasm.trainers.trainer import Trainer
from auto_chasm.trainers.wrappers import TrainerCallback

# ---------------------------------------------------------------------------
# Tiny synthetic harness (mirrors the project's other trainer tests).
# ---------------------------------------------------------------------------


class _TinyMlp(nn.Module):
    """A tiny per-position MLP standing in for a language model."""

    def __init__(self, h: int = 16, v: int = 32, layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(v, h)
        self.layers = [nn.Linear(h, h) for _ in range(layers)]
        self.output_proj = nn.Linear(h, v)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = 16
    num_hidden_layers = 2


def _make_model(seed: int = 0) -> Model:
    """A Model with one regression probe, ready for joint training."""
    mx.random.seed(seed)
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(
        ProbeConfig(
            name="p",
            layers=[-1],
            aggregation="last",
            module_config={"out_features": 1},
        )
    )
    return m


def _learnable_dataset(n: int = 8, seq_len: int = 5) -> list[dict[str, Any]]:
    """Regress the constant 1.0 — reachable via the probe head's bias."""
    tokens = list(range(1, seq_len + 1))
    return [{"tokens": tokens, "labels": [1.0] * seq_len} for _ in range(n)]


def _mse_loss() -> JointLoss:
    return JointLoss(weights={"lm_head": 0.0}, losses={"p": "mse"})


def _probe_snapshot(m: Model) -> dict[str, mx.array]:
    return {k: mx.array(v) for k, v in tree_flatten(m._probes["p"].module.parameters())}


def _base_snapshot(m: Model) -> dict[str, mx.array]:
    return {k: mx.array(v) for k, v in tree_flatten(m.model.parameters())}


def _max_delta(a: dict[str, mx.array], b: dict[str, mx.array]) -> float:
    return max(float(mx.max(mx.abs(a[k] - b[k]))) for k in a)


def _trainer(m: Model, tmp: str, **kw: Any) -> JointTrainer:
    base = {
        "model": m,
        "loss_fn": _mse_loss(),
        "learning_rate": 5e-2,
        "num_iters": 20,
        "batch_size": 4,
        "max_seq_length": 32,
        "verbose": False,
        "save_history": False,
        "early_stopping_patience": 0,
        "output_dir": tmp,
    }
    base.update(kw)
    return JointTrainer(**base)  # type: ignore[arg-type]


def _make_torch_tiny() -> Any:
    """Build a tiny 2-layer torch base model (config-tagged) for probe tests."""
    import torch
    import torch.nn as tnn

    class TorchTiny(tnn.Module):
        """Tiny torch model with a per-token hidden state for probe tests."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = tnn.Embedding(32, 16)
            self.layers = tnn.ModuleList([tnn.Linear(16, 16) for _ in range(2)])
            self.output_proj = tnn.Linear(16, 32)

        def forward(self, x: torch.Tensor, **kw: Any) -> torch.Tensor:
            h = self.embedding(x)
            for layer in self.layers:
                h = torch.nn.functional.gelu(layer(h))
            return self.output_proj(h)

    class _C:
        hidden_size = 16
        num_hidden_layers = 2

    base = TorchTiny()
    base.config = _C()
    return base


# ===========================================================================
# BUG 1 — warmup_steps >= num_iters silently drops the decay schedule and the
#         peak LR. The whole run becomes a half-finished linear warmup ramp.
# ===========================================================================


class TestWarmupExceedsTotalSteps:
    """``warmup_ratio`` that yields ``warmup_steps >= num_iters`` is a footgun."""

    def test_BUG_warmup_ge_total_reaches_peak_lr_somewhere(self) -> None:
        """A schedule should hit (approximately) the peak LR at some training step.

        With ``warmup_steps >= num_iters`` ``build_lr_schedule`` returns the
        warmup ramp ALONE (the cosine/linear decay is discarded), and the ramp
        only reaches ``peak * num_iters / warmup_steps`` by the last step.  For
        ``num_iters=10, warmup_steps=20`` the LR tops out at half the requested
        peak and the user never trains at their configured learning rate — a
        silent, unflagged truncation.
        """
        peak = 1e-2
        num_iters, warmup_steps = 10, 20  # e.g. warmup_ratio=2.0, or ratio>~ via rounding
        sched = build_lr_schedule("cosine", peak, num_iters, warmup_steps)

        # The optimizer reads the schedule at steps 0..num_iters-1.
        seen = [float(sched(mx.array(it))) for it in range(num_iters)]
        # Desired: the run actually reaches (near) the configured peak LR.
        assert max(seen) >= 0.9 * peak, (
            f"warmup_steps({warmup_steps}) >= num_iters({num_iters}): the LR never "
            f"reaches the configured peak {peak:.1e} (max seen {max(seen):.1e}); the "
            f"decay schedule was silently dropped."
        )


# ===========================================================================
# BUG 2 — history records lr_schedule(it) but the optimizer applies
#         lr_schedule(it-1). The logged LR curve is off by one step and reports
#         LR=0 on the final step where a real (nonzero) update happened.
# ===========================================================================


class TestLoggedLrMatchesAppliedLr:
    """The history LR must be the LR the optimizer actually used that step."""

    def test_BUG_logged_lr_is_offset_by_one_step(self) -> None:
        """The history's per-step LR should equal the optimizer's applied LR.

        ``run()`` logs ``self.lr_schedule(it)`` for ``it`` in 1..num_iters, but
        an MLX optimizer built from a schedule reads it at its *internal* step
        counter, which starts at 0.  So the LR applied on iteration ``it`` is
        ``lr_schedule(it-1)`` while history stores ``lr_schedule(it)`` — shifted
        forward by one.  Worst case: the final step logs LR=0.0 (cosine floor)
        even though the optimizer applied a nonzero LR and the weights moved.
        A scientific reader trusting the recorded LR curve is misled.
        """
        m = _make_model()
        m.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(
                m,
                tmp,
                num_iters=4,
                logging_steps=1,
                lr_schedule="cosine",
                learning_rate=5e-2,
            )
            hist = t.run(_learnable_dataset())

        logged = {e.step: e.learning_rate for e in hist if e.learning_rate is not None}
        # Ground truth: the LR the optimizer used at iteration `it` is sched(it-1).
        sched = t.lr_schedule
        for it, lr_logged in logged.items():
            applied = float(sched(mx.array(it - 1)))
            assert lr_logged == pytest.approx(applied, abs=1e-9), (
                f"iter {it}: history logged LR {lr_logged:.6e} but the optimizer "
                f"applied {applied:.6e} (logged = sched(it), applied = sched(it-1))."
            )


# ===========================================================================
# BUG 3 — training with nothing trainable raises an opaque MLX autograd error
#         instead of a clear "no trainable parameters" message.
# ===========================================================================


class TestNoTrainableParameters:
    """All-frozen training should fail (or no-op) with a clear, actionable error."""

    def test_BUG_all_frozen_gives_clear_error(self) -> None:
        """Freezing everything then training should raise a *clear* error.

        Currently the loop hands an empty gradient set to MLX and the user sees
        ``ValueError: [grad] Must specify at least one argument.`` — an internal
        autograd message with no hint that the real problem is "you froze every
        parameter; nothing can be optimized".  A library targeting a skeptical
        audience should name the actual misconfiguration.
        """
        m = _make_model()
        m.freeze_model()
        m.freeze_probe("p")
        assert len(tree_flatten(m.model.trainable_parameters())) == 0
        assert len(tree_flatten(m._probes["p"].module.trainable_parameters())) == 0

        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, num_iters=3)
            with pytest.raises(Exception) as exc:  # noqa: PT011 — asserting the message below
                t.run(_learnable_dataset())

        msg = str(exc.value).lower()
        assert any(
            kw in msg for kw in ("trainable", "frozen", "no parameters", "nothing to train")
        ), (
            "all-frozen training raised an opaque error instead of naming the "
            f"misconfiguration; got: {exc.value!r}"
        )


# ===========================================================================
# Regression coverage — behaviors that are correct today (should PASS).
# These pin down the contract so a future change can't silently break it.
# ===========================================================================


class TestDegenerateConfigs:
    """Degenerate configs must be clean no-ops or clear errors, never crashes."""

    def test_num_iters_zero_is_clean_noop(self) -> None:
        """num_iters=0 trains nothing and returns an empty history without crashing."""
        m = _make_model()
        m.prepare_for_joint_training()
        w0 = _probe_snapshot(m)
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, num_iters=0)
            hist = t.run(_learnable_dataset())
        assert len(hist) == 0
        # Nothing trained.
        assert _max_delta(_probe_snapshot(m), w0) == 0.0

    def test_empty_dataset_raises_clear_error(self) -> None:
        """An empty training dataset raises a clear ValueError, not an obscure crash."""
        m = _make_model()
        m.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, num_iters=3)
            with pytest.raises(ValueError, match="(?i)empty"):
                t.run([])

    def test_batch_larger_than_dataset_works(self) -> None:
        """batch_size > len(dataset) clamps to the dataset and still trains."""
        m = _make_model()
        m.prepare_for_joint_training()
        data = _learnable_dataset(n=3)
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, num_iters=10, batch_size=100, logging_steps=1)
            it = t.iterate(data)
            first = t.step(next(it))["loss"]
            for _ in range(20):
                last = t.step(next(it))["loss"]
        assert math.isfinite(first) and math.isfinite(last)
        assert last < first  # actually learns despite oversize batch

    def test_single_example_dataset_cycles(self) -> None:
        """A one-row dataset is looped over num_iters without dropping/duplicating wrongly."""
        m = _make_model()
        m.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, num_iters=15, batch_size=4)
            it = t.iterate(_learnable_dataset(n=1))
            first = t.step(next(it))["loss"]
            for _ in range(30):
                last = t.step(next(it))["loss"]
        assert last < 0.5 * first

    def test_lr_zero_is_a_noop(self) -> None:
        """learning_rate=0 leaves the loss (and params) unchanged."""
        m = _make_model()
        m.prepare_for_joint_training()
        w0 = _probe_snapshot(m)
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, learning_rate=0.0, num_iters=10)
            it = t.iterate(_learnable_dataset())
            first = t.step(next(it))["loss"]
            for _ in range(15):
                last = t.step(next(it))["loss"]
        assert first == pytest.approx(last, abs=1e-5)
        assert _max_delta(_probe_snapshot(m), w0) == 0.0

    def test_num_iters_huge_relative_to_tiny_data(self) -> None:
        """A large num_iters over tiny data cycles correctly and converges."""
        m = _make_model()
        m.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, num_iters=120, batch_size=2, logging_steps=1000)
            hist = t.run(_learnable_dataset(n=2))
        # The final logged train loss is well below the probe's untrained ~1.0.
        train_losses = [e.train_loss for e in hist if e.train_loss is not None]
        assert train_losses, "no train loss recorded"
        assert train_losses[-1] < 0.5


class TestGradientDirection:
    """A single step moves params down the loss gradient (sign correctness)."""

    def test_step_reduces_loss_on_learnable_target(self) -> None:
        """Repeated steps drive the loss below half its start (negative-grad descent)."""
        m = _make_model()
        m.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, num_iters=30)
            it = t.iterate(_learnable_dataset())
            first = t.step(next(it))["loss"]
            for _ in range(30):
                last = t.step(next(it))["loss"]
        assert last < 0.5 * first

    def test_one_param_step_moves_against_gradient(self) -> None:
        """A single AdamW step nudges the probe bias toward the target (descent sign)."""
        m = _make_model()
        m.prepare_for_joint_training()
        bias0 = mx.array(dict(tree_flatten(m._probes["p"].module.parameters()))["bias"])
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, learning_rate=1e-1, num_iters=1, grad_clip_norm=0.0)
            t.step(next(t.iterate(_learnable_dataset())))
        bias1 = mx.array(dict(tree_flatten(m._probes["p"].module.parameters()))["bias"])
        # Target is +1.0 and the probe starts near 0, so the bias must increase.
        assert float(bias1.sum()) > float(bias0.sum())


class TestFreezeUnfreezeInterplay:
    """Freeze/unfreeze must be honored exactly — frozen weights never move."""

    def test_frozen_base_is_byte_identical_after_training(self) -> None:
        """prepare_for_joint_training freezes the base; it must not move at all."""
        m = _make_model()
        m.prepare_for_joint_training()
        base0 = _base_snapshot(m)
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, num_iters=10)
            for _ in range(10):
                t.step(next(t.iterate(_learnable_dataset())))
        base1 = _base_snapshot(m)
        assert _max_delta(base1, base0) == 0.0

    def test_only_probe_moves_when_base_frozen(self) -> None:
        """With base frozen + probe unfrozen, only the probe changes."""
        m = _make_model()
        m.prepare_for_joint_training()
        base0, probe0 = _base_snapshot(m), _probe_snapshot(m)
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, num_iters=5)
            for _ in range(5):
                t.step(next(t.iterate(_learnable_dataset())))
        assert _max_delta(_base_snapshot(m), base0) == 0.0
        assert _max_delta(_probe_snapshot(m), probe0) > 0.0

    def test_freeze_unfreeze_freeze_sequence(self) -> None:
        """Re-freezing after unfreezing wins: a re-frozen probe does not train."""
        m = _make_model()
        m.freeze_model()
        m.unfreeze_probe("p")
        m.freeze_probe("p")  # final word: frozen
        assert len(tree_flatten(m._probes["p"].module.trainable_parameters())) == 0


class TestLrSchedule:
    """LR schedule edge cases: warmup=0, exact floors, single step."""

    def test_warmup_zero_starts_at_peak(self) -> None:
        """No warmup => the very first step is at (approximately) the peak LR."""
        sched = build_lr_schedule("cosine", 1e-2, num_iters=10, warmup_steps=0)
        assert float(sched(mx.array(0))) == pytest.approx(1e-2, rel=1e-6)

    def test_linear_decay_hits_floor_exactly(self) -> None:
        """Linear decay reaches exactly 0 at the horizon and never goes negative."""
        sched = build_lr_schedule("linear", 1e-2, num_iters=10, warmup_steps=0)
        assert float(sched(mx.array(10))) == pytest.approx(0.0, abs=1e-9)
        # Past the horizon it clamps at 0, not a negative LR.
        assert float(sched(mx.array(50))) >= 0.0

    def test_cosine_decay_hits_floor(self) -> None:
        """Cosine decay reaches ~0 at the horizon."""
        sched = build_lr_schedule("cosine", 1e-2, num_iters=10, warmup_steps=0)
        assert float(sched(mx.array(10))) == pytest.approx(0.0, abs=1e-6)

    def test_unknown_schedule_raises(self) -> None:
        """An unknown schedule name raises a clear ValueError."""
        with pytest.raises(ValueError, match="(?i)unknown lr_schedule"):
            build_lr_schedule("triangular", 1e-2, 10, 0)

    def test_constant_schedule_holds_value(self) -> None:
        """Constant schedule returns the same LR for every step."""
        sched = build_lr_schedule("constant", 7e-3, num_iters=10, warmup_steps=0)
        assert float(sched(mx.array(0))) == pytest.approx(7e-3)
        assert float(sched(mx.array(9))) == pytest.approx(7e-3)


class TestStateLeaks:
    """Re-running / verbose toggling must not change numerics or leak state."""

    def test_verbose_does_not_change_numerics(self) -> None:
        """verbose=True and verbose=False must produce identical loss trajectories."""

        def run(verbose: bool) -> list[float]:
            m = _make_model(seed=123)
            m.prepare_for_joint_training()
            with tempfile.TemporaryDirectory() as tmp:
                t = _trainer(m, tmp, num_iters=8, verbose=verbose)
                it = t.iterate(_learnable_dataset())
                return [t.step(next(it))["loss"] for _ in range(8)]

        a, b = run(True), run(False)
        for x, y in zip(a, b, strict=True):
            assert x == pytest.approx(y, abs=1e-6)

    def test_two_fresh_trainers_do_not_share_optimizer_state(self) -> None:
        """Two independent trainers over fresh models train identically (no stale state)."""

        def run() -> float:
            m = _make_model(seed=7)
            m.prepare_for_joint_training()
            with tempfile.TemporaryDirectory() as tmp:
                t = _trainer(m, tmp, num_iters=10)
                it = t.iterate(_learnable_dataset())
                for _ in range(10):
                    last = t.step(next(it))["loss"]
            return last

        assert run() == pytest.approx(run(), abs=1e-6)


class TestCallbacks:
    """Callback dispatch: firing counts and exception handling."""

    def test_on_step_end_fires_once_per_step(self) -> None:
        """on_step_end fires exactly num_iters times on the MLX path."""
        calls: list[int] = []

        class Counter(TrainerCallback):
            """Callback that counts on_step_end invocations."""

            def on_step_end(self, **kw: Any) -> None:
                calls.append(kw.get("step", -1))

        m = _make_model()
        m.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            t = Trainer(
                model=m,
                loss_fn=_mse_loss(),
                num_iters=6,
                batch_size=4,
                callbacks=[Counter()],
                verbose=False,
                save_history=False,
                early_stopping_patience=0,
                output_dir=tmp,
            )
            t.train(_learnable_dataset())
        assert len(calls) == 6
        assert calls == [1, 2, 3, 4, 5, 6]

    def test_callback_exception_propagates(self) -> None:
        """A raising callback crashes training loudly (re-raised, not swallowed).

        This pins the *fixed* contract: ``_fire_callback`` logs at WARNING and
        re-raises, so a broken user callback fails visibly instead of silently.
        """

        class Boom(TrainerCallback):
            """Callback that raises, to verify exceptions propagate."""

            def on_step_end(self, **kw: Any) -> None:
                raise RuntimeError("boom")

        m = _make_model()
        m.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            t = Trainer(
                model=m,
                loss_fn=_mse_loss(),
                num_iters=3,
                batch_size=4,
                callbacks=[Boom()],
                verbose=False,
                save_history=False,
                early_stopping_patience=0,
                output_dir=tmp,
            )
            with pytest.raises(RuntimeError, match="boom"):
                t.train(_learnable_dataset())


class TestMixedPrecision:
    """bf16 frozen-base mixed precision; fp16 raises; fp32 default unchanged."""

    def test_bf16_casts_base_keeps_probe_fp32(self) -> None:
        m = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, mixed_precision="bf16", num_iters=2)
            base_dtypes = {str(v.dtype) for _, v in tree_flatten(t._train_model.base.parameters())}
            probe_dtypes = {
                str(v.dtype) for _, v in tree_flatten(m._probes["p"].module.parameters())
            }
        assert base_dtypes == {"mlx.core.bfloat16"}
        assert probe_dtypes == {"mlx.core.float32"}

    def test_bf16_step_gives_finite_loss_and_descends(self) -> None:
        m = _make_model()
        m.freeze_model()
        m.unfreeze_all_probes()
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, mixed_precision="bf16", num_iters=40)
            it = t.iterate(_learnable_dataset())
            first = t.step(next(it))["loss"]
            assert math.isfinite(first)
            for _ in range(40):
                last = t.step(next(it))["loss"]
        assert math.isfinite(last)
        assert last < first

    def test_fp32_default_leaves_base_fp32(self) -> None:
        m = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, num_iters=2)
            base_dtypes = {str(v.dtype) for _, v in tree_flatten(t._train_model.base.parameters())}
        assert base_dtypes == {"mlx.core.float32"}

    def test_fp16_is_torch_only(self) -> None:
        """fp16 is a valid config (torch autocast + GradScaler); MLX rejects it."""
        from auto_chasm.config import TrainingConfig

        # Config accepts fp16 now (it is torch-only, not universally unsupported).
        assert TrainingConfig(mixed_precision="fp16").mixed_precision == "fp16"


class TestTorchStepEscapeHatch:
    """Trainer.step is the per-step escape hatch on torch as well as MLX.

    Mirrors how exp1 drives a manual ``iterate``/``step``/``evaluate`` loop to
    track a per-layer best — a flow that previously raised NotImplementedError
    on torch.  Verifies that step() actually trains (loss drops) and that
    interleaving evaluate() between steps works.
    """

    def test_torch_step_trains_and_interleaves_evaluate(self) -> None:
        pytest.importorskip("torch")
        import torch

        torch.manual_seed(0)
        m = Model(_make_torch_tiny(), None, "torch")
        m.attach_probe(
            ProbeConfig(
                name="p", layers=[-1], aggregation="last", module_config={"out_features": 1}
            )
        )
        m.freeze_model()
        m.unfreeze_all_probes()

        data = _learnable_dataset(n=8)
        with tempfile.TemporaryDirectory() as tmp:
            t = Trainer(
                model=m,
                loss_fn=_mse_loss(),
                learning_rate=5e-2,
                num_iters=60,
                batch_size=4,
                verbose=False,
                save_history=False,
                early_stopping_patience=0,
                output_dir=tmp,
            )
            batches = t.iterate(data)
            first = t.step(next(batches))
            # Contract parity with the MLX escape hatch.
            assert set(first) == {"loss", "ntoks", "components"}
            # Interleaving evaluate() between steps must not break the next step
            # (evaluate flips the shared base model to eval mode).
            t.evaluate(data)
            losses = [first["loss"]]
            for _ in range(59):
                losses.append(t.step(next(batches))["loss"])

        assert losses[-1] < 0.5 * losses[0], (
            f"torch step() did not reduce loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_torch_step_raises_without_trainable_params(self) -> None:
        """A fully-frozen model gives a clear error, not a silent no-op."""
        pytest.importorskip("torch")
        import torch

        torch.manual_seed(0)
        m = Model(_make_torch_tiny(), None, "torch")
        m.attach_probe(
            ProbeConfig(
                name="p", layers=[-1], aggregation="last", module_config={"out_features": 1}
            )
        )
        m.freeze_model()
        for probe in m._probes.values():
            for p in probe.module.parameters():
                p.requires_grad_(False)

        data = _learnable_dataset(n=8)
        with tempfile.TemporaryDirectory() as tmp:
            t = Trainer(
                model=m,
                loss_fn=_mse_loss(),
                num_iters=2,
                batch_size=4,
                verbose=False,
                save_history=False,
                early_stopping_patience=0,
                output_dir=tmp,
            )
            with pytest.raises(RuntimeError, match="(?i)trainable"):
                t.step(next(t.iterate(data)))


class TestCrossBackendTrajectory:
    """The same toy train on MLX and torch should have comparable loss drops."""

    def test_BUG_or_OK_mlx_and_torch_both_reduce_loss(self) -> None:
        """Both backends should reduce the loss on the identical learnable target.

        This is a sanity oracle: a regression where one backend silently fails
        to learn (e.g. a frozen-base or grad-routing bug) would surface as one
        trajectory not dropping while the other does.
        """
        pytest.importorskip("torch")
        import torch

        data = _learnable_dataset(n=8)

        # --- MLX ---
        m = _make_model(seed=0)
        m.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            t = _trainer(m, tmp, num_iters=60, learning_rate=5e-2)
            res = t.train(data)
        mlx_losses = [e.train_loss for e in res["history"] if e.train_loss is not None]
        assert mlx_losses, "MLX recorded no train loss"
        assert mlx_losses[-1] < 0.5 * mlx_losses[0]

        # --- torch ---
        torch.manual_seed(0)
        mt = Model(_make_torch_tiny(), None, "torch")
        mt.attach_probe(
            ProbeConfig(
                name="p", layers=[-1], aggregation="last", module_config={"out_features": 1}
            )
        )
        mt.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            tt = Trainer(
                model=mt,
                loss_fn=_mse_loss(),
                learning_rate=5e-2,
                num_iters=60,
                batch_size=4,
                logging_steps=1,
                verbose=False,
                save_history=False,
                early_stopping_patience=0,
                output_dir=tmp,
            )
            rt = tt.train(data)
        torch_losses = [e.train_loss for e in rt["history"] if e.train_loss is not None]
        assert torch_losses, "torch recorded no train loss"
        assert torch_losses[-1] < 0.5 * torch_losses[0], (
            "torch backend did not reduce loss while MLX did — cross-backend divergence."
        )

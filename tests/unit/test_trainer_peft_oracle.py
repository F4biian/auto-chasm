"""Oracle tests for PEFT (LoRA / full fine-tune) and the SFT / Joint trainers.

Each test asserts a *checkable* property against independent ground truth — not
merely "it runs":

- **LoRA**: applying LoRA to ``q_proj`` wraps exactly as many ``LoRALinear``
  modules as there are ``q_proj`` modules (counted independently); with a frozen
  base, ONLY adapter params are trainable (counted: 2 arrays per wrapped module,
  zero non-adapter); and the zero-initialized B matrix leaves the forward pass
  bit-identical immediately after wrapping.
- **full fine-tune (``none``)**: with no adapters and an unfrozen base, the base
  weights actually receive gradients and change after one optimizer step (and the
  base contributes trainable params), in contrast to the frozen LoRA path.
- **Trainer: SFT**: training a tiny model on a trivially-learnable synthetic
  target (regress a constant) DECREASES the loss to < 0.5x its initial value,
  i.e. it actually optimizes; ``.train()`` returns ``{"history", "output_dir"}``.
- **Joint trainer escape hatch**: ``iterate`` / ``step`` / ``evaluate`` produce
  the documented shapes/keys, ``step`` returns a finite loss dict, and a step
  reduces loss on a trivially-learnable batch.
- **early-stopping metric resolution**: ``resolve_early_stopping_metric`` returns
  the right value when the metric is present and RAISES (``KeyError``) when the
  requested metric is absent — there is deliberately no silent loss fallback.

The LoRA-on-a-real-model tests use the cached MLX checkpoint
``mlx-community/gemma-3-270m-it-8bit`` and SKIP cleanly when it (or ``mlx_lm``)
is unavailable.  Everything else uses a tiny synthetic MLX model.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from auto_chasm import JointLoss, Model, ProbeConfig
from auto_chasm.peft import apply_lora
from auto_chasm.trainers._metrics import resolve_early_stopping_metric
from auto_chasm.trainers.base import JointTrainer

# ---------------------------------------------------------------------------
# Tiny synthetic harness (mirrors test_classification_regression._TinyMlp and
# test_trainer_coverage.TinyMlp)
# ---------------------------------------------------------------------------


class _TinyMlp(nn.Module):
    """A tiny per-position MLP used as a stand-in language model."""

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
    """Mock model config exposing hidden_size / num_hidden_layers."""

    hidden_size = 16
    num_hidden_layers = 2


def _make_model() -> Model:
    """Build a Model wrapping a fresh _TinyMlp with one regression probe."""
    mx.random.seed(0)
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


def _learnable_dataset(n: int = 8, seq_len: int = 6) -> list[dict[str, Any]]:
    """A trivially-learnable target: regress the constant 1.0 over the whole region.

    The probe head is a Linear reading hidden states, so outputting a constant
    is reachable via its bias; MSE to this target must drive the loss toward 0.
    """
    tokens = list(range(1, seq_len + 1))
    return [{"tokens": tokens, "labels": [1.0] * seq_len} for _ in range(n)]


def _base_weight(model: Model) -> mx.array:
    """Snapshot the base model's output_proj weight (a representative base param)."""
    flat = dict(tree_flatten(model.model.parameters()))
    return mx.array(flat["output_proj.weight"])


def _load_cached_gemma() -> Any:
    """Load the cached 8-bit gemma MLX model, or skip if unavailable (offline)."""
    mlx_lm = pytest.importorskip("mlx_lm")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        model, _ = mlx_lm.load("mlx-community/gemma-3-270m-it-8bit")
    except Exception:
        pytest.skip("cached MLX model 'mlx-community/gemma-3-270m-it-8bit' unavailable")
    return model


# ---------------------------------------------------------------------------
# PEFT: LoRA
# ---------------------------------------------------------------------------


@pytest.mark.real_model
class TestLoraWrapsExactCount:
    """LoRA wraps exactly one LoRALinear per matched target module."""

    def test_qproj_lora_count_equals_qproj_module_count(self) -> None:
        """apply_lora(target=['q_proj']) wraps exactly as many LoRALinear as q_proj."""
        from mlx_lm.tuner.lora import LoRALinear

        from auto_chasm.backends import Backend

        model = _load_cached_gemma()

        # Independent ground truth: count q_proj modules BEFORE wrapping.
        n_qproj = sum(1 for n, _ in model.named_modules() if n.endswith("q_proj"))
        assert n_qproj > 0

        apply_lora(model, r=4, alpha=8, target_modules=["q_proj"], backend=Backend(force="mlx"))

        n_lora = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALinear))
        assert n_lora == n_qproj

    def test_layer_targeting_wraps_only_requested_layers(self) -> None:
        """target_layers=[0, 1] wraps exactly two LoRALinear modules."""
        from mlx_lm.tuner.lora import LoRALinear

        from auto_chasm.backends import Backend

        model = _load_cached_gemma()
        apply_lora(
            model,
            r=4,
            alpha=8,
            target_modules=["q_proj"],
            target_layers=[0, 1],
            backend=Backend(force="mlx"),
        )
        n_lora = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALinear))
        assert n_lora == 2


@pytest.mark.real_model
class TestLoraFrozenBaseOnlyAdaptersTrainable:
    """With a frozen base, only LoRA adapter params are trainable."""

    def test_only_adapter_params_trainable(self) -> None:
        """Frozen base + LoRA: trainable = 2 arrays/module (lora_a, lora_b), no base."""
        from mlx_lm.tuner.lora import LoRALinear

        from auto_chasm.backends import Backend

        model = _load_cached_gemma()
        model.freeze()  # freeze the entire base first

        n_qproj = sum(1 for n, _ in model.named_modules() if n.endswith("q_proj"))
        apply_lora(model, r=4, alpha=8, target_modules=["q_proj"], backend=Backend(force="mlx"))

        n_lora = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALinear))
        assert n_lora == n_qproj

        names = [k for k, _ in tree_flatten(model.trainable_parameters())]
        # Every wrapped LoRALinear contributes exactly lora_a + lora_b.
        assert len(names) == 2 * n_lora
        # And NOTHING outside the adapters is trainable.
        non_adapter = [k for k in names if "lora_" not in k]
        assert non_adapter == []


@pytest.mark.real_model
class TestLoraZeroInitForwardUnchanged:
    """Zero-initialized LoRA B leaves the forward pass bit-identical at init."""

    def test_forward_unchanged_immediately_after_lora(self) -> None:
        """Right after apply_lora, the model output is identical (B init = 0)."""
        from auto_chasm.backends import Backend

        model = _load_cached_gemma()
        toks = mx.array([[1, 2, 3, 4, 5]])

        out_before = model(toks)
        mx.eval(out_before)
        out_before = mx.array(out_before)

        apply_lora(model, r=4, alpha=8, target_modules=["q_proj"], backend=Backend(force="mlx"))

        out_after = model(toks)
        mx.eval(out_after)

        # Zero-init B => adapter contributes nothing yet => exact match.
        assert float(mx.max(mx.abs(out_after - out_before))) == 0.0


# ---------------------------------------------------------------------------
# PEFT: full fine-tune (peft="none")
# ---------------------------------------------------------------------------


class TestFullFineTuneTouchesBaseWeights:
    """Full fine-tune (no adapters) lets the base weights receive gradients."""

    def test_base_weights_change_after_step(self) -> None:
        """With an unfrozen base and no adapters, a base weight changes after one step."""
        m = _make_model()
        # Full fine-tune semantics: no adapters, base left trainable.
        m.unfreeze_model()
        m.unfreeze_all_probes()

        # The base must actually contribute trainable parameters.
        assert len(tree_flatten(m.model.trainable_parameters())) > 0

        w0 = _base_weight(m)
        loss = JointLoss(losses={"p": "mse"})
        trainer = JointTrainer(
            model=m,
            loss_fn=loss,
            learning_rate=1e-2,
            num_iters=5,
            batch_size=4,
            max_seq_length=32,
            verbose=False,
            save_history=False,
            early_stopping_patience=0,
        )
        trainer.step(next(trainer.iterate(_learnable_dataset())))
        w1 = _base_weight(m)

        # The base weight genuinely moved (gradients flowed into the base).
        assert float(mx.max(mx.abs(w1 - w0))) > 0.0

    def test_frozen_base_does_not_change_after_step(self) -> None:
        """Contrast: a frozen base (prepare_for_joint_training) stays put after a step."""
        m = _make_model()
        m.prepare_for_joint_training()  # freezes base, unfreezes probes

        # No adapters were applied, so with the base frozen nothing in the base trains.
        assert len(tree_flatten(m.model.trainable_parameters())) == 0

        w0 = _base_weight(m)
        loss = JointLoss(losses={"p": "mse"})
        trainer = JointTrainer(
            model=m,
            loss_fn=loss,
            learning_rate=1e-2,
            num_iters=5,
            batch_size=4,
            max_seq_length=32,
            verbose=False,
            save_history=False,
            early_stopping_patience=0,
        )
        trainer.step(next(trainer.iterate(_learnable_dataset())))
        w1 = _base_weight(m)

        assert float(mx.max(mx.abs(w1 - w0))) == 0.0


# ---------------------------------------------------------------------------
# Trainer: SFT (actually optimizes)
# ---------------------------------------------------------------------------


class TestSftActuallyOptimizes:
    """SFT training reduces the loss on a learnable target and returns a history."""

    def test_loss_decreases_below_half(self) -> None:
        """Final step loss is < 0.5x the initial step loss on a learnable target."""
        from auto_chasm.trainers.sft import SFTTrainer

        m = _make_model()
        m.prepare_for_joint_training()

        with tempfile.TemporaryDirectory() as tmp:
            sft = SFTTrainer(
                model=m,
                lm_weight=0.0,
                probe_weight=1.0,
                probe_loss="mse",
                learning_rate=5e-2,
                num_iters=60,
                batch_size=4,
                max_seq_length=32,
                early_stopping_patience=0,
                save_steps=0,
                output_dir=tmp,
                verbose=False,
            )
            # Use the underlying JointTrainer's step API for a clean, monotone
            # initial-vs-final comparison on a known-learnable batch stream.
            joint = sft._trainer
            it = joint.iterate(_learnable_dataset())
            initial = joint.step(next(it))["loss"]
            assert initial == initial  # finite (not NaN)
            for _ in range(60):
                final = joint.step(next(it))["loss"]

        assert final < 0.5 * initial

    def test_train_returns_history_dict(self) -> None:
        """SFTTrainer.train returns a dict with a 'history' (History) and 'output_dir'."""
        from auto_chasm.history import History
        from auto_chasm.trainers.sft import SFTTrainer

        m = _make_model()
        m.prepare_for_joint_training()

        with tempfile.TemporaryDirectory() as tmp:
            sft = SFTTrainer(
                model=m,
                lm_weight=0.0,
                probe_weight=1.0,
                probe_loss="mse",
                learning_rate=5e-2,
                num_iters=10,
                batch_size=4,
                max_seq_length=32,
                early_stopping_patience=0,
                save_steps=0,
                output_dir=tmp,
                verbose=False,
            )
            result = sft.train(_learnable_dataset())

        assert "history" in result
        assert isinstance(result["history"], History)
        assert result["output_dir"]


# ---------------------------------------------------------------------------
# Trainer: Joint (escape hatch: iterate / step / evaluate)
# ---------------------------------------------------------------------------


class TestJointEscapeHatch:
    """JointTrainer.iterate/step/evaluate expose correct shapes/keys and optimize."""

    def _trainer(self, m: Model, tmp: str) -> JointTrainer:
        """Build a JointTrainer over a learnable MSE target."""
        return JointTrainer(
            model=m,
            loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"p": "mse"}),
            learning_rate=5e-2,
            num_iters=30,
            batch_size=4,
            max_seq_length=32,
            verbose=False,
            save_history=False,
            early_stopping_patience=0,
            output_dir=tmp,
        )

    def test_iterate_yields_three_tuple(self) -> None:
        """``iterate`` yields (tokens, labels, lengths) with batch-aligned leading dim."""
        m = _make_model()
        m.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._trainer(m, tmp)
            tokens, labels, lengths = next(trainer.iterate(_learnable_dataset(n=8)))
            assert tokens.shape[0] == 4  # batch_size
            assert labels.shape[0] == 4
            assert lengths.shape[0] == 4
            assert lengths.shape[1] == 2  # [start, end) per sequence

    def test_step_returns_finite_loss_dict(self) -> None:
        """``step`` returns {'loss','ntoks','components'} with a finite loss and >0 ntoks."""
        import math

        m = _make_model()
        m.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._trainer(m, tmp)
            out = trainer.step(next(trainer.iterate(_learnable_dataset())))
        assert set(out.keys()) == {"loss", "ntoks", "components"}
        assert math.isfinite(out["loss"])
        assert out["ntoks"] > 0
        assert "p" in out["components"]
        assert math.isfinite(out["components"]["p"])

    def test_step_reduces_loss_on_learnable_batch(self) -> None:
        """Repeated steps on a learnable batch drive the loss below half its start."""
        m = _make_model()
        m.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._trainer(m, tmp)
            it = trainer.iterate(_learnable_dataset())
            initial = trainer.step(next(it))["loss"]
            for _ in range(30):
                final = trainer.step(next(it))["loss"]
        assert final < 0.5 * initial

    def test_evaluate_returns_loss_keys(self) -> None:
        """``evaluate`` returns a dict with a finite 'loss' and the probe component."""
        import math

        m = _make_model()
        m.prepare_for_joint_training()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._trainer(m, tmp)
            metrics = trainer.evaluate(_learnable_dataset())
        assert "loss" in metrics
        assert math.isfinite(metrics["loss"])
        assert "ntokens" in metrics
        assert "p" in metrics


# ---------------------------------------------------------------------------
# Trainer: early-stopping metric resolution
# ---------------------------------------------------------------------------


class TestEarlyStoppingMetricResolution:
    """resolve_early_stopping_metric returns present metrics and raises on absent ones."""

    def test_val_loss_resolves_to_loss(self) -> None:
        """'val_loss' resolves to the 'loss' entry."""
        assert resolve_early_stopping_metric("val_loss", {"loss": 0.42}) == pytest.approx(0.42)

    def test_loss_alias_resolves(self) -> None:
        """'loss' resolves directly to the 'loss' entry."""
        assert resolve_early_stopping_metric("loss", {"loss": 1.5}) == pytest.approx(1.5)

    def test_val_prefixed_custom_metric_resolves(self) -> None:
        """'val_f1' strips the 'val_' prefix and resolves to the 'f1' entry."""
        got = resolve_early_stopping_metric("val_f1", {"loss": 0.1, "f1": 0.87})
        assert got == pytest.approx(0.87)

    def test_missing_metric_raises_keyerror(self) -> None:
        """A requested metric absent from the dict raises KeyError (no silent fallback)."""
        with pytest.raises(KeyError, match="val_f1"):
            resolve_early_stopping_metric("val_f1", {"loss": 0.1})

    def test_missing_metric_does_not_fall_back_to_loss(self) -> None:
        """The absent-metric error mentions the loss alternative but does NOT return it."""
        with pytest.raises(KeyError) as exc:
            resolve_early_stopping_metric("val_accuracy", {"loss": 0.2, "f1": 0.5})
        # The message guides the user to a real fix; it must not silently use loss.
        assert "val_accuracy" in str(exc.value)

"""Oracle tests for bf16 mixed-precision training (frozen-base).

`mixed_precision="bf16"` casts the (frozen) base model to bf16 for its forward,
while the trainable probe/adapter params and the optimizer stay fp32. These
tests pin that the base really becomes bf16, the probes stay fp32, and training
still *works* (gradients flow through the bf16 base into the fp32 probe and the
loss drops) — not just "it runs".
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from auto_chasm import JointLoss, Model, ProbeConfig
from auto_chasm.config import TrainingConfig


class _TinyMlp(nn.Module):
    """Embedding -> linear blocks -> output projection."""

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
    """Minimal model config."""

    hidden_size = 16
    num_hidden_layers = 2
    vocab_size = 32


def _dtypes(module: object) -> set[str]:
    return {str(v.dtype) for _, v in tree_flatten(module.parameters())}  # type: ignore[attr-defined]


class TestMixedPrecisionConfig:
    """TrainingConfig accepts fp32/bf16/fp16 and rejects anything else."""

    def test_bf16_constructs(self) -> None:
        assert TrainingConfig(mixed_precision="bf16").mixed_precision == "bf16"

    def test_fp16_constructs(self) -> None:
        # fp16 is a valid config (torch-only); it raises later on the MLX trainer.
        assert TrainingConfig(mixed_precision="fp16").mixed_precision == "fp16"

    def test_invalid_precision_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid"):
            TrainingConfig(mixed_precision="fp8")  # type: ignore[arg-type]


class TestBf16FrozenBaseTraining:
    """bf16 casts the base but keeps probes fp32, and training still converges."""

    def _model(self) -> Model:
        m = Model(_TinyMlp(), None, "mlx")
        m.model.config = _Cfg()
        m.attach_probe(ProbeConfig(name="p", layers=[0]))
        return m

    def test_base_is_bf16_probe_is_fp32(self) -> None:
        from auto_chasm.trainers.base import JointTrainer

        m = self._model()
        trainer = JointTrainer(
            model=m,
            loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"p": "bce"}),
            mixed_precision="bf16",
        )
        # The base was cast to bf16; the trainable probe head stays fp32.
        assert _dtypes(trainer._train_model.base) == {"mlx.core.bfloat16"}
        assert _dtypes(m._probes["p"].module) == {"mlx.core.float32"}

    def test_bf16_training_reduces_loss(self) -> None:
        from auto_chasm.trainers.base import JointTrainer

        m = self._model()
        data = [{"tokens": [1, 2, 3, 4, 5], "labels": [1, 1, 1, 1, 1]} for _ in range(8)]
        trainer = JointTrainer(
            model=m,
            loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"p": "bce"}),
            mixed_precision="bf16",
            num_iters=50,
            batch_size=4,
            learning_rate=5e-2,
            verbose=False,
        )
        batches = trainer.iterate(data)
        first = trainer.step(next(batches))["loss"]
        for _ in range(40):
            trainer.step(next(batches))
        last = trainer.step(next(batches))["loss"]
        # Gradients flow through the bf16 base into the fp32 probe: the loss drops.
        assert last < 0.5 * first, f"bf16 training did not reduce loss: {first} -> {last}"

    def test_fp32_default_leaves_base_fp32(self) -> None:
        from auto_chasm.trainers.base import JointTrainer

        m = self._model()
        trainer = JointTrainer(
            model=m, loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"p": "bce"})
        )
        assert _dtypes(trainer._train_model.base) == {"mlx.core.float32"}

    def test_fp16_raises_on_mlx_trainer(self) -> None:
        """fp16 is torch-only: the MLX trainer refuses it (no GradScaler on MLX)."""
        from auto_chasm.trainers.base import JointTrainer

        m = self._model()
        with pytest.raises(NotImplementedError, match="fp16"):
            JointTrainer(
                model=m,
                loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"p": "bce"}),
                mixed_precision="fp16",
            )

    def test_bf16_via_training_config(self) -> None:
        from auto_chasm.trainers.base import JointTrainer

        m = self._model()
        trainer = JointTrainer(
            model=m,
            loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"p": "bce"}),
            config=TrainingConfig(mixed_precision="bf16"),
        )
        assert _dtypes(trainer._train_model.base) == {"mlx.core.bfloat16"}

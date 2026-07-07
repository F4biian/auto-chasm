"""Backward-compat: the pre-refactor exp1/exp2 usage patterns still work.

The DX refactor (ModuleSpec, Dataset, class_weights, metrics, LayerSweep) is
additive.  This locks in that the *old* hand-written patterns keep functioning:
a callable ``module_type`` head that ignores ``cfg``, a custom-callable
``probe_loss``, the ``Trainer.iterate``/``step``/``evaluate`` escape-hatch loop,
and ``build_dataset`` output shape.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from auto_chasm import JointLoss, Model, ProbeConfig, Trainer
from auto_chasm.data import build_dataset


class _TinyMlp(nn.Module):
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


def _make_mlp(backend: str):
    """The exact exp1 pattern: a closure that branches on backend, ignores cfg."""

    def build(in_features, _cfg):
        import mlx.nn as _nn

        class _MLP(_nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = _nn.Linear(in_features, 3)

            def __call__(self, x):
                return self.lin(x)

        return _MLP()

    return build


def test_callable_head_custom_loss_and_escape_hatch_loop() -> None:
    """exp1's escape-hatch loop with a callable head + custom weighted CE still runs."""
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    m.add_probes(
        [ProbeConfig(name=f"L{i}", layers=[i], module_type=_make_mlp("mlx")) for i in range(2)]
    )
    m.freeze_model()
    m.unfreeze_all_probes()

    def weighted_ce(logits, targets, mask):
        t = mx.maximum(targets, 0).astype(mx.int32)
        ce = nn.losses.cross_entropy(logits, t, reduction="none")
        wm = mask.astype(mx.float32)
        return (ce * wm).sum() / mx.maximum(wm.sum(), 1.0)

    data = [
        {"tokens": [1, 2, 3, 4, 5], "labels": [0, 1, 2, 1, 0]},
        {"tokens": [6, 7, 8, 9, 10], "labels": [1, 1, 0, 2, 2]},
    ]
    trainer = Trainer(
        model=m,
        loss_fn=JointLoss(
            weights={"lm_head": 0.0},
            losses={"L0": weighted_ce, "L1": weighted_ce},
        ),
        num_iters=4,
        batch_size=2,
        save_steps=0,
        verbose=False,
    )
    batches = trainer.iterate(data)
    for _ in range(4):
        out = trainer.step(next(batches))
        assert "loss" in out and "L0_acc" not in out  # step returns loss/ntoks/components
    metrics = trainer.evaluate(data)
    assert "loss" in metrics


def test_build_dataset_shape_unchanged() -> None:
    """build_dataset still emits the same {tokens, labels} shape (offset=0)."""

    class _Tok:
        eos_token_id = 0

        def encode(self, text: str) -> list[int]:
            return [ord(c) for c in text]

    conv = [
        [{"role": "user", "content": "abc", "labels": {"p": [{"start": 2, "end": 3, "label": 1}]}}]
    ]
    samples = build_dataset(conv, _Tok(), offset=0)
    assert samples[0]["tokens"] == [ord("a"), ord("b"), ord("c")]
    assert samples[0]["labels"] == [-100, -100, 1]

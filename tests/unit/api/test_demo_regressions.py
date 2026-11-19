"""Regression tests for bugs first surfaced by the end-to-end demo scripts.

Each test reduces a real-run demo bug to a fast, in-memory unit test, so a future
refactor that reintroduces the bug fails here instead of only in a manual demo run.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import JointLoss, Model, ProbeConfig, Trainer
from auto_chasm.outputs import ProbeOutput


class _TinyLM(nn.Module):
    """Tiny MLX LM with a per-token hidden state for probe training."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.layers = [nn.Linear(16, 16) for _ in range(2)]
        self.output_proj = nn.Linear(16, 32)

    def __call__(self, x: mx.array, **kw: Any) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    """Config stub exposing hidden size and layer count."""

    hidden_size = 16
    num_hidden_layers = 2


def _tiny_model() -> Model:
    mx.random.seed(0)
    base = _TinyLM()
    base.config = _Cfg()
    return Model(base, None, "mlx")


class _ProbeHolder:
    """Minimal ``train_model`` stand-in exposing ``get_probe`` for the metrics fn."""

    def __init__(self, name: str, module: Any) -> None:
        self._name = name
        self._module = module

    def get_probe(self, name: str) -> Any:
        return self._module


class TestProbeF1WiredAndExported:
    """``default_binary_metrics`` is exported and computes correct F1.

    Demo bug: the binary-metrics function existed but was never exported or wired,
    so training logged a misleading ``F1 0.0000`` while the probe was learning.
    The demo fixed it by exporting it and passing ``eval_metrics_fn=...``.
    """

    def test_default_binary_metrics_exported_from_trainers(self) -> None:
        from auto_chasm.trainers import default_binary_metrics

        assert callable(default_binary_metrics)

    def test_perfect_classifier_scores_f1_one(self) -> None:
        from auto_chasm.trainers import default_binary_metrics

        mx.random.seed(1)
        probe = nn.Linear(16, 1)
        hidden = mx.random.normal((1, 6, 16))
        logits = probe(hidden)[..., 0]  # [1, 6]
        # Targets = the probe's own decisions => a perfect classifier => F1 == 1.
        targets = (logits > 0).astype(mx.float32)
        # Ensure the degenerate all-negative case can't happen.
        assert float(targets.sum().item()) > 0
        mask = mx.ones((1, 6))
        metrics = default_binary_metrics(_ProbeHolder("p", probe), {"p": hidden}, targets, mask)
        assert metrics["p_f1"] == 1.0
        assert metrics["p_accuracy"] == 1.0

    def test_standalone_evaluate_after_training_surfaces_probe_f1(self) -> None:
        """`trainer.evaluate()` AFTER training must still report probe metrics.

        Bug: `restore_capture_fns()` at the end of `run()` tore down the
        capture-fn wrapping that populated `_captured_hidden`, so a standalone
        `evaluate()` on MLX silently dropped every `eval_metrics_fn` result
        (loss/perplexity still reported, but no probe F1/accuracy).
        """
        from auto_chasm.trainers import default_binary_metrics

        model = _tiny_model()
        model.attach_probe(ProbeConfig(name="p", layers=[-1]))
        data = [
            {"tokens": [1, 2, 3, 4, 5], "labels": [0, 1, 0, 1, 0]},
            {"tokens": [6, 7, 8, 9, 10], "labels": [1, 0, 1, 0, 1]},
        ]
        trainer = Trainer(
            model,
            JointLoss(losses={"p": "bce"}),
            eval_metrics_fn=default_binary_metrics,
            num_iters=3,
            batch_size=2,
            verbose=False,
        )
        trainer.train(data)
        metrics = trainer.evaluate(data)
        assert any(k.endswith("_f1") for k in metrics), (
            f"standalone evaluate() after training dropped probe metrics: {sorted(metrics)}"
        )


class TestEvalAcceptsPerProbeDictLabels:
    """`evaluate` must accept per-probe `{name: labels}` datasets, not crash.

    Demo bug (multi-head): the eval path did ``mx.array(batch_labels)`` directly,
    which raises ``Invalid type dict`` for per-probe targets — even though training
    routes the same dict through ``labels_to_mlx``. The crash made multi-head
    evaluation impossible despite multi-head *training* working.
    """

    def test_evaluate_does_not_crash_on_per_probe_dict_labels(self) -> None:
        model = _tiny_model()
        model.add_probes([ProbeConfig(name="a", layers=[-1]), ProbeConfig(name="b", layers=[-1])])
        # Per-probe independent targets => iterate_batches yields a dict per batch.
        data = [
            {"tokens": [1, 2, 3, 4], "labels": {"a": [0, 1, 0, 1], "b": [1, 1, 0, 0]}},
            {"tokens": [5, 6, 7, 8], "labels": {"a": [1, 0, 1, 0], "b": [0, 0, 1, 1]}},
        ]
        trainer = Trainer(
            model,
            JointLoss(losses={"a": "bce", "b": "bce"}),
            num_iters=2,
            batch_size=2,
            verbose=False,
        )
        trainer.train(data)
        metrics = trainer.evaluate(data)  # must not raise
        assert math.isfinite(metrics["loss"])


class TestProbeOutputIgnoresMinus100:
    """`ProbeOutput.bce/.mse/.mae` must ignore the `-100` sentinel like `.ce`.

    Demo bug (custom loss): only `.ce()` folded `-100` into the mask; `.bce/.mse/.mae`
    fed `-100` straight into the loss, so the DOCUMENTED recipe
    ``o.probes[name].bce(labels, mask=o.mask)`` exploded (bce hit -2068) whenever
    labels contained `-100` (special/padding tokens inside the length region).
    """

    def test_bce_masks_minus_100_inside_a_true_mask(self) -> None:
        full_mask = mx.array([[True, True, True]])
        with_sentinel = float(
            ProbeOutput(logits=mx.array([[8.0, -8.0, 5.0]])).bce(
                mx.array([[1.0, 0.0, -100.0]]), mask=full_mask
            )
        )
        without = float(
            ProbeOutput(logits=mx.array([[8.0, -8.0]])).bce(
                mx.array([[1.0, 0.0]]), mask=mx.array([[True, True]])
            )
        )
        assert with_sentinel == pytest.approx(without, abs=1e-4)
        assert with_sentinel < 1.0

    def test_mse_and_mae_mask_minus_100(self) -> None:
        m = mx.array([[1.0, 1.0, 1.0]])
        logits = mx.array([[1.0, 2.0, 3.0]])
        tgt = mx.array([[1.0, 2.0, -100.0]])
        mse = float(ProbeOutput(logits=logits).mse(tgt, mask=m))
        mae = float(ProbeOutput(logits=logits).mae(tgt, mask=m))
        # Only the two matching positions contribute => ~0; the -100 must not blow up.
        assert mse < 1e-4 and mae < 1e-4


class TestModelForwardAttentionMaskOnBareModel:
    """`Model.forward(attention_mask=...)` must not crash on a base with no mask kwarg.

    Demo bug (custom loss): forward passed ``mask=attention_mask`` to the base, but
    mlx_lm models accept no padding-mask kwarg (and transformers want
    ``attention_mask``), so real models raised ``TypeError``. Test models only
    survived because they accept ``**kwargs``.
    """

    def test_forward_with_attention_mask_does_not_raise(self) -> None:
        class _Bare(nn.Module):
            """A base whose __call__ takes ONLY inputs (like mlx_lm)."""

            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(32, 16)
                self.layers = [nn.Linear(16, 16)]
                self.output_proj = nn.Linear(16, 32)

            def __call__(self, x: mx.array) -> mx.array:
                h = self.embedding(x)
                for layer in self.layers:
                    h = nn.gelu(layer(h))
                return self.output_proj(h)

        mx.random.seed(0)
        base = _Bare()
        base.config = _Cfg()
        model = Model(base, None, "mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[-1]))
        out = model.forward(mx.array([[1, 2, 3, 4]]), attention_mask=mx.array([[1, 1, 1, 0]]))
        assert out.lm_logits.shape == (1, 4, 32)


class TestEnableSteeringHonorsNewConfig:
    """Re-enabling steering with a new config must update method/scale, not reuse it.

    Demo bug (checkpoint+steering): a second `enable_steering` for the same probe
    took the else-branch and kept the FIRST config, so switching method or scale
    (e.g. nullify->boundary, scale 0->8) silently did nothing.
    """

    def test_reenable_updates_scale_and_method(self) -> None:
        from auto_chasm.config import SteeringConfig

        model = _tiny_model()
        probe = model.attach_probe(ProbeConfig(name="p", layers=[-1]))
        # A non-degenerate head + distinct class means so geometry builds.
        probe.module.weight = mx.ones((1, 16))
        probe.module.bias = mx.zeros((1,))
        cm = {"mean_0": -mx.ones((16,)), "mean_1": mx.ones((16,))}

        model.enable_steering(
            "p", config=SteeringConfig(method="nullify", scale=0.0), class_means=cm
        )
        model.enable_steering(
            "p", config=SteeringConfig(method="boundary", scale=8.0), class_means=cm
        )

        hook = model._steering_hooks["p"]
        assert hook.config.method == "boundary"
        assert hook.config.scale == 8.0


class TestAddSpecialTokensGrowsUntiedHead:
    """`add_special_tokens` must grow an untied head when tokenizer vocab != embed rows.

    Demo bug (gemma): `_resize_mlx` derived `old_vocab` from the tokenizer count,
    which differed from the embedding rows, so the head match failed — the head
    never grew and the new id produced no logit.
    """

    def test_untied_head_grows_despite_tokenizer_count_mismatch(self) -> None:
        from auto_chasm.special_tokens import add_special_tokens

        class _Tok:
            """Reports ONE MORE than the embedding has (mimics gemma's offset)."""

            def __init__(self, n: int) -> None:
                self._n = n

            def __len__(self) -> int:
                return self._n

            def add_tokens(self, toks: list, special_tokens: bool = False) -> int:
                self._n += len(toks)
                return len(toks)

        class _M(nn.Module):
            """Tiny model with an UNTIED output head."""

            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(10, 4)
                self.lm_head = nn.Linear(4, 10, bias=False)

            def __call__(self, x: mx.array) -> mx.array:
                return self.lm_head(self.embedding(x))

        m = _M()
        n = add_special_tokens(m, _Tok(11), ["<x>", "<y>"], "mlx")
        assert n == 2
        assert m.embedding.weight.shape[0] == 12
        assert m.lm_head.weight.shape[0] == 12  # the head grew too (was the bug)
        assert m(mx.array([[11]])).shape[-1] == 12  # the new id produces a logit


class TestBf16Scale0SteeringIsIdentity:
    """scale=0 steering on a bf16 model must be a true identity (dtype preserved).

    Demo bug (checkpoint+steering): the MLX steering branch did not cast the steered
    hidden back to the residual's dtype (the torch branch did), so a bf16 model's
    residual silently became fp32 — even scale=0 then perturbed downstream logits.
    """

    def test_scale0_preserves_dtype_and_logits(self) -> None:
        from auto_chasm.config import SteeringConfig

        model = _tiny_model()
        model.model.set_dtype(mx.bfloat16)
        probe = model.attach_probe(ProbeConfig(name="p", layers=[-1]))
        probe.module.weight = mx.ones((1, 16))
        probe.module.bias = mx.zeros((1,))
        cm = {"mean_0": -mx.ones((16,)), "mean_1": mx.ones((16,))}
        ids = mx.array([[1, 2, 3, 4]])

        base = model.forward(ids).lm_logits
        model.enable_steering(
            "p", config=SteeringConfig(method="nullify", scale=0.0), class_means=cm
        )
        steered = model.forward(ids).lm_logits
        model.disable_steering("p")

        assert base.dtype == steered.dtype  # bf16 preserved (was upcast to fp32)
        assert mx.allclose(base.astype(mx.float32), steered.astype(mx.float32), atol=1e-2).item()

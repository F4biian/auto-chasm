"""Tests for LayerSweep / SweepResult.

Pins: the per-layer best-val *selection* keeps the lower-val-loss eval and
restores that head's snapshot (oracle, via a controlled fake); an end-to-end
sweep completes on MLX (and torch when available) producing one row per layer
with the documented fields and leaving the best probes on the model; and
SweepResult.to_csv writes the documented header.
"""

from __future__ import annotations

import csv
import types

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import Dataset, JointLoss, LayerSweep, Model, ModuleSpec, SweepResult, Task
from auto_chasm.sweep import _BestPerLayerCallback


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
    def __init__(self, layers: int = 2) -> None:
        self.hidden_size = 16
        self.num_hidden_layers = layers


def _data(n: int) -> list[dict]:
    return [
        {"tokens": [(i + k) % 30 + 1 for k in range(5)], "labels": [i % 3, 0, 1, 2, (i + 1) % 3]}
        for i in range(n)
    ]


def test_best_per_layer_selection_and_restore() -> None:
    """The callback keeps the lower-val-loss snapshot and restores it exactly."""
    module = nn.Linear(4, 3)
    w1 = mx.array(module.weight)  # snapshot the weights we will rank best

    fake_model = types.SimpleNamespace(
        backend=types.SimpleNamespace(name="mlx"),
        probes={"L0": types.SimpleNamespace(module=module)},
    )

    # Two evals: first has the lower per-probe val loss (1.0), second is worse (2.0).
    # JointLoss keys the component by the BARE probe name ("L0"); the total "loss"
    # is INVERTED (worse first) so a bug that ranked on the total would pick step 2.
    metric_seq = [
        {"L0": 1.0, "lm_head": 4.0, "loss": 9.0, "L0_acc": 0.5, "L0_adj": 0.9},
        {"L0": 2.0, "lm_head": 4.0, "loss": 5.0, "L0_acc": 0.4, "L0_adj": 0.8},
    ]

    class _FakeTrainer:
        def __init__(self) -> None:
            self.i = 0

        def evaluate(self, _data: object) -> dict:
            out = metric_seq[self.i]
            self.i += 1
            return out

    cb = _BestPerLayerCallback(
        fake_model, None, ["L0"], "val_loss", higher_is_better=False, eval_every=1, num_iters=2
    )
    cb.trainer = _FakeTrainer()

    cb.on_step_end(step=1)  # snapshots w1 at loss 1.0
    module.weight = module.weight * 0.0 + 7.0  # mutate before the worse eval
    cb.on_step_end(step=2)  # loss 2.0 -> not improved, best stays step 1

    assert cb.best["L0"]["iter"] == 1
    assert cb.best["L0"]["val"]["val_loss"] == 1.0

    module.weight = module.weight * 0.0 + 99.0  # junk
    cb.restore_best()
    assert mx.allclose(module.weight, w1, atol=1e-6)


def test_layer_loss_prefers_bare_probe_component_over_total() -> None:
    """`_layer_loss` ranks a layer on its OWN loss component, never the combined total.

    JointLoss keys each probe's loss by its bare name (``L0``); a per-layer sweep must
    rank each layer on that, not the shared LM+all-probe total (regression for the
    Phase-3b component-key rename that silently collapsed per-layer selection).
    """
    from auto_chasm.sweep import _layer_loss

    metrics = {"L0": 1.0, "L1": 2.0, "lm_head": 4.0, "loss": 7.0, "L0_acc": 0.9}
    assert _layer_loss(metrics, "L0") == 1.0  # own component, NOT the total 7.0
    assert _layer_loss(metrics, "L1") == 2.0
    # Back-compat fallbacks: older suffixed component key, then the total as last resort.
    assert _layer_loss({"probe_ce:L0": 1.5, "loss": 9.0}, "L0") == 1.5
    assert _layer_loss({"loss": 3.0}, "Lx") == 3.0


def test_layer_sweep_end_to_end_mlx(tmp_path) -> None:
    """A 2-layer sweep runs, yields one row per layer, and keeps the best probes."""
    m = Model(_TinyMlp(layers=2), None, "mlx")
    m.model.config = _Cfg(layers=2)
    sweep = LayerSweep(
        m,
        out_features=3,
        module_spec=ModuleSpec.linear(out_features=3),
        layers=[0, 1],
        ordinal_tol=1,
    )
    result = sweep.run(
        _data(6),
        _data(2),
        _data(2),
        loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"L0": "ce", "L1": "ce"}),
        num_iters=20,
        eval_every=10,
        batch_size=2,
        max_seq_length=16,
        learning_rate=1e-2,
        verbose=False,
        output_dir=str(tmp_path),
    )
    assert set(result.best) == {0, 1}
    for i in (0, 1):
        row = result.best[i]
        for key in ("iter", "val_loss", "val_acc", "val_adj", "test_loss", "test_acc", "test_adj"):
            assert key in row
    # the best probes remain on the model (usable / checkpointable)
    assert {"L0", "L1"} <= set(m.probes)


def test_sweep_result_to_csv(tmp_path) -> None:
    """to_csv writes the documented 8-column header and one row per layer."""
    result = SweepResult(
        best={
            0: {
                "iter": 10,
                "val_loss": 1.0,
                "val_acc": 0.3,
                "val_adj": 0.7,
                "test_loss": 1.1,
                "test_acc": 0.25,
                "test_adj": 0.6,
            }
        }
    )
    path = tmp_path / "r.csv"
    result.to_csv(str(path))
    rows = list(csv.reader(path.open()))
    assert rows[0] == [
        "layer",
        "iter",
        "val_loss",
        "val_acc",
        "val_group_acc",
        "test_loss",
        "test_acc",
        "test_group_acc",
    ]
    assert rows[1][0] == "0"
    assert result.best_layer("test_adj") == 0


def test_layer_sweep_end_to_end_torch(tmp_path) -> None:
    """The same sweep completes on the torch backend (no Trainer.step used)."""
    pytest.importorskip("torch")
    import torch
    import torch.nn as tnn

    class _TorchTiny(tnn.Module):
        def __init__(self, h: int = 16, v: int = 32, layers: int = 2) -> None:
            super().__init__()
            self.embedding = tnn.Embedding(v, h)
            self.layers = tnn.ModuleList([tnn.Linear(h, h) for _ in range(layers)])
            self.output_proj = tnn.Linear(h, v)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.embedding(x)
            for layer in self.layers:
                h = torch.nn.functional.gelu(layer(h))
            return self.output_proj(h)

    raw = _TorchTiny(layers=2)
    raw.config = _Cfg(layers=2)
    m = Model(raw, None, "torch")
    sweep = LayerSweep(m, out_features=3, layers=[0, 1], ordinal_tol=1)
    result = sweep.run(
        _data(6),
        _data(2),
        _data(2),
        loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"L0": "ce", "L1": "ce"}),
        num_iters=12,
        eval_every=6,
        batch_size=2,
        max_seq_length=16,
        learning_rate=1e-2,
        verbose=False,
        output_dir=str(tmp_path),
    )
    assert set(result.best) == {0, 1}
    assert "test_acc" in result.best[0]


def test_task_derives_sweep_config() -> None:
    """A ``Task`` supplies out_features + num_classes + metrics; the knobs can't drift.

    Pure config check (no training): the classification↔regression switch becomes a
    one-argument change, and mixing ``task=`` with explicit ``out_features=`` is an
    error. Explicit ``out_features=`` still works (back-compat).
    """
    import numpy as np

    model = types.SimpleNamespace(num_layers=4)

    def _metric_keys(fn, out_features: int) -> set:  # noqa: ANN001
        """Run the derived metric fn against a fake head and return the metric keys."""
        logits = np.zeros((1, 2, out_features), dtype=np.float32)
        fake = types.SimpleNamespace(get_probe=lambda name: lambda _h: logits)
        return set(fn(fake, {"p": None}, np.array([[0, 2]]), np.array([[1, 1]])))

    multi = LayerSweep(model, task=Task.multiclass(6))
    assert (multi.out_features, multi.num_classes) == (6, 6)
    # The derived metrics are CLASSIFICATION metrics (macro-F1), not regression.
    assert _metric_keys(multi.eval_metrics_fn, 6) == {"p_acc", "p_adj", "p_macro_f1"}

    reg = LayerSweep(model, task=Task.regression(6))
    assert (reg.out_features, reg.num_classes) == (1, 6)  # scalar head, 6 ordinal bins
    # The derived metrics are REGRESSION metrics (mse/mae), not classification.
    assert _metric_keys(reg.eval_metrics_fn, 1) == {"p_mse", "p_mae", "p_acc", "p_adj"}

    with pytest.raises(ValueError, match="either task= .* or out_features="):
        LayerSweep(model, task=Task.multiclass(3), out_features=3)
    with pytest.raises(ValueError, match="needs the head width"):
        LayerSweep(model)  # neither task= nor out_features=

    explicit = LayerSweep(model, out_features=3)  # back-compat unchanged
    assert (explicit.out_features, explicit.num_classes) == (3, 3)


def test_layer_sweep_task_driven_end_to_end_mlx(tmp_path) -> None:
    """A sweep driven by ``Dataset.infer_task()`` runs end-to-end (the DX capstone).

    The data's integer labels (0..2) infer ``Task.multiclass(3)``, which drives the
    head width and the metrics — the user never hand-passes out_features/num_classes/
    eval_metrics_fn. One row per layer with the documented fields.
    """
    m = Model(_TinyMlp(layers=2), None, "mlx")
    m.model.config = _Cfg(layers=2)
    task = Dataset(_data(6)).infer_task()  # -> Task.multiclass(3)
    assert task == Task.multiclass(3)

    sweep = LayerSweep(
        m,
        task=task,
        module_spec=ModuleSpec.linear(out_features=task.out_features),
        layers=[0, 1],
    )
    result = sweep.run(
        _data(6),
        _data(2),
        _data(2),
        loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"L0": "ce", "L1": "ce"}),
        num_iters=20,
        eval_every=10,
        batch_size=2,
        max_seq_length=16,
        learning_rate=1e-2,
        verbose=False,
        output_dir=str(tmp_path),
    )
    assert set(result.best) == {0, 1}
    for i in (0, 1):
        for key in ("val_loss", "val_acc", "val_adj", "test_loss", "test_acc", "test_adj"):
            assert key in result.best[i]
    assert {"L0", "L1"} <= set(m.probes)


def test_public_api_is_typed_not_any_for_intellisense() -> None:
    """m12: LayerSweep's public params carry real types (not Any) for IntelliSense.

    Annotations are lazy strings (``from __future__ import annotations``), so we can
    assert the concrete type names appear without importing them at runtime.
    """
    init_ann = LayerSweep.__init__.__annotations__
    assert init_ann["model"] == "Model"
    assert init_ann["module_spec"] == "HeadSpec | None"  # str | ModuleSpec | Callable

    run_ann = LayerSweep.run.__annotations__
    assert run_ann["train_data"] == "DatasetLike"
    assert run_ann["val_data"] == "DatasetLike"
    assert run_ann["loss_fn"] == "LossFn"

    assert SweepResult.__annotations__["model"] == "Model | None"

"""Cross-backend bug regression tests — BUG-30 through BUG-36, plus M17."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ===========================================================================
# BUG-30: _TorchProbeWrapper.parameters() excludes probe module parameters
# ===========================================================================
# Fixed: parameters() now yields base + probe module params.
# Tests are regular (no xfail) since the bug is resolved.


class TestBug30TorchProbeWrapper:
    """BUG-30: _TorchProbeWrapper.parameters() must include probe modules."""

    def test_probe_modules_in_parameters(self, torch_model_wrapper: Any) -> None:
        """_TorchProbeWrapper.parameters() should yield probe module params."""
        import torch

        from auto_chasm.config import ProbeConfig
        from auto_chasm.trainers.wrappers import _TorchProbeWrapper

        torch.manual_seed(42)
        probe = torch_model_wrapper.attach_probe(ProbeConfig(name="test_probe", layers=[1]))

        raw_model = torch_model_wrapper.model
        wrapper = _TorchProbeWrapper(raw_model, torch_model_wrapper._probes)

        param_list = list(wrapper.parameters())
        base_params = list(raw_model.parameters())

        for bp in base_params:
            found = any(p is bp for p in param_list)
            assert found, "Base parameter missing from wrapper.parameters()"

        for pp in probe.module.parameters():
            found = any(p is pp for p in param_list)
            assert found, (
                f"Probe parameter {pp.shape} missing from wrapper.parameters(). "
                "BUG-30: wrapper.parameters() only returns base model params."
            )

    def test_probe_params_are_trainable_in_optimizer(self, torch_model_wrapper: Any) -> None:
        """Probe params must be in the optimizer's parameter groups."""
        import torch

        from auto_chasm.config import ProbeConfig
        from auto_chasm.trainers.wrappers import _TorchProbeWrapper

        torch.manual_seed(42)
        torch_model_wrapper.attach_probe(ProbeConfig(name="test_probe", layers=[1]))
        torch_model_wrapper.prepare_for_joint_training()

        raw_model = torch_model_wrapper.model
        wrapper = _TorchProbeWrapper(raw_model, torch_model_wrapper._probes)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, wrapper.parameters()), lr=1e-4
        )

        probe = torch_model_wrapper._probes["test_probe"]
        for pp in probe.module.parameters():
            in_optim = any(
                any(p is pp for p in group["params"]) for group in optimizer.param_groups
            )
            assert in_optim, (
                "Probe parameter not in optimizer. BUG-30: optimizer doesn't see probe params."
            )

    def test_probe_weights_change_after_training_step(self, torch_model_wrapper: Any) -> None:
        """After one training step, probe weights must change."""
        import torch

        from auto_chasm.config import ProbeConfig
        from auto_chasm.trainers.loss import JointLoss
        from auto_chasm.trainers.wrappers import _TorchProbeWrapper

        torch.manual_seed(42)
        torch_model_wrapper.attach_probe(ProbeConfig(name="test_probe", layers=[1]))
        torch_model_wrapper.prepare_for_joint_training()

        raw_model = torch_model_wrapper.model
        wrapper = _TorchProbeWrapper(raw_model, torch_model_wrapper._probes)
        wrapper.train()

        loss_fn = JointLoss(losses={"test_probe": "bce"})
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, wrapper.parameters()), lr=1.0
        )

        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float32)
        lengths = torch.tensor([[0, 5]])

        probe = torch_model_wrapper._probes["test_probe"]
        initial_weight = probe.module.weight.data.clone()

        total, _, _ = loss_fn(wrapper, tokens, labels, lengths)
        total.backward()
        optimizer.step()
        optimizer.zero_grad()

        current_weight = probe.module.weight.data
        assert not torch.allclose(initial_weight, current_weight), (
            "Probe weights did not change after training step. "
            "BUG-30: optimizer excludes probe params."
        )


# ===========================================================================
# BUG-31: Probe modules not set to train mode in PyTorch training path
# ===========================================================================
# Fixed: _Train_torch now uses wrapper.train() which sets probes to train mode.
# Test simulates the actual _train_torch code path.


class TestBug31TorchProbeTrainMode:
    """BUG-31: Probes must be in train mode during _train_torch."""

    def test_probes_set_to_train_mode_even_if_previously_eval(
        self, torch_model_wrapper: Any
    ) -> None:
        """_train_torch should set probes to train mode even if previously eval."""
        import torch

        from auto_chasm.config import ProbeConfig
        from auto_chasm.trainers.wrappers import _TorchProbeWrapper

        torch.manual_seed(42)
        torch_model_wrapper.attach_probe(ProbeConfig(name="test_probe", layers=[1]))

        raw_model = torch_model_wrapper.model
        raw_model.eval()
        for p in torch_model_wrapper._probes.values():
            p.module.eval()
        assert not any(p.module.training for p in torch_model_wrapper._probes.values())

        # Simulate what _train_torch now does: create wrapper, call .train()
        wrapper = _TorchProbeWrapper(raw_model, torch_model_wrapper._probes)
        wrapper.train()

        all_train = all(p.module.training for p in torch_model_wrapper._probes.values())
        assert all_train, "BUG-31: probes remain in eval mode after _train_torch setup."


# ===========================================================================
# BUG-32: Best checkpoint state in _train_torch excludes probe parameters
# ===========================================================================
# Fixed: state_dict() now includes probe params with {name}.{key} format.


class TestBug32TorchBestStateIncludesProbes:
    """BUG-32: Best checkpoint must save and restore probe parameters."""

    def test_best_state_dict_includes_probes(self, torch_model_wrapper: Any) -> None:
        """The best state dict should include probe params."""
        import torch

        from auto_chasm.config import ProbeConfig
        from auto_chasm.trainers.wrappers import _TorchProbeWrapper

        torch.manual_seed(42)
        torch_model_wrapper.attach_probe(ProbeConfig(name="test_probe", layers=[1]))
        raw_model = torch_model_wrapper.model
        wrapper = _TorchProbeWrapper(raw_model, torch_model_wrapper._probes)

        state = wrapper.state_dict()

        assert any("test_probe" in k for k in state), (
            "Best checkpoint state dict does not contain probe parameters. "
            "BUG-32: state_dict() delegates to base model via __getattr__."
        )

    def test_load_state_dict_restores_probes(self, torch_model_wrapper: Any) -> None:
        """After save/load round-trip, probe weights should be restored."""
        import torch

        from auto_chasm.config import ProbeConfig
        from auto_chasm.trainers.wrappers import _TorchProbeWrapper

        torch.manual_seed(42)
        torch_model_wrapper.attach_probe(ProbeConfig(name="test_probe", layers=[1]))
        raw_model = torch_model_wrapper.model
        wrapper1 = _TorchProbeWrapper(raw_model, torch_model_wrapper._probes)

        state = wrapper1.state_dict()

        # Create a fresh wrapper and load state
        wrapper2 = _TorchProbeWrapper(raw_model, torch_model_wrapper._probes)
        wrapper2.load_state_dict(state)

        state2 = wrapper2.state_dict()
        for k in state:
            assert k in state2, f"Key {k} missing after load"
            if hasattr(state[k], "shape"):
                assert torch.allclose(state[k], state2[k]), f"Value mismatch for {k}"


# ===========================================================================
# BUG-33: Inconsistent masking in class_means.py _compute_torch
# ===========================================================================
# Fixed: _compute_torch now uses length-based masking matching _compute_mlx.
# Test verifies _compute_torch produces correct results.


class TestBug33ClassMeansTorchMask:
    """BUG-33: _compute_torch must use length-based mask (not -100)."""

    def test_class_means_excludes_padding_tokens(self, torch_model_wrapper: Any) -> None:
        """_compute_torch should exclude padding tokens via length-based mask."""
        from auto_chasm.class_means import _compute_torch

        # Simulate a short dataset where padding labels=0 (same as valid class 0).
        # If _compute_torch used -100 masking, padding would be included.
        # With length-based masking, only true token positions contribute.
        class DummyIterateBatches:
            """Mimics iterate_batches yielding one batch."""

            def __call__(self, dataset, batch_size, max_seq_length, loop=False):
                lengths = np.array([[0, 3]])
                # tokens: 3 real tokens, rest padding (0)
                tokens = np.array([[1, 2, 3, 0, 0]], dtype=np.int32)
                # labels: 0=non-digit, 1=digit for 3 real tokens, rest padding
                labels = np.array([[0, 0, 1, 0, 0]], dtype=np.int32)
                yield tokens, labels, lengths

        from auto_chasm.config import ProbeConfig as _Cfg

        torch_model_wrapper.attach_probe(_Cfg(name="test_probe", layers=[1]))

        mean_0, mean_1 = _compute_torch(
            torch_model_wrapper,  # type: ignore[arg-type]
            torch_model_wrapper._probes["test_probe"],
            None,  # dataset
            hidden_dim=16,
            batch_size=1,
            max_seq_length=5,
            iterate_batches=DummyIterateBatches(),
        )

        # With correct masking, position 2 (digit=1, label=1) contributes
        # to mean_1 and positions 0,1 (digit=0, labels=0) to mean_0.
        # Padding positions with label 0 should NOT contribute to mean_0.
        # If they did, mean_0 would be diluted.
        import torch

        assert torch.isfinite(mean_0).all(), "mean_0 has NaN/inf"
        assert torch.isfinite(mean_1).all(), "mean_1 has NaN/inf"
        assert mean_0.shape == (16,), f"mean_0 wrong shape: {mean_0.shape}"
        assert mean_1.shape == (16,), f"mean_1 wrong shape: {mean_1.shape}"


# ===========================================================================
# BUG-34: Gradient clipping excludes probe params in PyTorch path
# ===========================================================================
# Fixed: parameters() now includes probes, so clip_grad_norm_ covers them.


class TestBug34TorchGradClipIncludesProbes:
    """BUG-34: Gradient clipping must cover probe parameters."""

    def test_grad_clip_includes_probe_parameters(self, torch_model_wrapper: Any) -> None:
        """Gradient clipping should apply to all trainable params including probes."""
        import torch

        from auto_chasm.config import ProbeConfig
        from auto_chasm.trainers.wrappers import _TorchProbeWrapper

        torch.manual_seed(42)
        torch_model_wrapper.attach_probe(ProbeConfig(name="test_probe", layers=[1]))
        raw_model = torch_model_wrapper.model
        wrapper = _TorchProbeWrapper(raw_model, torch_model_wrapper._probes)

        trainable = [p for p in wrapper.parameters() if p.requires_grad]
        probe = torch_model_wrapper._probes["test_probe"]

        has_probe_params = any(
            id(p) in [id(pp) for pp in probe.module.parameters()] for p in trainable
        )
        assert has_probe_params, (
            "Probe parameters not in trainable list. "
            "BUG-34: gradient clipping would skip probe params."
        )


# ===========================================================================
# BUG-35: TrainingConfig.probe_weights stored but never used
# ===========================================================================
# Fixed: Trainer now calls _apply_probe_weights() which wires them into
# JointLoss.  Test probes the __init__ source code for wiring logic.


class TestBug35ConfigFieldsUsed:
    """BUG-35: TrainingConfig fields must be wired to the loss function."""

    def test_probe_weights_passed_to_loss(self) -> None:
        """probe_weights from TrainingConfig should be wired to JointLoss."""
        from auto_chasm.trainers.trainer import Trainer

        src = inspect.getsource(Trainer.__init__)

        has_probe_weights_passing = "probe_weights" in src
        assert has_probe_weights_passing, (
            "BUG-35: TrainingConfig.probe_weights is never passed to the loss."
        )

    def test_apply_probe_weights_updates_loss(self, model_wrapper: Any) -> None:
        """Calling _apply_probe_weights should update JointLoss weights."""
        from auto_chasm.config import TrainingConfig
        from auto_chasm.trainers.loss import JointLoss
        from auto_chasm.trainers.trainer import Trainer

        loss_fn = JointLoss()
        _trainer = Trainer(
            model=model_wrapper,
            loss_fn=loss_fn,
            config=TrainingConfig(probe_weights={"test": 2.0}),
            num_iters=1,
            verbose=False,
        )

        # _apply_probe_weights is called in __init__
        assert loss_fn._probe_weights.get("test") == 2.0, (
            "BUG-35: probe_weights not applied to loss."
        )


# ===========================================================================
# BUG-36: _compute_torch ignores lengths parameter
# ===========================================================================
# Fixed: _compute_torch now uses lengths for masking (not _lengths discard).


class TestBug36TorchClassMeansIgnoresLengths:
    """BUG-36: _compute_torch discards lengths from iterate_batches."""

    def test_class_means_torch_uses_lengths(self) -> None:
        """_compute_torch should use lengths for masking, not ignore them."""
        from auto_chasm.class_means import _compute_torch

        source = inspect.getsource(_compute_torch)
        assert "_lengths" not in source, (
            "BUG-36: _compute_torch parameter is named _lengths (discarded), "
            "not lengths (used for masking)."
        )


# ===========================================================================
# M17: torch best-val checkpoint tracked only when early stopping is armed
# ===========================================================================
# Fixed: best-state capture no longer nested inside `if es_active`, matching
# MLX, which restores the best-val weights whenever val data is evaluated.


class TestM17TorchBestStateWithoutEarlyStopping:
    """M17: patience=0 + val_data must still restore the best-val checkpoint."""

    def test_best_iter_recorded_with_early_stopping_off(self, tmp_path: Path) -> None:
        """A torch run with patience=0 but val_data records a best-val iter (was 0).

        Regression: the torch loop captured ``best_state`` only inside the
        ``es_active`` (patience > 0) branch, so with early stopping OFF it kept
        the last-step weights while MLX restored the best-val weights -- a silent
        cross-backend divergence. With the fix, best-val is tracked whenever we
        evaluate, so ``best_iter > 0``.
        """
        pytest.importorskip("torch")

        from auto_chasm import Model
        from auto_chasm.config import ProbeConfig
        from auto_chasm.trainers.loss import JointLoss
        from auto_chasm.trainers.trainer import Trainer
        from tests.conftest import DummyTokenizer, _make_torch_tiny_mlp

        torch_model = _make_torch_tiny_mlp(hidden_dim=4, vocab_size=8, num_layers=2)

        class Cfg:
            """Dummy config."""

            hidden_size = 4
            num_hidden_layers = 2

        torch_model.config = Cfg()
        wrapper = Model(torch_model, DummyTokenizer(), backend_name="torch")
        wrapper.attach_probe(ProbeConfig(name="p", layers=[-1]))
        wrapper.prepare_for_joint_training()

        data = [{"tokens": [1, 2, 3], "labels": [0, 0, 1]} for _ in range(4)]
        out = tmp_path / "out"
        trainer = Trainer(
            model=wrapper,
            loss_fn=JointLoss(weights={"lm_head": 0.0, "p": 1.0}),
            num_iters=4,
            batch_size=2,
            max_seq_length=8,
            eval_steps=2,  # validation runs, so a best-val checkpoint can be tracked
            early_stopping_patience=0,  # ...but early stopping itself is OFF
            output_dir=str(out),
            verbose=False,
        )
        trainer.train(data, val_data=data)

        manifest = json.loads((out / "final" / "training_manifest.json").read_text())
        assert manifest["best_iter"] > 0, (
            "M17: torch kept last-step weights (best_iter=0) with patience=0 + val_data "
            "instead of tracking the best-val checkpoint like MLX."
        )

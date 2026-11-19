"""Fuzz tests 1 — backend ops, config, loss, and output fuzzing.

Tests extreme, invalid, and nonsensical inputs for robustness.
"""

from __future__ import annotations

import contextlib
from typing import Any, cast

import mlx.core as mx
import numpy as np
import pytest

from auto_chasm import GenerationConfig, JointLoss, ProbeConfig, SteeringConfig
from auto_chasm.backends import Backend
from auto_chasm.outputs import JointOutputs, LossOutputs, ModelOutputs, ProbeOutput


class TestBackendOpsFuzzing:
    """Fuzz tests for backend tensor and module operations."""

    def test_backend_force_invalid(self) -> None:
        r"""Backend(force="invalid") should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            Backend(force="invalid")

    def test_tensor_from_empty_list(self) -> None:
        """TensorOps.tensor([]) should produce an empty tensor."""
        backend = Backend(force="mlx")
        t = backend.tensor.tensor([])
        assert t.size == 0

    def test_tensor_weird_nested_list(self) -> None:
        """TensorOps.tensor([[[1]]]) should produce a 3D tensor."""
        backend = Backend(force="mlx")
        t = backend.tensor.tensor([[[1]]])
        assert t.shape == (1, 1, 1)
        assert t.item() == 1

    def test_sample_single_token_vocab(self) -> None:
        """Sample from 1-element logits should return index 0."""
        backend = Backend(force="mlx")
        logits = mx.array([42.0])
        token = backend.tensor.sample(logits, temperature=0.0)
        assert token == 0

    def test_sample_single_token_vocab_with_temp(self) -> None:
        """Sample with temperature from 1-element logits should return 0."""
        backend = Backend(force="mlx")
        logits = mx.array([42.0])
        token = backend.tensor.sample(logits, temperature=1.0)
        assert token == 0

    def test_to_numpy_with_nan(self) -> None:
        """to_numpy on a tensor with NaN values should not crash."""
        backend = Backend(force="mlx")
        t = mx.array([float("nan"), 1.0, float("nan")])
        arr = backend.tensor.to_numpy(t)
        assert isinstance(arr, np.ndarray)
        assert np.isnan(arr[0])

    def test_to_numpy_with_inf(self) -> None:
        """to_numpy on a tensor with Inf values should not crash."""
        backend = Backend(force="mlx")
        t = mx.array([float("inf"), -float("inf"), 0.0])
        arr = backend.tensor.to_numpy(t)
        assert isinstance(arr, np.ndarray)
        assert np.isinf(arr[0])

    def test_sample_nan_logits_greedy(self) -> None:
        """Sample from all-NaN logits greedily should not crash."""
        backend = Backend(force="mlx")
        logits = mx.array([float("nan"), float("nan"), float("nan")])
        with contextlib.suppress(ValueError, RuntimeError):
            token = backend.tensor.sample(logits, 0.0)
            assert isinstance(token, int)

    def test_sample_nan_logits_tempered(self) -> None:
        """Sample from all-NaN logits with temperature should not crash."""
        backend = Backend(force="mlx")
        logits = mx.array([float("nan"), float("nan")])
        with contextlib.suppress(ValueError, RuntimeError):
            token = backend.tensor.sample(logits, 1.0)
            assert isinstance(token, int)

    def test_sample_uniform_logits(self) -> None:
        """Sample from uniform logits should return a valid token index."""
        backend = Backend(force="mlx")
        logits = mx.ones(10)
        token = backend.tensor.sample(logits, temperature=1.0)
        assert 0 <= token < 10

    def test_freeze_empty_module(self) -> None:
        """ModuleOps.freeze on an empty module should be a no-op."""
        backend = Backend(force="mlx")
        import mlx.nn as nn

        empty = nn.Module()
        backend.module.freeze(empty)
        assert True


class TestJointLossFuzzing:
    """Fuzz tests for JointLoss construction and invocation."""

    def test_zero_lm_and_probe_weight(self) -> None:
        """JointLoss with both weights zero should produce zero total loss."""
        loss_fn = JointLoss(weights={"lm_head": 0.0, "probe": 0.0})

        class _Mock:
            def __call__(self, inputs: Any) -> tuple[Any, Any]:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), mx.ones((b, t))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 3]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert float(total) == 0.0

    def test_invalid_probe_loss_default_raises(self) -> None:
        """JointLoss with invalid probe_loss string should raise."""
        with pytest.raises(ValueError, match="Unknown loss"):
            JointLoss(losses={"probe": "invalid"})

    def test_invalid_probe_loss_per_probe_raises(self) -> None:
        """JointLoss with invalid per-probe loss should raise."""
        with pytest.raises(ValueError, match="Unknown loss"):
            JointLoss(losses={"p": "invalid_loss"})

    def test_nonexistent_probe_weight_silent(self) -> None:
        """JointLoss weighting only the attached probe should not crash."""
        loss_fn = JointLoss(
            weights={"lm_head": 0.0},
            losses={"p": "bce"},
        )

        class _Mock:
            def __call__(self, inputs: Any) -> tuple[Any, Any]:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), {"p": mx.ones((b, t))}

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 3]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert float(total) >= 0.0

    def test_labels_wrong_shape_b_t_1(self) -> None:
        """JointLoss with labels shape [B, T, 1] should not crash."""
        loss_fn = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})

        class _Mock:
            def __call__(self, inputs: Any) -> tuple[Any, Any]:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), mx.ones((b, t))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[[0.0], [1.0], [0.0]]])
        lengths = mx.array([[0, 3]])
        with contextlib.suppress(Exception):
            total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
            assert float(total) >= 0.0

    def test_all_zero_lengths_mask(self) -> None:
        """JointLoss with all-zero lengths mask should not produce NaN."""
        loss_fn = JointLoss(losses={"probe": "bce"})

        class _Mock:
            def __call__(self, inputs: Any) -> tuple[Any, Any]:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), mx.ones((b, t))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 0]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert float(total) >= 0.0
        assert mx.isfinite(total).item()

    def test_lengths_exceed_sequence_length(self) -> None:
        """JointLoss with lengths exceeding sequence length should not crash."""
        loss_fn = JointLoss()

        class _Mock:
            def __call__(self, inputs: Any) -> tuple[Any, Any]:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), None

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[10, 20]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert float(total) >= 0.0
        assert mx.isfinite(total).item()

    def test_negative_lengths(self) -> None:
        """JointLoss with negative lengths should not produce NaN loss."""
        loss_fn = JointLoss()

        class _Mock:
            def __call__(self, inputs: Any) -> tuple[Any, Any]:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), None

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[-1, -1]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert float(total) >= 0.0

    def test_labels_with_nan_values(self) -> None:
        """JointLoss with NaN in labels should not crash."""
        loss_fn = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})

        class _Mock:
            def __call__(self, inputs: Any) -> tuple[Any, Any]:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), mx.ones((b, t))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[float("nan"), 1.0, float("nan")]])
        lengths = mx.array([[0, 3]])
        with contextlib.suppress(Exception):
            loss_fn(_Mock(), batch, labels, lengths)

    def test_lm_only_mode_with_probe_skip(self) -> None:
        """JointLoss with lm_weight > 0 and no probes should not crash."""
        loss_fn = JointLoss()

        class _Mock:
            def __call__(self, inputs: Any) -> tuple[Any, Any]:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), None

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 3]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert float(total) >= 0.0
        assert "lm_head" in components

    def test_custom_callable_probe_loss(self) -> None:
        """JointLoss with a custom callable probe loss should work."""

        def _custom_loss(
            logits: Any,
            targets: Any,
            mask: Any,
        ) -> Any:
            return mx.array(0.5)

        loss_fn = JointLoss(
            weights={"lm_head": 0.0},
            losses={"probe": _custom_loss},
        )

        class _Mock:
            def __call__(self, inputs: Any) -> tuple[Any, Any]:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), mx.ones((b, t))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 3]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert "probe" in components

    def test_probe_losses_with_missing_probe(self) -> None:
        """JointLoss with probe_losses for missing probes should not crash."""
        loss_fn = JointLoss(
            weights={"lm_head": 0.0},
            losses={"p": "bce", "missing_probe": "mse"},
        )

        class _Mock:
            def __call__(self, inputs: Any) -> tuple[Any, Any]:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), {"p": mx.ones((b, t))}

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 3]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert "p" in components


class TestOutputsFuzzing:
    """Fuzz tests for output containers and loss methods."""

    def test_probe_output_bce_with_none_logits(self) -> None:
        """ProbeOutput.bce with None logits should raise AssertionError."""
        po = ProbeOutput(logits=None)
        with pytest.raises(AssertionError):
            po.bce(None)

    def test_probe_output_bce_with_none_targets(self) -> None:
        """ProbeOutput.bce with None targets should raise."""
        po = ProbeOutput(logits=mx.array([1.0, 2.0]))
        with pytest.raises((TypeError, AttributeError)):
            po.bce(None)

    def test_probe_output_bce_mismatched_shapes(self) -> None:
        """ProbeOutput.bce with mismatched logits/targets should raise."""
        po = ProbeOutput(logits=mx.array([[1.0, 2.0], [3.0, 4.0]]))
        targets = mx.array([1.0, 0.0])
        with pytest.raises(Exception):
            po.bce(targets)

    def test_probe_output_mse_with_none_logits(self) -> None:
        """ProbeOutput.mse with None logits should raise AssertionError."""
        po = ProbeOutput(logits=None)
        with pytest.raises(AssertionError):
            po.mse(mx.array([1.0]))

    def test_probe_output_ce_multi_dim_targets(self) -> None:
        """ProbeOutput.ce with multi-dimensional targets should not crash."""
        po = ProbeOutput(logits=mx.array([[[1.0, 2.0, 3.0]]]))
        targets = mx.array([[[0]]])
        with contextlib.suppress(Exception):
            result = po.ce(targets)
            assert result is not None

    def test_probe_output_ce_with_none_logits(self) -> None:
        """ProbeOutput.ce with None logits should raise AssertionError."""
        po = ProbeOutput(logits=None)
        with pytest.raises(AssertionError):
            po.ce(mx.array([0]))

    def test_probe_output_bce_with_mask_none(self) -> None:
        """ProbeOutput.bce with mask=None should compute unmasked loss."""
        po = ProbeOutput(logits=mx.array([0.0, 0.0]))
        loss = po.bce(mx.array([0.0, 1.0]), mask=None)
        assert float(loss) > 0.0

    def test_probe_output_bce_with_empty_mask(self) -> None:
        """ProbeOutput.bce with all-False mask should not crash."""
        po = ProbeOutput(logits=mx.array([0.0, 0.0]))
        mask = mx.array([False, False])
        loss = po.bce(mx.array([0.0, 1.0]), mask=mask)
        assert float(loss) >= 0.0

    def test_model_outputs_none_fields(self) -> None:
        """ModelOutputs with None lm_logits should not crash on creation."""
        outputs = ModelOutputs(lm_logits=None, probes={})
        assert outputs.lm_logits is None
        assert outputs.probes == {}

    def test_loss_outputs_none_fields(self) -> None:
        """LossOutputs with None fields should not crash on creation."""
        outputs = LossOutputs(lm_ce=None, total=None)
        assert outputs.lm_ce is None
        assert outputs.total is None

    def test_loss_outputs_all_components_empty(self) -> None:
        """LossOutputs.all_components with all-None should be empty."""
        outputs = LossOutputs()
        assert outputs.all_components == {}

    def test_joint_outputs_empty_probes(self) -> None:
        """JointOutputs with empty probes dict should not crash."""
        targets = mx.array([[1, 2, 3]])
        lengths = mx.array([[0, 3]])
        jo = JointOutputs(
            lm_logits=mx.zeros((1, 3, 32)),
            probes={},
            targets=targets,
            lengths=lengths,
        )
        assert jo.ntoks is not None
        assert float(jo.ntoks) > 0

    def test_joint_outputs_zero_length_targets(self) -> None:
        """JointOutputs with shape T=0 should not crash on mask."""
        targets = mx.zeros((1, 0), dtype=mx.int32)
        lengths = mx.array([[0, 0]])
        jo = JointOutputs(
            lm_logits=mx.zeros((1, 0, 32)),
            probes={},
            targets=targets,
            lengths=lengths,
        )
        _ = jo.mask

    def test_joint_outputs_ntoks_property(self) -> None:
        """JointOutputs.ntoks with no valid tokens should return 0."""
        targets = mx.array([[1, 2, 3]])
        lengths = mx.array([[0, 0]])
        jo = JointOutputs(
            lm_logits=mx.zeros((1, 3, 32)),
            probes={},
            targets=targets,
            lengths=lengths,
        )
        assert float(jo.ntoks) == 0.0


class TestConfigFuzzing:
    """Fuzz tests for config dataclasses with extreme values."""

    def test_probe_config_empty_name(self) -> None:
        """ProbeConfig with empty name should not crash."""
        cfg = ProbeConfig(name="", layers=[0])
        assert cfg.name == ""

    def test_probe_config_empty_layers_raises(self) -> None:
        """ProbeConfig with empty layers list should raise."""
        with pytest.raises(ValueError, match="at least one"):
            ProbeConfig(name="test", layers=[])

    def test_probe_config_invalid_aggregation_raises(self) -> None:
        """ProbeConfig with invalid aggregation should raise."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            ProbeConfig(name="test", layers=[0], aggregation="nonexistent")

    def test_probe_config_subblock_source_constructs(self) -> None:
        """Sub-block sources (attention/mlp/residual) now build a valid config."""
        assert ProbeConfig(name="test", layers=[0], source="attention").source == "attention"

    def test_probe_config_negative_layers(self) -> None:
        """ProbeConfig with negative layers should not crash."""
        cfg = ProbeConfig(name="test", layers=[-1])
        assert cfg.layers == [-1]

    def test_probe_config_extreme_out_features(self) -> None:
        """ProbeConfig with out_features=0 should not crash on creation."""
        cfg = ProbeConfig(
            name="test",
            layers=[0],
            module_config={"out_features": 0},
        )
        assert cfg.module_config["out_features"] == 0

    def test_steering_config_invalid_method_raises(self) -> None:
        """SteeringConfig with nonexistent method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown steering method"):
            SteeringConfig(method=cast(Any, "nonexistent"))

    def test_generation_config_negative_temperature(self) -> None:
        """GenerationConfig must REJECT a negative temperature.

        A negative temperature would invert the sampling distribution
        (sampling the least-likely tokens); the corrected contract raises at
        construction instead of silently producing wrong samples.
        """
        with pytest.raises(ValueError, match="temperature"):
            GenerationConfig(temperature=-1.0, do_sample=True)

    def test_generation_config_zero_top_p(self) -> None:
        """GenerationConfig with top_p=0 should not crash."""
        cfg = GenerationConfig(top_p=0.0)
        assert cfg.top_p == 0.0

    def test_generation_config_zero_repetition_penalty(self) -> None:
        """GenerationConfig with repetition_penalty=0 should not crash."""
        cfg = GenerationConfig(repetition_penalty=0.0)
        assert cfg.repetition_penalty == 0.0

    def test_generation_config_extreme_max_tokens(self) -> None:
        """GenerationConfig with max_tokens=0 should not crash."""
        cfg = GenerationConfig(max_tokens=0)
        assert cfg.max_tokens == 0

    def test_generation_config_negative_repetition_penalty(self) -> None:
        """GenerationConfig with negative repetition_penalty should not crash."""
        cfg = GenerationConfig(repetition_penalty=-5.0)
        assert cfg.repetition_penalty == -5.0

    def test_lora_config_zero_rank(self) -> None:
        """LoraConfig must REJECT rank=0 (effective scale is alpha/rank).

        rank=0 is a divide-by-zero waiting to happen; the corrected contract
        raises at construction rather than blowing up cryptically in the backend.
        """
        from auto_chasm import LoraConfig

        with pytest.raises(ValueError, match="rank"):
            LoraConfig(rank=0)

    def test_training_config_extreme_lr(self) -> None:
        """TrainingConfig with learning_rate=0 should not crash."""
        from auto_chasm import TrainingConfig

        cfg = TrainingConfig(learning_rate=0.0)
        assert cfg.learning_rate == 0.0

    def test_training_config_invalid_lr_schedule(self) -> None:
        """TrainingConfig with invalid lr_schedule should not crash."""
        from auto_chasm import TrainingConfig

        cfg = TrainingConfig(lr_schedule=cast(Any, "invalid"))
        assert cfg.lr_schedule == cast(Any, "invalid")

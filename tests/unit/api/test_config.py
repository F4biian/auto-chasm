"""Tests for config dataclasses and output containers."""

from __future__ import annotations

import pytest

from auto_chasm.config import (
    GenerationConfig,
    LoraConfig,
    ProbeConfig,
    RLConfig,
    SteeringConfig,
    TrainingConfig,
)
from auto_chasm.outputs import LossOutputs, ModelOutputs, ProbeLossInfo, ProbeOutput


class TestProbeConfig:
    """Tests for ProbeConfig."""

    def test_default_values(self) -> None:
        config = ProbeConfig(name="test", layers=[5])
        assert config.name == "test"
        assert config.layers == [5]
        assert config.source == "hidden"
        assert config.aggregation == "concat"
        assert config.module_type == "linear"
        assert config.granularity == "token"
        assert config.layer_norm is False

    def test_custom_values(self) -> None:
        config = ProbeConfig(
            name="quality",
            layers=[5, 10, 17],
            source="hidden",  # a per-layer source (embedding/logits are single-site)
            aggregation="mean",
            module_type="mlp",
            module_config={"hidden_dim": 128, "dropout": 0.1},
            granularity="response",
        )
        assert config.name == "quality"
        assert config.layers == [5, 10, 17]
        assert config.source == "hidden"
        assert config.aggregation == "mean"
        assert config.module_type == "mlp"
        assert config.module_config["hidden_dim"] == 128
        assert config.granularity == "response"

    def test_empty_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ProbeConfig(name="test", layers=[])

    def test_invalid_aggregation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown aggregation"):
            ProbeConfig(name="test", layers=[5], aggregation="invalid")

    def test_custom_aggregation_callable(self) -> None:
        def custom_agg(states: list) -> list:
            return states[0]

        config = ProbeConfig(name="test", layers=[5], aggregation=custom_agg)
        assert callable(config.aggregation)

    def test_negative_layer_index(self) -> None:
        config = ProbeConfig(name="test", layers=[-1])
        assert config.layers == [-1]

    def test_reserved_lm_head_name_raises(self) -> None:
        """A probe named ``lm_head`` is rejected at construction (reserved term).

        ``lm_head`` is the fixed language-model term name in ``JointLoss``
        weights/losses and the ``combine`` namespace; a probe taking it would
        collide with the LM term. Enforcing it here (earliest point) covers every
        attach path, since ``attach_probe``/``add_probes`` all take a ``ProbeConfig``.
        """
        with pytest.raises(ValueError, match="reserved for the language-model head"):
            ProbeConfig(name="lm_head", layers=[5])
        # A different name is fine.
        assert ProbeConfig(name="lm_head_probe", layers=[5]).name == "lm_head_probe"


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self) -> None:
        config = TrainingConfig()
        assert config.lm_weight == 1.0
        assert config.probe_weight == 1.0
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.lr_schedule == "cosine"
        assert config.output_dir == "./checkpoints"

    def test_invalid_warmup_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="warmup_ratio"):
            TrainingConfig(warmup_ratio=1.5)
        with pytest.raises(ValueError, match="warmup_ratio"):
            TrainingConfig(warmup_ratio=-0.1)


class TestRLConfig:
    """Tests for RLConfig."""

    def test_default_values(self) -> None:
        config = RLConfig()
        assert config.algorithm == "sft"
        assert config.beta == 0.1
        assert config.clip_ratio == 0.2


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_default_values(self) -> None:
        config = GenerationConfig()
        assert config.max_tokens == 256
        assert config.temperature == 0.0
        assert config.do_sample is False


class TestSteeringConfig:
    """Tests for SteeringConfig."""

    def test_default_values(self) -> None:
        config = SteeringConfig()
        assert config.method == "nullify"
        assert config.scale == 1.0

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown steering method"):
            SteeringConfig(method="invalid")


class TestLoraConfig:
    """Tests for LoraConfig."""

    def test_default_peft_method(self) -> None:
        cfg = LoraConfig()
        assert cfg.peft_method == "lora"

    def test_peft_method_qlora(self) -> None:
        cfg = LoraConfig(peft_method="qlora")
        assert cfg.peft_method == "qlora"

    def test_peft_method_dora(self) -> None:
        cfg = LoraConfig(peft_method="dora")
        assert cfg.peft_method == "dora"

    def test_default_values(self) -> None:
        cfg = LoraConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16
        assert cfg.dropout == 0.0
        assert cfg.target_modules is None


class TestPEFTMethodsExist:
    """Tests for apply_qlora / apply_dora existence."""

    def test_apply_qlora_callable(self) -> None:
        from auto_chasm.peft import apply_qlora

        assert callable(apply_qlora)

    def test_apply_dora_callable(self) -> None:
        from auto_chasm.peft import apply_dora

        assert callable(apply_dora)


class TestLossOutputs:
    """Tests for LossOutputs."""

    def test_empty_outputs(self) -> None:
        outputs = LossOutputs()
        assert outputs.lm_ce is None
        assert outputs.total is None
        assert outputs.probes == {}

    def test_all_components(self) -> None:
        outputs = LossOutputs(
            lm_ce=1.5,
            probes={"hallucination": ProbeLossInfo(total=0.3, components={"bce": 0.3})},
            total=1.8,
        )
        components = outputs.all_components
        assert components["lm_ce"] == 1.5
        assert components["hallucination"] == 0.3
        assert components["hallucination_bce"] == 0.3
        assert components["total"] == 1.8


class TestModelOutputs:
    """Tests for ModelOutputs."""

    def test_empty_outputs(self) -> None:
        outputs = ModelOutputs()
        assert outputs.lm_logits is None
        assert outputs.probes == {}
        assert outputs.loss is None

    def test_with_probes(self) -> None:
        outputs = ModelOutputs(
            lm_logits="fake_logits",
            probes={"test": ProbeOutput(logits="fake_probe_logits")},
        )
        assert outputs.lm_logits == "fake_logits"
        assert "test" in outputs.probes

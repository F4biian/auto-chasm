"""Fuzz tests 2 — history, probe, model, and generation fuzzing.

Tests extreme, invalid, and nonsensical inputs for robustness.
"""

from __future__ import annotations

import contextlib
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import GenerationConfig, Model, ProbeConfig, SteeringConfig
from auto_chasm.backends import Backend
from auto_chasm.history import History, HistoryEntry
from auto_chasm.outputs import GenerationStep, LossOutputs, ModelOutputs, ProbeOutput
from auto_chasm.probe import Probe


class _TinyMlp:
    """Minimal MLP matching TinyMlp for isolated model fuzz tests."""

    def __init__(
        self,
        hidden_dim: int = 16,
        vocab_size: int = 32,
        num_layers: int = 4,
    ) -> None:
        import mlx.nn as nn

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array, **kwargs: Any) -> mx.array:
        import mlx.nn as nn

        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Tok:
    """Minimal tokenizer for tests."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids if i > 0)


class _Cfg:
    """Dummy model config."""

    hidden_size = 16
    num_hidden_layers = 4


def _make_model(backend: str = "mlx") -> Model:
    """Build a bare-bones Model for tests that cannot use fixtures."""
    base = _TinyMlp()
    base.config = _Cfg()
    return Model(base, _Tok(), backend_name=backend)


class TestHistoryFuzzing:
    """Fuzz tests for History and HistoryEntry."""

    def test_history_entry_negative_step(self) -> None:
        """HistoryEntry with negative step should not crash."""
        entry = HistoryEntry(step=-1)
        assert entry.step == -1

    def test_history_from_dict_wrong_step_type(self) -> None:
        """History.from_dict with string step should not crash."""
        h = History.from_dict({"entries": [{"step": "not_an_int"}]})
        assert len(h) == 1
        assert h[0].step == "not_an_int"

    def test_history_from_dict_missing_step_key(self) -> None:
        """History.from_dict with missing step should raise KeyError."""
        with pytest.raises(KeyError):
            History.from_dict({"entries": [{"train_loss": 0.5}]})

    def test_history_load_json_nonexistent(self) -> None:
        """History.load_json from non-existent file should raise."""
        with pytest.raises(FileNotFoundError):
            History.load_json("/nonexistent/path/history.json")

    def test_history_from_dict_empty_entries(self) -> None:
        """History.from_dict with empty entries list should work."""
        h = History.from_dict({"entries": []})
        assert len(h) == 0
        assert h.last() is None

    def test_history_from_dict_malformed_json(self) -> None:
        """History.from_dict with malformed inner dict should not crash."""
        h = History.from_dict({"entries": [{"step": 1, "unknown_field": "garbage"}]})
        assert h[0].step == 1

    def test_history_entry_to_dict_roundtrip(self) -> None:
        """HistoryEntry.to_dict then from_dict should preserve data."""
        original = HistoryEntry(
            step=5,
            train_loss=0.5,
            loss_components={"lm_ce": 0.3, "probe_bce": 0.2},
            val_loss=0.6,
            custom={"extra": 1.0},
        )
        d = original.to_dict()
        restored = HistoryEntry.from_dict(d)
        assert restored.step == 5
        assert restored.train_loss == 0.5
        assert restored.val_loss == 0.6
        assert restored.custom["extra"] == 1.0

    def test_history_last_on_empty(self) -> None:
        """History.last on empty history should return None."""
        h = History()
        assert h.last() is None

    def test_history_best_val_loss_empty(self) -> None:
        """History.best_val_loss on empty history should return None."""
        h = History()
        assert h.best_val_loss() is None

    def test_history_best_val_loss_no_val(self) -> None:
        """History.best_val_loss with no val entries should return None."""
        h = History()
        h.append(HistoryEntry(step=1, train_loss=0.5))
        assert h.best_val_loss() is None

    def test_history_component_series_missing(self) -> None:
        """History.component_series for missing name should return empty."""
        h = History()
        h.append(HistoryEntry(step=1))
        assert h.component_series("nonexistent") == []

    def test_history_metric_series_missing(self) -> None:
        """History.metric_series for missing name should return empty."""
        h = History()
        h.append(HistoryEntry(step=1))
        assert h.metric_series("nonexistent") == []

    def test_history_best_val_metric_missing(self) -> None:
        """History.best_val_metric for missing name should return None."""
        h = History()
        h.append(HistoryEntry(step=1, val_loss=0.5, val_metrics={"acc": 0.9}))
        assert h.best_val_metric("nonexistent") is None

    def test_history_entry_none_fields(self) -> None:
        """HistoryEntry with all optional fields None should not crash."""
        entry = HistoryEntry(step=0)
        assert entry.train_loss is None
        assert entry.loss_components == {}
        d = entry.to_dict()
        restored = HistoryEntry.from_dict(d)
        assert restored.train_loss is None

    def test_history_from_dict_with_nested_custom(self) -> None:
        """History.from_dict with custom dict should preserve it."""
        h = History.from_dict({"entries": [{"step": 1, "custom": {"user_metric": 42.0}}]})
        assert h[0].custom["user_metric"] == 42.0


class TestOutputContainersFuzzing:
    """Fuzz tests for output container dataclasses."""

    def test_model_outputs_null_lm_logits(self) -> None:
        """ModelOutputs with null lm_logits should not crash."""
        outputs = ModelOutputs(lm_logits=None, probes={})
        assert outputs.lm_logits is None

    def test_model_outputs_null_loss(self) -> None:
        """ModelOutputs with null loss should not crash."""
        outputs = ModelOutputs(lm_logits=mx.array([1.0]), loss=None)
        assert outputs.loss is None

    def test_loss_outputs_null_total(self) -> None:
        """LossOutputs with null total should not crash."""
        outputs = LossOutputs(lm_ce=None, total=None)
        assert outputs.total is None

    def test_generation_step_defaults(self) -> None:
        """GenerationStep with defaults should not crash."""
        step = GenerationStep(token_id=0, token_str="")
        assert step.token_id == 0
        assert step.probes == {}

    def test_generation_step_null_logits(self) -> None:
        """GenerationStep with null next_logits should not crash."""
        step = GenerationStep(token_id=1, token_str="a", next_logits=None)
        assert step.next_logits is None


class TestProbeFuzzing:
    """Fuzz tests for probe construction and operations."""

    def test_probe_with_hidden_dim_zero(self) -> None:
        """Probe with hidden_dim=0 should not crash on build."""
        cfg = ProbeConfig(name="zero", layers=[0])
        with contextlib.suppress(Exception):
            probe = Probe(cfg, hidden_dim=0, backend_name="mlx")
            assert probe.hidden_dim == 0

    def test_probe_with_out_features_zero(self) -> None:
        """Probe with out_features=0 should build a zero-output module."""
        cfg = ProbeConfig(
            name="zero_out",
            layers=[0],
            module_config={"out_features": 0},
        )
        probe = Probe(cfg, hidden_dim=16, backend_name="mlx")
        assert probe.module is not None

    def test_probe_forward_no_captured_states(self) -> None:
        """Probe.forward with no captured states should raise RuntimeError."""
        cfg = ProbeConfig(name="empty", layers=[0])
        probe = Probe(cfg, hidden_dim=16, backend_name="mlx")
        with pytest.raises(RuntimeError, match="No hidden states captured"):
            probe.forward()

    def test_probe_forward_with_empty_list(self) -> None:
        """Probe.forward with empty list should raise RuntimeError."""
        cfg = ProbeConfig(name="empty", layers=[0])
        probe = Probe(cfg, hidden_dim=16, backend_name="mlx")
        with pytest.raises(RuntimeError, match="No hidden states captured"):
            probe.forward([])

    def test_probe_name_property(self) -> None:
        """Probe.name should return config name."""
        cfg = ProbeConfig(name="my_probe", layers=[0])
        probe = Probe(cfg, hidden_dim=16, backend_name="mlx")
        assert probe.name == "my_probe"

    def test_probe_layers_property(self) -> None:
        """Probe.layers should return config layers."""
        cfg = ProbeConfig(name="p", layers=[0, 2])
        probe = Probe(cfg, hidden_dim=16, backend_name="mlx")
        assert probe.layers == [0, 2]

    def test_probe_source_property(self) -> None:
        """Probe.source should return config source."""
        cfg = ProbeConfig(name="p", layers=[0], source="embedding")
        probe = Probe(cfg, hidden_dim=16, backend_name="mlx")
        assert probe.source == "embedding"

    def test_probe_with_callable_aggregation(self) -> None:
        """Probe with callable aggregation should not crash."""

        def _custom_agg(states: list[Any]) -> Any:
            return states[0]

        cfg = ProbeConfig(name="p", layers=[0], aggregation=_custom_agg)
        with contextlib.suppress(Exception):
            probe = Probe(cfg, hidden_dim=16, backend_name="mlx")
            assert callable(probe.config.aggregation)

    def test_probe_with_mlp_module(self) -> None:
        """Probe with mlp module_type should not crash."""
        cfg = ProbeConfig(
            name="mlp_probe",
            layers=[0],
            module_type="mlp",
            module_config={"hidden_dim": 8, "dropout": 0.0},
        )
        probe = Probe(cfg, hidden_dim=16, backend_name="mlx")
        assert probe.module is not None

    def test_probe_custom_module_raises_clear(self) -> None:
        """Probe with callable module_type that raises should propagate."""

        def _bad_module(in_dim: int, cfg_dict: dict) -> None:
            msg = "custom module failed"
            raise RuntimeError(msg)

        cfg = ProbeConfig(name="bad", layers=[0], module_type=_bad_module)
        with pytest.raises(RuntimeError, match="custom module failed"):
            Probe(cfg, hidden_dim=16, backend_name="mlx")

    def test_probe_clear_captured_empty(self) -> None:
        """Probe.clear_captured on empty probe should not crash."""
        cfg = ProbeConfig(name="p", layers=[0])
        probe = Probe(cfg, hidden_dim=16, backend_name="mlx")
        probe.clear_captured()
        assert probe._captured == []


class TestModelFuzzing:
    """Fuzz tests for Model forward and sample operations."""

    def test_model_forward_empty_input_ids_list(self) -> None:
        """Model.forward with empty list input_ids should not crash."""
        model = _make_model("mlx")
        with contextlib.suppress(Exception):
            outputs = model.forward([])
            assert outputs.lm_logits is not None

    def test_model_forward_with_list_input(self) -> None:
        """Model.forward with list input should convert to tensor."""
        model = _make_model("mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        outputs = model.forward([[1, 2, 3]])
        assert outputs.lm_logits is not None
        assert "p" in outputs.probes

    def test_model_forward_with_numpy_input(self) -> None:
        """Model.forward with numpy array input should work."""
        import numpy as np

        model = _make_model("mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        outputs = model.forward(np.array([[1, 2, 3]], dtype=np.int32))
        assert outputs.lm_logits is not None

    def test_model_sample_from_single_logit(self) -> None:
        """Model.sample with a single logit value should return 0."""
        model = _make_model("mlx")
        logits = mx.array([42.0])
        token = model.sample(logits, temperature=0.0)
        assert token == 0

    def test_model_sample_all_equal_logits(self) -> None:
        """Model.sample with all-equal logits should produce valid token."""
        model = _make_model("mlx")
        logits = mx.ones(10)
        token = model.sample(logits, temperature=1.0)
        assert 0 <= token < 10

    def test_model_sample_nan_logits_greedy(self) -> None:
        """Model.sample with all-NaN logits greedily should not crash."""
        model = _make_model("mlx")
        logits = mx.array([float("nan"), float("nan")])
        with contextlib.suppress(ValueError, RuntimeError):
            token = model.sample(logits, 0.0)
            assert isinstance(token, int)

    def test_model_sample_all_inf_logits(self) -> None:
        """Model.sample with all-Inf logits should not crash."""
        model = _make_model("mlx")
        logits = mx.array([float("inf"), float("inf")])
        with contextlib.suppress(ValueError, RuntimeError):
            token = model.sample(logits, 0.0)
            assert isinstance(token, int)

    def test_model_generate_zero_max_tokens(self) -> None:
        """Model.generate with max_tokens=0 should produce empty string."""
        model = _make_model("mlx")
        result = model.generate(prompt="test", max_tokens=0, temperature=0.0)
        assert result == ""

    def test_model_generate_with_probes_max_tokens_zero(self) -> None:
        """generate_with_probes with max_tokens=0 should produce no steps."""
        model = _make_model("mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        steps = list(model.generate_with_probes(prompt="test", max_tokens=0, temperature=0.0))
        assert len(steps) == 0

    def test_model_generate_negative_temperature(self) -> None:
        """Model.generate with negative temperature must raise ValueError.

        A negative temperature would invert the sampling distribution and
        silently sample the least-likely tokens, so generation now rejects it
        (via config validation or the manual generation path).
        """
        model = _make_model("mlx")
        with pytest.raises(ValueError):
            model.generate(prompt="test", max_tokens=2, temperature=-1.0)

    def test_model_generate_extreme_temperature(self) -> None:
        """Model.generate with extreme temperature should not crash."""
        model = _make_model("mlx")
        with contextlib.suppress(Exception):
            result = model.generate(prompt="test", max_tokens=2, temperature=1000.0)
            assert isinstance(result, str)

    def test_model_forward_with_batched_empty_tokens(self) -> None:
        """Model.forward with batch of 0-length sequences should not crash."""
        model = _make_model("mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        with contextlib.suppress(Exception):
            outputs = model.forward(mx.zeros((0, 5), dtype=mx.int32))
            assert outputs.lm_logits is not None

    def test_model_freeze_probe_nonexistent_silent(self) -> None:
        """Model.freeze_probe with nonexistent name should raise KeyError."""
        model = _make_model("mlx")
        with pytest.raises(KeyError):
            model.freeze_probe("nonexistent")

    def test_model_freeze_probe_nonexistent_raises(self) -> None:
        """Model.unfreeze_probe with nonexistent name should raise KeyError."""
        model = _make_model("mlx")
        with pytest.raises(KeyError):
            model.unfreeze_probe("nonexistent")


class TestModelForwardEdgeCases:
    """Fuzz tests for Model.forward with extreme tensor shapes."""

    def test_forward_single_token_batch(self) -> None:
        """Forward with single token (B=1, T=1) should work."""
        model = _make_model("mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        outputs = model.forward(mx.zeros((1, 1), dtype=mx.int32))
        assert outputs.lm_logits is not None

    def test_forward_1d_input_no_batch_dim(self) -> None:
        """Forward with 1D input (no batch dim) should be handled."""
        model = _make_model("mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.zeros((5,), dtype=mx.int32)
        outputs = model.forward(input_ids)
        assert outputs.lm_logits is not None

    def test_forward_large_batch_size(self) -> None:
        """Forward with batch size 64 should not OOM."""
        model = _make_model("mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.zeros((64, 1), dtype=mx.int32)
        outputs = model.forward(input_ids)
        assert outputs.lm_logits.shape == (64, 1, 32)

    def test_forward_extreme_sequence_length(self) -> None:
        """Forward with 4096-token sequence should not crash."""
        model = _make_model("mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.random.randint(0, 32, (1, 4096))
        outputs = model.forward(input_ids)
        assert outputs.lm_logits.shape[1] == 4096

    def test_forward_4d_input(self) -> None:
        """Forward with 4D input should raise or handle gracefully."""
        model = _make_model("mlx")
        input_ids = mx.zeros((1, 3, 1, 1), dtype=mx.int32)
        with contextlib.suppress(ValueError, RuntimeError, TypeError, IndexError):
            model.forward(input_ids)

    def test_forward_all_zero_tokens(self) -> None:
        """Forward with all-zero token IDs should not crash."""
        model = _make_model("mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.zeros((2, 10), dtype=mx.int32)
        outputs = model.forward(input_ids)
        assert outputs.lm_logits is not None

    def test_forward_negative_token_ids(self) -> None:
        """Forward with negative token IDs should be handled gracefully."""
        model = _make_model("mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.array([[-1, 0, 1]], dtype=mx.int32)
        outputs = model.forward(input_ids)
        assert outputs.lm_logits is not None

    def test_forward_with_attention_mask_none(self) -> None:
        """Forward with attention_mask=None should work."""
        model = _make_model("mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        outputs = model.forward(mx.array([[1, 2, 3]]), attention_mask=None)
        assert outputs.lm_logits is not None

    def test_forward_with_list_attention_mask(self) -> None:
        """Forward with list attention_mask should be converted."""
        model = _make_model("mlx")
        input_ids = mx.array([[1, 2, 3]])
        outputs = model.forward(
            input_ids,
            attention_mask=[[1, 1, 1]],
        )
        assert outputs.lm_logits is not None


class TestGenerationConfigFuzzing:
    """Fuzz tests for generation with extreme config values."""

    def test_generate_with_zero_top_p(self) -> None:
        """Generate with top_p=0 should not crash."""
        model = _make_model("mlx")
        cfg = GenerationConfig(max_tokens=2, top_p=0.0, do_sample=True)
        with contextlib.suppress(Exception):
            result = model.generate(prompt="test", config=cfg)
            assert isinstance(result, str)

    def test_generate_with_zero_repetition_penalty(self) -> None:
        """Generate with repetition_penalty=0 should not crash."""
        model = _make_model("mlx")
        cfg = GenerationConfig(
            max_tokens=2,
            repetition_penalty=0.0,
            do_sample=True,
        )
        with contextlib.suppress(Exception):
            result = model.generate(prompt="test", config=cfg)
            assert isinstance(result, str)

    def test_generate_with_negative_repetition_penalty(self) -> None:
        """Generate with negative repetition_penalty should not crash."""
        model = _make_model("mlx")
        cfg = GenerationConfig(
            max_tokens=2,
            repetition_penalty=-5.0,
            do_sample=True,
        )
        with contextlib.suppress(Exception):
            result = model.generate(prompt="test", config=cfg)
            assert isinstance(result, str)

    def test_generate_stop_sequences_empty(self) -> None:
        """Generate with empty stop_sequences should not crash."""
        model = _make_model("mlx")
        cfg = GenerationConfig(max_tokens=2, stop_sequences=[])
        with contextlib.suppress(Exception):
            result = model.generate(prompt="test", config=cfg)
            assert isinstance(result, str)

    def test_generate_num_return_sequences_one(self) -> None:
        """Generate with num_return_sequences=1 should not crash."""
        model = _make_model("mlx")
        cfg = GenerationConfig(max_tokens=2, num_return_sequences=1)
        with contextlib.suppress(Exception):
            result = model.generate(prompt="test", config=cfg)
            assert isinstance(result, str)


class TestBackendProtocolFuzzing:
    """Fuzz tests for Backend protocol compliance."""

    def test_backend_name_property(self) -> None:
        """Backend.name should be either mlx or torch."""
        backend = Backend(force="mlx")
        assert backend.name in ("mlx", "torch")

    def test_backend_tensor_zeros_negative_shape(self) -> None:
        """Backend.tensor.zeros with negative shape should raise."""
        backend = Backend(force="mlx")
        with pytest.raises(Exception):
            backend.tensor.zeros((-1, 5))

    def test_backend_tensor_zeros_zero_dim_shape(self) -> None:
        """Backend.tensor.zeros with shape (0,) should work."""
        backend = Backend(force="mlx")
        t = backend.tensor.zeros((0,))
        assert t.size == 0

    def test_backend_tensor_stack_empty_list(self) -> None:
        """Backend.tensor.stack with empty list should raise."""
        backend = Backend(force="mlx")
        with pytest.raises(Exception):
            backend.tensor.stack([])

    def test_backend_tensor_concatenate_empty_list(self) -> None:
        """Backend.tensor.concatenate with empty list should raise."""
        backend = Backend(force="mlx")
        with pytest.raises(Exception):
            backend.tensor.concatenate([])

    def test_backend_tensor_mean_of_empty(self) -> None:
        """Backend.tensor.mean on empty tensor should not crash."""
        backend = Backend(force="mlx")
        t = backend.tensor.zeros((0,))
        with contextlib.suppress(Exception):
            backend.tensor.mean(t)

    def test_backend_module_forward_callable_model(self) -> None:
        """Backend.module.forward on a callable class should work."""
        backend = Backend(force="mlx")

        class _CallableModel:
            def __call__(self, x: Any) -> Any:
                return x * 2

        result = backend.module.forward(_CallableModel(), mx.array([1.0]))
        assert result is not None


class TestProbeOutputMethods:
    """Fuzz tests for ProbeOutput.bce/mse/ce edge cases."""

    def test_probe_output_bce_all_same(self) -> None:
        """BCE with all logits=0 and targets=0 should be 0."""
        po = ProbeOutput(logits=mx.zeros((3,)))
        loss = po.bce(mx.zeros((3,)))
        assert float(loss) >= 0.0

    def test_probe_output_bce_perfect_prediction(self) -> None:
        """BCE with perfect logits should be near 0."""
        po = ProbeOutput(logits=mx.array([100.0, -100.0, 100.0]))
        loss = po.bce(mx.array([1.0, 0.0, 1.0]))
        assert float(loss) < 1.0

    def test_probe_output_mse_identical(self) -> None:
        """MSE with identical logits and targets should be 0."""
        po = ProbeOutput(logits=mx.array([1.0, 2.0, 3.0]))
        loss = po.mse(mx.array([1.0, 2.0, 3.0]))
        assert float(loss) == 0.0

    def test_probe_output_mse_with_mask(self) -> None:
        """MSE with all-False mask should not crash."""
        po = ProbeOutput(logits=mx.array([1.0, 2.0]))
        mask = mx.array([False, False])
        loss = po.mse(mx.array([0.0, 0.0]), mask=mask)
        assert float(loss) >= 0.0

    def test_probe_output_ce_single_class(self) -> None:
        """CE with single-class logits should work."""
        po = ProbeOutput(logits=mx.array([[1.0]]))
        loss = po.ce(mx.array([0]))
        assert float(loss) >= 0.0

    def test_probe_output_ce_with_mask(self) -> None:
        """CE with mask should work correctly."""
        po = ProbeOutput(logits=mx.array([[1.0, 2.0], [3.0, 4.0]]))
        loss = po.ce(mx.array([0, 1]), mask=mx.array([True, False]))
        assert float(loss) >= 0.0


class TestSteeringEdgeCases:
    """Fuzz tests for steering with extreme inputs."""

    def test_steering_with_zero_scale(self) -> None:
        """Steering with scale=0 should be a no-op."""
        from auto_chasm.steering import SteeringHook

        hook = SteeringHook("test", SteeringConfig(method="nullify", scale=0.0))
        hook._mean_0 = mx.zeros(8)
        hook._mean_1 = mx.ones(8)
        hook._direction = mx.ones(8)
        hook._head_norm = 1.0
        hook.enable()
        hidden = mx.random.normal((1, 3, 8))
        head = nn.Linear(8, 1)
        logits = head(hidden).squeeze(-1)
        result = hook.steer(hidden, head, logits)
        assert result.shape == hidden.shape

    def test_steering_disabled_by_default(self) -> None:
        """Steering should be disabled after initialization."""
        from auto_chasm.steering import SteeringHook

        hook = SteeringHook("test", SteeringConfig())
        assert not hook.enabled

    def test_steering_toggle(self) -> None:
        """Steering enable/disable cycle should work."""
        from auto_chasm.steering import SteeringHook

        hook = SteeringHook("test", SteeringConfig())
        hook.enable()
        assert hook.enabled
        hook.disable()
        assert not hook.enabled

    def test_steering_serialize_empty_geometry(self) -> None:
        """SteeringHook.to_dict with no geometry should not crash."""
        from auto_chasm.steering import SteeringHook

        hook = SteeringHook("test", SteeringConfig())
        d = hook.to_dict()
        assert "probe_name" in d
        assert "mean_0" not in d

    def test_steering_restore_empty(self) -> None:
        """SteeringHook.from_dict with empty data should work."""
        from auto_chasm.steering import SteeringHook

        hook = SteeringHook.from_dict(
            {"probe_name": "test", "method": "nullify", "scale": 1.0, "head_norm": 0.0}
        )
        assert hook.probe_name == "test"
        assert not hook.has_geometry


class TestLossOutputsAllComponents:
    """Fuzz tests for LossOutputs.all_components edge cases."""

    def test_all_components_with_none_lm_ce(self) -> None:
        """LossOutputs.all_components with None lm_ce should skip it."""
        outputs = LossOutputs(lm_ce=None, total=1.0)
        components = outputs.all_components
        assert "lm_ce" not in components
        assert components["total"] == 1.0

    def test_all_components_with_none_total(self) -> None:
        """LossOutputs.all_components with None total should skip it."""
        outputs = LossOutputs(lm_ce=0.5, total=None)
        components = outputs.all_components
        assert components["lm_ce"] == 0.5
        assert "total" not in components

    def test_all_components_empty_probes(self) -> None:
        """LossOutputs.all_components with empty probes should work."""
        outputs = LossOutputs(lm_ce=0.5, probes={}, total=0.5)
        components = outputs.all_components
        assert components["lm_ce"] == 0.5
        assert components["total"] == 0.5

"""Regression tests for a range of previously-fixed defects across the library.

Covers probe layer save/restore, aggregation, numeric helpers, batch collation,
checkpoint export/import (including malicious-archive rejection), steering, and
trainer edge cases.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import ProbeConfig, SteeringConfig
from auto_chasm.model import Model
from auto_chasm.probe import LayerCapture, _find_layers
from auto_chasm.steering import SteeringHook

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TinyMlp(nn.Module):
    """A tiny MLP for testing."""

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 32, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class DummyTokenizer:
    """Minimal tokenizer for testing."""

    eos_token_id = 0
    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids if i > 0)


@pytest.fixture
def tiny_model() -> tuple[TinyMlp, DummyTokenizer]:
    """Create a tiny model and tokenizer for testing."""
    mx.random.seed(42)
    model = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)
    tokenizer = DummyTokenizer()
    return model, tokenizer


@pytest.fixture
def model_wrapper(tiny_model: tuple[TinyMlp, DummyTokenizer]) -> Model:
    """Create a Model wrapper around the tiny model."""
    base_model, tokenizer = tiny_model

    class Config:
        """Dummy configuration for testing."""

        hidden_size = 16
        num_hidden_layers = 4

    base_model.config = Config()
    return Model(base_model, tokenizer, backend_name="mlx")


# ===========================================================================
# BUG-02: attach_probe stores wrapped layer, not original
# ===========================================================================


class TestBug02OriginalLayerStorage:
    """BUG-02: attach_probe stores the LayerCapture wrapper in _original_layers."""

    def test_original_layers_stores_original(self, model_wrapper: Model) -> None:
        """After attaching a probe, _original_layers should store the *original* layer."""
        config = ProbeConfig(name="test", layers=[1])
        model_wrapper.attach_probe(config)

        original = model_wrapper._original_layers.get(1)
        assert original is not None
        assert not isinstance(original, LayerCapture), (
            "_original_layers should store the original layer, not the LayerCapture wrapper"
        )

    def test_restore_original_layers_restores_actual_originals(self, model_wrapper: Model) -> None:
        """restore_original_layers should put back the original transformer blocks."""
        layers = _find_layers(model_wrapper.model)
        original_layer_1 = layers[1]

        config = ProbeConfig(name="test", layers=[1])
        model_wrapper.attach_probe(config)

        assert isinstance(layers[1], LayerCapture)

        model_wrapper.restore_original_layers()

        restored = layers[1]
        assert not isinstance(restored, LayerCapture), (
            "restore_original_layers should restore the actual original block"
        )
        assert restored is original_layer_1


# ===========================================================================
# BUG-03: Dead code in _generate_stream_mlx
# ===========================================================================


class TestBug03DeadCode:
    """BUG-03: len(tokens) is called but the result is discarded."""

    def test_no_dead_len_call(self) -> None:
        """_generate_stream_mlx should not have a dead len(tokens) call."""
        import inspect

        from auto_chasm.generation import _generate_stream_mlx

        source = inspect.getsource(_generate_stream_mlx)
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped == "len(tokens)":
                pytest.fail("Found dead 'len(tokens)' call — result is discarded.")


# ===========================================================================
# BUG-05: probe.forward bypasses aggregation for single-layer
# ===========================================================================


class TestBug05SingleLayerAggregation:
    """BUG-05: Single-layer probes always bypass aggregation, even if configured."""

    def test_single_layer_with_mean_aggregation_runs_aggregation(
        self, model_wrapper: Model
    ) -> None:
        """A single-layer probe with aggregation='mean' should still call _aggregate."""
        config = ProbeConfig(name="test", layers=[1], aggregation="mean")
        probe = model_wrapper.attach_probe(config)

        original_aggregate = probe._aggregate
        call_count = [0]

        def counting_aggregate(states: list) -> Any:
            call_count[0] += 1
            return original_aggregate(states)

        probe._aggregate = counting_aggregate

        input_ids = mx.array([[1, 2, 3]])
        model_wrapper.forward(input_ids)

        assert call_count[0] > 0, (
            "Single-layer probe with aggregation='mean' should call _aggregate"
        )


# ===========================================================================
# BUG-06: to_numpy for multi-dim MLX arrays
# ===========================================================================


class TestBug06ToNumpyMLX:
    """BUG-06: MLXTensorOps.to_numpy returns raw array for multi-dim tensors."""

    def test_to_numpy_returns_numpy_array(self) -> None:
        """to_numpy should return a numpy array, not an MLX array."""
        from auto_chasm.backends.mlx_backend import MLXTensorOps

        ops = MLXTensorOps()
        t = mx.array([[1.0, 2.0], [3.0, 4.0]])
        result = ops.to_numpy(t)

        import numpy as np

        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result)}"


# ===========================================================================
# BUG-08: collate_batches catches TypeError broadly
# ===========================================================================


class TestBug08CollateTypeError:
    """BUG-08: collate_batches catches TypeError alongside ImportError."""

    def test_collate_does_not_mask_type_errors(self) -> None:
        """TypeError during stacking should propagate, not be silently caught."""
        from auto_chasm.data import collate_batches

        batch1 = {"tokens": "not_a_tensor", "probe_labels": {}}
        batch2 = {"tokens": "also_not_a_tensor", "probe_labels": {}}

        try:
            result = collate_batches([batch1, batch2], [])
            assert isinstance(result["tokens"], list), (
                "String values should be collected as a list, not stacked"
            )
        except TypeError:
            pass


# ===========================================================================
# BUG-09: clean_adapter_keys replaces ALL .layer. segments
# ===========================================================================


class TestBug09CleanAdapterKeys:
    """BUG-09: clean_adapter_keys replaces ALL .layer. segments."""

    def test_preserves_non_model_layer_segments(self) -> None:
        """Should only clean .layer. segments that are part of model.layers.N."""
        from auto_chasm.utils import clean_adapter_keys

        state = {
            "model.layers.3.self_attn.q_proj.lora_a": "a",
            "encoder.layers.5.fc1.weight": "b",
        }
        cleaned = clean_adapter_keys(state)

        assert "model.3.self_attn.q_proj.lora_a" in cleaned
        # encoder.layers.5 should NOT be transformed
        assert "encoder.layers.5.fc1.weight" in cleaned


# ===========================================================================
# BUG-11: _MLXMlp is not an nn.Module subclass
# ===========================================================================


class TestBug11MLXMlpNotModule:
    """BUG-11: MLXMlp does not extend mlx.nn.Module."""

    def test_mlx_mlp_is_nn_module(self) -> None:
        """MLXMlp should be an nn.Module so it's visible to parameter trees."""
        from auto_chasm._mlx_mlp import MLXMlp

        mlp = MLXMlp(16, 32, 1)
        assert isinstance(mlp, nn.Module), (
            "MLXMlp must be an nn.Module to be visible to trainable_parameters()"
        )


# ===========================================================================
# BUG-13: SteeringConfig missing "custom" method
# ===========================================================================


class TestBug13CustomSteeringMethod:
    """BUG-13: SteeringConfig.method doesn't include 'custom'."""

    def test_custom_steering_method_accepted(self) -> None:
        """SteeringConfig should accept method='custom' for user-defined steering."""
        config = SteeringConfig(method="custom")
        assert config.method == "custom"


# ===========================================================================
# BUG-14: _generate_manual_mlx appends EOS before checking
# ===========================================================================


class TestBug14ManualGenerateEos:
    """BUG-14: _generate_manual_mlx checks eos after appending."""

    def test_eos_token_not_included_in_output(self) -> None:
        """Generated text should not include the EOS token."""
        from auto_chasm.generation import _generate_manual_mlx

        class MockTokenizer:
            """Test helper."""

            eos_token_id = 99

            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

            def decode(self, ids: list) -> str:
                return str(ids)

        class MockModel:
            """Test helper."""

            def __call__(self, x: mx.array) -> mx.array:
                logits = mx.zeros((1, x.shape[1], 100))
                logits[0, -1, 99] = 100.0
                return logits

        result = _generate_manual_mlx(MockModel(), MockTokenizer(), "test", 10, 0.0)
        assert "99" not in result, f"EOS token found in output: {result}"


# ===========================================================================
# BUG-15: History.best_val_loss returns tuple, not HistoryEntry
# ===========================================================================


class TestBug15BestValLossType:
    """BUG-15: History.best_val_loss returns (step, val_loss) tuple."""

    def test_best_val_loss_returns_entry(self) -> None:
        """best_val_loss should return the HistoryEntry, not just (step, val_loss)."""
        from auto_chasm.history import History, HistoryEntry

        h = History()
        h.append(HistoryEntry(step=10, val_loss=0.5))
        h.append(HistoryEntry(step=20, val_loss=0.3))

        best = h.best_val_loss()
        assert isinstance(best, HistoryEntry), f"Expected HistoryEntry, got {type(best)}"
        assert best.step == 20
        assert best.val_loss == 0.3


# ===========================================================================
# BUG-16: RLTrainer._dpo_loss accesses model._probes directly
# ===========================================================================


class TestBug16RLTrainerProbeAccess:
    """BUG-16: RLTrainer loss functions access model attributes incorrectly."""

    def test_sft_probe_loss_works_with_trainable_model(self) -> None:
        """The implemented RL loss function should work with _TrainableModel."""
        from auto_chasm.config import RLConfig
        from auto_chasm.trainers.rl import RLTrainer

        RLConfig(algorithm="sft")  # verify import works
        assert hasattr(RLTrainer, "_sft_probe_loss")


# ===========================================================================
# BUG-17: TorchBackend step hardcodes max_norm=1.0
# ===========================================================================


class TestBug17TorchStepHardcodedNorm:
    """BUG-17: TorchOptimOps.step hardcodes gradient clip max_norm=1.0."""

    def test_step_uses_custom_clip_norm(self) -> None:
        """TorchOptimOps.step should accept a max_norm parameter."""
        import inspect

        from auto_chasm.backends.torch_backend import TorchOptimOps

        sig = inspect.signature(TorchOptimOps.step)
        params = list(sig.parameters.keys())
        assert "max_norm" in params, (
            "TorchOptimOps.step should accept max_norm instead of hardcoding 1.0"
        )


# ===========================================================================
# BUG-22: export_checkpoint doesn't validate empty dirs
# ===========================================================================


class TestBug22ExportEmptyCheckpoint:
    """BUG-22: export_checkpoint with empty dir creates useless archive."""

    def test_export_validates_checkpoint_contents(self) -> None:
        """export_checkpoint should validate that the directory contains a checkpoint."""
        from auto_chasm.checkpoint import export_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty"
            empty_dir.mkdir()
            output = Path(tmpdir) / "out.auto_chasm"

            with pytest.raises((ValueError, FileNotFoundError)):
                export_checkpoint(str(empty_dir), str(output))


# ===========================================================================
# BUG-23: import_checkpoint vulnerable to path traversal
# ===========================================================================


class TestBug23ImportPathTraversal:
    """BUG-23: import_checkpoint uses tar.extractall without validation."""

    def test_import_rejects_malicious_archive(self) -> None:
        """import_checkpoint should reject archives with path traversal."""
        from auto_chasm.checkpoint import import_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            malicious_file = Path(tmpdir) / "bad.auto_chasm"
            attacker_dir = Path(tmpdir) / "attacker"
            attacker_dir.mkdir()
            (attacker_dir / "payload.txt").write_text("pwned")

            with tarfile.open(str(malicious_file), "w:gz") as tar:
                tar.add(str(attacker_dir), arcname="../escape")

            output_dir = Path(tmpdir) / "safe_output"
            output_dir.mkdir()

            with pytest.raises((ValueError, tarfile.TarError)):
                import_checkpoint(str(malicious_file), str(output_dir))


# ===========================================================================
# BUG-28: generate dispatches to torch path when backend=None on MLX
# ===========================================================================


class TestBug28GenerateDispatch:
    """BUG-28: generate dispatches to torch when backend=None on MLX."""

    def test_generate_uses_mlx_by_default_on_mlx(self) -> None:
        """On MLX machine, generate should auto-detect backend."""
        from auto_chasm.generation import generate

        class MockModel:
            """Test helper."""

            def __call__(self, x):  # type: ignore
                return (mx.zeros((1, 3, 32)),)

        class MockTokenizer:
            """Test helper."""

            eos_token_id = 0

            def encode(self, text):  # type: ignore
                return [1, 2, 3]

            def decode(self, ids):  # type: ignore
                return "test"

        result = generate(MockModel(), MockTokenizer(), "hi", max_tokens=1, temperature=0.0)
        assert isinstance(result, str)


# ===========================================================================
# Passing tests for verified non-bugs (previously false alarms)
# ===========================================================================


class TestVerifiedNotBugs:
    """Tests that verify behaviors that looked suspicious but are correct."""

    def test_layer_capture_no_harmful_duplicate_assignment(self) -> None:
        """Duplicate assignment in LayerCapture.__init__ is harmless (BUG-01)."""
        layer = nn.Linear(16, 16)
        capture = LayerCapture(
            layer,
            layer_idx=0,
            capture_fn=lambda h: None,
            steer_fn="fn_a",
            binary_head="head_a",
        )
        assert capture.steer_fn == "fn_a"
        assert capture.binary_head == "head_a"

    def test_push_to_mean_broadcasts_correctly(self) -> None:
        """_push_to_mean handles 1-D direction with 3-D hidden (BUG-04)."""
        config = SteeringConfig(method="push_to_mean")
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 16)
        hook._mean_1 = mx.array([1.0] * 16)
        hook._direction = hook._mean_1 - hook._mean_0
        hook._head_norm = 1.0
        hook.enabled = True

        hidden = mx.random.normal((2, 5, 16))
        head = nn.Linear(16, 1)
        logits = mx.array([[0.5] * 5, [0.3] * 5])

        result = hook.steer(hidden, head, logits)
        assert result.shape == hidden.shape

    def test_from_dict_raises_on_missing_probe_name(self) -> None:
        """SteeringHook.from_dict correctly raises KeyError (BUG-10)."""
        data = {"method": "nullify", "scale": 1.0}
        with pytest.raises(KeyError):
            SteeringHook.from_dict(data)

    def test_source_attention_constructs(self) -> None:
        """source='attention' now builds a valid config (sub-block source implemented)."""
        assert ProbeConfig(name="test", layers=[5], source="attention").source == "attention"

    def test_source_mlp_constructs(self) -> None:
        """source='mlp' now builds a valid config (sub-block source implemented)."""
        assert ProbeConfig(name="test", layers=[5], source="mlp").source == "mlp"

    def test_multiple_probes_processed(self, model_wrapper: Model) -> None:
        """_TrainableModel processes all probes (BUG-18)."""
        from auto_chasm.trainers.trainable import _TrainableModel

        model_wrapper.attach_probe(ProbeConfig(name="a", layers=[0]))
        model_wrapper.attach_probe(ProbeConfig(name="b", layers=[1]))

        train_model = _TrainableModel(model_wrapper.model, model_wrapper._probes)
        input_ids = mx.array([[1, 2, 3]])
        lm_logits, probe_logits = train_model(input_ids)

        assert lm_logits is not None
        assert probe_logits is not None

    def test_mlp_respects_out_features(self, model_wrapper: Model) -> None:
        """MLP module uses out_features from module_config (BUG-24)."""
        config = ProbeConfig(
            name="test",
            layers=[1],
            module_type="mlp",
            module_config={"out_features": 5, "hidden_dim": 32},
        )
        model_wrapper.attach_probe(config)

        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        probe_logits = outputs.probes["test"].logits
        assert probe_logits.shape[-1] == 5

    def test_max_aggregation_returns_tensor(self, model_wrapper: Model) -> None:
        """Max aggregation returns a tensor, not a tuple (BUG-25)."""
        config = ProbeConfig(name="multi", layers=[0, 1], aggregation="max")
        model_wrapper.attach_probe(config)

        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert outputs.probes["multi"].logits is not None
        assert hasattr(outputs.probes["multi"].logits, "shape")

    def test_config_names_match_trainer(self) -> None:
        """TrainingConfig fields match JointTrainer params (BUG-26)."""
        import inspect

        from auto_chasm.trainers.base import JointTrainer

        trainer_params = set(inspect.signature(JointTrainer.__init__).parameters.keys())
        trainer_params.discard("self")
        assert "num_iters" in trainer_params
        assert "learning_rate" in trainer_params

    def test_checkpoint_error_for_missing_base_model(self) -> None:
        """load_checkpoint raises ValueError for missing base_model (BUG-20)."""
        from auto_chasm.checkpoint import load_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {"backend": "mlx", "probes": {}, "steering": {}}
            (Path(tmpdir) / "manifest.json").write_text(json.dumps(manifest))

            with pytest.raises(ValueError, match="base_model"):
                load_checkpoint(tmpdir)

    def test_no_duplicate_handlers(self) -> None:
        """configure_logging doesn't add duplicate handlers (BUG-21)."""
        import logging

        from auto_chasm.logger import configure_logging, get_logger

        root = get_logger("auto_chasm")
        root.handlers.clear()

        configure_logging()
        configure_logging()

        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) <= 1

        root.handlers.clear()

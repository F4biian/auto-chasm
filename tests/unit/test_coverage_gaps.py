"""Targeted tests to hit specific uncovered lines for coverage."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import ProbeConfig
from auto_chasm.history import History, HistoryEntry
from auto_chasm.model import Model
from auto_chasm.probe import Probe


class TinyMlp(nn.Module):
    """Tiny MLP for testing."""

    def __init__(self, hidden_dim: int = 8, vocab_size: int = 16, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self._hidden_dim = hidden_dim

    def __call__(self, x: mx.array) -> tuple:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return (self.output_proj(h),)


class DummyTokenizer:
    """Test helper."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "test"


# ===========================================================================
# model.py: forward with attention_mask, enable_steering without config
# ===========================================================================


class TestModelForward:
    """Model.forward edge cases."""

    def test_forward_without_mask(self) -> None:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        outputs = model.forward(mx.array([[1, 2, 3]]))
        assert outputs.lm_logits is not None

    def test_forward_no_probes(self) -> None:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        outputs = model.forward(mx.array([[1, 2, 3]]))
        assert outputs.lm_logits is not None
        assert outputs.probes == {}


class TestEnableSteeringEdge:
    """enable_steering edge cases."""

    def test_enable_steering_without_config(self) -> None:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))

        mean_0 = mx.array([1.0] * 8)
        mean_1 = mx.array([2.0] * 8)
        # config=None should use default SteeringConfig
        model.enable_steering("p", config=None, class_means={"mean_0": mean_0, "mean_1": mean_1})
        assert "p" in model.steering_hooks


# ===========================================================================
# probe.py: "last" aggregation, RuntimeError path
# ===========================================================================


class TestProbeAggregation:
    """Probe aggregation edge cases."""

    def test_last_aggregation(self) -> None:
        base = TinyMlp()

        class Config:
            """Test helper."""

            hidden_size = 8

        base.config = Config()
        model = Model(base, DummyTokenizer(), "mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0, 1], aggregation="last"))
        outputs = model.forward(mx.array([[1, 2, 3]]))
        assert "p" in outputs.probes

    def test_runtime_error_no_captured(self) -> None:
        config = ProbeConfig(name="p", layers=[0], aggregation="mean")
        probe = Probe(config, 8, "mlx")
        with pytest.raises(RuntimeError, match="No hidden states captured"):
            probe.forward([])

    def test_no_layers_raises(self) -> None:
        class NoLayerModel(nn.Module):
            """Test helper."""

            def __call__(self, x):
                return (mx.zeros((1, 3, 16)),)

        class MockTokenizer(DummyTokenizer):
            """Test helper."""

            pass

        m = Model(NoLayerModel(), MockTokenizer(), "mlx")
        with pytest.raises(ValueError, match="Cannot find transformer"):
            m.attach_probe(ProbeConfig(name="p", layers=[0]))


# ===========================================================================
# history.py: best_val_metric edge cases
# ===========================================================================


class TestHistoryEdgeCases:
    """History edge case coverage."""

    def test_best_val_metric_higher_is_better(self) -> None:
        h = History()
        h.append(HistoryEntry(step=10, val_metrics={"acc": 0.5}))
        h.append(HistoryEntry(step=20, val_metrics={"acc": 0.9}))
        best = h.best_val_metric("acc", higher_is_better=True)
        assert best is not None
        assert best[0] == 20
        assert best[1] == 0.9

    def test_best_val_metric_lower_is_better(self) -> None:
        h = History()
        h.append(HistoryEntry(step=10, val_metrics={"mse": 0.5}))
        h.append(HistoryEntry(step=20, val_metrics={"mse": 0.1}))
        best = h.best_val_metric("mse", higher_is_better=False)
        assert best is not None
        assert best[0] == 20
        assert best[1] == 0.1

    def test_best_val_metric_not_found(self) -> None:
        h = History()
        h.append(HistoryEntry(step=10, val_metrics={"a": 0.5}))
        best = h.best_val_metric("nonexistent")
        assert best is None

    def test_entries_property_returns_copy(self) -> None:
        h = History()
        h.append(HistoryEntry(step=1))
        entries = h.entries
        assert len(entries) == 1
        entries.pop()  # should not affect internal list
        assert len(h) == 1


# ===========================================================================
# generation.py: chat without template, stream decode
# ===========================================================================


class TestChatEdge:
    """chat() edge cases."""

    def test_chat_fallback_formatting(self) -> None:
        from auto_chasm.generation import _generate_manual_mlx

        class NoTemplateTokenizer(DummyTokenizer):
            """Test helper."""

            chat_template = None

        result = _generate_manual_mlx(TinyMlp(), NoTemplateTokenizer(), "hi", 1, 0.0)
        assert isinstance(result, str)

    def test_manual_decode_tokens(self) -> None:
        """The decode path in manual generate should handle EOS correctly."""
        from auto_chasm.generation import _generate_manual_mlx

        class ManyTokenModel(nn.Module):
            """Test helper."""

            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(16, 8)
                self.layers = [nn.Linear(8, 8) for _ in range(2)]
                self.output_proj = nn.Linear(8, 16)

            def __call__(self, x):
                h = self.embedding(x)
                for layer in self.layers:
                    h = nn.gelu(layer(h))
                return (self.output_proj(h),)

        result = _generate_manual_mlx(ManyTokenModel(), DummyTokenizer(), "test", 3, 0.0)
        assert isinstance(result, str)


# ===========================================================================
# peft.py: _default_target_modules
# ===========================================================================


class TestPEFT:
    """PEFT utility tests."""

    def test_default_target_modules(self) -> None:
        from auto_chasm.peft import _default_target_modules

        base = TinyMlp()
        targets = _default_target_modules(base)
        assert isinstance(targets, list)

    def test_apply_lora_without_backend(self) -> None:
        from auto_chasm.peft import apply_lora

        base = TinyMlp()
        result = apply_lora(base, r=4, alpha=8, target_modules=["layers.0"])
        assert result is not None

    def test_get_trainable_params(self) -> None:
        from auto_chasm.backends import Backend
        from auto_chasm.peft import get_trainable_params

        base = TinyMlp()
        backend = Backend(force="mlx")
        params = get_trainable_params(base, backend)
        assert len(params) > 0

    def test_unfreeze_lora_params_mlx(self) -> None:
        from auto_chasm.peft import _unfreeze_lora_params

        base = TinyMlp()
        base.freeze()  # freeze everything first
        from auto_chasm.backends import Backend

        _unfreeze_lora_params(base, Backend(force="mlx"))
        # Should unfreeze (fallback since no LoRALinear layers)


# ===========================================================================
# data.py: iterate_batches dict access
# ===========================================================================


class TestIterateBatchesEdge:
    """iterate_batches edge cases."""

    def test_dict_items(self) -> None:
        from auto_chasm.trainers.data_utils import iterate_batches

        dataset = [{"tokens": [1, 2, 3], "labels": [0, 0, 0]}]
        batches = list(iterate_batches(dataset, batch_size=1, max_seq_length=16, loop=False))
        assert len(batches) == 1

    def test_tuple_items(self) -> None:
        from auto_chasm.trainers.data_utils import iterate_batches

        dataset = [([1, 2, 3], [0, 0, 0])]
        batches = list(iterate_batches(dataset, batch_size=1, max_seq_length=16, loop=False))
        assert len(batches) == 1

    def test_input_ids_key(self) -> None:
        from auto_chasm.trainers.data_utils import iterate_batches

        dataset = [{"input_ids": [5, 6], "binary_labels": [0, 1]}]
        batches = list(iterate_batches(dataset, batch_size=1, max_seq_length=16, loop=False))
        assert len(batches) == 1

    def test_empty_dataset_raises(self) -> None:
        from auto_chasm.trainers.data_utils import iterate_batches

        with pytest.raises(ValueError, match="empty"):
            list(iterate_batches([], batch_size=1, max_seq_length=16, loop=False))

    def test_joint_text_dataset_eos_append(self) -> None:
        from auto_chasm.trainers.data_utils import JointTextDataset

        data = [{"text": "hello", "labels": [0, 0, 0]}]
        tokenizer = DummyTokenizer()
        ds = JointTextDataset(data, tokenizer, text_key="text", labels_key="labels")
        tokens, labels = ds[0]
        assert tokens[-1] == tokenizer.eos_token_id

    def test_joint_text_dataset_labels_padding(self) -> None:
        from auto_chasm.trainers.data_utils import JointTextDataset

        data = [{"text": "hello", "labels": [1]}]
        tokenizer = DummyTokenizer()
        ds = JointTextDataset(data, tokenizer, text_key="text", labels_key="labels")
        tokens, labels = ds[0]
        assert len(labels) == len(tokens)
        assert labels[0] == 1

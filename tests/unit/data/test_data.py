"""Tests for auto_chasm.data — collation, linking, JointDataset."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from auto_chasm.config import ProbeConfig
from auto_chasm.data import JointDataset, collate_batches, link_columns, span_labels_to_tokens


class MockTransformersTokenizer:
    """Mock transformers-style tokenizer that returns offset mapping."""

    def __init__(self, offset_mapping: list[tuple[int, int]]) -> None:
        self._offset_mapping = offset_mapping

    def __call__(self, text: str, return_offsets_mapping: bool = False) -> Any:
        class Encoding:
            """Mock tokenizer encoding with offset mapping."""

            def __init__(self, om: list[tuple[int, int]]) -> None:
                self.offset_mapping = om

        return Encoding(self._offset_mapping)


class MockMLXTokenizer:
    """Mock MLX-style tokenizer with only an encode method."""

    def __init__(self, num_tokens: int) -> None:
        self._num_tokens = num_tokens
        self.eos_token_id = -1

    def encode(self, text: str) -> list[int]:
        return list(range(self._num_tokens))


class TestLinkColumns:
    """Tests for link_columns utility."""

    def test_links_by_convention(self) -> None:
        batch = {"hallucination_labels": [0, 0, 1], "text": "hello"}
        probes = [ProbeConfig(name="hallucination", layers=[0])]
        result = link_columns(batch, probes)
        assert "probe_labels" in result
        assert result["probe_labels"]["hallucination"] == [0, 0, 1]

    def test_links_with_custom_map(self) -> None:
        batch = {"custom_col": [1, 0, 1]}
        probes = [ProbeConfig(name="probe_a", layers=[0])]
        result = link_columns(batch, probes, column_map={"probe_a": "custom_col"})
        assert result["probe_labels"]["probe_a"] == [1, 0, 1]

    def test_missing_column_silent(self) -> None:
        batch = {"text": "hello"}
        probes = [ProbeConfig(name="missing_probe", layers=[0])]
        result = link_columns(batch, probes)
        assert result["probe_labels"] == {}

    def test_multiple_probes(self) -> None:
        batch = {"a_labels": [0, 0], "b_labels": [1, 1]}
        probes = [
            ProbeConfig(name="a", layers=[0]),
            ProbeConfig(name="b", layers=[1]),
        ]
        result = link_columns(batch, probes)
        assert result["probe_labels"]["a"] == [0, 0]
        assert result["probe_labels"]["b"] == [1, 1]

    def test_empty_probes(self) -> None:
        batch = {"text": "hello"}
        result = link_columns(batch, [])
        assert result["probe_labels"] == {}


class TestJointDataset:
    """Tests for JointDataset wrapper."""

    @pytest.fixture
    def sample_data(self) -> list[dict]:
        return [
            {"text": "hello", "p_labels": [0, 0, 1]},
            {"text": "world", "p_labels": [0, 1]},
        ]

    def test_length(self, sample_data: list) -> None:
        ds = JointDataset(sample_data, [ProbeConfig(name="p", layers=[0])])
        assert len(ds) == 2

    def test_getitem_links(self, sample_data: list) -> None:
        ds = JointDataset(sample_data, [ProbeConfig(name="p", layers=[0])])
        item = ds[0]
        assert "probe_labels" in item
        assert item["probe_labels"]["p"] == [0, 0, 1]

    def test_iteration(self, sample_data: list) -> None:
        ds = JointDataset(sample_data, [ProbeConfig(name="p", layers=[0])])
        items = list(ds)
        assert len(items) == 2

    def test_custom_column_map(self, sample_data: list) -> None:
        ds = JointDataset(
            sample_data,
            [ProbeConfig(name="my_probe", layers=[0])],
            column_map={"my_probe": "p_labels"},
        )
        item = ds[0]
        assert item["probe_labels"]["my_probe"] == [0, 0, 1]


class TestCollateBatches:
    """Tests for collate_batches utility."""

    def test_collates_tensors(self) -> None:
        batch1 = {"input_ids": mx.array([1, 2, 3]), "probe_labels": {}}
        batch2 = {"input_ids": mx.array([4, 5, 6]), "probe_labels": {}}
        result = collate_batches([batch1, batch2], [])
        assert result["input_ids"].shape == (2, 3)

    def test_collates_probe_labels(self) -> None:
        batch1 = {"input_ids": mx.array([1]), "probe_labels": {"p": mx.array([0])}}
        batch2 = {"input_ids": mx.array([2]), "probe_labels": {"p": mx.array([1])}}
        result = collate_batches([batch1, batch2], [])
        assert "probe_labels" in result
        assert result["probe_labels"]["p"].shape == (2, 1)

    def test_empty_batches(self) -> None:
        result = collate_batches([], [])
        assert result == {}

    def test_mixed_probe_labels(self) -> None:
        batch1 = {"input_ids": mx.array([1]), "probe_labels": {"a": mx.array([0])}}
        batch2 = {"input_ids": mx.array([2]), "probe_labels": {"b": mx.array([1])}}
        result = collate_batches([batch1, batch2], [])
        assert "a" in result["probe_labels"]
        assert "b" in result["probe_labels"]

    def test_preserves_probe_labels_key(self) -> None:
        batch1 = {"input_ids": mx.array([1]), "probe_labels": {"p": mx.array([0])}}
        batch2 = {"input_ids": mx.array([2]), "probe_labels": {"p": mx.array([1])}}
        result = collate_batches([batch1, batch2], [])
        assert "probe_labels" in result
        # Only probe_labels key is present, no duplicate "probe_labels" raw key
        for key in result:
            assert key == "input_ids" or key == "probe_labels"


class TestSpanLabelsToTokens:
    """Tests for span_labels_to_tokens utility."""

    def test_basic_transformers_tokenizer(self) -> None:
        tokenizer = MockTransformersTokenizer([(0, 4), (5, 9), (10, 15)])
        spans = [{"start": 0, "end": 9, "label": 1}]
        # Default masks unmarked tokens (-100): only the first two overlap.
        result = span_labels_to_tokens("hello world test", tokenizer, spans)
        assert result == [1, 1, -100]

    def test_default_label_zero_fills_negatives(self) -> None:
        # Opt-in default_label=0 reproduces "unmarked = negative class 0".
        tokenizer = MockTransformersTokenizer([(0, 4), (5, 9), (10, 15)])
        spans = [{"start": 0, "end": 9, "label": 1}]
        result = span_labels_to_tokens("hello world test", tokenizer, spans, default_label=0)
        assert result == [1, 1, 0]

    def test_basic_mlx_tokenizer(self) -> None:
        tokenizer = MockMLXTokenizer(num_tokens=5)
        spans = [{"start": 0, "end": 5, "label": 1}]
        result = span_labels_to_tokens("hello", tokenizer, spans)
        assert len(result) == 5
        assert result[0] == 1

    def test_aggregation_max(self) -> None:
        tokenizer = MockTransformersTokenizer([(0, 10)])
        spans = [
            {"start": 0, "end": 5, "label": 0},
            {"start": 3, "end": 10, "label": 1},
        ]
        result = span_labels_to_tokens("hellohello", tokenizer, spans, aggregation="max")
        assert result == [1]

    def test_aggregation_min(self) -> None:
        tokenizer = MockTransformersTokenizer([(0, 10)])
        spans = [
            {"start": 0, "end": 5, "label": 0},
            {"start": 3, "end": 10, "label": 1},
        ]
        result = span_labels_to_tokens("hellohello", tokenizer, spans, aggregation="min")
        assert result == [0]

    def test_aggregation_mean(self) -> None:
        tokenizer = MockTransformersTokenizer([(0, 10)])
        spans = [
            {"start": 0, "end": 5, "label": 0},
            {"start": 3, "end": 10, "label": 1},
        ]
        result = span_labels_to_tokens("hellohello", tokenizer, spans, aggregation="mean")
        # mean is no longer rounded to int — float (soft/regression) labels survive.
        assert result == [0.5]

    def test_aggregation_callable(self) -> None:
        tokenizer = MockTransformersTokenizer([(0, 10)])
        spans = [
            {"start": 0, "end": 5, "label": 2},
            {"start": 3, "end": 10, "label": 4},
        ]
        result = span_labels_to_tokens(
            "hellohello", tokenizer, spans, aggregation=lambda x: sum(x) * 10
        )
        assert result == [60]

    def test_default_label_no_overlap(self) -> None:
        tokenizer = MockTransformersTokenizer([(0, 5), (5, 10)])
        spans = [{"start": 10, "end": 15, "label": 1}]
        result = span_labels_to_tokens("hello world", tokenizer, spans, default_label=99)
        assert result == [99, 99]

    def test_empty_spans(self) -> None:
        # No spans → every token masked (-100) by default.
        tokenizer = MockTransformersTokenizer([(0, 5)])
        result = span_labels_to_tokens("hello", tokenizer, [])
        assert result == [-100]

    def test_empty_spans_with_default_label(self) -> None:
        # With default_label=0, no spans → every token is the negative class 0.
        tokenizer = MockTransformersTokenizer([(0, 5)])
        result = span_labels_to_tokens("hello", tokenizer, [], default_label=0)
        assert result == [0]

    def test_mlx_empty_text(self) -> None:
        tokenizer = MockMLXTokenizer(num_tokens=0)
        result = span_labels_to_tokens("", tokenizer, [])
        assert result == []


class TestRouteProbeLabels:
    """Tests for route_probe_labels utility."""

    def test_routes_matching_columns(self) -> None:
        from auto_chasm.trainers.data_utils import route_probe_labels

        sample = {
            "text": "hello",
            "tokens": [1, 2, 3],
            "digit_labels": [0, 0, 1],
            "hallucination_labels": [0, 1, 0],
        }
        result = route_probe_labels(sample, ["digit", "hallucination"])
        assert result["digit"] == [0, 0, 1]
        assert result["hallucination"] == [0, 1, 0]

    def test_missing_column_returns_empty(self) -> None:
        from auto_chasm.trainers.data_utils import route_probe_labels

        sample = {"text": "hello", "tokens": [1, 2, 3]}
        result = route_probe_labels(sample, ["digit"])
        assert result == {}

    def test_empty_probe_names(self) -> None:
        from auto_chasm.trainers.data_utils import route_probe_labels

        sample = {"text": "hello", "digit_labels": [1, 0]}
        result = route_probe_labels(sample, [])
        assert result == {}

    def test_some_missing_some_present(self) -> None:
        from auto_chasm.trainers.data_utils import route_probe_labels

        sample = {"a_labels": [0, 1], "text": "hi"}
        result = route_probe_labels(sample, ["a", "b"])
        assert result["a"] == [0, 1]
        assert "b" not in result


class TestJointTextDatasetWithProbeNames:
    """Tests for JointTextDataset with probe_names auto-routing."""

    def test_probe_names_stored(self) -> None:
        from auto_chasm.trainers.data_utils import JointTextDataset

        data = [{"text": "hello", "digit_labels": [0, 0], "labels": [0, 0]}]
        tokenizer = MockMLXTokenizer(num_tokens=3)
        ds = JointTextDataset(data, tokenizer, probe_names=["digit"])
        assert ds.probe_names == ["digit"]

    def test_probe_names_none_by_default(self) -> None:
        from auto_chasm.trainers.data_utils import JointTextDataset

        data = [{"text": "hello", "labels": [0, 0]}]
        tokenizer = MockMLXTokenizer(num_tokens=3)
        ds = JointTextDataset(data, tokenizer)
        assert ds.probe_names is None

    def test_backward_compat_no_probe_names(self) -> None:
        from auto_chasm.trainers.data_utils import JointTextDataset

        data = [{"text": "hello", "labels": [0, 0, 1]}]
        tokenizer = MockMLXTokenizer(num_tokens=3)
        ds = JointTextDataset(data, tokenizer, labels_key="labels")
        tokens, labels = ds[0]
        assert len(tokens) > 0
        assert len(labels) > 0


class TestSpanLabelsEdgeCases:
    """Edge cases for span_labels_to_tokens."""

    def test_unknown_aggregation_raises(self) -> None:
        """Unknown aggregation string raises ValueError."""
        from auto_chasm.data import span_labels_to_tokens

        class Tok:
            """Simple tokenizer."""

            eos_token_id = 0

            def encode(self, text: str) -> list[int]:
                return [1]

        # Must provide overlapping spans to reach the aggregation logic
        spans = [{"start": 0, "end": 5, "label": 1}]
        with pytest.raises(ValueError):
            span_labels_to_tokens("hello", Tok(), spans, aggregation="invalid")

    def test_offset_mapping_none_fallback(self) -> None:
        """When return_offsets_mapping is None, fallback to char heuristic."""
        from auto_chasm.data import span_labels_to_tokens

        class NoOffsetTokenizer:
            """Tokenizer that doesn't support offset_mapping."""

            eos_token_id = 0

            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

            def decode(self, ids: list[int]) -> str:
                return "abc"

            def __call__(
                self, text: str, return_offsets_mapping: bool = False, **kwargs: object
            ) -> object:
                class Encoding:
                    """Mock encoding."""

                    offset_mapping = None

                return Encoding()

        labels = span_labels_to_tokens("abc", NoOffsetTokenizer(), [], default_label=0)
        assert len(labels) == 3
        assert all(l == 0 for l in labels)

    def test_empty_text_returns_empty(self) -> None:
        """Empty text returns empty list."""
        from auto_chasm.data import span_labels_to_tokens

        class EmptyTokenizer:
            """Tokenizer that produces no tokens."""

            eos_token_id = 0

            def encode(self, text: str) -> list[int]:
                return []

        labels = span_labels_to_tokens("", EmptyTokenizer(), [], default_label=0)
        assert labels == []


class TestDataTorchCollation:
    """Torch backend paths for collate_batches."""

    def test_collate_torch_tensors(self) -> None:
        """collate_batches stacks MLX arrays correctly."""
        import mlx.core as mx

        from auto_chasm.data import collate_batches

        batches = [
            {"tokens": mx.array([1, 2, 3]), "probe_labels": {}},
            {"tokens": mx.array([4, 5, 6]), "probe_labels": {}},
        ]
        result = collate_batches(batches, [])
        assert "tokens" in result
        assert result["tokens"].shape == (2, 3)

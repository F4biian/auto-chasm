"""Tests for the auto-chasm data pipeline.

Covers ``auto_chasm.data`` (``build_dataset``, ``span_labels_to_tokens``,
``_shift_and_fit``, masking contract, per-probe dict labels, offset) and
``auto_chasm.trainers.data_utils`` (``iterate_batches``, ``_infer_label_dtype``,
``_pad_label_matrix``, ``_build_label_output``, ``_pad_labels_to_len``).

Tests named ``test_BUG_*`` are regression tests for specific past defects; the
remaining tests are general regression coverage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from auto_chasm.data import (
    IGNORE_INDEX,
    _shift_and_fit,  # noqa: PLC2701 (intentional internal probe)
    build_dataset,
    span_labels_to_tokens,
)
from auto_chasm.trainers.data_utils import (
    _build_label_output,
    _infer_label_dtype,
    _pad_labels_to_len,
    iterate_batches,
)

# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


class WordTokenizer:
    """Encode-only tokenizer: one id per whitespace word, plus an EOS id."""

    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        return [100 + i for i, _ in enumerate(text.split())]


class OffsetWordTokenizer:
    """Transformers-style tokenizer: word-level ids + matching char offsets.

    ``encode()`` deliberately returns a DIFFERENT (per-character) token count to
    catch any path that mixes the two encodings.
    """

    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        return [200 + i for i in range(len(text))]

    def __call__(self, text: str, return_offsets_mapping: bool = False) -> Any:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        pos = 0
        for i, word in enumerate(text.split()):
            start = text.index(word, pos)
            end = start + len(word)
            ids.append(300 + i)
            offsets.append((start, end))
            pos = end

        class Encoding:
            """Fake tokenizer encoding exposing input_ids and offset_mapping."""

            def __init__(self, input_ids: list[int], om: list[tuple[int, int]]) -> None:
                self.input_ids = input_ids
                self.offset_mapping = om

        return Encoding(ids, offsets)


def _real_tokenizer() -> Any:
    """Load the cached SmolLM tokenizer, or skip if unavailable (no network)."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"real tokenizer unavailable: {exc}")


# ===========================================================================
# FINDING 1 (critical): mixing a single-probe (plain-list) sample and a
# multi-probe (dict) sample in one batch silently DROPS the plain-list sample's
# labels.  build_dataset emits a plain list when a conversation labels <=1 probe
# and a dict when it labels >=2 — so an ordinary multi-head dataset hits this.
# ===========================================================================


class TestMixedListAndDictLabelsInOneBatch:
    """A plain-list sample batched with a dict sample must keep its labels."""

    def test_BUG_plain_list_sample_labels_dropped_in_mixed_batch(self) -> None:
        """_build_label_output zeroes a plain-list sample to all -100.

        labels_list = [[1, 0], {"a": ..., "b": ...}].  The plain list belongs to
        whatever head(s) exist; instead every probe gets all -100 for that
        sample, silently discarding real supervision (research-poisoning).
        """
        labels_list = [[1, 0], {"a": [0, 1], "b": [1, 1]}]
        out = _build_label_output(labels_list, [2, 2], 2, 2, IGNORE_INDEX)
        assert isinstance(out, dict)
        # Sample 0 (the plain list [1, 0]) must NOT be all-masked for the probes.
        row0 = np.concatenate([out[name][0] for name in out])
        assert not np.all(row0 == IGNORE_INDEX), (
            "plain-list sample's labels were silently dropped to all -100 when "
            "batched alongside a dict-labelled sample"
        )

    def test_BUG_build_dataset_then_batch_drops_single_probe_labels(self) -> None:
        """A single-probe conversation keeps its labels AND doesn't bleed onto other heads.

        In a multi-probe dataset, conversation A labels only probe 'a' (unique label
        7) and conversation B labels 'a' and 'b'. build_dataset now emits BOTH as full
        {probe: labels} dicts (global 2-probe set), so A's 7 survives batching AND A's
        'b' stream is -100 (A never bleeds onto head 'b').
        """
        tok = OffsetWordTokenizer()
        conv_a = [
            {
                "role": "user",
                "content": "alpha beta",
                "labels": {"a": [{"start": 0, "end": 5, "label": 7}]},
            }
        ]
        conv_b = [
            {
                "role": "user",
                "content": "gamma delta",
                "labels": {
                    "a": [{"start": 0, "end": 5, "label": 1}],
                    "b": [{"start": 6, "end": 11, "label": 1}],
                },
            }
        ]
        dataset = build_dataset([conv_a, conv_b], tok, default_label=0)
        # Both samples are now full dicts over {a, b} — no ambiguous plain list.
        assert isinstance(dataset[0]["labels"], dict) and set(dataset[0]["labels"]) == {"a", "b"}
        assert isinstance(dataset[1]["labels"], dict)
        assert dataset[0]["labels"]["a"][0] == 7  # A's unique supervision exists pre-batch
        assert set(dataset[0]["labels"]["b"]) == {IGNORE_INDEX}  # A never labels 'b' -> no bleed

        _tokens, labels, _lengths = next(
            iterate_batches(dataset, batch_size=2, max_seq_length=32, loop=False)
        )
        assert isinstance(labels, dict) and "a" in labels
        # Conv A's unique label 7 survives in probe 'a'; probe 'b' has no 7 (no bleed).
        assert (labels["a"] == 7).any()
        assert not (labels["b"] == 7).any()

    def test_two_dict_samples_batch_correctly(self) -> None:
        """Control: two dict samples (no plain list) batch without loss."""
        labels_list = [{"a": [1, 0]}, {"a": [0, 1]}]
        out = _build_label_output(labels_list, [2, 2], 2, 2, IGNORE_INDEX)
        assert (out["a"] == 1).sum() == 2  # one positive per sample preserved

    def test_all_plain_lists_batch_correctly(self) -> None:
        """Control: a homogeneous plain-list batch is fine."""
        labels_list = [[1, 0], [0, 1]]
        out = _build_label_output(labels_list, [2, 2], 2, 2, IGNORE_INDEX)
        assert not isinstance(out, dict)
        assert (out == 1).sum() == 2


# ===========================================================================
# Masking contract: an unmarked token MUST be exactly -100 when default is None.
# ===========================================================================


class TestMaskingContract:
    """Every unmarked token defaults to -100; nothing is silently trained."""

    def test_message_without_labels_key_all_masked(self) -> None:
        tok = WordTokenizer()
        result = build_dataset([[{"role": "user", "content": "a b c"}]], tok)
        assert result[0]["labels"] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]

    def test_empty_labels_dict_all_masked(self) -> None:
        tok = WordTokenizer()
        result = build_dataset([[{"role": "user", "content": "a b c", "labels": {}}]], tok)
        assert result[0]["labels"] == [IGNORE_INDEX] * 3

    def test_unmarked_tokens_inside_labeled_message_are_masked(self) -> None:
        tok = WordTokenizer()
        result = build_dataset(
            [
                [
                    {
                        "role": "user",
                        "content": "a b c",
                        "labels": {"p": [{"start": 0, "end": 1, "label": 1}]},
                    }
                ]
            ],
            tok,
        )
        # Only the first word is marked; the other two stay -100 (not class 0).
        assert result[0]["labels"] == [1, IGNORE_INDEX, IGNORE_INDEX]

    def test_default_label_zero_is_distinct_from_mask(self) -> None:
        tok = WordTokenizer()
        result = build_dataset(
            [
                [
                    {
                        "role": "user",
                        "content": "a b c",
                        "labels": {"p": [{"start": 0, "end": 1, "label": 1}]},
                    }
                ]
            ],
            tok,
            default_label=0,
        )
        assert result[0]["labels"] == [1, 0, 0]

    def test_default_label_float_zero_does_not_mask(self) -> None:
        """default_label=0.0 must fill 0.0, never collapse to the -100 sentinel."""
        tok = WordTokenizer()
        result = build_dataset(
            [
                [
                    {
                        "role": "user",
                        "content": "a b c",
                        "labels": {"p": [{"start": 0, "end": 1, "label": 1}]},
                    }
                ]
            ],
            tok,
            default_label=0.0,
        )
        assert result[0]["labels"] == [1, 0.0, 0.0]
        assert IGNORE_INDEX not in result[0]["labels"]


# ===========================================================================
# Span edge cases.
# ===========================================================================


class TestSpanEdges:
    """Span-to-token labeling edge cases: clamping, reversal, and max aggregation."""

    def test_reversed_span_matches_nothing(self) -> None:
        """Start > end is an empty interval; it must label no token."""
        tok = WordTokenizer()
        labels = span_labels_to_tokens(
            "a b c", tok, [{"start": 5, "end": 0, "label": 1}], default_label=0
        )
        assert labels == [0, 0, 0]

    def test_negative_start_clamps(self) -> None:
        tok = WordTokenizer()
        labels = span_labels_to_tokens(
            "a b c", tok, [{"start": -5, "end": 1, "label": 1}], default_label=0
        )
        assert labels[0] == 1

    def test_span_beyond_end_clamps(self) -> None:
        tok = WordTokenizer()
        labels = span_labels_to_tokens(
            "a b c", tok, [{"start": 2, "end": 9999, "label": 1}], default_label=0
        )
        # The span covers the back portion of the text; the first token is outside.
        assert labels[0] == 0
        assert labels[-1] == 1

    def test_touching_spans_do_not_double_count_under_max(self) -> None:
        tok = OffsetWordTokenizer()
        # Two abutting spans [0,5) and [5,11); 'max' aggregation.
        labels = span_labels_to_tokens(
            "alpha betas",
            tok,
            [
                {"start": 0, "end": 5, "label": 1},
                {"start": 5, "end": 11, "label": 2},
            ],
            aggregation="max",
        )
        assert labels == [1, 2]

    def test_overlapping_spans_aggregate_max(self) -> None:
        from tests.unit.test_data import MockTransformersTokenizer  # reuse

        tok = MockTransformersTokenizer([(0, 10)])
        labels = span_labels_to_tokens(
            "helloworld",
            tok,
            [{"start": 0, "end": 5, "label": 0}, {"start": 3, "end": 10, "label": 1}],
            aggregation="max",
        )
        assert labels == [1]


# ===========================================================================
# _shift_and_fit: offset arithmetic must never desync length.
# ===========================================================================


class TestShiftAndFit:
    """_shift_and_fit offset arithmetic keeps length fixed and masks the gap."""

    @pytest.mark.parametrize("offset", [-3, -2, -1, 0, 1, 2, 3, 5, 10])
    def test_length_is_exactly_n_tokens(self, offset: int) -> None:
        out = _shift_and_fit([1, 2, 3], offset, 3, IGNORE_INDEX)
        assert len(out) == 3

    def test_positive_offset_opens_masked_gap_at_front(self) -> None:
        out = _shift_and_fit([1, 2, 3, 4], 1, 4, IGNORE_INDEX)
        assert out == [IGNORE_INDEX, 1, 2, 3]

    def test_negative_offset_opens_masked_gap_at_back(self) -> None:
        out = _shift_and_fit([1, 2, 3, 4], -1, 4, IGNORE_INDEX)
        assert out == [2, 3, 4, IGNORE_INDEX]

    def test_offset_exceeds_length_all_masked(self) -> None:
        out = _shift_and_fit([1, 2, 3], 10, 3, IGNORE_INDEX)
        assert out == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]


class TestBuildDatasetOffsetLengths:
    """build_dataset keeps tokens and labels aligned across label offsets."""

    @pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2, 5, 10])
    def test_tokens_and_labels_aligned(self, offset: int) -> None:
        tok = OffsetWordTokenizer()
        conv = [
            {
                "role": "user",
                "content": "alpha beta gamma delta",
                "labels": {"p": [{"start": 0, "end": 11, "label": 1}]},
            }
        ]
        result = build_dataset([conv], tok, offset=offset)
        assert len(result[0]["tokens"]) == len(result[0]["labels"])


# ===========================================================================
# dtype inference.
# ===========================================================================


class TestDtypeInference:
    """Per-probe label dtype inference (int vs float) and its stability across batches."""

    def test_minus100_survives_float_cast_exactly(self) -> None:
        out = _build_label_output([[IGNORE_INDEX, 0.8, 0.3]], [3], 1, 3, IGNORE_INDEX)
        assert out.dtype.kind == "f"
        assert out[0, 0] == IGNORE_INDEX  # exact, not -99.99

    def test_all_ints_stay_int(self) -> None:
        assert _infer_label_dtype([[0, 1, 0]]) is np.int32

    def test_any_float_promotes(self) -> None:
        assert _infer_label_dtype([[0, 1], [0.5, 1.0]]) is np.float32

    def test_empty_list_does_not_force_float(self) -> None:
        assert _infer_label_dtype([[], [0, 1]]) is np.int32

    def test_bool_labels_treated_as_int(self) -> None:
        out = _build_label_output([[True, False, True]], [3], 1, 3, IGNORE_INDEX)
        assert out.dtype.kind in ("i", "u", "b")
        assert out.tolist()[0] == [1, 0, 1]

    def test_per_probe_independent_dtype(self) -> None:
        out = _build_label_output([{"a": [0.1, 0.9], "b": [0, 1]}], [2, 2], 1, 2, IGNORE_INDEX)
        assert out["a"].dtype.kind == "f"
        assert out["b"].dtype.kind in ("i", "u")

    def test_BUG_regression_probe_dtype_unstable_across_batches(self) -> None:
        """The SAME probe must keep a stable label dtype across batches.

        A regression (float) probe whose labels happen to be all-masked in one
        batch falls back to int32, while a batch with real float values is
        float32.  ``labels_to_torch``/``labels_to_mlx`` then build tensors of
        different dtypes for the same probe across steps — a silent type
        instability that can surface as a dtype mismatch downstream.
        """
        batch_with_floats = _build_label_output([{"p": [0.5, 0.5]}], [2], 1, 2, IGNORE_INDEX)
        batch_all_masked = _build_label_output([{"p": []}], [2], 1, 2, IGNORE_INDEX)
        assert batch_with_floats["p"].dtype == batch_all_masked["p"].dtype, (
            "probe 'p' has float32 labels in one batch and int32 in an "
            "all-masked batch — dtype is not stable across batches"
        )


# ===========================================================================
# Padding / batching.
# ===========================================================================


class TestPaddingAndBatching:
    """Batch padding, truncation, and label masking for ragged token lengths."""

    def test_ragged_lengths_lengths_array_correct(self) -> None:
        data = [
            {"tokens": [1], "labels": [0]},
            {"tokens": [1, 2, 3], "labels": [0, 1, 0]},
            {"tokens": [1, 2], "labels": [0, 1]},
        ]
        _tokens, _labels, lengths = next(
            iterate_batches(data, batch_size=3, max_seq_length=32, loop=False)
        )
        # third column entry is the per-sample valid length; sorted ascending.
        valid = sorted(int(r[1]) for r in lengths)
        assert valid == [1, 2, 3]

    def test_length_one_example(self) -> None:
        data = [{"tokens": [7], "labels": [1]}]
        tokens, labels, lengths = next(iterate_batches(data, 1, 32, loop=False))
        assert tokens[0, 0] == 7
        assert labels[0, 0] == 1
        assert int(lengths[0, 1]) == 1

    def test_length_zero_token_list_does_not_crash(self) -> None:
        data = [{"tokens": [], "labels": []}, {"tokens": [1, 2], "labels": [0, 1]}]
        tokens, labels, lengths = next(iterate_batches(data, 2, 32, loop=False))
        assert tokens.shape[0] == 2
        # The empty sample has valid length 0 (fully masked).
        zero_rows = [int(r[1]) for r in lengths if int(r[1]) == 0]
        assert 0 in zero_rows

    def test_pad_positions_are_masked_in_labels(self) -> None:
        data = [{"tokens": [1], "labels": [5]}, {"tokens": [1, 2, 3], "labels": [5, 6, 7]}]
        _tokens, labels, _lengths = next(iterate_batches(data, 2, 32, loop=False))
        # The length-1 sample's columns 1.. must be the ignore sentinel, not 0.
        short_row = labels[np.argmin([np.count_nonzero(r != -100) for r in labels])]
        assert short_row[1] == IGNORE_INDEX

    def test_label_shorter_than_tokens_pads_with_minus100(self) -> None:
        out = _pad_labels_to_len([1, 0], 5)
        assert out == [1, 0, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]

    def test_label_longer_than_tokens_truncates(self) -> None:
        out = _pad_labels_to_len([1, 0, 1, 0, 1], 3)
        assert out == [1, 0, 1]

    def test_empty_dataset_raises_cleanly(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            next(iterate_batches([], 4, 32, loop=False))


# ===========================================================================
# Mutation: build_dataset must not mutate the caller's conversations.
# ===========================================================================


class TestNoMutation:
    """build_dataset and iterate_batches must not mutate their inputs."""

    def test_build_dataset_does_not_mutate_input(self) -> None:
        import copy

        tok = WordTokenizer()
        conv = [
            [
                {
                    "role": "user",
                    "content": "alpha beta",
                    "labels": {"a": [{"start": 0, "end": 5, "label": 1}]},
                }
            ]
        ]
        snapshot = copy.deepcopy(conv)
        build_dataset(conv, tok)
        assert conv == snapshot

    def test_iterate_batches_does_not_mutate_dataset(self) -> None:
        import copy

        data = [{"tokens": [1, 2, 3], "labels": [0, 1, 0]} for _ in range(4)]
        snapshot = copy.deepcopy(data)
        for _ in iterate_batches(data, 2, 32, loop=False):
            pass
        assert data == snapshot


# ===========================================================================
# Real tokenizer: leading-space (Ġ) tokenization + unicode offset alignment.
# ===========================================================================


@pytest.mark.real_model
class TestRealTokenizerOffsets:
    """Span labeling against a real tokenizer's offset mapping, including unicode."""

    def test_token_label_lengths_match_unicode(self) -> None:
        tok = _real_tokenizer()
        labels = span_labels_to_tokens("café 🌟 ok", tok, [{"start": 0, "end": 4, "label": 1}])
        ids = tok("café 🌟 ok", return_offsets_mapping=True)["input_ids"]
        assert len(labels) == len(ids)

    def test_leading_space_token_still_labeled(self) -> None:
        """A span on 'is' (no leading space) still hits the ' is' token."""
        tok = _real_tokenizer()
        # "Paris is nice"; "is" is chars 6-8, but its token offset is (5, 8).
        labels = span_labels_to_tokens("Paris is nice", tok, [{"start": 6, "end": 8, "label": 1}])
        assert labels.count(1) == 1
        assert labels[1] == 1  # the ' is' token

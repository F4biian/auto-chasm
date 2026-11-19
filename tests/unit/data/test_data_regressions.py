"""Regression tests for the auto_chasm data layer.

Oracle tests for four previously-fixed data-layer bugs:

1. ``iterate_batches`` dropped the last partial batch (up to ~44% of samples)
   and corrupted eval metrics on small datasets.  It must now yield ALL
   samples across its batches.
2. ``build_dataset`` silently max-merged multiple probes' spans into one shared
   ``labels`` array.  A multi-probe message must now raise
   ``NotImplementedError`` loudly.
3. ``build_dataset`` could misalign span labels when ``encode()`` and the
   offset-mapping path produced different token counts.  Tokens and labels must
   always have matching length.
4. ``JointTextDataset`` silently produced all-``-100`` labels on a mistyped
   ``labels_key``.  It must now warn loudly, like ``JointDataset`` does.

Plus: float regression labels must survive ``iterate_batches`` (dtype inferred
as float), preserving a prior fix.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from auto_chasm.data import build_dataset, span_labels_to_tokens
from auto_chasm.trainers.data_utils import JointTextDataset, iterate_batches


class MockEncodeTokenizer:
    """Tokenizer with only ``encode`` — one id per word, plus an EOS id."""

    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        """Return one token id per whitespace-delimited word."""
        return [100 + i for i, _ in enumerate(text.split())]


class MockOffsetTokenizer:
    """Transformers-style tokenizer exposing ``input_ids`` + ``offset_mapping``.

    Tokenizes on whitespace.  Crucially, ``encode()`` returns a DIFFERENT token
    count than the offset-mapping callable path: ``encode()`` splits on every
    character while the callable splits on words.  This exposes the alignment
    bug — only a build that reuses the callable encoding keeps tokens and labels
    the same length.
    """

    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        """Return one id per character (deliberately the WRONG count)."""
        return [200 + i for i in range(len(text))]

    def __call__(self, text: str, return_offsets_mapping: bool = False) -> Any:
        """Return an encoding with word-level ids and matching offsets."""
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
            """Mock encoding carrying ids and offset mapping."""

            def __init__(self, input_ids: list[int], om: list[tuple[int, int]]) -> None:
                self.input_ids = input_ids
                self.offset_mapping = om

        return Encoding(ids, offsets)


class TestIterateBatchesKeepsAllSamples:
    """Oracle: every sample appears across the yielded batches (no tail drop)."""

    @pytest.mark.parametrize(
        ("n", "batch_size"),
        [(10, 4), (7, 4), (5, 3), (16, 9), (3, 2), (9, 4), (1, 4)],
    )
    def test_all_samples_present(self, n: int, batch_size: int) -> None:
        """A dataset of N samples (N not a multiple of batch_size) yields all N."""
        # Each sample has a unique token id so we can count distinct samples.
        data = [{"tokens": [1000 + i], "labels": [0]} for i in range(n)]

        seen: set[int] = set()
        total_rows = 0
        for tokens, _labels, _lengths in iterate_batches(
            data, batch_size, max_seq_length=16, loop=False
        ):
            for row in tokens:
                seen.add(int(row[0]))
                total_rows += 1

        expected = {1000 + i for i in range(n)}
        assert seen == expected, f"dropped samples: {sorted(expected - seen)}"
        assert total_rows == n, f"expected {n} rows across batches, got {total_rows}"

    def test_partial_last_batch_is_smaller(self) -> None:
        """The final partial batch is smaller, not padded to batch_size."""
        data = [{"tokens": [1000 + i], "labels": [0]} for i in range(7)]
        sizes = [
            tokens.shape[0]
            for tokens, _l, _len in iterate_batches(
                data, batch_size=4, max_seq_length=16, loop=False
            )
        ]
        # 7 samples / batch_size 4 -> one full batch of 4 and one of 3.
        assert sorted(sizes) == [3, 4]
        assert sum(sizes) == 7

    def test_batch_count_matches_eval_divisor(self) -> None:
        """floor- and ceil-based callers both see the true number of batches.

        On a small dataset the eval-metric average divides by the number of
        batches actually produced; dropping the tail used to deflate it.
        """
        data = [{"tokens": [1, 2, 3], "labels": [0, 1, 0]} for _ in range(10)]
        n_batches = sum(1 for _ in iterate_batches(data, 4, 16, loop=False))
        # ceil(10 / 4) == 3, NOT floor == 2.
        assert n_batches == 3


class TestIterateBatchesFloatLabels:
    """Oracle: float regression targets survive (dtype inferred as float)."""

    def test_float_labels_not_truncated(self) -> None:
        """Float labels keep a floating dtype and exact values."""
        data = [{"tokens": [1, 2, 3], "labels": [0.1, 0.8, 0.3]} for _ in range(5)]
        _tokens, labels, _lengths = next(iterate_batches(data, 2, 16, loop=False))
        assert labels.dtype.kind == "f"
        # 0.8 would collapse to 0 under int truncation.
        assert np.isclose(float(labels[labels > 0.5].max()), 0.8)

    def test_partial_batch_float_labels_preserved(self) -> None:
        """The trailing partial batch also keeps float labels."""
        data = [{"tokens": [1, 2], "labels": [0.25, 0.75]} for _ in range(5)]
        seen_float = False
        for _t, labels, _len in iterate_batches(data, 2, 16, loop=False):
            assert labels.dtype.kind == "f"
            if np.any(np.isclose(labels, 0.75)):
                seen_float = True
        assert seen_float

    def test_int_labels_stay_int(self) -> None:
        """Integer class labels keep an integer dtype."""
        data = [{"tokens": [1, 2, 3], "labels": [0, 1, 0]} for _ in range(4)]
        _tokens, labels, _lengths = next(iterate_batches(data, 2, 16, loop=False))
        assert labels.dtype.kind in ("i", "u")


class TestBuildDatasetMultiProbe:
    """Oracle: two probes in one message emit independent per-probe label arrays."""

    def test_two_probes_emit_label_dict(self) -> None:
        """A message labelling two probes yields a {probe: labels} dict."""
        tok = MockEncodeTokenizer()
        conversations = [
            [
                {
                    "role": "user",
                    "content": "alpha beta gamma",
                    "labels": {
                        "probe_a": [{"start": 0, "end": 5, "label": 1}],
                        "probe_b": [{"start": 6, "end": 10, "label": 1}],
                    },
                }
            ]
        ]
        labels = build_dataset(conversations, tok)[0]["labels"]
        assert isinstance(labels, dict)
        assert set(labels) == {"probe_a", "probe_b"}
        # "alpha" (0-5) is probe_a's positive; "beta" (6-10) is probe_b's.
        # Each head sees ONLY its own positive; the rest is masked (-100).
        assert labels["probe_a"] == [1, -100, -100]
        assert labels["probe_b"] == [-100, 1, -100]

    def test_two_probes_independent_default_label(self) -> None:
        """default_label=0 fills unmarked tokens per head, heads stay independent."""
        tok = MockEncodeTokenizer()
        conversations = [
            [
                {
                    "role": "user",
                    "content": "alpha beta gamma",
                    "labels": {
                        "a": [{"start": 0, "end": 5, "label": 1}],
                        "b": [{"start": 6, "end": 10, "label": 1}],
                    },
                }
            ]
        ]
        labels = build_dataset(conversations, tok, default_label=0)[0]["labels"]
        # Within each head, the message IS labeled → unmarked tokens become 0.
        assert labels["a"] == [1, 0, 0]
        assert labels["b"] == [0, 1, 0]

    def test_probe_only_labeled_in_one_message_masks_other_messages(self) -> None:
        """A head is masked (-100) in messages that do not name it."""
        tok = MockEncodeTokenizer()
        conversations = [
            [
                {
                    "role": "user",
                    "content": "alpha beta",
                    "labels": {"a": [{"start": 0, "end": 5, "label": 1}]},
                },
                {
                    "role": "assistant",
                    "content": "gamma delta",
                    "labels": {"b": [{"start": 0, "end": 5, "label": 1}]},
                },
            ]
        ]
        labels = build_dataset(conversations, tok, default_label=0)[0]["labels"]
        # 4 tokens total (alpha beta | gamma delta).  Head "a" is labeled only in
        # the first message → masked (-100) over the second; head "b" the reverse.
        assert labels["a"] == [1, 0, -100, -100]
        assert labels["b"] == [-100, -100, 1, 0]

    def test_single_probe_still_works(self) -> None:
        """A single-probe dataset builds without raising and labels the span."""
        tok = MockEncodeTokenizer()
        conversations = [
            [
                {
                    "role": "user",
                    "content": "alpha beta gamma",
                    "labels": {"probe_a": [{"start": 0, "end": 5, "label": 1}]},
                }
            ]
        ]
        result = build_dataset(conversations, tok)
        assert len(result) == 1
        labels = result[0]["labels"]
        # "alpha" (chars 0-5) overlaps the first word token -> label 1.
        assert labels[0] == 1
        # Other words are outside the span -> MASKED (-100) by default: labels
        # are opt-in, so unmarked tokens are never silently trained as class 0.
        assert labels[1] == -100
        assert labels[2] == -100

    def test_single_probe_default_label_zero_fills_negatives(self) -> None:
        """default_label=0 turns unmarked tokens into the negative class 0."""
        tok = MockEncodeTokenizer()
        conversations = [
            [
                {
                    "role": "user",
                    "content": "alpha beta gamma",
                    "labels": {"probe_a": [{"start": 0, "end": 5, "label": 1}]},
                }
            ]
        ]
        result = build_dataset(conversations, tok, default_label=0)
        labels = result[0]["labels"]
        assert labels[0] == 1
        assert labels[1] == 0
        assert labels[2] == 0

    def test_empty_span_probe_is_a_declared_probe(self) -> None:
        """An EMPTY span list DECLARES the probe (all-negative), so this is multi-probe.

        It used to be read as "unlabeled", which silently dropped every
        negative-only example -- in a span-annotated corpus, exactly the clean
        ones. Declaring probe_b therefore now yields per-probe dict labels:
        probe_a carries its span, probe_b is all-fill.
        """
        tok = MockEncodeTokenizer()
        conversations = [
            [
                {
                    "role": "user",
                    "content": "alpha beta gamma",
                    "labels": {
                        "probe_a": [{"start": 0, "end": 5, "label": 1}],
                        "probe_b": [],
                    },
                }
            ]
        ]
        result = build_dataset(conversations, tok, default_label=0)
        labels = result[0]["labels"]
        assert set(labels) == {"probe_a", "probe_b"}
        assert labels["probe_a"][0] == 1
        assert labels["probe_b"] == [0, 0, 0]

    def test_no_labels_still_works(self) -> None:
        """A message with no labels masks all tokens to -100."""
        tok = MockEncodeTokenizer()
        conversations = [[{"role": "user", "content": "alpha beta gamma"}]]
        result = build_dataset(conversations, tok)
        assert result[0]["labels"] == [-100, -100, -100]


class TestBuildDatasetAlignment:
    """Oracle: tokens and labels always have matching length after build."""

    def test_tokens_labels_same_length_offset_tokenizer(self) -> None:
        """With an offset tokenizer whose encode() disagrees, lengths still match."""
        tok = MockOffsetTokenizer()
        conversations = [
            [
                {
                    "role": "user",
                    "content": "alpha beta gamma",
                    "labels": {"p": [{"start": 0, "end": 5, "label": 1}]},
                }
            ]
        ]
        result = build_dataset(conversations, tok)
        sample = result[0]
        assert len(sample["tokens"]) == len(sample["labels"])
        # The callable path yields 3 word tokens, not 16 char tokens from encode().
        assert len(sample["tokens"]) == 3
        # Tokens come from the callable encoding (300-range), not encode() (200-range).
        assert all(t >= 300 for t in sample["tokens"])
        # First word "alpha" is in the span -> label 1; the rest are masked (-100).
        assert sample["labels"] == [1, -100, -100]

    @pytest.mark.parametrize("offset", [0, 1, -1, 2])
    def test_lengths_match_under_offsets(self, offset: int) -> None:
        """Label shifting never desyncs token and label counts."""
        tok = MockOffsetTokenizer()
        conversations = [
            [
                {
                    "role": "user",
                    "content": "alpha beta gamma delta",
                    "labels": {"p": [{"start": 0, "end": 11, "label": 1}]},
                }
            ]
        ]
        result = build_dataset(conversations, tok, offset=offset)
        assert len(result[0]["tokens"]) == len(result[0]["labels"])

    def test_multi_message_lengths_match(self) -> None:
        """Concatenating multiple messages keeps tokens and labels aligned."""
        tok = MockOffsetTokenizer()
        conversations = [
            [
                {"role": "user", "content": "alpha beta"},
                {
                    "role": "assistant",
                    "content": "gamma delta epsilon",
                    "labels": {"p": [{"start": 0, "end": 5, "label": 1}]},
                },
            ]
        ]
        result = build_dataset(conversations, tok)
        assert len(result[0]["tokens"]) == len(result[0]["labels"])

    def test_span_labels_alignment_oracle(self) -> None:
        """span_labels_to_tokens itself stays aligned to the offset encoding."""
        tok = MockOffsetTokenizer()
        labels = span_labels_to_tokens(
            "alpha beta gamma", tok, [{"start": 0, "end": 5, "label": 1}]
        )
        # 3 word tokens; only the first overlaps the span; rest masked (-100).
        assert labels == [1, -100, -100]


class TestJointTextDatasetWarnsOnMistypedKey:
    """Oracle: a mistyped labels_key warns loudly instead of silent -100s."""

    def test_mistyped_labels_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """An absent labels_key produces a warning naming the bad key."""
        data = [{"text": "hi", "lables": [0, 1]} for _ in range(4)]  # typo: lables
        tok = MockEncodeTokenizer()
        with caplog.at_level(logging.WARNING, logger="auto_chasm.trainers.data_utils"):
            JointTextDataset(data, tok, labels_key="labels")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a loud warning for the mistyped labels_key"
        assert any("labels" in r.getMessage() for r in warnings)

    def test_correct_labels_key_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A correct labels_key present in all samples does not warn."""
        data = [{"text": "hi", "labels": [0, 1]} for _ in range(4)]
        tok = MockEncodeTokenizer()
        with caplog.at_level(logging.WARNING, logger="auto_chasm.trainers.data_utils"):
            JointTextDataset(data, tok, labels_key="labels")
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "labels_key" in r.getMessage()
        ]
        assert not warnings

    def test_mistyped_labels_key_trains_on_nothing(self) -> None:
        """Confirms the silent failure mode the warning guards: all -100 labels."""
        data = [{"text": "a b", "lables": [0, 1]}]  # typo
        tok = MockEncodeTokenizer()
        ds = JointTextDataset(data, tok, labels_key="labels")
        _tokens, labels = ds[0]
        # Without the correct key, every label is the ignore sentinel.
        assert all(label == -100 for label in labels)

    def test_mistyped_tokens_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """An absent tokens_key (when one is requested) warns loudly."""
        data = [{"text": "hi", "labels": [0, 1], "toks": [1, 2]} for _ in range(4)]
        tok = MockEncodeTokenizer()
        with caplog.at_level(logging.WARNING, logger="auto_chasm.trainers.data_utils"):
            JointTextDataset(data, tok, tokens_key="tokens")
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "tokens_key" in r.getMessage()
        ]
        assert warnings, "expected a loud warning for the mistyped tokens_key"


def test_seed_zero_is_deterministic_and_does_not_stomp_global_rng() -> None:
    """seed=0 reproduces the batch order and leaves the global np.random stream intact.

    Regression: `if seed:` treated seed=0 as "no seed" (nondeterministic), and seeding
    used the GLOBAL np.random (stomping the user's stream).
    """
    data = [{"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 0, 0]} for _ in range(20)]

    def _order(seed: int) -> list:
        it = iterate_batches(data, batch_size=2, max_seq_length=16, loop=True, seed=seed)
        return [tuple(np.array(next(it)[0]).ravel()[:2].tolist()) for _ in range(6)]

    np.random.seed(111)
    a = _order(0)
    np.random.seed(222)  # a DIFFERENT ambient global state
    b = _order(0)
    assert a == b  # seed=0 is honored, independent of global state

    # The user's global np.random stream is untouched by iterating.
    np.random.seed(5)
    before = np.random.rand()
    list(iterate_batches(data, batch_size=2, max_seq_length=16, loop=False, seed=0))
    np.random.seed(5)
    assert before == np.random.rand()

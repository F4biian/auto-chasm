"""Oracle tests for the OOP Dataset front-end.

Pins: ``from_conversations`` is identical to ``build_dataset`` (back-compat,
including the >=2-probe dict branch and a nonzero offset); ``from_texts`` places
the right per-token labels for each ``label_site``; ``label_site='sentence'``
without delimiters raises; ``class_weights`` uses the balanced formula; and
``split`` is deterministic, disjoint, and size-preserving.
"""

from __future__ import annotations

import pytest

from auto_chasm import Dataset
from auto_chasm.data import build_dataset


class _CharTok:
    """One token id per character; the build_dataset char-offset fallback then
    maps character ``i`` to token ``i`` exactly, so spans are unambiguous."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        """Return one id per character."""
        return [ord(c) for c in text]


def test_from_conversations_matches_build_dataset() -> None:
    """from_conversations(...).samples == build_dataset(...) element-for-element."""
    tok = _CharTok()
    conv = [
        [
            {
                "role": "user",
                "content": "abcdef",
                "labels": {
                    "a": [{"start": 0, "end": 2, "label": 1}],
                    "b": [{"start": 4, "end": 6, "label": 1}],
                },
            }
        ]
    ]
    direct = build_dataset(conv, tok, 1, "max", 0)
    via_cls = Dataset.from_conversations(conv, tok, offset=1, aggregation="max", default_label=0)
    assert via_cls.samples == direct
    # the >=2-probe branch yields per-probe dict labels
    assert isinstance(via_cls[0]["labels"], dict)
    assert set(via_cls[0]["labels"]) == {"a", "b"}


def test_no_cross_probe_label_bleed_in_multi_probe_dataset() -> None:
    """A sample labeling one probe emits -100 for the others (no cross-head bleed).

    Regression: a conversation that labeled exactly one probe dropped the probe name
    (plain list), which then broadcast to EVERY head at batch time — head A trained
    on head B's labels. With ≥2 probes named anywhere, every sample must be a full
    {probe: labels} dict with -100 for the probes it does not label.
    """
    tok = _CharTok()
    conv = [
        [{"role": "user", "content": "ab", "labels": {"a": [{"start": 0, "end": 2, "label": 5}]}}],
        [{"role": "user", "content": "ab", "labels": {"b": [{"start": 0, "end": 2, "label": 6}]}}],
    ]
    samples = build_dataset(conv, tok, 0, "max", None)
    # Every sample is a full dict over BOTH probes.
    for s in samples:
        assert isinstance(s["labels"], dict) and set(s["labels"]) == {"a", "b"}
    # Sample 0 labeled only "a": its "b" stream is entirely -100 (no bleed), and vice versa.
    assert set(samples[0]["labels"]["b"]) == {-100}
    assert 5 in samples[0]["labels"]["a"]
    assert set(samples[1]["labels"]["a"]) == {-100}
    assert 6 in samples[1]["labels"]["b"]


def test_from_texts_response_site() -> None:
    """label_site='response' labels only the last character/token."""
    ds = Dataset.from_texts(["abc"], [2], _CharTok(), label_site="response", probe_name="p")
    assert ds[0]["labels"] == [-100, -100, 2]


def test_from_texts_response_append_eos_reads_last_token() -> None:
    """append_eos moves the response label onto an EOS so the WHOLE text is read.

    Without it, the label sits on the last content token and the loss (next-token
    aligned) supervises the second-to-last token's state — the last token is
    never an input. With it, the supervised state has read all content tokens.
    """
    ds = Dataset.from_texts(
        ["abcde"], [3], _CharTok(), label_site="response", probe_name="p", append_eos=True
    )
    sample = ds[0]
    assert sample["tokens"] == [ord(c) for c in "abcde"] + [0]  # EOS (id 0) appended
    assert sample["labels"] == [-100, -100, -100, -100, -100, 3]  # label moved onto the EOS
    # next-token shift: the EOS label supervises probe output 4 = the state after
    # reading tokens[0..4] = the full text 'abcde' (incl the final token).
    shifted = sample["labels"][1:]
    assert [k for k, v in enumerate(shifted) if v != -100] == [4]


def test_append_eos_requires_eos_token() -> None:
    """append_eos with a tokenizer lacking eos_token_id raises clearly."""

    class _NoEos:
        eos_token_id = None

        def encode(self, text: str) -> list[int]:
            return [ord(c) for c in text]

    with pytest.raises(ValueError, match="eos_token_id"):
        Dataset.from_texts(["ab"], [0], _NoEos(), label_site="response", append_eos=True)


def test_from_texts_token_site() -> None:
    """label_site='token' labels every token from warmup_chars to the end."""
    ds = Dataset.from_texts(
        ["abcd"], [3], _CharTok(), label_site="token", warmup_chars=1, probe_name="p"
    )
    assert ds[0]["labels"] == [-100, 3, 3, 3]


def test_from_texts_sentence_site() -> None:
    """label_site='sentence' labels each sentence-ending delimiter position."""
    ds = Dataset.from_texts(
        ["ab.cd."],
        [1],
        _CharTok(),
        label_site="sentence",
        sentence_delimiters=["."],
        probe_name="p",
    )
    # '.' is at char offsets 2 and 5.
    assert ds[0]["labels"] == [-100, -100, 1, -100, -100, 1]


def test_sentence_without_delimiters_raises() -> None:
    """label_site='sentence' without delimiters raises (no auto-detect)."""
    with pytest.raises(ValueError, match="sentence_delimiters"):
        Dataset.from_texts(["a.b"], [0], _CharTok(), label_site="sentence")


def test_unknown_label_site_raises() -> None:
    """An unknown label_site raises."""
    with pytest.raises(ValueError, match="label_site"):
        Dataset.from_texts(["abc"], [0], _CharTok(), label_site="bogus")


def test_class_weights_balanced_formula() -> None:
    """Dataset.class_weights = total / (C * max(count, 1)) over non-(-100) labels."""
    samples = [
        {"tokens": [1, 2, 3], "labels": [0, 0, 1]},
        {"tokens": [4, 5], "labels": [2, -100]},
    ]
    ds = Dataset(samples)
    assert ds.class_weights(3) == pytest.approx([4 / 6, 4 / 3, 4 / 3])
    assert ds.label_counts(3) == [2, 1, 1]
    with pytest.raises(ValueError, match="scheme"):
        ds.class_weights(3, scheme="median")


def test_split_is_deterministic_disjoint_and_sized() -> None:
    """split is reproducible, partitions the data, and sizes sum to the original."""
    ds = Dataset([{"tokens": [i], "labels": [0]} for i in range(20)])
    train, val = ds.split(0.25, seed=0)
    assert len(train) + len(val) == 20
    assert len(val) == 5
    train_ids = {s["tokens"][0] for s in train}
    val_ids = {s["tokens"][0] for s in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(range(20))
    # deterministic
    train2, val2 = ds.split(0.25, seed=0)
    assert [s["tokens"] for s in val] == [s["tokens"] for s in val2]
    assert [s["tokens"] for s in train] == [s["tokens"] for s in train2]


class _MultiByteTok:
    """A byte-level tokenizer: a multi-byte char yields several tokens (like ByteLevel BPE)."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        """One token per UTF-8 byte, so a multi-byte final char spans several tokens."""
        ids: list[int] = []
        for c in text:
            ids += [ord(c[0]) % 200 + 1] * len(c.encode("utf-8"))
        return ids


def test_response_multibyte_final_char_gets_exactly_one_label() -> None:
    """A multi-byte final char yields ONE response label, not several (M4, end-to-end)."""
    ds = Dataset.from_texts(["ok 👍"], [1], _MultiByteTok(), label_site="response", probe_name="p")
    labeled = [v for v in ds[0]["labels"] if v != -100]
    assert labeled == [1]  # exactly one labeled token, value 1


def test_collapse_response_label_keeps_last_and_warns_on_none(caplog) -> None:  # noqa: ANN001
    """collapse_response_label keeps the last labeled token (M4) and warns on none (M3)."""
    import logging

    from auto_chasm.data import collapse_response_label

    # M4: several labeled tokens (multi-byte final char) collapse to just the last.
    assert collapse_response_label([-100, 1, 1, 1]) == [-100, -100, -100, 1]
    # M3: zero labeled tokens (uncovered last char) -> unchanged + a clear warning.
    with caplog.at_level(logging.WARNING, "auto_chasm"):
        assert collapse_response_label([-100, -100]) == [-100, -100]
    assert "NO labeled token" in caplog.text


def test_offset_shifting_off_the_only_label_warns(caplog) -> None:  # noqa: ANN001
    """offset=1 that pushes a response label off the end warns (M1)."""
    import logging

    with caplog.at_level(logging.WARNING, "auto_chasm"):
        Dataset.from_texts(
            ["aa bb"], [1], _CharTok(), label_site="response", probe_name="p", offset=1
        )
    assert "NO supervision" in caplog.text

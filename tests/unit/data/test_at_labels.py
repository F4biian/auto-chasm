r"""Positional ("at") label specs address the template tokens a char span cannot.

A character span reaches only a message's own content. The tokens a chat template
renders BETWEEN two turns -- ``<|im_end|>\\n<|im_start|>assistant\\n``, ``[/INST]`` --
belong to no message, yet that is where "has the model decided whether it can answer?"
lives: the last token it reads before writing anything.

These tests pin the two properties that make the feature usable across models: the
sites are derived from the template's OWN rendering (so ChatML and Llama-2 agree on
what "the gap between these turns" means, even though they split that text
differently), and a spec that selects nothing warns instead of silently training on
an empty target.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from auto_chasm._at_labels import at_positions, is_at_spec, validate_at_spec
from auto_chasm.data import IGNORE_INDEX, build_dataset

# ---------------------------------------------------------------------------
# Tokenizers: word-level ids + offsets, one per template family
# ---------------------------------------------------------------------------


class _WordTok:
    """Whitespace tokenizer with a stable vocab, char offsets and a chat template."""

    chat_template = "fake"
    eos_token_id = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _id(self, word: str) -> int:
        return self.vocab.setdefault(word, 1000 + len(self.vocab))

    def encode(self, text: str, **_: Any) -> list[int]:
        return [self._id(w) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        back = {i: w for w, i in self.vocab.items()}
        return " ".join(back.get(i, "?") for i in ids)

    def __call__(self, text: str, return_offsets_mapping: bool = False) -> Any:
        ids, offsets, pos = [], [], 0
        for word in text.split():
            start = text.index(word, pos)
            ids.append(self._id(word))
            offsets.append((start, start + len(word)))
            pos = start + len(word)

        class Encoding:
            """Fake encoding exposing input_ids and offset_mapping."""

            def __init__(self) -> None:
                self.input_ids = ids
                self.offset_mapping = offsets

        return Encoding()

    def apply_chat_template(
        self, messages: list[dict[str, Any]], tokenize: bool = False,
        add_generation_prompt: bool = False, **_: Any,
    ) -> str:
        raise NotImplementedError


class ChatMLTok(_WordTok):
    """Closes a turn and opens the next in ONE run of text, like ChatML."""

    def apply_chat_template(
        self, messages: list[dict[str, Any]], tokenize: bool = False,
        add_generation_prompt: bool = False, **_: Any,
    ) -> str:
        out = "".join(f"<start> {m['role']} : {m['content']} <end> " for m in messages)
        return out + ("<start> assistant : " if add_generation_prompt else "")


class LlamaTok(_WordTok):
    """Brackets the user turn instead, like Llama-2's [INST] ... [/INST]."""

    def apply_chat_template(
        self, messages: list[dict[str, Any]], tokenize: bool = False,
        add_generation_prompt: bool = False, **_: Any,
    ) -> str:
        out = "<s> "
        for m in messages:
            out += (
                f"[INST] {m['content']} [/INST] " if m["role"] == "user"
                else f"{m['content']} </s> "
            )
        return out


def _conv(user_labels: Any = None, assistant_labels: Any = None) -> list[dict[str, Any]]:
    user: dict[str, Any] = {"role": "user", "content": "is the answer here"}
    assistant: dict[str, Any] = {"role": "assistant", "content": "yes it is"}
    if user_labels is not None:
        user["labels"] = user_labels
    if assistant_labels is not None:
        assistant["labels"] = assistant_labels
    return [user, assistant]


def _labeled(sample: dict[str, Any], probe: str | None = None) -> list[int]:
    labels = sample["labels"]
    seq = labels if probe is None else labels[probe]
    return [i for i, v in enumerate(seq) if v != IGNORE_INDEX]


def _answer_start(tok: _WordTok, sample: dict[str, Any]) -> int:
    """Index of the first token of the assistant's own content."""
    first = tok.encode("yes")[0]
    return sample["tokens"].index(first)


# ---------------------------------------------------------------------------
# The decision point: "after this user message" == "before this answer"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tok_cls", [ChatMLTok, LlamaTok])
def test_after_user_is_the_last_token_before_the_answer(tok_cls: type[_WordTok]) -> None:
    """The site is derived from the template, so both families agree on the gap.

    ChatML puts the turn separator in the user's suffix; the Llama-2-style template
    writes ``[/INST]`` there instead. "The token just before the answer" must be the
    same position either way -- that is the whole point of before/after.
    """
    tok = tok_cls()
    sample = build_dataset(
        [_conv(user_labels={"answ": [{"at": "after", "index": -1, "label": 1}]})],
        tok, default_label=0,
    )[0]
    positions = _labeled(sample)
    assert positions == [_answer_start(tok, sample) - 1]


@pytest.mark.parametrize("tok_cls", [ChatMLTok, LlamaTok])
def test_before_assistant_names_the_same_token(tok_cls: type[_WordTok]) -> None:
    tok = tok_cls()
    convo = _conv(
        user_labels={"a": [{"at": "after", "index": -1, "label": 1}]},
        assistant_labels={"b": [{"at": "before", "index": -1, "label": 1}]},
    )
    sample = build_dataset([convo], tok, default_label=0)[0]
    assert _labeled(sample, "a") == _labeled(sample, "b")


def test_sites_cover_exactly_their_tokens() -> None:
    """prefix / content / suffix partition a message; message is their union."""
    tok = ChatMLTok()
    convo = _conv(user_labels={
        "pre": [{"at": "prefix", "label": 1}],
        "con": [{"at": "content", "label": 1}],
        "suf": [{"at": "suffix", "label": 1}],
        "msg": [{"at": "message", "label": 1}],
    })
    sample = build_dataset([convo], tok)[0]
    pre, con, suf, msg = (_labeled(sample, k) for k in ("pre", "con", "suf", "msg"))
    assert pre and con and suf
    assert pre + con + suf == msg == list(range(msg[0], msg[-1] + 1))
    # The content site is the message's own words, nothing else.
    assert [sample["tokens"][i] for i in con] == tok.encode("is the answer here")


def test_gaps_of_a_multi_turn_conversation() -> None:
    """after(i) and before(i+1) name the same tokens in the middle of a chat."""
    tok = ChatMLTok()
    convo = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "first question",
         "labels": {"a": [{"at": "after", "label": 1}]}},
        {"role": "assistant", "content": "first answer",
         "labels": {"b": [{"at": "before", "label": 1}]}},
        {"role": "user", "content": "second question"},
    ]
    sample = build_dataset([convo], tok)[0]
    gap = _labeled(sample, "a")
    assert gap == _labeled(sample, "b")
    # It sits strictly between the two contents.
    first_answer = sample["tokens"].index(tok.encode("first")[0], gap[-1])
    assert gap[-1] == first_answer - 1


# ---------------------------------------------------------------------------
# Composition with character spans
# ---------------------------------------------------------------------------


def test_positional_only_leaves_the_content_masked() -> None:
    """A pinned position must not turn every content token into class 0.

    default_label=0 exists for "mark the positive spans, the rest is negative". A
    spec that deliberately names ONE token says nothing about the message's text, and
    filling it would label every prompt token with the prompt-level answer.
    """
    tok = ChatMLTok()
    sample = build_dataset(
        [_conv(user_labels={"answ": [{"at": "after", "index": -1, "label": 1}]})],
        tok, default_label=0,
    )[0]
    assert len(_labeled(sample)) == 1
    assert sum(1 for v in sample["labels"] if v == 0) == 0


def test_spans_still_fill_when_both_kinds_are_present() -> None:
    """With a char span present the content keeps its usual fill semantics."""
    tok = ChatMLTok()
    convo = _conv(user_labels={"answ": [
        {"start": 0, "end": 2, "label": 1},                 # "is"
        {"at": "after", "index": -1, "label": 1},
    ]})
    sample = build_dataset([convo], tok, default_label=0)[0]
    labels = sample["labels"]
    assert labels.count(1) == 2          # the span's token + the gap token
    assert labels.count(0) == 3          # the rest of the content, filled


def test_empty_span_list_still_declares_the_probe() -> None:
    tok = ChatMLTok()
    sample = build_dataset([_conv(user_labels={"answ": []})], tok, default_label=0)[0]
    assert sample["labels"].count(0) == 4      # every content token, none positive


@pytest.mark.parametrize(("aggregation", "expected"), [("max", 5), ("min", 2)])
def test_overlapping_specs_aggregate(aggregation: str, expected: int) -> None:
    """A span and a positional spec on the same token combine like two spans do."""
    tok = ChatMLTok()
    convo = _conv(user_labels={"answ": [
        {"start": 0, "end": 2, "label": 2},
        {"at": "content", "index": 0, "label": 5},
    ]})
    sample = build_dataset([convo], tok, default_label=0, aggregation=aggregation)[0]
    assert sample["labels"][_labeled(sample)[0]] == expected


# ---------------------------------------------------------------------------
# token_ids: the escape hatch for templates this module cannot reason about
# ---------------------------------------------------------------------------


def test_token_ids_match_inside_the_site() -> None:
    tok = ChatMLTok()
    marker = tok.encode("<start> assistant")
    convo = _conv(assistant_labels={"open": [
        {"at": "before", "token_ids": marker, "label": 1},
    ]})
    sample = build_dataset([convo], tok)[0]
    positions = _labeled(sample)
    assert [sample["tokens"][i] for i in positions] == marker


def test_token_ids_default_to_the_whole_message() -> None:
    tok = ChatMLTok()
    target = tok.encode("answer")
    sample = build_dataset([_conv(user_labels={"p": [{"token_ids": target, "label": 1}]})], tok)[0]
    assert [sample["tokens"][i] for i in _labeled(sample)] == target


def test_token_ids_then_index_selects_one() -> None:
    tok = ChatMLTok()
    marker = tok.encode("<start> assistant")
    convo = _conv(assistant_labels={"open": [
        {"at": "before", "token_ids": marker, "index": -1, "label": 1},
    ]})
    sample = build_dataset([convo], tok)[0]
    assert [sample["tokens"][i] for i in _labeled(sample)] == marker[-1:]


# ---------------------------------------------------------------------------
# A spec that selects nothing must say so
# ---------------------------------------------------------------------------


def test_missing_token_ids_warn_and_label_nothing(caplog: pytest.LogCaptureFixture) -> None:
    tok = ChatMLTok()
    with caplog.at_level(logging.WARNING):
        sample = build_dataset(
            [_conv(user_labels={"p": [{"token_ids": [424242], "label": 1}]})], tok)[0]
    assert _labeled(sample) == []
    assert "do not occur" in caplog.text


def test_index_out_of_range_warns_and_labels_nothing(caplog: pytest.LogCaptureFixture) -> None:
    tok = ChatMLTok()
    with caplog.at_level(logging.WARNING):
        sample = build_dataset(
            [_conv(user_labels={"p": [{"at": "content", "index": 99, "label": 1}]})], tok)[0]
    assert _labeled(sample) == []
    assert "out of range" in caplog.text


def test_scaffolding_site_without_a_template_warns(caplog: pytest.LogCaptureFixture) -> None:
    """chat_template=False leaves no scaffolding, so 'after' cannot exist."""
    tok = ChatMLTok()
    with caplog.at_level(logging.WARNING):
        sample = build_dataset(
            [_conv(user_labels={"p": [{"at": "after", "label": 1}]})],
            tok, chat_template=False,
        )[0]
    assert _labeled(sample) == []
    assert "is empty here" in caplog.text


def test_content_site_works_without_a_template() -> None:
    tok = ChatMLTok()
    sample = build_dataset(
        [_conv(user_labels={"p": [{"at": "content", "index": -1, "label": 1}]})],
        tok, chat_template=False,
    )[0]
    assert len(_labeled(sample)) == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("spec", "match"), [
    ({"at": "nowhere", "label": 1}, "unknown site"),
    ({"at": "after"}, "no 'label'"),
    ({"at": "after", "start": 0, "end": 1, "label": 1}, "mixes a character span"),
    ({"at": "after", "index": "last", "label": 1}, "non-integer 'index'"),
    ({"token_ids": [], "label": 1}, "empty or non-sequence"),
    ({"token_ids": ["x"], "label": 1}, "non-integer entries"),
])
def test_malformed_specs_raise(spec: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_at_spec(spec, "p")
    # and the same failure surfaces through the public builder
    with pytest.raises(ValueError, match=match):
        build_dataset([_conv(user_labels={"p": [spec]})], ChatMLTok())


def test_is_at_spec_distinguishes_the_two_forms() -> None:
    assert is_at_spec({"at": "after", "label": 1})
    assert is_at_spec({"token_ids": [1], "label": 1})
    assert not is_at_spec({"start": 0, "end": 1, "label": 1})


def test_at_positions_is_usable_directly() -> None:
    """The resolver is a pure function of the layout, not of any tokenizer."""
    layout = [{"pre": (0, 2), "content": (2, 5), "post": (5, 7)},
              {"pre": (7, 7), "content": (7, 9), "post": (9, 10)}]
    tokens = list(range(10))
    assert at_positions({"at": "after", "label": 1}, 0, layout, tokens, "p", "user") == [5, 6]
    assert at_positions({"at": "before", "label": 1}, 1, layout, tokens, "p", "assistant") == [5, 6]
    assert at_positions({"at": "message", "label": 1}, 0, layout, tokens, "p", "user") == list(
        range(0, 7)
    )


# ---------------------------------------------------------------------------
# Interaction with the rest of the pipeline
# ---------------------------------------------------------------------------


def test_positional_labels_coexist_with_lm_weights() -> None:
    """The LM-weight channel is untouched, and the probe keeps its pinned token."""
    tok = ChatMLTok()
    convo = _conv(user_labels={"answ": [{"at": "after", "index": -1, "label": 1}]})
    sample = build_dataset([convo], tok, lm_train_on="assistant")[0]
    labels = sample["labels"]
    assert set(labels) == {"answ", "lm_head"}
    assert len(_labeled(sample, "answ")) == 1
    start = _answer_start(tok, sample)
    assert labels["lm_head"][start] == 1.0        # the answer trains
    assert labels["lm_head"][start - 1] == 0.0    # its opener does not
    assert len(labels["lm_head"]) == len(sample["tokens"])


def test_two_probes_stay_independent() -> None:
    tok = ChatMLTok()
    convo = _conv(
        user_labels={"answ": [{"at": "after", "index": -1, "label": 1}]},
        assistant_labels={"halluc": [{"start": 0, "end": 3, "label": 1}]},
    )
    sample = build_dataset([convo], tok, default_label=0)[0]
    assert _labeled(sample, "answ") != _labeled(sample, "halluc")
    # each head masks the other's tokens
    for i in _labeled(sample, "answ"):
        assert sample["labels"]["halluc"][i] == IGNORE_INDEX


def test_every_sample_has_one_label_across_a_batch() -> None:
    """The prompt-level use case: one supervised position per conversation."""
    tok = ChatMLTok()
    convos = [
        _conv(user_labels={"answ": [{"at": "after", "index": -1, "label": int(i % 2)}]})
        for i in range(4)
    ]
    samples = build_dataset(convos, tok, default_label=0)
    assert [len(_labeled(s)) for s in samples] == [1, 1, 1, 1]
    assert [s["labels"][_labeled(s)[0]] for s in samples] == [0, 1, 0, 1]


@pytest.mark.real_model
def test_real_template_places_the_gap_token() -> None:
    """Against a real tokenizer, the label lands on the token before the answer."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct", local_files_only=True
        )
    except Exception as exc:  # pragma: no cover - environment without the cache
        pytest.skip(f"tokenizer unavailable: {exc}")
    convo = [
        {"role": "user", "content": "Is the answer here?",
         "labels": {"answ": [{"at": "after", "index": -1, "label": 1}]}},
        {"role": "assistant", "content": "Yes it is."},
    ]
    sample = build_dataset([convo], tok, default_label=0)[0]
    positions = _labeled(sample)
    assert len(positions) == 1
    answer_ids = tok.encode("Yes", add_special_tokens=False)
    assert sample["tokens"][positions[0] + 1] == answer_ids[0]


# ---------------------------------------------------------------------------
# The label has to survive the whole pipeline, on both backends
# ---------------------------------------------------------------------------


class SmallVocabTok(ChatMLTok):
    """ChatML-ish tokenizer whose ids fit a tiny model's embedding table."""

    def _id(self, word: str) -> int:
        return self.vocab.setdefault(word, 1 + len(self.vocab) % 60)


def _tiny_model(backend: str, vocab_size: int = 64, hidden: int = 16, layers: int = 4) -> Any:
    """A minimal model of the shape the Model wrapper expects, per backend."""
    if backend == "torch":
        torch = pytest.importorskip("torch")
        tnn = torch.nn

        class TorchTiny(tnn.Module):
            """Embedding -> stack of linears -> vocab logits."""

            def __init__(self) -> None:
                super().__init__()
                self.embedding = tnn.Embedding(vocab_size, hidden)
                self.layers = tnn.ModuleList(
                    [tnn.Linear(hidden, hidden) for _ in range(layers)]
                )
                self.output_proj = tnn.Linear(hidden, vocab_size)

            def forward(self, x: Any, **_: Any) -> Any:
                h = self.embedding(x)
                for layer in self.layers:
                    h = torch.nn.functional.gelu(layer(h))
                return self.output_proj(h)

        torch.manual_seed(0)
        return TorchTiny()

    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class MlxTiny(nn.Module):
        """The same network under MLX."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden)
            self.layers = [nn.Linear(hidden, hidden) for _ in range(layers)]
            self.output_proj = nn.Linear(hidden, vocab_size)

        def __call__(self, x: Any, **_: Any) -> Any:
            h = self.embedding(x)
            for layer in self.layers:
                h = nn.gelu(layer(h))
            return self.output_proj(h)

    mx.random.seed(0)
    return MlxTiny()


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_a_pinned_label_trains_on_both_backends(backend: str, tmp_path: Any) -> None:
    """One label per conversation, at a template position, reaches the probe loss.

    The label array is mostly -100 by construction, which is the case a loss that
    averages over "all tokens" gets wrong; and the scored position must be the one
    the spec pinned, not the last token of the sequence.
    """
    from auto_chasm import Model, ProbeConfig, Trainer
    from auto_chasm.data import build_dataset
    from auto_chasm.dataset import Dataset
    from auto_chasm.trainers.loss import JointLoss

    tok = SmallVocabTok()
    convos = [
        _conv(user_labels={"answ": [{"at": "after", "index": -1, "label": i % 2}]})
        for i in range(8)
    ]
    samples = build_dataset(convos, tok, default_label=0)
    assert all(len(_labeled(s)) == 1 for s in samples)

    base = _tiny_model(backend)

    class Config:
        """Minimal config the Model wrapper reads."""

        hidden_size = 16
        num_hidden_layers = 4

    base.config = Config()
    model = Model(base, tok, backend_name=backend)
    model.add_probes([ProbeConfig(name="answ", layers=[2], module_config={"out_features": 1})])
    data = Dataset(samples, tokenizer=tok)
    trainer = Trainer(
        model, JointLoss(weights={"lm_head": 1.0, "answ": 1.0}),
        num_iters=2, batch_size=2, learning_rate=1e-3, max_seq_length=64,
        logging_steps=1, output_dir=str(tmp_path), save_history=False,
    )
    losses = trainer.train(data)["history"].train_losses
    assert losses and all(float(x) == float(x) for x in losses)    # finite, not NaN

    scores = model.probe_scores(data, batch_size=2)
    labels_out = scores.labels["answ"] if isinstance(scores.labels, dict) else scores.labels
    # Exactly one scored position per conversation, carrying the pinned label.
    assert len(list(labels_out)) == len(convos)
    assert sorted({int(v) for v in labels_out}) == [0, 1]

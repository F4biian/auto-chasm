"""Training data must be rendered the way the model is prompted at inference.

``build_dataset`` used to concatenate raw message text, so a three-turn chat
tokenized to ``'You are helpful.What is 2+2?It is four.'`` -- no role markers, no
turn delimiters -- while generation used the full template. A model fine-tuned on
one format and sampled from the other never sees its training distribution.

The hard part is that span labels are CHARACTER offsets into ``msg["content"]``,
so the scaffolding has to be spliced around separately-tokenized content rather
than rendered into one string.
"""

from __future__ import annotations

from typing import Any

import pytest

from auto_chasm._chat_template import (
    has_chat_template,
    message_wrappers,
    set_default_thinking,
    template_kwargs,
)


class _Tok:
    """Minimal chat tokenizer: ``<s>{role}:{content}</s>`` per turn."""

    chat_template = "dummy"

    def apply_chat_template(
        self, messages: list[dict[str, Any]], tokenize: bool = False,
        add_generation_prompt: bool = False, **kw: Any
    ) -> str:
        out = "".join(f"<s>{m['role']}:{m['content']}</s>" for m in messages)
        if add_generation_prompt:
            out += "<s>assistant:"
            if kw.get("enable_thinking") is False:
                out += "<think></think>"
        return out


def test_wrappers_reconstruct_the_template_exactly() -> None:
    tok = _Tok()
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    wrappers, _ = message_wrappers(tok, msgs)
    rebuilt = "".join(p + m["content"] + s for (p, s), m in zip(wrappers, msgs, strict=True))
    assert rebuilt == tok.apply_chat_template(msgs)


def test_closing_tag_belongs_to_the_turn_it_closes() -> None:
    """The assistant must keep its own end tag, or it never learns to stop."""
    tok = _Tok()
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    wrappers, _ = message_wrappers(tok, msgs)
    assert wrappers[-1][1].startswith("</s>")


def test_generation_prompt_is_returned_separately() -> None:
    tok = _Tok()
    msgs = [{"role": "user", "content": "hi"}]
    _, trailing = message_wrappers(tok, msgs, add_generation_prompt=True)
    assert trailing == "<s>assistant:"


def test_template_that_drops_content_is_rejected() -> None:
    """Silently mislabelling is worse than failing."""

    class Dropping(_Tok):
        def apply_chat_template(self, messages: Any, **kw: Any) -> str:
            return "<s>nothing</s>"

    with pytest.raises(ValueError, match="did not round-trip"):
        message_wrappers(Dropping(), [{"role": "user", "content": "hi"}])


def test_no_template_is_detected() -> None:
    class Bare:
        chat_template = None

    assert has_chat_template(Bare()) is False
    assert has_chat_template(_Tok()) is True


def test_empty_conversation_is_not_an_error() -> None:
    assert message_wrappers(_Tok(), []) == ([], "")


# --- the unified reasoning switch -------------------------------------------


def test_explicit_argument_beats_the_global() -> None:
    try:
        set_default_thinking(False)
        assert template_kwargs(True) == {"enable_thinking": True}
        assert template_kwargs(None) == {"enable_thinking": False}
    finally:
        set_default_thinking(None)


def test_global_none_leaves_the_template_alone() -> None:
    set_default_thinking(None)
    assert template_kwargs(None) == {}

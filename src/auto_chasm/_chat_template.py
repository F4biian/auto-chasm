r"""Chat-template rendering that preserves per-message character offsets.

``build_dataset`` labels tokens from CHARACTER SPANS into ``msg["content"]``. A
naive ``apply_chat_template(conversation)`` returns one string in which those
offsets no longer mean anything, which is why the dataset builder originally
concatenated raw contents and emitted no role markers at all::

    [system, user, assistant]  ->  'You are helpful.What is 2+2?It is four.'

That trains a chat model on a format it never sees at inference, where the same
turns are rendered ``<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n``.

This module keeps both properties. It asks the tokenizer to render the
conversation with each message's content replaced by a unique SENTINEL, then
splits the result on those sentinels. What falls between them is exactly the
template's own scaffolding, per message::

    ('<|im_start|>user\\n', '<|im_end|>\\n')

The content is then tokenized on its own, so span offsets stay valid, and the
scaffolding is tokenized separately and spliced around it. Nothing about the
template is hardcoded, so this works for any model whose template round-trips a
sentinel (Qwen, Llama, Gemma, Mistral, ...).

**Tokenization at the seams.** Tokenizing ``prefix``, ``content`` and ``suffix``
separately can differ from tokenizing their concatenation, because BPE may merge
across a boundary. Here the boundaries are special tokens and newlines, which do
not merge with ordinary text in the templates we support -- :func:`verify_render`
checks exactly this and is used by the tests.
"""

from __future__ import annotations

from typing import Any

from auto_chasm.logger import get_logger

logger = get_logger(__name__)

#: Placeholder substituted for each message's content while probing the template.
#: Must survive templating unchanged and never appear in real text.
_SENTINEL = "\x00\x00AUTOCHASM{}\x00\x00"


def has_chat_template(tokenizer: Any) -> bool:
    """Whether ``tokenizer`` can render chat turns."""
    return bool(getattr(tokenizer, "chat_template", None)) and hasattr(
        tokenizer, "apply_chat_template"
    )


#: Process-wide fallback for ``enable_thinking``; see :func:`set_default_thinking`.
_DEFAULT_THINKING: bool | None = None


def set_default_thinking(enable: bool | None) -> None:
    """Set reasoning mode ONCE for dataset building and generation alike.

    Every call site takes an ``enable_thinking`` argument, but they are far apart
    (data prep, generation, evaluation) and the template's own default cannot be
    relied on: for Qwen3.5 a plain ``AutoTokenizer`` renders a CLOSED
    ``<think></think>`` block (reasoning off) while mlx-lm's ``TokenizerWrapper``
    leaves it OPEN (reasoning on) -- from the identical template string. Training
    with one and generating with the other is silent and costly.

    Args:
        enable: ``False`` to suppress reasoning, ``True`` to request it, ``None``
            to defer to each template's own default.
    """
    global _DEFAULT_THINKING
    _DEFAULT_THINKING = enable


def get_default_thinking() -> bool | None:
    """The current process-wide reasoning default (see :func:`set_default_thinking`)."""
    return _DEFAULT_THINKING


def template_kwargs(enable_thinking: bool | None) -> dict[str, Any]:
    """Keyword arguments selecting a model's reasoning mode.

    Falls back to :func:`set_default_thinking` when the caller passes ``None``,
    so one setting covers data prep and generation.
    """
    resolved = _DEFAULT_THINKING if enable_thinking is None else enable_thinking
    return {} if resolved is None else {"enable_thinking": resolved}


def message_wrappers(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool | None = None,
) -> tuple[list[tuple[str, str]], str]:
    r"""Per-message ``(prefix, suffix)`` scaffolding, plus any trailing prompt.

    ONE render of the whole conversation with sentinel contents, split on the
    sentinels. Incremental rendering (apply the template to the first ``k``
    messages, diff against ``k+1``) would separate a message's closing tag from
    the next message's opener exactly, but many templates refuse partial
    conversations -- Qwen3.5 raises ``No user query found in messages`` for a
    system-only prefix -- so it is not available in general.

    The text BETWEEN two sentinels therefore belongs to two messages at once
    (``'<|im_end|>\n<|im_start|>assistant\n'`` closes one turn and opens the
    next). It is assigned wholly to the earlier message's SUFFIX, so that an
    assistant turn keeps its own closing tag: with ``lm_train_on="assistant"``
    the model still learns to emit ``<|im_end|>`` and stop, which is the failure
    mode that matters (a model that never learns to stop runs to max_tokens).

    The cost is that when an assistant turn is FOLLOWED by another turn, the next
    turn's opener is also LM-trained -- a couple of structural tokens the model
    would emit anyway. Single-turn data (prompt + one reply, the usual SFT shape)
    is unaffected, because the assistant message is last and its suffix is just
    the closing tag.

    Args:
        tokenizer: Tokenizer exposing ``apply_chat_template``.
        messages: The conversation; only ``role`` is read (content is replaced).
        add_generation_prompt: Append the assistant-turn opener after the last
            message (what generation uses).
        enable_thinking: Reasoning mode; see :func:`template_kwargs`.

    Returns:
        ``(wrappers, trailing)`` where ``wrappers[i]`` brackets message ``i`` and
        ``trailing`` is the generation prompt when requested, else ``""``.

    Raises:
        ValueError: If the template did not round-trip a sentinel, so the split
            cannot be trusted.
    """
    if not messages:
        return [], ""
    probes = [{**m, "content": _SENTINEL.format(i)} for i, m in enumerate(messages)]
    kw = template_kwargs(enable_thinking)
    rendered = str(
        tokenizer.apply_chat_template(probes, tokenize=False, add_generation_prompt=False, **kw)
    )

    parts: list[str] = []
    rest = rendered
    for i in range(len(messages)):
        head, sep, rest = rest.partition(_SENTINEL.format(i))
        if not sep:
            raise ValueError(
                f"The chat template did not round-trip message {i}'s content, so its "
                "scaffolding cannot be located. Pass chat_template=False to build the "
                "dataset from raw message text instead."
            )
        parts.append(head)
    parts.append(rest)  # tail after the final sentinel

    # parts[i] is the glue BEFORE message i; parts[-1] is the tail after the last.
    # Message 0 owns the leading preamble (BOS / injected system turn); every other
    # message's opener is folded into the previous message's suffix (see docstring).
    wrappers = [
        (parts[0] if i == 0 else "", parts[i + 1]) for i in range(len(messages))
    ]

    trailing = ""
    if add_generation_prompt:
        with_gen = str(
            tokenizer.apply_chat_template(
                probes, tokenize=False, add_generation_prompt=True, **kw
            )
        )
        if with_gen.startswith(rendered):
            trailing = with_gen[len(rendered) :]
        else:  # template rewrites the tail; recover it after the last sentinel
            trailing = with_gen.rpartition(_SENTINEL.format(len(messages) - 1))[2]
            wrappers[-1] = (wrappers[-1][0], "")
    return wrappers, trailing


def verify_render(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tokens: list[int],
    *,
    enable_thinking: bool | None = None,
) -> tuple[bool, str, str]:
    """Check that ``tokens`` decode to what the template would produce.

    Returns:
        ``(matches, decoded, expected)``. Used by the tests to prove the splice
        is faithful rather than merely plausible.
    """
    expected = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        **template_kwargs(enable_thinking),
    )
    decoded = tokenizer.decode(tokens)
    return decoded == expected, decoded, expected

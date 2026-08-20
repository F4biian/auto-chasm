"""Pure helpers for text generation — prompt resolution, stop control, guards.

Split out of :mod:`auto_chasm.generation` so that module stays focused on the
backend generation loops.  Everything here is backend-free (no ``mlx``/``torch``
imports) and re-exported from ``auto_chasm.generation`` for backwards
compatibility, so ``from auto_chasm.generation import _resolve_prompt`` keeps
working.
"""

from __future__ import annotations

from typing import Any

from auto_chasm.logger import get_logger

logger = get_logger(__name__)


def _resolve_prompt(
    tokenizer: Any,
    prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
) -> str:
    """Resolve a prompt string from either a direct string or chat messages.

    If ``messages`` is provided, the tokenizer's ``apply_chat_template``
    is used to format it.  Raises ``ValueError`` if the tokenizer has
    no chat template.

    Args:
        tokenizer: The tokenizer.
        prompt: Direct prompt string (ignored if ``messages`` is set).
        messages: Chat messages as ``[{"role": ..., "content": ...}, ...]``.

    Returns:
        A formatted prompt string.

    Raises:
        ValueError: If neither prompt nor messages is provided, or if
            the tokenizer lacks a chat template when messages are used.
    """
    if messages is not None:
        tok = tokenizer
        if not hasattr(tok, "apply_chat_template") or not getattr(tok, "chat_template", None):
            raise ValueError(
                "Tokenizer has no chat_template. Pass a string prompt instead of messages."
            )
        # Same reasoning mode as the training data: a model fine-tuned with the
        # <think> block CLOSED and then sampled with it OPEN sees a prompt shape
        # it never trained on.
        from auto_chasm._chat_template import template_kwargs

        return tok.apply_chat_template(  # type: ignore[no-any-return]
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs(None),
        )
    if prompt is not None:
        return prompt
    raise ValueError("Either prompt or messages must be provided.")


def _auto_stop_tokens(tokenizer: Any) -> list[int] | None:
    """Best-effort scan of the tokenizer vocab for common turn-end tokens.

    Chat models end a turn with a dedicated token (``<|im_end|>``,
    ``<end_of_turn>``, ...) rather than the bare ``eos_token_id``.  Detecting it
    lets the manual/streaming loops stop at the end of a turn without the caller
    naming it.  Returns ``None`` when nothing extra is found.

    Args:
        tokenizer: The tokenizer (``get_vocab`` is used if present).

    Returns:
        A list of extra stop token IDs, or ``None``.
    """
    auto_stop: list[int] = []
    turn_patterns = ("<end_of_turn>", "<|im_end|>", "<|end|>", "<|endoftext|>", "</s>")
    vocab = getattr(tokenizer, "get_vocab", None)
    if vocab:
        try:
            v = vocab()
            if isinstance(v, dict):
                for pattern in turn_patterns:
                    tid = v.get(pattern)
                    if tid is not None and tid != tokenizer.eos_token_id:
                        auto_stop.append(tid)
        except Exception:
            pass
    return auto_stop or None


def _extract_stop_tokens(tokenizer: Any, kwargs: dict[str, Any]) -> list[int] | None:
    """Explicit ``stop_tokens`` (token IDs) if given, else auto-detected turn tokens.

    ``stop_sequences`` (arbitrary strings) are intentionally NOT handled here: a
    multi-token string is matched as a substring of the decoded text by the
    caller, because reducing it to a set of single stop-token IDs would fire on
    each constituent token wherever it appears.

    Args:
        tokenizer: The tokenizer.
        kwargs: Generation kwargs; ``stop_tokens`` is popped if present.

    Returns:
        A list of stop token IDs, or ``None`` for eos-only stopping.
    """
    explicit = kwargs.pop("stop_tokens", None)
    if explicit is not None:
        return list(explicit)
    return _auto_stop_tokens(tokenizer)


def _earliest_stop_index(text: str, stop_sequences: list[str] | None) -> int | None:
    """Start index of the earliest stop-sequence occurrence in ``text``, or ``None``.

    Args:
        text: Decoded text generated so far.
        stop_sequences: Substrings that end generation; empty/``None`` means no
            substring stopping.

    Returns:
        The start index of the earliest-occurring stop sequence, or ``None``.
    """
    best: int | None = None
    for seq in stop_sequences or ():
        if not seq:
            continue
        idx = text.find(seq)
        if idx != -1 and (best is None or idx < best):
            best = idx
    return best


def _repeat_guard_triggered(same_token_count: int, max_repeat: int | None) -> bool:
    """Decide whether the repetition guard should stop generation.

    The guard is non-silent: when it fires it logs a warning so a truncated
    generation is never mistaken for a natural stop.  A ``max_repeat`` of
    ``None`` disables the guard entirely.

    Args:
        same_token_count: Number of consecutive identical tokens seen so far.
        max_repeat: Consecutive-repeat cap, or ``None`` to disable the guard.

    Returns:
        ``True`` if generation should stop due to repetition.
    """
    if max_repeat is None:
        return False
    if same_token_count >= max_repeat:
        logger.warning(
            "Repetition guard: stopped after %d consecutive identical tokens "
            "(max_repeat=%d). Pass max_repeat=None to disable or a higher value.",
            same_token_count,
            max_repeat,
        )
        return True
    return False


def _check_temperature(temperature: float) -> None:
    """Validate a sampling temperature for the manual generation paths.

    A negative temperature would pass the ``temperature > 0`` sampling check
    yet make ``logits / temperature`` invert the distribution, silently
    sampling the *least* likely tokens.  A ``NaN`` temperature also passes every
    comparison and falls through to greedy.  Reject both explicitly so callers
    get a clear error instead of wrong numbers.  ``temperature == 0`` is valid
    and selects greedy decoding.

    Args:
        temperature: Sampling temperature.

    Raises:
        ValueError: If ``temperature`` is negative or ``NaN``.
    """
    if not temperature >= 0:  # False for negatives AND for NaN
        raise ValueError(
            f"temperature must be >= 0 (0 = greedy), got {temperature!r}. "
            "A negative temperature would invert the sampling distribution."
        )


def reject_num_return_sequences(kwargs: dict[str, Any]) -> None:
    """Reject any ``num_return_sequences`` other than 1 — generation returns one string.

    Silently generating and discarding the extra sequences (the old behaviour on both
    backends) wastes compute and hides that only one result comes back.  Uses ``!= 1``
    rather than ``int(n) > 1`` so a fractional ``1.5`` (``int`` -> 1) or ``0`` is caught
    too, not silently accepted and dropped.

    Args:
        kwargs: Generation kwargs; ``num_return_sequences`` is inspected, not popped.

    Raises:
        ValueError: If ``num_return_sequences`` is set to anything other than 1.
    """
    n = kwargs.get("num_return_sequences")
    if n is not None and n != 1:
        raise ValueError(
            f"num_return_sequences={n!r} is not supported: generate()/generate_stream() "
            "return a single sequence. Call generate() in a loop for multiple samples."
        )


def check_sampling_params(kwargs: dict[str, Any]) -> None:
    """Validate the sampling controls so a degenerate value fails loudly, not silently.

    ``repetition_penalty <= 0`` silently no-ops (falsy 0) or inverts logits (negative);
    ``top_p`` outside ``(0, 1]`` silently kept the whole vocab; a bad ``top_k`` mis-slices.

    Args:
        kwargs: Generation kwargs; ``repetition_penalty``/``top_p``/``top_k`` are read.

    Raises:
        ValueError: If any sampling control is out of range.
    """
    rep = kwargs.get("repetition_penalty")
    if rep is not None and rep <= 0:
        raise ValueError(f"repetition_penalty must be > 0 (1.0 = no penalty), got {rep!r}.")
    top_p = kwargs.get("top_p")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p!r}.")
    top_k = kwargs.get("top_k")
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k!r}.")


def _softmax_np(logits: Any) -> Any:
    """Numerically-stable softmax of a 1-D numpy array."""
    import numpy as np

    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def filter_logits(
    logits: Any,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float | None,
    generated: list[int],
) -> Any:
    """Apply repetition penalty, temperature, top-k and top-p to a 1-D logit vector.

    Backend-free (numpy) so the MLX and PyTorch fallback loops share one correct
    implementation instead of silently ignoring these sampling controls.  The
    order matches HuggingFace: repetition penalty, then temperature, then top-k,
    then top-p (nucleus).  Greedy decoding (``temperature == 0``) applies only the
    repetition penalty — top-k/top-p do not affect an argmax — and the caller
    takes the argmax of the returned logits.

    Args:
        logits: A 1-D array of next-token logits (numpy or convertible).
        temperature: Sampling temperature (0 = greedy).
        top_k: Keep only the ``top_k`` highest-logit tokens, or ``None``.
        top_p: Nucleus threshold in ``(0, 1)``, or ``None``.
        repetition_penalty: HF-style penalty (>1 discourages seen tokens), or ``None``.
        generated: Token IDs seen so far, penalised by ``repetition_penalty``.

    Returns:
        The processed logits as a ``float64`` numpy array.
    """
    import numpy as np

    out = np.asarray(logits, dtype=np.float64).copy()
    if repetition_penalty and repetition_penalty != 1.0 and len(generated) > 0:
        uniq = np.unique(np.asarray(generated, dtype=np.int64))
        uniq = uniq[(uniq >= 0) & (uniq < out.shape[-1])]  # ignore any out-of-range id
        vals = out[uniq]
        out[uniq] = np.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
    if not (temperature and temperature > 0):
        return out  # greedy: top-k/top-p are irrelevant to argmax
    out = out / temperature
    if top_k:
        k = min(int(top_k), out.shape[-1])
        if k > 0:
            kth = np.partition(out, -k)[-k]
            out[out < kth] = -np.inf
    if top_p is not None and 0.0 < top_p < 1.0:
        order = np.argsort(out)[::-1]  # descending
        cumulative = np.cumsum(_softmax_np(out[order]))
        remove = cumulative > top_p
        remove[1:] = remove[:-1].copy()  # keep the token that crosses the threshold
        remove[0] = False
        out[order[remove]] = -np.inf
    return out


def stream_flush(pending: str, stop_sequences: list[str] | None) -> tuple[str, str, bool]:
    """Split a streaming buffer into (safe-to-emit, still-pending, stop-now).

    Ensures a stop sequence never leaks into streamed output: a completed stop
    ends generation with only the text before it emitted, and any trailing suffix
    that could still begin a stop sequence is withheld until the next piece.

    Args:
        pending: Decoded text not yet emitted.
        stop_sequences: Stop strings, or ``None`` for no substring stopping.

    Returns:
        ``(emit, still_pending, stop_now)`` — text to yield now, text to carry
        forward, and whether generation should stop.
    """
    if not stop_sequences:
        return pending, "", False
    cut = _earliest_stop_index(pending, stop_sequences)
    if cut is not None:
        return pending[:cut], "", True
    hold = held_back_len(pending, stop_sequences)
    if hold == 0:
        return pending, "", False
    keep = len(pending) - hold
    return pending[:keep], pending[keep:], False


def held_back_len(buffer: str, stop_sequences: list[str] | None) -> int:
    """Length of the longest suffix of ``buffer`` that is a proper prefix of a stop.

    Streaming must not emit text that could still turn out to be the start of a
    stop sequence, or a partial stop leaks before the full match triggers.  That
    trailing suffix is held back until the next decoded piece resolves it.  A
    *complete* stop match (the whole sequence) is not held here — the caller
    detects and cuts on it separately.

    Args:
        buffer: The not-yet-emitted decoded text.
        stop_sequences: Stop strings, or ``None``.

    Returns:
        Number of trailing characters to withhold (0 if none could start a stop).
    """
    hold = 0
    for seq in stop_sequences or ():
        if not seq:
            continue
        limit = min(len(buffer), len(seq) - 1)  # a full match is handled by the caller
        for k in range(limit, 0, -1):
            if buffer[-k:] == seq[:k]:
                hold = max(hold, k)
                break
    return hold


def _strip_leading_space(text: str) -> str:
    """Strip a single leading space from manually-decoded output.

    ``mlx_lm``'s non-streaming path drops the leading space its detokenizer
    emits before the first sub-word, but a manual ``tokenizer.decode`` keeps
    it.  Stripping one leading space makes the manual/stream output match the
    non-stream path for the same greedy prompt.

    Args:
        text: Decoded continuation text.

    Returns:
        The text with at most one leading space removed.
    """
    return text[1:] if text.startswith(" ") else text

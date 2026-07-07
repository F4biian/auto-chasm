"""Text generation — generate, generate_stream, chat.

Provides a transformers-like generation API that works out of the box
with any base model.  Steering is applied transparently through the
patched layers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from auto_chasm._gen_cache import MlxDecoder, TorchDecoder
from auto_chasm._gen_sampling import next_token_mlx, next_token_torch
from auto_chasm._generation_utils import (
    _auto_stop_tokens,
    _check_temperature,
    _earliest_stop_index,
    _extract_stop_tokens,
    _repeat_guard_triggered,
    _resolve_prompt,
    _strip_leading_space,
    check_sampling_params,
    reject_num_return_sequences,
    stream_flush,
)
from auto_chasm._mlx_compat import ensure_mlx_lm_compat

# Re-exported for backwards compatibility (callers import these from here).
__all__ = [
    "_check_temperature",
    "_extract_stop_tokens",
    "_repeat_guard_triggered",
    "_resolve_prompt",
    "chat",
    "chat_repl",
    "generate",
    "generate_stream",
]

# Default repetition guard: cap on consecutive identical tokens before stopping (None disables).
DEFAULT_MAX_REPEAT = 256


@contextmanager
def _eval_mode(model: Any) -> Iterator[None]:
    """Put a torch/mlx model in eval mode for inference, restoring the prior state.

    Inference must not apply LoRA (or any) dropout: a model left in train mode
    after a fine-tune would sample through active dropout, making generation
    non-deterministic and degrading quality.  Restores the previous ``training``
    flag afterward so a mid-training generation (e.g. a logging sample) does not
    silently switch the model to eval.  A no-op for objects without the API.

    Args:
        model: The framework model (or anything; non-modules pass through).

    Yields:
        Nothing; used purely for its enter/exit side effects.
    """
    was_training = getattr(model, "training", None)
    set_eval = getattr(model, "eval", None)
    if callable(set_eval):
        set_eval()
    try:
        yield
    finally:
        if was_training and hasattr(model, "train"):
            model.train()


def _torch_model_device(model: Any) -> Any:
    """Best-effort device for a torch ``model`` (its ``.device`` or first param, else CPU)."""
    dev = getattr(model, "device", None)
    if dev is not None:
        return dev
    try:
        return next(iter(model.parameters())).device
    except (StopIteration, AttributeError, TypeError):
        return "cpu"


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    backend: Any = None,
    max_repeat: int | None = DEFAULT_MAX_REPEAT,
    use_cache: bool = True,
    **kwargs: Any,
) -> str:
    """Generate text from a prompt.

    Works with both MLX and PyTorch models.  If steering hooks are
    installed, they are applied transparently during generation.

    Args:
        model: The base language model.
        tokenizer: The tokenizer.
        prompt: Input text prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        backend: The backend instance.
        max_repeat: Consecutive-identical-token cap for the repetition guard.
            ``None`` disables the guard (generate up to ``max_tokens``).  When
            it triggers, a warning is logged rather than stopping silently.
        use_cache: Use an incremental KV cache for O(n) decoding (default). The
            primary mlx_lm/HF paths cache regardless; this controls the manual /
            streaming loops, which fall back to a full re-forward when unsupported.
        **kwargs: Additional generation arguments.

    Returns:
        Generated text string.
    """
    _check_temperature(temperature)  # guard ALL paths (the mlx_lm/HF primaries too)
    reject_num_return_sequences(kwargs)
    check_sampling_params(kwargs)
    kwargs["use_cache"] = use_cache
    if backend is None:
        from auto_chasm.backends.loaders import detect_backend

        backend_name = detect_backend()
    else:
        backend_name = backend.name

    with _eval_mode(model):  # disable LoRA/dropout for deterministic inference
        if backend_name == "mlx":
            return _generate_mlx(
                model, tokenizer, prompt, max_tokens, temperature, max_repeat=max_repeat, **kwargs
            )
        return _generate_torch(
            model, tokenizer, prompt, max_tokens, temperature, max_repeat=max_repeat, **kwargs
        )


def generate_stream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    backend: Any = None,
    max_repeat: int | None = DEFAULT_MAX_REPEAT,
    use_cache: bool = True,
    **kwargs: Any,
) -> Iterator[str]:
    """Stream generated tokens one at a time.

    Args:
        model: The base language model.
        tokenizer: The tokenizer.
        prompt: Input text prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        backend: The backend instance.
        max_repeat: Consecutive-identical-token cap for the repetition guard.
            ``None`` disables the guard.  Logs a warning when it triggers.
        use_cache: Use an incremental KV cache for O(n) streaming (default); falls
            back to a full re-forward when the model does not support one.
        **kwargs: Additional generation arguments.

    Yields:
        Individual token strings.
    """
    _check_temperature(temperature)  # guard ALL paths (the mlx_lm/HF primaries too)
    reject_num_return_sequences(kwargs)
    check_sampling_params(kwargs)
    kwargs["use_cache"] = use_cache
    if backend is None:
        from auto_chasm.backends.loaders import detect_backend

        backend_name = detect_backend()
    else:
        backend_name = backend.name

    with _eval_mode(model):  # disable LoRA/dropout for deterministic inference
        if backend_name == "mlx":
            yield from _generate_stream_mlx(
                model, tokenizer, prompt, max_tokens, temperature, max_repeat=max_repeat, **kwargs
            )
        else:
            yield from _generate_stream_torch(
                model, tokenizer, prompt, max_tokens, temperature, max_repeat=max_repeat, **kwargs
            )


def chat(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.0,
    backend: Any = None,
    **kwargs: Any,
) -> str:
    """Generate a response in a chat conversation.

    Applies the tokenizer's chat template if available.

    Args:
        model: The base language model.
        tokenizer: The tokenizer.
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        backend: The backend instance.
        **kwargs: Additional generation arguments.

    Returns:
        Generated response text.
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = (
            "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
            + "\nAssistant:"
        )

    return generate(model, tokenizer, prompt, max_tokens, temperature, backend, **kwargs)


def chat_repl(
    model: Any,
    tokenizer: Any,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    backend: Any = None,
) -> None:
    """Start an interactive chat REPL.

    Maintains a multi-turn conversation in memory.  Type
    ``"exit"`` or ``"quit"`` to stop.  ``Ctrl-C`` or ``Ctrl-D``
    also exits gracefully.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        system_prompt: Optional system prompt prepended to conversation.
        max_tokens: Maximum tokens per response.
        temperature: Sampling temperature.
        backend: Backend instance.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    print("Chat REPL — type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        messages.append({"role": "user", "content": user_input})

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = (
                "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
                + "\nAssistant:"
            )

        response = generate(model, tokenizer, prompt, max_tokens, temperature, backend)
        messages.append({"role": "assistant", "content": response})
        print(f"\nAssistant: {response}")


def _generate_mlx(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    max_repeat: int | None = DEFAULT_MAX_REPEAT,
    **kwargs: Any,
) -> str:
    """MLX generation implementation."""
    # mlx_lm.generate honours only the model's own eos tokens, so stop control routes to
    # the manual loop (exact token ids via stop_tokens, or decoded substrings).
    explicit_stop = kwargs.pop("stop_tokens", None)
    stop_sequences = kwargs.pop("stop_sequences", None)
    # Extract sampling params up front so BOTH the manual loop and the mlx_lm
    # sampler honour them (the manual loop previously ignored top_p/top_k/rep).
    top_p = kwargs.pop("top_p", None)
    top_k = kwargs.pop("top_k", None)
    rep = kwargs.pop("repetition_penalty", None)
    kwargs.pop("num_return_sequences", None)
    use_cache = kwargs.pop("use_cache", True)
    # Route to the manual full-forward loop for stop control OR when caching is off (e.g.
    # steering): mlx_lm.generate always caches internally and can't honour use_cache=False.
    if explicit_stop or stop_sequences or not use_cache:
        return _generate_manual_mlx(
            model,
            tokenizer,
            prompt,
            max_tokens,
            temperature,
            stop_tokens=list(explicit_stop) if explicit_stop else None,
            stop_sequences=list(stop_sequences) if stop_sequences else None,
            max_repeat=max_repeat,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep,
            use_cache=use_cache,
        )
    ensure_mlx_lm_compat()
    try:
        from mlx_lm import generate as mlx_generate

        gen_kwargs: dict[str, Any] = {}
        # Translate sampling params into mlx_lm constructs (raw forwarding would be ignored).
        kwargs.pop("do_sample", None)
        if temperature > 0 or top_p is not None or top_k is not None:
            from mlx_lm.generate import make_sampler

            sampler_kw: dict[str, Any] = {}
            if top_p is not None:
                sampler_kw["top_p"] = top_p
            if top_k is not None:
                sampler_kw["top_k"] = top_k
            gen_kwargs["sampler"] = make_sampler(max(temperature, 1e-6), **sampler_kw)
        if rep is not None and rep != 1.0:
            from mlx_lm.generate import make_logits_processors

            gen_kwargs["logits_processors"] = make_logits_processors(repetition_penalty=rep)
        return mlx_generate(  # type: ignore[no-any-return]
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            **gen_kwargs,
            **kwargs,
        )
    except Exception:
        # mlx_lm unavailable: manual path, honouring auto-detected turn tokens.
        return _generate_manual_mlx(
            model,
            tokenizer,
            prompt,
            max_tokens,
            temperature,
            stop_tokens=_auto_stop_tokens(tokenizer),
            max_repeat=max_repeat,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep,
            use_cache=use_cache,
        )


def _generate_manual_mlx(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop_tokens: list[int] | None = None,
    stop_sequences: list[str] | None = None,
    max_repeat: int | None = DEFAULT_MAX_REPEAT,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    use_cache: bool = True,
) -> str:
    """Manual MLX generation (fallback when mlx_lm is unavailable).

    Args:
        model: The base language model.
        tokenizer: The tokenizer.
        prompt: Input text prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        stop_tokens: Additional token IDs that should stop generation
            (e.g., ``<end_of_turn>``).  ``None`` uses only ``eos_token_id``.
        stop_sequences: Strings that stop generation when they appear in the
            decoded output; the returned text is truncated before the match.
        max_repeat: Consecutive-identical-token cap for the repetition guard.
            ``None`` disables it; logs a warning when it triggers.
        top_k: Keep only the ``top_k`` highest-logit tokens when sampling.
        top_p: Nucleus-sampling threshold in ``(0, 1)``.
        repetition_penalty: HF-style penalty (>1 discourages repeats).
        use_cache: Use an incremental KV cache (O(n) not O(n^2)); falls back to
            full-forward when the model does not support one.
    """
    _check_temperature(temperature)
    prompt_len = len(tokenizer.encode(prompt))
    tokens = tokenizer.encode(prompt)
    eos_id = tokenizer.eos_token_id
    stop_ids: set[int] = {eos_id}
    if stop_tokens:
        stop_ids.update(stop_tokens)
    same_token_count = 0
    prev_token: int | None = None
    decoder = MlxDecoder(model, use_cache)

    for _ in range(max_tokens):
        next_logits = decoder.next_logits(tokens)
        next_token = next_token_mlx(
            next_logits, temperature, top_k, top_p, repetition_penalty, tokens
        )

        if next_token in stop_ids:
            break

        if next_token == prev_token:
            same_token_count += 1
        else:
            same_token_count = 0
        if _repeat_guard_triggered(same_token_count, max_repeat):
            break

        prev_token = next_token
        tokens.append(next_token)
        if stop_sequences:
            text = tokenizer.decode(tokens[prompt_len:])
            cut = _earliest_stop_index(text, stop_sequences)
            if cut is not None:
                return _strip_leading_space(text[:cut])

    return _strip_leading_space(tokenizer.decode(tokens[prompt_len:]))


def _generate_stream_mlx(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    max_repeat: int | None = DEFAULT_MAX_REPEAT,
    **kwargs: Any,
) -> Iterator[str]:
    """Streaming MLX generation."""
    _check_temperature(temperature)
    stop_tokens = _extract_stop_tokens(tokenizer, kwargs)
    stop_sequences = kwargs.pop("stop_sequences", None)
    top_k = kwargs.pop("top_k", None)
    top_p = kwargs.pop("top_p", None)
    rep = kwargs.pop("repetition_penalty", None)
    kwargs.pop("num_return_sequences", None)
    decoder = MlxDecoder(model, kwargs.pop("use_cache", True))
    tokens = tokenizer.encode(prompt)
    eos_id = tokenizer.eos_token_id
    stop_ids: set[int] = {eos_id}
    if stop_tokens:
        stop_ids.update(stop_tokens)
    same_token_count = 0
    prev_token: int | None = None
    first_emitted = False
    generated: list[int] = []
    decoded_so_far = ""
    pending = ""  # decoded but not yet safe to emit (may start a stop sequence)

    for _ in range(max_tokens):
        next_logits = decoder.next_logits(tokens)
        next_token = next_token_mlx(next_logits, temperature, top_k, top_p, rep, tokens)

        if next_token in stop_ids:
            break

        if next_token == prev_token:
            same_token_count += 1
        else:
            same_token_count = 0
        if _repeat_guard_triggered(same_token_count, max_repeat):
            break
        prev_token = next_token

        tokens.append(next_token)
        generated.append(next_token)
        # Incremental detokenization: decode the whole generated suffix and emit only the
        # NEW text (decoding one token in isolation splits multi-byte graphemes into U+FFFD,
        # diverging from the full decode). Hold emission while the decode ends in a
        # replacement char (an incomplete grapheme) until a later token completes it.
        full = tokenizer.decode(generated)
        if full.endswith("�"):
            continue
        piece = full[len(decoded_so_far) :]
        decoded_so_far = full
        if not first_emitted and piece:
            piece = _strip_leading_space(piece)
            first_emitted = True
        pending += piece
        emit, pending, stop = stream_flush(pending, stop_sequences)
        if emit:
            yield emit
        if stop:
            return

    if pending:  # flush text held back for a stop that never completed
        yield pending


def _generate_torch(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    max_repeat: int | None = DEFAULT_MAX_REPEAT,
    **kwargs: Any,
) -> str:
    """PyTorch generation implementation."""
    import torch

    # Pull out stop strings before tokenizing; HF's native stop support takes
    # the raw strings plus the tokenizer (it tokenizes them internally).
    stop_strings = kwargs.pop("stop_sequences", None)
    stop_tokens = _extract_stop_tokens(tokenizer, kwargs)

    if not hasattr(model, "generate"):
        # The manual fallback only needs ``.encode``; do NOT require a callable
        # tokenizer here (an encode-only tokenizer must still work, as on MLX).
        # HF's generate() honours top_p/top_k/rep natively on the primary path
        # below; the manual loop must apply them itself or they'd be dropped.
        return _generate_manual_torch(
            model,
            tokenizer,
            prompt,
            max_tokens,
            temperature,
            stop_tokens=stop_tokens,
            stop_sequences=list(stop_strings) if stop_strings else None,
            max_repeat=max_repeat,
            top_k=kwargs.get("top_k"),
            top_p=kwargs.get("top_p"),
            repetition_penalty=kwargs.get("repetition_penalty"),
            use_cache=kwargs.get("use_cache", True),
        )

    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.device if hasattr(model, "device") else "cpu"
    input_ids = inputs["input_ids"].to(device)

    gen_kwargs: dict[str, Any] = dict(kwargs)
    # ``do_sample``/``temperature`` are passed explicitly below; pop any
    # caller-supplied copies so they don't collide as duplicate keyword
    # arguments (mirrors the MLX path, which also pops ``do_sample``). The
    # explicit ``temperature`` argument is authoritative and overrides a kwarg.
    gen_kwargs.pop("do_sample", None)
    gen_kwargs.pop("temperature", None)
    # Forward the attention mask the tokenizer already produced. For a single
    # unpadded prompt it is all-ones, so the generated text is identical -- but
    # passing it explicitly silences HF's "attention mask ... not set" warning
    # and stays correct if batched/padded generation is ever added. (overridable)
    attn = inputs.get("attention_mask") if hasattr(inputs, "get") else None
    if attn is not None and hasattr(attn, "to"):
        gen_kwargs.setdefault("attention_mask", attn.to(device))
    # Qwen BASE tokenizers leave ``pad_token_id`` unset, so HF warns and silently
    # falls back to ``eos_token_id``. Set it explicitly (a dedicated pad if the
    # tokenizer has one, else eos) for clean, deterministic logs; it is unused for
    # batch-1 decoding, so this changes no output -- only the spurious warning.
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(pad_id, int):
        gen_kwargs.setdefault("pad_token_id", pad_id)
    if stop_strings:
        # HF GenerationMixin supports stop_strings natively, but only when the
        # tokenizer is also supplied so it can tokenize the stop strings.
        gen_kwargs["stop_strings"] = list(stop_strings)
        gen_kwargs.setdefault("tokenizer", tokenizer)
    if stop_tokens:
        # HF stops on any id in ``eos_token_id``; extend it with the extra stop
        # tokens (explicit stop_tokens or auto-detected turn-end tokens) so they
        # are honoured natively instead of being silently dropped.
        base_eos = getattr(tokenizer, "eos_token_id", None)
        eos_ids = [base_eos] if isinstance(base_eos, int) else []
        for tid in stop_tokens:
            if isinstance(tid, int) and tid not in eos_ids:
                eos_ids.append(tid)
        if eos_ids:
            gen_kwargs["eos_token_id"] = eos_ids

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            **gen_kwargs,
        )

    generated = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)  # type: ignore[no-any-return]


def _generate_manual_torch(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop_tokens: list[int] | None = None,
    stop_sequences: list[str] | None = None,
    max_repeat: int | None = DEFAULT_MAX_REPEAT,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    use_cache: bool = True,
) -> str:
    """Manual PyTorch generation (fallback when model has no .generate()).

    Args:
        model: The base language model.
        tokenizer: The tokenizer (only ``.encode``/``.decode`` are required).
        prompt: Input text prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        stop_tokens: Extra stop token IDs in addition to ``eos_token_id``.
        stop_sequences: Strings that stop generation when they appear in the
            decoded output; the returned text is truncated before the match.
        max_repeat: Consecutive-identical-token cap for the repetition guard.
            ``None`` disables it; logs a warning when it triggers.
        top_k: Keep only the ``top_k`` highest-logit tokens when sampling.
        top_p: Nucleus-sampling threshold in ``(0, 1)``.
        repetition_penalty: HF-style penalty (>1 discourages repeats).
        use_cache: Use an incremental KV cache (O(n) not O(n^2)); falls back to
            full-forward when the model does not support one.
    """
    import torch

    _check_temperature(temperature)
    prompt_len = len(tokenizer.encode(prompt))
    tokens = tokenizer.encode(prompt)
    eos_id = tokenizer.eos_token_id
    stop_ids: set[int] = {eos_id}
    if stop_tokens:
        stop_ids.update(stop_tokens)
    same_token_count = 0
    prev_token: int | None = None
    device = _torch_model_device(model)
    decoder = TorchDecoder(model, use_cache)

    for _ in range(max_tokens):
        with torch.no_grad():
            next_logits = decoder.next_logits(tokens, device)
        next_token = next_token_torch(
            next_logits, temperature, top_k, top_p, repetition_penalty, tokens
        )

        if next_token in stop_ids:
            break

        if next_token == prev_token:
            same_token_count += 1
        else:
            same_token_count = 0
        if _repeat_guard_triggered(same_token_count, max_repeat):
            break

        prev_token = next_token
        tokens.append(next_token)
        if stop_sequences:
            text: str = tokenizer.decode(tokens[prompt_len:])
            cut = _earliest_stop_index(text, stop_sequences)
            if cut is not None:
                return _strip_leading_space(text[:cut])

    return _strip_leading_space(tokenizer.decode(tokens[prompt_len:]))


def _generate_stream_torch(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    max_repeat: int | None = DEFAULT_MAX_REPEAT,
    **kwargs: Any,
) -> Iterator[str]:
    """Streaming PyTorch generation."""
    import torch

    _check_temperature(temperature)
    stop_tokens = _extract_stop_tokens(tokenizer, kwargs)
    stop_sequences = kwargs.pop("stop_sequences", None)
    top_k = kwargs.pop("top_k", None)
    top_p = kwargs.pop("top_p", None)
    rep = kwargs.pop("repetition_penalty", None)
    kwargs.pop("num_return_sequences", None)
    device = _torch_model_device(model)
    decoder = TorchDecoder(model, kwargs.pop("use_cache", True))
    tokens = tokenizer.encode(prompt)
    eos_id = tokenizer.eos_token_id
    stop_ids: set[int] = {eos_id}
    if stop_tokens:
        stop_ids.update(stop_tokens)
    same_token_count = 0
    prev_token: int | None = None
    first_emitted = False
    generated: list[int] = []
    decoded_so_far = ""
    pending = ""  # decoded but not yet safe to emit (may start a stop sequence)

    for _ in range(max_tokens):
        with torch.no_grad():
            next_logits = decoder.next_logits(tokens, device)

        # Penalise the full context (prompt + generated), matching the manual
        # loops and HF's native repetition_penalty on the primary path.
        next_id = next_token_torch(next_logits, temperature, top_k, top_p, rep, tokens)
        if next_id in stop_ids:
            break

        if next_id == prev_token:
            same_token_count += 1
        else:
            same_token_count = 0
        if _repeat_guard_triggered(same_token_count, max_repeat):
            break
        prev_token = next_id

        tokens.append(next_id)
        # Incremental detokenization (see the MLX stream): emit the growing decode's
        # diff, holding while it ends in a replacement char so multi-byte graphemes
        # aren't split.
        generated.append(next_id)
        full = tokenizer.decode(generated)
        if full.endswith("�"):
            continue
        piece = full[len(decoded_so_far) :]
        decoded_so_far = full
        if not first_emitted and piece:
            piece = _strip_leading_space(piece)
            first_emitted = True
        pending += piece
        emit, pending, stop = stream_flush(pending, stop_sequences)
        if emit:
            yield emit
        if stop:
            return

    if pending:  # flush text held back for a stop that never completed
        yield pending

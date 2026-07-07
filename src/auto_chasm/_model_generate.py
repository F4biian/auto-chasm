"""Extracted probe-inspecting generation loop.

Holds the body of ``Model.generate_with_probes`` so ``model.py`` stays under the
file-length cap.  It takes the live ``Model`` and drives its ``forward`` / ``sample``
just like the inline loop it replaced, so behaviour is identical.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from auto_chasm._generation_utils import (
    _check_temperature,
    _extract_stop_tokens,
    _repeat_guard_triggered,
    _resolve_prompt,
)
from auto_chasm.outputs import GenerationStep


def generate_with_probes(
    model: Any,
    prompt: str | None,
    max_tokens: int,
    temperature: float,
    stop_tokens: list[int] | None,
    messages: list[dict[str, str]] | None,
    max_repeat: int | None,
) -> Iterator[GenerationStep]:
    """Generate tokens one at a time with full probe inspection.

    Args:
        model: The :class:`~auto_chasm.model.Model` to drive.
        prompt: Input text prompt (ignored if ``messages`` is set).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy; negative raises).
        stop_tokens: Extra stop token IDs; auto-detected from the tokenizer if ``None``.
        messages: Chat messages (requires the tokenizer's ``chat_template``).
        max_repeat: Repetition-guard cap; ``None`` disables the guard.

    Yields:
        ``GenerationStep`` per generated token.
    """
    resolved = _resolve_prompt(model.tokenizer, prompt, messages)
    _check_temperature(temperature)  # negative temp is a config error (M5)

    tokens = model.tokenizer.encode(resolved)
    prompt_len = len(tokens)
    eos_id = model.tokenizer.eos_token_id
    stop_ids: set[int] = {eos_id}
    extra = _extract_stop_tokens(model.tokenizer, {"stop_tokens": stop_tokens})
    if extra:
        stop_ids.update(extra)
    same_token_count = 0
    prev_token: int | None = None

    # M7: a granularity="response" probe pools over the RESPONSE region only during
    # training (labels sit after the prompt). Setting prompt_len makes the inference
    # pool exclude the prompt too, so generate_with_probes reads the same region the
    # probe was trained on — otherwise it pooled prompt+response and diverged. Reset
    # afterward (even on early consumer break) so a later forward() is unaffected.
    for probe in model._probes.values():
        probe.prompt_len = prompt_len
    try:
        for _ in range(max_tokens):
            outputs = model.forward([tokens])
            assert outputs.lm_logits is not None
            next_logits = outputs.lm_logits[0, -1, :]
            next_token = model.sample(next_logits, temperature)

            if next_token in stop_ids:
                break

            if next_token == prev_token:
                same_token_count += 1
            else:
                same_token_count = 0
            if _repeat_guard_triggered(same_token_count, max_repeat):
                break
            prev_token = next_token

            token_str = model.tokenizer.decode([next_token])
            tokens.append(next_token)

            yield GenerationStep(
                token_id=next_token,
                token_str=token_str,
                probes=outputs.probes,
                next_logits=next_logits,
            )
    finally:
        for probe in model._probes.values():
            probe.prompt_len = None

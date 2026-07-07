"""Backend glue for single-token sampling in the fallback/streaming loops.

The sampling *math* (repetition penalty, temperature, top-k, top-p) lives in the
backend-free :mod:`auto_chasm._generation_utils`.  This module holds the thin
MLX/PyTorch adapters that convert a logit tensor to numpy, run that shared filter,
and draw the next token with the backend's own RNG (so seeding still works).
Kept out of :mod:`auto_chasm.generation` to stay under the file-length cap.
"""

from __future__ import annotations

from typing import Any

from auto_chasm._generation_utils import filter_logits


def next_token_mlx(
    logits: Any,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float | None,
    generated: list[int],
) -> int:
    """Sample one token id from 1-D MLX logits, honouring all sampling controls.

    Args:
        logits: A 1-D ``mlx.core.array`` of next-token logits.
        temperature: Sampling temperature (0 = greedy).
        top_k: Keep only the ``top_k`` highest-logit tokens, or ``None``.
        top_p: Nucleus threshold in ``(0, 1)``, or ``None``.
        repetition_penalty: HF-style penalty (>1 discourages seen tokens), or ``None``.
        generated: Token IDs seen so far (penalised by ``repetition_penalty``).

    Returns:
        The sampled token id.
    """
    import mlx.core as mx
    import numpy as np

    filtered = filter_logits(
        np.asarray(logits.astype(mx.float32)),
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        generated,
    )
    if temperature and temperature > 0:
        return int(mx.random.categorical(mx.array(filtered)).item())
    return int(np.argmax(filtered))


def next_token_torch(
    logits: Any,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float | None,
    generated: list[int],
) -> int:
    """Sample one token id from 1-D torch logits, honouring all sampling controls.

    Args:
        logits: A 1-D ``torch.Tensor`` of next-token logits.
        temperature: Sampling temperature (0 = greedy).
        top_k: Keep only the ``top_k`` highest-logit tokens, or ``None``.
        top_p: Nucleus threshold in ``(0, 1)``, or ``None``.
        repetition_penalty: HF-style penalty (>1 discourages seen tokens), or ``None``.
        generated: Token IDs seen so far (penalised by ``repetition_penalty``).

    Returns:
        The sampled token id.
    """
    import numpy as np
    import torch

    filtered = filter_logits(
        logits.detach().cpu().float().numpy(),
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        generated,
    )
    if temperature and temperature > 0:
        probs = torch.softmax(torch.from_numpy(filtered).float(), dim=-1)
        return int(torch.multinomial(probs, 1).item())
    return int(np.argmax(filtered))

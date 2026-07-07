"""Incremental KV-cache decoders for the fallback / streaming generation loops.

The primary paths (``mlx_lm.generate`` / HF ``model.generate``) cache internally.
The manual and streaming loops, however, re-run the whole growing sequence each
step (O(n^2)); these decoders make them O(n) by feeding only the newest token(s)
through a KV cache and reading the last position's logits.

Both decoders fall back **transparently** to a full re-forward when the model
does not support caching — e.g. a plain callable used in a test, or a model that
ignores ``use_cache`` — so the generated text is byte-for-byte unchanged.

Caching is *not* transparent under steering: a KV cache freezes each past token's
key/value (computed, and steered, when it was the newest token), whereas the
full-forward path re-steers only the current last position. The ``Model`` facade
therefore disables caching while steering hooks are active; see ``Model.generate``.
"""

from __future__ import annotations

from typing import Any


def _last_logits(out: Any) -> Any:
    """Return the last-position logits from a model output (ModelOutput/tuple/tensor)."""
    if hasattr(out, "logits"):
        logits = out.logits
    elif isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    return logits[0, -1, :]


class MlxDecoder:
    """Feeds newest tokens through an mlx_lm KV cache; full-forward if unsupported."""

    def __init__(self, model: Any, use_cache: bool = True) -> None:
        """Create a decoder, building a prompt cache when the model supports one.

        Args:
            model: The MLX base model.
            use_cache: Whether to attempt KV caching (``False`` forces full-forward).
        """
        self.model = model
        self.cache: Any = None
        self._fed = 0
        if use_cache:
            try:
                from mlx_lm.models.cache import make_prompt_cache

                self.cache = make_prompt_cache(model)
            except Exception:
                self.cache = None  # not an mlx_lm model — full-forward each step

    def next_logits(self, tokens: list[int]) -> Any:
        """Return the last-position logits for ``tokens`` (the full sequence so far).

        Args:
            tokens: Prompt + all generated tokens up to now.

        Returns:
            A 1-D array of next-token logits.
        """
        import mlx.core as mx

        if self.cache is not None:
            new = tokens[self._fed :]
            try:
                out = self.model(mx.array([new]), cache=self.cache)
            except TypeError:
                self.cache = None  # model does not accept cache= — fall back
                self._fed = 0
                return self.next_logits(tokens)
            self._fed = len(tokens)
        else:
            out = self.model(mx.array([tokens]))
        return _last_logits(out)


class TorchDecoder:
    """Feeds newest tokens through HF ``past_key_values``; full-forward if unsupported."""

    def __init__(self, model: Any, use_cache: bool = True) -> None:
        """Create a decoder that threads ``past_key_values`` when the model supports it.

        Args:
            model: The torch base model.
            use_cache: Whether to attempt KV caching (``False`` forces full-forward).
        """
        self.model = model
        self.past: Any = None
        self._fed = 0
        self._ok = use_cache

    def next_logits(self, tokens: list[int], device: Any) -> Any:
        """Return the last-position logits for ``tokens`` (the full sequence so far).

        Args:
            tokens: Prompt + all generated tokens up to now.
            device: The device to place the input ids on.

        Returns:
            A 1-D tensor of next-token logits.
        """
        import torch

        if self._ok:
            new = tokens[self._fed :]
            try:
                out = self.model(
                    torch.tensor([new], device=device),
                    past_key_values=self.past,
                    use_cache=True,
                )
            except TypeError:
                self._ok = False  # model does not accept the cache kwargs
                self._fed = 0
                return self.next_logits(tokens, device)
            past = getattr(out, "past_key_values", None)
            if past is None:
                self._ok = False  # model ignored use_cache — cannot cache
                self._fed = 0
                return self.next_logits(tokens, device)
            self.past = past
            self._fed = len(tokens)
            return _last_logits(out)
        out = self.model(torch.tensor([tokens], device=device))
        return _last_logits(out)

"""Stream generation with and without the KV cache, and time the difference.

Decoding without a cache re-reads the whole growing sequence every step (O(n^2)).
The KV cache (on by default) feeds only the newest token each step (O(n)), so it
is much faster and the gap widens with length. The cache is an *optimization only*:
the output is bit-identical to full-forward in float32, and numerically equivalent
in bfloat16 (the default port here is bf16, so a long greedy run can differ by a
token or two — the same property every KV cache has in low precision).

The same code runs on MLX and PyTorch. Streaming always uses the manual decode
loop (where this cache lives); ``model.generate`` uses the backend's own cached
fast path.

    python demo/demo_kv_cache.py
"""

from __future__ import annotations

import time

from auto_chasm import Model

MODEL = "HuggingFaceTB/SmolLM2-135M"  # small standard-Llama arch; loads on both backends

model = Model.from_pretrained(MODEL)
prompt = "In a distant kingdom, a young inventor discovered that"
max_tokens = 128


def timed_stream(*, use_cache: bool) -> tuple[float, str]:
    """Stream ``max_tokens`` greedily and return (seconds, text)."""
    start = time.perf_counter()
    text = "".join(
        model.generate_stream(prompt, max_tokens=max_tokens, temperature=0.0, use_cache=use_cache)
    )
    return time.perf_counter() - start, text


cached_time, cached_text = timed_stream(use_cache=True)
uncached_time, _ = timed_stream(use_cache=False)

print(f"with    KV cache: {cached_time:.2f}s")
print(f"without KV cache: {uncached_time:.2f}s")
print(f"speedup: {uncached_time / cached_time:.1f}x faster for {max_tokens} tokens\n")
print(f"generated text:\n{cached_text}")

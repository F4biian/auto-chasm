"""Oracle tests: streaming generation must not leak the stop/EOS token.

The bug: the streaming loops yielded the decoded next token *before* the
stop-token check, so an EOS/stop token's text (e.g. ``</s>``) leaked into
the stream even though generation then stopped.
"""

from __future__ import annotations

import numpy as np

EOS = 0
VOCAB = 16


class _Tok:
    eos_token_id = EOS

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int], **kwargs: object) -> str:
        return "".join(str(i) for i in ids)


def _logits_favoring(idx: int, shape) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.float32)
    arr[..., idx] = 10.0
    return arr


class _AlwaysEOSMlx:
    def __call__(self, x):  # type: ignore[no-untyped-def]
        import mlx.core as mx

        b, t = x.shape
        return mx.array(_logits_favoring(EOS, (b, t, VOCAB)))


class _AlwaysTokenMlx:
    def __init__(self, tok: int) -> None:
        self.tok = tok

    def __call__(self, x):  # type: ignore[no-untyped-def]
        import mlx.core as mx

        b, t = x.shape
        return mx.array(_logits_favoring(self.tok, (b, t, VOCAB)))


def test_stream_mlx_does_not_emit_eos():
    from auto_chasm.generation import _generate_stream_mlx

    out = list(_generate_stream_mlx(_AlwaysEOSMlx(), _Tok(), "hi", max_tokens=5, temperature=0.0))
    assert out == []  # EOS chosen first => nothing emitted, no leak


def test_stream_mlx_emits_non_stop_tokens():
    from auto_chasm.generation import _generate_stream_mlx

    out = list(
        _generate_stream_mlx(_AlwaysTokenMlx(5), _Tok(), "hi", max_tokens=3, temperature=0.0)
    )
    # token 5 is not a stop token; it repeats, so the repetition guard caps it,
    # but the first emissions must be the decoded token 5 and never EOS.
    assert len(out) >= 1
    assert all("0" not in tok or tok == "5" for tok in out[:1])
    assert out[0] == "5"


class _TorchTok(_Tok):
    def __call__(self, text: str, return_tensors: str = "pt"):  # type: ignore[override]
        import torch

        return {"input_ids": torch.tensor([self.encode(text)])}


class _AlwaysEOSTorch:
    def __call__(self, x):  # type: ignore[no-untyped-def]
        import torch

        b, t = x.shape
        return torch.tensor(_logits_favoring(EOS, (b, t, VOCAB)))


def test_stream_torch_does_not_emit_eos():
    from auto_chasm.generation import _generate_stream_torch

    out = list(
        _generate_stream_torch(_AlwaysEOSTorch(), _TorchTok(), "hi", max_tokens=5, temperature=0.0)
    )
    assert out == []

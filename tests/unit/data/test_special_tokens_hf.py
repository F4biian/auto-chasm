"""Special-token regressions: real-HF-tokenizer path, bias dtype, vocab sync."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from auto_chasm.model import Model
from auto_chasm.special_tokens import _append_head_bias, _vocab_size


class _RustTok:
    """A low-level tokenizer (like tokenizers.Tokenizer): no __len__, no vocab_size."""


class _FakeHFTok:
    """Mimics a HuggingFace PreTrainedTokenizerFast: has __len__ AND a ._tokenizer."""

    def __init__(self, vocab: int) -> None:
        self._vocab = vocab
        self._extra: list[str] = []
        self._tokenizer = _RustTok()

    def __len__(self) -> int:
        return self._vocab + len(self._extra)

    def add_tokens(self, tokens, special_tokens=False):  # noqa: ANN001, ANN201
        new = [t for t in tokens if t not in self._extra]
        self._extra += new
        return len(new)


class _TinyMlp(nn.Module):
    def __init__(self, vocab: int = 32, h: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, h)
        self.output_proj = nn.Linear(h, vocab)  # untied head

    def __call__(self, x: mx.array) -> mx.array:
        return self.output_proj(self.embedding(x))


def _cfg(vocab: int) -> object:
    class _C:
        hidden_size = 8
        num_hidden_layers = 1
        vocab_size = vocab

    return _C()


def test_st_f1_hf_style_tokenizer_vocab_size_not_zero() -> None:
    """_vocab_size reads the outer HF tokenizer's len, not the ._tokenizer (which is 0)."""
    tok = _FakeHFTok(50257)
    assert _vocab_size(tok) == 50257  # was 0 -> resize_token_embeddings(0) wiped the model
    tok.add_tokens(["<a>", "<b>"], special_tokens=True)
    assert _vocab_size(tok) == 50259


def test_st_f1_add_special_tokens_grows_embedding_not_wipes() -> None:
    """add_special_tokens on a HF-style tokenizer grows the tables (not to 0 rows)."""
    m = Model(_TinyMlp(vocab=32), _FakeHFTok(32), "mlx")
    m.model.config = _cfg(32)
    assert m.add_special_tokens(["<x>", "<y>"]) == 2
    assert m.model.embedding.weight.shape[0] == 34  # grew by 2 (NOT wiped to 0)
    assert m.model.output_proj.weight.shape[0] == 34
    assert m.model.config.vocab_size == 34
    m.forward([[32, 33, 1]])  # brand-new ids do not index out of range


def test_st_f3_head_bias_dtype_preserved() -> None:
    """_append_head_bias keeps a bf16 head bias in bf16 (no silent float32 upcast)."""
    head = nn.Linear(8, 4)
    head.bias = head.bias.astype(mx.bfloat16)
    _append_head_bias(head, 2)
    assert head.bias.shape[0] == 6
    assert head.bias.dtype == mx.bfloat16


def test_st_f4_config_vocab_syncs_to_real_table_not_tokenizer() -> None:
    """config.vocab_size tracks the ACTUAL grown table, not a tokenizer that runs ahead."""
    # Tokenizer already reports 34 while the embedding has only 32 rows (pre-existing gap).
    m = Model(_TinyMlp(vocab=32), _FakeHFTok(34), "mlx")
    m.model.config = _cfg(34)
    m.add_special_tokens(["<x>", "<y>"])  # tokenizer -> 36, embedding 32 -> 34
    assert m.model.embedding.weight.shape[0] == 34
    assert m.model.config.vocab_size == 34  # the real 34-row table, not the tokenizer's 36

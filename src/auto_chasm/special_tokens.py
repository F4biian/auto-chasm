"""Add new (special) tokens to a loaded model + tokenizer.

Adding a token means two things must stay in sync: the tokenizer must know
the new string, and the model's input-embedding table (and any untied output
head) must grow a row for the new id.  This module keeps them in sync for
both backends.

Support (full on every path):
- **PyTorch**: via ``model.resize_token_embeddings`` (transformers).
- **MLX, plain ``nn.Embedding`` / ``nn.Linear``**: rows appended, initialised to
  the mean of the existing rows.
- **MLX, ``QuantizedEmbedding`` / ``QuantizedLinear``**: the new rows are
  quantized on their own and concatenated onto the packed table. Because each
  row is its own quantization group (the group size divides the feature dim),
  the **existing rows stay byte-identical** — no dequantize/requantize of the
  base, no precision loss for the original tokens.
"""

from __future__ import annotations

from typing import Any

from auto_chasm.logger import get_logger

logger = get_logger(__name__)


def add_special_tokens(model: Any, tokenizer: Any, tokens: list[str], backend_name: str) -> int:
    """Add ``tokens`` to the tokenizer and grow the model embeddings to match.

    Args:
        model: The underlying framework model.
        tokenizer: The tokenizer (HuggingFace or mlx-lm ``TokenizerWrapper``).
        tokens: New token strings to add as special tokens.
        backend_name: ``"mlx"`` or ``"torch"``.

    Returns:
        The number of tokens actually added (0 if all already existed).

    Raises:
        NotImplementedError: If no embedding can be located on an MLX model.
        TypeError: If the tokenizer cannot add tokens.
    """
    if not tokens:
        return 0
    if not hasattr(tokenizer, "add_tokens"):
        raise TypeError("Tokenizer does not support add_tokens(); cannot add special tokens.")

    n_added = int(tokenizer.add_tokens(tokens, special_tokens=True))
    if n_added == 0:
        logger.info("add_special_tokens: all %d tokens already present.", len(tokens))
        return 0

    new_vocab = _vocab_size(tokenizer)
    if backend_name == "torch":
        model.resize_token_embeddings(new_vocab)
    else:
        _resize_mlx(model, new_vocab, n_added)

    logger.info("Added %d special token(s); vocab is now %d.", n_added, new_vocab)
    return n_added


def _vocab_size(tokenizer: Any) -> int:
    """Return the tokenizer's vocab size, including added tokens.

    Handles plain HuggingFace tokenizers (``len()``) and the mlx-lm
    ``TokenizerWrapper`` (no ``__len__``; unwrap to ``_tokenizer``).

    Args:
        tokenizer: The tokenizer or wrapper.

    Returns:
        Total vocabulary size after any added tokens.
    """
    # A plain HuggingFace tokenizer has __len__ that already includes added tokens —
    # use it directly. Only the mlx-lm TokenizerWrapper lacks __len__; unwrap THAT to
    # its ._tokenizer. (A plain HF tokenizer ALSO has a ._tokenizer — the low-level
    # Rust tokenizer with no __len__/vocab_size — so unwrapping unconditionally would
    # read 0 and silently resize the embedding to zero rows.)
    try:
        return len(tokenizer)
    except TypeError:
        inner = getattr(tokenizer, "_tokenizer", tokenizer)
    try:
        return len(inner)
    except TypeError:
        base = int(getattr(inner, "vocab_size", 0))
        added = inner.get_added_vocab() if hasattr(inner, "get_added_vocab") else {}
        return base + len(added)


def _resize_mlx(model: Any, new_vocab: int, n_added: int) -> None:
    """Grow an MLX model's input embedding (and untied output head) by ``n_added`` rows.

    Handles both full-precision (``nn.Embedding``/``nn.Linear``) and quantized
    (``nn.QuantizedEmbedding``/``nn.QuantizedLinear``) tables.

    Args:
        model: The MLX model.
        new_vocab: The target vocabulary size.
        n_added: Number of new rows to append.

    Raises:
        NotImplementedError: If no embedding can be located on the model.
    """
    import mlx.nn as nn

    embedding = None
    for _name, module in model.named_modules():
        if isinstance(module, nn.QuantizedEmbedding):
            _append_quantized_rows(module, n_added)
            module.num_embeddings = new_vocab
            embedding = module
            break
        if isinstance(module, nn.Embedding):
            _append_full_rows(module, n_added)
            if hasattr(module, "num_embeddings"):
                module.num_embeddings = new_vocab
            embedding = module
            break
    if embedding is None:
        raise NotImplementedError("Could not locate an embedding to resize on this model.")

    # The untied head's old out_features == the embedding's OLD row count. Derive
    # it from the (now-grown) embedding, NOT the tokenizer count: some tokenizers
    # report a different size than the embedding table (e.g. gemma's reports one
    # more), so ``new_vocab - n_added`` missed the head and it never grew — the new
    # id then produced no logit.
    old_vocab = embedding.weight.shape[0] - n_added

    # Resize an untied output head (quantized or full precision) whose
    # out_features == the old vocab.  Tied heads grow with the embedding.
    for _name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear) and module.weight.shape[0] == old_vocab:
            _append_quantized_rows(module, n_added)
            _append_head_bias(module, n_added)
            break
        if isinstance(module, nn.Linear) and module.weight.shape[0] == old_vocab:
            _append_full_rows(module, n_added)
            _append_head_bias(module, n_added)
            break

    # Sync the DECLARED vocab to the ACTUAL grown table row count, not the tokenizer
    # count (which can exceed the table when the tokenizer already reported more than
    # the embedding had — then a source="logits" probe would oversize and mismatch the
    # head). Otherwise a logits probe sizing from a stale config dies with a matmul error.
    real_vocab = int(embedding.weight.shape[0])
    for attr in ("config", "args"):
        cfg = getattr(model, attr, None)
        if cfg is not None and hasattr(cfg, "vocab_size"):
            cfg.vocab_size = real_vocab


def _append_full_rows(module: Any, n_added: int) -> None:
    """Append ``n_added`` mean-initialised rows to a full-precision weight, in place."""
    import mlx.core as mx

    w = module.weight
    new_rows = mx.broadcast_to(mx.mean(w, axis=0, keepdims=True), (n_added, w.shape[1]))
    module.weight = mx.concatenate([w, new_rows], axis=0)


def _append_quantized_rows(module: Any, n_added: int) -> None:
    """Append ``n_added`` mean-init rows to a quantized weight, losslessly for the rest.

    Each row is an independent quantization group, so the new rows are quantized
    on their own and concatenated; the existing packed rows are untouched.

    Args:
        module: A ``QuantizedEmbedding`` or ``QuantizedLinear``.
        n_added: Number of rows to append.
    """
    import mlx.core as mx

    gs, bits = module.group_size, module.bits
    full = mx.dequantize(module.weight, module.scales, module.biases, gs, bits)
    new_full = mx.broadcast_to(mx.mean(full, axis=0, keepdims=True), (n_added, full.shape[1]))
    n_w, n_s, n_b = mx.quantize(new_full, gs, bits)
    module.weight = mx.concatenate([module.weight, n_w], axis=0)
    module.scales = mx.concatenate([module.scales, n_s], axis=0)
    module.biases = mx.concatenate([module.biases, n_b], axis=0)


def _append_head_bias(module: Any, n_added: int) -> None:
    """Append ``n_added`` zero biases to an output head, if it has a bias."""
    import mlx.core as mx

    if "bias" in module and module.bias is not None:
        # Match the existing bias dtype so a bf16 head bias is not silently upcast
        # (concatenating float32 zeros would promote the whole bias to float32).
        zeros = mx.zeros((n_added,), dtype=module.bias.dtype)
        module.bias = mx.concatenate([module.bias, zeros])

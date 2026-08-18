"""Config and module lookup must see past wrapper architectures.

Checking only ``model.config``/``model.args`` missed Qwen3.5's MLX build, a
multimodal-style shell whose ``args`` holds only a ``text_config`` and whose real
dimensions live under ``model.language_model``. Every probe attach on that model
raised before a single training step ran.

The hidden-dim lookup was fixed first; the vocab-size lookup, the module search
and ``stats()``'s config reads kept the old top-level-only assumption, so
``Model.stats()`` still raised on that model, ``_find_embedding`` returned
``(None, None)`` (no embedding probe could attach), and ``num_attention_heads``
silently reported ``None``. They now share one wrapper-aware walk.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from auto_chasm._model_stats import _cfg_value
from auto_chasm.model import _explicit_in_features
from auto_chasm.probe import _find_embedding, _get_hidden_dim, _get_vocab_size


def test_plain_config() -> None:
    assert _get_hidden_dim(SimpleNamespace(config=SimpleNamespace(hidden_size=896))) == 896


def test_plain_args() -> None:
    assert _get_hidden_dim(SimpleNamespace(args=SimpleNamespace(d_model=512))) == 512


def test_wrapper_with_language_model() -> None:
    """The Qwen3.5 shape: outer args has only text_config, inner holds the size."""
    model = SimpleNamespace(
        args=SimpleNamespace(model_type="qwen3_5", text_config=SimpleNamespace()),
        language_model=SimpleNamespace(args=SimpleNamespace(hidden_size=1024)),
    )
    assert _get_hidden_dim(model) == 1024


def test_text_config_one_level_down() -> None:
    model = SimpleNamespace(args=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2048)))
    assert _get_hidden_dim(model) == 2048


def test_unknown_still_raises() -> None:
    with pytest.raises(ValueError, match="Cannot determine hidden dimension"):
        _get_hidden_dim(SimpleNamespace())


def test_explicit_in_features_is_honoured() -> None:
    """The escape hatch the error message advertises must actually work."""
    assert _explicit_in_features(SimpleNamespace(module_config={"in_features": 777})) == 777
    assert _explicit_in_features(SimpleNamespace(module_config={"out_features": 1})) is None
    assert _explicit_in_features(SimpleNamespace(module_config=None)) is None


# --- vocab size: the same walk, previously top-level only -------------------


def _qwen35_shaped() -> SimpleNamespace:
    """The wrapper shape that broke ``Model.stats()``, with real Qwen3.5 values."""
    return SimpleNamespace(
        args=SimpleNamespace(model_type="qwen3_5", text_config=SimpleNamespace()),
        language_model=SimpleNamespace(
            args=SimpleNamespace(hidden_size=1024, vocab_size=248320, num_attention_heads=8),
            model=SimpleNamespace(embed_tokens="EMBED"),
        ),
    )


def test_vocab_size_plain_config() -> None:
    assert _get_vocab_size(SimpleNamespace(config=SimpleNamespace(vocab_size=151936))) == 151936


def test_vocab_size_through_wrapper() -> None:
    assert _get_vocab_size(_qwen35_shaped()) == 248320


def test_vocab_size_unknown_still_raises() -> None:
    with pytest.raises(ValueError, match="Cannot determine vocabulary size"):
        _get_vocab_size(SimpleNamespace())


def test_cfg_value_through_wrapper() -> None:
    """stats() read num_attention_heads as None on wrapped models."""
    assert _cfg_value(_qwen35_shaped(), ("num_attention_heads", "n_heads")) == 8
    assert _cfg_value(_qwen35_shaped(), ("nonexistent_field",)) is None


# --- module search ----------------------------------------------------------


def test_find_embedding_through_wrapper() -> None:
    module, path = _find_embedding(_qwen35_shaped())
    assert (module, path) == ("EMBED", "language_model.model.embed_tokens")


def test_find_embedding_prefers_the_shortest_path() -> None:
    """An unwrapped model must keep resolving to its own short path."""
    model = SimpleNamespace(embed_tokens="TOP", model=SimpleNamespace(embed_tokens="NESTED"))
    assert _find_embedding(model) == ("TOP", "embed_tokens")


def test_find_embedding_missing_is_none() -> None:
    assert _find_embedding(SimpleNamespace()) == (None, None)

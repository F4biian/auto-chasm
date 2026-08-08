"""Hidden-dim inference must see past wrapper architectures.

Checking only ``model.config``/``model.args`` missed Qwen3.5's MLX build, a
multimodal-style shell whose ``args`` holds only a ``text_config`` and whose real
hidden size lives at ``model.language_model.args.hidden_size``. Every probe
attach on that model raised before a single training step ran.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from auto_chasm.model import _explicit_in_features
from auto_chasm.probe import _get_hidden_dim


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

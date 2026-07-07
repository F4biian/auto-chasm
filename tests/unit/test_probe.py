"""Tests for Probe injection — source, granularity, and pooling.

Covers ProbeConfig.source (hidden, embedding, logits, attention, mlp,
residual) and granularity/pooling (token, response, sentence, custom).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import ProbeConfig
from auto_chasm.model import Model
from auto_chasm.probe import (
    _find_embedding,
    _find_output_head,
    _get_hidden_dim,
    _get_vocab_size,
)


class TinyMlp(nn.Module):
    """A tiny MLP for testing probe injection."""

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 32, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self._hidden_dim = hidden_dim

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class DummyTokenizer:
    """Minimal tokenizer for testing."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids if i > 0)


class Config:
    """Dummy configuration."""

    hidden_size = 16
    num_hidden_layers = 4
    vocab_size = 32


@pytest.fixture
def tiny_model() -> TinyMlp:
    """Create a TinyMlp for testing."""
    mx.random.seed(42)
    model = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)
    model.config = Config()
    return model


@pytest.fixture
def model_wrapper(tiny_model: TinyMlp) -> Model:
    """Create a Model wrapper."""
    return Model(tiny_model, DummyTokenizer(), backend_name="mlx")


# ---------------------------------------------------------------------------
# Source tests
# ---------------------------------------------------------------------------


class TestSourceHidden:
    """Tests for source='hidden' (default, backward-compatible)."""

    def test_default_source_is_hidden(self) -> None:
        cfg = ProbeConfig(name="p", layers=[1])
        assert cfg.source == "hidden"

    def test_hidden_inject_and_forward(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert "p" in outputs.probes
        # [B, T, out_dim] — token granularity (default)
        assert outputs.probes["p"].logits.shape == (1, 3, 1)


class TestSourceEmbedding:
    """Tests for source='embedding'."""

    def test_embedding_source_config(self) -> None:
        cfg = ProbeConfig(name="p", layers=[1], source="embedding")
        assert cfg.source == "embedding"

    def test_find_embedding(self, tiny_model: TinyMlp) -> None:
        embed, path = _find_embedding(tiny_model)
        assert embed is not None
        assert isinstance(embed, nn.Embedding)
        assert path == "embedding"

    def test_embedding_inject_and_forward(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0], source="embedding"))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert "p" in outputs.probes
        # Embedding output is [B, T, hidden_dim], probe output is [B, T, 1]
        assert outputs.probes["p"].logits.shape == (1, 3, 1)

    def test_embedding_output_dim_matches_hidden(self, model_wrapper: Model) -> None:
        """Embedding source uses hidden_dim, same as hidden source."""
        model_wrapper.attach_probe(ProbeConfig(name="emb", layers=[0], source="embedding"))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert outputs.probes["emb"].logits.ndim == 3


class TestSourceLogits:
    """Tests for source='logits'."""

    def test_logits_source_config(self) -> None:
        cfg = ProbeConfig(name="p", layers=[1], source="logits")
        assert cfg.source == "logits"

    def test_find_output_head(self, tiny_model: TinyMlp) -> None:
        head, path = _find_output_head(tiny_model)
        assert head is not None
        assert isinstance(head, nn.Linear)
        assert path == "output_proj"

    def test_logits_inject_and_forward(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0], source="logits"))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert "p" in outputs.probes
        # Logits output is [B, T, vocab_size], probe output is [B, T, 1]
        assert outputs.probes["p"].logits.shape == (1, 3, 1)

    def test_logits_probe_uses_vocab_dim(self) -> None:
        """When source='logits', in_features should be vocab_size."""
        from auto_chasm.probe import _get_vocab_size

        model = TinyMlp(hidden_dim=16, vocab_size=64, num_layers=4)
        model.config = type(
            "C", (), {"hidden_size": 16, "num_hidden_layers": 4, "vocab_size": 64}
        )()
        vocab_dim = _get_vocab_size(model)
        assert vocab_dim == 64


class TestSubBlockSourcesConstruct:
    """attention/mlp/residual sources now build a valid config (no longer raise).

    Their capture behavior is oracle-checked in test_probe_subblock_oracle.py.
    """

    def test_attention_constructs(self) -> None:
        assert ProbeConfig(name="p", layers=[1], source="attention").source == "attention"

    def test_mlp_constructs(self) -> None:
        assert ProbeConfig(name="p", layers=[1], source="mlp").source == "mlp"

    def test_residual_constructs(self) -> None:
        assert ProbeConfig(name="p", layers=[1], source="residual").source == "residual"


# ---------------------------------------------------------------------------
# Granularity tests
# ---------------------------------------------------------------------------


class TestGranularityToken:
    """Tests for granularity='token' (default)."""

    def test_default_granularity_is_token(self) -> None:
        cfg = ProbeConfig(name="p", layers=[1])
        assert cfg.granularity == "token"

    def test_token_produces_3d_output(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1], granularity="token"))
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        outputs = model_wrapper.forward(input_ids)
        logits = outputs.probes["p"].logits
        assert logits.ndim == 3
        assert logits.shape == (1, 5, 1)


class TestGranularityResponse:
    """Tests for granularity='response'."""

    def test_response_produces_2d_output(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1], granularity="response"))
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        outputs = model_wrapper.forward(input_ids)
        logits = outputs.probes["p"].logits
        assert logits.ndim == 2
        assert logits.shape == (1, 1)

    def test_response_with_embedding_source(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(
            ProbeConfig(name="p", layers=[0], source="embedding", granularity="response")
        )
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        logits = outputs.probes["p"].logits
        assert logits.ndim == 2
        assert logits.shape == (1, 1)

    def test_response_with_logits_source(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(
            ProbeConfig(name="p", layers=[0], source="logits", granularity="response")
        )
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        logits = outputs.probes["p"].logits
        assert logits.ndim == 2
        assert logits.shape == (1, 1)


class TestGranularityCustom:
    """Tests for granularity='custom' with a pooling callable."""

    def test_custom_pooling_with_callable(self, model_wrapper: Model) -> None:
        def last_token_pool(logits: mx.array) -> mx.array:
            return logits[:, -1, :]

        model_wrapper.attach_probe(
            ProbeConfig(
                name="p",
                layers=[1],
                granularity="custom",
                pooling=last_token_pool,
            )
        )
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        outputs = model_wrapper.forward(input_ids)
        logits = outputs.probes["p"].logits
        # Last-token pooling → [B, out_dim]
        assert logits.ndim == 2
        assert logits.shape == (1, 1)

    def test_custom_pooling_is_called(self, model_wrapper: Model) -> None:
        called: list[bool] = [False]

        def tracking_pool(logits: mx.array) -> mx.array:
            called[0] = True
            return logits.mean(axis=1)

        model_wrapper.attach_probe(
            ProbeConfig(
                name="p",
                layers=[1],
                granularity="custom",
                pooling=tracking_pool,
            )
        )
        input_ids = mx.array([[1, 2, 3]])
        model_wrapper.forward(input_ids)
        assert called[0] is True

    def test_custom_without_pooling_falls_back(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(
            ProbeConfig(name="p", layers=[1], granularity="custom", pooling=None)
        )
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        # Falls back to raw [B, T, out_dim]
        logits = outputs.probes["p"].logits
        assert logits.ndim == 3
        assert logits.shape == (1, 3, 1)


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Ensure existing probes with default config still work."""

    def test_default_source_and_granularity(self, model_wrapper: Model) -> None:
        """Probe with only name+layers should still produce [B, T, 1]."""
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert outputs.probes["p"].logits.shape == (1, 3, 1)

    def test_multi_layer_concatenation(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0, 1], aggregation="concat"))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert outputs.probes["p"].logits is not None

    def test_multi_layer_mean_aggregation(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0, 1], aggregation="mean"))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert outputs.probes["p"].logits is not None

    def test_negative_index_still_works(self, model_wrapper: Model) -> None:
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[-1]))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert outputs.probes["p"].logits.shape == (1, 3, 1)

    def test_custom_module_type(self, model_wrapper: Model) -> None:
        def custom_module(in_dim: int, cfg: dict) -> nn.Linear:
            return nn.Linear(in_dim, cfg.get("out_features", 1))

        model_wrapper.attach_probe(
            ProbeConfig(
                name="p",
                layers=[1],
                module_type=custom_module,
                module_config={"out_features": 1},
            )
        )
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert outputs.probes["p"].logits.shape == (1, 3, 1)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for _get_vocab_size and related helpers."""

    def test_get_vocab_size_from_config(self) -> None:
        model = TinyMlp()
        model.config = type("C", (), {"vocab_size": 64})()
        assert _get_vocab_size(model) == 64

    def test_get_vocab_size_raises_when_missing(self) -> None:
        model = TinyMlp()
        model.config = type("C", (), {})()
        with pytest.raises(ValueError, match="vocabulary"):
            _get_vocab_size(model)

    def test_get_hidden_dim(self) -> None:
        model = TinyMlp()
        model.config = type("C", (), {"hidden_size": 128})()
        assert _get_hidden_dim(model) == 128

    def test_find_embedding_none_when_missing(self) -> None:
        class NoEmbedding(nn.Module):
            """Model without an embedding layer."""

            def __init__(self) -> None:
                super().__init__()
                self.layers = [nn.Linear(16, 16)]

        embed, path = _find_embedding(NoEmbedding())
        assert embed is None
        assert path is None

    def test_find_output_head_none_when_missing(self) -> None:
        class NoHead(nn.Module):
            """Model without an output head."""

            def __init__(self) -> None:
                super().__init__()
                self.layers = [nn.Linear(16, 16)]

        head, path = _find_output_head(NoHead())
        assert head is None
        assert path is None

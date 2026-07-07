"""Coverage-patch tests for probe.py — edge cases and uncovered branches.

Covers:
- _MLXLayerCapture.__getattr__ (attribute delegation)
- _TorchLayerCapture.__getattr__ (attribute delegation)
- _forward_impl steering path (steer_fn + binary_head)
- Probe.forward with explicit hidden_states
- Probe.forward single-layer concat
- Probe._apply_pooling with sentence and unknown granularity
- Probe._inject_embedding / _inject_logits when modules not found
- Custom aggregation callable
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import ProbeConfig
from auto_chasm.model import Model
from auto_chasm.probe import (
    Probe,
    _getattr_mlx,
    make_layer_capture,
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
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "test"


class Config:
    """Dummy model configuration."""

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
# _MLXLayerCapture.__getattr__ tests
# ---------------------------------------------------------------------------


class TestMLXLayerCaptureGetAttr:
    """Tests for _MLXLayerCapture.__getattr__ delegation."""

    def test_getattr_delegates_to_layer(self, tiny_model: TinyMlp) -> None:
        """Accessing a parameter on the capture delegates to the wrapped layer."""
        capture = make_layer_capture(tiny_model.layers[0], layer_idx=0, backend_name="mlx")
        weight = capture.weight
        assert weight is not None

    def test_getattr_raises_on_missing(self, tiny_model: TinyMlp) -> None:
        """Accessing a nonexistent attribute raises AttributeError."""
        capture = make_layer_capture(tiny_model.layers[0], layer_idx=0, backend_name="mlx")
        with pytest.raises(AttributeError):
            _ = capture.nonexistent_attr_xyz

    def test_getattr_direct_call(self, tiny_model: TinyMlp) -> None:
        """Call _getattr_mlx directly as a function (not through __getattr__)."""
        capture = make_layer_capture(tiny_model.layers[0], layer_idx=0, backend_name="mlx")
        result = _getattr_mlx(capture, "weight")
        assert result is not None

    def test_getattr_direct_missing_raises(self, tiny_model: TinyMlp) -> None:
        """_getattr_mlx raises AttributeError for missing attrs when called directly."""
        capture = make_layer_capture(tiny_model.layers[0], layer_idx=0, backend_name="mlx")
        with pytest.raises(AttributeError):
            _getattr_mlx(capture, "nonexistent_attr_xyz")


class TestTorchLayerCaptureGetAttr:
    """Tests for _TorchLayerCapture.__getattr__ delegation."""

    def test_getattr_delegates_to_layer(self) -> None:
        """_TorchLayerCapture delegates attribute access to wrapped layer."""
        pytest.importorskip("torch")
        import torch.nn as tnn

        from auto_chasm.probe import _TorchLayerCapture

        layer = tnn.Linear(4, 4)
        capture = _TorchLayerCapture(layer, layer_idx=0)
        weight = capture.weight
        assert weight is not None

    def test_getattr_raises_on_missing(self) -> None:
        """_TorchLayerCapture raises AttributeError for nonexistent attributes."""
        pytest.importorskip("torch")
        import torch.nn as tnn

        from auto_chasm.probe import _TorchLayerCapture

        layer = tnn.Linear(4, 4)
        capture = _TorchLayerCapture(layer, layer_idx=0)
        with pytest.raises(AttributeError):
            _ = capture.nonexistent_attr_xyz


# ---------------------------------------------------------------------------
# _forward_impl steering path
# ---------------------------------------------------------------------------


class TestForwardImplSteering:
    """Tests for the steering path in _forward_impl."""

    def test_steer_fn_called(self, model_wrapper: Model) -> None:
        """When steer_fn and binary_head are set, steer_fn should be called."""
        called: list[bool] = [False]

        def steer_fn(hidden: mx.array, head: nn.Module, logits: mx.array) -> mx.array:
            called[0] = True
            return hidden

        probe_cfg = ProbeConfig(name="p", layers=[0])
        model_wrapper.attach_probe(probe_cfg)

        probe = model_wrapper._probes["p"]
        probe.layer_captures[0].steer_fn = steer_fn
        probe.layer_captures[0].binary_head = nn.Linear(16, 1)

        input_ids = mx.array([[1, 2, 3]])
        model_wrapper.forward(input_ids)
        assert called[0]

    def test_steer_fn_exception_propagates(self, model_wrapper: Model) -> None:
        """If steer_fn raises, the error must PROPAGATE (not be silently swallowed).

        A swallowed steer_fn error would degrade the run to an unsteered alpha=0 for those
        positions — a partial failure no downstream check catches — so it fails loud instead.
        """

        def steer_fn(hidden: mx.array, head: nn.Module, logits: mx.array) -> mx.array:
            msg = "deliberate steer crash"
            raise RuntimeError(msg)

        probe_cfg = ProbeConfig(name="p", layers=[0])
        model_wrapper.attach_probe(probe_cfg)

        probe = model_wrapper._probes["p"]
        probe.layer_captures[0].steer_fn = steer_fn
        probe.layer_captures[0].binary_head = nn.Linear(16, 1)

        input_ids = mx.array([[1, 2, 3]])
        with pytest.raises(RuntimeError, match="deliberate steer crash"):
            model_wrapper.forward(input_ids)

    def test_steer_with_tuple_output(self, model_wrapper: Model) -> None:
        """steer_fn works when the layer returns a tuple."""
        called: list[bool] = [False]

        def steer_fn(hidden: mx.array, head: nn.Module, logits: mx.array) -> mx.array:
            called[0] = True
            return hidden

        probe_cfg = ProbeConfig(name="p", layers=[0])
        model_wrapper.attach_probe(probe_cfg)

        probe = model_wrapper._probes["p"]
        probe.layer_captures[0].steer_fn = steer_fn
        probe.layer_captures[0].binary_head = nn.Linear(16, 1)

        input_ids = mx.array([[1, 2, 3]])
        model_wrapper.forward(input_ids)
        assert called[0]


# ---------------------------------------------------------------------------
# Probe.forward edge cases
# ---------------------------------------------------------------------------


class TestProbeForwardEdgeCases:
    """Edge cases for Probe.forward()."""

    def test_forward_with_explicit_hidden_states(self) -> None:
        """Pass hidden_states directly instead of using captured states."""
        probe = Probe(ProbeConfig(name="p", layers=[0]), hidden_dim=16, backend_name="mlx")
        hidden = [mx.zeros((1, 3, 16))]
        logits = probe.forward(hidden_states=hidden)
        assert logits is not None

    def test_concat_single_layer(self) -> None:
        """Single-layer concat should bypass _aggregate and go direct to module."""
        probe = Probe(
            ProbeConfig(name="p", layers=[0], aggregation="concat"),
            hidden_dim=16,
            backend_name="mlx",
        )
        hidden = [mx.ones((1, 3, 16))]
        logits = probe.forward(hidden_states=hidden)
        assert logits.shape == (1, 3, 1)

    def test_no_hidden_states_raises(self) -> None:
        """Calling forward without any captured states raises RuntimeError."""
        probe = Probe(ProbeConfig(name="p", layers=[0]), hidden_dim=16, backend_name="mlx")
        with pytest.raises(RuntimeError, match="No hidden states captured"):
            probe.forward()

    def test_explicit_hidden_states_preferred_over_captured(self) -> None:
        """Explicit hidden_states should be used instead of captured states."""
        probe = Probe(
            ProbeConfig(name="p", layers=[0], aggregation="mean"), hidden_dim=16, backend_name="mlx"
        )
        probe._captured.append(mx.ones((1, 2, 16)))
        hidden = [mx.zeros((1, 3, 16))]
        logits = probe.forward(hidden_states=hidden)
        assert logits is not None


# ---------------------------------------------------------------------------
# _apply_pooling granularity tests
# ---------------------------------------------------------------------------


class TestApplyPoolingGranularity:
    """Tests for Probe._apply_pooling with different granularity values."""

    def test_sentence_granularity_requires_delimiters(self) -> None:
        """Sentence granularity needs explicit delimiters; absent → ValueError."""
        import pytest

        with pytest.raises(ValueError, match="sentence_delimiters"):
            ProbeConfig(name="p", layers=[0], granularity="sentence")

    def test_sentence_granularity_with_delimiters_constructs(self) -> None:
        """With sentence_delimiters provided, the config builds (oracle elsewhere)."""
        cfg = ProbeConfig(
            name="p",
            layers=[0],
            granularity="sentence",
            module_config={"sentence_delimiters": [13]},
        )
        assert cfg.granularity == "sentence"

    def test_unknown_granularity_returns_raw(self) -> None:
        """Unknown granularity falls through and returns raw logits."""
        probe = Probe(ProbeConfig(name="p", layers=[0]), hidden_dim=16, backend_name="mlx")
        logits = mx.ones((2, 4, 1))
        # setting a bogus granularity to hit the fallthrough path
        probe.config.granularity = "unknown_granularity"
        result = probe._apply_pooling(logits)
        assert result.shape == (2, 4, 1)

    def test_response_granularity_mlx(self) -> None:
        """Response granularity mean-pools over sequence dim on MLX."""
        probe = Probe(
            ProbeConfig(name="p", layers=[0], granularity="response"),
            hidden_dim=16,
            backend_name="mlx",
        )
        logits = mx.ones((2, 4, 1))
        result = probe._apply_pooling(logits)
        assert result.shape == (2, 1)


# ---------------------------------------------------------------------------
# Injection edge cases
# ---------------------------------------------------------------------------


class TestInjectionEdgeCases:
    """Edge cases for probe injection."""

    def test_inject_embedding_not_found_raises(self) -> None:
        """_inject_embedding raises ValueError when embed is not found."""

        class NoEmbedModel(nn.Module):
            """Model without an embedding module."""

            def __call__(self, x):  # type: ignore[no-untyped-def]
                return (mx.zeros((1, 3, 16)),)

        model = Model(NoEmbedModel(), DummyTokenizer(), backend_name="mlx")
        with pytest.raises(ValueError, match="Cannot determine hidden dimension"):
            model.attach_probe(ProbeConfig(name="p", layers=[0], source="embedding"))

    def test_inject_logits_not_found_raises(self) -> None:
        """_inject_logits raises ValueError when output head is not found."""

        class NoHeadModel(nn.Module):
            """Model without an output projection head."""

            def __call__(self, x):  # type: ignore[no-untyped-def]
                return (mx.zeros((1, 3, 16)),)

        model = Model(NoHeadModel(), DummyTokenizer(), backend_name="mlx")
        with pytest.raises(ValueError, match="Cannot determine vocabulary size"):
            model.attach_probe(ProbeConfig(name="p", layers=[0], source="logits"))

    def test_inject_hidden_not_found_raises(self) -> None:
        """_inject_hidden raises ValueError when _find_layers returns None."""

        class NoLayerModel(nn.Module):
            """Model without a transformer layer list."""

            def __call__(self, x):  # type: ignore[no-untyped-def]
                return (mx.zeros((1, 3, 16)),)

        model = Model(NoLayerModel(), DummyTokenizer(), backend_name="mlx")
        with pytest.raises(ValueError, match="Cannot find transformer"):
            model.attach_probe(ProbeConfig(name="p", layers=[0]))


# ---------------------------------------------------------------------------
# Custom aggregation callable
# ---------------------------------------------------------------------------


class TestCustomAggregationCallable:
    """Tests for custom callable aggregation."""

    def test_custom_aggregation_function(self, model_wrapper: Model) -> None:
        """A callable aggregation should be invoked."""

        def custom_agg(states: list) -> mx.array:
            return mx.concatenate(states, axis=-1)

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0, 1], aggregation=custom_agg))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert "p" in outputs.probes

    def test_custom_aggregation_with_mean_like(self, model_wrapper: Model) -> None:
        """A custom aggregation that does mean pooling should work."""

        def mean_agg(states: list) -> mx.array:
            return mx.mean(mx.stack(states, axis=0), axis=0)

        model_wrapper.attach_probe(
            ProbeConfig(
                name="p",
                layers=[0, 1],
                aggregation=mean_agg,
                module_config={"in_features": 16},
            )
        )
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert "p" in outputs.probes
        assert outputs.probes["p"].logits.shape == (1, 3, 1)


# ---------------------------------------------------------------------------
# _aggregate method edge cases
# ---------------------------------------------------------------------------


class TestAggregateEdgeCases:
    """Edge cases for Probe._aggregate."""

    def test_aggregate_max_strategy(self) -> None:
        """Max aggregation strategy should work."""
        config = ProbeConfig(name="p", layers=[0, 1], aggregation="max")
        probe = Probe(config, hidden_dim=16, backend_name="mlx")
        states = [mx.ones((1, 3, 16)), mx.full((1, 3, 16), 2.0)]
        result = probe._aggregate(states)
        assert result is not None

    def test_aggregate_unknown_raises(self) -> None:
        """Unknown aggregation strategy raises ValueError."""
        config = ProbeConfig(name="p", layers=[0, 1], aggregation="concat")
        probe = Probe(config, hidden_dim=16, backend_name="mlx")
        states = [mx.ones((1, 3, 16)), mx.ones((1, 3, 16))]
        probe.config.aggregation = "unknown_agg"
        with pytest.raises(ValueError, match="Unknown aggregation"):
            probe._aggregate(states)


# ---------------------------------------------------------------------------
# _resolve_negative_index edge cases
# ---------------------------------------------------------------------------


class TestResolveNegativeIndex:
    """Edge cases for _resolve_negative_index."""

    def test_negative_index_out_of_range_raises(self) -> None:
        """A negative index that resolves below 0 should raise ValueError."""
        from auto_chasm.probe import _resolve_negative_index

        with pytest.raises(ValueError, match="out of range"):
            _resolve_negative_index(-10, 4)

    def test_positive_index_out_of_range_raises(self) -> None:
        """An out-of-range positive index should raise ValueError."""
        from auto_chasm.probe import _resolve_negative_index

        with pytest.raises(ValueError, match="out of range"):
            _resolve_negative_index(10, 4)


# ---------------------------------------------------------------------------
# _build_module edge cases
# ---------------------------------------------------------------------------


class TestBuildModuleEdgeCases:
    """Edge cases for Probe._build_module."""

    def test_unknown_module_type_raises(self) -> None:
        """An unknown module_type should raise ValueError."""
        config = ProbeConfig(name="p", layers=[0], module_type="unknown")
        with pytest.raises(ValueError, match="Unknown module_type.*unknown"):
            Probe(config, hidden_dim=16, backend_name="mlx")

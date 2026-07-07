"""Oracle tests for four documented behaviours — correct values vs ground truth.

Each test asserts a CORRECT result against an independently computed oracle,
never merely "it runs":

* **Class means** (``auto_chasm.class_means``): per-class mean hidden states
  must equal the mean of the class-0 / class-1 vectors with ``-100`` (ignore)
  positions EXCLUDED.  Flipping the token at a ``-100``-labeled position must
  leave the means unchanged (proves ``-100`` is dropped), while flipping a
  real class position's token must move that class's mean (proves the harness
  is live).  Both the MLX and PyTorch paths are checked.
* **Steering custom fn** (``Model.enable_steering(steer_fn=...)``): a custom
  ``steer_fn(hidden, head, logits)`` returning ``hidden + C`` must actually be
  invoked, and its output must flow downstream — a capture one layer below the
  steered layer (made an exact identity) shifts by EXACTLY ``C``.
* **Generation: greedy** (``auto_chasm.generation``): with ``temperature=0``
  generation is deterministic and equals the per-step argmax of the model
  logits, hand-verified against a model whose argmax token equals the current
  sequence length.
* **Granularity token** (``auto_chasm.probe``): a ``granularity="token"`` probe
  keeps the time axis — output is ``[B, T, out]`` — whereas ``"response"``
  pools it away to ``[B, out]``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from auto_chasm import Model, ProbeConfig
from auto_chasm.class_means import compute_class_means
from auto_chasm.config import SteeringConfig
from auto_chasm.generation import _generate_manual_mlx, _generate_manual_torch

# --- shared tiny synthetic models (no network) -----------------------------


class _TinyMlx(nn.Module):
    """Tiny MLX transformer-shaped model: embedding -> linear layers -> head."""

    def __init__(self, h: int = 4, v: int = 8, layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(v, h)
        self.layers = [nn.Linear(h, h) for _ in range(layers)]
        self.output_proj = nn.Linear(h, v)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h)


class _TinyMlxGelu(nn.Module):
    """Tiny MLX model with GELU non-linearities (varied per-token activations)."""

    def __init__(self, h: int = 16, v: int = 32, layers: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(v, h)
        self.layers = [nn.Linear(h, h) for _ in range(layers)]
        self.output_proj = nn.Linear(h, v)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    """Minimal config exposing the attributes the probe engine reads."""

    def __init__(self, h: int, v: int, layers: int) -> None:
        self.hidden_size = h
        self.num_hidden_layers = layers
        self.vocab_size = v


def _mlx_model(h: int = 4, v: int = 8, layers: int = 2, gelu: bool = False) -> Model:
    """Build a ``Model`` wrapping a tiny MLX network with a config attached."""
    base = _TinyMlxGelu(h, v, layers) if gelu else _TinyMlx(h, v, layers)
    m = Model(base, None, "mlx")
    m.model.config = _Cfg(h, v, layers)
    return m


def _torch_model(h: int = 4, v: int = 8, layers: int = 2) -> Model:
    """Build a ``Model`` wrapping a tiny PyTorch network with a config attached."""
    import torch.nn as tnn

    class _TinyTorch(tnn.Module):
        """Tiny PyTorch transformer-shaped model."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = tnn.Embedding(v, h)
            self.layers = tnn.ModuleList([tnn.Linear(h, h) for _ in range(layers)])
            self.output_proj = tnn.Linear(h, v)

        def forward(self, x: Any) -> Any:
            out = self.embedding(x)
            for layer in self.layers:
                out = layer(out)
            return self.output_proj(out)

    m = Model(_TinyTorch(), None, "torch")
    m.model.config = _Cfg(h, v, layers)
    return m


# The probe captures hidden states from ``model.forward(tokens[:, :-1])`` and the
# harness aligns them to ``labels[1:]``.  So label index ``j+1`` (== ``b_labels``
# index ``j``) is the class of the hidden state produced from token index ``j``.
# Labels below: original idx 1,2 -> class 0,0; idx 3 -> -100; idx 4 -> class 1.
# The -100 position's hidden state is produced from token index 2.
_LABELS = [0, 0, 1, -100, 1]
_TOKENS = [1, 2, 3, 4, 5]
_TOKENS_FLIP_IGNORED = [1, 2, 6, 4, 5]  # differ only at token idx 2 (the -100 pos)
_TOKENS_FLIP_CLASS0 = [7, 2, 3, 4, 5]  # differ only at token idx 0 (a class-0 pos)


def _ground_truth_means(probe: Any, model: Model, tokens: list[int], labels: list[int]):
    """Independently compute class means from captured states (MLX/torch)."""
    probe.clear_captured()
    if model.backend.name == "torch":
        import torch

        with torch.no_grad():
            model.forward(torch.tensor([tokens])[:, :-1])
        h = probe.get_captured_states()[0].float().detach().cpu().numpy()[0]
    else:
        model.forward(mx.array([tokens])[:, :-1])
        cap = probe.get_captured_states()[0]
        mx.eval(cap)
        h = np.array(cap)[0]
    b_labels = np.array(labels[1:])
    mean_0 = h[b_labels == 0].mean(axis=0)
    mean_1 = h[b_labels == 1].mean(axis=0)
    return mean_0, mean_1


def _as_np(t: Any) -> np.ndarray:
    """Convert an MLX or torch tensor to a numpy array."""
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    mx.eval(t)
    return np.array(t)


class TestClassMeansOracle:
    """Class means equal the masked per-class mean and exclude -100 positions."""

    def test_mlx_means_match_independent_ground_truth(self) -> None:
        m = _mlx_model()
        probe = m.attach_probe(
            ProbeConfig(name="p", layers=[0], aggregation="last", module_config={"out_features": 1})
        )
        res = compute_class_means(m, {"p": probe}, [(_TOKENS, _LABELS)], "mlx")
        gt0, gt1 = _ground_truth_means(probe, m, _TOKENS, _LABELS)
        assert np.allclose(_as_np(res["p"]["mean_0"]), gt0, atol=1e-5)
        assert np.allclose(_as_np(res["p"]["mean_1"]), gt1, atol=1e-5)

    def test_mlx_flipping_ignored_position_leaves_means_unchanged(self) -> None:
        m = _mlx_model()
        probe = m.attach_probe(
            ProbeConfig(name="p", layers=[0], aggregation="last", module_config={"out_features": 1})
        )
        base = compute_class_means(m, {"p": probe}, [(_TOKENS, _LABELS)], "mlx")
        # Flipping the token at the -100 position changes that position's hidden
        # state, yet the means must be identical because -100 is excluded.
        flipped = compute_class_means(m, {"p": probe}, [(_TOKENS_FLIP_IGNORED, _LABELS)], "mlx")
        assert np.allclose(_as_np(base["p"]["mean_0"]), _as_np(flipped["p"]["mean_0"]), atol=1e-6)
        assert np.allclose(_as_np(base["p"]["mean_1"]), _as_np(flipped["p"]["mean_1"]), atol=1e-6)

    def test_mlx_flipping_real_class_position_moves_that_mean(self) -> None:
        # Guards the test above against a vacuous pass: a live harness must move
        # mean_0 when a genuine class-0 position's hidden state changes.
        m = _mlx_model()
        probe = m.attach_probe(
            ProbeConfig(name="p", layers=[0], aggregation="last", module_config={"out_features": 1})
        )
        base = compute_class_means(m, {"p": probe}, [(_TOKENS, _LABELS)], "mlx")
        flipped = compute_class_means(m, {"p": probe}, [(_TOKENS_FLIP_CLASS0, _LABELS)], "mlx")
        assert not np.allclose(
            _as_np(base["p"]["mean_0"]), _as_np(flipped["p"]["mean_0"]), atol=1e-6
        )

    def test_torch_means_match_and_exclude_ignored(self) -> None:
        m = _torch_model()
        probe = m.attach_probe(
            ProbeConfig(name="p", layers=[0], aggregation="last", module_config={"out_features": 1})
        )
        res = compute_class_means(m, {"p": probe}, [(_TOKENS, _LABELS)], "torch")
        gt0, gt1 = _ground_truth_means(probe, m, _TOKENS, _LABELS)
        assert np.allclose(_as_np(res["p"]["mean_0"]), gt0, atol=1e-5)
        assert np.allclose(_as_np(res["p"]["mean_1"]), gt1, atol=1e-5)
        flipped = compute_class_means(m, {"p": probe}, [(_TOKENS_FLIP_IGNORED, _LABELS)], "torch")
        assert np.allclose(_as_np(res["p"]["mean_0"]), _as_np(flipped["p"]["mean_0"]), atol=1e-6)
        assert np.allclose(_as_np(res["p"]["mean_1"]), _as_np(flipped["p"]["mean_1"]), atol=1e-6)


class TestSteeringCustomFnOracle:
    """A custom steer_fn is invoked and its exact output flows downstream."""

    def _model_with_identity_second_layer(self) -> Model:
        """Tiny model whose layer 1 is an exact identity (preserves the shift)."""
        m = _mlx_model(h=4, v=8, layers=2)
        # layer[1] := identity so a constant added at layer-0 output reaches the
        # layer-1 capture unchanged.
        m.model.layers[1].weight = mx.eye(4)
        m.model.layers[1].bias = mx.zeros(4)
        return m

    def test_custom_fn_invoked_with_expected_signature(self) -> None:
        m = self._model_with_identity_second_layer()
        m.attach_probe(
            ProbeConfig(
                name="p0", layers=[0], aggregation="last", module_config={"out_features": 1}
            )
        )
        seen: list[tuple] = []

        def steer_fn(hidden: Any, head: Any, logits: Any) -> Any:
            seen.append((tuple(hidden.shape), tuple(logits.shape)))
            return hidden + 1.0

        m.enable_steering("p0", config=SteeringConfig(method="custom"), steer_fn=steer_fn)
        m.forward(mx.array([[1, 2, 3]]))
        assert len(seen) == 1, "custom steer_fn must be invoked exactly once per forward"
        hidden_shape, logits_shape = seen[0]
        assert hidden_shape == (1, 3, 4)  # [B, T, hidden_dim]
        assert logits_shape == (1, 3)  # head logits squeezed to [B, T]

    def test_custom_fn_output_shifts_downstream_by_exact_constant(self) -> None:
        const = 7.0
        m = self._model_with_identity_second_layer()
        m.attach_probe(
            ProbeConfig(
                name="p0", layers=[0], aggregation="last", module_config={"out_features": 1}
            )
        )
        # A second probe one layer below the steered layer observes the
        # downstream hidden state (layer 1 is identity, so it equals the
        # steered layer-0 output).
        p1 = m.attach_probe(
            ProbeConfig(
                name="p1", layers=[1], aggregation="last", module_config={"out_features": 1}
            )
        )
        x = mx.array([[1, 2, 3]])

        m.forward(x)
        base = _as_np(p1.get_captured_states()[0])

        def steer_fn(hidden: Any, head: Any, logits: Any) -> Any:
            return hidden + const

        m.enable_steering("p0", config=SteeringConfig(method="custom"), steer_fn=steer_fn)
        m.forward(x)
        steered = _as_np(p1.get_captured_states()[0])

        # The downstream capture must be shifted by EXACTLY the constant the
        # user fn added — proves the fn's output is what flows forward.
        assert np.allclose(steered - base, const, atol=1e-4)

    def test_disabling_steering_restores_unsteered_output(self) -> None:
        m = self._model_with_identity_second_layer()
        m.attach_probe(
            ProbeConfig(
                name="p0", layers=[0], aggregation="last", module_config={"out_features": 1}
            )
        )
        p1 = m.attach_probe(
            ProbeConfig(
                name="p1", layers=[1], aggregation="last", module_config={"out_features": 1}
            )
        )
        x = mx.array([[1, 2, 3]])
        m.forward(x)
        base = _as_np(p1.get_captured_states()[0])

        m.enable_steering(
            "p0",
            config=SteeringConfig(method="custom"),
            steer_fn=lambda hidden, head, logits: hidden + 5.0,
        )
        m.disable_steering("p0")
        m.forward(x)
        restored = _as_np(p1.get_captured_states()[0])
        assert np.allclose(restored, base, atol=1e-5)


class _StepArgmaxModel:
    """MLX model whose argmax token id equals the current sequence length."""

    def __init__(self, vocab: int = 64) -> None:
        self.vocab = vocab

    def __call__(self, x: mx.array) -> tuple:
        b, t = x.shape
        logits = mx.full((b, t, self.vocab), -50.0)
        logits[:, -1, t % self.vocab] = mx.array(100.0)
        return (logits,)


class _StepArgmaxModelTorch:
    """PyTorch model whose argmax token id equals the current sequence length."""

    def __init__(self, vocab: int = 64) -> None:
        self.vocab = vocab

    def __call__(self, x: Any) -> tuple:
        import torch

        b, t = x.shape
        logits = torch.full((b, t, self.vocab), -50.0)
        logits[:, -1, t % self.vocab] = 100.0
        return (logits,)


class _IdTokenizer:
    """Tokenizer encoding to a fixed 3-token prompt and decoding ids to text."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids if i > 0)


def _generated_ids(text: str) -> list[int]:
    """Recover token ids from the id-tokenizer's space-joined decode."""
    return [int(s) for s in text.split()] if text else []


class TestGreedyGenerationOracle:
    """temperature=0 generation is deterministic and equals per-step argmax."""

    def test_greedy_equals_per_step_argmax_mlx(self) -> None:
        # Prompt length 3.  Step t feeds a length-(t) sequence, so the argmax
        # token is exactly the current length: 3, 4, 5, 6.
        out = _generate_manual_mlx(_StepArgmaxModel(), _IdTokenizer(), "x", 4, 0.0)
        assert _generated_ids(out) == [3, 4, 5, 6]

    def test_greedy_is_deterministic_mlx(self) -> None:
        a = _generate_manual_mlx(_StepArgmaxModel(), _IdTokenizer(), "x", 4, 0.0)
        b = _generate_manual_mlx(_StepArgmaxModel(), _IdTokenizer(), "x", 4, 0.0)
        assert a == b

    def test_first_token_is_argmax_of_fixed_logits_mlx(self) -> None:
        # A fixed-logits model with a hand-chosen argmax; the first greedy token
        # must equal that argmax index.
        scores = [-1.0, 0.0, 9.0, 3.0, 2.0, 1.0]
        expected = int(np.argmax(scores))

        class _Fixed:
            """Returns the same logit vector at the last position every step."""

            def __call__(self, x: mx.array) -> tuple:
                b, t = x.shape
                logits = mx.full((b, t, len(scores)), -50.0)
                logits[:, -1, :] = mx.array(scores)
                return (logits,)

        out = _generate_manual_mlx(_Fixed(), _IdTokenizer(), "x", 1, 0.0)
        assert _generated_ids(out) == [expected]

    def test_greedy_equals_per_step_argmax_torch(self) -> None:
        out = _generate_manual_torch(_StepArgmaxModelTorch(), _IdTokenizer(), "x", 4, 0.0)
        assert _generated_ids(out) == [3, 4, 5, 6]

    def test_greedy_is_deterministic_torch(self) -> None:
        a = _generate_manual_torch(_StepArgmaxModelTorch(), _IdTokenizer(), "x", 4, 0.0)
        b = _generate_manual_torch(_StepArgmaxModelTorch(), _IdTokenizer(), "x", 4, 0.0)
        assert a == b


class TestGranularityTokenOracle:
    """granularity='token' preserves the time axis; 'response' pools it away."""

    def _token_probe(self, out_features: int) -> tuple[Model, ProbeConfig]:
        m = _mlx_model(h=16, v=32, layers=4, gelu=True)
        cfg = ProbeConfig(
            name="p",
            layers=[0],
            aggregation="last",
            granularity="token",
            module_config={"out_features": out_features},
        )
        m.attach_probe(cfg)
        return m, cfg

    def test_token_binary_preserves_time_axis(self) -> None:
        m, _ = self._token_probe(out_features=1)
        out = m.forward(mx.array([[1, 2, 3, 4, 5]]))
        # One prediction per token: time axis T=5 preserved (not pooled).
        assert out.probes["p"].logits.shape == (1, 5, 1)

    def test_token_multiclass_preserves_time_axis(self) -> None:
        m, _ = self._token_probe(out_features=3)
        out = m.forward(mx.array([[1, 2, 3, 4, 5]]))
        assert out.probes["p"].logits.shape == (1, 5, 3)

    def test_response_pools_time_axis_away(self) -> None:
        # Contrast: 'response' removes the time axis -> [B, out].
        m = _mlx_model(h=16, v=32, layers=4, gelu=True)
        m.attach_probe(
            ProbeConfig(
                name="r",
                layers=[0],
                aggregation="last",
                granularity="response",
                module_config={"out_features": 3},
            )
        )
        out = m.forward(mx.array([[1, 2, 3, 4, 5]]))
        assert out.probes["r"].logits.shape == (1, 3)

    def test_token_output_is_not_constant_over_time(self) -> None:
        # The time axis must carry real per-token information, not a broadcast
        # of a single pooled value.
        m, _ = self._token_probe(out_features=3)
        out = m.forward(mx.array([[1, 2, 3, 4, 5]]))
        tok = _as_np(out.probes["p"].logits)
        assert not np.allclose(tok[0, 0], tok[0, 1], atol=1e-4)

    def test_response_equals_mean_over_token_outputs(self) -> None:
        # Ground-truth tie between the two granularities: with no padding mask,
        # the 'response' pooled vector equals the time-mean of the per-token
        # outputs on the same hidden states.
        m, _ = self._token_probe(out_features=3)
        out = m.forward(mx.array([[1, 2, 3, 4, 5]]))
        tok = _as_np(out.probes["p"].logits)
        probe = m.probes["p"]
        captured = probe.get_captured_states()
        pooled = probe._masked_mean_over_time(probe.module(captured[0]), None)
        assert np.allclose(_as_np(pooled), tok.mean(axis=1), atol=1e-5)

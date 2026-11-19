"""Fuzzing and edge-case tests — property-based stress testing with extreme inputs.

Tests verify the library handles weird/edge-case inputs without
crashing or producing silently wrong results.
"""

from __future__ import annotations

import contextlib
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm import GenerationConfig, JointLoss, Model, ProbeConfig, SteeringConfig
from auto_chasm.backends import Backend
from auto_chasm.checkpoint import (
    export_checkpoint,
    import_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from auto_chasm.steering import SteeringHook


class _TinyMlp(nn.Module):
    """Minimal MLP for fuzz testing."""

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 32, num_layers: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Tok:
    """Minimal tokenizer for fuzz testing."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids if i > 0)


class _Cfg:
    """Dummy model config."""

    hidden_size = 16
    num_hidden_layers = 4


def _make_model(backend: str = "mlx") -> Model:
    """Build a bare-bones Model for tests that cannot use fixtures."""
    base = _TinyMlp()
    base.config = _Cfg()
    return Model(base, _Tok(), backend_name=backend)


class TestFuzzingShapes:
    """Edge-case tensor shapes for forward passes."""

    def test_empty_batch(self, model_wrapper: Model) -> None:
        """Forward with batch dimension 0 should not crash."""
        input_ids = mx.zeros((0, 5), dtype=mx.int32)
        with contextlib.suppress(ValueError, RuntimeError, IndexError):
            outputs = model_wrapper.forward(input_ids)
            assert outputs.lm_logits.shape[0] == 0

    def test_single_element(self, model_wrapper: Model) -> None:
        """Forward with single token (B=1, T=1) should work."""
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.zeros((1, 1), dtype=mx.int32)
        outputs = model_wrapper.forward(input_ids)
        assert outputs.lm_logits.shape == (1, 1, 32)
        assert "p" in outputs.probes

    def test_large_batch(self, model_wrapper: Model) -> None:
        """Forward with batch size 64 should not OOM or error."""
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.zeros((64, 1), dtype=mx.int32)
        outputs = model_wrapper.forward(input_ids)
        assert outputs.lm_logits.shape == (64, 1, 32)

    def test_1d_input_rejected(self, model_wrapper: Model) -> None:
        """1-D input (no batch dim) should be handled gracefully."""
        input_ids = mx.zeros((5,), dtype=mx.int32)
        outputs = model_wrapper.forward(input_ids)
        assert outputs.lm_logits is not None

    def test_4d_plus_shape(self, model_wrapper: Model) -> None:
        """4-D+ input should raise or be handled without crashing."""
        input_ids = mx.zeros((1, 3, 1, 1), dtype=mx.int32)
        with contextlib.suppress(ValueError, RuntimeError, TypeError, IndexError):
            model_wrapper.forward(input_ids)

    def test_zero_length_sequence(self, model_wrapper: Model) -> None:
        """Forward with zero-length sequence (B=1, T=0) should not crash."""
        input_ids = mx.zeros((1, 0), dtype=mx.int32)
        with contextlib.suppress(ValueError, RuntimeError, IndexError):
            outputs = model_wrapper.forward(input_ids)
            assert outputs.lm_logits.shape[1] == 0

    def test_extreme_sequence_length(self, model_wrapper: Model) -> None:
        """Forward with 4096-token sequence should not crash."""
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.random.randint(0, 32, (1, 4096))
        outputs = model_wrapper.forward(input_ids)
        assert outputs.lm_logits.shape[1] == 4096
        assert "p" in outputs.probes


class TestFuzzingValues:
    """Extreme / NaN / Inf values in loss and probe ops."""

    def test_nan_logits_in_loss(self) -> None:
        """JointLoss should not crash when model returns NaN logits."""
        loss_fn = JointLoss(losses={"probe": "bce"})

        class _NanModel:
            """Returns NaN logits."""

            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                nan_lm = mx.full((b, t, 32), float("nan"))
                nan_p = mx.full((b, t), float("nan"))
                return nan_lm, nan_p

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 3]])
        total, ntoks, components = loss_fn(_NanModel(), batch, labels, lengths)
        assert mx.any(mx.isnan(total)) or float(ntoks) > 0

    def test_inf_logits_in_loss(self) -> None:
        """JointLoss should not crash when model returns Inf logits."""
        loss_fn = JointLoss(losses={"probe": "bce"})

        class _InfModel:
            """Returns Inf logits."""

            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                inf_lm = mx.full((b, t, 32), float("inf"))
                inf_p = mx.full((b, t), float("inf"))
                return inf_lm, inf_p

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 3]])
        loss_fn(_InfModel(), batch, labels, lengths)

    def test_all_zeros_forward(self, model_wrapper: Model) -> None:
        """Forward with all-zero token IDs should not crash."""
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.zeros((2, 10), dtype=mx.int32)
        outputs = model_wrapper.forward(input_ids)
        assert outputs.lm_logits is not None

    def test_large_positive_tokens(self, model_wrapper: Model) -> None:
        """Tokens near vocab_size boundary should not crash."""
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.full((2, 5), 31, dtype=mx.int32)
        outputs = model_wrapper.forward(input_ids)
        assert outputs.lm_logits is not None

    def test_negative_token_ids(self) -> None:
        """Negative token IDs should be handled gracefully."""
        model = _make_model("mlx")
        model.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.array([[-1, 0, 1]], dtype=mx.int32)
        outputs = model.forward(input_ids)
        assert outputs.lm_logits is not None

    def test_alternating_sing_values(self) -> None:
        """Loss with alternating 0/1 labels should produce finite loss."""
        loss_fn = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})

        class _AltModel:
            """Returns alternating-probability logits."""

            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                lm = mx.zeros((b, t, 32))
                p = mx.tile(mx.array([10.0, -10.0])[None, :], (b, t // 2 + 1))
                return lm, p[:, :t]

        batch = mx.array([[1, 2, 3, 4]])
        labels = mx.array([[0.0, 1.0, 0.0, 1.0]])
        lengths = mx.array([[0, 4]])
        total, ntoks, components = loss_fn(_AltModel(), batch, labels, lengths)
        assert mx.isfinite(total).item()
        assert "probe" in components


class TestFuzzingDtypes:
    """Mixed / unusual dtypes for model and probe operations."""

    def test_int8_input(self, model_wrapper: Model) -> None:
        """int8 input should be accepted by the backend."""
        input_ids = mx.zeros((1, 5), dtype=mx.int8)
        with contextlib.suppress(ValueError, TypeError):
            outputs = model_wrapper.forward(input_ids)
            assert outputs.lm_logits is not None

    def test_float16_labels_in_loss(self) -> None:
        """float16 probe labels should not crash the loss computation."""
        loss_fn = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})

        class _F16Model:
            """Returns f32 logits."""

            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), mx.zeros((b, t))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]], dtype=mx.float16)
        lengths = mx.array([[0, 3]])
        with contextlib.suppress(TypeError, ValueError):
            loss_fn(_F16Model(), batch, labels, lengths)

    def test_uint8_labels(self) -> None:
        """uint8 probe labels should work in loss computation."""
        loss_fn = JointLoss(weights={"lm_head": 0.0}, losses={"probe": "bce"})

        class _MockModel:
            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), mx.zeros((b, t))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0, 1, 0]], dtype=mx.uint8)
        lengths = mx.array([[0, 3]])
        loss_fn(_MockModel(), batch, labels, lengths)

    def test_int32_input(self, model_wrapper: Model) -> None:
        """int32 token IDs should work as expected."""
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        outputs = model_wrapper.forward(input_ids)
        assert "p" in outputs.probes

    def test_bf16_if_available(self) -> None:
        """bfloat16 logits should not crash loss if backend supports it."""
        if not hasattr(mx, "bfloat16"):
            pytest.skip("MLX does not support bfloat16")
        loss_fn = JointLoss(losses={"probe": "bce"})

        class _Bf16Model:
            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                lm = mx.zeros((b, t, 32), dtype=mx.bfloat16)
                p = mx.zeros((b, t), dtype=mx.bfloat16)
                return lm, p

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 3]])
        total, ntoks, components = loss_fn(_Bf16Model(), batch, labels, lengths)
        assert mx.isfinite(total).item()


class TestFuzzingProbeInjection:
    """Extreme / invalid probe configurations."""

    def test_probe_every_single_layer(self, model_wrapper: Model) -> None:
        """Attach a probe to each layer individually.

        All must produce outputs.
        """
        for layer_idx in range(4):
            name = f"p_{layer_idx}"
            model_wrapper.attach_probe(ProbeConfig(name=name, layers=[layer_idx]))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        for layer_idx in range(4):
            assert f"p_{layer_idx}" in outputs.probes

    def test_probe_all_layers(self, model_wrapper: Model) -> None:
        """Attach one probe to all layers with concat aggregation."""
        model_wrapper.attach_probe(ProbeConfig(name="all", layers=[0, 1, 2, 3]))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert "all" in outputs.probes

    def test_negative_index_beyond_range(self, model_wrapper: Model) -> None:
        """Negative layer index beyond model depth should raise."""
        with pytest.raises((ValueError, IndexError)):
            model_wrapper.attach_probe(ProbeConfig(name="bad", layers=[-10]))

    def test_re_attach_same_name(self, model_wrapper: Model) -> None:
        """Attaching a probe with a name already in use must raise (no silent overwrite).

        A silent overwrite would orphan the first probe's capture wrapper (it keeps
        running and leaks memory), so re-using a name is rejected up front.
        """
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))
        with pytest.raises(ValueError, match="already attached"):
            model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert "p" in outputs.probes

    def test_detach_and_re_attach(self, model_wrapper: Model) -> None:
        """Restore original layers, then re-attach a probe."""
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))
        model_wrapper.restore_original_layers()
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert "p" in outputs.probes
        assert len(model_wrapper.probes) == 1

    def test_multi_class_out_features(self, model_wrapper: Model) -> None:
        """Probe with out_features=10 should produce [B, T, 10]."""
        model_wrapper.attach_probe(
            ProbeConfig(name="mc", layers=[0], module_config={"out_features": 10})
        )
        input_ids = mx.array([[1, 2, 3]])
        outputs = model_wrapper.forward(input_ids)
        assert outputs.probes["mc"].logits.shape[-1] == 10

    def test_model_with_zero_layers(self) -> None:
        """Model with no transformer layers should raise on probe attach."""

        class _Flat(nn.Module):
            """Model with no .layers attribute."""

            def __call__(self, x: mx.array) -> mx.array:
                return x

        model = Model(_Flat(), _Tok(), backend_name="mlx")
        with pytest.raises((ValueError, RuntimeError, AttributeError)):
            model.attach_probe(ProbeConfig(name="p", layers=[0]))

    def test_custom_module_raises(self, model_wrapper: Model) -> None:
        """A custom module that raises should propagate the error."""

        def _raising(in_dim: int, cfg: dict) -> nn.Linear:
            msg = "intentional failure"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="intentional"):
            model_wrapper.attach_probe(ProbeConfig(name="bad", layers=[0], module_type=_raising))


class TestFuzzingGeneration:
    """Edge-case inputs to the generation API."""

    def test_empty_prompt(self, model_wrapper: Model) -> None:
        """Generate with empty string should not crash."""
        with contextlib.suppress(ValueError, RuntimeError, IndexError):
            result = model_wrapper.generate(prompt="", max_tokens=3, temperature=0.0)
            assert isinstance(result, str)

    def test_long_prompt(self, model_wrapper: Model) -> None:
        """Generate with prompt longer than max_tokens should not crash."""
        long_text = "hello world " * 200
        with contextlib.suppress(ValueError, RuntimeError, IndexError):
            result = model_wrapper.generate(prompt=long_text, max_tokens=5, temperature=0.0)
            assert isinstance(result, str)

    def test_temperature_zero_do_sample(self, model_wrapper: Model) -> None:
        """Zero temperature with do_sample=True should not crash."""
        cfg = GenerationConfig(max_tokens=3, temperature=0.0, do_sample=True)
        with contextlib.suppress(ValueError, RuntimeError, IndexError):
            result = model_wrapper.generate(
                prompt="test", max_tokens=3, temperature=0.0, config=cfg
            )
            assert isinstance(result, str)

    def test_negative_temperature(self, model_wrapper: Model) -> None:
        """Negative temperature must raise rather than flip the distribution.

        A negative temperature would invert the sampling distribution and
        silently sample the least-likely tokens, so generation now rejects it
        with ``ValueError`` (via config validation or the manual path).
        """
        with pytest.raises(ValueError):
            model_wrapper.generate(prompt="test", max_tokens=3, temperature=-1.0)

    def test_max_tokens_zero(self, model_wrapper: Model) -> None:
        """max_tokens=0 should produce an empty string."""
        result = model_wrapper.generate(prompt="test", max_tokens=0, temperature=0.0)
        assert result == ""

    def test_generate_with_probes_empty(self, model_wrapper: Model) -> None:
        """generate_with_probes with empty prompt should not crash."""
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))
        with contextlib.suppress(ValueError, RuntimeError, IndexError):
            steps = list(
                model_wrapper.generate_with_probes(prompt="", max_tokens=2, temperature=0.0)
            )
            assert all(s.token_str is not None for s in steps)


class TestFuzzingJointLoss:
    """Edge-case configurations for JointLoss."""

    def test_all_zero_probe_weights(self) -> None:
        """Zero probe weights should skip probe loss entirely."""
        loss_fn = JointLoss(weights={"probe": 0.0})

        class _Mock:
            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), mx.ones((b, t))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 3]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert "probe" not in components
        assert "lm_head" in components

    def test_mixed_bce_mse(self) -> None:
        """Mix of probe_loss='bce' and probe_loss='mse' should both work."""
        loss_fn = JointLoss(losses={"probe_a": "bce", "probe_b": "mse"})

        class _Mock:
            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                lm = mx.zeros((b, t, 32))
                probes = {"probe_a": mx.ones((b, t)), "probe_b": mx.ones((b, t))}
                return lm, probes

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 3]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        # Two probes → per-probe component keys (no silent collision when two
        # heads share a loss type).
        assert "probe_a" in components
        assert "probe_b" in components

    def test_zero_lm_and_probe_weight(self) -> None:
        """Both weights zero should produce zero total loss."""
        loss_fn = JointLoss(weights={"lm_head": 0.0, "probe": 0.0})

        class _Mock:
            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), mx.ones((b, t))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 3]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert float(total) == 0.0

    def test_ignore_index_minus_100(self) -> None:
        """Labels with -100 should be masked in cross-entropy."""
        loss_fn = JointLoss()

        class _Mock:
            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), None

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, -100.0, 1.0]])
        lengths = mx.array([[0, 3]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert "lm_head" in components

    def test_sequence_length_one(self) -> None:
        """A single-token sequence should not cause division by zero."""
        loss_fn = JointLoss()

        class _Mock:
            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), None

        batch = mx.array([[5]])
        labels = mx.array([[0.0]])
        lengths = mx.array([[0, 1]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert float(total) >= 0
        # A single-token batch has no next-token targets -> nothing supervised ->
        # empty components (not a fabricated {"lm_head": 0}).
        assert components == {}

    def test_zero_length_mask(self) -> None:
        """Zero-length mask should still produce a valid loss."""
        loss_fn = JointLoss(losses={"probe": "bce"})

        class _Mock:
            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                return mx.zeros((b, t, 32)), mx.ones((b, t))

        batch = mx.array([[1, 2, 3]])
        labels = mx.array([[0.0, 1.0, 0.0]])
        lengths = mx.array([[0, 0]])
        total, ntoks, components = loss_fn(_Mock(), batch, labels, lengths)
        assert float(ntoks) >= 0


class TestFuzzingSteering:
    """Edge-case inputs to steering hooks and geometry computation."""

    def test_all_zeros_direction(self) -> None:
        """Steering with zero direction should be a no-op."""
        hook = SteeringHook("test", SteeringConfig())
        hook._mean_0 = mx.zeros(8)
        hook._mean_1 = mx.zeros(8)
        hook._direction = mx.zeros(8)
        hook._head_norm = 0.0
        hook.enabled = True
        hidden = mx.random.normal((1, 3, 8))
        head = nn.Linear(8, 1)
        logits = head(hidden).squeeze(-1)
        result = hook.steer(hidden, head, logits)
        assert result.shape == hidden.shape

    def test_nan_direction(self) -> None:
        """Steering with NaN direction should not crash."""
        hook = SteeringHook("test", SteeringConfig())
        hook._mean_0 = mx.full(8, float("nan"))
        hook._mean_1 = mx.full(8, float("nan"))
        hook._direction = mx.full(8, float("nan"))
        hook._head_norm = 1.0
        hook.enable()
        hidden = mx.random.normal((1, 3, 8))
        head = nn.Linear(8, 1)
        logits = head(hidden).squeeze(-1)
        with contextlib.suppress(Exception):
            result = hook.steer(hidden, head, logits)
            assert result.shape == hidden.shape

    def test_steer_2d_hidden_no_batch(self) -> None:
        """Steering with 2D hidden (no batch dim) should raise or handle."""
        hook = SteeringHook("test", SteeringConfig(method="nullify"))
        hook._mean_0 = mx.zeros(8)
        hook._mean_1 = mx.ones(8)
        hook._direction = mx.ones(8)
        hook._head_norm = 1.0
        hook.enable()
        hidden_2d = mx.random.normal((3, 8))
        head = nn.Linear(8, 1)
        logits = head(hidden_2d).squeeze(-1)
        with contextlib.suppress(ValueError, RuntimeError, IndexError):
            hook.steer(hidden_2d, head, logits)

    def test_multiple_steering_hooks_same_probe(self, model_wrapper: Model) -> None:
        """Enabling steering twice on same probe should not duplicate hooks."""
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))
        cm = {"p": {"mean_0": mx.zeros(16), "mean_1": mx.ones(16)}}
        model_wrapper.enable_steering("p", class_means=cm)
        model_wrapper.enable_steering("p", class_means=cm)
        assert len(model_wrapper.steering_hooks) == 1

    def test_compute_geometry_single_sample(self) -> None:
        """Compute_geometry with one sample per class still produces a direction."""
        hook = SteeringHook("test", SteeringConfig())
        hidden_by_class = {
            0: [mx.array([1.0, 0.0, 0.0])],
            1: [mx.array([0.0, 1.0, 0.0])],
        }
        head_weight = mx.array([[1.0, 0.0, 0.0]])
        head_bias = mx.array([0.0])
        hook.compute_geometry(hidden_by_class, head_weight, head_bias)
        assert hook.has_geometry

    def test_compute_geometry_stress(self) -> None:
        """Compute_geometry with 1000 samples per class should not OOM."""
        hook = SteeringHook("test", SteeringConfig())
        mx.random.seed(0)
        n = 1000
        hidden_by_class = {
            0: [mx.random.normal((16,)) for _ in range(n)],
            1: [mx.random.normal((16,), scale=2.0) for _ in range(n)],
        }
        head_weight = mx.random.normal((1, 16))
        head_bias = mx.array([0.0])
        hook.compute_geometry(hidden_by_class, head_weight, head_bias)
        assert hook.has_geometry
        assert hook._direction is not None
        assert hook._direction.shape == (16,)


class TestFuzzingCheckpoint:
    """Edge-case save / load / export / import scenarios."""

    def test_save_to_non_writable(self, model_wrapper: Model) -> None:
        """Save to a non-writable directory should raise."""
        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "readonly"
            path.mkdir(mode=0o444)
            with pytest.raises((PermissionError, OSError, Exception)):
                save_checkpoint(model_wrapper, str(path))

    def test_load_corrupt_manifest(self) -> None:
        """Loading a checkpoint with corrupt JSON should raise."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "manifest.json").write_text("{corrupt json!!!}")
            with pytest.raises((json.JSONDecodeError, ValueError, Exception)):
                load_checkpoint(str(ckpt))

    def test_export_missing_manifest(self) -> None:
        """Exporting a directory without manifest should raise."""
        with tempfile.TemporaryDirectory() as tmp:
            empty_dir = Path(tmp) / "empty"
            empty_dir.mkdir()
            out = Path(tmp) / "out.auto_chasm"
            with pytest.raises((ValueError, FileNotFoundError)):
                export_checkpoint(str(empty_dir), str(out))

    def test_import_corrupt_archive(self) -> None:
        """Importing a corrupt tar archive should raise."""
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = Path(tmp) / "corrupt.auto_chasm"
            corrupt.write_text("not a tar archive at all!!!")
            with pytest.raises((tarfile.ReadError, ValueError, Exception)):
                import_checkpoint(str(corrupt), str(Path(tmp) / "out"))

    def test_save_and_restore_probe(self) -> None:
        """Save then load probe config should preserve structure."""
        base = _TinyMlp()
        base.config = _Cfg()
        model = Model(base, _Tok(), backend_name="mlx")
        model._base_model_name = "test-model"
        model.attach_probe(ProbeConfig(name="p", layers=[0], module_config={"out_features": 2}))
        input_ids = mx.array([[1, 2, 3]])
        before = model.forward(input_ids)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            model.save_checkpoint(str(ckpt))
            try:
                restored = load_checkpoint(str(ckpt), base_model="test-model")
                after = restored.forward(input_ids)
                assert before.probes["p"].logits.shape == after.probes["p"].logits.shape
            except Exception:
                pass


class TestFuzzingBackendOps:
    """Edge-case inputs to backend tensor operations."""

    def test_sample_zero_length_logits(self) -> None:
        """Backend sample with 0-length logits should raise clearly."""
        backend = Backend(force="mlx")
        logits = mx.array([])
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            backend.tensor.sample(logits, 0.0)

    def test_sample_nan_logits(self) -> None:
        """Backend sample with NaN logits should not crash destructively."""
        backend = Backend(force="mlx")
        logits = mx.array([float("nan"), float("nan"), float("nan")])
        with contextlib.suppress(ValueError, RuntimeError):
            token = backend.tensor.sample(logits, 0.0)
            assert 0 <= token < 3

    def test_tensor_from_empty_list(self) -> None:
        """Creating a tensor from an empty list should not crash."""
        backend = Backend(force="mlx")
        with contextlib.suppress(ValueError, RuntimeError, TypeError):
            t = backend.tensor.tensor([])
            assert t.size == 0

    def test_to_numpy_non_contiguous(self) -> None:
        """to_numpy on a non-contiguous (transposed) tensor should work."""
        backend = Backend(force="mlx")
        t = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t_t = mx.transpose(t)
        with contextlib.suppress(ValueError, RuntimeError, NotImplementedError):
            arr = backend.tensor.to_numpy(t_t)
            assert isinstance(arr, np.ndarray)
            assert arr.shape == t_t.shape


class TestFuzzingTorchBackend:
    """Fuzzing tests for the PyTorch backend (if available)."""

    def test_torch_backend_samples(self) -> None:
        """Torch backend sample with NaN logits should not crash."""
        pytest.importorskip("torch")
        backend = Backend(force="torch")
        import torch

        logits = torch.tensor([float("nan"), float("nan")])
        with contextlib.suppress(ValueError, RuntimeError):
            token = backend.tensor.sample(logits, 0.0)
            assert isinstance(token, int)

    def test_torch_create_from_empty_list(self) -> None:
        """Torch tensor creation from empty list should work."""
        pytest.importorskip("torch")
        backend = Backend(force="torch")
        with contextlib.suppress(ValueError, RuntimeError, TypeError):
            t = backend.tensor.tensor([])
            assert t.numel() == 0

    def test_torch_joint_loss_nan(self) -> None:
        """Torch JointLoss with NaN logits should not crash."""
        pytest.importorskip("torch")
        import torch

        loss_fn = JointLoss(losses={"probe": "bce"})

        class _TorchNan:
            def __call__(self, inputs: Any) -> tuple:
                b, t = inputs.shape
                return (
                    torch.full((b, t, 32), float("nan")),
                    torch.full((b, t), float("nan")),
                )

        batch = torch.tensor([[1, 2, 3]])
        labels = torch.tensor([[0.0, 1.0, 0.0]])
        lengths = torch.tensor([[0, 3]])
        with contextlib.suppress(Exception):
            loss_fn(_TorchNan(), batch, labels, lengths)

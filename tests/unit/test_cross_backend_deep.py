"""Deep coverage tests for cross-backend modules.

Covers class_means, checkpoint, steering, trainable, loss torch path,
peft execution, and cross-backend steering parity.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

# ===========================================================================
# Section 2: class_means.py coverage
# ===========================================================================


class TestClassMeansCompute:
    """Tests for class_means.py — compute_class_means and internal functions."""

    def test_compute_class_means_via_api(self, model_wrapper: Any, sample_dataset: Any) -> None:
        """Model.compute_class_means() via public API with tiny MLX model."""
        from auto_chasm.config import ProbeConfig

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))

        result = model_wrapper.compute_class_means(sample_dataset, batch_size=8, max_seq_length=5)

        assert "p" in result
        assert "mean_0" in result["p"]
        assert "mean_1" in result["p"]
        assert result["p"]["mean_0"].shape == (16,)
        assert result["p"]["mean_1"].shape == (16,)

    def test_compute_mlx_directly(self, model_wrapper: Any) -> None:
        """_compute_mlx with a tiny MLX model and custom iterate_batches."""
        from auto_chasm.class_means import _compute_mlx
        from auto_chasm.config import ProbeConfig

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        probe = model_wrapper._probes["p"]

        class StaticBatches:
            """Iterate batches returning one fixed batch."""

            def __call__(self, dataset, batch_size, max_seq_length, loop=False):
                tokens = np.array([[1, 2, 3, 0, 0]], dtype=np.int32)
                labels = np.array([[0, 0, 1, 0, 0]], dtype=np.int32)
                lengths = np.array([[0, 3]])
                yield tokens, labels, lengths

        mean_0, mean_1 = _compute_mlx(
            model_wrapper,
            probe,
            None,
            hidden_dim=16,
            batch_size=1,
            max_seq_length=5,
            iterate_batches=StaticBatches(),
        )

        assert mean_0.shape == (16,)
        assert mean_1.shape == (16,)
        assert mx.isfinite(mean_0).all()
        assert mx.isfinite(mean_1).all()

    def test_compute_torch_directly(self, torch_model_wrapper: Any) -> None:
        """_compute_torch with real iterate_batches loop."""
        import torch

        from auto_chasm.class_means import _compute_torch
        from auto_chasm.config import ProbeConfig

        torch.manual_seed(42)
        torch_model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        probe = torch_model_wrapper._probes["p"]

        class StaticBatches:
            """Iterate batches returning one fixed batch."""

            def __call__(self, dataset, batch_size, max_seq_length, loop=False):
                tokens = np.array([[1, 2, 3, 0, 0]], dtype=np.int32)
                labels = np.array([[0, 0, 1, 0, 0]], dtype=np.int32)
                lengths = np.array([[0, 3]])
                yield tokens, labels, lengths

        mean_0, mean_1 = _compute_torch(
            torch_model_wrapper,
            probe,
            None,
            hidden_dim=16,
            batch_size=1,
            max_seq_length=5,
            iterate_batches=StaticBatches(),
        )

        assert mean_0.shape == (16,)
        assert mean_1.shape == (16,)
        assert torch.isfinite(mean_0).all()
        assert torch.isfinite(mean_1).all()

    def test_all_same_class_labels(self, model_wrapper: Any) -> None:
        """compute_class_means with all labels being class 0 does not crash."""
        from auto_chasm.config import ProbeConfig

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))

        data = [{"tokens": [1, 2, 3], "labels": [0, 0, 0]} for _ in range(8)]
        result = model_wrapper.compute_class_means(data, batch_size=8, max_seq_length=5)

        assert "p" in result
        assert result["p"]["mean_0"].shape == (16,)
        assert result["p"]["mean_1"].shape == (16,)
        # mean_0 has data, mean_1 is 0/0 = NaN (no class-1 samples)
        assert mx.isfinite(result["p"]["mean_0"]).all()


# ===========================================================================
# Section 3: checkpoint.py load_checkpoint coverage
# ===========================================================================


class TestCheckpointCoverage:
    """Checkpoint save/load/import/export coverage tests."""

    def test_save_load_checkpoint_roundtrip(
        self, model_wrapper: Any, tiny_model: Any, monkeypatch: Any
    ) -> None:
        """Model.from_checkpoint() full roundtrip — save then load."""
        from auto_chasm.config import ProbeConfig
        from auto_chasm.model import Model

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        model_wrapper._base_model_name = "test-model"  # type: ignore[attr-defined]

        def mock_from_pretrained(
            model_name: str,
            backend_name: str | None = None,
            lora: Any = None,
            **kwargs: Any,
        ) -> Model:
            base_model, tokenizer = tiny_model
            instance = Model(base_model, tokenizer, backend_name=backend_name, lora_config=lora)
            instance._base_model_name = model_name
            return instance

        monkeypatch.setattr(Model, "from_pretrained", mock_from_pretrained)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "ckpt"
            model_wrapper.save_checkpoint(str(ckpt_path))
            loaded = Model.from_checkpoint(str(ckpt_path))

        assert "p" in loaded.probes

    def test_probe_weights_preserved_after_save_load(
        self, model_wrapper: Any, tiny_model: Any, monkeypatch: Any
    ) -> None:
        """Probe weights preserved after save/load roundtrip."""
        from auto_chasm.config import ProbeConfig
        from auto_chasm.model import Model

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        model_wrapper._base_model_name = "test-model"  # type: ignore[attr-defined]
        initial_weight = model_wrapper._probes["p"].module.weight

        def mock_from_pretrained(
            model_name: str,
            backend_name: str | None = None,
            lora: Any = None,
            **kwargs: Any,
        ) -> Model:
            base_model, tokenizer = tiny_model
            instance = Model(base_model, tokenizer, backend_name=backend_name, lora_config=lora)
            instance._base_model_name = model_name
            return instance

        monkeypatch.setattr(Model, "from_pretrained", mock_from_pretrained)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "ckpt"
            model_wrapper.save_checkpoint(str(ckpt_path))
            loaded = Model.from_checkpoint(str(ckpt_path))

        loaded_weight = loaded._probes["p"].module.weight
        diff = mx.abs(initial_weight - loaded_weight)
        assert float(diff.max().item()) < 1e-6, "Probe weights changed after save/load"

    def test_load_probe_weights_mlx(self, model_wrapper: Any, tmp_path: Any) -> None:
        """_load_probe_weights with MLX backend."""
        from auto_chasm._checkpoint_weights import (
            load_probe_weights as _load_probe_weights,
        )
        from auto_chasm._checkpoint_weights import (
            save_probe_weights as _save_probe_weights,
        )
        from auto_chasm.config import ProbeConfig

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        probe = model_wrapper._probes["p"]
        original_weight = probe.module.weight

        probe_path = tmp_path / "probe.safetensors"
        _save_probe_weights(probe, probe_path, model_wrapper.backend)

        # Zero out weights, then load
        probe.module.weight = mx.zeros_like(original_weight)
        _load_probe_weights(probe, probe_path, model_wrapper.backend)

        restored = probe.module.weight
        diff = mx.abs(original_weight - restored)
        assert float(diff.max().item()) < 1e-6, "MLX probe weight restore failed"

    def test_load_probe_weights_torch(self, torch_model_wrapper: Any, tmp_path: Any) -> None:
        """_load_probe_weights with torch backend."""
        import torch

        from auto_chasm._checkpoint_weights import (
            load_probe_weights as _load_probe_weights,
        )
        from auto_chasm._checkpoint_weights import (
            save_probe_weights as _save_probe_weights,
        )
        from auto_chasm.config import ProbeConfig

        torch.manual_seed(42)
        torch_model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        probe = torch_model_wrapper._probes["p"]
        original_weight = probe.module.weight.data.clone()

        probe_path = tmp_path / "probe.safetensors"
        _save_probe_weights(probe, probe_path, torch_model_wrapper.backend)

        # Zero out weights, then load
        probe.module.weight.data.zero_()
        _load_probe_weights(probe, probe_path, torch_model_wrapper.backend)

        assert torch.allclose(original_weight, probe.module.weight.data)

    def test_missing_probe_weights_raises(self, model_wrapper: Any, tmp_path: Any) -> None:
        """Missing probe weights file must raise, not silently keep untrained weights.

        Swallowing a missing-file load would leave the probe with random,
        untrained weights while the model looks "restored" — a research-poisoning
        footgun. A broken checkpoint must fail loudly.
        """
        import pytest

        from auto_chasm._checkpoint_weights import load_probe_weights as _load_probe_weights
        from auto_chasm.config import ProbeConfig

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))
        probe = model_wrapper._probes["p"]

        missing_path = tmp_path / "nonexistent.safetensors"
        with pytest.raises(FileNotFoundError):
            _load_probe_weights(probe, missing_path, model_wrapper.backend)

    def test_export_import_checkpoint_roundtrip(self, model_wrapper: Any) -> None:
        """export_checkpoint followed by import_checkpoint roundtrip."""
        from auto_chasm.checkpoint import export_checkpoint, import_checkpoint, save_checkpoint
        from auto_chasm.config import ProbeConfig

        model_wrapper.attach_probe(ProbeConfig(name="p", layers=[1]))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            save_checkpoint(model_wrapper, str(ckpt_dir))

            archive = Path(tmp) / "exported.auto_chasm"
            export_checkpoint(str(ckpt_dir), str(archive))

            import_dir = Path(tmp) / "imported"
            extracted = import_checkpoint(str(archive), str(import_dir))
            extracted_path = Path(extracted)

            manifest = json.loads((extracted_path / "manifest.json").read_text())
            assert "p" in manifest["probes"]


# ===========================================================================
# Section 4: steering.py deep coverage
# ===========================================================================


class TestSteeringDeepCoverage:
    """Deep coverage of steering module edge cases."""

    def test_disable_stops_steering(self) -> None:
        """SteeringHook.disable() actually stops steering."""
        from auto_chasm.config import SteeringConfig
        from auto_chasm.steering import SteeringHook

        config = SteeringConfig(method="nullify", scale=1.0)
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 16)
        hook._mean_1 = mx.array([1.0] * 16)
        hook._direction = hook._mean_1 - hook._mean_0
        hook._head_norm = 1.0
        hook.enable()

        hidden = mx.random.normal((2, 5, 16))
        head = nn.Linear(16, 1)
        logits = head(hidden).squeeze(-1)

        steered = hook.steer(hidden, head, logits)
        assert not float(mx.allclose(hidden, steered).item())

        hook.disable()
        assert not hook.enabled

        after_disable = hook.steer(hidden, head, logits)
        assert float(mx.allclose(hidden, after_disable).item())

    def test_nullify_deterministic(self) -> None:
        """Nullify method produces deterministic results."""
        from auto_chasm.config import SteeringConfig
        from auto_chasm.steering import SteeringHook

        config = SteeringConfig(method="nullify", scale=1.0)
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 16)
        hook._mean_1 = mx.array([2.0] * 16)
        hook._direction = hook._mean_1 - hook._mean_0
        hook._head_norm = 1.0
        hook.enabled = True

        mx.random.seed(42)
        hidden = mx.random.normal((2, 5, 16))
        head = nn.Linear(16, 1)
        logits = head(hidden).squeeze(-1)

        r1 = hook.steer(hidden, head, logits)
        r2 = hook.steer(hidden, head, logits)
        assert float(mx.allclose(r1, r2).item())

    def test_push_to_mean_deterministic(self) -> None:
        """Push_to_mean method produces deterministic results."""
        from auto_chasm.config import SteeringConfig
        from auto_chasm.steering import SteeringHook

        config = SteeringConfig(method="push_to_mean", scale=1.0)
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 16)
        hook._mean_1 = mx.array([2.0] * 16)
        hook._direction = hook._mean_1 - hook._mean_0
        hook._head_norm = 1.0
        hook.enabled = True

        mx.random.seed(42)
        hidden = mx.random.normal((2, 5, 16))
        head = nn.Linear(16, 1)
        logits = head(hidden).squeeze(-1)

        r1 = hook.steer(hidden, head, logits)
        r2 = hook.steer(hidden, head, logits)
        assert float(mx.allclose(r1, r2).item())

    def test_boundary_deterministic(self) -> None:
        """Boundary method produces deterministic results."""
        from auto_chasm.config import SteeringConfig
        from auto_chasm.steering import SteeringHook

        config = SteeringConfig(method="boundary", scale=1.0)
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 16)
        hook._mean_1 = mx.array([2.0] * 16)
        hook._direction = hook._mean_1 - hook._mean_0
        hook._head_norm = 1.0
        hook.enabled = True

        mx.random.seed(42)
        hidden = mx.random.normal((2, 5, 16))
        head = nn.Linear(16, 1)
        logits = head(hidden).squeeze(-1)

        r1 = hook.steer(hidden, head, logits)
        r2 = hook.steer(hidden, head, logits)
        assert float(mx.allclose(r1, r2).item())

    def test_scale_zero_produces_no_change(self) -> None:
        """Steering with scale=0 produces no change."""
        from auto_chasm.config import SteeringConfig
        from auto_chasm.steering import SteeringHook

        config = SteeringConfig(method="nullify", scale=0.0)
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 16)
        hook._mean_1 = mx.array([2.0] * 16)
        hook._direction = hook._mean_1 - hook._mean_0
        hook._head_norm = 1.0
        hook.enabled = True

        hidden = mx.random.normal((2, 5, 16))
        head = nn.Linear(16, 1)
        logits = head(hidden).squeeze(-1)

        result = hook.steer(hidden, head, logits)
        assert float(mx.allclose(hidden, result).item())

    def test_large_scale_does_not_crash(self) -> None:
        """Extremely large steering scale doesn't crash."""
        from auto_chasm.config import SteeringConfig
        from auto_chasm.steering import SteeringHook

        config = SteeringConfig(method="nullify", scale=1e6)
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 16)
        hook._mean_1 = mx.array([2.0] * 16)
        hook._direction = hook._mean_1 - hook._mean_0
        hook._head_norm = 1.0
        hook.enabled = True

        hidden = mx.ones((2, 5, 16))
        head = nn.Linear(16, 1)
        logits = head(hidden).squeeze(-1)

        result = hook.steer(hidden, head, logits)
        assert result.shape == hidden.shape
        assert mx.isfinite(result).all()

    def test_one_token_sequences(self) -> None:
        """Steering with 1-token sequences (edge case in _steer_mlx)."""
        from auto_chasm.config import SteeringConfig
        from auto_chasm.steering import SteeringHook

        config = SteeringConfig(method="nullify", scale=1.0)
        hook = SteeringHook("test", config)
        hook._mean_0 = mx.array([0.0] * 16)
        hook._mean_1 = mx.array([2.0] * 16)
        hook._direction = hook._mean_1 - hook._mean_0
        hook._head_norm = 1.0
        hook.enabled = True

        hidden = mx.random.normal((2, 1, 16))
        head = nn.Linear(16, 1)
        logits = head(hidden).squeeze(-1)

        result = hook.steer(hidden, head, logits)
        assert result.shape == (2, 1, 16)
        assert mx.isfinite(result).all()

    def test_steer_mlx_torch_parity(self) -> None:
        """_steer_mlx and _steer_torch produce identical results on same data."""
        pytest.importorskip("torch")
        import torch

        from auto_chasm.steering import _steer_mlx, _steer_torch

        hidden_dim = 16
        batch, seq_len = 2, 5

        mx.random.seed(42)
        torch.manual_seed(42)

        hidden_mlx = mx.random.normal((batch, seq_len, hidden_dim))
        hidden_torch = torch.tensor(np.array(hidden_mlx))

        mean_0_mlx = mx.array([0.0] * hidden_dim)
        mean_1_mlx = mx.array([1.0] * hidden_dim)
        direction_mlx = mean_1_mlx - mean_0_mlx

        mean_0_torch = torch.zeros(hidden_dim)
        mean_1_torch = torch.ones(hidden_dim)
        direction_torch = mean_1_torch - mean_0_torch

        head_mlx = nn.Linear(hidden_dim, 1)
        head_torch = torch.nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            head_torch.weight.copy_(torch.tensor(np.array(head_mlx.weight)))
            head_torch.bias.copy_(torch.tensor(np.array(head_mlx.bias)))

        logits_mlx = head_mlx(hidden_mlx).squeeze(-1)
        logits_torch = head_torch(hidden_torch).squeeze(-1)

        for method in ("nullify", "push_to_mean", "boundary"):
            result_mlx = _steer_mlx(
                hidden_mlx,
                head_mlx,
                logits_mlx,
                method,
                mean_0_mlx,
                mean_1_mlx,
                direction_mlx,
            )
            result_torch = _steer_torch(
                hidden_torch,
                head_torch,
                logits_torch,
                method,
                mean_0_torch,
                mean_1_torch,
                direction_torch,
            )

            diff = mx.abs(result_mlx - mx.array(result_torch.detach().numpy()))
            max_diff = float(diff.max().item())
            assert max_diff < 1e-4, (
                f"Steering method '{method}': MLX and torch disagree (max diff={max_diff})"
            )

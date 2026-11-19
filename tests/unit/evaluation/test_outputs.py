"""Tests for output container convenience methods."""

from __future__ import annotations

import pytest

from auto_chasm.outputs import JointOutputs, ProbeOutput


def test_probe_output_bce():
    """BCE loss should return a positive scalar."""
    import mlx.core as mx

    logits = mx.array([[0.5, -0.5]])
    targets = mx.array([[1.0, 0.0]])
    po = ProbeOutput(logits=logits)
    loss = po.bce(targets)
    assert float(loss) > 0


def test_probe_output_mse():
    """MSE loss should return a positive scalar."""
    import mlx.core as mx

    logits = mx.array([[1.0, 2.0]])
    targets = mx.array([[0.5, 1.5]])
    po = ProbeOutput(logits=logits)
    loss = po.mse(targets)
    assert float(loss) > 0


def test_probe_output_ce():
    """CE loss should return a positive scalar for class-indices."""
    import mlx.core as mx

    logits = mx.array([[[0.5, 0.5], [0.5, 0.5]]])
    targets = mx.array([[0, 0]])
    po = ProbeOutput(logits=logits)
    loss = po.ce(targets)
    assert float(loss) > 0


def test_probe_output_bce_with_mask():
    """Masking should change the BCE loss value."""
    import mlx.core as mx

    logits = mx.array([[0.5, -0.5, 0.0]])
    targets = mx.array([[1.0, 0.0, 0.0]])
    mask = mx.array([[True, False, True]])
    po = ProbeOutput(logits=logits)
    loss_masked = po.bce(targets, mask=mask)
    loss_unmasked = po.bce(targets)
    assert float(loss_masked) != float(loss_unmasked)


def test_joint_outputs_constructs():
    """JointOutputs should compute mask and ntoks from lengths."""
    import mlx.core as mx

    lm_logits = mx.zeros((1, 5, 32))
    probes = {"digit": ProbeOutput(logits=mx.zeros((1, 5)))}
    targets = mx.array([[1, 2, 3, 4, 5]])
    lengths = mx.array([[0, 5]])
    outputs = JointOutputs(lm_logits, probes, targets, lengths)
    assert outputs.ntoks > 0
    assert outputs.mask is not None


def test_joint_outputs_lm_ce():
    """lm_ce property should return a positive scalar."""
    import mlx.core as mx

    lm_logits = mx.random.normal((1, 5, 32))
    probes: dict[str, ProbeOutput] = {}
    targets = mx.array([[1, 2, 3, 4, 5]])
    lengths = mx.array([[0, 5]])
    outputs = JointOutputs(lm_logits, probes, targets, lengths)
    ce = outputs.lm_ce
    assert float(ce) > 0


def test_joint_outputs_imported():
    """JointOutputs is importable from its submodule (``auto_chasm.outputs``).

    It is intentionally NOT part of the curated top-level surface (Phase 4b slim) —
    custom-loss authors receive a ``JointOutputs``/``ProbeOutput`` as an argument
    rather than importing it — but must stay reachable from its home module.
    """
    from auto_chasm.outputs import JointOutputs

    assert JointOutputs is not None


class TestOutputsTorchBackend:
    """Torch backend paths for output loss methods."""

    def test_probe_output_bce_torch(self) -> None:
        """BCE loss works with torch tensors."""
        pytest.importorskip("torch")
        import torch

        logits = torch.tensor([[0.5, -0.5]])
        targets = torch.tensor([[1.0, 0.0]])
        po = ProbeOutput(logits=logits)
        loss = po.bce(targets)
        assert float(loss) > 0

    def test_probe_output_mse_torch(self) -> None:
        """MSE loss works with torch tensors."""
        pytest.importorskip("torch")
        import torch

        logits = torch.tensor([[1.0, 2.0]])
        targets = torch.tensor([[0.5, 1.5]])
        po = ProbeOutput(logits=logits)
        loss = po.mse(targets)
        assert float(loss) > 0

    def test_probe_output_ce_torch(self) -> None:
        """CE loss works with torch tensors."""
        pytest.importorskip("torch")
        import torch

        logits = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
        targets = torch.tensor([[0, 0]])
        po = ProbeOutput(logits=logits)
        loss = po.ce(targets)
        assert float(loss) > 0

    def test_joint_outputs_torch(self) -> None:
        """JointOutputs mask/ntoks/lm_ce work with torch tensors."""
        pytest.importorskip("torch")
        import torch

        lm_logits = torch.zeros((1, 5, 32))
        probes: dict[str, ProbeOutput] = {}
        targets = torch.tensor([[1, 2, 3, 4, 5]])
        lengths = torch.tensor([[0, 5]])
        outputs = JointOutputs(lm_logits, probes, targets, lengths)
        assert outputs.ntoks > 0
        assert outputs.mask is not None
        ce = outputs.lm_ce
        assert float(ce) > 0

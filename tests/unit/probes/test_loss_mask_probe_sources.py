"""Regression tests: loss-method masking, probe source/layer, param count.

- Finding B: ProbeOutput.bce/ce/mse/mae fall back to the bound ``.mask`` (a 2-arg
  custom loss no longer silently averages padding into the loss).
- Probe Finding 1: ``embedding``/``logits`` sources reject multi-layer configs
  (they read a single site; multi-layer crashed or silently dropped layers).
- F4: ``count_parameters`` dispatches on the tensor backend, not MLX-importability.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from auto_chasm import ProbeConfig
from auto_chasm.outputs import ProbeOutput
from auto_chasm.utils import count_parameters


def test_finding_b_loss_methods_honor_bound_mask() -> None:
    """bce/mse fall back to the bound mask, so padding no longer contaminates the loss."""
    logits = mx.array([[10.0, 10.0, -10.0, -10.0]])  # last two are padding
    targets = mx.array([[1.0, 1.0, 1.0, 1.0]])
    mask = mx.array([[True, True, False, False]])

    bound = ProbeOutput(logits=logits, mask=mask)
    # With the bound mask, the padding positions (wrong predictions) are excluded.
    assert float(bound.bce(targets)) < 1e-3
    assert float(bound.bce(targets)) == pytest.approx(float(bound.bce(targets, mask=mask)))
    # Unmasked ProbeOutput still averages over everything (behavior unchanged).
    assert float(ProbeOutput(logits=logits, mask=None).bce(targets)) > 1.0


def test_finding_b_ce_honors_bound_mask() -> None:
    """ce falls back to the bound mask too."""
    logits = mx.array([[[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0]]])  # [B=1, T=3, C=2]
    targets = mx.array([[0, 1, 1]])  # last position is padding-labeled wrong
    mask = mx.array([[True, True, False]])
    bound = ProbeOutput(logits=logits, mask=mask)
    assert float(bound.ce(targets)) < 1e-2  # padding position excluded -> tiny loss


def test_probe1_multilayer_single_site_source_rejected() -> None:
    """embedding/logits sources reject a multi-layer config (single-site sources)."""
    for source in ("embedding", "logits"):
        with pytest.raises(ValueError, match="single site"):
            ProbeConfig(name="p", layers=[0, 1, 2], source=source)
        ProbeConfig(name="ok", layers=[0], source=source)  # single layer is fine
    # A genuine per-layer source spans multiple layers without complaint.
    ProbeConfig(name="h", layers=[0, 1, 2], source="hidden")


def test_f4_count_parameters_dispatches_on_backend() -> None:
    """count_parameters works on both an MLX and a torch module (no importability bug)."""
    import mlx.nn as mnn

    assert count_parameters(mnn.Linear(4, 2)) == 10  # 4*2 weights + 2 bias
    torch = pytest.importorskip("torch")
    assert count_parameters(torch.nn.Linear(4, 2)) == 10

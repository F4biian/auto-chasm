"""Capture-wrapper bookkeeping for embedding/logits probe sources.

``embedding``/``logits`` probes replace a single module (``embed_tokens`` /
``lm_head``) at a dotted attribute path with a capture wrapper.  Unlike block
sources — tracked by index in ``Model._original_layers`` — these need their
original module remembered by path so the wrapper can be removed on restore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from auto_chasm.outputs import ProbeOutput
from auto_chasm.probe import (
    _find_embedding,
    _find_output_head,
    _set_module_by_path,
)

if TYPE_CHECKING:
    from auto_chasm.probe import Probe


def prepare_inputs(
    model: Any,
    backend: Any,
    input_ids: Any,
    attention_mask: Any,
    pool_mask: Any,
) -> tuple[Any, Any, Any]:
    """Convert list/ndarray inputs to backend tensors and move them to the device.

    Args:
        model: The base model (its first parameter gives the torch device).
        backend: The active :class:`~auto_chasm.backends.Backend`.
        input_ids: Token ids (tensor, ndarray, or nested list).
        attention_mask: Optional attention mask in the same forms.
        pool_mask: Optional response-region pool mask in the same forms.

    Returns:
        ``(input_ids, attention_mask, pool_mask)`` as backend tensors, on the
        model's device under PyTorch.
    """
    import numpy as np

    def _to_tensor(x: Any) -> Any:
        return backend.tensor.tensor(x) if isinstance(x, (list, np.ndarray)) else x

    input_ids = _to_tensor(input_ids)
    attention_mask = _to_tensor(attention_mask)
    pool_mask = _to_tensor(pool_mask)

    if backend.name == "torch":
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = None if attention_mask is None else attention_mask.to(device)
        pool_mask = None if pool_mask is None else pool_mask.to(device)
    return input_ids, attention_mask, pool_mask


def extract_lm_logits(lm_out: Any) -> Any:
    """Pull the LM logits out of a base-model forward result.

    Args:
        lm_out: The base model's output (a ``.logits`` object, a tuple whose
            first element is the logits, or the logits tensor itself).

    Returns:
        The LM logits tensor.
    """
    if hasattr(lm_out, "logits"):
        return lm_out.logits
    if isinstance(lm_out, tuple):
        return lm_out[0]
    return lm_out


def run_probes(
    probes: dict[str, Probe],
    input_ids: Any,
    pool_mask: Any,
) -> dict[str, ProbeOutput]:
    """Run every probe on its captured states and collect the outputs.

    The ``if captured`` guard handles the legitimate "no captured states" case,
    so genuine shape/compute errors from ``probe.forward`` propagate instead of
    being silently swallowed.

    Args:
        probes: Attached probes keyed by name.
        input_ids: Token ids ``[B, T]`` (needed for sentence granularity).
        pool_mask: Mask ``[B, T]`` threaded into pooling (response/padding).

    Returns:
        Per-probe :class:`~auto_chasm.outputs.ProbeOutput`, keyed by name.
    """
    probe_outputs: dict[str, ProbeOutput] = {}
    for name, probe in probes.items():
        captured = probe.get_captured_states()
        if captured:
            logits = probe.forward(captured, mask=pool_mask, input_ids=input_ids)
            probe_outputs[name] = ProbeOutput(
                logits=logits,
                hidden_states=captured,
                aggregated=len(captured) > 1,
            )
    return probe_outputs


def track_single_module(model: Any, source: str, originals: dict[str, Any]) -> None:
    """Record the original embedding/logits module before it is wrapped.

    Args:
        model: The base language model.
        source: ``"embedding"`` or ``"logits"``.
        originals: Map ``attr_path -> original module`` to update in place; an
            already-tracked path is left untouched so the first (genuine)
            original survives repeated attaches.
    """
    module, attr_path = (
        _find_embedding(model) if source == "embedding" else _find_output_head(model)
    )
    if attr_path is not None and attr_path not in originals:
        originals[attr_path] = module


def restore_single_modules(model: Any, originals: dict[str, Any]) -> None:
    """Reinstall every tracked original embedding/logits module, then clear.

    Args:
        model: The base language model.
        originals: Map ``attr_path -> original module`` produced by
            :func:`track_single_module`; emptied after restoration.
    """
    for attr_path, original in originals.items():
        _set_module_by_path(model, attr_path, original)
    originals.clear()

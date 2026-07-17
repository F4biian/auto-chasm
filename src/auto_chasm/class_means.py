"""Per-class mean hidden state computation for steering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from auto_chasm.logger import get_logger

if TYPE_CHECKING:
    import mlx.core as mx
    import torch

logger = get_logger(__name__)


def compute_class_means(
    model: Any,
    probes: dict[str, Any],
    dataset: Any,
    backend_name: str,
    batch_size: int = 8,
    max_seq_length: int = 256,
) -> dict[str, dict[str, Any]]:
    """Compute per-class mean hidden states for steering.

    Runs the model over the dataset and accumulates hidden states
    at each probe's injection layer, separated by binary label.

    Args:
        model: The ``Model`` instance.
        probes: Dict of probe name to ``Probe``.
        dataset: Dataset yielding ``(tokens, labels)`` tuples.
        backend_name: ``"mlx"`` or ``"torch"``.
        batch_size: Batch size for iteration.
        max_seq_length: Maximum sequence length.

    Returns:
        Dict mapping probe name to ``{"mean_0": tensor, "mean_1": tensor}``.
    """
    from auto_chasm.trainers.data_utils import iterate_batches

    result: dict[str, dict[str, Any]] = {}

    for probe_name, probe in probes.items():
        hidden_dim = probe.hidden_dim

        if backend_name == "mlx":
            mean_0, mean_1 = _compute_mlx(
                model, probe, dataset, hidden_dim, batch_size, max_seq_length, iterate_batches
            )
        else:
            mean_0, mean_1 = _compute_torch(
                model, probe, dataset, hidden_dim, batch_size, max_seq_length, iterate_batches
            )

        result[probe_name] = {"mean_0": mean_0, "mean_1": mean_1}

    return result


def _labels_for_probe(raw_labels: Any, probe_name: str) -> Any:
    """Select a probe's label array from a per-probe dict batch (or pass through).

    A dataset carrying the reserved ``"lm_head"`` LM-weight channel (or several
    probes) batches labels as a ``{name: array}`` dict; the class means must be
    computed from THIS probe's own labels — never the LM weights or another
    head's classes.

    Args:
        raw_labels: The batch labels — a numpy array or a ``{name: array}`` dict.
        probe_name: The probe whose labels to select.

    Returns:
        The label array for this probe.

    Raises:
        KeyError: If ``raw_labels`` is a dict without this probe's key (and no
            unambiguous single-probe fallback exists).
    """
    if not isinstance(raw_labels, dict):
        return raw_labels
    if probe_name in raw_labels:
        return raw_labels[probe_name]
    probe_keys = [k for k in raw_labels if k != "lm_head"]
    if len(probe_keys) == 1:  # single-probe data labeled under a different name
        return raw_labels[probe_keys[0]]
    raise KeyError(
        f"compute_class_means: batch labels are a per-probe dict without a "
        f"{probe_name!r} entry (keys: {sorted(raw_labels)}); cannot pick the "
        "class labels unambiguously."
    )


def _compute_mlx(
    model: Any,
    probe: Any,
    dataset: Any,
    hidden_dim: int,
    batch_size: int,
    max_seq_length: int,
    iterate_batches: Any,
) -> tuple[mx.array, mx.array]:
    """Compute class means for MLX backend."""
    import mlx.core as mx

    sum_0 = mx.zeros(hidden_dim)
    sum_1 = mx.zeros(hidden_dim)
    count_0 = mx.array(0.0)
    count_1 = mx.array(0.0)

    for raw_tokens, raw_labels, lengths in iterate_batches(
        dataset, batch_size, max_seq_length, loop=False
    ):
        tokens = mx.array(raw_tokens)
        labels = mx.array(_labels_for_probe(raw_labels, probe.name))
        probe.clear_captured()
        model.forward(tokens[:, :-1])
        captured = probe.get_captured_states()
        if not captured:
            continue
        h = captured[0]
        b_labels = labels[:, 1:].astype(mx.float32)
        steps = mx.arange(1, b_labels.shape[1] + 1)
        length_mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
        # Count by explicit class membership, gated by the length mask. This
        # excludes the -100 ignore sentinel (it is neither class 0 nor 1) and
        # any padding — counting it as a class silently corrupts the axis.
        m0 = mx.logical_and(b_labels == 0, length_mask).astype(mx.float32)
        m1 = mx.logical_and(b_labels == 1, length_mask).astype(mx.float32)
        sum_0 = sum_0 + mx.sum(h * mx.expand_dims(m0, -1), axis=(0, 1))
        sum_1 = sum_1 + mx.sum(h * mx.expand_dims(m1, -1), axis=(0, 1))
        count_0 = count_0 + mx.sum(m0)
        count_1 = count_1 + mx.sum(m1)
        mx.eval(sum_0, sum_1, count_0, count_1)

    eps = 1e-8
    return sum_0 / (count_0 + eps), sum_1 / (count_1 + eps)


def _compute_torch(
    model: Any,
    probe: Any,
    dataset: Any,
    hidden_dim: int,
    batch_size: int,
    max_seq_length: int,
    iterate_batches: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute class means for PyTorch backend."""
    import torch

    device = next(model.model.parameters()).device
    sum_0 = torch.zeros(hidden_dim, device=device)
    sum_1 = torch.zeros(hidden_dim, device=device)
    count_0: float = 0.0
    count_1: float = 0.0

    for raw_tokens, raw_labels, lengths in iterate_batches(
        dataset, batch_size, max_seq_length, loop=False
    ):
        device = next(model.model.parameters()).device
        tokens = torch.from_numpy(raw_tokens).long().to(device)
        labels = torch.from_numpy(_labels_for_probe(raw_labels, probe.name)).long().to(device)
        lengths_t = torch.from_numpy(lengths).to(device)
        probe.clear_captured()
        with torch.no_grad():
            model.forward(tokens[:, :-1])
        captured = probe.get_captured_states()
        if not captured:
            continue
        h = captured[0].float()
        b_labels = labels[:, 1:].float()
        steps = torch.arange(1, b_labels.shape[1] + 1, device=device)
        length_mask = (steps >= lengths_t[:, 0:1]) & (steps < lengths_t[:, 1:])
        # Count by explicit class membership; excludes -100 and padding.
        m0 = ((b_labels == 0) & length_mask).float()
        m1 = ((b_labels == 1) & length_mask).float()
        sum_0 += (h * m0.unsqueeze(-1)).sum(dim=(0, 1))
        sum_1 += (h * m1.unsqueeze(-1)).sum(dim=(0, 1))
        count_0 += m0.sum().item()
        count_1 += m1.sum().item()

    return sum_0 / max(count_0, 1), sum_1 / max(count_1, 1)

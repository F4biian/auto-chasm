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

    # ONE pass, however many probes. Every probe captures from the SAME forward,
    # so looping probes on the outside re-ran the whole corpus per probe -- a
    # 24-layer mass-mean sweep cost 24 passes to compute what one pass already
    # produced. The per-probe helpers still exist for a single probe.
    result: dict[str, dict[str, Any]] = {}
    if len(probes) == 1:
        (probe_name, probe), = probes.items()
        fn = _compute_mlx if backend_name == "mlx" else _compute_torch
        mean_0, mean_1 = fn(
            model, probe, dataset, probe.hidden_dim, batch_size, max_seq_length, iterate_batches
        )
        return {probe_name: {"mean_0": mean_0, "mean_1": mean_1}}

    accum = _MultiProbeAccumulator(probes, backend_name, model)
    for raw_tokens, raw_labels, lengths in iterate_batches(
        dataset, batch_size, max_seq_length, loop=False
    ):
        accum.step(raw_tokens, raw_labels, lengths)
    for probe_name in probes:
        mean_0, mean_1 = accum.means(probe_name)
        result[probe_name] = {"mean_0": mean_0, "mean_1": mean_1}
    return result


class _MultiProbeAccumulator:
    """Token-level per-class sums for EVERY probe, filled from one forward pass."""

    def __init__(self, probes: dict[str, Any], backend_name: str, model: Any) -> None:
        """Zero the running sums for each probe."""
        self.probes = probes
        self.backend = backend_name
        self.model = model
        self.sums: dict[str, list[Any]] = {}
        self.counts: dict[str, list[float]] = {name: [0.0, 0.0] for name in probes}
        if backend_name == "mlx":
            import mlx.core as mx

            self.sums = {n: [mx.zeros(p.hidden_dim), mx.zeros(p.hidden_dim)]
                         for n, p in probes.items()}
        else:
            import torch

            dev = next(model.model.parameters()).device
            self.sums = {n: [torch.zeros(p.hidden_dim, device=dev),
                             torch.zeros(p.hidden_dim, device=dev)]
                         for n, p in probes.items()}

    def step(self, raw_tokens: Any, raw_labels: Any, lengths: Any) -> None:
        """Run one batch and add each probe's masked per-class sums."""
        for probe in self.probes.values():
            probe.clear_captured()
        if self.backend == "mlx":
            import mlx.core as mx

            tokens = mx.array(raw_tokens)
            self.model.forward(tokens[:, :-1])
            for name, probe in self.probes.items():
                captured = probe.get_captured_states()
                if not captured:
                    continue
                h = captured[0]
                labels = mx.array(_labels_for_probe(raw_labels, name))
                b = labels[:, 1:].astype(mx.float32)
                steps = mx.arange(1, b.shape[1] + 1)
                lm = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
                for cls in (0, 1):
                    m = mx.logical_and(b == cls, lm).astype(mx.float32)
                    self.sums[name][cls] = self.sums[name][cls] + mx.sum(
                        h * mx.expand_dims(m, -1), axis=(0, 1)
                    )
                    self.counts[name][cls] += float(mx.sum(m))
                mx.eval(*self.sums[name])
            return
        import torch

        dev = next(self.model.model.parameters()).device
        tokens = torch.from_numpy(raw_tokens).long().to(dev)
        lengths_t = torch.from_numpy(lengths).to(dev)
        with torch.no_grad():
            self.model.forward(tokens[:, :-1])
        for name, probe in self.probes.items():
            captured = probe.get_captured_states()
            if not captured:
                continue
            h = captured[0].float()
            labels = torch.from_numpy(_labels_for_probe(raw_labels, name)).long().to(dev)
            b = labels[:, 1:].float()
            steps = torch.arange(1, b.shape[1] + 1, device=dev)
            lm = (steps >= lengths_t[:, 0:1]) & (steps < lengths_t[:, 1:])
            for cls in (0, 1):
                m = ((b == cls) & lm).float()
                self.sums[name][cls] += (h * m.unsqueeze(-1)).sum(dim=(0, 1))
                self.counts[name][cls] += float(m.sum().item())

    def means(self, name: str) -> tuple[Any, Any]:
        """Class means for ``name`` (sums / counts, guarded against empty classes)."""
        c0, c1 = self.counts[name]
        return self.sums[name][0] / max(c0, 1e-8), self.sums[name][1] / max(c1, 1e-8)


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


def save_class_means(model: Any, class_means: dict[str, dict[str, Any]], path: str) -> None:
    """Save class-mean vectors to one file (``.safetensors``/``.pth``).

    Lives here rather than inline on ``Model`` so all class-mean logic —
    computing the means and persisting them — sits in one module.

    Args:
        model: The ``Model`` whose backend performs the write.
        class_means: ``{probe_name: {"mean_0": tensor, "mean_1": tensor}}``.
        path: File path (``.safetensors`` for MLX, ``.pth`` for PyTorch).
    """
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.backend.wrapping.save_class_means(class_means, path)
    logger.info("Class means saved to %s", path)


def load_class_means(model: Any, path: str) -> dict[str, Any]:
    """Load class-mean vectors from a file (backend-agnostic, auto-format).

    Args:
        model: The ``Model`` whose backend performs the read.
        path: File path (auto-detected format).

    Returns:
        Dict with ``"mean_0"`` / ``"mean_1"`` tensors.
    """
    return model.backend.wrapping.load_class_means(path)  # type: ignore[no-any-return]

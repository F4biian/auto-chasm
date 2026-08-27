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
    second_moment: bool = False,
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
        second_moment: Also accumulate the centered second moment, adding
            ``"scatter"`` (about the overall mean), ``"mean"`` and ``"counts"``
            to each entry. Needed for a
            whitened direction; costs one ``hidden x hidden`` matrix per probe.

    Returns:
        Dict mapping probe name to ``{"mean_0": tensor, "mean_1": tensor}``, plus
        ``"scatter"``/``"mean"``/``"counts"`` when ``second_moment`` is set.
    """
    from auto_chasm.trainers.data_utils import iterate_batches

    # ONE pass, however many probes. Every probe captures from the SAME forward,
    # so looping probes on the outside re-ran the whole corpus per probe -- a
    # 24-layer mass-mean sweep cost 24 passes to compute what one pass already
    # produced. The per-probe helpers still exist for a single probe.
    result: dict[str, dict[str, Any]] = {}
    if len(probes) == 1 and not second_moment:
        (probe_name, probe), = probes.items()
        fn = _compute_mlx if backend_name == "mlx" else _compute_torch
        mean_0, mean_1 = fn(
            model, probe, dataset, probe.hidden_dim, batch_size, max_seq_length, iterate_batches
        )
        return {probe_name: {"mean_0": mean_0, "mean_1": mean_1}}

    accum = _MultiProbeAccumulator(probes, backend_name, model, second_moment=second_moment)
    for raw_tokens, raw_labels, lengths in iterate_batches(
        dataset, batch_size, max_seq_length, loop=False
    ):
        accum.step(raw_tokens, raw_labels, lengths)
    for probe_name in probes:
        mean_0, mean_1 = accum.means(probe_name)
        result[probe_name] = {"mean_0": mean_0, "mean_1": mean_1}
        if second_moment:
            result[probe_name]["scatter"] = accum.scatter(probe_name)
            result[probe_name]["mean"] = accum.mean(probe_name)
            result[probe_name]["counts"] = tuple(accum.counts[probe_name])
    return result


class _MultiProbeAccumulator:
    """Token-level per-class sums for EVERY probe, filled from one forward pass."""

    def __init__(self, probes: dict[str, Any], backend_name: str, model: Any,
                 second_moment: bool = False) -> None:
        """Zero the running sums for each probe.

        ``second_moment`` additionally accumulates ``sum(h h^T)`` per probe, which
        is what a whitened (LDA) direction needs. It costs one ``hidden x hidden``
        matrix per probe -- 26 MB at hidden=2560 in float32, so ~950 MB across 36
        layers -- and is therefore opt-in.
        """
        self.probes = probes
        self.backend = backend_name
        self.second_moment = second_moment
        self.model = model
        self.sums: dict[str, list[Any]] = {}
        self.counts: dict[str, list[float]] = {name: [0.0, 0.0] for name in probes}
        # Fixed shift for the second moment, frozen from the first batch holding
        # any labeled token. See .scatter() for why it is not accumulated raw.
        self.offset: dict[str, Any] = {}
        if backend_name == "mlx":
            import mlx.core as mx

            self.sums = {n: [mx.zeros(p.hidden_dim), mx.zeros(p.hidden_dim)]
                         for n, p in probes.items()}
            self.m2 = ({n: mx.zeros((p.hidden_dim, p.hidden_dim)) for n, p in probes.items()}
                       if second_moment else {})
        else:
            import torch

            dev = next(model.model.parameters()).device
            self.sums = {n: [torch.zeros(p.hidden_dim, device=dev),
                             torch.zeros(p.hidden_dim, device=dev)]
                         for n, p in probes.items()}
            self.m2 = ({n: torch.zeros((p.hidden_dim, p.hidden_dim), device=dev)
                        for n, p in probes.items()} if second_moment else {})

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
                if self.second_moment:
                    valid = mx.logical_and(mx.logical_or(b == 0, b == 1), lm)
                    vf = mx.expand_dims(valid.astype(mx.float32), -1)
                    nv = float(mx.sum(vf))
                    if nv:
                        if name not in self.offset:
                            self.offset[name] = mx.sum(h * vf, axis=(0, 1)) / nv
                            mx.eval(self.offset[name])
                        # Subtract BEFORE masking, so padded rows stay exactly zero.
                        hv = ((h - self.offset[name]) * vf).reshape(-1, h.shape[-1])
                        self.m2[name] = self.m2[name] + (hv.T @ hv)
                        mx.eval(self.m2[name])
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
            if self.second_moment:
                valid = (((b == 0) | (b == 1)) & lm).float().unsqueeze(-1)
                nv = float(valid.sum().item())
                if nv:
                    if name not in self.offset:
                        self.offset[name] = (h * valid).sum(dim=(0, 1)) / nv
                    # Subtract BEFORE masking, so padded rows stay exactly zero.
                    hv = ((h - self.offset[name]) * valid).reshape(-1, h.shape[-1])
                    self.m2[name] += hv.T @ hv

    def scatter(self, name: str) -> Any:
        """Scatter about the OVERALL mean: ``sum_i (h_i - mu)(h_i - mu)^T``.

        Accumulated already centered, around a fixed offset ``o`` frozen from the
        first batch, and corrected once at the end::

            S = sum_i (h_i - o)(h_i - o)^T - n (mu - o)(mu - o)^T

        The textbook identity (``sum h h^T - n mu mu^T``, i.e. ``o = 0``) is the
        same quantity algebraically and the wrong way to compute it. LLM hidden
        states have massive-activation dimensions, so ``||mu||^2`` runs orders of
        magnitude above the variance and the two terms nearly cancel: measured on
        a 135M model that lost ~3 of float32's ~7 digits and returned a
        mathematically PSD matrix with NEGATIVE eigenvalues, which then corrupts
        the ``Sigma^-1/2`` root. With ``o`` near the mean the correction term is
        small and the cancellation goes away, at no extra pass or memory.
        """
        mu = self.mean(name)
        n = sum(self.counts[name])
        if self.backend == "mlx":
            import mlx.core as mx

            d = mu - self.offset.get(name, mx.zeros_like(mu))
            return self.m2[name] - n * mx.outer(d, d)
        import torch

        d = mu - self.offset.get(name, torch.zeros_like(mu))
        return self.m2[name] - n * torch.outer(d, d)

    def mean(self, name: str) -> Any:
        """The OVERALL mean hidden state over both classes' labeled tokens."""
        m0, m1 = self.means(name)
        c0, c1 = self.counts[name]
        return (c0 * m0 + c1 * m1) / max(c0 + c1, 1.0)

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


def fit_mass_mean(
    model: Any,
    dataset: Any,
    *,
    probe_names: list[str] | None = None,
    whiten: bool = False,
    shrinkage: float = 1e-2,
    calibrate_scale: bool = False,
    calibrate_bias: bool = False,
    batch_size: int = 8,
    max_seq_length: int = 1024,
) -> dict[str, dict[str, Any]]:
    """Fit a MASS-MEAN probe per attached head: no training, one streaming pass.

    The direction is ``theta = mean_1 - mean_0`` over token-level hidden states,
    and it is written INTO each probe's linear head (``weight = theta``, bias set
    so the midpoint of the two class means scores 0). Every downstream tool then
    works unchanged — ``model.probe_scores`` for per-token scores, its clustered
    bootstrap for confidence intervals — because the head simply computes
    ``h . theta`` like any other linear probe.

    AUROC depends only on ``theta / |theta|``, so the scale and the bias are free
    parameters: they set where the decision threshold sits, never the ranking.

    Memory is O(hidden) per probe — sums are accumulated, states never stored — so
    this is safe on a corpus of any size, and all probes are filled from ONE pass.

    Args:
        model: A ``Model`` with single-logit linear probes attached.
        dataset: The data to fit the means on (the TRAIN split).
        probe_names: Which probes to fit (``None`` = all attached).
        whiten: Fit the overall mean ``mu`` and covariance ``Sigma`` of the hidden
            states as well, and score the WHITENED state instead of the raw one::

                h_white = Sigma^-1/2 (h - mu)

            The direction is then the same difference of class means, measured in
            that space. OFF by default, so the probe stays the plain projection.

            Worth trying when a mass-mean probe sits far below a trained linear
            one: hidden states are strongly anisotropic, so ``mu_1 - mu_0`` picks
            up whatever high-variance nuisance direction happens to lie between
            the centroids, and a plain projection cannot discount it. Whitening is
            still closed-form and still ONE pass — the covariance accumulates
            alongside the means — but it costs one ``hidden x hidden`` matrix per
            probe (26 MB at hidden=2560, so ~950 MB across 36 layers) and needs
            far more states than dimensions to be estimated well.

            Note this is the OVERALL covariance, not the pooled within-class one:
            the transform describes the hidden states, and knows nothing about the
            labels. (Using the within-class covariance instead would make it LDA.)
        shrinkage: Ridge added to the covariance as a fraction of its mean
            eigenvalue, before the inverse square root. Without it the small
            eigenvalues are noise and ``1/sqrt`` of them is enormous.
        calibrate_scale: Scale the direction so the class means sit at logits
            ``-+2``. OFF by default: a mass-mean probe is the plain projection
            ``h . theta``, and scaling it is a separate choice.
        calibrate_bias: Offset so the midpoint of the class means scores 0. OFF by
            default, and the head's bias is ZEROED either way — leaving the random
            initialisation there would shift every score by an arbitrary constant.
            Under ``whiten`` the bias instead carries the ``- mu`` of the whitening
            transform, unless this picks the class midpoint as the reference.

    Neither affects AUROC, which reads only the ranking and is invariant to any
    positive rescaling or shift. They exist solely to make ``loss`` (and, for the
    bias, ``acc``/``macro_f1``) interpretable: uncalibrated, ``|theta|`` is 38-67
    on a 576-dim model, so cross-entropy lands in the tens.
        batch_size: Batch size for the forward passes.
        max_seq_length: Truncation length.

    Returns:
        ``{probe_name: {"mean_0", "mean_1", "theta", "scale", "bias"}}`` as NumPy
        arrays, plus ``"mean"``, ``"cov"``, ``"whitener"`` (``Sigma^-1/2``) and
        ``"theta_whitened"`` when ``whiten=True`` — so the transform can be
        applied to states directly, not only through the probe.

    Raises:
        ValueError: If a named probe's head is not a single-logit linear layer,
            since there would be nowhere to write the direction.
    """
    import numpy as np

    from auto_chasm.metrics import to_numpy

    names = list(probe_names if probe_names is not None else model.probes)
    probes = {n: model.probes[n] for n in names}
    raw = compute_class_means(model, probes, dataset, model.backend.name,
                              batch_size=batch_size, max_seq_length=max_seq_length,
                              second_moment=whiten)

    out: dict[str, dict[str, Any]] = {}
    for name in names:
        m0 = to_numpy(raw[name]["mean_0"]).astype(np.float64)
        m1 = to_numpy(raw[name]["mean_1"]).astype(np.float64)
        theta = m1 - m0
        wh = _whiten(theta, raw[name], shrinkage, name) if whiten else None
        # Scoring the whitened state is still a plain linear read of the raw one:
        #   theta_w . (Sigma^-1/2 (h - mu)) == (Sigma^-1/2 theta_w) . (h - mu)
        # so the transform folds into the weight and the centering into the bias.
        direction = wh["whitener"] @ wh["theta_whitened"] if wh else theta
        module = probes[name].module
        weight = getattr(module, "weight", None)
        if weight is None or tuple(weight.shape) != (1, direction.shape[0]):
            raise ValueError(
                f"Probe {name!r} has no single-logit linear head to write the mass-mean "
                f"direction into (expected weight of shape (1, {direction.shape[0]}), got "
                f"{None if weight is None else tuple(weight.shape)}). Build the sweep with "
                "ModuleSpec.linear(out_features=1)."
            )
        scale = 1.0
        if calibrate_scale:
            spread = float(direction @ (m1 - m0))
            scale = 4.0 / spread if spread > 0 else 1.0
        w = direction * scale
        if calibrate_bias:
            bias = -float(w @ (m0 + m1) / 2.0)
        elif wh:
            bias = -float(w @ wh["mean"])  # the "- mu" of the whitening transform
        else:
            # ZERO unless asked for: the head arrives randomly initialised, and
            # leaving that bias in place offsets every score by a constant.
            bias = 0.0
        _write_linear(module, w, bias, model.backend.name)
        out[name] = {"mean_0": m0, "mean_1": m1, "theta": theta,
                     "scale": scale, "bias": bias}
        # Hang the transform off the probe so it is saved with the checkpoint and
        # can be applied to states directly; None when refitting without whitening.
        probes[name].whitening = (
            {"mean": wh["mean"], "whitener": wh["whitener"], "cov": wh["cov"]}
            if wh else None
        )
        if wh:
            out[name].update(wh)
    return out


def _whiten(
    theta: Any, entry: dict[str, Any], shrinkage: float, name: str
) -> dict[str, Any]:
    """Fit the whitening transform: the overall mean and ``Sigma^-1/2``.

    Fitting produces two fixed quantities, the mean ``mu`` and the covariance
    ``Sigma`` of the hidden states. Any state is then whitened by centering and
    applying the inverse square root::

        h_white = Sigma^-1/2 (h - mu)

    and the mass-mean direction is simply the difference of class means measured
    in that space, ``theta_white = Sigma^-1/2 theta``.

    ``Sigma^-1/2`` is taken through an eigendecomposition (``Sigma`` is symmetric),
    which also yields the transform itself rather than only its action on
    ``theta`` -- useful for whitening states directly.

    Returns:
        ``{"mean", "cov", "whitener", "theta_whitened"}``.
    """
    import numpy as np

    from auto_chasm.metrics import to_numpy

    n = float(sum(entry["counts"]))
    cov = to_numpy(entry["scatter"]).astype(np.float64) / max(n - 1.0, 1.0)
    cov = (cov + cov.T) / 2.0  # kill any drift from the accumulation
    dim = cov.shape[0]
    if n < 2 * dim:
        logger.warning(
            "Probe %r: whitening a %d-dim covariance from only %d hidden states. The "
            "estimate is poor below a few states per dimension — raise the data or the "
            "shrinkage, or leave whiten=False.",
            name, dim, int(n),
        )
    # Ridge toward the identity before the root: the covariance is hidden x hidden
    # estimated from token counts that are not always comfortably larger, so the
    # small eigenvalues are noise and 1/sqrt(noise) is enormous.
    cov.flat[:: dim + 1] += shrinkage * float(np.trace(cov)) / dim

    evals, evecs = np.linalg.eigh(cov)
    floor = max(float(evals.max()), 1e-12) * 1e-10
    whitener = (evecs * np.clip(evals, floor, None) ** -0.5) @ evecs.T
    return {
        "mean": to_numpy(entry["mean"]).astype(np.float64),
        "cov": cov,
        "whitener": whitener,
        "theta_whitened": whitener @ np.asarray(theta, dtype=np.float64),
    }


def _write_linear(module: Any, theta: Any, bias: float, backend_name: str) -> None:
    """Set a linear head's weight/bias, on either backend."""
    import numpy as np

    w = np.asarray(theta, dtype=np.float32).reshape(1, -1)
    if backend_name == "mlx":
        import mlx.core as mx

        module.weight = mx.array(w)
        if getattr(module, "bias", None) is not None:
            module.bias = mx.array(np.array([bias], dtype=np.float32))
        return
    import torch

    with torch.no_grad():
        dev, dtype = module.weight.device, module.weight.dtype
        module.weight.copy_(torch.tensor(w, device=dev, dtype=dtype))
        if getattr(module, "bias", None) is not None:
            module.bias.copy_(torch.tensor([bias], device=dev, dtype=dtype))

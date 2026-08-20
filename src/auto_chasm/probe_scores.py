"""Per-token probe scores, with clustered bootstrap confidence intervals.

An eval loop returns one aggregate number per metric and discards the underlying
``(score, label)`` pairs, so there is nothing left to put an error bar around.
This runs the dataset once and keeps those pairs — for EVERY attached probe from
the same forward pass — which is all a confidence interval needs.

**Bootstrap by RESPONSE, never by token.** Tokens inside one response share a
prompt, a model, and a hallucination span, so they are nowhere near independent.
Resampling tokens pretends a corpus of ~1200 responses is ~78k independent
observations, and the interval comes out several times too narrow (measured on
realistic correlated data: 0.026 wide token-level vs 0.070 clustered — a 2.7x
understatement). :meth:`ProbeScores.bootstrap` therefore resamples whole GROUPS
by default, and the group defaults to the response.

The AUROC here is also the CORPUS value, not the token-weighted mean of per-batch
AUROCs an eval loop reports — close, but not the same number, and only the corpus
one is what a confidence interval is around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from auto_chasm.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProbeScores:
    """Per-token scores and labels for one or more probes, plus their groups.

    Attributes:
        scores: ``{probe_name: [N] float}`` — the head's raw output per token.
        labels: ``[N]`` targets, with ignored/padding positions already removed.
        groups: ``[N]`` cluster id per token (the response, or a dataset
            ``"group"`` field when the samples carry one).
        probe_names: The probes present, in attachment order.
    """

    scores: dict[str, np.ndarray]
    labels: np.ndarray
    groups: np.ndarray
    probe_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        """Number of scored tokens."""
        return int(self.labels.shape[0])

    def statistic(self, name: str, fn: Any = None) -> float:
        """Corpus value of ``fn`` (default AUROC) for one probe.

        Args:
            name: Probe name.
            fn: ``(scores, labels) -> float``. ``None`` uses AUROC.

        Returns:
            The statistic over every scored token.
        """
        return float(_resolve_statistic(fn)(self.scores[name], self.labels))

    def auroc(self, name: str) -> float:
        """Corpus AUROC for one probe (``nan`` if only one class is present)."""
        return self.statistic(name)

    def aurocs(self) -> dict[str, float]:
        """Corpus AUROC for every probe."""
        return {n: self.auroc(n) for n in self.scores}

    def bootstrap(
        self,
        name: str | None = None,
        *,
        n_boot: int = 1000,
        ci: float = 95.0,
        seed: int = 0,
        cluster: bool = True,
        method: str = "percentile",
        statistic: Any = None,
    ) -> dict[str, tuple[float, float, float]]:
        """Bootstrap a confidence interval per probe, on SHARED resamples.

        Every probe is evaluated on the same resampled draws, so differences
        between probes (adjacent layers, say) share their sampling noise and the
        curves are comparable — bootstrapping each one independently makes every
        layer wobble on its own and hides whether a peak is real.

        Args:
            name: A single probe, or ``None`` for all of them.
            n_boot: Resamples to draw.
            ci: Central interval width in percent (95 -> 2.5/97.5).
            seed: Seed for the resampling.
            cluster: Resample whole GROUPS (correct). ``False`` resamples tokens
                independently, which is wrong for token-level data and only
                exists so the difference can be demonstrated.
            method: ``"percentile"`` (default) takes the empirical quantiles of
                the draws. ``"basic"`` (reverse percentile) reflects them through
                the point estimate, ``[2t - q_hi, 2t - q_lo]``, which corrects
                first-order bias when the draws sit systematically off the point
                estimate — worth checking when an interval looks lopsided.
            statistic: ``(scores, labels) -> float`` to bootstrap. ``None`` uses
                AUROC; pass your own for accuracy, F1, or anything else.

        Returns:
            ``{probe_name: (point_estimate, lo, hi)}``.

        Raises:
            ValueError: On an unknown ``method``.
        """
        if method not in ("percentile", "basic"):
            raise ValueError(f"method must be 'percentile' or 'basic', got {method!r}.")
        stat = _resolve_statistic(statistic)
        names = list(self.scores) if name is None else [name]
        rng = np.random.default_rng(seed)
        uniq = np.unique(self.groups)
        # Precompute each group's token positions once; the resample loop then
        # concatenates slices instead of scanning the whole array n_boot times.
        by_group = {g: np.flatnonzero(self.groups == g) for g in uniq} if cluster else {}
        draws: dict[str, list[float]] = {n: [] for n in names}
        n_tokens = len(self.labels)

        for _ in range(n_boot):
            if cluster:
                pick = rng.choice(uniq, size=uniq.size, replace=True)
                idx = np.concatenate([by_group[g] for g in pick])
            else:
                idx = rng.integers(0, n_tokens, n_tokens)
            y = self.labels[idx]
            if y.min() == y.max():  # single-class resample: AUROC undefined
                continue
            for n in names:
                draws[n].append(stat(self.scores[n][idx], y))

        lo_q, hi_q = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0
        out: dict[str, tuple[float, float, float]] = {}
        for n in names:
            vals = np.asarray(draws[n], dtype=float)
            vals = vals[~np.isnan(vals)]
            point = self.statistic(n, statistic)
            if vals.size == 0:
                out[n] = (point, float("nan"), float("nan"))
                continue
            lo, hi = float(np.percentile(vals, lo_q)), float(np.percentile(vals, hi_q))
            if method == "basic":
                lo, hi = 2.0 * point - hi, 2.0 * point - lo
            out[n] = (point, lo, hi)
        return out

    def to_csv(self, path: str, **bootstrap_kwargs: Any) -> None:
        """Write ``probe,auroc,ci_lo,ci_hi,n_tokens,n_groups`` — ready to plot.

        Args:
            path: Destination CSV.
            **bootstrap_kwargs: Forwarded to :meth:`bootstrap` (``n_boot``,
                ``ci``, ``seed``, ``cluster``, ``method``, ``statistic``).
        """
        import csv

        stats = self.bootstrap(**bootstrap_kwargs)
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["probe", "auroc", "ci_lo", "ci_hi", "n_tokens", "n_groups"])
            for n in self.scores:
                point, lo, hi = stats[n]
                writer.writerow(
                    [n, point, lo, hi, len(self), int(np.unique(self.groups).size)]
                )


def _resolve_statistic(fn: Any) -> Any:
    """``fn`` or the default AUROC, as a ``(scores, labels) -> float`` callable."""
    if fn is not None:
        return fn
    from auto_chasm.metrics import auroc as _auroc

    def _default(scores: np.ndarray, labels: np.ndarray) -> float:
        return float(_auroc(scores, labels, np.ones_like(labels, dtype=bool)))

    return _default


def _labels_for(raw_labels: Any, name: str) -> np.ndarray:
    """This probe's target array, whether labels are a dict or a shared list."""
    if isinstance(raw_labels, dict):
        if name in raw_labels:
            return np.asarray(raw_labels[name])
        # A single labeled probe emits ONE shared list under its own name; every
        # attached head (L0..L23 in a sweep) trains on it.
        others = [k for k in raw_labels if k != "lm_head"]
        if len(others) == 1:
            return np.asarray(raw_labels[others[0]])
        raise KeyError(
            f"No targets for probe {name!r}: the dataset labels {sorted(raw_labels)}. "
            "Name the probe after the labeled key, or label exactly one probe so the "
            "array is shared."
        )
    return np.asarray(raw_labels)


def collect_probe_scores(
    model: Any,
    dataset: Any,
    *,
    probe_names: list[str] | None = None,
    batch_size: int = 8,
    max_seq_length: int = 1024,
) -> ProbeScores:
    """Score every token of ``dataset`` with each attached probe, in ONE pass.

    Batches the dataset IN ORDER rather than through ``iterate_batches``, which
    sorts by length and shuffles: that returns only ``(tokens, labels, lengths)``,
    so a row could not be traced back to the sample it came from, and the group id
    a clustered bootstrap needs would be guesswork.

    Args:
        model: A ``Model`` with probes already attached (their current weights are
            used — after ``LayerSweep.run`` those are each layer's best).
        dataset: The dataset to score.
        probe_names: Which probes to score (``None`` = all attached).
        batch_size: Batch size for the forward passes.
        max_seq_length: Truncation length, as in training.

    Returns:
        A :class:`ProbeScores` holding one score per token per probe, the labels,
        and a group id per token for clustered bootstrapping.

    Raises:
        ValueError: If no probes are attached, or a probe is not a single-logit
            token-granularity head.
    """
    from auto_chasm.metrics import to_numpy

    names = list(probe_names if probe_names is not None else model.probes)
    if not names:
        raise ValueError(
            "No probes attached, so there is nothing to score. Attach probes "
            "(or run a LayerSweep) before calling collect_probe_scores."
        )

    samples = list(dataset)
    # A per-sample group keeps the bootstrap honest. Datasets built with ``groups=``
    # carry one; otherwise each sample IS its own cluster -- still correct (one
    # response = one cluster), just less conservative than clustering by prompt.
    if not any(isinstance(s, dict) and "group" in s for s in samples):
        logger.info(
            "Dataset carries no 'group' field; clustering the bootstrap by SAMPLE. "
            "Build it with groups= (e.g. the prompt id) to cluster one level up."
        )

    chunks: dict[str, list[np.ndarray]] = {n: [] for n in names}
    label_chunks: list[np.ndarray] = []
    group_chunks: list[np.ndarray] = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        toks = [list(s["tokens"])[:max_seq_length] for s in batch]
        labs = [np.asarray(_labels_for(s["labels"], names[0]))[:max_seq_length] for s in batch]
        width = max(len(t) for t in toks)
        if width < 2:  # nothing survives the next-token shift
            continue

        tok_arr = np.zeros((len(batch), width), dtype=np.int32)
        lab_arr = np.full((len(batch), width), -100, dtype=np.float64)
        for i, (t, y) in enumerate(zip(toks, labs, strict=True)):
            tok_arr[i, : len(t)] = t
            lab_arr[i, : len(y)] = y

        for n in names:
            model.probes[n].clear_captured()
        model.forward(model.to_tensor(tok_arr)[:, :-1])

        y = lab_arr[:, 1:]                       # aligns with forward(tokens[:, :-1])
        pos = np.arange(1, width)[None, :]
        true_len = np.array([len(t) for t in toks])[:, None]
        keep = (pos < true_len) & (y != -100)
        if not keep.any():
            continue

        for n in names:
            logits = to_numpy(model.probes[n].forward())
            if logits.ndim == 3 and logits.shape[-1] == 1:
                logits = logits[..., 0]
            if logits.shape != y.shape:
                raise ValueError(
                    f"Probe {n!r} produced {logits.shape} for a [B, T] target — per-token "
                    "scoring needs a single-logit, token-granularity head."
                )
            chunks[n].append(logits[keep])
        label_chunks.append(y[keep])
        gids = np.array([s.get("group", start + i) for i, s in enumerate(batch)], dtype=object)
        group_chunks.append(np.repeat(gids, keep.sum(axis=1)))

    if not label_chunks:
        raise ValueError("No labeled tokens found — every position was masked or padding.")
    return ProbeScores(
        scores={n: np.concatenate(chunks[n]) for n in names},
        labels=np.concatenate(label_chunks).astype(np.int64),
        groups=np.concatenate(group_chunks),
        probe_names=names,
    )

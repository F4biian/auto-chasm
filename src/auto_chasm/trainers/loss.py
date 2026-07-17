"""Backend-agnostic joint LM + probe loss.

Provides :class:`JointLoss` — a callable class that combines the language-model
cross-entropy with any number of probe-head losses on a **single** code path
(one graph built through :mod:`auto_chasm.ops`, no per-backend duplication).

**Loss function contract:**

Calling a ``JointLoss`` with ``(model, batch, labels, lengths)`` returns
``(total, ntoks, components)`` where:

- ``total`` — the differentiable scalar loss.
- ``ntoks`` — number of valid (non-padding) tokens.
- ``components`` — ``dict[str, tensor]`` of the per-term scalar losses, keyed by
  **term name**: ``"lm_head"`` for the LM cross-entropy and the probe's own name
  for each probe head (e.g. ``{"lm_head": ..., "digit": ...}``).  Each value is a
  backend scalar tensor (MLX ``mx.array`` or torch ``Tensor``), not a ``float``.

Users who need full control can subclass ``JointLoss`` or write their own
callable with the same ``(model, batch, labels, lengths)`` signature.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from auto_chasm.config import LM_HEAD
from auto_chasm.logger import get_logger
from auto_chasm.outputs import JointOutputs, ProbeOutput, _check_class_indices
from auto_chasm.trainers._loss_ce import (
    _SEQ_CW_MSG,
    _require_resolved_class_weights,
    _seq_target_and_mask,
    _validate_class_weights,
    check_class_weights_applicable,
    weighted_ce,
    weighted_lm_ce,
)
from auto_chasm.trainers._loss_routing import (
    _probe_granularities,
    _required_positional_arity,
    _sequence_level,
)
from auto_chasm.trainers.loss_terms import LossTerm, LossTerms
from auto_chasm.utils import tensor_backend

if TYPE_CHECKING:
    import mlx.core as mx
    import torch

logger = get_logger(__name__)

# Back-compat re-export: the legacy keyword adapter moved to _loss_legacy.py
# (file-length cap); sft.py / trainable.py / tests import it from here.
from auto_chasm.trainers._loss_legacy import (  # noqa: E402
    _joint_loss_from_legacy as _joint_loss_from_legacy,  # explicit re-export
)

LossFn = Callable[[Any, Any, Any, Any], tuple[Any, Any, dict[str, Any]]]

# ``LM_HEAD`` (the reserved term name for the language-model cross-entropy) is
# imported from ``config`` above — one source of truth, so ``ProbeConfig`` can
# reject a probe named ``lm_head`` at construction and the loss uses the same name.

#: Built-in string loss specs mapped to their canonical lower-case name.
_BUILTIN_LOSSES = frozenset({"bce", "ce", "mse", "mae"})


class JointLoss:
    """Backend-agnostic joint LM + probe loss.

    Callable with ``(model, batch, labels, lengths)``; returns
    ``(total_loss, ntoks, components)`` where ``components`` is keyed by term name
    (``"lm_head"`` and each probe name).  One code path serves both MLX and torch.

    The loss over the terms ``{"lm_head"} ∪ {probe names}`` is composed one of two
    ways:

    - **weighted sum** (default / ``weights=``): ``total = Σ w(term) · term_loss``
      where ``w(term) = weights.get(term, 1.0)``.  A term with weight ``<= 0`` is
      skipped (not computed).  ``weights=None`` weights every term ``1.0`` (joint
      LM + all probes); ``weights={"lm_head": 0.0}`` is pure-probe mode; a per-probe
      ``0.0`` drops that probe.
    - **arbitrary composition** (``combine=``): a ``Callable[[LossTerms], LossTerm |
      scalar]`` such as ``lambda L: L.lm_head ** L.p1 - L.p2``.  All terms are
      computed and passed as a :class:`~auto_chasm.trainers.loss_terms.LossTerms`;
      the returned term/scalar is the total.  Mutually exclusive with ``weights``.

    Args:
        weights: Per-term weight ``dict[str, float]``, including the reserved key
            ``"lm_head"``.  A key that is neither ``"lm_head"`` nor an attached probe
            name raises ``ValueError`` (typo protection) at compute time.  Mutually
            exclusive with ``combine``.
        losses: Per-term loss spec ``dict[str, str | Callable]``.  Values are
            ``"bce"``/``"ce"``/``"mse"``/``"mae"`` or a callable.  A probe absent
            from ``losses`` defaults to ``"bce"``.  ``"lm_head"`` defaults to token
            cross-entropy but may be overridden by a callable here.  Custom callables
            support two signatures, detected by arity: ``(probe, target)`` (called
            with a :class:`~auto_chasm.outputs.ProbeOutput` whose ``.mask`` is bound)
            or ``(logits, target, mask)`` (the legacy signature).
        combine: A ``Callable[[LossTerms], LossTerm | scalar]`` for arbitrary
            composition.  Mutually exclusive with ``weights``.
        class_weights: Per-CLASS weights for the built-in ``"ce"`` and ``"bce"``
            probe losses.  For ``"ce"`` a length-``C`` vector scales each class
            (token-level); for ``"bce"`` a length-2 ``[w_neg, w_pos]`` weights the
            0- and 1-class (token- **or** sequence-level).  A flat sequence applies
            to every weightable probe; a ``{probe_name: sequence}`` dict targets
            individual heads.  Pass ``"balanced"`` to have ``Trainer.train`` compute
            inverse-frequency weights from the training data.  ``None`` (default) is
            unweighted.  Setting it on an ``"mse"``/``"mae"``/custom probe, or a
            sequence-level ``"ce"`` probe, raises at compute time.

    Raises:
        ValueError: If both ``weights`` and ``combine`` are given, or if an unknown
            loss spec is passed.
    """

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        losses: dict[str, str | Callable[..., Any]] | None = None,
        combine: Callable[[LossTerms], Any] | None = None,
        class_weights: Sequence[float] | dict[str, Sequence[float]] | str | None = None,
    ) -> None:
        """Initialize the joint loss (see the class docstring for semantics)."""
        if weights is not None and combine is not None:
            raise ValueError("weights and combine are mutually exclusive: pass one or the other.")
        self._weights: dict[str, float] = dict(weights) if weights is not None else {}
        self._combine = combine
        self._losses: dict[str, str | Callable[..., Any]] = dict(losses) if losses else {}
        # Defaults for terms/probes not explicitly listed. The public API lists
        # everything; the legacy adapter (`_joint_loss_from_legacy`) sets these to
        # carry the old global `probe_loss`/`probe_weight` defaults.
        self._default_loss: str | Callable[..., Any] = "bce"
        self._default_weight: float = 1.0

        # Validate every loss spec up front (fail fast on a typo / wrong kind).
        for name, spec in self._losses.items():
            if name == LM_HEAD:
                # The LM head defaults to token cross-entropy; only a callable can
                # override it. A string loss name (bce/ce/mse/mae) is a probe-head
                # spec and is meaningless here — reject it at construction, not at
                # compute time.
                if not callable(spec):
                    raise ValueError(
                        f"losses[{LM_HEAD!r}] must be a callable that overrides the "
                        f"language-model cross-entropy — signature (outputs, target) or "
                        f"(probe, target); got {spec!r}. String loss names apply to "
                        f"probe heads, not the LM head."
                    )
                continue
            if not callable(spec):
                _canonical_loss_name(spec)
        # ``class_weights`` only reaches CE probes; validate against the probe loss
        # specs + the current default loss (the legacy adapter sets a non-"bce"
        # default via ``set_class_weights`` after construction).
        self._class_weights = _validate_class_weights(
            class_weights, self._default_loss, self._losses
        )

    def set_class_weights(self, class_weights: Any) -> None:
        """Set (and validate) per-class weights after construction.

        Used by the legacy adapter (and ``Trainer``) to apply class weights once the
        default probe loss is known — so ``make_joint_loss(probe_loss="ce",
        class_weights=[...])`` validates against ``"ce"``, not the ``"bce"`` default.

        Args:
            class_weights: A sequence, a ``{probe: sequence}`` dict, ``"balanced"``,
                or ``None``.
        """
        self._class_weights = _validate_class_weights(
            class_weights, self._default_loss, self._losses
        )

    @property
    def _probe_weights(self) -> dict[str, float]:
        """The live per-term weight dict (Phase 3b shim for ``trainer.py``).

        ``Trainer._apply_probe_weights`` still calls ``loss._probe_weights.update(...)``
        with ``TrainingConfig.probe_weights``; those probe names are valid term keys in
        the new ``weights`` dict, so returning the live ``_weights`` mapping keeps that
        call working until Phase 3b-2 rewrites the trainer onto ``weights=``.

        Returns:
            The mutable ``{term: weight}`` mapping backing this loss.
        """
        return self._weights

    # ------------------------------------------------------------------
    # Per-term configuration helpers
    # ------------------------------------------------------------------

    def _weight_for(self, term: str) -> float:
        """Return the weight for ``term`` (``self._default_weight`` by default)."""
        return float(self._weights.get(term, self._default_weight))

    def _class_weights_for(self, probe_name: str) -> Any:
        """Return the per-class weights this probe should use, or ``None``."""
        spec = self._class_weights
        if spec is None:
            return None
        if isinstance(spec, dict):
            return spec.get(probe_name)
        return spec

    def _loss_spec_for(self, probe_name: str) -> str | Callable[..., Any]:
        """Return the loss spec for ``probe_name`` (``self._default_loss`` by default)."""
        return self._losses.get(probe_name, self._default_loss)

    def _validate_weight_keys(self, probe_names: Sequence[str]) -> None:
        """Raise if a ``weights`` (or dict ``class_weights``) key is not a valid term.

        Args:
            probe_names: The names of the model's attached probes.

        Raises:
            ValueError: On an unknown ``weights`` or ``class_weights`` key (typo
                protection — a typo would otherwise silently skip that weighting).
        """
        allowed = {LM_HEAD, *probe_names}
        unknown = [k for k in self._weights if k not in allowed]
        if unknown:
            known = ", ".join(sorted(allowed)) or "(none)"
            raise ValueError(
                f"Unknown weights key(s) {unknown}: expected 'lm_head' or a probe "
                f"name. Attached terms are: {known}."
            )
        if isinstance(self._class_weights, dict):
            cw_unknown = [k for k in self._class_weights if k not in probe_names]
            if cw_unknown:
                probes = ", ".join(sorted(probe_names)) or "(none)"
                raise ValueError(
                    f"Unknown class_weights key(s) {cw_unknown}: expected a probe name "
                    f"(class weights apply per probe). Probes are: {probes}."
                )

    # ------------------------------------------------------------------
    # The single backend-agnostic compute path
    # ------------------------------------------------------------------

    def __call__(
        self,
        model: Any,
        batch: Any,
        labels: Any,
        lengths: Any,
    ) -> tuple[mx.array | torch.Tensor, mx.array | torch.Tensor, dict[str, Any]]:
        """Compute the joint LM + probe loss (single backend-agnostic path).

        Args:
            model: The model to evaluate.
            batch: Tokenized input batch of shape ``[B, T]``.
            labels: Probe label tensor (or ``{probe: array}`` dict) aligned with
                ``batch``.
            lengths: Per-sequence token ranges for the probe target region.

        Returns:
            Tuple of ``(total_loss, ntoks, components)`` where ``components`` is a
            dict of the per-term scalar losses keyed by term name.
        """
        from auto_chasm import ops

        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Single-token edge: no next-token targets, so NOTHING (LM or probe) can be
        # supervised. Return a finite 0 with EMPTY components -- emitting {lm_head: 0}
        # here fabricated a term a pure-probe run never has and mismatched the keys of
        # a normal batch. ntoks is 0 so token-weighted eval correctly ignores it.
        if int(targets.shape[1]) == 0:
            z = ops.zeros_like(_scalar_from(batch))
            return z, z, {}

        mask = _length_mask(targets, lengths)
        lm_logits, raw_probe_outputs = _call_model(model, inputs, mask)
        probe_dict = _normalize_probe_outputs(raw_probe_outputs)

        self._validate_weight_keys(list(probe_dict))
        if LM_HEAD in probe_dict:
            raise ValueError("'lm_head' is reserved for the language-model head; rename the probe.")

        o = JointOutputs(lm_logits, probe_dict, targets, lengths)
        n_time = int(targets.shape[1])
        # Route each probe by its declared granularity (reliable), not tensor shape.
        granularities = _probe_granularities(model)
        # A labels dict whose keys match NO probe silently skips every probe term
        # (loss becomes just the LM term, or a constant 0 in pure-probe mode).
        # That is almost always a probe-name typo, so surface it. The reserved
        # "lm_head" key is the per-token LM WEIGHT channel, not a probe target —
        # a pure-SFT-with-weights dataset must not trip this warning.
        probe_label_keys = set(labels) - {LM_HEAD} if isinstance(labels, dict) else set()
        if isinstance(labels, dict) and probe_label_keys and probe_label_keys.isdisjoint(o.probes):
            logger.warning(
                "labels dict keys %s match none of the attached probes %s; every "
                "probe term is skipped. Check for a probe-name typo.",
                sorted(probe_label_keys),
                sorted(o.probes),
            )

        terms: dict[str, LossTerm] = {}

        # LM term.  Skip only in weighted-sum mode with weight <= 0 (combine may
        # still reference it, so it is always computed when combine is given).
        # The reserved labels["lm_head"] channel carries PER-TOKEN weights
        # (1=train, 0=mask, negative=unlearn; -100=unspecified->1) — see
        # ``weighted_lm_ce``. With no channel this is the plain masked-mean CE.
        lm_weights = labels.get(LM_HEAD) if isinstance(labels, dict) else None
        if self._combine is not None or self._weight_for(LM_HEAD) > 0:
            lm_spec = self._losses.get(LM_HEAD)
            if lm_spec is not None and lm_weights is not None:
                raise ValueError(
                    "The data carries a per-token labels['lm_head'] weight channel, "
                    "but losses['lm_head'] overrides the LM loss with a custom "
                    "callable — the weights would be silently ignored. Apply them "
                    "inside your callable, or drop one of the two."
                )
            if lm_spec is None:
                if lm_weights is None:
                    terms[LM_HEAD] = LossTerm(o.lm_ce)
                else:
                    terms[LM_HEAD] = LossTerm(
                        weighted_lm_ce(o.lm_logits, o.targets, o.mask, lm_weights[:, 1:])
                    )
            else:
                terms[LM_HEAD] = LossTerm(self._lm_term(lm_spec, o))

        for probe_name, probe in o.probes.items():
            if probe.logits is None:
                continue
            if self._combine is None and self._weight_for(probe_name) <= 0:
                continue
            target = _labels_for(labels, probe_name)
            if target is None:
                continue  # per-probe labels given but none for this head this batch
            shifted = target[:, 1:]
            value = self._probe_term(
                probe, shifted, o, n_time, probe_name, granularities.get(probe_name)
            )
            terms[probe_name] = LossTerm(value)

        total = self._reduce(terms, _scalar_from(batch))
        return total, o.ntoks, {name: term.value for name, term in terms.items()}

    def _reduce(self, terms: dict[str, LossTerm], zero: Any) -> Any:
        """Compose the per-term losses into the differentiable scalar total.

        Args:
            terms: The computed ``{name: LossTerm}`` terms.
            zero: A backend scalar zero returned when no term contributes (every
                term dropped/absent), so the total stays a 0-d backend tensor.

        Returns:
            The scalar total (a backend tensor).
        """
        if self._combine is not None:
            result = self._combine(LossTerms(terms))
            return result.value if isinstance(result, LossTerm) else result
        total: Any = None
        for name, term in terms.items():
            contrib = self._weight_for(name) * term.value
            total = contrib if total is None else total + contrib
        return zero if total is None else total

    def _probe_term(
        self,
        probe: ProbeOutput,
        shifted: Any,
        o: JointOutputs,
        n_time: int,
        probe_name: str,
        granularity: str | None = None,
    ) -> Any:
        """Compute a single probe's scalar loss via the reusable methods.

        Args:
            probe: The probe output (``.logits`` and, once bound below, ``.mask``).
            shifted: The probe's shifted labels ``[B, T-1]``.
            o: The :class:`~auto_chasm.outputs.JointOutputs` for this batch.
            n_time: Number of target tokens (``targets.shape[1]``).
            probe_name: The probe's name (for per-probe class weights / loss spec).
            granularity: The probe's declared granularity, used to route per-token
                vs sequence-level losses without a shape guess (``None`` = unknown).

        Returns:
            The probe's scalar loss.
        """
        spec = self._loss_spec_for(probe_name)
        class_weights = self._class_weights_for(probe_name)
        check_class_weights_applicable(probe_name, spec, class_weights)
        probe_mask = _combine_masks(o.mask, shifted)
        probe.mask = probe_mask  # bind so a 2-arg custom loss can ``probe.reduce(...)``

        if _sequence_level(probe.logits, n_time, granularity):
            return self._seq_term(probe, shifted, probe_mask, spec, class_weights)
        if callable(spec):
            # Pass the RAW (int) shifted labels: a 2-param (probe, target) loss can
            # then call probe.ce(target) safely; the legacy 3-param path float-casts
            # internally. Passing a float target to probe.ce triggers an UNCATCHABLE
            # C++ abort (MLX gather on float indices), so this must stay int here.
            return self._call_custom(spec, probe, shifted, probe_mask)
        name = _canonical_loss_name(spec)
        if name == "bce":
            if class_weights is not None:
                _require_resolved_class_weights(class_weights)
                return probe.bce(_as_float(shifted), mask=o.mask, weights=class_weights)
            return probe.bce(_as_float(shifted), mask=o.mask)
        if name == "mse":
            return _scalar_probe(probe).mse(_as_float(shifted), mask=o.mask)
        if name == "mae":
            return _scalar_probe(probe).mae(_as_float(shifted), mask=o.mask)
        # name == "ce"
        _check_class_indices(shifted, probe.n_classes)
        if class_weights is not None:
            _require_resolved_class_weights(class_weights)
            # Float-cast before comparing to the ignore sentinel: a uint8 label dtype
            # would wrap `-100` and mis-mark ignores (mirrors the `_combine_masks` fix).
            label_valid = _as_float(shifted) != -100.0
            return weighted_ce(probe.logits, shifted, label_valid, probe_mask, class_weights)
        return probe.ce(shifted, mask=o.mask)

    def _seq_term(
        self,
        probe: ProbeOutput,
        shifted: Any,
        probe_mask: Any,
        spec: str | Callable[..., Any],
        class_weights: Any,
    ) -> Any:
        """Compute a sequence-level (response/sentence) probe loss.

        The per-token labels are pooled to one float target per row; ``bce``/``mse``/
        ``mae`` use that float target, ``ce`` rounds it to an int class index, and a
        custom callable is dispatched by arity.

        Args:
            probe: The pooled probe output ``[B, out]`` or ``[B]``.
            shifted: The probe's shifted labels ``[B, T-1]``.
            probe_mask: The per-token validity mask ``[B, T-1]``.
            spec: The resolved loss spec.
            class_weights: Per-class weights, or ``None``.

        Returns:
            The probe's scalar loss.
        """
        seq_target, seq_valid = _seq_target_and_mask(shifted, probe_mask)
        flat = _squeeze_seq_head(probe.logits, spec)
        pooled = ProbeOutput(logits=flat, mask=seq_valid)
        if callable(spec):
            return self._call_custom(spec, pooled, seq_target, seq_valid)
        name = _canonical_loss_name(spec)
        if name == "bce":
            if class_weights is not None:
                _require_resolved_class_weights(class_weights)
                return pooled.bce(seq_target, mask=seq_valid, weights=class_weights)
            return pooled.bce(seq_target, mask=seq_valid)
        if name == "mse":
            return pooled.mse(seq_target, mask=seq_valid)
        if name == "mae":
            return pooled.mae(seq_target, mask=seq_valid)
        # name == "ce"
        if class_weights is not None:
            raise NotImplementedError(_SEQ_CW_MSG)
        return pooled.ce(_round_int(seq_target), mask=seq_valid)

    def _lm_term(self, spec: str | Callable[..., Any], o: JointOutputs) -> Any:
        """Compute the LM term when ``"lm_head"`` is overridden by a callable.

        A 2-param ``(outputs, targets)`` callable receives the whole
        :class:`~auto_chasm.outputs.JointOutputs` and the shifted target tokens; a
        3-param ``(logits, targets, mask)`` callable receives the raw LM logits, the
        targets, and the length mask.  A string spec is rejected (the LM head has no
        class structure for ``bce``/``ce``/``mse``/``mae``).

        Args:
            spec: The ``"lm_head"`` loss spec.
            o: The :class:`~auto_chasm.outputs.JointOutputs` for this batch.

        Returns:
            The LM term's scalar loss.

        Raises:
            ValueError: If ``spec`` is a built-in string rather than a callable.
        """
        if not callable(spec):
            raise ValueError(
                "losses['lm_head'] must be a callable; the built-in string losses "
                "('bce'/'ce'/'mse'/'mae') apply to probe heads, not the LM head."
            )
        if _required_positional_arity(spec) <= 2:
            return spec(o, o.targets)
        return spec(o.lm_logits, o.targets, o.mask)

    def _call_custom(
        self,
        fn: Callable[..., Any],
        probe: Any,
        target: Any,
        probe_mask: Any = None,
    ) -> Any:
        """Dispatch a custom-loss callable by its arity.

        Supports both signatures: a 2-param ``(probe, target)`` fn is called with a
        :class:`~auto_chasm.outputs.ProbeOutput` whose ``.mask`` is bound (so it can
        use ``probe.reduce``/``probe.bce``/etc.), and a 3-param ``(logits, target,
        mask)`` fn is called with ``(probe.logits, target, probe_mask)`` (legacy).

        Args:
            fn: The user callable.
            probe: A :class:`~auto_chasm.outputs.ProbeOutput` (for a 2-param fn) or a
                ``{name: ProbeOutput}`` dict (LM-head override).
            target: The (shifted / pooled) target for this term.
            probe_mask: The validity mask passed to a 3-param fn.

        Returns:
            The scalar the callable returns.
        """
        if _required_positional_arity(fn) <= 2:
            return fn(probe, target)
        logits = probe.logits if isinstance(probe, ProbeOutput) else probe
        # Legacy 3-param losses expect the float-cast target (the old ``p_targets``).
        return fn(logits, _as_float(target), probe_mask)


# --------------------------------------------------------------------------- #
# Free helpers (module-level so subclasses and custom losses can reuse them).  #
# --------------------------------------------------------------------------- #


def _canonical_loss_name(spec: str) -> str:
    """Return the canonical lower-case name of a built-in string loss spec.

    Args:
        spec: One of ``"bce"``/``"ce"``/``"mse"``/``"mae"`` (case-insensitive).

    Returns:
        The lower-cased name.

    Raises:
        ValueError: If ``spec`` is not a known built-in loss.
    """
    name = spec.lower()
    if name in _BUILTIN_LOSSES:
        return name
    raise ValueError(f"Unknown loss={spec!r}. Expected 'bce', 'ce', 'mse', 'mae', or a callable.")


def _scalar_from(batch: Any) -> Any:
    """Return a backend scalar derived from ``batch`` (for a device-correct zero)."""
    from auto_chasm import ops

    flat = batch.reshape(-1)
    return ops.sum(_as_float(flat[:1])) * 0.0


def _length_mask(targets: Any, lengths: Any) -> Any:
    """Boolean ``[B, T-1]`` mask for the valid ``lengths[:,0] <= step < lengths[:,1]``.

    Args:
        targets: The shifted targets ``[B, T-1]`` (for shape and backend).
        lengths: Per-sequence ``[B, 2]`` token ranges.

    Returns:
        A boolean mask ``[B, T-1]``.
    """
    from auto_chasm import ops

    steps = ops.arange(int(targets.shape[1]), like=targets, start=1)
    lo = steps >= lengths[:, 0:1]
    hi = steps < lengths[:, 1:]
    if tensor_backend(targets) == "torch":
        return lo & hi
    import mlx.core as mx

    return mx.logical_and(lo, hi)


def _combine_masks(length_mask: Any, shifted: Any) -> Any:
    """Combine the length mask with a ``shifted != -100`` label-valid mask.

    Args:
        length_mask: The length-window mask ``[B, T-1]``.
        shifted: The probe's shifted labels ``[B, T-1]``.

    Returns:
        The boolean intersection mask ``[B, T-1]``.
    """
    # Cast to float before ``!= -100`` so an unsigned label dtype (e.g. uint8) does
    # not overflow converting the sentinel (matches the pre-3b float-cast behavior).
    label_valid = _as_float(shifted) != -100.0
    if tensor_backend(length_mask) == "torch":
        return length_mask & label_valid
    import mlx.core as mx

    return mx.logical_and(length_mask, label_valid)


def _as_float(x: Any) -> Any:
    """Cast ``x`` to float32 on its own backend."""
    if tensor_backend(x) == "torch":
        import torch

        return x.to(torch.float32)
    import mlx.core as mx

    return x.astype(mx.float32)


def _round_int(x: Any) -> Any:
    """Round a float pooled target to the nearest integer class index (any backend)."""
    if tensor_backend(x) == "torch":
        return x.round().long()
    import mlx.core as mx

    return mx.round(x).astype(mx.int32)


def _scalar_probe(probe: ProbeOutput) -> ProbeOutput:
    """Return a probe whose ``[B, T, 1]`` scalar head is squeezed to ``[B, T]``.

    A token-level ``out_features=1`` regression head outputs ``[B, T, 1]`` while the
    targets are ``[B, T]``; without collapsing the trailing ``1`` the ``mse``/``mae``
    elementwise ops broadcast instead of aligning.  The bound ``.mask`` is preserved
    so a downstream reduce still sees the validity mask.

    Args:
        probe: The probe output (``[B, T, 1]`` for a scalar head).

    Returns:
        A :class:`~auto_chasm.outputs.ProbeOutput` with the trailing singleton
        removed when it is a scalar head, else ``probe`` unchanged.
    """
    logits = probe.logits
    assert logits is not None
    if logits.ndim == 3 and int(logits.shape[-1]) == 1:
        return ProbeOutput(logits=logits[..., 0], mask=probe.mask)
    return probe


def _squeeze_seq_head(probe_logits: Any, spec: str | Callable[..., Any]) -> Any:
    """Collapse a ``[B, 1]`` sequence head to ``[B]`` for non-CE losses.

    A pooled scalar head is ``[B, 1]``; ``bce``/``mse``/``mae`` and a custom callable
    want ``[B]`` (matching the ``[B]`` pooled target), while a multi-class ``ce`` head
    is ``[B, C]`` and must be left untouched.

    Args:
        probe_logits: The pooled probe output.
        spec: The loss spec (a callable, or a built-in name).

    Returns:
        The (possibly squeezed) logits.
    """
    is_ce = (not callable(spec)) and _canonical_loss_name(spec) == "ce"
    if probe_logits.ndim == 2 and not is_ce and int(probe_logits.shape[-1]) == 1:
        return probe_logits[..., 0]
    return probe_logits


def _labels_for(labels: Any, probe_name: str) -> Any:
    """Return the label array a probe should train on.

    ``labels`` is either a single per-token array shared by every probe, or a
    ``{probe_name: array}`` dict of independent per-probe targets.  For the dict
    form, a probe absent from the dict returns ``None`` (it is skipped this batch —
    it simply has no targets here).

    Args:
        labels: A single array, or a ``{probe_name: array}`` dict.
        probe_name: The probe whose targets are requested.

    Returns:
        The probe's label array, or ``None`` when the dict omits it.
    """
    if isinstance(labels, dict):
        return labels.get(probe_name)
    return labels


def _normalize_probe_outputs(raw: Any) -> dict[str, Any]:
    """Normalize model probe outputs to a ``{name: logits}`` dict.

    Handles legacy single-tensor and ``None`` returns for backward compatibility
    with tests and older model wrappers.

    Args:
        raw: The second element of the model return tuple, which may be a dict, a
            single tensor, or ``None``.

    Returns:
        A ``{probe_name: logits}`` dict.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        return {"probe": raw}
    return raw


def _call_model(model: Any, inputs: Any, mask: Any) -> Any:
    """Call the trainable model, passing a pooling mask when it is accepted.

    The trainer wrappers (``_TrainableModel``, ``_TorchProbeWrapper``) accept a
    ``mask`` keyword so ``granularity="response"`` pooling ignores padding.
    Legacy/custom callables that take only ``inputs`` are still supported — their
    signature is inspected once so a genuine ``TypeError`` raised inside the forward
    pass is never swallowed.

    Args:
        model: The trainable model wrapper or a callable.
        inputs: Tokenized input batch ``[B, T-1]``.
        mask: Valid-region mask ``[B, T-1]`` or ``None``.

    Returns:
        The ``(lm_logits, probe_outputs)`` tuple from the model.
    """
    try:
        sig = inspect.signature(model)
    except (TypeError, ValueError):
        return model(inputs)

    params = sig.parameters
    accepts_mask = "mask" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_mask:
        return model(inputs, mask=mask)
    return model(inputs)

"""Data-driven supervision-task inference for probe heads.

A probe's task type (binary / multi-class classification, or ordinal
regression) fixes **four** things that must stay mutually consistent: the head's
output width (``out_features``), the probe loss (``bce``/``ce``/``mse``), the
evaluation metrics factory (:func:`classification_metrics` vs
:func:`regression_metrics`), and how the integer/float labels are read.  Hand-
coordinating those four across a training script is exactly where they drift
apart (a multi-class head trained with a single-logit ``bce``, a regressor scored
with classification metrics, ...).

:class:`Task` bundles those decisions behind one object, either **inferred from
the data** (:meth:`~auto_chasm.Dataset.infer_task`) or **declared explicitly**
(:meth:`Task.regression`, :meth:`Task.classification`, ...).  From a ``Task`` the
consistent head width, loss, and metrics all follow — they can no longer
disagree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from auto_chasm.config import LM_HEAD
from auto_chasm.metrics import classification_metrics, regression_metrics

if TYPE_CHECKING:
    from auto_chasm.trainers.loss import JointLoss

#: The three supervision kinds a probe head can carry.
TaskKind = Literal["binary", "multiclass", "regression"]

#: Accepted values for the ``kind`` selector on ``infer_task`` — the three
#: concrete kinds, ``"classification"`` (binary-or-multiclass, chosen by the
#: label range), and ``"auto"`` (chosen by the label dtype AND range).
TaskSelector = Literal["auto", "classification", "binary", "multiclass", "regression"]


@dataclass(frozen=True)
class Task:
    """The supervision task for one probe head: what head width, loss, and metrics agree.

    Attributes:
        kind: ``"binary"``, ``"multiclass"``, or ``"regression"``.
        num_classes: The class count (classification) or the ordinal-bin count
            used to discretize a regressor's predictions for the comparable
            accuracy metrics (regression).  ``None`` only for a *continuous*
            regression task with no natural bin count.
    """

    kind: TaskKind
    num_classes: int | None = None

    def __post_init__(self) -> None:
        """Validate the kind/num_classes pairing."""
        if self.kind in ("binary", "multiclass"):
            if self.num_classes is None or self.num_classes < 2:
                raise ValueError(
                    f"A {self.kind!r} task needs num_classes >= 2, got {self.num_classes!r}."
                )
        elif self.kind == "regression":
            if self.num_classes is not None and self.num_classes < 2:
                raise ValueError(
                    f"A regression task's num_classes (ordinal bins) must be >= 2 or None, "
                    f"got {self.num_classes!r}."
                )
        else:  # pragma: no cover - dataclass typing already constrains kind
            raise ValueError(f"Unknown task kind {self.kind!r}.")

    # ------------------------------------------------------------------ #
    # Explicit constructors (when you want to declare, not infer).        #
    # ------------------------------------------------------------------ #

    @classmethod
    def binary(cls) -> Task:
        """A 2-class task on a single-logit head (``bce``)."""
        return cls("binary", 2)

    @classmethod
    def multiclass(cls, num_classes: int) -> Task:
        """A ``num_classes``-way task on a ``num_classes``-logit head (``ce``)."""
        return cls("multiclass", num_classes)

    @classmethod
    def classification(cls, num_classes: int) -> Task:
        """Binary when ``num_classes == 2``, else multi-class."""
        return cls.binary() if num_classes == 2 else cls.multiclass(num_classes)

    @classmethod
    def regression(cls, num_classes: int | None = None) -> Task:
        """A scalar-head (``out_features=1``) ordinal regression task (``mse``).

        Args:
            num_classes: The ordinal-bin count used to discretize predictions for
                the comparable accuracy metrics (e.g. ``6`` for CEFR A1..C2).
                ``None`` for a continuous target with no natural bins (then
                :meth:`build_metrics` cannot report discretized accuracy).
        """
        return cls("regression", num_classes)

    # ------------------------------------------------------------------ #
    # The consistent modeling knobs that follow from the task.            #
    # ------------------------------------------------------------------ #

    @property
    def out_features(self) -> int:
        """The head's output width: ``num_classes`` for multi-class, else ``1``."""
        if self.kind == "multiclass":
            assert self.num_classes is not None  # guaranteed by __post_init__
            return self.num_classes
        return 1

    @property
    def loss_spec(self) -> str:
        """The ``JointLoss`` loss name for this task: ``bce`` / ``ce`` / ``mse``."""
        return {"binary": "bce", "multiclass": "ce", "regression": "mse"}[self.kind]

    @property
    def is_classification(self) -> bool:
        """``True`` for a binary or multi-class task, ``False`` for regression."""
        return self.kind in ("binary", "multiclass")

    def build_metrics(self, *, ordinal_tol: int = 1) -> Callable[..., dict[str, float]]:
        """Build the ``eval_metrics_fn`` matching this task.

        Classification → :func:`classification_metrics`; regression →
        :func:`regression_metrics` (which also reports the discretized accuracy so
        a regressor is directly comparable to a classifier).

        Args:
            ordinal_tol: Tolerance for the adjacent (``_adj``) accuracy metric.

        Returns:
            An ``eval_metrics_fn`` with the trainer's metric signature.

        Raises:
            ValueError: For a continuous regression task (``num_classes is None``),
                which has no bins to discretize predictions into.
        """
        if self.kind == "regression":
            if self.num_classes is None:
                raise ValueError(
                    "regression metrics need num_classes (ordinal bins) to discretize "
                    "predictions; declare Task.regression(num_classes=...) or infer from "
                    "integer-valued labels."
                )
            return regression_metrics(self.num_classes, ordinal_tol=ordinal_tol)
        return classification_metrics(self.num_classes, ordinal_tol=ordinal_tol)

    def build_loss(
        self,
        probe_name: str,
        *,
        lm_weight: float = 0.0,
        class_weights: Any = None,
    ) -> JointLoss:
        """Build a single-probe ``JointLoss`` matching this task.

        Args:
            probe_name: The head this loss supervises.  Must not be the reserved
                ``"lm_head"`` name.
            lm_weight: Weight on the language-model cross-entropy term.  Defaults
                to ``0.0`` (pure-probe training — the LM head is not computed).
            class_weights: Per-class weights, valid only for a multi-class task
                (``ce``).  A list, or ``"balanced"`` to resolve from the training
                data at ``Trainer.train`` time.

        Returns:
            A ``JointLoss`` with ``weights={"lm_head": lm_weight}`` and
            ``losses={probe_name: <this task's loss>}``.

        Raises:
            ValueError: If ``probe_name`` is ``"lm_head"``, or ``class_weights`` is
                given for a non-multi-class task (only CE weights per class).
        """
        from auto_chasm.trainers.loss import JointLoss

        if probe_name == LM_HEAD:
            raise ValueError(f"probe_name {LM_HEAD!r} is reserved for the language-model head.")
        if class_weights is not None and self.kind != "multiclass":
            raise ValueError(
                f"class_weights apply only to a multi-class (ce) task; this task is "
                f"{self.kind!r} ({self.loss_spec!r})."
            )
        return JointLoss(
            weights={LM_HEAD: lm_weight},
            losses={probe_name: self.loss_spec},
            class_weights=class_weights,
        )

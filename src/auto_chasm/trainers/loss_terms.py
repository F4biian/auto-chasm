"""Composable loss-term objects for the OOP ``JointLoss`` redesign.

A :class:`LossTerm` wraps a single backend scalar loss (an ``mlx.core.array``
or a ``torch.Tensor``) and overloads the Python arithmetic operators so that a
user-supplied ``combine=`` lambda reads like ordinary math::

    JointLoss(combine=lambda L: L.lm_head ** L.p1 - L.p2)

Here ``L`` is a :class:`LossTerms` namespace and every ``L.<name>`` is a
:class:`LossTerm`.  Because MLX and PyTorch scalars already implement
``+ - * / ** -`` natively, the overloads just forward to the wrapped ``.value``'s
own operators — no backend dispatch is needed, so the same lambda runs
identically on both backends.

The wrappers deliberately do **not** implement ``__float__`` or ``__bool__``:
those would force an evaluation of a traced graph tensor and crash inside
``mlx.core.value_and_grad`` / ``mx.compile``.
"""

from __future__ import annotations

from typing import Any


class LossTerm:
    """A single backend scalar loss that composes via Python operators.

    Wraps one ``mlx.core.array`` or ``torch.Tensor`` scalar and overloads the
    arithmetic operators so terms combine like math while staying
    backend-agnostic.  Every operator returns a **new** ``LossTerm`` whose
    ``.value`` is the corresponding operation on the underlying tensors, so a
    chain such as ``a ** b - c`` builds the differentiable graph without ever
    evaluating it.

    Operands may be another ``LossTerm`` (its ``.value`` is used) or a plain
    Python number.  The reflected operators (``__rsub__`` etc.) preserve operand
    order so non-commutative operations (``-``, ``/``, ``**``) are correct.

    ``__float__`` / ``__bool__`` are intentionally **not** defined: forcing a
    scalar out of a traced tensor raises inside ``value_and_grad`` / ``compile``.

    Attributes:
        value: The wrapped backend scalar tensor.
    """

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        """Wrap a backend scalar loss tensor.

        Args:
            value: An ``mlx.core.array`` or ``torch.Tensor`` scalar loss.
        """
        self._value = value

    @property
    def value(self) -> Any:
        """The wrapped backend scalar tensor."""
        return self._value

    @staticmethod
    def _operand(other: Any) -> Any:
        """Unwrap ``other`` to its raw operand (a tensor or a number).

        Args:
            other: A ``LossTerm`` or a Python number.

        Returns:
            ``other.value`` when ``other`` is a ``LossTerm``, else ``other``.
        """
        return other.value if isinstance(other, LossTerm) else other

    def __add__(self, other: Any) -> LossTerm:
        """Return ``LossTerm(self.value + other)``."""
        return LossTerm(self._value + self._operand(other))

    def __radd__(self, other: Any) -> LossTerm:
        """Return ``LossTerm(other + self.value)`` for a left number operand."""
        return LossTerm(self._operand(other) + self._value)

    def __sub__(self, other: Any) -> LossTerm:
        """Return ``LossTerm(self.value - other)``."""
        return LossTerm(self._value - self._operand(other))

    def __rsub__(self, other: Any) -> LossTerm:
        """Return ``LossTerm(other - self.value)`` for a left number operand."""
        return LossTerm(self._operand(other) - self._value)

    def __mul__(self, other: Any) -> LossTerm:
        """Return ``LossTerm(self.value * other)``."""
        return LossTerm(self._value * self._operand(other))

    def __rmul__(self, other: Any) -> LossTerm:
        """Return ``LossTerm(other * self.value)`` for a left number operand."""
        return LossTerm(self._operand(other) * self._value)

    def __truediv__(self, other: Any) -> LossTerm:
        """Return ``LossTerm(self.value / other)``."""
        return LossTerm(self._value / self._operand(other))

    def __rtruediv__(self, other: Any) -> LossTerm:
        """Return ``LossTerm(other / self.value)`` for a left number operand."""
        return LossTerm(self._operand(other) / self._value)

    def __pow__(self, other: Any) -> LossTerm:
        """Return ``LossTerm(self.value ** other)``."""
        return LossTerm(self._value ** self._operand(other))

    def __rpow__(self, other: Any) -> LossTerm:
        """Return ``LossTerm(other ** self.value)`` for a left number operand."""
        return LossTerm(self._operand(other) ** self._value)

    def __neg__(self) -> LossTerm:
        """Return ``LossTerm(-self.value)``."""
        return LossTerm(-self._value)

    def __repr__(self) -> str:
        """Return a debug representation naming the wrapped value."""
        return f"LossTerm({self._value!r})"


class LossTerms:
    """A named namespace of :class:`LossTerm` objects for a ``combine=`` lambda.

    Wraps ``dict[str, LossTerm]`` and exposes each term both as an attribute
    (``L.lm_head``) and by key (``L["lm_head"]``).  An unknown name raises with a
    message listing the available term names, so a typo in a user's ``combine=``
    lambda is reported clearly instead of surfacing as an opaque ``KeyError``.
    """

    __slots__ = ("_terms",)

    _terms: dict[str, LossTerm]

    def __init__(self, terms: dict[str, LossTerm]) -> None:
        """Wrap a mapping of term name to :class:`LossTerm`.

        Args:
            terms: Mapping from loss-term name to its :class:`LossTerm`.
        """
        object.__setattr__(self, "_terms", dict(terms))

    def _get(self, name: str) -> LossTerm:
        """Return the term named ``name`` or build the not-found message.

        Args:
            name: The requested term name.

        Returns:
            The :class:`LossTerm` registered under ``name``.

        Raises:
            KeyError: If ``name`` is not a registered term; the message lists
                the available term names.
        """
        try:
            return self._terms[name]
        except KeyError:
            available = ", ".join(self._terms) or "(none)"
            raise KeyError(f"unknown loss term {name!r}; available: {available}") from None

    def __getitem__(self, name: str) -> LossTerm:
        """Return the term named ``name`` (subscript access).

        Args:
            name: The requested term name.

        Returns:
            The :class:`LossTerm` registered under ``name``.

        Raises:
            KeyError: If ``name`` is not a registered term.
        """
        return self._get(name)

    def __getattr__(self, name: str) -> LossTerm:
        """Return the term named ``name`` (attribute access).

        The ``_terms`` slot and dunder names (``__copy__``/``__deepcopy__``/
        ``__reduce_ex__`` etc., probed by ``copy``/``pickle``) are rejected up front
        with a plain ``AttributeError`` so an instance whose slot is unset — created
        via ``copy``/``deepcopy``/``pickle``/``__new__`` — cannot recurse (accessing
        ``self._terms`` on such an instance would otherwise re-enter ``__getattr__``).

        Args:
            name: The requested term name.

        Returns:
            The :class:`LossTerm` registered under ``name``.

        Raises:
            AttributeError: If ``name`` is a reserved/dunder name, if ``_terms`` is
                unset, or if ``name`` is not a registered term (the message then
                lists the available term names).
        """
        if name == "_terms" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        terms: dict[str, LossTerm] = object.__getattribute__(self, "_terms")
        try:
            return terms[name]
        except KeyError:
            available = ", ".join(terms) or "(none)"
            raise AttributeError(f"unknown loss term {name!r}; available: {available}") from None

    def __contains__(self, name: object) -> bool:
        """Return whether ``name`` is a registered term."""
        return name in self._terms

    def __iter__(self) -> Any:
        """Iterate over the registered term names."""
        return iter(self._terms)

    def __repr__(self) -> str:
        """Return a debug representation listing the registered term names."""
        names = ", ".join(self._terms) or "(none)"
        return f"LossTerms({names})"

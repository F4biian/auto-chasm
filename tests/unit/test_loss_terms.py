"""Oracle tests for ``LossTerm`` / ``LossTerms`` operator algebra.

Every operator (including the reversed forms with a plain-number operand) is
checked on **both** MLX and PyTorch against a hand-computed, numpy-verified
value, so the OOP ``JointLoss`` ``combine=`` lambda composes identically on both
backends.
"""

from __future__ import annotations

import pytest

from auto_chasm.metrics import to_numpy
from auto_chasm.trainers.loss_terms import LossTerm, LossTerms


def _mlx_scalar(value: float):
    """Return an MLX scalar array for ``value``."""
    import mlx.core as mx

    return mx.array(value)


def _torch_scalar(value: float):
    """Return a torch scalar tensor for ``value``."""
    import torch

    return torch.tensor(value)


BUILDERS = [
    pytest.param(_mlx_scalar, id="mlx"),
    pytest.param(_torch_scalar, id="torch"),
]


def _num(term: LossTerm) -> float:
    """Convert a ``LossTerm``'s wrapped scalar to a Python float via to_numpy."""
    return float(to_numpy(term.value).reshape(()))


@pytest.mark.parametrize("scalar", BUILDERS)
def test_exotic_combination_pow_sub(scalar):
    """``L.lm_head ** L.p1 - L.p2`` equals ``2**3 - 1 == 7`` on both backends."""
    pytest.importorskip("torch")
    terms = LossTerms(
        {
            "lm_head": LossTerm(scalar(2.0)),
            "p1": LossTerm(scalar(3.0)),
            "p2": LossTerm(scalar(1.0)),
        }
    )
    result = terms.lm_head**terms.p1 - terms.p2
    assert isinstance(result, LossTerm)
    assert _num(result) == pytest.approx(7.0)


@pytest.mark.parametrize("scalar", BUILDERS)
def test_forward_operators(scalar):
    """Each forward operator with a number operand matches its hand value."""
    pytest.importorskip("torch")
    a = LossTerm(scalar(3.0))
    b = LossTerm(scalar(2.0))
    assert _num(a + b) == pytest.approx(5.0)
    assert _num(a - b) == pytest.approx(1.0)
    assert _num(a * b) == pytest.approx(6.0)
    assert _num(a / b) == pytest.approx(1.5)
    assert _num(a**b) == pytest.approx(9.0)
    assert _num(-a) == pytest.approx(-3.0)
    # Right operand a plain number:
    assert _num(a + 4.0) == pytest.approx(7.0)
    assert _num(a - 4.0) == pytest.approx(-1.0)
    assert _num(a * 4.0) == pytest.approx(12.0)
    assert _num(a / 4.0) == pytest.approx(0.75)
    assert _num(a**2.0) == pytest.approx(9.0)


@pytest.mark.parametrize("scalar", BUILDERS)
def test_reversed_operators_with_number(scalar):
    """Reversed operators (number on the left) keep operand order correct."""
    pytest.importorskip("torch")
    p1 = LossTerm(scalar(3.0))
    assert _num(4.0 + p1) == pytest.approx(7.0)
    assert _num(2.0 - p1) == pytest.approx(-1.0)  # non-commutative
    assert _num(0.5 * p1) == pytest.approx(1.5)
    assert _num(2.0 / p1) == pytest.approx(2.0 / 3.0)  # non-commutative
    assert _num(2.0**p1) == pytest.approx(8.0)  # non-commutative


@pytest.mark.parametrize("scalar", BUILDERS)
def test_lossterms_attribute_and_item_access(scalar):
    """Attribute and subscript access return the same ``LossTerm``."""
    pytest.importorskip("torch")
    terms = LossTerms({"lm_head": LossTerm(scalar(2.0)), "p1": LossTerm(scalar(3.0))})
    assert terms.lm_head is terms["lm_head"]
    assert _num(terms.p1) == pytest.approx(3.0)
    assert "p1" in terms
    assert set(terms) == {"lm_head", "p1"}


def test_lossterms_unknown_attribute_lists_available():
    """Unknown attribute raises ``AttributeError`` naming available terms."""
    terms = LossTerms(
        {
            "lm_head": LossTerm(_mlx_scalar(2.0)),
            "p1": LossTerm(_mlx_scalar(3.0)),
            "p2": LossTerm(_mlx_scalar(1.0)),
        }
    )
    with pytest.raises(AttributeError) as exc:
        _ = terms.p3
    msg = str(exc.value)
    assert "p3" in msg
    assert "lm_head" in msg and "p1" in msg and "p2" in msg


def test_lossterms_unknown_item_lists_available():
    """Unknown subscript raises ``KeyError`` naming available terms."""
    terms = LossTerms({"lm_head": LossTerm(_mlx_scalar(2.0)), "p1": LossTerm(_mlx_scalar(3.0))})
    with pytest.raises(KeyError) as exc:
        _ = terms["p3"]
    msg = str(exc.value)
    assert "p3" in msg
    assert "lm_head" in msg and "p1" in msg


def test_lossterm_repr_names_value():
    """``repr`` includes the wrapped value for debugging."""
    term = LossTerm(_mlx_scalar(2.0))
    assert "LossTerm(" in repr(term)


def test_lossterms_copy_pickle_no_recursion():
    """copy/deepcopy/pickle must not RecursionError (the unset-``_terms``-slot guard).

    Regression: ``__getattr__`` accessed ``self._terms``, which on an instance whose
    slot is unset (copy/pickle/``__new__``) re-invoked ``__getattr__('_terms')`` ->
    infinite recursion. Plain-float values exercise the container's picklability
    (not the tensor's).
    """
    import copy
    import pickle

    terms = LossTerms({"a": LossTerm(1.0), "b": LossTerm(2.0)})
    for clone in (copy.copy(terms), copy.deepcopy(terms), pickle.loads(pickle.dumps(terms))):
        assert set(clone) == {"a", "b"}
        assert clone["a"].value == 1.0
    # A bare __new__ instance raises a clean AttributeError, not RecursionError:
    bare = LossTerms.__new__(LossTerms)
    with pytest.raises(AttributeError):
        _ = bare.anything


def test_lossterm_has_no_float_or_bool():
    """``__float__`` / ``__bool__`` are absent (would crash traced tensors)."""
    assert "__float__" not in LossTerm.__dict__
    assert "__bool__" not in LossTerm.__dict__

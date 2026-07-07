"""Regression: degenerate sampling controls fail loudly, not silently.

repetition_penalty<=0 silently no-op'd (falsy 0) or inverted logits; top_p outside
(0,1] silently kept the whole vocab; temperature=NaN slipped past the sign check into
greedy; num_return_sequences=1.5/0 was int()-truncated to 1 and dropped.
"""

from __future__ import annotations

import pytest

from auto_chasm._generation_utils import (
    _check_temperature,
    check_sampling_params,
    reject_num_return_sequences,
)


def test_temperature_nan_raises() -> None:
    with pytest.raises(ValueError, match="temperature must be >= 0"):
        _check_temperature(float("nan"))


def test_temperature_valid_ok() -> None:
    _check_temperature(0.0)  # greedy
    _check_temperature(0.7)


@pytest.mark.parametrize("rep", [0.0, -1.0, -0.5])
def test_repetition_penalty_non_positive_raises(rep: float) -> None:
    with pytest.raises(ValueError, match="repetition_penalty must be > 0"):
        check_sampling_params({"repetition_penalty": rep})


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.5])
def test_top_p_out_of_range_raises(top_p: float) -> None:
    with pytest.raises(ValueError, match="top_p must be in"):
        check_sampling_params({"top_p": top_p})


def test_top_k_negative_raises() -> None:
    with pytest.raises(ValueError, match="top_k must be >= 0"):
        check_sampling_params({"top_k": -3})


def test_valid_sampling_params_ok() -> None:
    check_sampling_params({"repetition_penalty": 1.3, "top_p": 0.9, "top_k": 40})
    check_sampling_params({"top_p": 1.0})  # 1.0 = keep all (no nucleus filtering)
    check_sampling_params({})  # nothing set


@pytest.mark.parametrize("n", [1.5, 0, 2, 3])
def test_num_return_sequences_not_one_raises(n: object) -> None:
    with pytest.raises(ValueError, match="num_return_sequences"):
        reject_num_return_sequences({"num_return_sequences": n})


def test_num_return_sequences_one_ok() -> None:
    reject_num_return_sequences({"num_return_sequences": 1})
    reject_num_return_sequences({"num_return_sequences": 1.0})  # equals 1
    reject_num_return_sequences({})


def test_validation_reaches_public_generate() -> None:
    """The public generate() entry rejects a bad sampling value before dispatch."""
    from auto_chasm.generation import generate

    with pytest.raises(ValueError, match="repetition_penalty must be > 0"):
        generate(object(), object(), "hi", repetition_penalty=0.0)

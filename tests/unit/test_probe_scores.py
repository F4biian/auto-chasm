"""Per-token probe scores and clustered bootstrap intervals.

An eval loop reports an aggregate and discards the ``(score, label)`` pairs, so
there is nothing left to put an error bar around. These pin the two things that
are easy to get wrong once you do keep them: the AUROC must be the CORPUS value
(not a mean of per-batch ones), and the bootstrap must resample RESPONSES —
tokens inside a response share a prompt and a hallucination span, so resampling
them independently reports intervals several times too narrow.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from auto_chasm.metrics import auroc
from auto_chasm.probe_scores import ProbeScores, _labels_for


def _correlated(n_groups: int = 200, per: int = 30, seed: int = 0) -> ProbeScores:
    """The real correlation structure, not just a per-group score offset.

    What makes token-level bootstrapping wrong is that a response is MOSTLY
    hallucinated or MOSTLY clean -- its label RATE is a group property, not an
    independent per-token draw. A fixture that varies only the score offset while
    every group keeps the same label rate shows almost no clustering effect,
    because the resampled class balance barely moves.
    """
    rng = np.random.default_rng(seed)
    offs = rng.normal(0, 1.2, n_groups)
    y, s, g = [], [], []
    for i in range(n_groups):
        rate = 0.5 if rng.random() < 0.3 else 0.02   # this response's own rate
        lab = (rng.random(per) < rate).astype(int)
        y.append(lab)
        s.append(rng.normal(lab * 0.9 + offs[i], 1.0))
        g.append(np.full(per, i))
    return ProbeScores(scores={"p": np.concatenate(s)}, labels=np.concatenate(y),
                       groups=np.concatenate(g), probe_names=["p"])


def test_point_estimate_is_the_corpus_auroc() -> None:
    ps = _correlated()
    expected = auroc(ps.scores["p"], ps.labels, np.ones_like(ps.labels, dtype=bool))
    assert ps.auroc("p") == pytest.approx(expected)


def test_clustered_interval_is_wider_than_token_level() -> None:
    """The whole reason clustering is the default."""
    ps = _correlated()
    _, clo, chi = ps.bootstrap("p", n_boot=200, seed=0)["p"]
    _, tlo, thi = ps.bootstrap("p", n_boot=200, seed=0, cluster=False)["p"]
    assert (chi - clo) > 1.5 * (thi - tlo)


def test_interval_brackets_the_point_estimate() -> None:
    ps = _correlated()
    point, lo, hi = ps.bootstrap("p", n_boot=200, seed=0)["p"]
    assert lo <= point <= hi


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    ps = _correlated()
    assert ps.bootstrap("p", n_boot=100, seed=7) == ps.bootstrap("p", n_boot=100, seed=7)


def test_all_probes_share_the_same_resamples() -> None:
    """Independent draws per probe would hide whether a peak between layers is real."""
    base = _correlated()
    scores = {"a": base.scores["p"], "b": base.scores["p"].copy()}
    ps = ProbeScores(scores=scores, labels=base.labels, groups=base.groups,
                     probe_names=["a", "b"])
    out = ps.bootstrap(n_boot=100, seed=0)
    assert out["a"] == out["b"]        # identical scores + shared draws -> identical CI


def test_ci_width_argument_is_honoured() -> None:
    ps = _correlated()
    _, lo95, hi95 = ps.bootstrap("p", n_boot=200, seed=0, ci=95.0)["p"]
    _, lo50, hi50 = ps.bootstrap("p", n_boot=200, seed=0, ci=50.0)["p"]
    assert (hi50 - lo50) < (hi95 - lo95)


def test_to_csv_is_plot_ready() -> None:
    ps = _correlated(n_groups=40)
    path = Path(tempfile.mkdtemp()) / "ci.csv"
    ps.to_csv(str(path), n_boot=50)
    rows = list(csv.DictReader(path.open()))
    assert rows[0].keys() >= {"probe", "auroc", "ci_lo", "ci_hi", "n_tokens", "n_groups"}
    assert int(rows[0]["n_groups"]) == 40


def test_shared_label_list_is_found_for_any_probe_name() -> None:
    """A sweep's heads are L0..L23 while the data labels one probe by its own name."""
    assert _labels_for({"halluc": [1, 0]}, "L7").tolist() == [1, 0]
    assert _labels_for([1, 0], "L7").tolist() == [1, 0]
    assert _labels_for({"halluc": [1, 0], "lm_head": [0.0, 0.0]}, "L7").tolist() == [1, 0]


def test_ambiguous_labels_raise_rather_than_guess() -> None:
    with pytest.raises(KeyError, match="No targets for probe"):
        _labels_for({"a": [1], "b": [0]}, "L7")


def test_no_probes_attached_is_a_clear_error() -> None:
    from auto_chasm.probe_scores import collect_probe_scores

    class _M:
        probes: dict[str, object] = {}

    with pytest.raises(ValueError, match="No probes attached"):
        collect_probe_scores(_M(), [])

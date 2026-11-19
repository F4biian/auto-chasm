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
from typing import Any

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


# --- bootstrap knobs --------------------------------------------------------


def test_method_basic_reflects_through_the_point_estimate() -> None:
    ps = _correlated(n_groups=60)
    pt, plo, phi = ps.bootstrap("p", n_boot=200, seed=0, method="percentile")["p"]
    _, blo, bhi = ps.bootstrap("p", n_boot=200, seed=0, method="basic")["p"]
    assert blo == pytest.approx(2 * pt - phi)
    assert bhi == pytest.approx(2 * pt - plo)


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="method must be"):
        _correlated(n_groups=10).bootstrap("p", n_boot=5, method="bca")


def test_custom_statistic_is_bootstrapped() -> None:
    """Anything (accuracy, F1, ...) — not just AUROC."""
    ps = _correlated(n_groups=60)

    def base_rate(scores: np.ndarray, labels: np.ndarray) -> float:
        return float(labels.mean())

    pt, lo, hi = ps.bootstrap("p", n_boot=200, seed=0, statistic=base_rate)["p"]
    assert pt == pytest.approx(ps.labels.mean())
    assert lo <= pt <= hi


def test_to_csv_forwards_bootstrap_kwargs() -> None:
    ps = _correlated(n_groups=30)
    path = Path(tempfile.mkdtemp()) / "c.csv"
    ps.to_csv(str(path), n_boot=40, ci=50.0, seed=3, method="basic")
    row = next(iter(csv.DictReader(path.open())))
    assert float(row["ci_lo"]) <= float(row["auroc"]) <= float(row["ci_hi"])


# --- collect_probe_scores: the extraction itself ----------------------------


@pytest.fixture(scope="module")
def tiny_scored() -> Any:
    """A real model, a real dataset, real captured states."""
    from auto_chasm import Dataset, Model, ProbeConfig

    m = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    convos, gids = [], []
    for p in range(6):
        for r in range(2):
            dirty = (p + r) % 2 == 0
            text = "It was Nikola Tesla." if dirty else "It was Bell."
            spans = [{"start": 7, "end": 19, "label": 1}] if dirty else []
            convos.append([{"role": "user", "content": f"Who invented {p}?"},
                           {"role": "assistant", "content": text,
                            "labels": {"halluc": spans}}])
            gids.append(p)
    d = Dataset.from_conversations(conversations=convos, tokenizer=m.tokenizer,
                                   default_label=0, groups=gids)
    m.add_probes([ProbeConfig(name=f"L{i}", layers=[i], module_config={"out_features": 1})
                  for i in (2, 5)])
    return m, d, m.probe_scores(d, batch_size=3, max_seq_length=128)


def test_extraction_scores_every_labeled_token_once(tiny_scored: Any) -> None:
    """N must equal the number of non-masked labels in the dataset — no drops, no dupes."""
    import numpy as np

    _, d, ps = tiny_scored
    expected = sum(int((np.asarray(s["labels"]) != -100).sum()) for s in d)
    assert len(ps) == expected


def test_extraction_aligns_scores_labels_and_groups(tiny_scored: Any) -> None:
    _, _, ps = tiny_scored
    for arr in (*ps.scores.values(), ps.labels, ps.groups):
        assert arr.shape[0] == len(ps)


def test_extraction_keeps_both_classes(tiny_scored: Any) -> None:
    _, _, ps = tiny_scored
    assert set(np.unique(ps.labels)) == {0, 1}


def test_extraction_uses_the_dataset_group_field(tiny_scored: Any) -> None:
    """Group ids must be the PROMPT ids passed to from_conversations, not row numbers."""
    _, _, ps = tiny_scored
    assert set(np.unique(ps.groups)) == set(range(6))


def test_every_attached_probe_is_scored_from_one_pass(tiny_scored: Any) -> None:
    _, _, ps = tiny_scored
    assert sorted(ps.scores) == ["L2", "L5"]
    # different layers must give DIFFERENT scores, or capture is reading one layer
    assert not np.allclose(ps.scores["L2"], ps.scores["L5"])


def test_probe_names_selects_a_subset(tiny_scored: Any) -> None:
    m, d, _ = tiny_scored
    ps = m.probe_scores(d, probe_names=["L5"], batch_size=3, max_seq_length=128)
    assert sorted(ps.scores) == ["L5"]


def test_scoring_is_reproducible(tiny_scored: Any) -> None:
    m, d, ps = tiny_scored
    again = m.probe_scores(d, batch_size=3, max_seq_length=128)
    assert np.allclose(ps.scores["L2"], again.scores["L2"])
    assert np.array_equal(ps.labels, again.labels)


def test_batch_size_does_not_change_the_result(tiny_scored: Any) -> None:
    """Padding differs between batch shapes; the masked output must not."""
    m, d, ps = tiny_scored
    other = m.probe_scores(d, batch_size=1, max_seq_length=128)
    assert np.allclose(np.sort(ps.scores["L2"]), np.sort(other.scores["L2"]), atol=1e-4)
    assert ps.labels.sum() == other.labels.sum()


def test_to_csv_accepts_a_precomputed_bootstrap() -> None:
    """Calling bootstrap() then to_csv() silently wrote the DEFAULT interval.

    bootstrap() is a pure function, not a setting, so its result has to be handed
    back explicitly — otherwise an expensive 90%/4200-draw run is discarded and
    the file holds a 95%/1000-draw one instead.
    """
    ps = _correlated(n_groups=40)
    stats = ps.bootstrap(n_boot=60, ci=90.0, seed=5)
    path = Path(tempfile.mkdtemp()) / "given.csv"
    ps.to_csv(str(path), stats=stats)
    row = next(iter(csv.DictReader(path.open())))
    assert float(row["ci_lo"]) == pytest.approx(stats["p"][1])
    assert float(row["ci_hi"]) == pytest.approx(stats["p"][2])


def test_to_csv_rejects_stats_plus_options() -> None:
    ps = _correlated(n_groups=20)
    with pytest.raises(TypeError, match="not both"):
        ps.to_csv("/tmp/x.csv", stats=ps.bootstrap(n_boot=20), n_boot=999)


def test_to_csv_writes_only_the_probes_in_stats() -> None:
    """bootstrap(name=...) returns one probe; the CSV must not KeyError on the rest."""
    base = _correlated(n_groups=20)
    ps = ProbeScores(scores={"a": base.scores["p"], "b": base.scores["p"]},
                     labels=base.labels, groups=base.groups, probe_names=["a", "b"])
    path = Path(tempfile.mkdtemp()) / "one.csv"
    ps.to_csv(str(path), stats=ps.bootstrap("a", n_boot=20))
    rows = list(csv.DictReader(path.open()))
    assert [r["probe"] for r in rows] == ["a"]


def test_to_csv_options_are_explicit_for_editors() -> None:
    """``**kwargs`` forwarded the options but gave editors nothing to complete."""
    import inspect

    params = set(inspect.signature(ProbeScores.to_csv).parameters) - {"self"}
    assert {"n_boot", "ci", "seed", "cluster", "method", "statistic", "stats"} <= params
    assert not any(
        q.kind is inspect.Parameter.VAR_KEYWORD
        for q in inspect.signature(ProbeScores.to_csv).parameters.values()
    )


def test_probe_scores_return_type_is_concrete() -> None:
    """Annotated ``Any``, so no editor could complete .bootstrap()/.to_csv()."""
    import inspect

    from auto_chasm import Model

    assert inspect.signature(Model.probe_scores).return_annotation == "ProbeScores"


def test_to_csv_honours_explicit_options() -> None:
    ps = _correlated(n_groups=30)
    path = Path(tempfile.mkdtemp()) / "opt.csv"
    ps.to_csv(str(path), n_boot=50, ci=50.0, seed=1)
    narrow = next(iter(csv.DictReader(path.open())))
    ps.to_csv(str(path), n_boot=50, ci=99.0, seed=1)
    wide = next(iter(csv.DictReader(path.open())))
    w_n = float(narrow["ci_hi"]) - float(narrow["ci_lo"])
    w_w = float(wide["ci_hi"]) - float(wide["ci_lo"])
    assert w_w > w_n


def test_sweep_test_data_is_optional() -> None:
    """Scoring the test set inside run() AND again via probe_scores is a wasted pass.

    ``run`` restores each layer's best head either way, so ``probe_scores`` reads
    exactly those weights — the internal pass buys nothing when the reported test
    numbers (and their confidence intervals) come from there.
    """
    import inspect

    from auto_chasm import LayerSweep

    assert inspect.signature(LayerSweep.run).parameters["test_data"].default is None
    src = inspect.getsource(LayerSweep.run)
    assert "{} if test_data is None else trainer.evaluate(test_data)" in src

"""Mass-mean probing: fit a direction with no training, reuse every scoring tool.

``theta = mean_1 - mean_0`` is written INTO each linear head, so a mass-mean probe
is just a linear probe whose weights were computed rather than learned — and
``probe_scores``, its clustered bootstrap and the CSV all work on it unchanged.
AUROC depends only on ``theta/|theta|``, so scale and bias set the threshold and
never the ranking.
"""

from __future__ import annotations

import numpy as np
import pytest

from auto_chasm import Dataset, Model, ProbeConfig
from auto_chasm.metrics import auroc, to_numpy


def _data(model: Model) -> tuple[Dataset, Dataset]:
    convos, groups = [], []
    for p in range(20):
        for r in range(2):
            dirty = (p + r) % 2 == 0
            convos.append([
                {"role": "user", "content": f"Who invented {p}?"},
                {"role": "assistant",
                 "content": "It was Nikola Tesla." if dirty else "It was Bell.",
                 "labels": {"halluc": [{"start": 7, "end": 19, "label": 1}] if dirty else []}},
            ])
            groups.append(p)
    d = Dataset.from_conversations(conversations=convos, tokenizer=model.tokenizer,
                                   default_label=0, groups=groups)
    return d.split(val_fraction=0.4, seed=0, groups="group")


@pytest.fixture(scope="module")
def fitted() -> tuple[Model, Dataset, dict]:
    m = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tr, te = _data(m)
    m.add_probes([ProbeConfig(name=f"L{i}", layers=[i], module_config={"out_features": 1})
                  for i in (2, 5)])
    means = m.fit_mass_mean(tr, batch_size=4, max_seq_length=128)
    return m, te, means


def test_default_is_the_plain_projection(fitted: tuple) -> None:
    """By default a mass-mean probe IS ``h . theta`` — no scale, no bias.

    The head arrives randomly initialised, so the bias must be actively ZEROED;
    leaving it would offset every score by an arbitrary constant.
    """
    m, _, means = fitted
    w = to_numpy(m.probes["L5"].module.weight).reshape(-1).astype(np.float64)
    assert np.allclose(w, means["L5"]["theta"], rtol=1e-4, atol=1e-3)
    assert float(to_numpy(m.probes["L5"].module.bias)[0]) == pytest.approx(0.0, abs=1e-6)
    assert means["L5"]["scale"] == 1.0 and means["L5"]["bias"] == 0.0


def test_theta_is_the_difference_of_class_means(fitted: tuple) -> None:
    _, _, means = fitted
    r = means["L2"]
    assert np.allclose(r["theta"], r["mean_1"] - r["mean_0"])


def test_every_probe_is_fitted_from_one_pass(fitted: tuple) -> None:
    _, _, means = fitted
    assert sorted(means) == ["L2", "L5"]
    assert not np.allclose(means["L2"]["theta"], means["L5"]["theta"])


def test_scores_flow_through_the_normal_tooling(fitted: tuple) -> None:
    m, te, _ = fitted
    ps = m.probe_scores(te, batch_size=4, max_seq_length=128)
    assert sorted(ps.scores) == ["L2", "L5"]
    for value in ps.aurocs().values():
        assert 0.0 <= value <= 1.0


def test_auroc_ignores_scale_and_bias(fitted: tuple) -> None:
    """Why the bias choice is free: only theta's DIRECTION can change the ranking."""
    m, te, _ = fitted
    ps = m.probe_scores(te, batch_size=4, max_seq_length=128)
    s, y = ps.scores["L5"], ps.labels
    ones = np.ones_like(y, dtype=bool)
    base = auroc(s, y, ones)
    assert auroc(s * 7.5 + 3.0, y, ones) == pytest.approx(base)


def test_non_linear_head_is_rejected(fitted: tuple) -> None:
    """An MLP head has nowhere to put a direction; say so instead of guessing."""
    from auto_chasm import ModuleSpec

    m2 = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tr, _ = _data(m2)
    m2.add_probes([ProbeConfig(name="M", layers=[2],
                               module_type=ModuleSpec.mlp(hidden_dims=(8,), out_features=1),
                               module_config={"out_features": 1})])
    with pytest.raises(ValueError, match="single-logit linear head"):
        m2.fit_mass_mean(tr, batch_size=4, max_seq_length=128)


# --- hidden states ----------------------------------------------------------


def test_hidden_states_are_capped(fitted: tuple) -> None:
    m, te, _ = fitted
    hs = m.hidden_states(te, layers=[5], max_tokens=25, batch_size=4, max_seq_length=128)
    assert len(hs) == 25
    assert hs.n_seen > 25                    # it really did subsample
    assert hs.states[5].shape == (25, m.hidden_size)


def test_hidden_states_align_with_labels_and_groups(fitted: tuple) -> None:
    m, te, _ = fitted
    hs = m.hidden_states(te, layers=[2, 5], max_tokens=30, batch_size=4, max_seq_length=128)
    assert sorted(hs.states) == [2, 5]
    for arr in (*hs.states.values(), hs.labels, hs.groups):
        assert arr.shape[0] == len(hs)


def test_hidden_states_class_means_helper(fitted: tuple) -> None:
    m, te, _ = fitted
    hs = m.hidden_states(te, layers=[5], max_tokens=None, batch_size=4, max_seq_length=128)
    cm = hs.class_means(5)
    h = hs.states[5]
    assert np.allclose(cm["mean_1"], h[hs.labels == 1].mean(axis=0))
    assert np.allclose(cm["theta"], cm["mean_1"] - cm["mean_0"])


def test_hidden_states_needs_a_probe_at_that_layer(fitted: tuple) -> None:
    m, te, _ = fitted
    with pytest.raises(ValueError, match="No probe is attached at layer"):
        m.hidden_states(te, layers=[17], batch_size=4, max_seq_length=128)


def test_hidden_states_without_any_probe_explains_how() -> None:
    m = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tr, _ = _data(m)
    with pytest.raises(ValueError, match="No single-layer probes attached"):
        m.hidden_states(tr, batch_size=2, max_seq_length=128)


# --- the comparable per-probe table -----------------------------------------


def test_metrics_covers_the_headline_four(fitted: tuple) -> None:
    m, te, _ = fitted
    ps = m.probe_scores(te, batch_size=4, max_seq_length=128)
    got = ps.metrics("L5")
    assert set(got) == {"loss", "acc", "macro_f1", "auroc"}
    assert 0.0 <= got["acc"] <= 1.0
    assert got["auroc"] == pytest.approx(ps.auroc("L5"))


def test_calibration_changes_loss_but_never_auroc() -> None:
    """The point of AUROC here: invariant to any positive rescale and shift.

    So the opt-in knobs can only move ``loss`` (and, via the bias, the threshold
    metrics) -- never the layer ranking the experiment is about.
    """
    m = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tr, te = _data(m)
    m.add_probes([ProbeConfig(name="L5", layers=[5], module_config={"out_features": 1})])

    m.fit_mass_mean(tr, batch_size=4, max_seq_length=128)
    raw = m.probe_scores(te, batch_size=4, max_seq_length=128).metrics("L5")
    m.fit_mass_mean(tr, calibrate_scale=True, batch_size=4, max_seq_length=128)
    scaled = m.probe_scores(te, batch_size=4, max_seq_length=128).metrics("L5")

    assert scaled["loss"] < raw["loss"] / 10.0
    assert scaled["auroc"] == pytest.approx(raw["auroc"])
    assert scaled["acc"] == pytest.approx(raw["acc"])     # threshold is still 0


def test_calibrated_class_means_land_near_plus_minus_two(fitted: tuple) -> None:
    m, _, _ = fitted
    means = m.fit_mass_mean(_data(m)[0], calibrate_scale=True, calibrate_bias=True,
                            batch_size=4, max_seq_length=128)
    r = means["L5"]
    w, b = r["theta"] * r["scale"], r["bias"]
    assert float(w @ r["mean_1"] + b) == pytest.approx(2.0, abs=1e-3)
    assert float(w @ r["mean_0"] + b) == pytest.approx(-2.0, abs=1e-3)


def test_report_merges_every_split_into_one_row(fitted: tuple) -> None:
    m, te, _ = fitted
    tr, _ = _data(m)
    rep = m.evaluate_probes({"val": tr, "test": te}, n_boot=30, seed=0,
                            batch_size=4, max_seq_length=128)
    row = rep.rows["L5"]
    assert row["layer"] == 5
    for split in ("val", "test"):
        for metric in ("loss", "acc", "macro_f1", "auroc", "auroc_lo", "auroc_hi"):
            assert f"{split}_{metric}" in row


def test_report_csv_has_probe_and_layer_first(fitted: tuple) -> None:
    import csv
    import tempfile
    from pathlib import Path

    m, te, _ = fitted
    rep = m.evaluate_probes({"test": te}, n_boot=0, batch_size=4, max_seq_length=128)
    path = Path(tempfile.mkdtemp()) / "rep.csv"
    rep.to_csv(str(path))
    header = next(iter(csv.reader(path.open())))
    assert header[:2] == ["probe", "layer"]
    assert "test_auroc" in header


def test_report_can_skip_bootstrapping(fitted: tuple) -> None:
    """n_boot=0 for a quick pass: metrics without paying for intervals."""
    m, te, _ = fitted
    rep = m.evaluate_probes({"test": te}, n_boot=0, batch_size=4, max_seq_length=128)
    assert "test_auroc" in rep.rows["L5"]
    assert "test_auroc_lo" not in rep.rows["L5"]

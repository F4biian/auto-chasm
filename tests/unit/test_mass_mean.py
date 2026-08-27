"""Mass-mean probing: fit a direction with no training, reuse every scoring tool.

``theta = mean_1 - mean_0`` is written INTO each linear head, so a mass-mean probe
is just a linear probe whose weights were computed rather than learned — and
``probe_scores``, its clustered bootstrap and the CSV all work on it unchanged.
AUROC depends only on ``theta/|theta|``, so scale and bias set the threshold and
never the ranking.
"""

from __future__ import annotations

from pathlib import Path

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


def test_report_keeps_the_scores_it_built_from(fitted: tuple) -> None:
    """Discarding them meant probe_scores() repeated the whole forward pass.

    One pass per split is all the table needs; the raw per-token scores fall out
    of it for free, and re-running them for a CSV or a plot is pure waste.
    """
    m, te, _ = fitted
    tr, _ = _data(m)
    rep = m.evaluate_probes({"val": tr, "test": te}, n_boot=20, seed=0,
                            batch_size=4, max_seq_length=128)
    assert sorted(rep.scores) == ["test", "val"]
    kept = rep.scores["test"]
    assert kept.auroc("L5") == pytest.approx(rep.rows["L5"]["test_auroc"])
    assert len(kept) > 0


def test_report_runs_one_pass_per_split(fitted: tuple) -> None:
    import auto_chasm.probe_scores as ps_mod

    m, te, _ = fitted
    tr, _ = _data(m)
    calls = []
    original = ps_mod.collect_probe_scores
    ps_mod.collect_probe_scores = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    try:
        m.evaluate_probes({"val": tr, "test": te}, n_boot=0,
                          batch_size=4, max_seq_length=128)
    finally:
        ps_mod.collect_probe_scores = original
    assert len(calls) == 2


# --- label-free whitening ---------------------------------------------------


def test_whitening_recovers_a_direction_a_plain_projection_cannot() -> None:
    """The mechanism, on data built to have exactly that structure.

    Hidden states are strongly anisotropic, so ``mu_1 - mu_0`` picks up whatever
    high-variance nuisance direction lies between the centroids. A plain
    projection cannot discount it; whitening the states can. Note the transform
    itself never sees ``y`` -- only the direction does.
    """
    rng = np.random.default_rng(0)
    dim, n = 60, 8000
    mixing = rng.normal(0, 1, (dim, dim))
    mixing[:, 0] *= 12                       # a dominant nuisance direction
    y = rng.integers(0, 2, n)
    signal = np.zeros(dim)
    signal[3] = 1.0                          # the true class axis, low variance
    x = rng.normal(0, 1, (n, dim)) @ mixing.T + y[:, None] * signal * 6.0
    tr, te = slice(0, n // 2), slice(n // 2, None)

    xtr, ytr = x[tr], y[tr]
    theta = xtr[ytr == 1].mean(0) - xtr[ytr == 0].mean(0)
    mu = xtr.mean(0)                                        # label-free
    cov = (xtr - mu).T @ (xtr - mu) / (len(xtr) - 1)        # label-free
    cov.flat[:: dim + 1] += 1e-3 * np.trace(cov) / dim
    evals, evecs = np.linalg.eigh(cov)
    whitener = (evecs * evals**-0.5) @ evecs.T

    ones = np.ones(n - n // 2, dtype=bool)
    plain = auroc(x[te] @ theta, y[te], ones)
    whitened = auroc((x[te] - mu) @ whitener.T @ (whitener @ theta), y[te], ones)
    assert plain < 0.65
    assert whitened > 0.90


def test_whitening_matches_lda_on_auroc_exactly() -> None:
    """Two-class LDA and label-free whitening give the SAME ranking.

    ``S_total = S_within + (n0 n1 / n) theta theta^T`` -- the between-class term
    is rank one ALONG theta -- so by Sherman-Morrison ``Sigma_t^-1 theta`` is a
    POSITIVE multiple of ``Sigma_w^-1 theta``. Same direction, different length,
    and AUROC reads only the ranking. This is why the label-free transform costs
    nothing in AUROC while still applying to unlabelled states.
    """
    rng = np.random.default_rng(0)
    dim, n = 40, 6000
    mixing = rng.normal(0, 1, (dim, dim))
    mixing[:, 0] *= 12
    y = (rng.random(n) < 0.166).astype(int)          # imbalanced, as in real data
    signal = np.zeros(dim)
    signal[3] = 1.0
    x = rng.normal(0, 1, (n, dim)) @ mixing.T + y[:, None] * signal * 6.0

    m0, m1 = x[y == 0].mean(0), x[y == 1].mean(0)
    theta = m1 - m0
    mu = x.mean(0)
    s_total = (x - mu).T @ (x - mu)
    s_within = ((x[y == 0] - m0).T @ (x[y == 0] - m0)
                + (x[y == 1] - m1).T @ (x[y == 1] - m1))

    n0, n1 = float((y == 0).sum()), float((y == 1).sum())
    assert np.allclose(s_total - s_within, (n0 * n1 / n) * np.outer(theta, theta))

    free = np.linalg.solve(s_total, theta)
    lda = np.linalg.solve(s_within, theta)
    cos = free @ lda / (np.linalg.norm(free) * np.linalg.norm(lda))
    assert cos == pytest.approx(1.0, abs=1e-9)


def test_whiten_is_off_by_default() -> None:
    import inspect

    from auto_chasm.class_means import fit_mass_mean

    assert inspect.signature(fit_mass_mean).parameters["whiten"].default is False


def test_plain_fit_leaves_no_whitening_on_the_probe(fitted: tuple) -> None:
    model, _, _ = fitted
    assert all(p.whitening is None for p in model.probes.values())


def test_whitened_direction_differs_from_theta() -> None:
    m = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tr, _ = _data(m)
    m.add_probes([ProbeConfig(name="L5", layers=[5], module_config={"out_features": 1})])

    m.fit_mass_mean(tr, batch_size=4, max_seq_length=128)
    plain = to_numpy(m.probes["L5"].module.weight).reshape(-1).astype(np.float64)
    m.fit_mass_mean(tr, whiten=True, batch_size=4, max_seq_length=128)
    white = to_numpy(m.probes["L5"].module.weight).reshape(-1).astype(np.float64)

    cos = float(plain @ white / (np.linalg.norm(plain) * np.linalg.norm(white)))
    assert cos < 0.99          # genuinely rotated, not merely rescaled


def test_refitting_without_whiten_clears_the_transform() -> None:
    m = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tr, _ = _data(m)
    m.add_probes([ProbeConfig(name="L5", layers=[5], module_config={"out_features": 1})])
    m.fit_mass_mean(tr, whiten=True, batch_size=4, max_seq_length=128)
    assert m.probes["L5"].whitening is not None
    m.fit_mass_mean(tr, batch_size=4, max_seq_length=128)
    assert m.probes["L5"].whitening is None


def test_probe_scores_equal_scoring_the_whitened_state_by_hand() -> None:
    """The folded weight/bias must agree with the exposed transform.

    The head computes ``w . h + b``; the transform says the score is
    ``theta_white . Sigma^-1/2 (h - mu)``. If these ever drift apart, the probe
    and the geometry would be describing different spaces.
    """
    m = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tr, te = _data(m)
    m.add_probes([ProbeConfig(name="L5", layers=[5], module_config={"out_features": 1})])
    res = m.fit_mass_mean(tr, whiten=True, batch_size=4, max_seq_length=128)

    hs = m.hidden_states(te, batch_size=4, max_seq_length=128)
    h = hs.states[5].astype(np.float64)
    probe = m.probes["L5"]
    w = to_numpy(probe.module.weight).reshape(-1).astype(np.float64)
    b = float(to_numpy(probe.module.bias).reshape(-1)[0])

    by_head = h @ w + b
    by_hand = probe.whiten(h) @ res["L5"]["theta_whitened"]
    assert np.allclose(by_head, by_hand, rtol=1e-4, atol=1e-4)


def test_whiten_without_a_fit_says_how_to_get_one(fitted: tuple) -> None:
    model, _, _ = fitted
    probe = next(iter(model.probes.values()))
    with pytest.raises(RuntimeError, match="whiten=True"):
        probe.whiten(np.zeros((3, probe.hidden_dim)))


def test_whitening_warns_when_tokens_are_scarce(caplog: pytest.LogCaptureFixture) -> None:
    """A hidden x hidden covariance from fewer states than dimensions is noise."""
    import logging

    m = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tr, _ = _data(m)
    m.add_probes([ProbeConfig(name="L5", layers=[5], module_config={"out_features": 1})])
    with caplog.at_level(logging.WARNING):
        m.fit_mass_mean(tr, whiten=True, batch_size=4, max_seq_length=128)
    # getMessage() applies the args; .message is the raw format string.
    assert any("whitening a" in r.getMessage() for r in caplog.records)


def test_whitening_survives_a_checkpoint_round_trip(tmp_path: Path) -> None:
    """Saving a whitened probe must carry mu and Sigma^-1/2 with it."""
    m = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tr, te = _data(m)
    m.add_probes([ProbeConfig(name="L5", layers=[5], module_config={"out_features": 1})])
    m.fit_mass_mean(tr, whiten=True, batch_size=4, max_seq_length=128)
    before = m.probes["L5"].whitening
    assert before is not None
    hs = m.hidden_states(te, batch_size=4, max_seq_length=128)
    h = hs.states[5].astype(np.float64)
    whitened_before = m.probes["L5"].whiten(h)
    scores_before = np.asarray(
        m.probe_scores(te, batch_size=4, max_seq_length=128).scores["L5"]
    )

    m.save_checkpoint(str(tmp_path / "ck"))
    restored = Model.from_checkpoint(str(tmp_path / "ck"),
                                     base_model="HuggingFaceTB/SmolLM2-135M")

    after = restored.probes["L5"].whitening
    assert after is not None
    for key in ("mean", "whitener", "cov"):
        assert np.allclose(before[key], after[key]), key
    assert np.allclose(whitened_before, restored.probes["L5"].whiten(h))
    # ...and the reloaded probe must actually SCORE the same, not merely carry
    # the same arrays: the transform is folded into the head's weight and bias.
    scores_after = np.asarray(
        restored.probe_scores(te, batch_size=4, max_seq_length=128).scores["L5"]
    )
    assert np.allclose(scores_before, scores_after)


def test_checkpoint_without_whitening_restores_none(tmp_path: Path) -> None:
    m = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tr, _ = _data(m)
    m.add_probes([ProbeConfig(name="L5", layers=[5], module_config={"out_features": 1})])
    m.fit_mass_mean(tr, batch_size=4, max_seq_length=128)
    m.save_checkpoint(str(tmp_path / "ck"))
    restored = Model.from_checkpoint(str(tmp_path / "ck"),
                                     base_model="HuggingFaceTB/SmolLM2-135M")
    assert restored.probes["L5"].whitening is None


def test_resaving_without_whitening_removes_the_stale_file(tmp_path: Path) -> None:
    """A refit that drops whitening must not leave the old transform on disk."""
    m = Model.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tr, _ = _data(m)
    m.add_probes([ProbeConfig(name="L5", layers=[5], module_config={"out_features": 1})])
    ck = tmp_path / "ck"
    m.fit_mass_mean(tr, whiten=True, batch_size=4, max_seq_length=128)
    m.save_checkpoint(str(ck))
    stale = ck / "probes" / "L5.whitening.safetensors"
    assert stale.exists()

    m.fit_mass_mean(tr, batch_size=4, max_seq_length=128)
    m.save_checkpoint(str(ck))
    assert not stale.exists()
    assert (ck / "probes" / "L5.safetensors").exists()   # weights kept, not pruned

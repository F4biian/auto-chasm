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


def test_theta_is_written_into_the_head(fitted: tuple) -> None:
    """The head must BE the direction, or downstream scoring measures something else."""
    m, _, means = fitted
    w = to_numpy(m.probes["L5"].module.weight).reshape(-1)
    assert np.allclose(w, means["L5"]["theta"].astype(np.float32), rtol=1e-4, atol=1e-4)


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

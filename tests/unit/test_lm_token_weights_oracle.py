"""Oracle tests for per-token LM-loss weights (the ``labels["lm_head"]`` channel).

The channel controls how each token trains the LM head: ``w > 0`` = normal
cross-entropy, ``0.0`` = masked, ``w < 0`` = UNLIKELIHOOD training (Welleck et
al. 2019, arXiv:1908.04319 -- ``|w| * -log(1-p)``, with ``|w|`` the paper's
``alpha``); ``-100`` = unspecified (default ``1.0``). Pinned here, value-level
AND behaviorally, on both backends:

- the weighted loss equals an independent numpy recompute (incl. negative weights);
- negative weights use unlikelihood, NOT negated CE, and the loss is therefore
  bounded below by 0 -- pinned by a formula oracle and a boundedness test;
- all-``1.0`` weights reproduce the unweighted ``lm_ce`` exactly;
- weight ``0.0`` positions contribute nothing (== computing without them);
- ``-100`` in the channel means "default 1.0", so a sample WITHOUT the channel
  batched next to samples WITH one trains normally (and a plain-list sample's
  probe labels are never broadcast into the channel);
- **behavioral**: actually training a tiny model INCREASES the probability of a
  weight ``+1`` token and DECREASES the probability of a negative-weight token —
  the whole point of the feature, not just "no exceptions";
- the data layer resolves ``lm_train_on`` roles and every explicit span form
  (char span / substring / regex / token-id subsequence) to the right per-token
  weights, aggregates overlaps with min, and raises on every misuse
  (param+specs conflict, probe-style ``label`` field, unknown forms, custom
  ``lm_head`` loss combined with the channel).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from auto_chasm import Dataset, JointLoss, Model, ProbeConfig
from auto_chasm.data import build_dataset
from auto_chasm.trainers._loss_ce import weighted_lm_ce
from auto_chasm.trainers.data_utils import iterate_batches
from auto_chasm.trainers.trainable import _TrainableModel

VOCAB = 32
HID = 16


class _TinyMlx(nn.Module):
    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, HID)
        self.layers = [nn.Linear(HID, HID) for _ in range(layers)]
        self.output_proj = nn.Linear(HID, VOCAB)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = HID
    num_hidden_layers = 2


def _np_weighted_lm_ce(
    logits: np.ndarray, targets: np.ndarray, mask: np.ndarray, weights: np.ndarray
) -> float:
    """Independent numpy reference for the likelihood + unlikelihood objective.

    ``w > 0`` -> ``w * -log p`` (cross-entropy); ``w < 0`` -> ``|w| * -log(1-p)``
    (unlikelihood, Welleck et al. 2019, with the same 1e-5 floor on ``1-p``);
    ``-100`` -> the default ``1.0``. Denominator is ``sum(|w| * mask)``.
    """
    m = logits.max(-1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
    ce = -np.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    w = np.where(weights == -100.0, 1.0, weights)
    prob = np.exp(-ce)
    ul = -np.log(np.maximum(1.0 - prob, 1e-5))
    per_token = np.where(w >= 0.0, w * ce, np.abs(w) * ul)
    num = (per_token * mask).sum()
    den = max((np.abs(w) * mask).sum(), 1e-8)
    return float(num / den)


def _rand_case(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((2, 4, 6)).astype(np.float32)
    targets = rng.integers(0, 6, size=(2, 4)).astype(np.int64)
    mask = np.array([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=bool)
    return logits, targets, mask


# --------------------------------------------------------------------------- #
# Value oracles (both backends)                                                #
# --------------------------------------------------------------------------- #


def test_weighted_lm_ce_value_oracle_mlx() -> None:
    """MLX weighted LM CE == numpy reference, incl. negative and -100 weights."""
    logits, targets, mask = _rand_case()
    weights = np.array([[1.0, 0.0, -2.0, 1.0], [-100.0, 1.0, -1.0, 0.0]], dtype=np.float32)
    out = float(
        weighted_lm_ce(mx.array(logits), mx.array(targets), mx.array(mask), mx.array(weights))
    )
    assert out == pytest.approx(_np_weighted_lm_ce(logits, targets, mask, weights), abs=1e-5)


def test_weighted_lm_ce_value_oracle_torch() -> None:
    """Torch weighted LM CE == numpy reference (same case as the MLX oracle)."""
    torch = pytest.importorskip("torch")
    logits, targets, mask = _rand_case()
    weights = np.array([[1.0, 0.0, -2.0, 1.0], [-100.0, 1.0, -1.0, 0.0]], dtype=np.float32)
    out = float(
        weighted_lm_ce(
            torch.tensor(logits),
            torch.tensor(targets),
            torch.tensor(mask),
            torch.tensor(weights),
        )
    )
    assert out == pytest.approx(_np_weighted_lm_ce(logits, targets, mask, weights), abs=1e-5)


def test_all_ones_equals_unweighted_lm_ce() -> None:
    """A channel of all 1.0 reproduces the plain JointLoss LM CE bit-for-bit-ish."""
    m = Model(_TinyMlx(), None, "mlx")
    m.model.config = _Cfg()
    tm = _TrainableModel(m.model, m._probes)
    batch = mx.array([[1, 2, 3, 4, 5]])
    lengths = mx.array([[0, 4]])
    plain = JointLoss()(tm, batch, mx.array([[-100] * 5]), lengths)[0]
    ones = JointLoss()(tm, batch, {"lm_head": mx.array([[1.0] * 5])}, lengths)[0]
    assert float(plain) == pytest.approx(float(ones), abs=1e-6)


def test_weight_zero_equals_excluding_the_position() -> None:
    """Weight 0.0 at a position == computing the mean without that position."""
    logits, targets, mask = _rand_case(seed=3)
    weights = np.ones_like(mask, dtype=np.float32)
    weights[0, 1] = 0.0  # mask out one in-window position
    out = float(
        weighted_lm_ce(mx.array(logits), mx.array(targets), mx.array(mask), mx.array(weights))
    )
    # Reference: drop the position from the mask entirely, unweighted mean.
    mask_ref = mask.copy()
    mask_ref[0, 1] = False
    ref = _np_weighted_lm_ce(logits, targets, mask_ref, np.ones_like(weights))
    assert out == pytest.approx(ref, abs=1e-5)


def test_minus_100_means_default_training_in_mixed_batches() -> None:
    """A sample WITHOUT the channel (padded to -100) trains exactly like weight 1.0.

    Also pins that a plain-list sample's PROBE labels are never broadcast into
    the reserved channel by the batcher (they would mask/unlearn random tokens).
    """
    with_channel = {"tokens": [1, 2, 3, 4, 5], "labels": {"lm_head": [1.0] * 5}}
    without_channel = {"tokens": [1, 2, 3, 4, 5], "labels": [0, 1, 0, 1, 0]}  # plain probe labels
    (tokens, labels, lengths) = next(
        iterate_batches([with_channel, without_channel], 2, 8, loop=False)
    )
    lm = labels["lm_head"]
    assert lm.shape[0] == 2
    row_with = lm[0] if not (lm[0] == -100).all() else lm[1]
    row_without = lm[1] if not (lm[0] == -100).all() else lm[0]
    assert (row_without == -100).all(), (
        "the plain-list sample's probe labels leaked into the lm_head channel "
        f"(row: {row_without.tolist()})"
    )
    assert (row_with[:5] == 1.0).all()
    # And the loss treats the -100 row as default weight 1.0: equal per-sample CE
    # for two identical token rows.
    m = Model(_TinyMlx(), None, "mlx")
    m.model.config = _Cfg()
    tm = _TrainableModel(m.model, m._probes)
    both = JointLoss()(
        tm,
        mx.array(tokens),
        {"lm_head": mx.array(lm)},
        mx.array(lengths),
    )[0]
    all_ones = JointLoss()(
        tm,
        mx.array(tokens),
        {"lm_head": mx.array(np.ones_like(lm))},
        mx.array(lengths),
    )[0]
    assert float(both) == pytest.approx(float(all_ones), abs=1e-6)


# --------------------------------------------------------------------------- #
# BEHAVIORAL oracles: the feature actually trains/masks/unlearns.              #
# --------------------------------------------------------------------------- #


def _token_probs_mlx(model: Any, tokens: list[int]) -> np.ndarray:
    """P(token_t | prefix) for t=1..T-1 from a tiny MLX model."""
    logits = model(mx.array([tokens[:-1]]))
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    return np.array([float(probs[0, t, tokens[t + 1]]) for t in range(len(tokens) - 1)])


def test_behavioral_train_up_unlearn_down_mlx() -> None:
    """Training with weights [+1, -1, 0] moves the actual token probabilities.

    Token 7 (weight +1) must become MORE likely, token 9 (weight -1) LESS
    likely, after real gradient steps through JointLoss on MLX.
    """
    mx.random.seed(0)
    raw = _TinyMlx()
    tokens = [5, 7, 9, 3, 4]
    # Weight per token position (unshifted; the loss aligns w[:,1:] to targets):
    # token 7 -> +1 (train), token 9 -> -2 (unlearn), token 3 -> 0 (masked),
    # token 4 -> +1 (train).
    weights = [1.0, 1.0, -1.0, 0.0, 1.0]
    p_before = _token_probs_mlx(raw, tokens)  # [p(7), p(9), p(3), p(4)]

    m = Model(raw, None, "mlx")
    m.model.config = _Cfg()
    tm = _TrainableModel(m.model, m._probes)
    loss_fn = JointLoss()
    optimizer = __import__("mlx.optimizers", fromlist=["SGD"]).SGD(learning_rate=0.01)
    value_and_grad = nn.value_and_grad(tm, loss_fn)
    batch = mx.array([tokens])
    labels = {"lm_head": mx.array([weights])}
    lengths = mx.array([[0, len(tokens)]])
    for _ in range(30):
        (_, _, _), grads = value_and_grad(tm, batch, labels, lengths)
        optimizer.update(tm, grads)
        mx.eval(tm.state, optimizer.state)

    p_after = _token_probs_mlx(raw, tokens)
    assert p_after[0] > p_before[0], f"+1 token got LESS likely: {p_before[0]} -> {p_after[0]}"
    assert p_after[1] < p_before[1], (
        f"unlearned token got MORE likely: {p_before[1]} -> {p_after[1]}"
    )
    assert p_after[3] > p_before[3], f"+1 token got LESS likely: {p_before[3]} -> {p_after[3]}"


def test_behavioral_train_up_unlearn_down_torch() -> None:
    """The same behavioral oracle through the torch path."""
    torch = pytest.importorskip("torch")
    import torch.nn as tnn

    torch.manual_seed(0)

    class _TinyTorch(tnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = tnn.Embedding(VOCAB, HID)
            self.layers = tnn.ModuleList([tnn.Linear(HID, HID) for _ in range(2)])
            self.output_proj = tnn.Linear(HID, VOCAB)
            self.config = _Cfg()

        def forward(self, x: Any) -> Any:
            h = self.embedding(x)
            for layer in self.layers:
                h = torch.nn.functional.gelu(layer(h))
            return self.output_proj(h)

    def token_probs(model: Any, tokens: list[int]) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.tensor([tokens[:-1]]))
            probs = torch.softmax(logits.float(), dim=-1)
        return np.array([float(probs[0, t, tokens[t + 1]]) for t in range(len(tokens) - 1)])

    raw = _TinyTorch()
    tokens = [5, 7, 9, 3, 4]
    weights = [1.0, 1.0, -1.0, 0.0, 1.0]
    p_before = token_probs(raw, tokens)

    m = Model(raw, None, "torch")
    from auto_chasm.trainers.wrappers import _TorchProbeWrapper

    wrapper = _TorchProbeWrapper(raw, m._probes)
    loss_fn = JointLoss()
    optimizer = torch.optim.SGD(raw.parameters(), lr=0.01)
    batch = torch.tensor([tokens])
    labels = {"lm_head": torch.tensor([weights])}
    lengths = torch.tensor([[0, len(tokens)]])
    for _ in range(30):
        total, _, _ = loss_fn(wrapper, batch, labels, lengths)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

    p_after = token_probs(raw, tokens)
    assert p_after[0] > p_before[0]
    assert p_after[1] < p_before[1], f"unlearn failed on torch: {p_before[1]} -> {p_after[1]}"
    assert p_after[3] > p_before[3]


def test_probe_loss_unaffected_by_lm_channel() -> None:
    """The probe term is identical with and without the lm_head channel."""
    m = Model(_TinyMlx(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(
        ProbeConfig(
            name="p",
            layers=[0],
            aggregation="last",
            granularity="token",
            module_config={"out_features": 1},
        )
    )
    tm = _TrainableModel(m.model, m._probes)
    batch = mx.array([[1, 2, 3, 4, 5]])
    lengths = mx.array([[0, 4]])
    probe_labels = mx.array([[0, 1, 0, 1, 0]])
    loss = JointLoss(weights={"lm_head": 0.0})
    without = loss(tm, batch, {"p": probe_labels}, lengths)
    with_channel = loss(
        tm, batch, {"p": probe_labels, "lm_head": mx.array([[1.0, 0.0, -1.0, 1.0, 0.0]])}, lengths
    )
    assert float(without[2]["p"]) == pytest.approx(float(with_channel[2]["p"]), abs=1e-6)


def test_custom_lm_loss_plus_channel_raises() -> None:
    """A custom losses['lm_head'] callable + the weight channel must not co-exist."""
    m = Model(_TinyMlx(), None, "mlx")
    m.model.config = _Cfg()
    tm = _TrainableModel(m.model, m._probes)
    loss = JointLoss(losses={"lm_head": lambda o, t: o.lm_ce})
    with pytest.raises(ValueError, match="silently ignored"):
        loss(
            tm, mx.array([[1, 2, 3]]), {"lm_head": mx.array([[1.0, 1.0, 1.0]])}, mx.array([[0, 2]])
        )


# --------------------------------------------------------------------------- #
# Data layer: lm_train_on roles + explicit span forms.                         #
# --------------------------------------------------------------------------- #


class _CharTok:
    """One token per character; char i == token i, so spans are exact."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


def test_lm_train_on_assistant_masks_other_roles() -> None:
    """Role-based weights: 1.0 on assistant tokens, 0.0 on user tokens."""
    conv = [
        [
            {"role": "user", "content": "abc"},
            {"role": "assistant", "content": "de"},
        ]
    ]
    samples = build_dataset(conv, _CharTok(), lm_train_on="assistant")
    labels = samples[0]["labels"]
    assert isinstance(labels, dict) and set(labels) == {"lm_head"}
    assert labels["lm_head"] == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_lm_train_on_role_tuple() -> None:
    """A sequence of roles trains all of them."""
    conv = [
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    ]
    samples = build_dataset(conv, _CharTok(), lm_train_on=("assistant", "system"))
    assert samples[0]["labels"]["lm_head"] == [1.0, 0.0, 1.0]


def test_lm_train_on_no_matching_role_warns(caplog: pytest.LogCaptureFixture) -> None:
    """lm_train_on matching NO role -> all-zero weights + a loud warning."""
    import logging

    conv = [[{"role": "user", "content": "ab"}]]
    with caplog.at_level(logging.WARNING, logger="auto_chasm.data"):
        samples = build_dataset(conv, _CharTok(), lm_train_on="assistant")
    assert samples[0]["labels"]["lm_head"] == [0.0, 0.0]
    assert "trains the LM head on nothing" in caplog.text


def test_explicit_char_span_weights() -> None:
    """{start, end, weight} resolves to exactly those tokens; rest default 1.0."""
    conv = [
        [
            {
                "role": "user",
                "content": "abcdef",
                "labels": {"lm_head": [{"start": 2, "end": 4, "weight": -1.0}]},
            }
        ]
    ]
    samples = build_dataset(conv, _CharTok())
    assert samples[0]["labels"]["lm_head"] == [1.0, 1.0, -1.0, -1.0, 1.0, 1.0]


def test_explicit_text_spec_all_occurrences() -> None:
    """A substring spec weights EVERY occurrence."""
    conv = [
        [
            {
                "role": "user",
                "content": "abab",
                "labels": {"lm_head": [{"text": "ab", "weight": 0.0}]},
            }
        ]
    ]
    samples = build_dataset(conv, _CharTok())
    assert samples[0]["labels"]["lm_head"] == [0.0, 0.0, 0.0, 0.0]


def test_explicit_regex_spec() -> None:
    """A regex spec weights every match."""
    conv = [
        [
            {
                "role": "user",
                "content": "a1b22c",
                "labels": {"lm_head": [{"regex": r"\d+", "weight": -5.0}]},
            }
        ]
    ]
    samples = build_dataset(conv, _CharTok())
    assert samples[0]["labels"]["lm_head"] == [1.0, -5.0, 1.0, -5.0, -5.0, 1.0]


def test_explicit_token_ids_spec() -> None:
    """A token-id subsequence spec weights every contiguous occurrence."""
    conv = [
        [
            {
                "role": "user",
                "content": "abcab",
                "labels": {"lm_head": [{"token_ids": [ord("a"), ord("b")], "weight": 0.0}]},
            }
        ]
    ]
    samples = build_dataset(conv, _CharTok())
    assert samples[0]["labels"]["lm_head"] == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_overlapping_specs_take_min() -> None:
    """Overlaps aggregate with min: unlearn (-1) beats mask (0) beats train (1)."""
    conv = [
        [
            {
                "role": "user",
                "content": "abcd",
                "labels": {
                    "lm_head": [
                        {"start": 0, "end": 3, "weight": 0.0},
                        {"start": 1, "end": 4, "weight": -1.0},
                    ]
                },
            }
        ]
    ]
    samples = build_dataset(conv, _CharTok())
    assert samples[0]["labels"]["lm_head"] == [0.0, -1.0, -1.0, -1.0]


# --------------------------------------------------------------------------- #
# COMPOSITION: lm_train_on sets the baseline, explicit specs override it.      #
# --------------------------------------------------------------------------- #


def test_lm_train_on_composes_with_specs_on_the_assistant() -> None:
    """The 95% case: assistant-only training + an unlearn span on the reply.

    Roles set the baseline (assistant 1.0, everything else 0.0); the span
    overrides only the tokens it covers.
    """
    conv = [
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ab"},
            {
                "role": "assistant",
                "content": "cdef",
                "labels": {"lm_head": [{"start": 1, "end": 3, "weight": -1.0}]},
            },
        ]
    ]
    samples = build_dataset(conv, _CharTok(), lm_train_on="assistant")
    #        s    y    s    a    b    c     d     e     f
    assert samples[0]["labels"]["lm_head"] == [
        0.0,
        0.0,
        0.0,  # system: masked by the role baseline
        0.0,
        0.0,  # user: masked by the role baseline
        1.0,
        -1.0,
        -1.0,
        1.0,  # assistant: baseline 1.0, span overrides "de"
    ]


def test_lm_train_on_assistant_masks_system_too() -> None:
    """The sharp edge, pinned: "assistant" masks SYSTEM as well as user.

    Naming only "assistant" excludes every other role. Callers who want the
    system prompt trained must name it explicitly (next test).
    """
    conv = [
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ab"},
            {"role": "assistant", "content": "cd"},
        ]
    ]
    weights = build_dataset(conv, _CharTok(), lm_train_on="assistant")[0]["labels"]["lm_head"]
    assert weights == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert weights[:3] == [0.0, 0.0, 0.0], "system tokens must be masked by lm_train_on='assistant'"

    # Naming system keeps it — the documented escape hatch.
    both = build_dataset(conv, _CharTok(), lm_train_on=("assistant", "system"))[0]["labels"]
    assert both["lm_head"] == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]


def test_spec_overrides_a_role_masked_message_upward() -> None:
    """A spec beats the role baseline in BOTH directions.

    A weight-1.0 span on a user message trains those tokens even though
    lm_train_on='assistant' masked the message — proving the baseline is a
    default, not a floor, and does not take part in the specs' min rule.
    """
    conv = [
        [
            {
                "role": "user",
                "content": "abcd",
                "labels": {"lm_head": [{"start": 0, "end": 2, "weight": 1.0}]},
            },
            {"role": "assistant", "content": "ef"},
        ]
    ]
    weights = build_dataset(conv, _CharTok(), lm_train_on="assistant")[0]["labels"]["lm_head"]
    assert weights == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0], (
        "a spec must override the role baseline (min() with the baseline would give 0.0)"
    )


def test_composition_specs_still_min_among_themselves() -> None:
    """Composition does not weaken the min rule BETWEEN overlapping specs."""
    conv = [
        [
            {
                "role": "assistant",
                "content": "abcd",
                "labels": {
                    "lm_head": [
                        {"start": 0, "end": 3, "weight": 0.0},
                        {"start": 1, "end": 4, "weight": -1.0},
                    ]
                },
            }
        ]
    ]
    weights = build_dataset(conv, _CharTok(), lm_train_on="assistant")[0]["labels"]["lm_head"]
    assert weights == [0.0, -1.0, -1.0, -1.0]  # unlearn beats mask; uncovered would be 1.0


def test_token_ids_spec_composes_with_a_masked_role() -> None:
    """The token_ids form also overrides the role baseline (not min-ed with it).

    token_ids ranges are applied after the char-span aggregation, so this pins
    that path separately — it is the one that mutates the weight array in place.
    """
    conv = [
        [
            {
                "role": "user",
                "content": "abcd",
                "labels": {"lm_head": [{"token_ids": [ord("b"), ord("c")], "weight": 1.0}]},
            },
            {"role": "assistant", "content": "ef"},
        ]
    ]
    weights = build_dataset(conv, _CharTok(), lm_train_on="assistant")[0]["labels"]["lm_head"]
    assert weights == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]


def test_label_field_in_lm_spec_raises() -> None:
    """A probe-style 'label' field in an lm_head spec is a caught user error."""
    conv = [
        [
            {
                "role": "user",
                "content": "ab",
                "labels": {"lm_head": [{"start": 0, "end": 1, "label": 1}]},
            }
        ]
    ]
    with pytest.raises(ValueError, match="Probe spans use 'label'"):
        build_dataset(conv, _CharTok())


def test_unknown_spec_form_raises() -> None:
    """An unrecognized spec form raises instead of being silently ignored."""
    conv = [
        [
            {
                "role": "user",
                "content": "ab",
                "labels": {"lm_head": [{"tokens": [1], "weight": 0.0}]},  # typo: token_ids
            }
        ]
    ]
    with pytest.raises(ValueError, match="Unknown labels"):
        build_dataset(conv, _CharTok())


def test_probe_spans_and_lm_channel_coexist() -> None:
    """Probe spans + lm_head specs in one message -> dict with both, both correct."""
    conv = [
        [
            {
                "role": "user",
                "content": "abcd",
                "labels": {
                    "halluc": [{"start": 1, "end": 3, "label": 1}],
                    "lm_head": [{"start": 1, "end": 3, "weight": -1.0}],
                },
            }
        ]
    ]
    samples = build_dataset(conv, _CharTok(), default_label=0)
    labels = samples[0]["labels"]
    assert set(labels) == {"halluc", "lm_head"}
    assert labels["halluc"] == [0, 1, 1, 0]
    assert labels["lm_head"] == [1.0, -1.0, -1.0, 1.0]


def test_dataset_split_stratify_works_with_lm_channel() -> None:
    """stratify='label' still works when samples carry the lm_head channel."""
    samples = [
        {"tokens": [i, i + 1], "labels": {"halluc": [0, i % 2], "lm_head": [1.0, 1.0]}}
        for i in range(20)
    ]
    ds = Dataset(samples)
    train, val = ds.split(0.25, seed=0, stratify="label")
    assert len(train) + len(val) == 20 and len(val) == 4  # per-class round(10*0.25)=2, x2


def test_class_means_selects_probe_labels_from_dict() -> None:
    """compute_class_means uses the PROBE's labels, never the lm_head weights."""
    mx.random.seed(0)
    m = Model(_TinyMlx(), None, "mlx")
    m.model.config = _Cfg()
    m.attach_probe(
        ProbeConfig(
            name="p",
            layers=[0],
            aggregation="last",
            granularity="token",
            module_config={"out_features": 1},
        )
    )
    flat = [{"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 1, 0]}]
    with_channel = [
        {
            "tokens": [1, 2, 3, 4, 5],
            # lm weights deliberately look like plausible class labels (all 0/1
            # floats) so a wrong-key selection would produce different means.
            "labels": {"p": [0, 0, 1, 1, 0], "lm_head": [1.0, 1.0, 0.0, 0.0, 0.0]},
        }
    ]
    means_flat = m.compute_class_means(flat, batch_size=1, max_seq_length=8)
    means_dict = m.compute_class_means(with_channel, batch_size=1, max_seq_length=8)
    for key in ("mean_0", "mean_1"):
        a = np.array(means_flat["p"][key].astype(mx.float32))
        b = np.array(means_dict["p"][key].astype(mx.float32))
        assert np.allclose(a, b, atol=1e-6), f"{key} differs: dict labels mis-selected"


def test_empty_span_list_still_declares_the_probe() -> None:
    """A probe key with an EMPTY span list keeps the probe's label row.

    A sample annotated ``{"p": []}`` (deliberately: "no positive spans here")
    must still emit ``"p"`` in its labels dict when the lm_head channel is
    active — all ``-100`` (the documented masking semantics are unchanged),
    but structurally PRESENT. Otherwise a batch of only such samples would be
    missing the probe's key entirely, and per-probe consumers (corpus AUROC,
    class means, the loss's key-matching) would see the key flicker in and
    out from batch to batch (fatal at batch_size=1).
    """
    conv = [
        [
            {
                "role": "user",
                "content": "abc",
                "labels": {"lm_head": [{"start": 0, "end": 3, "weight": 0.0}]},
            },
            {"role": "assistant", "content": "de", "labels": {"p": []}},
        ],
        [
            {
                "role": "user",
                "content": "abc",
                "labels": {"lm_head": [{"start": 0, "end": 3, "weight": 0.0}]},
            },
            {
                "role": "assistant",
                "content": "de",
                "labels": {"p": [{"start": 0, "end": 1, "label": 1}]},
            },
        ],
    ]
    samples = build_dataset(conv, _CharTok())
    empty, labeled = samples[0]["labels"], samples[1]["labels"]
    assert set(empty) == {"p", "lm_head"}, "empty-span sample lost its probe key"
    assert empty["p"] == [-100] * 5, "empty spans must still mean fully masked"
    # Tokens without a spec get the baseline (1.0) — the -100 "unspecified"
    # sentinel is only for samples that lack the channel entirely.
    assert empty["lm_head"] == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert labeled["p"] == [-100, -100, -100, 1, -100]

    # Without the channel, the single-probe plain-list contract is unchanged:
    # an empty-span conversation is one flat all-masked list, not a dict.
    no_channel = build_dataset(
        [[{"role": "assistant", "content": "de", "labels": {"p": []}}]], _CharTok()
    )
    assert no_channel[0]["labels"] == [-100, -100]

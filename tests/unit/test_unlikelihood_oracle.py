"""Unlikelihood-training oracles for negative ``labels["lm_head"]`` weights.

A negative weight runs UNLIKELIHOOD training (Welleck et al. 2019,
arXiv:1908.04319) on that token — the LM loss there is ``|w| * -log(1 - p)``,
not the negated cross-entropy — with ``|w|`` as the paper's ``alpha``.

Every expected number here is derived BY HAND from ``-log p`` / ``-log(1-p)``
rather than from ``test_lm_token_weights_oracle``'s shared numpy reference: a
mistake duplicated in the implementation and that reference would pass there
but fail here. Pinned: the exact value (and that it is NOT the negated CE),
alpha scaling the CE/UL mix, decay to 0 for an already-unlikely token, the
``_UL_EPS`` cap for a token at ``p ~ 1``, non-negativity over a weight/scale
sweep, and — behaviorally, on torch — that the settings which diverged to NaN
under negated CE now train stably.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from auto_chasm import JointLoss, Model
from auto_chasm.trainers._loss_ce import weighted_lm_ce

VOCAB = 32
HID = 16


class _Cfg:
    hidden_size = HID
    num_hidden_layers = 2


# --------------------------------------------------------------------------- #
# UNLIKELIHOOD: hand-computed pins that do NOT reuse the shared numpy oracle.  #
# Every expected number below is derived by hand from -log p / -log(1-p) so a  #
# mistake duplicated in _np_weighted_lm_ce could not hide here.                #
# --------------------------------------------------------------------------- #

LN3 = float(np.log(3.0))
# A 2-token vocab with logits [ln3, 0] -> probs [0.75, 0.25]; target 0 -> p=0.75.
CE_P75 = float(-np.log(0.75))  # 0.2876821
UL_P75 = float(-np.log(0.25))  # 1.3862944


def _one_token_case(weight: float) -> tuple[Any, Any, Any, Any]:
    """[1,1,2] logits [ln3, 0] with target 0 (p=0.75) and one given weight."""
    logits = mx.array([[[LN3, 0.0]]])
    targets = mx.array([[0]])
    mask = mx.array([[True]])
    weights = mx.array([[weight]])
    return logits, targets, mask, weights


def test_unlikelihood_hand_computed_value() -> None:
    """A negative weight yields -log(1-p) exactly, NOT the negated CE."""
    loss = float(weighted_lm_ce(*_one_token_case(-1.0)))
    assert loss == pytest.approx(UL_P75, abs=1e-6), "negative weight is not -log(1-p)"
    # The old (rejected) semantics would have produced -CE = -0.2877 here.
    assert loss != pytest.approx(-CE_P75, abs=1e-3), "negative weight fell back to negated CE"
    # A positive weight is still plain CE.
    assert float(weighted_lm_ce(*_one_token_case(1.0))) == pytest.approx(CE_P75, abs=1e-6)


def test_unlikelihood_alpha_scales_the_mix() -> None:
    """|w| is the paper's alpha: it changes the CE/UL mix, not the objective.

    Two tokens, one trained (+1) and one unlearned. Going from alpha=1 to
    alpha=2 must reweight the mean towards the unlikelihood term by exactly
    the hand-computed amount — proving magnitudes stay meaningful rather than
    collapsing to a -1/0/+1 trichotomy.
    """
    logits = mx.array([[[LN3, 0.0], [LN3, 0.0]]])
    targets = mx.array([[0, 0]])
    mask = mx.array([[True, True]])

    a1 = float(weighted_lm_ce(logits, targets, mask, mx.array([[1.0, -1.0]])))
    a2 = float(weighted_lm_ce(logits, targets, mask, mx.array([[1.0, -2.0]])))
    assert a1 == pytest.approx((CE_P75 + UL_P75) / 2.0, abs=1e-6)
    assert a2 == pytest.approx((CE_P75 + 2.0 * UL_P75) / 3.0, abs=1e-6)
    assert a2 > a1, "alpha=2 must put MORE weight on the unlikelihood term"


def test_unlikelihood_decays_to_zero_for_an_already_unlikely_token() -> None:
    """The bounded-ness property: -log(1-p) -> 0 as p -> 0.

    This is the whole reason for preferring unlikelihood over negated CE. With
    negated CE this same input would score about -20 (and keep falling); here
    it is ~0 because there is nothing left to suppress.
    """
    logits = mx.array([[[-20.0, 0.0]]])  # target 0 -> p ~= 2e-9
    loss = float(weighted_lm_ce(logits, mx.array([[0]]), mx.array([[True]]), mx.array([[-1.0]])))
    assert loss == pytest.approx(0.0, abs=1e-6), f"UL term did not decay to 0 (got {loss})"
    assert loss >= 0.0


def test_unlikelihood_is_clamped_for_a_certain_token() -> None:
    """p ~= 1 hits the _UL_EPS floor: a finite cap, not inf/NaN."""
    logits = mx.array([[[20.0, 0.0]]])  # target 0 -> p ~= 1 - 2e-9
    loss = float(weighted_lm_ce(logits, mx.array([[0]]), mx.array([[True]]), mx.array([[-1.0]])))
    assert loss == pytest.approx(-float(np.log(1e-5)), abs=1e-4)  # ~11.5129
    assert np.isfinite(loss)


@pytest.mark.parametrize("weight", [-0.5, -1.0, -2.0, -5.0, 0.0, 1.0, 3.0])
def test_loss_is_never_negative(weight: float) -> None:
    """Bounded below by 0 for ANY weight and ANY logits — pinned over a sweep.

    The property that makes unlearning safe to mix into a joint objective: no
    weight can drive this term to -inf and swamp the probe/LM terms.
    """
    rng = np.random.default_rng(7)
    for scale in (0.1, 1.0, 10.0, 50.0):
        logits = mx.array((rng.standard_normal((2, 5, 8)) * scale).astype(np.float32))
        targets = mx.array(rng.integers(0, 8, size=(2, 5)).astype(np.int32))
        mask = mx.array(np.ones((2, 5), dtype=bool))
        weights = mx.array(np.full((2, 5), weight, dtype=np.float32))
        loss = float(weighted_lm_ce(logits, targets, mask, weights))
        assert np.isfinite(loss), f"non-finite loss at scale={scale}, w={weight}"
        assert loss >= -1e-6, f"loss went negative ({loss}) at scale={scale}, w={weight}"


def test_behavioral_strong_alpha_does_not_collapse_the_trained_token_torch() -> None:
    """Stability regression: the exact case that COLLAPSED under negated CE.

    At lr=0.05 with alpha=2 on the torch path, negated-CE unlearning ran away:
    the unbounded term dominated the objective and dragged the +1 tokens'
    probabilities toward 0 along with the unlearned one (which is why the
    original behavioral test had to be tuned down to lr=0.01 / w=-1). With the
    bounded unlikelihood term the SAME settings behave: the -2 token goes
    down, the +1 tokens still go UP.
    """
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
    weights = [1.0, 1.0, -2.0, 0.0, 1.0]  # token 9 unlearned at alpha=2
    p_before = token_probs(raw, tokens)

    m = Model(raw, None, "torch")
    from auto_chasm.trainers.wrappers import _TorchProbeWrapper

    wrapper = _TorchProbeWrapper(raw, m._probes)
    optimizer = torch.optim.SGD(raw.parameters(), lr=0.05)
    batch = torch.tensor([tokens])
    labels = {"lm_head": torch.tensor([weights])}
    lengths = torch.tensor([[0, len(tokens)]])
    for _ in range(30):
        total, _, _ = JointLoss()(wrapper, batch, labels, lengths)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

    p_after = token_probs(raw, tokens)
    assert p_after[1] < p_before[1], "the alpha=2 token was not unlearned"
    assert p_after[0] > p_before[0], (
        f"+1 token collapsed alongside the unlearned one: {p_before[0]} -> {p_after[0]}"
    )
    assert p_after[3] > p_before[3], (
        f"+1 token collapsed alongside the unlearned one: {p_before[3]} -> {p_after[3]}"
    )

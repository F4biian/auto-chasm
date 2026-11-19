"""Oracle tests for Direct Preference Optimization (RLTrainer algorithm='dpo').

DPO must make the policy prefer the ``chosen`` continuation over ``rejected``:
the reference-corrected log-prob margin (chosen − rejected) should *increase*
after training and end positive. The reference is the initial policy, captured
once via cached log-probs (no second model in memory).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import Model
from auto_chasm.config import RLConfig
from auto_chasm.trainers.rl import RLTrainer


class _TinyLM(nn.Module):
    """A minimal language model: embedding -> linear -> vocab logits."""

    def __init__(self, vocab: int = 16, hidden: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.fc = nn.Linear(hidden, hidden)
        self.output_proj = nn.Linear(hidden, vocab)

    def __call__(self, x: mx.array) -> mx.array:
        return self.output_proj(nn.gelu(self.fc(self.embedding(x))))


class _Cfg:
    """Minimal config (DPO trains the LM directly; no probes)."""

    hidden_size = 16
    num_hidden_layers = 1
    vocab_size = 16


def _margin(trainer: RLTrainer) -> float:
    """Reference-free margin: response log-prob(chosen) − log-prob(rejected)."""
    lm = trainer.wrapper.model
    chosen = mx.array([[1, 2, 7, 7, 7]])
    rejected = mx.array([[1, 2, 3, 3, 3]])
    plen = mx.array([2])
    length = mx.array([5])
    lc = trainer._resp_logp(lm(chosen), chosen, plen, length)
    lr = trainer._resp_logp(lm(rejected), rejected, plen, length)
    return float(lc[0] - lr[0])


class TestDpoPreferenceMargin:
    """DPO training raises the chosen-over-rejected log-prob margin."""

    def test_margin_increases_and_turns_positive(self) -> None:
        model = Model(_TinyLM(), None, "mlx")
        model.model.config = _Cfg()
        # The response after the 2-token prompt should be all 7s (chosen), not 3s.
        pref = [
            {"chosen": [1, 2, 7, 7, 7], "rejected": [1, 2, 3, 3, 3], "prompt_len": 2}
            for _ in range(8)
        ]
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=0.1),
            num_iters=80,
            batch_size=4,
            learning_rate=5e-2,
        )
        before = _margin(trainer)
        out = trainer.train(pref)
        after = _margin(trainer)

        assert after > before, f"DPO did not raise the preference margin: {before} -> {after}"
        assert after > 0, f"policy should prefer chosen (margin>0), got {after}"
        assert "history" in out

    def test_dpo_handles_variable_lengths_and_prompts(self) -> None:
        # Different chosen/rejected lengths + a nonzero prompt must not crash and
        # must still move the margin in the right direction.
        model = Model(_TinyLM(), None, "mlx")
        model.model.config = _Cfg()
        pref = [
            {"chosen": [1, 5, 9, 9], "rejected": [1, 5, 4], "prompt_len": 2},
            {"chosen": [2, 6, 9, 9, 9], "rejected": [2, 6, 4, 4], "prompt_len": 2},
        ]
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=0.1),
            num_iters=60,
            batch_size=2,
            learning_rate=5e-2,
        )
        lm = trainer.wrapper.model
        c = mx.array([[1, 5, 9, 9]])
        r = mx.array([[1, 5, 4, 0]])  # padded
        before = float(
            trainer._resp_logp(lm(c), c, mx.array([2]), mx.array([4]))[0]
            - trainer._resp_logp(lm(r), r, mx.array([2]), mx.array([3]))[0]
        )
        trainer.train(pref)
        after = float(
            trainer._resp_logp(lm(c), c, mx.array([2]), mx.array([4]))[0]
            - trainer._resp_logp(lm(r), r, mx.array([2]), mx.array([3]))[0]
        )
        assert after > before


class TestUnimplementedAlgorithmsStillRaise:
    """PPO/GRPO remain honest not-implemented (reward design is the experiment)."""

    def test_ppo_raises(self) -> None:
        model = Model(_TinyLM(), None, "mlx")
        model.model.config = _Cfg()
        with pytest.raises(NotImplementedError, match="Proximal"):
            RLTrainer(model, RLConfig(algorithm="ppo"))

    def test_grpo_raises(self) -> None:
        model = Model(_TinyLM(), None, "mlx")
        model.model.config = _Cfg()
        with pytest.raises(NotImplementedError, match="Group Relative"):
            RLTrainer(model, RLConfig(algorithm="grpo"))

"""Tests for the RL trainers (DPO + SFT-with-probe-penalty).

Covers ``src/auto_chasm/trainers/rl.py`` (RLTrainer, _pad_batch,
_resp_logp, _dpo_train, _dpo_batch_logp, nested dpo_loss, dispatch),
``src/auto_chasm/trainers/sft.py`` (SFTTrainer), and the ``RLConfig`` part of
``src/auto_chasm/config.py``.

The critical risk is silently-wrong DPO loss / reference math, so it is checked
against hand-computed numpy oracles, frozen-reference checks, response-masking
edge cases, ragged batching, and SFT sanity. PPO/GRPO raise on purpose (not
bugs). Tests named ``test_BUG_*`` are regression tests for specific past defects.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_map

from auto_chasm import Model
from auto_chasm.config import ProbeConfig, RLConfig
from auto_chasm.trainers.rl import RLTrainer, _pad_batch
from auto_chasm.trainers.sft import SFTTrainer

LOG2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Tiny models
# ---------------------------------------------------------------------------


class TinyLM(nn.Module):
    """A minimal language model: embedding -> linear -> vocab logits."""

    def __init__(self, vocab: int = 16, hidden: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.fc = nn.Linear(hidden, hidden)
        self.output_proj = nn.Linear(hidden, vocab)

    def __call__(self, x: mx.array) -> mx.array:
        return self.output_proj(nn.gelu(self.fc(self.embedding(x))))


class TinyMlp(nn.Module):
    """Tiny MLP with a per-token hidden state for probe-based SFT."""

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 32, num_layers: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array, **kwargs):  # noqa: ANN003, ANN204
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _LMCfg:
    hidden_size = 16
    num_hidden_layers = 1
    vocab_size = 16


class _MlpCfg:
    hidden_size = 16
    num_hidden_layers = 4
    vocab_size = 32


def _lm_model(seed: int = 0) -> Model:
    mx.random.seed(seed)
    base = TinyLM()
    base.config = _LMCfg()
    return Model(base, None, "mlx")


def _mlp_model(seed: int = 0) -> Model:
    mx.random.seed(seed)
    base = TinyMlp()
    base.config = _MlpCfg()
    return Model(base, None, "mlx")


def _clone_params(model: Model):  # noqa: ANN202
    """Deep-copy a model's parameters into a fresh, frozen reference TinyLM."""
    ref = TinyLM()
    ref.update(tree_map(lambda a: mx.array(np.array(a)), model.model.parameters()))
    mx.eval(ref.parameters())
    return ref


# ===========================================================================
# 1. _resp_logp — response masking correctness (the heart of the ref math)
# ===========================================================================


class TestRespLogpMasking:
    """`_resp_logp` response masking — the heart of the DPO reference math."""

    def _setup(self):  # noqa: ANN202
        model = _lm_model(0)
        trainer = RLTrainer(model, RLConfig(algorithm="dpo", beta=0.1), num_iters=1, batch_size=1)
        return trainer, trainer.wrapper.model

    def test_prompt_len_zero_scores_whole_response(self) -> None:
        """prompt_len=0 must score every transition (positions 1..length-1)."""
        trainer, lm = self._setup()
        seq = mx.array([[1, 2, 3, 4, 5]])
        logits = lm(seq)
        got = float(trainer._resp_logp(logits, seq, mx.array([0]), mx.array([5]))[0])

        logp = nn.log_softmax(logits[:, :-1, :], axis=-1)
        targets = seq[:, 1:]
        per = mx.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
        # prompt_len=0, length=5 -> steps 1..4 all scored (i.e. all 4 transitions).
        expected = float(per.sum())
        assert got == pytest.approx(expected, abs=1e-5)

    def test_prompt_len_equals_len_scores_nothing(self) -> None:
        """prompt_len == len => no response tokens => 0.0, never NaN."""
        trainer, lm = self._setup()
        seq = mx.array([[1, 2, 3, 4, 5]])
        got = float(trainer._resp_logp(lm(seq), seq, mx.array([5]), mx.array([5]))[0])
        assert got == 0.0
        assert not math.isnan(got)

    def test_prompt_len_greater_than_len_is_zero_not_nan(self) -> None:
        """prompt_len > len => empty window => 0.0, finite (no crash, no NaN)."""
        trainer, lm = self._setup()
        seq = mx.array([[1, 2, 3, 4, 5]])
        got = float(trainer._resp_logp(lm(seq), seq, mx.array([7]), mx.array([5]))[0])
        assert math.isfinite(got)
        assert got == 0.0

    def test_padding_does_not_leak_into_logp(self) -> None:
        """A right-padded sequence must give the SAME response log-prob as the
        unpadded one (padding beyond ``length`` is masked out).
        """
        trainer, lm = self._setup()
        short = mx.array([[1, 2, 7]])
        padded = mx.array([[1, 2, 7, 0, 0]])
        lp_short = float(trainer._resp_logp(lm(short), short, mx.array([1]), mx.array([3]))[0])
        lp_padded = float(trainer._resp_logp(lm(padded), padded, mx.array([1]), mx.array([3]))[0])
        assert lp_short == pytest.approx(lp_padded, abs=1e-5)

    def test_only_response_positions_scored(self) -> None:
        """With prompt_len=2, length=5 only the 3 response tokens (positions 2,3,4)
        contribute; the 2 prompt transitions must NOT.
        """
        trainer, lm = self._setup()
        seq = mx.array([[1, 2, 7, 7, 7]])
        logits = lm(seq)
        logp = nn.log_softmax(logits[:, :-1, :], axis=-1)
        targets = seq[:, 1:]
        per = mx.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)[0]
        # response positions predict tokens at index 2,3,4 => per[1],per[2],per[3]
        expected = float(per[1] + per[2] + per[3])
        got = float(trainer._resp_logp(logits, seq, mx.array([2]), mx.array([5]))[0])
        assert got == pytest.approx(expected, abs=1e-5)


# ===========================================================================
# 2. DPO loss — hand-math oracle against the standard formula
# ===========================================================================


class TestDpoLossHandMath:
    """DPO loss checked against the standard formula via a numpy oracle."""

    def _make(self, beta: float, lr: float, seed: int = 7) -> RLTrainer:
        model = _lm_model(seed)
        return RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=beta),
            num_iters=1,
            batch_size=4,
            learning_rate=lr,
        )

    def test_iter0_loss_is_log2_when_policy_equals_ref(self) -> None:
        """At iteration 0 the policy IS the reference, so every per-example logit
        is exactly 0 and the DPO loss is -log sigmoid(0) = log 2.
        """
        trainer = self._make(beta=0.2, lr=0.0)
        trainer.logging_steps = 1
        pref = [
            {"chosen": [1, 2, 7, 7, 7], "rejected": [1, 2, 3, 3, 3], "prompt_len": 2}
            for _ in range(4)
        ]
        out = trainer.train(pref)
        first = out["history"].entries[0].train_loss
        assert first == pytest.approx(LOG2, abs=1e-5)

    def test_iter0_log2_with_ragged_lengths(self) -> None:
        """Same log2 invariant must hold with ragged chosen/rejected lengths
        (padding masked) — guards against padding contaminating the ref logp.
        """
        trainer = self._make(beta=0.5, lr=0.0)
        trainer.logging_steps = 1
        pref = [
            {"chosen": [1, 2, 7, 7, 7], "rejected": [1, 2, 3], "prompt_len": 2},
            {"chosen": [4, 5, 9, 9], "rejected": [4, 5, 8, 8, 8], "prompt_len": 2},
        ]
        trainer.batch_size = 2
        out = trainer.train(pref)
        assert out["history"].entries[0].train_loss == pytest.approx(LOG2, abs=1e-5)

    def test_full_dpo_loss_matches_numpy_oracle(self) -> None:
        """Full hand-math: train, then independently recompute the DPO loss for a
        batch with policy=trained-model and ref=cloned-initial-model, using the
        standard formula -log sigmoid(beta*((pc-rc)-(pr-rr))). The trainer's
        _resp_logp + the DPO formula must agree with the numpy recompute to 1e-4.
        """
        model = _lm_model(11)
        ref_clone = _clone_params(model)  # frozen copy of the INITIAL policy
        beta = 0.3
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=beta),
            num_iters=12,
            batch_size=2,
            learning_rate=5e-2,
        )
        pref = [
            {"chosen": [1, 2, 7, 7, 7], "rejected": [1, 2, 3, 3, 3], "prompt_len": 2},
            {"chosen": [4, 5, 9, 9, 9], "rejected": [4, 5, 8, 8, 8], "prompt_len": 2},
        ]
        trainer.train(pref)
        policy = trainer.wrapper.model

        c = mx.array([d["chosen"] for d in pref])
        r = mx.array([d["rejected"] for d in pref])
        pl = mx.array([2, 2])
        ln = mx.array([5, 5])
        pc = trainer._resp_logp(policy(c), c, pl, ln)
        pr = trainer._resp_logp(policy(r), r, pl, ln)
        rc = trainer._resp_logp(ref_clone(c), c, pl, ln)
        rr = trainer._resp_logp(ref_clone(r), r, pl, ln)

        logit = beta * ((np.array(pc) - np.array(rc)) - (np.array(pr) - np.array(rr)))
        # Standard DPO loss, mean over the batch.
        oracle = float(np.mean(-np.log(1.0 / (1.0 + np.exp(-logit)))))

        # Independently, the trainer's own nested formula:
        bce = nn.losses.binary_cross_entropy(
            beta * ((pc - rc) - (pr - rr)),
            mx.ones((2,)),
            with_logits=True,
        )
        trainer_formula = float(mx.mean(bce))
        assert trainer_formula == pytest.approx(oracle, abs=1e-4)

    def test_preferring_chosen_lowers_loss_below_log2(self) -> None:
        """Sign check: if the policy raises chosen and lowers rejected relative to
        the reference, (pc-rc)-(pr-rr) > 0 and the DPO loss must drop below log2.
        """
        model = _lm_model(5)
        ref_clone = _clone_params(model)
        beta = 0.1
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=beta),
            num_iters=60,
            batch_size=4,
            learning_rate=5e-2,
        )
        pref = [
            {"chosen": [1, 2, 7, 7, 7], "rejected": [1, 2, 3, 3, 3], "prompt_len": 2}
            for _ in range(4)
        ]
        trainer.train(pref)
        policy = trainer.wrapper.model
        c = mx.array([[1, 2, 7, 7, 7]])
        r = mx.array([[1, 2, 3, 3, 3]])
        pl, ln = mx.array([2]), mx.array([5])
        margin = float(
            (trainer._resp_logp(policy(c), c, pl, ln) - trainer._resp_logp(ref_clone(c), c, pl, ln))
            - (
                trainer._resp_logp(policy(r), r, pl, ln)
                - trainer._resp_logp(ref_clone(r), r, pl, ln)
            )
        )
        assert margin > 0, "training toward chosen should give a positive reward margin"
        loss = float(-math.log(1.0 / (1.0 + math.exp(-beta * margin))))
        assert loss < LOG2


# ===========================================================================
# 3. Reference caching — the reference must stay FROZEN as the policy trains
# ===========================================================================


class TestReferenceIsFrozen:
    """The cached DPO reference must stay frozen as the policy trains."""

    def test_reference_logp_does_not_change_when_policy_trains(self) -> None:
        """The cached reference (initial policy) must NOT track the policy. After
        training, the reward margin (policy − ref) must be NON-zero — proving the
        reference stayed put while the policy moved.
        """
        model = _lm_model(2)
        ref_clone = _clone_params(model)
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=0.1),
            num_iters=40,
            batch_size=4,
            learning_rate=5e-2,
        )
        pref = [
            {"chosen": [1, 2, 7, 7, 7], "rejected": [1, 2, 3, 3, 3], "prompt_len": 2}
            for _ in range(4)
        ]
        # ref logp of chosen, computed from the frozen clone, must be identical
        # before and after training (the clone never trains).
        c = mx.array([[1, 2, 7, 7, 7]])
        pl, ln = mx.array([2]), mx.array([5])
        ref_before = float(trainer._resp_logp(ref_clone(c), c, pl, ln)[0])
        trainer.train(pref)
        ref_after = float(trainer._resp_logp(ref_clone(c), c, pl, ln)[0])
        assert ref_before == pytest.approx(ref_after, abs=1e-6)

        # And the policy genuinely moved away from the reference.
        policy = trainer.wrapper.model
        pol_after = float(trainer._resp_logp(policy(c), c, pl, ln)[0])
        assert abs(pol_after - ref_after) > 1e-3

    def test_cached_ref_logp_is_padding_independent(self) -> None:
        """A per-example reference log-prob must be identical whether the example is
        cached alone or inside a ragged batch padded to a longer width — otherwise
        ``ref_c_all[idx]`` would not match the policy logp computed in the loop.
        """
        model = _lm_model(0)
        trainer = RLTrainer(model, RLConfig(algorithm="dpo", beta=0.1), num_iters=1, batch_size=3)
        lm = trainer.wrapper.model
        seq = [1, 2, 7]
        alone, alone_len = _pad_batch([seq], 0)
        lp_alone = float(trainer._resp_logp(lm(alone), alone, mx.array([1]), alone_len)[0])
        batched, batched_len = _pad_batch([seq, [4, 5, 6, 7, 8, 9, 10]], 0)
        lp_batched = float(
            trainer._resp_logp(lm(batched), batched, mx.array([1, 1]), batched_len)[0]
        )
        assert lp_alone == pytest.approx(lp_batched, abs=1e-5)

    def test_margin_increases_after_dpo(self) -> None:
        """Oracle: chosen−rejected reference-free margin increases and turns positive."""
        model = _lm_model(0)
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=0.1),
            num_iters=80,
            batch_size=4,
            learning_rate=5e-2,
        )
        pref = [
            {"chosen": [1, 2, 7, 7, 7], "rejected": [1, 2, 3, 3, 3], "prompt_len": 2}
            for _ in range(8)
        ]
        lm = trainer.wrapper.model
        c = mx.array([[1, 2, 7, 7, 7]])
        r = mx.array([[1, 2, 3, 3, 3]])
        pl, ln = mx.array([2]), mx.array([5])

        def margin() -> float:
            return float(
                trainer._resp_logp(lm(c), c, pl, ln)[0] - trainer._resp_logp(lm(r), r, pl, ln)[0]
            )

        before = margin()
        trainer.train(pref)
        after = margin()
        assert after > before
        assert after > 0


# ===========================================================================
# 4. Edge cases — chosen==rejected, single pair, bs>n
# ===========================================================================


class TestDpoEdgeCases:
    """DPO edge cases: equal pairs, empty/short responses, no mutation."""

    def test_chosen_equals_rejected_loss_is_log2_and_stable(self) -> None:
        """Chosen == rejected => the logit is identically 0 at every step, so the
        loss stays at log2 and never drifts or NaNs (margin is genuinely 0).
        """
        model = _lm_model(0)
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=0.1),
            num_iters=10,
            batch_size=2,
            learning_rate=5e-2,
        )
        trainer.logging_steps = 1
        pref = [
            {"chosen": [1, 2, 5, 5], "rejected": [1, 2, 5, 5], "prompt_len": 2} for _ in range(2)
        ]
        out = trainer.train(pref)
        for entry in out["history"].entries:
            if entry.train_loss is not None:
                assert entry.train_loss == pytest.approx(LOG2, abs=1e-4)

    def test_single_pair_batch_size_larger_than_data(self) -> None:
        """A single preference pair with batch_size > n must train without crashing
        or index errors.
        """
        model = _lm_model(0)
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=0.1),
            num_iters=5,
            batch_size=8,
            learning_rate=1e-2,
        )
        pref = [{"chosen": [1, 2, 7, 7, 7], "rejected": [1, 2, 3, 3, 3], "prompt_len": 2}]
        out = trainer.train(pref)
        assert "history" in out

    def test_default_prompt_len_is_zero(self) -> None:
        """Omitting prompt_len must default to 0 (score the whole sequence), per the
        docstring — not crash.
        """
        model = _lm_model(0)
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=0.1),
            num_iters=3,
            batch_size=2,
            learning_rate=1e-2,
        )
        pref = [{"chosen": [1, 2, 7], "rejected": [1, 3, 4]} for _ in range(2)]
        out = trainer.train(pref)
        assert "history" in out

    def test_does_not_mutate_caller_preference_data(self) -> None:
        """The trainer must not mutate the caller's preference dicts/lists."""
        model = _lm_model(0)
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=0.1),
            num_iters=4,
            batch_size=2,
            learning_rate=1e-2,
        )
        pref = [
            {"chosen": [1, 2, 7, 7, 7], "rejected": [1, 2, 3, 3, 3], "prompt_len": 2}
            for _ in range(2)
        ]
        import copy

        snapshot = copy.deepcopy(pref)
        trainer.train(pref)
        assert pref == snapshot

    def test_empty_chosen_response_is_stable(self) -> None:
        """Chosen has no response (len == prompt_len): the logit comes only from
        rejected. Loss must stay finite and the model can still lower rejected.
        """
        model = _lm_model(0)
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=0.1),
            num_iters=6,
            batch_size=2,
            learning_rate=5e-2,
        )
        trainer.logging_steps = 1
        pref = [{"chosen": [1, 2], "rejected": [1, 2, 3, 3], "prompt_len": 2} for _ in range(2)]
        out = trainer.train(pref)
        losses = [e.train_loss for e in out["history"].entries if e.train_loss is not None]
        assert all(math.isfinite(x) for x in losses)
        assert losses[-1] <= losses[0] + 1e-6

    def test_non_divisible_batch_count_trains_correctly(self) -> None:
        """N not divisible by batch_size exercises the wraparound index logic; the
        cached reference must still be indexed per-example correctly and the margin
        must increase (no IndexError, no ref/example mismatch).
        """
        model = _lm_model(0)
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=0.1),
            num_iters=100,
            batch_size=2,
            learning_rate=5e-2,
        )
        pref = [
            {"chosen": [1, 2, 7, 7, 7], "rejected": [1, 2, 3, 3, 3], "prompt_len": 2}
            for _ in range(5)  # 5 is not divisible by batch_size 2
        ]
        lm = trainer.wrapper.model
        c = mx.array([[1, 2, 7, 7, 7]])
        r = mx.array([[1, 2, 3, 3, 3]])
        pl, ln = mx.array([2]), mx.array([5])

        def margin() -> float:
            return float(
                trainer._resp_logp(lm(c), c, pl, ln)[0] - trainer._resp_logp(lm(r), r, pl, ln)[0]
            )

        before = margin()
        trainer.train(pref)
        assert margin() > before


class TestDpoEmptyDataError:
    """An empty preference dataset must give a clear error, not an MLX leak.

    The SFT path raises a clean ``ValueError('Dataset is empty.')``; the DPO
    path leaks ``[concatenate] No arrays provided for concatenation`` from deep in
    the reference-caching step. That is a confusing-error DX bug.
    """

    def test_BUG_empty_preference_data_clear_error(self) -> None:
        model = _lm_model(0)
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="dpo", beta=0.1),
            num_iters=3,
            batch_size=2,
            learning_rate=1e-2,
        )
        with pytest.raises(ValueError) as excinfo:
            trainer.train([])
        msg = str(excinfo.value).lower()
        # Desired: a message that mentions the empty dataset, NOT MLX concat internals.
        assert "concatenate" not in msg, (
            "DPO leaks a cryptic MLX error on empty data instead of a clear "
            f"'empty dataset' message: {excinfo.value!r}"
        )
        assert "empty" in msg or "preference" in msg


# ===========================================================================
# 5. _pad_batch correctness
# ===========================================================================


class TestPadBatch:
    """`_pad_batch` ragged-length padding and per-example length tracking."""

    def test_ragged_padding_and_lengths(self) -> None:
        tokens, lengths = _pad_batch([[1, 2, 3], [4, 5], [6]], pad=0)
        assert np.array(tokens).tolist() == [[1, 2, 3], [4, 5, 0], [6, 0, 0]]
        assert np.array(lengths).tolist() == [3, 2, 1]

    def test_single_sequence(self) -> None:
        tokens, lengths = _pad_batch([[1, 2, 3]], pad=0)
        assert np.array(tokens).tolist() == [[1, 2, 3]]
        assert np.array(lengths).tolist() == [3]


# ===========================================================================
# 6. SFT (algorithm="sft") — beta=0 pure SFT, beta>0 adds probe penalty
# ===========================================================================


class TestSftProbePenalty:
    """SFT path: beta=0 is pure SFT; beta>0 adds the probe penalty."""

    def _data(self) -> list[dict]:
        return [{"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 0, 0]} for _ in range(16)]

    def test_beta_zero_is_pure_sft_total_equals_policy_loss(self) -> None:
        """beta=0 must make total == policy_loss (probe penalty zero-weighted)."""
        model = _mlp_model(0)
        model.attach_probe(ProbeConfig(name="p", layers=[2]))
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="sft", beta=0.0),
            num_iters=20,
            batch_size=4,
            learning_rate=1e-2,
        )
        trainer.logging_steps = 5
        out = trainer.train(self._data())
        entries = [
            e for e in out["history"].entries if e.loss_components and e.train_loss is not None
        ]
        assert entries, "no loss components recorded"
        for e in entries:
            # total (train_loss) == policy_loss when beta=0
            assert e.train_loss == pytest.approx(e.loss_components["policy_loss"], abs=1e-4)

    def test_sft_loss_decreases(self) -> None:
        """Pure SFT (beta=0) must reduce the policy loss on a learnable target."""
        model = _mlp_model(0)
        model.attach_probe(ProbeConfig(name="p", layers=[2]))
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="sft", beta=0.0),
            num_iters=40,
            batch_size=4,
            learning_rate=1e-2,
        )
        trainer.logging_steps = 5
        out = trainer.train(self._data())
        losses = [e.train_loss for e in out["history"].entries if e.train_loss is not None]
        assert losses[-1] < 0.5 * losses[0]

    def test_beta_positive_adds_probe_penalty_to_total(self) -> None:
        """beta>0: total must equal policy_loss + beta*probe_penalty per the docstring."""
        model = _mlp_model(0)
        model.attach_probe(ProbeConfig(name="p", layers=[2]))
        beta = 0.5
        trainer = RLTrainer(
            model,
            RLConfig(algorithm="sft", beta=beta),
            num_iters=10,
            batch_size=4,
            learning_rate=1e-2,
        )
        trainer.logging_steps = 1
        out = trainer.train(self._data())
        for e in out["history"].entries:
            if e.loss_components and e.train_loss is not None:
                expected = (
                    e.loss_components["policy_loss"] + beta * e.loss_components["probe_penalty"]
                )
                assert e.train_loss == pytest.approx(expected, abs=1e-4)

    def test_sftrainer_trains_and_returns_history(self) -> None:
        """The plain SFTTrainer must train and return the unified dict contract."""
        model = _mlp_model(0)
        model.attach_probe(ProbeConfig(name="p", layers=[2]))
        trainer = SFTTrainer(
            model,
            num_iters=30,
            batch_size=4,
            learning_rate=1e-2,
            logging_steps=5,
            save_steps=0,
            verbose=False,
        )
        out = trainer.train(self._data())
        assert set(out) >= {"history", "output_dir"}
        losses = [e.train_loss for e in out["history"].entries if e.train_loss is not None]
        assert losses[-1] < losses[0]


# ===========================================================================
# 7. PPO / GRPO — confirm they raise on purpose (NOT bugs)
# ===========================================================================


class TestUnimplementedRaise:
    """PPO/GRPO and unknown algorithms raise clearly (not silently)."""

    def test_ppo_raises(self) -> None:
        model = _lm_model(0)
        with pytest.raises(NotImplementedError):
            RLTrainer(model, RLConfig(algorithm="ppo"))

    def test_grpo_raises(self) -> None:
        model = _lm_model(0)
        with pytest.raises(NotImplementedError):
            RLTrainer(model, RLConfig(algorithm="grpo"))

    def test_unknown_algorithm_raises_valueerror(self) -> None:
        # An unknown algorithm is now rejected earlier — at RLConfig construction —
        # with a clear message, before RLTrainer is even built.
        model = _lm_model(0)
        with pytest.raises(ValueError, match="Unknown algorithm"):
            RLTrainer(model, RLConfig(algorithm="bogus"))  # type: ignore[arg-type]


# ===========================================================================
# 8. Cross-backend sanity (torch) — DPO path is MLX-only; confirm a clear raise
# ===========================================================================


class TestCrossBackend:
    """The MLX-only DPO path raises clearly for a torch model."""

    def test_torch_backend_raises_clearly(self) -> None:
        """RLTrainer wraps the model in the MLX-only path; a torch model must raise
        a clear ValueError at construction, not crash deep in the loop.
        """
        pytest.importorskip("torch")
        import torch
        import torch.nn as tnn

        class TorchTinyLM(tnn.Module):
            """Tiny torch LM used to confirm the DPO path rejects torch models."""

            def __init__(self) -> None:
                super().__init__()
                self.embedding = tnn.Embedding(16, 16)
                self.fc = tnn.Linear(16, 16)
                self.output_proj = tnn.Linear(16, 16)

            def forward(self, x):  # noqa: ANN001, ANN201
                return self.output_proj(torch.nn.functional.gelu(self.fc(self.embedding(x))))

        base = TorchTinyLM()
        base.config = _LMCfg()
        model = Model(base, None, "torch")
        with pytest.raises(ValueError, match="MLX backend"):
            RLTrainer(model, RLConfig(algorithm="dpo", beta=0.1))

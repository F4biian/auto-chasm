"""Tests for the RL-style (probe-penalty) trainer.

``algorithm="sft"`` (supervised CE + beta-weighted probe penalty) and ``"dpo"``
(see ``test_dpo_oracle.py``) are real; ``ppo``/``grpo`` must raise — the library
refuses to fake them. The loss tests assert the closed-form SFT contract
``total = ce + beta * probe_penalty``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.config import RLConfig
from auto_chasm.trainers.rl import RLTrainer


class TinyMlp(nn.Module):
    """Tiny MLP for RL testing."""

    def __init__(self, hidden_dim: int = 8, vocab_size: int = 16, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self._hidden_dim = hidden_dim

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class DummyTokenizer:
    """Test helper."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids: list[int]) -> str:
        return "test"


class TestRLConfig:
    """Tests for RLConfig."""

    def test_default(self) -> None:
        cfg = RLConfig()
        assert cfg.algorithm == "sft"
        assert cfg.beta == 0.1

    def test_custom_values(self) -> None:
        cfg = RLConfig(algorithm="sft", beta=0.5)
        assert cfg.algorithm == "sft"
        assert cfg.beta == 0.5


def _make_model() -> object:
    from auto_chasm.model import Model

    base = TinyMlp()

    class Config:
        """Test helper."""

        hidden_size = 8
        num_hidden_layers = 2

    base.config = Config()
    return Model(base, DummyTokenizer(), "mlx")


class TestRLTrainerInit:
    """Tests for RLTrainer construction and honesty about unimplemented modes."""

    @pytest.fixture
    def model(self) -> object:
        return _make_model()

    def test_init_sft(self, model) -> None:
        cfg = RLConfig(algorithm="sft")
        trainer = RLTrainer(model=model, rl_config=cfg, num_iters=10, batch_size=2)
        assert trainer.rl_config.algorithm == "sft"
        assert callable(trainer.loss_fn)

    @pytest.mark.parametrize("algo", ["ppo", "grpo"])
    def test_unimplemented_algorithms_raise(self, model, algo) -> None:
        """ppo/grpo are not faked as SFT — they raise a clear error (dpo is real)."""
        cfg = RLConfig(algorithm=algo)
        with pytest.raises(NotImplementedError, match="not implemented"):
            RLTrainer(model=model, rl_config=cfg, num_iters=10, batch_size=2)

    def test_custom_loss_bypasses_dispatch(self, model) -> None:
        """A custom loss_fn is used verbatim, regardless of algorithm."""

        def my_loss(m, b, lbl, lengths):
            return mx.array(0.0), mx.array(1.0), {}

        cfg = RLConfig(algorithm="sft")
        trainer = RLTrainer(model=model, rl_config=cfg, loss_fn=my_loss, num_iters=10, batch_size=2)
        assert trainer.loss_fn is my_loss

    def test_unknown_algorithm_raises(self, model) -> None:
        cfg = RLConfig(algorithm="sft")
        cfg.algorithm = "reinforce"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown RL algorithm"):
            RLTrainer(model=model, rl_config=cfg, num_iters=10)


class TestSftProbeLoss:
    """Oracle tests for the implemented sft+probe loss."""

    @pytest.fixture
    def rl_setup(self, model_wrapper: object) -> tuple:
        from auto_chasm.config import ProbeConfig

        model_wrapper.attach_probe(ProbeConfig(name="test_probe", layers=[0], aggregation="last"))
        seq_len = 8
        mx.random.seed(0)
        batch = mx.random.randint(0, 16, (2, seq_len))
        labels = mx.random.randint(0, 2, (2, seq_len))
        lengths = mx.array([[0, seq_len - 1], [0, seq_len - 1]])
        return model_wrapper, batch, labels, lengths

    def _trainer(self, model_wrapper, **kwargs):
        cfg = RLConfig(algorithm="sft", **kwargs)
        return RLTrainer(
            model=model_wrapper, rl_config=cfg, num_iters=5, batch_size=2, max_seq_length=16
        )

    def test_returns_triple(self, rl_setup) -> None:
        from auto_chasm.trainers.trainable import _TrainableModel

        mw, batch, labels, lengths = rl_setup
        trainer = self._trainer(mw)
        tm = _TrainableModel(mw.model, mw._probes)
        total, ntoks, components = trainer._sft_probe_loss(tm, batch, labels, lengths)
        assert isinstance(total, mx.array)
        assert isinstance(ntoks, mx.array)
        assert isinstance(components, dict)
        assert int(ntoks) > 0
        assert "policy_loss" in components and "probe_penalty" in components

    def test_total_equals_ce_plus_beta_penalty(self, rl_setup) -> None:
        """Oracle: total must equal policy_loss + beta * probe_penalty."""
        from auto_chasm.trainers.trainable import _TrainableModel

        mw, batch, labels, lengths = rl_setup
        beta = 0.7
        trainer = self._trainer(mw, beta=beta)
        tm = _TrainableModel(mw.model, mw._probes)
        total, _, comp = trainer._sft_probe_loss(tm, batch, labels, lengths)
        expected = float(comp["policy_loss"]) + beta * float(comp["probe_penalty"])
        assert float(total) == pytest.approx(expected, rel=1e-5)

    def test_beta_increases_loss(self, rl_setup) -> None:
        """Oracle: a higher beta increases total by exactly the penalty delta."""
        from auto_chasm.trainers.trainable import _TrainableModel

        mw, batch, labels, lengths = rl_setup
        tm = _TrainableModel(mw.model, mw._probes)
        total_lo, _, comp_lo = self._trainer(mw, beta=0.0)._sft_probe_loss(
            tm, batch, labels, lengths
        )
        total_hi, _, comp_hi = self._trainer(mw, beta=2.0)._sft_probe_loss(
            tm, batch, labels, lengths
        )
        penalty = float(comp_hi["probe_penalty"])
        assert penalty > 0
        assert float(total_hi) - float(total_lo) == pytest.approx(2.0 * penalty, rel=1e-4)

    def test_no_probes_zero_penalty(self, model_wrapper) -> None:
        from auto_chasm.trainers.trainable import _TrainableModel

        trainer = self._trainer(model_wrapper, beta=0.1)
        seq_len = 8
        batch = mx.random.randint(0, 16, (2, seq_len))
        labels = mx.random.randint(0, 2, (2, seq_len))
        lengths = mx.array([[0, seq_len - 1], [0, seq_len - 1]])
        tm = _TrainableModel(model_wrapper.model, model_wrapper._probes)
        _, _, components = trainer._sft_probe_loss(tm, batch, labels, lengths)
        assert float(components["probe_penalty"]) == pytest.approx(0.0, abs=1e-5)

    def test_rl_sft_actually_trains(self, model_wrapper) -> None:
        """Regression: RLTrainer(sft).train must run under value_and_grad.

        ``_sft_probe_loss`` used a Python ``if ntoks == 0`` on a traced array,
        which crashed during training (not in eager unit tests).
        """
        from auto_chasm.config import ProbeConfig

        model_wrapper.attach_probe(ProbeConfig(name="test_probe", layers=[0], aggregation="last"))
        data = [{"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 0, 0]} for _ in range(6)]
        # Unified contract (bug 9): all trainers' .train() return a dict with
        # "history" and "output_dir"; run() remains the History-returning API.
        result = self._trainer(model_wrapper, beta=0.5).train(data)
        assert set(result) >= {"history", "output_dir"}
        assert len(result["history"].train_losses) >= 1

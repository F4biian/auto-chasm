"""Trainer extra coverage tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_trainer import DummyTokenizer, TinyMlp


class TestTrainerCoverage:
    """Miscellaneous trainer coverage tests."""

    def test_callback_exception_propagates(self) -> None:
        """A callback that raises propagates loudly (re-raised, not swallowed)."""
        import tempfile

        import pytest

        from auto_chasm.config import ProbeConfig
        from auto_chasm.trainers.loss import JointLoss
        from auto_chasm.trainers.trainer import Trainer, TrainerCallback

        class BrokenCallback(TrainerCallback):
            """Callback that raises on every event."""

            def on_step_end(self, **kwargs: object) -> None:
                msg = "deliberate crash"
                raise RuntimeError(msg)

        data = [{"tokens": [1, 2, 3], "labels": [0, 0, 1]}]
        from auto_chasm.trainers.data_utils import JointTextDataset

        with tempfile.TemporaryDirectory() as tmp:
            ds = JointTextDataset(data, DummyTokenizer(), tokens_key="tokens")
            m = TinyMlp(hidden_dim=4, vocab_size=8, num_layers=2)

            class Cfg:
                """Dummy config."""

                hidden_size = 4
                num_hidden_layers = 2

            m.config = Cfg()  # type: ignore[attr-defined]
            from auto_chasm.model import Model

            wrapper = Model(m, DummyTokenizer(), backend_name="mlx")
            wrapper.attach_probe(ProbeConfig(name="p", layers=[0]))

            loss_fn = JointLoss()
            trainer = Trainer(
                model=wrapper,
                loss_fn=loss_fn,
                learning_rate=1e-3,
                num_iters=2,
                batch_size=2,
                max_seq_length=8,
                output_dir=str(Path(tmp) / "out"),
                callbacks=[BrokenCallback()],
                verbose=False,
            )
            with pytest.raises(RuntimeError, match="deliberate crash"):
                trainer.train(ds)

    def test_unsupported_backend_raises(self) -> None:
        """Trainer with unsupported backend raises RuntimeError."""
        from auto_chasm.trainers.loss import JointLoss
        from auto_chasm.trainers.trainer import Trainer

        data = [{"tokens": [1, 2, 3], "labels": [0, 0, 1]}]

        m = TinyMlp(hidden_dim=4, vocab_size=8, num_layers=2)

        class Cfg:
            """Dummy config."""

            hidden_size = 4
            num_hidden_layers = 2

        m.config = Cfg()  # type: ignore[attr-defined]
        from auto_chasm.model import Model

        wrapper = Model(m, DummyTokenizer(), backend_name="mlx")
        # Override backend name to unsupported value
        wrapper.backend.name = "cuda"

        loss_fn = JointLoss()
        trainer = Trainer(
            model=wrapper,
            loss_fn=loss_fn,
            num_iters=1,
            verbose=False,
        )
        with pytest.raises(RuntimeError):
            trainer.train(data)  # type: ignore[arg-type]
        wrapper.backend.name = "mlx"


class TestTrainerConfigOverride:
    """TrainingConfig parameter override behavior."""

    def test_config_overrides_defaults(self) -> None:
        """TrainingConfig values are used as defaults when params not explicitly set."""
        from auto_chasm.config import TrainingConfig
        from auto_chasm.trainers.loss import JointLoss
        from auto_chasm.trainers.trainer import Trainer

        config = TrainingConfig(learning_rate=1e-3)

        m = TinyMlp(hidden_dim=4, vocab_size=8, num_layers=2)

        class Cfg:
            """Dummy config."""

            hidden_size = 4
            num_hidden_layers = 2

        m.config = Cfg()  # type: ignore[attr-defined]
        from auto_chasm.model import Model

        wrapper = Model(m, DummyTokenizer(), backend_name="mlx")
        loss_fn = JointLoss()
        trainer = Trainer(
            model=wrapper,
            loss_fn=loss_fn,
            num_iters=1,
            verbose=False,
            config=config,
        )
        assert trainer.learning_rate == 1e-3

    def test_explicit_param_overrides_config(self) -> None:
        """Explicitly set params take priority over TrainingConfig."""
        from auto_chasm.config import TrainingConfig
        from auto_chasm.trainers.loss import JointLoss
        from auto_chasm.trainers.trainer import Trainer

        config = TrainingConfig(learning_rate=1e-3)

        m = TinyMlp(hidden_dim=4, vocab_size=8, num_layers=2)

        class Cfg:
            """Dummy config."""

            hidden_size = 4
            num_hidden_layers = 2

        m.config = Cfg()  # type: ignore[attr-defined]
        from auto_chasm.model import Model

        wrapper = Model(m, DummyTokenizer(), backend_name="mlx")
        loss_fn = JointLoss()
        trainer = Trainer(
            model=wrapper,
            loss_fn=loss_fn,
            learning_rate=5e-4,
            num_iters=1,
            verbose=False,
            config=config,
        )
        assert trainer.learning_rate == 5e-4  # explicit wins

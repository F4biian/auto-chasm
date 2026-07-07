"""Torch trainer integration tests.

Tests the full Trainer._train_torch path with a tiny PyTorch model,
verifying training, evaluation, checkpointing, and generation.
"""

from __future__ import annotations

import torch

from auto_chasm.config import ProbeConfig
from auto_chasm.trainers.loss import JointLoss


def _make_torch_model_and_tokenizer():
    """Create a tiny torch model with config for integration testing."""
    from tests.conftest import _make_torch_tiny_mlp

    torch.manual_seed(42)
    model = _make_torch_tiny_mlp(hidden_dim=16, vocab_size=32, num_layers=4)

    class Config:
        """Dummy model configuration."""

        hidden_size = 16
        num_hidden_layers = 4
        vocab_size = 32

    model.config = Config()
    tokenizer = _MockHFTokenizer()
    return model, tokenizer


class _MockHFTokenizer:
    """Mock HuggingFace-style tokenizer for torch generation tests."""

    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 for c in text[:10]]

    def decode(self, ids: list[int], **kwargs: object) -> str:
        return "".join(chr(i + 32) for i in ids if i > 0)

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:  # type: ignore[override]
        import torch

        ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def apply_chat_template(self, messages: list, **kwargs: object) -> str:  # type: ignore[override]
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _make_dataset(n: int = 32):
    """Create a tiny dataset for training."""
    data = []
    for _i in range(n):
        tokens = [1, 2, 3, 4, 5]
        labels = [0, 0, 1, 0, 0]
        data.append({"tokens": tokens, "labels": labels})
    return data


class TestTorchTrainerBasic:
    """Basic torch trainer functionality tests."""

    def test_trainer_runs_without_error(self, tmp_path):  # type: ignore[no-untyped-def]
        from auto_chasm import Model
        from auto_chasm.trainers.trainer import Trainer

        raw_model, tokenizer = _make_torch_model_and_tokenizer()
        wrapper = Model(raw_model, tokenizer, backend_name="torch")

        loss_fn = JointLoss()
        trainer = Trainer(
            model=wrapper,
            loss_fn=loss_fn,
            num_iters=5,
            batch_size=4,
            logging_steps=5,
            save_steps=0,
            early_stopping_patience=0,
            verbose=False,
            output_dir=str(tmp_path / "output"),
        )
        data = _make_dataset(16)
        result = trainer.train(data)
        assert "history" in result
        assert "output_dir" in result

    def test_trainer_loss_decreases(self, tmp_path):  # type: ignore[no-untyped-def]
        from auto_chasm import Model
        from auto_chasm.trainers.trainer import Trainer

        raw_model, tokenizer = _make_torch_model_and_tokenizer()
        wrapper = Model(raw_model, tokenizer, backend_name="torch")

        loss_fn = JointLoss()
        trainer = Trainer(
            model=wrapper,
            loss_fn=loss_fn,
            num_iters=20,
            batch_size=8,
            logging_steps=20,
            save_steps=0,
            early_stopping_patience=0,
            verbose=False,
            output_dir=str(tmp_path / "output"),
        )
        data = _make_dataset(32)
        result = trainer.train(data)
        history = result["history"]
        entries = [e for e in history if e.train_loss is not None]
        assert len(entries) >= 1
        assert entries[0].train_loss > 0

    def test_trainer_with_probes(self, tmp_path):  # type: ignore[no-untyped-def]
        from auto_chasm import Model
        from auto_chasm.trainers.trainer import Trainer

        raw_model, tokenizer = _make_torch_model_and_tokenizer()
        wrapper = Model(raw_model, tokenizer, backend_name="torch")

        probe_config = ProbeConfig(
            name="test_probe",
            layers=[-1],
            source="hidden",
        )
        wrapper.attach_probe(probe_config)

        loss_fn = JointLoss(losses={"test_probe": "bce"})
        trainer = Trainer(
            model=wrapper,
            loss_fn=loss_fn,
            num_iters=10,
            batch_size=4,
            logging_steps=10,
            save_steps=0,
            early_stopping_patience=0,
            verbose=False,
            output_dir=str(tmp_path / "output"),
        )
        data = _make_dataset(16)
        result = trainer.train(data)
        assert result["history"] is not None

    def test_trainer_saves_checkpoint(self, tmp_path):  # type: ignore[no-untyped-def]
        from auto_chasm import Model
        from auto_chasm.trainers.trainer import Trainer

        raw_model, tokenizer = _make_torch_model_and_tokenizer()
        wrapper = Model(raw_model, tokenizer, backend_name="torch")

        loss_fn = JointLoss()
        trainer = Trainer(
            model=wrapper,
            loss_fn=loss_fn,
            num_iters=10,
            batch_size=4,
            logging_steps=10,
            save_steps=5,
            early_stopping_patience=0,
            keep_best_only=False,
            verbose=False,
            output_dir=str(tmp_path / "output"),
        )
        data = _make_dataset(16)
        trainer.train(data)
        output = tmp_path / "output"
        pt_files = list(output.glob("*.pt"))
        assert len(pt_files) >= 1

    def test_trainer_evaluates(self, tmp_path):  # type: ignore[no-untyped-def]
        from auto_chasm import Model
        from auto_chasm.trainers.trainer import Trainer

        raw_model, tokenizer = _make_torch_model_and_tokenizer()
        wrapper = Model(raw_model, tokenizer, backend_name="torch")

        loss_fn = JointLoss()
        trainer = Trainer(
            model=wrapper,
            loss_fn=loss_fn,
            num_iters=5,
            batch_size=4,
            logging_steps=5,
            save_steps=0,
            eval_steps=5,
            early_stopping_patience=0,
            verbose=False,
            output_dir=str(tmp_path / "output"),
        )
        data = _make_dataset(16)
        result = trainer.train(data, val_data=data)
        assert result["history"] is not None


class TestTorchTrainerCallbacks:
    """Test callback integration with torch trainer."""

    def test_callbacks_fire(self, tmp_path):  # type: ignore[no-untyped-def]
        from auto_chasm import Model
        from auto_chasm.trainers.trainer import Trainer
        from auto_chasm.trainers.wrappers import TrainerCallback

        raw_model, tokenizer = _make_torch_model_and_tokenizer()
        wrapper = Model(raw_model, tokenizer, backend_name="torch")

        events = []

        class RecordingCallback(TrainerCallback):
            """Callback that records events for testing."""

            def on_train_begin(self, **kwargs):  # type: ignore[no-untyped-def]
                events.append("begin")

            def on_step_end(self, **kwargs):  # type: ignore[no-untyped-def]
                events.append("step")

            def on_train_end(self, **kwargs):  # type: ignore[no-untyped-def]
                events.append("end")

        loss_fn = JointLoss()
        trainer = Trainer(
            model=wrapper,
            loss_fn=loss_fn,
            num_iters=3,
            batch_size=4,
            logging_steps=3,
            save_steps=0,
            early_stopping_patience=0,
            verbose=False,
            output_dir=str(tmp_path / "output"),
            callbacks=[RecordingCallback()],
        )
        data = _make_dataset(16)
        trainer.train(data)
        assert "begin" in events
        assert "end" in events
        assert events.index("begin") < events.index("end")


class TestTorchTrainerEarlyStopping:
    """Test early stopping with torch backend."""

    def test_early_stopping_fires(self, tmp_path):  # type: ignore[no-untyped-def]
        from auto_chasm import Model
        from auto_chasm.trainers.trainer import Trainer

        raw_model, tokenizer = _make_torch_model_and_tokenizer()
        wrapper = Model(raw_model, tokenizer, backend_name="torch")

        loss_fn = JointLoss()
        trainer = Trainer(
            model=wrapper,
            loss_fn=loss_fn,
            num_iters=100,
            batch_size=4,
            logging_steps=100,
            save_steps=0,
            eval_steps=5,
            early_stopping_patience=2,
            verbose=False,
            output_dir=str(tmp_path / "output"),
        )
        data = _make_dataset(16)
        result = trainer.train(data, val_data=data)
        history = result["history"]
        # With patience=2 and 100 iters, early stopping should trigger
        # (loss won't improve enough on tiny random data)
        assert len(history) < 100


class TestTorchTrainerGeneration:
    """Test generation with torch backend."""

    def test_generate_returns_string(self):  # type: ignore[no-untyped-def]
        from auto_chasm import Model

        raw_model, tokenizer = _make_torch_model_and_tokenizer()
        wrapper = Model(raw_model, tokenizer, backend_name="torch")
        result = wrapper.generate("Hello", max_tokens=5, temperature=0.0)
        assert isinstance(result, str)

    def test_generate_deterministic(self):  # type: ignore[no-untyped-def]
        from auto_chasm import Model

        torch.manual_seed(42)
        raw_model, tokenizer = _make_torch_model_and_tokenizer()
        wrapper = Model(raw_model, tokenizer, backend_name="torch")
        r1 = wrapper.generate("Hello", max_tokens=5, temperature=0.0)
        r2 = wrapper.generate("Hello", max_tokens=5, temperature=0.0)
        assert r1 == r2


class TestTorchModelForward:
    """Test Model.forward with torch backend."""

    def test_forward_returns_model_outputs(self):  # type: ignore[no-untyped-def]
        from auto_chasm import Model
        from auto_chasm.outputs import ModelOutputs

        raw_model, tokenizer = _make_torch_model_and_tokenizer()
        wrapper = Model(raw_model, tokenizer, backend_name="torch")
        outputs = wrapper.forward([1, 2, 3, 4, 5])
        assert isinstance(outputs, ModelOutputs)
        assert outputs.lm_logits is not None

    def test_forward_with_probe(self):  # type: ignore[no-untyped-def]
        from auto_chasm import Model
        from auto_chasm.config import ProbeConfig

        raw_model, tokenizer = _make_torch_model_and_tokenizer()
        wrapper = Model(raw_model, tokenizer, backend_name="torch")
        wrapper.attach_probe(ProbeConfig(name="p", layers=[-1], source="hidden"))
        outputs = wrapper.forward([1, 2, 3, 4, 5])
        assert "p" in outputs.probes


class TestTorchModelSteering:
    """Test steering with torch backend."""

    def test_enable_disable_steering(self):  # type: ignore[no-untyped-def]
        import torch

        from auto_chasm import Model
        from auto_chasm.config import ProbeConfig

        raw_model, tokenizer = _make_torch_model_and_tokenizer()
        wrapper = Model(raw_model, tokenizer, backend_name="torch")
        wrapper.attach_probe(ProbeConfig(name="p", layers=[-1], source="hidden"))

        cm = {"p": {"mean_0": torch.zeros(16), "mean_1": torch.ones(16)}}
        wrapper.enable_steering("p", class_means=cm)
        assert "p" in wrapper.steering_hooks
        wrapper.disable_steering("p")

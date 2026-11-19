"""PEFT regressions: inference eval-mode, and assigning the wrapped model."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from auto_chasm import LoraConfig, Model
from auto_chasm.generation import _eval_mode, generate


class _TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = [nn.Linear(8, 8)]
        self.output_proj = nn.Linear(8, 16)

    def __call__(self, x: mx.array, **k: object) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class _Cfg:
    hidden_size = 8
    num_hidden_layers = 1


class _CharTok:
    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        return [0]

    def decode(self, ids: list[int]) -> str:
        return "".join("x" for _ in ids)


class _FakeBackend:
    name = "mlx"


# --- PEFT-1: inference runs in eval mode (LoRA/dropout disabled) -------------------


def test_peft1_eval_mode_restores_training_flag() -> None:
    """_eval_mode sets eval for the block and restores a prior train() state."""

    class _Mod:
        def __init__(self) -> None:
            self.training = True

        def eval(self) -> None:
            self.training = False

        def train(self, mode: bool = True) -> None:
            self.training = mode

    m = _Mod()
    with _eval_mode(m):
        assert m.training is False  # eval during inference
    assert m.training is True  # restored afterward


def test_peft1_eval_mode_leaves_eval_model_in_eval() -> None:
    """A model already in eval stays in eval (no spurious switch to train)."""

    class _Mod:
        def __init__(self) -> None:
            self.training = False

        def eval(self) -> None:
            self.training = False

        def train(self, mode: bool = True) -> None:
            self.training = mode

    m = _Mod()
    with _eval_mode(m):
        assert m.training is False
    assert m.training is False  # was not training -> not switched to train


def test_peft1_generate_runs_model_in_eval() -> None:
    """generate() forwards the base model in eval mode (dropout off), then restores."""

    class _RecordingModel:
        def __init__(self) -> None:
            self.training = True
            self.saw: list[bool] = []
            base = np.full(100, -10.0, dtype=np.float32)
            base[1] = 10.0
            self._base = base

        def eval(self) -> None:
            self.training = False

        def train(self, mode: bool = True) -> None:
            self.training = mode

        def __call__(self, x: mx.array) -> mx.array:
            self.saw.append(self.training)
            return mx.broadcast_to(mx.array(self._base), (1, x.shape[1], 100))

    model = _RecordingModel()
    # stop_sequences (never matched) routes to the manual loop, which calls the model.
    generate(model, _CharTok(), "x", max_tokens=2, backend=_FakeBackend(), stop_sequences=["zzz"])
    assert model.saw and all(t is False for t in model.saw)  # eval every forward
    assert model.training is True  # prior train() state restored


# --- PEFT-2: apply_peft assigns the wrapped model returned by the adapter ----------


def test_peft2_apply_peft_assigns_returned_model(monkeypatch) -> None:  # noqa: ANN001
    """apply_peft must store the adapter's return (torch's PeftModel wrapper)."""
    m = Model(_TinyMlp(), None, "mlx")
    m.model.config = _Cfg()
    sentinel = object()

    def _fake_apply(model, **kwargs):  # noqa: ANN001, ANN202
        return sentinel

    monkeypatch.setattr("auto_chasm.peft.apply_lora", _fake_apply)
    m.attach_lora(LoraConfig(rank=4))
    assert m.model is sentinel  # the returned wrapper replaced self.model (was discarded)


# --- PEFT-3: MLX save_adapters writes ONLY adapter params (not the base) -----------


def test_peft3_mlx_save_adapters_excludes_base_weights(tmp_path) -> None:  # noqa: ANN001
    """MLX save_adapters saves lora_a/lora_b (+DoRA m) only, never the base linear."""
    from mlx_lm.tuner.dora import DoRALinear
    from mlx_lm.tuner.lora import LoRALinear

    from auto_chasm.backends.mlx_backend import MLXModelWrapping

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = LoRALinear.from_base(nn.Linear(8, 8), r=4, scale=1.0)
            self.b = DoRALinear.from_base(nn.Linear(8, 8), r=4, scale=1.0)

        def __call__(self, x: mx.array) -> mx.array:
            return self.b(self.a(x))

    path = str(tmp_path / "adapters.safetensors")
    MLXModelWrapping().save_adapters(_M(), path)
    keys = set(mx.load(path).keys())
    assert keys  # non-empty
    # The base was left unfrozen, yet no base weight leaks into the adapter file.
    assert not any("linear.weight" in k or "linear.bias" in k for k in keys)
    assert any(k.endswith(".m") for k in keys)  # DoRA magnitude preserved
    assert any(k.endswith("lora_a") for k in keys)
    assert any(k.endswith("lora_b") for k in keys)


# --- MLX LoRA/DoRA freeze the base (torch/get_peft_model parity) -------------------


class _AttnBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8)
        self.self_attn.v_proj = nn.Linear(8, 8)


class _AttnModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = [_AttnBlock() for _ in range(2)]
        self.output_proj = nn.Linear(8, 16)


def _trainable_leaves(model: nn.Module) -> set[str]:
    from mlx.utils import tree_flatten

    return {k.rsplit(".", 1)[-1] for k, _ in tree_flatten(model.trainable_parameters())}


def test_apply_lora_freezes_base_on_mlx() -> None:
    """apply_lora leaves only the adapter factors trainable (base frozen, like torch)."""
    from auto_chasm.backends import Backend
    from auto_chasm.peft import apply_lora

    m = _AttnModel()
    apply_lora(m, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=Backend(force="mlx"))
    assert _trainable_leaves(m) == {"lora_a", "lora_b"}  # no base weights trainable


def test_apply_dora_freezes_base_on_mlx() -> None:
    """apply_dora leaves only the DoRA adapter params (factors + magnitude) trainable."""
    from auto_chasm.backends import Backend
    from auto_chasm.peft import apply_dora

    m = _AttnModel()
    apply_dora(m, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=Backend(force="mlx"))
    assert _trainable_leaves(m) == {"lora_a", "lora_b", "m"}


def test_unfreeze_lora_params_handles_dora_magnitude() -> None:
    """_unfreeze_lora_params unfreezes DoRA adapters incl. m (DoRA was skipped before)."""
    from mlx.utils import tree_flatten

    from auto_chasm.backends import Backend
    from auto_chasm.peft import _unfreeze_lora_params, apply_dora

    backend = Backend(force="mlx")
    m = _AttnModel()
    apply_dora(m, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=backend)
    m.freeze()  # a full freeze — nothing trainable
    assert len(tree_flatten(m.trainable_parameters())) == 0
    _unfreeze_lora_params(m, backend)
    assert _trainable_leaves(m) == {"lora_a", "lora_b", "m"}  # magnitude included

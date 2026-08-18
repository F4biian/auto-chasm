"""Gradient checkpointing and the unrolled-recurrence diagnostic.

Both exist because a 0.8B model exhausted a 64 GB machine. The cause was not the
parameter count and not the loss: linear-attention blocks fall back to an
unrolled per-timestep loop during training (mlx-lm selects it with
``use_kernel=not self.training``, because the fused kernel has no vjp), so one
``[B, heads, Dv, Dk]`` state per timestep is retained for backward. Measured on
Qwen3.5-0.8B: 1.0 MB per timestep per layer x 18 such layers, ~48 MB/token in
total, and 3.6-3.8x less with checkpointing on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from auto_chasm import _grad_checkpoint as gc


class _Block:
    """A block that looks like a linear-attention layer to the detector."""

    def __init__(self) -> None:
        self.linear_attn = object()


class _DenseBlock:
    def __init__(self) -> None:
        self.self_attn = object()


def _model(blocks: list[object]) -> SimpleNamespace:
    return SimpleNamespace(layers=blocks)


def test_detects_linear_attention_blocks() -> None:
    m = _model([_Block(), _DenseBlock(), _Block()])
    assert gc.unrolled_recurrence_layers(m) == 2


def test_dense_model_is_not_flagged() -> None:
    assert gc.unrolled_recurrence_layers(_model([_DenseBlock()] * 4)) == 0
    assert gc.memory_warning(_model([_DenseBlock()] * 4)) is None


def test_warning_names_the_cause_and_the_remedy() -> None:
    text = gc.memory_warning(_model([_Block(), _DenseBlock()]))
    assert text is not None
    assert "1 block(s)" in text
    # It must say what to DO, or it is just noise before a crash.
    assert "enable_gradient_checkpointing" in text
    assert "grad_accum_steps does NOT" in text  # the thing users try first and wastes time


def test_no_layers_found_is_not_a_crash() -> None:
    """The detector runs on every training start; it must never be the thing that fails."""
    assert gc.unrolled_recurrence_layers(SimpleNamespace()) == 0
    assert gc.memory_warning(SimpleNamespace()) is None


def test_enable_is_idempotent_per_class() -> None:
    """A second enable() must not wrap the wrapper (that recomputes twice)."""

    class Blk:
        def __call__(self, x: int) -> int:
            return x + 1

    m = _model([Blk(), Blk()])
    original = Blk.__call__
    try:
        assert gc._enable_mlx(m) == 1  # two instances, ONE class
        assert getattr(Blk, gc._FLAG, False) is True
        assert gc._enable_mlx(m) == 0  # already patched -> no-op
    finally:
        gc._disable_mlx(m)
    assert Blk.__call__ is original
    assert not hasattr(Blk, gc._FLAG)


def test_missing_block_list_raises_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="Could not locate the transformer block list"):
        gc._blocks(SimpleNamespace())


# --- torch backend ----------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch backend not installed")


def _tiny_causal_lm() -> object:
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model("gpt2", n_layer=2, n_head=2, n_embd=32, vocab_size=64)
    return AutoModelForCausalLM.from_config(cfg)


def test_torch_enable_disable_round_trip() -> None:
    m = _tiny_causal_lm()
    assert m.is_gradient_checkpointing is False
    assert gc._enable_torch(m) == 1
    assert m.is_gradient_checkpointing is True
    assert gc._disable_torch(m) == 1
    assert m.is_gradient_checkpointing is False


def test_torch_gradients_survive_a_frozen_base() -> None:
    """The LoRA shape: base frozen, one trainable tensor.

    Without ``enable_input_require_grads`` no input to a checkpointed block
    requires grad, and autograd returns nothing for the block interior — silently.
    """
    m = _tiny_causal_lm()
    gc._enable_torch(m)
    for p_ in m.parameters():
        p_.requires_grad_(False)
    head = m.get_output_embeddings()
    head.weight.requires_grad_(True)
    ids = torch.randint(0, 64, (1, 16))
    m(input_ids=ids, labels=ids).loss.backward()
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all()


def test_torch_target_rejects_a_plain_module() -> None:
    import torch.nn as nn

    with pytest.raises(RuntimeError, match="does not expose gradient_checkpointing_enable"):
        gc._torch_target(nn.Linear(2, 2))

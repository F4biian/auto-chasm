"""Regression: an embedding-source probe must attach after LoRA on torch.

get_peft_model wraps the torch model in a PeftModel, nesting the base one level
deeper (embed_tokens moves from model.model.embed_tokens to
model.model.model.embed_tokens). _find_embedding only walked two levels, so
attach_probe(source="embedding") raised after attach_lora — and since
Model.from_checkpoint restores adapters BEFORE probes, that broke reloading any
torch checkpoint holding an embedding probe + LoRA.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from auto_chasm import LoraConfig, Model, ProbeConfig  # noqa: E402
from auto_chasm.probe import _find_embedding  # noqa: E402


def _model() -> Model:
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=64,
    )
    return Model(LlamaForCausalLM(cfg), None, "torch")


def test_find_embedding_reaches_through_peftmodel() -> None:
    """_find_embedding locates embed_tokens inside a PeftModel wrapper."""
    m = _model()
    m.attach_lora(LoraConfig(rank=4, target_modules=["q_proj", "v_proj"]))
    assert type(m.model).__name__ == "PeftModelForCausalLM"
    module, path = _find_embedding(m.model)
    assert module is not None and path == "model.model.embed_tokens"


def test_embedding_probe_attaches_after_lora_and_forwards() -> None:
    """attach_lora then an embedding-source probe attaches, and forward runs."""
    m = _model()
    m.attach_lora(LoraConfig(rank=4, target_modules=["q_proj", "v_proj"]))
    m.attach_probe(
        ProbeConfig(name="e", source="embedding", layers=[0], module_config={"out_features": 2})
    )
    out = m.forward([[5, 6, 7, 8]])
    assert tuple(out.probes["e"].logits.shape) == (1, 4, 2)


def test_detach_restores_embedding_after_lora() -> None:
    """Detaching the embedding probe restores the wrapped module cleanly (path-based)."""
    m = _model()
    m.attach_lora(LoraConfig(rank=4, target_modules=["q_proj"]))
    m.attach_probe(
        ProbeConfig(name="e", source="embedding", layers=[0], module_config={"out_features": 2})
    )
    m.restore_original_layers()
    # The embedding is back to a plain module and a fresh forward still works.
    module, _ = _find_embedding(m.model)
    assert module is not None
    m.forward([[5, 6, 7, 8]])


def test_plain_model_embedding_probe_unaffected() -> None:
    """A non-LoRA model's embedding probe still attaches (no regression)."""
    m = _model()
    m.attach_probe(
        ProbeConfig(name="e", source="embedding", layers=[0], module_config={"out_features": 2})
    )
    m.forward([[5, 6, 7, 8]])

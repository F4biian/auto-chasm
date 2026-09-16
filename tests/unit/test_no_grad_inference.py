"""Inference passes must not build an autograd graph.

``probe_scores`` and ``hidden_states`` run forwards to READ a model, never to train
it, yet PyTorch records a graph whenever a parameter requires grad -- which a LoRA
checkpoint always has. The retained activations dwarf the weights (a 7B whose weights
are 13.5 GB was measured at 46 GB, dying on a 48 GB card), so this is a memory bug
rather than a style preference. ``eval()`` is NOT the fix: it switches dropout, not
graph building, which the first test pins down.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest

from auto_chasm import ProbeConfig, no_grad


def _samples(n: int = 4, length: int = 8, vocab: int = 32) -> list[dict[str, Any]]:
    """Pre-tokenized samples: the collectors consume any iterable of these."""
    return [
        {"tokens": [(g + i) % vocab for i in range(length)],
         "labels": [i % 2 for i in range(length)],
         "group": g}
        for g in range(n)
    ]


def _probe(model: Any, name: str = "p", layer: int = 1) -> Any:
    model.add_probes([ProbeConfig(name=name, layers=[layer],
                                  module_config={"out_features": 1})])
    return model.probes[name]


def test_eval_alone_does_not_stop_the_graph(torch_model_wrapper: Any) -> None:
    """The premise: eval() switches dropout, not autograd."""
    import torch

    model = torch_model_wrapper
    probe = _probe(model)
    model.model.eval()
    model.forward(model.to_tensor([[1, 2, 3, 4]]))
    captured = probe.get_captured_states()[0]
    assert isinstance(captured, torch.Tensor)
    assert captured.requires_grad and captured.grad_fn is not None


def test_hidden_states_builds_no_graph_by_default(torch_model_wrapper: Any) -> None:
    model = torch_model_wrapper
    probe = _probe(model)
    hs = model.hidden_states(_samples(), layers=[1], max_tokens=None,
                             batch_size=2, max_seq_length=8)
    assert hs.states[1].shape[0] > 0
    captured = probe.get_captured_states()[0]
    assert not captured.requires_grad
    assert captured.grad_fn is None


def test_probe_scores_builds_no_graph_by_default(torch_model_wrapper: Any) -> None:
    model = torch_model_wrapper
    probe = _probe(model)
    scores = model.probe_scores(_samples(), batch_size=2, max_seq_length=8)
    assert scores.scores["p"].shape[0] > 0
    captured = probe.get_captured_states()[0]
    assert not captured.requires_grad
    assert captured.grad_fn is None


@pytest.mark.parametrize("call", ["hidden_states", "probe_scores"])
def test_no_grad_false_keeps_the_graph(torch_model_wrapper: Any, call: str) -> None:
    """The opt-out stays available for anyone who wants to backpropagate."""
    model = torch_model_wrapper
    probe = _probe(model)
    kwargs: dict[str, Any] = {"batch_size": 2, "max_seq_length": 8, "no_grad": False}
    if call == "hidden_states":
        kwargs |= {"layers": [1], "max_tokens": None}
    getattr(model, call)(_samples(), **kwargs)
    captured = probe.get_captured_states()[0]
    assert captured.requires_grad and captured.grad_fn is not None


def test_public_context_disables_grad_for_a_model(torch_model_wrapper: Any) -> None:
    model = torch_model_wrapper
    probe = _probe(model)
    with no_grad(model):
        model.forward(model.to_tensor([[1, 2, 3, 4]]))
    assert probe.get_captured_states()[0].grad_fn is None
    # ...and the surrounding state is untouched once the block exits.
    model.forward(model.to_tensor([[1, 2, 3, 4]]))
    assert probe.get_captured_states()[0].grad_fn is not None


def test_public_context_accepts_a_backend_name() -> None:
    assert type(no_grad("mlx")) is type(nullcontext())   # MLX records no graph
    with no_grad("torch"):
        import torch

        assert not torch.is_grad_enabled()


def test_public_context_rejects_nonsense() -> None:
    with pytest.raises(TypeError, match="Model, a Backend"):
        no_grad(42)

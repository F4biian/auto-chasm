"""Model statistics — architecture dimensions and parameter counts.

Backs the ``Model`` facade's ``hidden_size``/``vocab_size``/``num_parameters``/
``stats`` accessors so the facade stays small.  Everything here reads the wrapped
base model (its ``config``/``args`` and modules) and counts parameters by the
tensor's real backend, so it works on MLX and PyTorch alike.
"""

from __future__ import annotations

from typing import Any

from auto_chasm.probe import _find_embedding, _get_hidden_dim, _get_vocab_size
from auto_chasm.utils import tensor_backend


def hidden_size(base: Any) -> int:
    """The base model's hidden dimension (from its config)."""
    return _get_hidden_dim(base)


def vocab_size(base: Any) -> int:
    """The vocabulary size — from the config, else the embedding's row count."""
    try:
        return _get_vocab_size(base)
    except ValueError:
        module, _ = _find_embedding(base)
        if module is not None:
            num = getattr(module, "num_embeddings", None)  # MLX / torch nn.Embedding
            if num is not None:
                return int(num)
            weight = getattr(module, "weight", None)
            if weight is not None:
                return int(weight.shape[0])
        raise


def _cfg_value(base: Any, names: tuple[str, ...]) -> int | None:
    """First integer config value found under any of ``names`` (else ``None``)."""
    for holder in (getattr(base, "config", None), getattr(base, "args", None)):
        for name in names:
            value = getattr(holder, name, None)
            if value is not None:
                return int(value)
    return None


def _count(module: Any, *, trainable: bool) -> int:
    """Count a module's parameters (all, or only trainable), backend-agnostic."""
    if trainable and hasattr(module, "trainable_parameters"):  # MLX: trainable subset
        params: Any = module.trainable_parameters()
    else:
        params = module.parameters()
    if isinstance(params, dict):  # MLX nn.Module.parameters() -> nested dict
        from mlx.utils import tree_flatten

        arrays = [arr for _, arr in tree_flatten(params)]
    else:  # torch: generator/list of tensors (filter by requires_grad for trainable)
        arrays = [p for p in params if not trainable or getattr(p, "requires_grad", True)]
    return sum(int(a.numel()) if tensor_backend(a) == "torch" else int(a.size) for a in arrays)


def num_parameters(model: Any, *, trainable: bool = False) -> int:
    """Total (or trainable) parameter count of the base model plus every probe head."""
    modules = [model.model, *(probe.module for probe in model._probes.values())]
    return sum(_count(m, trainable=trainable) for m in modules)


def model_stats(model: Any) -> dict[str, Any]:
    """A dict of the model's architecture dimensions and parameter counts.

    Keys: ``backend``, ``num_layers``, ``hidden_size``, ``vocab_size``,
    ``num_attention_heads``, ``intermediate_size`` (the MLP width; ``None`` when the
    config does not expose it), ``num_parameters``, ``num_trainable_parameters``,
    ``num_probes``, and ``probe_parameters`` (a ``{probe_name: count}`` map).
    """
    base = model.model
    return {
        "backend": model.backend.name,
        "num_layers": model.num_layers,
        "hidden_size": hidden_size(base),
        "vocab_size": vocab_size(base),
        "num_attention_heads": _cfg_value(base, ("num_attention_heads", "n_heads", "num_heads")),
        "intermediate_size": _cfg_value(base, ("intermediate_size", "ffn_dim", "n_inner")),
        "num_parameters": num_parameters(model),
        "num_trainable_parameters": num_parameters(model, trainable=True),
        "num_probes": len(model._probes),
        "probe_parameters": {
            name: _count(probe.module, trainable=False) for name, probe in model._probes.items()
        },
    }

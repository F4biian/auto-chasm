"""Backend-agnostic gradient control for inference passes.

A forward pass run for ANALYSIS -- scoring probes, collecting hidden states -- needs
no autograd graph, but PyTorch records one whenever some parameter requires grad, and
a LoRA checkpoint always has some. The retained activations then dwarf the weights:
a 7B in bf16 is ~13.5 GB of weights, while the same analysis pass under autograd was
measured at 46 GB on a 48 GB card, dying in an unrelated matmul.

``model.eval()`` does NOT cover this. It switches dropout and BatchNorm; graph
building is a separate mechanism (measured: under ``eval()`` alone the output still
carries a ``grad_fn`` and the graph still holds its saved tensors).

MLX records no graph implicitly -- gradients come from ``mx.grad`` /
``mx.value_and_grad`` -- so there this context is a no-op and the same caller code
runs unchanged on both backends.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

__all__ = ["no_grad"]


def no_grad(target: Any = None) -> AbstractContextManager[Any]:
    """Context manager that disables gradient tracking on the matching backend.

    Args:
        target: Where to read the backend from -- a :class:`~auto_chasm.Model`, a
            ``Backend``, the string ``"torch"`` or ``"mlx"``, or ``None`` to
            auto-detect. PASS THE MODEL whenever there is one: auto-detection
            prefers MLX on a machine that has both installed, which would silently
            no-op for a torch model.

    Returns:
        The backend's no-grad context (``torch.no_grad()`` on torch, a no-op
        context on MLX).

    Raises:
        TypeError: If ``target`` is not a Model, a Backend, ``"torch"``/``"mlx"``
            or ``None``.

    Example::

        from auto_chasm import no_grad

        with no_grad(model):
            out = model.forward(tokens)
    """
    from auto_chasm.backends.base import Backend

    backend = getattr(target, "backend", target)   # a Model carries its Backend
    if backend is None:
        backend = Backend()
    elif isinstance(backend, str):
        backend = Backend(force=backend)
    if not hasattr(backend, "module"):
        raise TypeError(
            "no_grad() takes a Model, a Backend, 'torch'/'mlx' or None; got "
            f"{type(target).__name__}."
        )
    return backend.module.no_grad()

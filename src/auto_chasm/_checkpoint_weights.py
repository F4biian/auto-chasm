"""Backend-agnostic probe-weight serialization for checkpoints.

Splits the safetensors/torch read-write primitives out of ``checkpoint.py`` so
each file stays within the project's line budget.  All functions keep numpy as
the cross-backend intermediate: weights saved on MLX load on PyTorch and vice
versa.  bfloat16 (which numpy cannot represent) is widened to float32 for the
round-trip and cast back to the probe's dtype on load.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from auto_chasm.logger import get_logger
from auto_chasm.utils import tensor_backend

if TYPE_CHECKING:
    from auto_chasm.backends import Backend
    from auto_chasm.probe import Probe

logger = get_logger(__name__)


def save_probe_weights(probe: Probe, path: Path, backend: Backend) -> None:
    """Save probe module weights in safetensors format (backend-agnostic).

    Args:
        probe: The ``Probe`` instance.
        path: File path to save to.
        backend: The backend instance.
    """
    try:
        if backend.name == "mlx":
            import mlx.core as mx
            from mlx.utils import tree_flatten

            flat = tree_flatten(probe.module.parameters())
            weights = dict(flat)
            mx.save_safetensors(str(path), weights)
        else:
            import torch
            from safetensors.torch import save_file as st_save

            # Preserve the probe's native dtype — safetensors supports fp64, so a
            # blanket ``.float()`` would silently truncate a float64 probe on a
            # path that claims to be lossless. Only cast genuinely-unsupported
            # dtypes (e.g. bfloat16 round-trips fine, but exotic/complex dtypes
            # are coerced to float32 to keep the file readable cross-backend).
            supported = {
                torch.float64,
                torch.float32,
                torch.float16,
                torch.bfloat16,
            }
            state = {}
            for k, v in probe.module.state_dict().items():
                t = v.contiguous().cpu()
                if t.dtype not in supported:
                    t = t.float()
                state[k] = t
            st_save(state, str(path))
    except Exception as e:
        logger.warning("Could not save probe weights for '%s': %s", probe.name, e)


def save_probe_whitening(probe: Probe, path: Path) -> None:
    """Write a probe's whitening transform, if it has one.

    Kept in its own file rather than folded into the probe weights: the loader
    validates those keys against the module's ``state_dict`` and would reject the
    extra entries. Written through NumPy, so an MLX-fitted transform reloads on
    torch and vice versa.

    Args:
        probe: The ``Probe`` instance.
        path: Destination ``*.whitening.safetensors`` path.
    """
    if probe.whitening is None:
        if path.exists():
            path.unlink()  # a refit without whitening must not leave a stale one
        return
    try:
        import numpy as np
        from safetensors.numpy import save_file as np_save

        np_save({k: np.ascontiguousarray(v, dtype=np.float64)
                 for k, v in probe.whitening.items()}, str(path))
    except Exception as e:
        logger.warning("Could not save whitening for '%s': %s", probe.name, e)


def load_probe_whitening(probe: Probe, path: Path) -> None:
    """Restore a probe's whitening transform if one was saved beside it.

    Absence is not an error: only mass-mean probes fitted with ``whiten=True``
    have one.

    Args:
        probe: The ``Probe`` instance.
        path: The ``*.whitening.safetensors`` path.
    """
    if not path.exists():
        return
    try:
        from safetensors.numpy import load_file as np_load

        probe.whitening = dict(np_load(str(path)))
    except Exception as e:
        logger.warning("Could not load whitening for '%s': %s", probe.name, e)


def _expected_mlx_shapes(probe: Probe) -> dict[str, tuple[int, ...]]:
    """Return the probe module's parameter shapes as a flat ``{key: shape}`` map.

    Args:
        probe: The ``Probe`` whose MLX module to inspect.

    Returns:
        Flat mapping from dotted parameter name to its current shape.
    """
    from mlx.utils import tree_flatten

    return {k: tuple(v.shape) for k, v in tree_flatten(probe.module.parameters())}


def _validate_shapes_against_probe(
    probe: Probe,
    on_disk: dict[str, tuple[int, ...]],
    expected: dict[str, tuple[int, ...]],
) -> None:
    """Raise if any on-disk tensor's shape differs from the probe's config.

    Catches the silent-reshape footgun: MLX ``module.update`` replaces arrays
    without a shape check, so a config/weights mismatch would otherwise mutate
    the probe into a head that contradicts its own config and the manifest.

    Args:
        probe: The probe being loaded into (named in the error).
        on_disk: ``{key: shape}`` of the tensors read from disk.
        expected: ``{key: shape}`` of the probe module's current parameters.

    Raises:
        ValueError: If a shared key has mismatched shapes, or the key sets
            differ (so the on-disk weights cannot populate this probe).
    """
    if set(on_disk) != set(expected):
        raise ValueError(
            f"Probe '{probe.name}' weight keys do not match the checkpoint: "
            f"probe has {sorted(expected)}, on-disk file has {sorted(on_disk)}. "
            f"The checkpoint does not correspond to this probe configuration."
        )
    mismatches = {k: (expected[k], on_disk[k]) for k in expected if expected[k] != on_disk[k]}
    if mismatches:
        detail = "; ".join(
            f"{k}: expected {exp}, on-disk {got}" for k, (exp, got) in mismatches.items()
        )
        raise ValueError(
            f"Probe '{probe.name}' weight shape mismatch ({detail}). Refusing to "
            f"silently reshape the probe to match incompatible on-disk weights."
        )


def safetensors_framework() -> str:
    """Return the safetensors read framework for the installed backend.

    Returns:
        ``"pt"`` when PyTorch is importable, else ``"mlx"``.
    """
    try:
        import torch  # noqa: F401

        return "pt"
    except Exception:
        return "mlx"


def _tensor_to_numpy(tensor: Any) -> Any:
    """Convert a torch/mlx tensor to numpy, upcasting bfloat16 losslessly.

    numpy has no bfloat16 dtype, so reading safetensors with ``framework="np"``
    raises on a bf16 tensor — the reason this goes through a real framework
    first.  bf16 is widened to float32 (a lossless superset) before the numpy
    hand-off; the loader casts back to the probe's dtype.

    Args:
        tensor: A ``torch.Tensor`` or ``mlx.core.array``.

    Returns:
        The values as a numpy array.
    """
    import numpy as np

    if tensor_backend(tensor) == "torch":
        import torch

        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.detach().cpu().numpy()

    import mlx.core as mx

    if tensor.dtype == mx.bfloat16:
        tensor = tensor.astype(mx.float32)
    return np.array(tensor)


def read_probe_weights_numpy(path: Path) -> dict[str, Any]:
    """Read probe weights from disk into a backend-neutral ``{key: ndarray}`` map.

    Detects the on-disk format by *content*, not extension: ``save_probe_weights``
    always writes safetensors regardless of the chosen filename, so dispatching on
    the suffix alone would misread a safetensors blob named ``*.pth``.  Safetensors
    is tried first (through the installed framework, so bf16 is handled); anything
    else falls back to ``torch.load`` (legacy ``.pth``/pickle).  Reading into numpy
    lets either backend consume weights saved by the other.

    Args:
        path: File path to read.

    Returns:
        Flat mapping from parameter name to its numpy array.
    """
    try:
        from safetensors import safe_open

        weights: dict[str, Any] = {}
        with safe_open(str(path), framework=safetensors_framework()) as f:
            for key in f.keys():  # noqa: SIM118  # safe_open is not a dict
                weights[key] = _tensor_to_numpy(f.get_tensor(key))
        return weights
    except Exception:
        # Not a safetensors file — fall back to a torch pickle.
        import torch

        state_dict = torch.load(str(path), map_location="cpu", weights_only=False)
        return {k: _tensor_to_numpy(v) for k, v in state_dict.items()}


def load_probe_weights(probe: Probe, path: Path, backend: Backend) -> None:
    """Load probe module weights (backend-agnostic; safetensors or torch ``.pth``).

    Weights are read into numpy first, validated against the probe's parameter
    shapes, then applied to the active backend — so a checkpoint saved on one
    backend loads cleanly into the other.

    Args:
        probe: The ``Probe`` instance.
        path: File path to load from.
        backend: The backend instance.

    Raises:
        FileNotFoundError: If the probe-weights file is missing — a checkpoint
            with no weights would otherwise "load" into an untrained (random)
            probe with no signal (research-poisoning footgun).
        ValueError: If the on-disk weights cannot populate this probe (shape or
            key mismatch against its config).
        Exception: Propagated if the underlying load/decode fails, so a corrupt
            checkpoint fails loudly instead of leaving an untrained probe.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Probe weights not found for '{probe.name}': {path}. The checkpoint is "
            f"incomplete; refusing to return a probe with random, untrained weights."
        )

    import numpy as np

    weights = read_probe_weights_numpy(path)
    on_disk = {k: tuple(np.asarray(v).shape) for k, v in weights.items()}

    if backend.name == "mlx":
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten

        _validate_shapes_against_probe(probe, on_disk, _expected_mlx_shapes(probe))
        # Cast each array back to the probe param's current dtype so a bf16 probe
        # stays bf16 — the numpy round-trip upcast bf16 to float32 (numpy has no
        # bf16), and MLX's update() replaces arrays outright without re-casting.
        current = dict(tree_flatten(probe.module.parameters()))
        restored = []
        for k, v in weights.items():
            arr = mx.array(v)
            if k in current:
                arr = arr.astype(current[k].dtype)
            restored.append((k, arr))
        # Un-flatten dotted safetensors keys (``norm.bias``, ``layers.0.weight``)
        # into the nested tree update() expects; flat keys fail for submodule heads.
        probe.module.update(tree_unflatten(restored))
    else:
        import torch

        expected = {k: tuple(v.shape) for k, v in probe.module.state_dict().items()}
        _validate_shapes_against_probe(probe, on_disk, expected)
        torch_sd = {k: torch.tensor(np.array(v)) for k, v in weights.items()}
        try:
            target_device = next(probe.module.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
        probe.module.load_state_dict(torch_sd)
        probe.module = probe.module.to(target_device)

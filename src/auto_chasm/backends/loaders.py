"""Model loading helpers — backend-specific model + tokenizer loading.

Keeps ``model.py`` free of direct framework imports.
"""

from __future__ import annotations

from typing import Any, Literal, cast


def detect_backend() -> Literal["mlx", "torch"]:
    """Auto-detect the available backend.

    Returns:
        ``"mlx"`` on macOS with mlx installed, ``"torch"`` otherwise.

    Raises:
        RuntimeError: If neither backend is available.
    """
    try:
        import mlx.core  # noqa: F401

        return "mlx"
    except ImportError:
        pass
    # Only testable when torch is installed (pragma: no cover)
    try:  # pragma: no cover
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        pass
    raise RuntimeError("No supported backend found. Install 'mlx' or 'torch'.")


def resolve_backend_name(backend_name: str | None) -> Literal["mlx", "torch"]:
    """Validate a backend name up front, auto-detecting when ``None``.

    Validating BEFORE a model is loaded turns a typo (``"pytorch"``, ``"MLX"``) into
    an immediate clear error, instead of silently loading the entire model and then
    failing with a misleading "No supported backend found".

    Args:
        backend_name: ``"mlx"``, ``"torch"``, or ``None`` (auto-detect).

    Returns:
        The validated backend name.

    Raises:
        ValueError: If ``backend_name`` is neither ``"mlx"`` nor ``"torch"``.
    """
    if backend_name is None:
        return detect_backend()
    if backend_name not in ("mlx", "torch"):
        raise ValueError(
            f"Unknown backend_name {backend_name!r}. Use 'mlx' or 'torch' (or None to auto-detect)."
        )
    return cast(Literal["mlx", "torch"], backend_name)


def load_mlx(model_name: str, **kwargs: Any) -> tuple[Any, Any]:
    """Load a model and tokenizer via mlx-lm.

    Args:
        model_name: Model name or path.
        **kwargs: Additional arguments passed to ``mlx_lm.load``.

    Returns:
        Tuple of ``(model, tokenizer)``.
    """
    from mlx_lm import load

    kwargs.pop("dtype", None)  # mlx-community ports are already bf16; mlx_lm takes no torch dtype
    result = load(model_name, **kwargs)
    return result[0], result[1]


def load_torch(model_name: str, **kwargs: Any) -> tuple[Any, Any]:  # pragma: no cover
    """Load a model and tokenizer via transformers.

    Args:
        model_name: Model name or path.
        **kwargs: Additional arguments passed to the model loader.

    Returns:
        Tuple of ``(model, tokenizer)``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Optional load dtype (e.g. "bfloat16"): applies to the MODEL only — torch_dtype is
    # not a valid tokenizer arg — so pop it out of the shared kwargs first. None -> the
    # transformers default (fp32). Probing experiments pass bf16 to control the geometry.
    dtype = kwargs.pop("dtype", None)
    model_kwargs = dict(kwargs)
    if dtype is not None:
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)  # "bfloat16" -> torch.bfloat16
        model_kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except ValueError as exc:
        # MLX-format checkpoints (e.g. ``mlx-community/*-8bit``) carry an MLX
        # ``quantization`` block that transformers cannot read — it raises a
        # cryptic "no `quant_method` attribute". Translate it into actionable
        # guidance: the torch backend needs a torch-native model.
        if "quant_method" in str(exc) or "quantization" in str(exc).lower():
            raise RuntimeError(
                f"'{model_name}' looks like an MLX-format (mlx-community) checkpoint, which "
                "the PyTorch/transformers backend cannot load — its quantization metadata is "
                "MLX-specific.\n"
                "On the torch backend use a torch-native model instead, e.g. "
                "'HuggingFaceTB/SmolLM2-135M' (base) or 'google/gemma-3-270m-it' (instruct), "
                "or load the full-precision/torch-quantized variant of this model.\n"
                f"(original transformers error: {exc})"
            ) from exc
        raise
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer

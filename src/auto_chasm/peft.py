"""PEFT integration — thin wrapper around backend adapter operations.

Provides a unified interface for applying LoRA, QLoRA, and DoRA
regardless of whether the backend is MLX or PyTorch.
"""

from __future__ import annotations

import re
from typing import Any

from auto_chasm._mlx_compat import ensure_mlx_lm_compat
from auto_chasm.logger import get_logger

logger = get_logger(__name__)

# Default LoRA target keys — full paths matching mlx_lm conventions
DEFAULT_LORA_KEYS = ["self_attn.q_proj", "self_attn.v_proj"]


def _validate_lora_hparams(r: int, alpha: int) -> None:
    """Validate LoRA rank and alpha before any backend work.

    Catches footguns that would otherwise surface as cryptic low-level errors:
    ``r=0`` (``ZeroDivisionError`` from ``alpha / r``), ``r<0`` (``Negative
    dimensions not allowed`` from allocating a rank-``r`` factor), and ``alpha=0``
    (scale ``0/r == 0`` makes every adapter a permanent silent no-op).

    Args:
        r: LoRA rank.
        alpha: LoRA alpha scaling.

    Raises:
        ValueError: If ``r`` is not a positive integer.
    """
    if r < 1:
        raise ValueError(
            f"LoRA rank must be a positive integer (got r={r}); rank 0 or negative "
            f"is meaningless (no low-rank bottleneck)."
        )
    if alpha == 0:
        logger.warning(
            "LoRA alpha=0 yields scale=alpha/rank=0, so the adapters are permanent "
            "no-ops: training them cannot change the output. Use a non-zero alpha "
            "(commonly alpha == rank or 2*rank)."
        )


def _extract_layer_index(name: str) -> int | None:
    """Extract layer index from a module name like ``model.layers.3.self_attn.q_proj``.

    Args:
        name: The fully qualified module name.

    Returns:
        The layer index, or ``None`` if it cannot be determined.
    """
    for pattern in (
        r"(?:^|\.)layers\.(\d+)\.",
        r"(?:^|\.)h\.(\d+)\.",
        r"(?:^|\.)blocks\.(\d+)\.",
    ):
        m = re.search(pattern, name)
        if m:
            return int(m.group(1))
    return None


def _filter_lora_targets(
    target_modules: list[str],
    target_layers: list[int] | None = None,
    until_layer: int | None = None,
    after_layer: int | None = None,
) -> list[str]:
    """Filter LoRA target module names by layer index.

    Args:
        target_modules: Full list of candidate module names.
        target_layers: Only include modules in these layer indices.
        until_layer: Only include modules with layer index < this value.
        after_layer: Only include modules with layer index >= this value.

    Returns:
        Filtered list of module names.
    """
    if target_layers is None and until_layer is None and after_layer is None:
        return target_modules

    result: list[str] = []
    for name in target_modules:
        idx = _extract_layer_index(name)
        if idx is None:
            if target_layers is not None and len(target_layers) == 0:
                continue
            # Can't determine layer — include it by default
            result.append(name)
            continue
        if target_layers is not None and idx not in target_layers:
            continue
        if until_layer is not None and idx >= until_layer:
            continue
        if after_layer is not None and idx < after_layer:
            continue
        result.append(name)
    return result


# Linear-like leaf module class names LoRA can wrap, framework-agnostic:
# torch nn.Linear / GPT-2-style Conv1D, MLX nn.Linear / nn.QuantizedLinear.
_LINEAR_CLASS_NAMES = ("Linear", "QuantizedLinear", "Conv1D")


def targetable_lora_modules(model: Any) -> list[str]:
    """Every module LoRA can adapt in ``model`` — full dotted paths, in model order.

    "Targetable" means a linear-like LEAF module (``Linear``/``QuantizedLinear``/
    ``Conv1D``), excluding:

    - the LM head (any path whose last component is ``lm_head``) — adapting the
      output embedding is excluded by convention (mirrors PEFT's
      ``target_modules="all-linear"``), and a tied head would double-adapt the
      input embedding;
    - modules that are already LoRA internals (path contains ``lora``), so
      listing an adapted model describes the BASE architecture, not the
      adapters.

    This is also the DEFAULT target set when ``LoraConfig.target_modules`` is
    ``None`` — i.e. by default LoRA adapts everything it can. Exposed as
    ``Model.lora_targetable_modules`` and in ``Model.stats()``.

    Args:
        model: The (raw) language model.

    Returns:
        List of full module paths, order-stable, possibly empty for a model
        that exposes no ``named_modules``.
    """
    targets: list[str] = []
    adapter_prefixes: list[str] = []
    try:
        for name, mod in model.named_modules():
            if not name:
                continue
            cls = type(mod).__name__.lower()
            # An adapter wrapper (LoRALinear/DoRALinear/peft LoraLayer...) IS the
            # canonical linear position — report it once and skip its internals
            # (its inner base .linear and its lora_A/lora_B would otherwise be
            # listed as three bogus extra targets).
            if "lora" in cls or "dora" in cls:
                adapter_prefixes.append(name + ".")
                if name.rsplit(".", 1)[-1] != "lm_head":
                    targets.append(name)
                continue
            if any(name.startswith(pre) for pre in adapter_prefixes):
                continue
            if type(mod).__name__ not in _LINEAR_CLASS_NAMES:
                continue
            last = name.rsplit(".", 1)[-1]
            if last == "lm_head" or "lora" in name.lower():
                continue
            targets.append(name)
    except Exception:
        pass
    return targets


def _default_target_modules(model: Any) -> list[str]:
    """The default LoRA target set: EVERY adaptable linear module.

    ``target_modules=None`` means "adapt everything you can" — all linear-like
    leaf modules except the LM head (see :func:`targetable_lora_modules`), not
    merely the attention q/k/v projections. Callers who want a narrower scope
    pass it explicitly.

    Args:
        model: The language model.

    Returns:
        List of module name keys to target.
    """
    targets = targetable_lora_modules(model)
    if not targets:
        targets = list(DEFAULT_LORA_KEYS)
        logger.warning("Could not infer target modules; using defaults: %s", targets)
    return targets


def _model_module_names(model: Any) -> list[str]:
    """Return the model's fully qualified module names, or an empty list.

    Args:
        model: The language model.

    Returns:
        List of dotted module paths from ``model.named_modules()``, or an
        empty list if the model does not expose ``named_modules``.
    """
    try:
        return [name for name, _ in model.named_modules() if name]
    except Exception:
        return []


def _name_matches_target(name: str, target: str) -> bool:
    """Check whether a full module name matches a user-supplied target.

    Matching is suffix/component based (the HuggingFace/PEFT convention):
    a target matches a name if the name equals the target, ends with
    ``.<target>``, or — when the target itself contains dots — the dotted
    target appears as a trailing path component sequence.

    Args:
        name: Fully qualified module name (e.g. ``layers.0.self_attn.q_proj``).
        target: User-supplied target (e.g. ``q_proj`` or ``self_attn.q_proj``).

    Returns:
        ``True`` if the name matches the target.
    """
    return name == target or name.endswith("." + target)


def _resolve_target_modules(model: Any, target_modules: list[str]) -> list[str]:
    """Expand user target modules to the model's concrete full module paths.

    Resolves short names such as ``"q_proj"`` to every matching full module
    path (e.g. ``"layers.0.self_attn.q_proj"``) so that the same request wraps
    the same modules on both backends — MLX's ``linear_to_lora_layers`` matches
    full paths by exact equality, and PyTorch/PEFT suffix-matches them too.

    If the model exposes no inspectable modules, the request is passed through
    unchanged (the backend performs its own matching).

    Args:
        model: The language model to inspect.
        target_modules: User-supplied module names (short or full).

    Returns:
        Concrete full module paths to target, de-duplicated and order-stable.

    Raises:
        ValueError: If the model exposes modules but none match the request.
    """
    module_names = _model_module_names(model)
    if not module_names:
        return target_modules

    resolved: list[str] = []
    seen: set[str] = set()
    unmatched: list[str] = []
    for target in target_modules:
        matched = [n for n in module_names if _name_matches_target(n, target)]
        if not matched:
            unmatched.append(target)
            continue
        for n in matched:
            if n not in seen:
                seen.add(n)
                resolved.append(n)

    if not resolved:
        raise ValueError(
            f"target_modules={target_modules!r} matched no modules in this model "
            f"(inspected {len(module_names)} named modules). Nothing would be "
            f"adapted. Check the names against model.named_modules() — common "
            f"choices are 'q_proj', 'v_proj', or 'self_attn.q_proj'."
        )
    if unmatched:
        logger.warning(
            "Some target_modules matched no modules and were skipped: %s. "
            "Adapting %d module(s) from the rest.",
            unmatched,
            len(resolved),
        )
    return resolved


def get_num_layers(model: Any) -> int:
    """Return the number of transformer layers in a model.

    Args:
        model: The base language model.

    Returns:
        Number of transformer layers.

    Raises:
        ValueError: If the number of layers cannot be determined.
    """
    from auto_chasm.probe import _find_layers

    layers = _find_layers(model)
    if layers is None:
        raise ValueError("Cannot determine the number of layers in this model.")
    return len(layers)


def apply_lora(
    model: Any,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    target_layers: list[int] | None = None,
    until_layer: int | None = None,
    after_layer: int | None = None,
    backend: Any = None,
) -> Any:
    """Apply LoRA adapters to a model.

    Args:
        model: The base language model.
        r: LoRA rank.
        alpha: LoRA alpha scaling.
        dropout: LoRA dropout.
        target_modules: Module names to apply LoRA to (full paths like
            ``"self_attn.q_proj"``).  ``None`` uses sensible defaults.
        target_layers: Only apply LoRA to these specific layer indices.
        until_layer: Only apply LoRA to layers with index < this value.
        after_layer: Only apply LoRA to layers with index >= this value.
        backend: The backend instance.

    Returns:
        Model with LoRA applied.

    Raises:
        ValueError: If ``r`` is not a positive integer, or if targeting filters
            out every module.
    """
    _validate_lora_hparams(r, alpha)

    if backend is None:
        from auto_chasm.backends import Backend

        backend = Backend()

    if target_modules is None:
        target_modules = _default_target_modules(model)

    target_modules = _resolve_target_modules(model, target_modules)

    target_modules = _filter_lora_targets(
        target_modules,
        target_layers=target_layers,
        until_layer=until_layer,
        after_layer=after_layer,
    )
    if not target_modules:
        raise ValueError(
            "Layer targeting (target_layers/until_layer/after_layer) filtered out "
            "every target module — nothing would be adapted. Widen the layer range."
        )

    adapter_config = {"r": r, "alpha": alpha, "dropout": dropout, "num_layers": -1}
    logger.info(
        "Applying LoRA (r=%d, alpha=%d, targets=%s)",
        r,
        alpha,
        target_modules,
    )
    return backend.wrapping.apply_adapters(model, adapter_config, target_modules)


def apply_qlora(
    model: Any,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    target_layers: list[int] | None = None,
    until_layer: int | None = None,
    after_layer: int | None = None,
    bits: int = 4,
    group_size: int = 64,
    backend: Any = None,
) -> Any:
    """Apply true QLoRA — quantize the base model, then LoRA on the frozen base.

    On **MLX** the base linear layers are quantized in place to ``bits``-bit
    (skipping layers already quantized or whose width is not a multiple of
    ``group_size``), then LoRA adapters are applied on top of the quantized base.

    On **PyTorch**, 4-bit (nf4) quantization is a *load-time* concern that needs
    ``bitsandbytes`` + CUDA (``load_in_4bit=True``). Post-hoc quantization of an
    already-loaded fp model is not done here, so this requires a base that is
    already 4-bit-loaded; otherwise it raises (rather than silently applying
    plain LoRA and calling it QLoRA).

    Args:
        model: The base language model.
        r: LoRA rank.
        alpha: LoRA alpha scaling.
        dropout: LoRA dropout.
        target_modules: Module names to apply LoRA to.  ``None`` uses defaults.
        target_layers: Only apply LoRA to these specific layer indices.
        until_layer: Only apply LoRA to layers with index < this value.
        after_layer: Only apply LoRA to layers with index >= this value.
        bits: Quantization bit-width for the base (MLX). Default ``4``.
        group_size: Quantization group size (MLX). Default ``64``.
        backend: The backend instance.

    Returns:
        Model with the base quantized and LoRA applied.

    Raises:
        NotImplementedError: On torch when the base is not already 4-bit-loaded.
    """
    if backend is None:
        from auto_chasm.backends import Backend

        backend = Backend()

    if backend.name == "mlx":
        _quantize_mlx_base(model, bits=bits, group_size=group_size)
        logger.info(
            "QLoRA: quantized base to %d-bit (group_size=%d); applying LoRA (r=%d).",
            bits,
            group_size,
            r,
        )
    else:
        _require_torch_4bit_base(model)

    return apply_lora(
        model,
        r,
        alpha,
        dropout,
        target_modules,
        target_layers=target_layers,
        until_layer=until_layer,
        after_layer=after_layer,
        backend=backend,
    )


def _quantize_mlx_base(model: Any, *, bits: int, group_size: int) -> None:
    """Quantize an MLX base model in place, verifying that something was quantized.

    ``nn.quantize`` only quantizes ``Linear``/``Embedding`` layers whose last
    weight dimension is a multiple of ``group_size``; layers that don't divide
    are silently skipped. If *no* layer qualifies, ``nn.quantize`` is a no-op and
    the caller would proceed to wrap full-precision Linears with LoRA while
    believing the base is quantized. This helper detects that and raises a clear
    error naming ``group_size`` and the offending feature widths instead.

    Args:
        model: The MLX model to quantize (modified in place).
        bits: Quantization bit-width.
        group_size: Quantization group size; the per-layer input width must be a
            multiple of this for the layer to be quantizable.

    Raises:
        ValueError: If no candidate layer is a multiple of ``group_size`` (so
            nothing would be quantized).
    """
    import mlx.nn as nn

    quantizable_types = (nn.Linear, nn.Embedding)
    candidate_widths: set[int] = set()
    has_quantizable = False
    for _name, module in model.named_modules():
        if isinstance(module, quantizable_types) and not isinstance(
            module, (nn.QuantizedLinear, nn.QuantizedEmbedding)
        ):
            width = int(module.weight.shape[-1])
            candidate_widths.add(width)
            if width % group_size == 0:
                has_quantizable = True

    if not has_quantizable:
        widths = ", ".join(str(w) for w in sorted(candidate_widths)) or "<none>"
        raise ValueError(
            f"QLoRA: nothing to quantize — no base layer has an input width "
            f"divisible by group_size={group_size} (candidate widths: {widths}). "
            f"nn.quantize would be a silent no-op, leaving a full-precision base. "
            f"Choose a group_size that divides the model width (MLX supports "
            f"32, 64, 128), or load an already-quantized base."
        )

    def _quantizable(_path: str, module: Any) -> bool:
        return (
            isinstance(module, quantizable_types) and int(module.weight.shape[-1]) % group_size == 0
        )

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=_quantizable)

    n_quantized = sum(
        1
        for _name, module in model.named_modules()
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    )
    if n_quantized == 0:
        widths = ", ".join(str(w) for w in sorted(candidate_widths)) or "<none>"
        raise ValueError(
            f"QLoRA: nn.quantize produced zero QuantizedLinear/QuantizedEmbedding "
            f"layers with group_size={group_size} (candidate widths: {widths}); the "
            f"base remains full-precision. Choose a compatible group_size (MLX "
            f"supports 32, 64, 128) or load an already-quantized base."
        )


def _require_torch_4bit_base(model: Any) -> None:
    """Raise unless a torch model is already loaded in 4-bit (bitsandbytes).

    Args:
        model: The torch base model.

    Raises:
        NotImplementedError: If the base is not 4-bit-loaded.
    """
    is_4bit = getattr(model, "is_loaded_in_4bit", False) or any(
        "4bit" in type(m).__name__.lower() for _, m in model.named_modules()
    )
    if not is_4bit:
        raise NotImplementedError(
            "torch QLoRA requires a base loaded in 4-bit (bitsandbytes "
            "load_in_4bit=True, CUDA). Post-hoc 4-bit quantization of an already-"
            "loaded fp model is not performed here. Load the base 4-bit, then call "
            "apply_qlora (LoRA on the quantized base). On MLX, apply_qlora quantizes "
            "for you."
        )


def apply_dora(
    model: Any,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    target_layers: list[int] | None = None,
    until_layer: int | None = None,
    after_layer: int | None = None,
    backend: Any = None,
) -> Any:
    """Apply Weight-Decomposed LoRA (DoRA) adapters to a model.

    DoRA decomposes pre-trained weights into magnitude and direction
    components and applies LoRA to the directional component.

    DoRA is native on both backends: PyTorch via ``peft``'s
    ``LoraConfig(use_dora=True)``, and MLX via ``mlx_lm``'s ``DoRALinear``
    (``linear_to_lora_layers(..., use_dora=True)``).

    Args:
        model: The base language model.
        r: LoRA rank.
        alpha: LoRA alpha scaling.
        dropout: LoRA dropout.
        target_modules: Module names to apply LoRA to.  ``None`` uses
            sensible defaults.
        target_layers: Only apply LoRA to these specific layer indices.
        until_layer: Only apply LoRA to layers with index < this value.
        after_layer: Only apply LoRA to layers with index >= this value.
        backend: The backend instance.

    Returns:
        Model with DoRA applied.

    Raises:
        ValueError: If ``r`` is not a positive integer, or if targeting filters
            out every module.
    """
    _validate_lora_hparams(r, alpha)

    if backend is None:
        from auto_chasm.backends import Backend

        backend = Backend()

    if target_modules is None:
        target_modules = _default_target_modules(model)

    target_modules = _resolve_target_modules(model, target_modules)
    target_modules = _filter_lora_targets(
        target_modules,
        target_layers=target_layers,
        until_layer=until_layer,
        after_layer=after_layer,
    )
    if not target_modules:
        raise ValueError(
            "Layer targeting (target_layers/until_layer/after_layer) filtered out "
            "every target module — nothing would be adapted. Widen the layer range."
        )

    adapter_config = {"r": r, "alpha": alpha, "dropout": dropout, "num_layers": -1}
    logger.info(
        "Applying DoRA (r=%d, alpha=%d, targets=%s)",
        r,
        alpha,
        target_modules,
    )
    return backend.wrapping.apply_adapters(model, adapter_config, target_modules, method="dora")


def get_trainable_params(model: Any, backend: Any) -> list[Any]:
    """Get trainable parameters after adapter application.

    Args:
        model: The model with adapters applied.
        backend: The backend instance.

    Returns:
        List of trainable parameter tensors.
    """
    return backend.wrapping.get_trainable_params(model)  # type: ignore[no-any-return]


def _unfreeze_lora_params(model: Any, backend: Any) -> None:
    """Unfreeze only the adapter parameters after a full freeze.

    Unfreezes the low-rank factors (``lora_a``/``lora_b``) plus DoRA's magnitude
    (``m``) inside every adapter module — LoRA, switch-LoRA, and DoRA (linear and
    embedding). DoRA modules were previously skipped entirely, so their adapters
    (and the magnitude vector) never trained.

    Args:
        model: The model with LoRA/DoRA adapters applied.
        backend: The backend instance.
    """
    if backend.name == "mlx":
        ensure_mlx_lm_compat()
        try:
            from mlx_lm.tuner.lora import LoRALinear, LoRASwitchLinear

            adapter_types: tuple[type, ...] = (LoRALinear, LoRASwitchLinear)
            try:
                from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear

                adapter_types += (DoRALinear, DoRAEmbedding)
            except ImportError:
                pass
            for _, module in model.named_modules():
                if isinstance(module, adapter_types):
                    mod: Any = module
                    mod.freeze()
                    mod.unfreeze(keys=["lora_a", "lora_b", "m"])  # 'm' = DoRA magnitude
        except (ImportError, AttributeError):
            model.unfreeze()
    else:
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

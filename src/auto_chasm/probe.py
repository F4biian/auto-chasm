"""Probe injection engine — attach heads to transformer layers.

The capture wrapper is an ``nn.Module`` so wrapped layer parameters stay in
the module tree.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from auto_chasm.config import ProbeConfig
from auto_chasm.logger import get_logger

try:
    import mlx.nn as nn

    _MLX = True
except ImportError:
    _MLX = False
    try:
        import torch.nn as nn
    except ImportError:
        # Neither backend is installed. ``import auto_chasm`` must still succeed —
        # a backend is required only to USE a model, at which point
        # ``Model.from_pretrained`` raises a clear "install auto-chasm[mlx] or
        # auto-chasm[torch]" error. The capture base falls back to ``object``.
        nn = None

logger = get_logger(__name__)


def _find_layers(model: Any) -> Any:
    """Find the transformer block list in a model (or ``None``)."""
    for attr in (
        "layers",
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "transformer.blocks",
    ):
        parts = attr.split(".")
        obj = model
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                break
        else:
            if isinstance(obj, (list, tuple)) or (
                hasattr(obj, "__len__") and hasattr(obj, "__getitem__")
            ):
                return obj
    return None


_HIDDEN_DIM_ATTRS = ("hidden_size", "d_model", "n_embd", "dim")


#: Sub-model attributes a wrapper architecture hides the real model behind, in
#: search order. ``""`` is the model itself, so an unwrapped model is unaffected.
_SUBMODEL_PATHS: tuple[str, ...] = ("", "language_model", "model", "model.model",
                                    "language_model.model")


def _resolve_path(model: Any, path: str) -> Any | None:
    """Walk a dotted attribute path; ``None`` if any component is missing."""
    obj = model
    for part in filter(None, path.split(".")):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def config_holders(model: Any) -> list[Any]:
    """Every config-like object on a model, wrapper sub-models included.

    Checking only ``model.config`` and ``model.args`` misses shells that hold a
    sub-model: Qwen3.5's MLX build is a multimodal-style wrapper whose ``args``
    carries just ``{model_type, text_config}``, with the real dimensions at
    ``model.language_model.args``. Reading a dimension off such a model returned
    nothing -- ``hidden_size`` raised, and ``num_attention_heads`` silently
    reported ``None``.

    Ordered OUTERMOST FIRST, so a wrapper that does expose a value still wins and
    behaviour on unwrapped models is unchanged.
    """
    seen: list[Any] = []
    for path in _SUBMODEL_PATHS:
        holder = _resolve_path(model, path)
        if holder is None:
            continue
        for cfg in (getattr(holder, "config", None), getattr(holder, "args", None)):
            if cfg is None:
                continue
            seen.append(cfg)
            sub = getattr(cfg, "text_config", None)
            if sub is not None:
                seen.append(sub)
    return seen


def _get_hidden_dim(model: Any) -> int:
    """Infer hidden dimension from a model's config (raises if unknown)."""
    for cfg in config_holders(model):
        for attr in _HIDDEN_DIM_ATTRS:
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)

    raise ValueError(
        "Cannot determine hidden dimension. Pass module_config={'in_features': N} to ProbeConfig."
    )


def _get_vocab_size(model: Any) -> int:
    """Infer vocabulary size from a model's config (raises if unknown)."""
    for cfg in config_holders(model):
        for attr in ("vocab_size", "n_vocab", "vocabulary_size", "num_embeddings"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)

    raise ValueError(
        "Cannot determine vocabulary size: no config on this model or its sub-models "
        f"exposes vocab_size/n_vocab (searched {len(config_holders(model))} config objects). "
        "Pass module_config={'out_features': N} to ProbeConfig to set it explicitly."
    )


def _find_in_submodels(model: Any, names: tuple[str, ...]) -> tuple[Any | None, str | None]:
    """First ``(module, dotted_path)`` matching any of ``names``, wrappers included.

    Tries every name at the model itself before descending, so an unwrapped model
    resolves to the same short path it always did; the returned path is dotted and
    is resolved generically by ``_set_module_by_path``, so deeper hits restore
    correctly.
    """
    for path in _SUBMODEL_PATHS:
        holder = _resolve_path(model, path)
        if holder is None:
            continue
        for name in names:
            found = getattr(holder, name, None)
            if found is not None:
                return found, f"{path}.{name}" if path else name
    return None, None


def _find_embedding(model: Any) -> tuple[Any | None, str | None]:
    """Find the embedding module in a model.

    Args:
        model: The base language model.

    Returns:
        Tuple ``(module, attr_path)`` (dotted path, e.g. ``"embed_tokens"``),
        or ``(None, None)`` if not found.
    """
    # Searched across WRAPPER sub-models too: ``model.embed_tokens`` for a plain
    # HF causal-LM, ``model.model.embed_tokens`` once a torch PeftModel wrapper
    # (get_peft_model) nests the base one level deeper — without the deeper path
    # an embedding-source probe cannot attach after LoRA, which broke
    # Model.from_checkpoint reloads (probes are restored after adapters) — and
    # ``language_model.model.embed_tokens`` for Qwen3.5's multimodal-style shell,
    # where the plain search returned (None, None) and probing failed outright.
    return _find_in_submodels(model, ("embedding", "embed_tokens", "wte"))


def _find_output_head(model: Any) -> tuple[Any | None, str | None]:
    """Find the output projection / LM head module in a model.

    Args:
        model: The base language model.

    Returns:
        Tuple ``(module, attr_path)`` (dotted path, e.g. ``"lm_head"``), or
        ``(None, None)`` if not found.
    """
    return _find_in_submodels(model, ("output_proj", "lm_head", "output", "head"))


# Attribute names a transformer block uses for its attention / MLP submodule,
# across common architectures (Llama/Gemma/Qwen, GPT-2, etc.).
_SUBMODULE_ATTRS: dict[str, tuple[str, ...]] = {
    "attention": ("self_attn", "attn", "attention", "self_attention"),
    "mlp": ("mlp", "feed_forward", "ffn", "feedforward", "mlp_block"),
}


def _find_named_child(block: Any, names: tuple[str, ...]) -> tuple[Any | None, str | None]:
    """Return the first ``(submodule, attr_name)`` on ``block`` matching ``names``."""
    for name in names:
        if hasattr(block, name):
            return getattr(block, name), name
    return None, None


def _set_module_by_path(model: Any, path: str, new_module: Any) -> None:
    """Install ``new_module`` at a dotted attribute path on ``model``."""
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)


def _resolve_negative_index(idx: int, num_layers: int) -> int:
    """Convert a (possibly negative) layer index to a positive one."""
    if idx < 0:
        idx = num_layers + idx
    if idx < 0 or idx >= num_layers:
        raise ValueError(f"Layer index {idx} out of range for model with {num_layers} layers.")
    return idx


# LayerCapture — backend-aware factory. MLX and PyTorch each need a base
# extending their own nn.Module, so two private subclasses exist and
# ``make_layer_capture`` picks the right one at call time.


def _forward_impl(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Core forward logic — delegates to original layer and optionally steers."""
    # ``residual`` source captures the INPUT (the residual stream entering the
    # block) instead of the output; everything else captures the output.
    capture_input = getattr(self, "capture_input", False)
    if capture_input and self.capture_fn is not None and args:
        self.capture_fn(args[0])

    output = self.layer(*args, **kwargs)

    hidden = output[0] if isinstance(output, tuple) else output

    if not capture_input and self.capture_fn is not None:
        self.capture_fn(hidden)

    if self.steer_fn is not None and self.binary_head is not None:
        from auto_chasm.utils import tensor_backend

        try:
            if tensor_backend(hidden) == "torch":
                # Cast around the head (it may be fp32 while the model runs bf16).
                orig_dtype = hidden.dtype
                head_dtype = next(self.binary_head.parameters()).dtype
                h = hidden.to(head_dtype)
                logits = self.binary_head(h).squeeze(-1)
                hidden = self.steer_fn(h, self.binary_head, logits).to(orig_dtype)
            else:
                # Mirror the torch branch: cast around the (possibly fp32) head and
                # cast the steered hidden BACK to the residual's dtype. Otherwise a
                # bf16 model's residual silently became fp32 after steering — so
                # even scale=0 perturbed downstream logits (a dtype change, not a
                # value change).
                from mlx.utils import tree_flatten

                orig_dtype = hidden.dtype
                head_params = tree_flatten(self.binary_head.parameters())
                head_dtype = head_params[0][1].dtype if head_params else orig_dtype
                h = hidden.astype(head_dtype)
                logits = self.binary_head(h).squeeze(-1)
                hidden = self.steer_fn(h, self.binary_head, logits).astype(orig_dtype)
        except Exception as e:
            # Do NOT silently pass the hidden through: a steer_fn error is a real bug (shape/dtype/
            # head mismatch) and silently skipping it degrades the run to an unsteered alpha=0 for
            # those positions — a partial failure no downstream check would catch. Fail loud.
            logger.error("steer_fn FAILED at a capture (layer %s): %s", self.layer_idx, e)
            raise

    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def _getattr_mlx(self: Any, name: str) -> Any:
    """Delegate attribute access to the wrapped layer (MLX path)."""
    try:
        return self[name]
    except (KeyError, TypeError):
        pass
    try:
        layer = self["layer"]
    except (KeyError, TypeError) as err:
        raise AttributeError(name) from err
    return getattr(layer, name)


def _getattr_torch(self: Any, name: str) -> Any:
    """Delegate attribute access to the wrapped layer (PyTorch path)."""
    try:
        return super(_TorchLayerCapture, self).__getattr__(name)  # noqa: UP008
    except AttributeError:
        pass
    try:
        layer = super(_TorchLayerCapture, self).__getattr__("layer")  # noqa: UP008
    except AttributeError as err:
        raise AttributeError(name) from err
    return getattr(layer, name)


_MLX_CAPTURE_BASE: type = nn.Module if _MLX else object


class _MLXLayerCapture(_MLX_CAPTURE_BASE):  # type: ignore[misc]
    """LayerCapture for MLX backend (extends mlx.nn.Module)."""

    def __init__(
        self,
        layer: Any,
        layer_idx: int,
        capture_fn: Callable[[Any], None] | None = None,
        steer_fn: Any = None,
        binary_head: Any = None,
        capture_input: bool = False,
    ) -> None:
        """Initialize the capture wrapper."""
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.capture_fn = capture_fn
        self.steer_fn = steer_fn
        self.binary_head = binary_head
        self.capture_input = capture_input

    __call__ = _forward_impl
    __getattr__ = _getattr_mlx


try:
    import torch.nn as _torch_nn

    _TORCH_NN: type = _torch_nn.Module
except ImportError:
    _TORCH_NN = object


class _TorchLayerCapture(_TORCH_NN):  # type: ignore[misc]
    """LayerCapture for PyTorch backend (extends torch.nn.Module)."""

    def __init__(
        self,
        layer: Any,
        layer_idx: int,
        capture_fn: Callable[[Any], None] | None = None,
        steer_fn: Any = None,
        binary_head: Any = None,
        capture_input: bool = False,
    ) -> None:
        """Initialize the capture wrapper."""
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.capture_fn = capture_fn
        self.steer_fn = steer_fn
        self.binary_head = binary_head
        self.capture_input = capture_input

    forward = _forward_impl
    __getattr__ = _getattr_torch


def make_layer_capture(
    layer: Any,
    layer_idx: int,
    capture_fn: Callable[[Any], None] | None = None,
    steer_fn: Any = None,
    binary_head: Any = None,
    backend_name: str = "mlx",
    capture_input: bool = False,
) -> Any:
    """Create a LayerCapture wrapper for the correct backend.

    Args:
        layer: The original transformer block (or submodule).
        layer_idx: Index of this layer in the model.
        capture_fn: Callback ``(hidden_state) -> None``.
        steer_fn: Optional ``(hidden, head, logits) -> modified_hidden``.
        binary_head: Optional probe head module (needed by steer_fn).
        backend_name: ``"mlx"`` or ``"torch"``.
        capture_input: If ``True``, capture the wrapped module's INPUT (the
            residual stream entering it) instead of its output.

    Returns:
        A ``LayerCapture`` instance extending the correct ``nn.Module``.
    """
    if backend_name == "torch":
        return _TorchLayerCapture(
            layer, layer_idx, capture_fn, steer_fn, binary_head, capture_input
        )
    return _MLXLayerCapture(layer, layer_idx, capture_fn, steer_fn, binary_head, capture_input)


# Keep the old name as an alias for backward compatibility with existing tests.
LayerCapture = _MLXLayerCapture


class Probe:
    """A single auxiliary probe head attached to one or more layers.

    A probe captures hidden states from specified layers, optionally
    aggregates them, and passes the result through a trainable module.

    Args:
        config: Probe configuration.
        hidden_dim: Hidden dimension of the model.
        backend_name: ``"mlx"`` or ``"torch"``.
    """

    def __init__(
        self,
        config: ProbeConfig,
        hidden_dim: int,
        backend_name: str = "mlx",
    ) -> None:
        """Initialize the probe."""
        self.config = config
        self.hidden_dim = hidden_dim
        self.backend_name = backend_name
        # Each capture appends ``(resolved_layer_idx, hidden)`` so states can be
        # reordered to ``config.layers`` order regardless of execution order.
        self._captured: list[tuple[int, Any]] = []
        # Resolved (positive) layer indices in ``config.layers`` order; set at
        # injection time and used to reorder captures before aggregation.
        self._resolved_layers: list[int] = []
        self.layer_captures: list[LayerCapture] = []
        # Prompt length for ``granularity="response"`` pooling.  When set, the
        # response pool excludes positions ``< prompt_len`` (the prompt), so
        # inference pools the same response region the trainer trains on.  ``None``
        # pools over all valid (non-padding) positions — the documented default.
        self.prompt_len: int | None = None

        #: Label-free whitening transform fitted by ``fit_mass_mean(whiten=True)``:
        #: ``{"mean", "whitener", "cov"}`` as NumPy arrays, or None. Saved and
        #: restored with the checkpoint so the transform outlives the process.
        self.whitening: dict[str, Any] | None = None
        self.module = self._build_module()
        self._log_params()

    def whiten(self, states: Any) -> Any:
        """Apply the fitted whitening transform: ``Sigma^-1/2 (h - mu)``.

        The mean and covariance are fitted label-free over the hidden states, so
        this applies to ANY state — including tokens with no label, at generation
        time. The probe already scores whitened states internally (the transform
        is folded into its weight and bias); this exposes the transform itself,
        for looking at the geometry of the whitened space directly.

        Args:
            states: ``[..., hidden]`` array of hidden states, any backend or NumPy.

        Returns:
            The whitened states as a NumPy array of the same shape.

        Raises:
            RuntimeError: If no whitening has been fitted for this probe.
        """
        import numpy as np

        from auto_chasm.metrics import to_numpy

        if self.whitening is None:
            raise RuntimeError(
                f"Probe {self.name!r} has no whitening transform. Fit one with "
                "model.fit_mass_mean(train_data, whiten=True)."
            )
        h = np.asarray(to_numpy(states), dtype=np.float64)
        return (h - self.whitening["mean"]) @ self.whitening["whitener"].T

    @property
    def name(self) -> str:
        """Probe name from config."""
        return self.config.name

    @property
    def layers(self) -> list[int]:
        """Layer indices this probe is attached to."""
        return self.config.layers

    @property
    def source(self) -> str:
        """What this probe captures (hidden / embedding / logits)."""
        return self.config.source

    def _build_module(self) -> Any:
        """Build the probe module from config.

        Returns:
            A trainable module (MLX nn.Linear, MLP, or custom callable).
        """
        cfg = dict(self.config.module_config)

        aggregation = self.config.aggregation
        is_concat = isinstance(aggregation, str) and aggregation == "concat"
        if is_concat:
            in_dim = len(self.config.layers) * self.hidden_dim
        elif callable(aggregation) and not isinstance(aggregation, str):
            # A custom callable may concat (-> hidden*L), reduce (-> hidden), or
            # emit any width.  Probe it so the head matches what it ACTUALLY
            # returns — never assume concat width (that broke reducing callables).
            from auto_chasm import _probe_agg

            in_dim = _probe_agg.infer_callable_agg_dim(
                aggregation, len(self.config.layers), self.hidden_dim, self.backend_name
            )
        else:
            in_dim = self.hidden_dim

        in_features = cfg.pop("in_features", in_dim)

        from auto_chasm import modules

        return modules.build_probe_module(self.config, in_features, cfg, self.backend_name)

    def _log_params(self) -> None:
        """Log module parameter count."""
        try:
            from auto_chasm.utils import count_parameters

            n = count_parameters(self.module)
            logger.info("Probe '%s': %d parameters (%s)", self.name, n, self.config.module_type)
        except Exception:
            logger.debug("Could not count parameters for probe '%s'.", self.name)

    def _make_capture_fn(self, layer_idx: int) -> Callable[[Any], None]:
        """Build a capture callback that tags a captured state with its layer."""

        def _capture(h: Any) -> None:
            self._captured.append((layer_idx, h))

        return _capture

    def inject(self, model: Any, resolved_layers: list[int]) -> None:
        """Insert capture wrappers into the model for the configured source.

        ``hidden``/``residual`` wrap each block (output / INPUT); ``attention``/
        ``mlp`` wrap that submodule; ``embedding``/``logits`` the embedding/head.

        Args:
            model: The base language model.
            resolved_layers: Positive layer indices (per-block sources only).
        """
        if self.config.source in ("embedding", "logits"):
            self._inject_single(model)
        else:
            self._inject_layers(model, resolved_layers)

    def _inject_single(self, model: Any) -> None:
        """Wrap a single module (embedding or LM head) with a capture wrapper."""
        if self.config.source == "embedding":
            module, attr_path = _find_embedding(model)
            err = "Cannot find embedding module (tried embedding/embed_tokens/wte)."
        else:
            module, attr_path = _find_output_head(model)
            err = "Cannot find LM head module (tried output_proj/lm_head/output/head)."
        if module is None:
            raise ValueError(err)
        self._resolved_layers = [-1]
        capture = make_layer_capture(
            module,
            layer_idx=-1,
            capture_fn=self._make_capture_fn(-1),
            backend_name=self.backend_name,
        )
        _set_module_by_path(model, attr_path, capture)  # type: ignore[arg-type]
        self.layer_captures.append(capture)

    def _inject_layers(self, model: Any, resolved_layers: list[int]) -> None:
        """Wrap each requested block (or its attention/mlp submodule) to capture.

        Handles ``hidden`` (block output), ``residual`` (block input), and
        ``attention``/``mlp`` (submodule output); sub-block sources unwrap any
        prior block capture so several probes can share a layer.

        Args:
            model: The base language model.
            resolved_layers: Positive layer indices.
        """
        layers = _find_layers(model)
        if layers is None:
            raise ValueError("Cannot find transformer layers in model.")
        self._resolved_layers = list(resolved_layers)

        source = self.config.source
        sub_names = _SUBMODULE_ATTRS.get(source)
        capture_input = source == "residual"

        for idx in resolved_layers:
            block = layers[idx]
            if sub_names is not None:
                # Unwrap any prior block-level capture(s) so probes can share a layer.
                host = block
                while hasattr(host, "capture_fn"):
                    host = host.layer
                target, name = _find_named_child(host, sub_names)
                if target is None or name is None:
                    raise ValueError(
                        f"Probe {self.name!r}: no {source!r} submodule in block {idx} "
                        f"(looked for {sub_names}). Use source='hidden' or a custom probe."
                    )
                capture = make_layer_capture(
                    target,
                    layer_idx=idx,
                    capture_fn=self._make_capture_fn(idx),
                    backend_name=self.backend_name,
                )
                setattr(host, name, capture)
            else:
                capture = make_layer_capture(
                    block,
                    layer_idx=idx,
                    capture_fn=self._make_capture_fn(idx),
                    backend_name=self.backend_name,
                    capture_input=capture_input,
                )
                layers[idx] = capture
            self.layer_captures.append(capture)

    def clear_captured(self) -> None:
        """Clear stored hidden states (call before each forward pass)."""
        self._captured.clear()

    def get_captured_states(self) -> list[Any]:
        """Retrieve captured hidden states in ``config.layers`` order.

        Captures arrive in model-execution (ascending) order; each is tagged
        with its resolved layer index and reordered to match
        ``self._resolved_layers`` so concat columns / weighting align with the
        user's requested order.

        Returns:
            List of hidden-state tensors, one per layer, in config order.
        """
        if not self._resolved_layers or len(self._captured) <= 1:
            return [h for _, h in self._captured]
        by_idx: dict[int, Any] = dict(self._captured)
        if len(by_idx) != len(self._resolved_layers):
            # Repeated layers or partial captures: fall back to capture order.
            return [h for _, h in self._captured]
        return [by_idx[idx] for idx in self._resolved_layers]

    def forward(
        self,
        hidden_states: list[Any] | None = None,
        mask: Any | None = None,
        input_ids: Any | None = None,
    ) -> Any:
        """Run the probe on captured (or provided) hidden states.

        Args:
            hidden_states: Override hidden states.  If ``None``, uses
                captured states from the last forward pass.
            mask: Optional boolean ``[B, T]`` mask of valid positions, used by
                ``granularity="response"`` so pooling ignores padding.
            input_ids: Token ids ``[B, T]``, required for
                ``granularity="sentence"`` to locate sentence boundaries.

        Returns:
            Probe logits (raw).  Shape depends on ``granularity``: per-token
            ``[B, T, out_dim]`` or pooled ``[B, out_dim]``.
        """
        if hidden_states is None:
            hidden_states = self.get_captured_states()

        if not hidden_states:
            raise RuntimeError("No hidden states captured. Run a forward pass first.")

        # On PyTorch, cast to module weight dtype for bf16 hidden states (e.g.
        # Gemma) when the probe was built in fp32.  Gate on the probe's actual
        # backend, not on whether MLX is importable (both can be installed).
        if self.backend_name == "torch":
            try:
                target_dtype = next(self.module.parameters()).dtype
                hidden_states = [h.to(target_dtype) for h in hidden_states]
            except (StopIteration, AttributeError):
                pass

        if len(hidden_states) == 1 and self.config.aggregation == "concat":
            logits = self.module(hidden_states[0])
        else:
            aggregated = self._aggregate(hidden_states)
            logits = self.module(aggregated)

        return self._apply_pooling(logits, mask, input_ids)

    def _response_mask(self, mask: Any | None, seq_len: int) -> Any | None:
        """Narrow a pooling mask to the response region (``>= self.prompt_len``).

        Used by ``granularity="response"`` so inference pools over the same
        response-only region the trainer trains on. When ``self.prompt_len`` is
        unset, the mask is returned unchanged (pool over all valid positions).

        Args:
            mask: Optional valid-position mask ``[B, T]`` (padding excluded), or
                ``None`` (all positions valid).
            seq_len: Sequence length ``T`` of the per-token logits.

        Returns:
            A mask ``[B, T]`` with prompt positions zeroed, or ``None`` when no
            ``prompt_len`` is set and no input mask was given.
        """
        if self.prompt_len is None:
            return mask
        if self.backend_name == "mlx":
            import mlx.core as mx

            response = (mx.arange(seq_len) >= self.prompt_len).astype(mx.int32)[None, :]
            return response if mask is None else mask.astype(mx.int32) * response

        import torch

        response = (torch.arange(seq_len) >= self.prompt_len).long().unsqueeze(0)
        if mask is None:
            return response
        return mask.long() * response.to(mask.device)

    def _apply_pooling(
        self, logits: Any, mask: Any | None = None, input_ids: Any | None = None
    ) -> Any:
        """Apply granularity pooling (delegates to :mod:`auto_chasm._probe_agg`).

        For ``granularity="response"`` the mask is first narrowed to the response
        region via :meth:`_response_mask` (when ``self.prompt_len`` is set), so the
        prompt never contaminates the pooled prediction.
        """
        from auto_chasm import _probe_agg

        if self.config.granularity == "response" and self.prompt_len is not None:
            mask = self._response_mask(mask, logits.shape[1])

        return _probe_agg.apply_pooling(
            self.config, self.name, self.backend_name, logits, mask, input_ids
        )

    def _masked_mean_over_time(self, logits: Any, mask: Any | None) -> Any:
        """Masked mean-pool over time (delegates to :mod:`auto_chasm._probe_agg`)."""
        from auto_chasm import _probe_agg

        return _probe_agg.masked_mean_over_time(logits, mask, self.backend_name)

    def _aggregate(self, hidden_states: list[Any]) -> Any:
        """Aggregate multi-layer hidden states (see :mod:`auto_chasm._probe_agg`).

        Args:
            hidden_states: List of hidden-state tensors.

        Returns:
            Aggregated tensor.
        """
        from auto_chasm import _probe_agg

        return _probe_agg.aggregate(hidden_states, self.config.aggregation, self.backend_name)

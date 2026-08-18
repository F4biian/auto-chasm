"""Model facade — the user-facing entry point. Wraps a base language model."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from auto_chasm.backends import Backend
from auto_chasm.backends.loaders import load_mlx as _load_mlx
from auto_chasm.backends.loaders import load_torch as _load_torch
from auto_chasm.backends.loaders import resolve_backend_name as _resolve_backend_name
from auto_chasm.config import GenerationConfig, LoraConfig, ProbeConfig, SteeringConfig
from auto_chasm.generation import DEFAULT_MAX_REPEAT, _resolve_prompt
from auto_chasm.logger import get_logger
from auto_chasm.outputs import GenerationStep, ModelOutputs
from auto_chasm.probe import (
    Probe,
    _find_layers,
    _get_hidden_dim,
    _get_vocab_size,
    _resolve_negative_index,
)
from auto_chasm.steering import SteeringHook, build_auto_steer_fn, validate_steering

if TYPE_CHECKING:
    import mlx.core as mx
    import torch
logger = get_logger(__name__)


def _explicit_in_features(config: Any) -> int | None:
    """An ``in_features`` the caller put in ``module_config``, if any.

    ``_get_hidden_dim``'s error tells the caller to "Pass
    module_config={'in_features': N}" -- but nothing read it, so that advice did
    not work. It does now, which also gives any exotic architecture an escape
    hatch that needs no library change.

    Args:
        config: The ``ProbeConfig`` being attached.

    Returns:
        The declared input width, or None when unset.
    """
    module_config = getattr(config, "module_config", None) or {}
    value = module_config.get("in_features")
    return int(value) if isinstance(value, int) and value > 0 else None


class Model:
    """User-facing model with probes and steering.

    Wraps a base language model with a unified API for attaching probes,
    training, steering, and generation.

    Args:
        model: The base language model (pre-loaded).
        tokenizer: The tokenizer.
        backend_name: ``"mlx"`` or ``"torch"``.  Auto-detected if ``None``.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        backend_name: Literal["mlx", "torch"] | None = None,
        lora_config: LoraConfig | None = None,
    ) -> None:
        """Initialize the probe-aware language model."""
        self.model = model
        self.tokenizer = tokenizer
        self.backend = Backend(force=backend_name)
        self._probes: dict[str, Probe] = {}
        self._steering_hooks: dict[str, SteeringHook] = {}
        self._original_layers: dict[int, Any] = {}
        # ``{attr_path: original}`` for embedding/logits probes (unwrap on restore).
        self._original_modules: dict[str, Any] = {}
        self._lora_config: LoraConfig | None = lora_config
        self._n_added_special_tokens = 0  # save_checkpoint warns: grown vocab not persisted

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        backend_name: Literal["mlx", "torch"] | None = None,
        lora: LoraConfig | None = None,
        **kwargs: Any,
    ) -> Model:
        """Load a base model and tokenizer, optionally applying LoRA.

        Args:
            model_name: Model name or path.
            backend_name: ``"mlx"`` or ``"torch"``.
            lora: LoRA configuration.  If ``None``, no adapters are applied.
            **kwargs: Additional arguments passed to the model loader.

        Returns:
            A ``Model`` instance ready for probe attachment and training.
        """
        backend_name = _resolve_backend_name(backend_name)  # validate + auto-detect (M15)
        if backend_name == "mlx":
            model, tokenizer = _load_mlx(model_name, **kwargs)
        else:
            model, tokenizer = _load_torch(model_name, **kwargs)

        instance = cls(model, tokenizer, backend_name, lora_config=lora)
        instance._base_model_name = model_name  # type: ignore[attr-defined]

        if lora is not None:
            instance.attach_lora(lora)

        return instance

    @property
    def probes(self) -> dict[str, Probe]:
        """Attached probes keyed by name."""
        return self._probes

    @property
    def raw_model(self) -> Any:
        """The underlying framework model, unwrapped from any wrappers."""
        return self.model

    @property
    def lora_config(self) -> LoraConfig | None:
        """The LoRA configuration, or ``None`` if no adapters are applied."""
        return self._lora_config

    @property
    def steering_hooks(self) -> dict[str, SteeringHook]:
        """Steering hooks keyed by probe name."""
        return self._steering_hooks

    @property
    def base_model_name(self) -> str | None:
        """The model name/path, or ``None`` if constructed directly."""
        return getattr(self, "_base_model_name", None)

    def attach_lora(self, config: LoraConfig | None = None, **kwargs: Any) -> None:
        """Apply PEFT adapters to the model.

        Dispatches to the correct adapter function based on
        ``config.peft_method``.

        Args:
            config: PEFT configuration.  Uses stored config if ``None``.
            **kwargs: Override fields on the config (e.g. ``rank=16``).
        """
        if config is None:
            config = self._lora_config
        if config is None:
            raise ValueError("No LoraConfig provided and none stored on the model.")

        if kwargs:
            import dataclasses

            config = dataclasses.replace(config, **kwargs)

        method = config.peft_method

        _apply: Callable[..., Any]  # generic: qlora adds bits/group_size extras
        if method == "qlora":
            from auto_chasm.peft import apply_qlora as _apply
        elif method == "dora":
            from auto_chasm.peft import apply_dora as _apply
        else:
            from auto_chasm.peft import apply_lora as _apply

        # Assign the return: torch's get_peft_model returns a NEW PeftModel wrapper (MLX in-place).
        self.model = _apply(
            self.model,
            r=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            target_modules=config.target_modules,
            target_layers=config.target_layers,
            until_layer=config.until_layer,
            after_layer=config.after_layer,
            backend=self.backend,
        )

        self._lora_config = config
        logger.info(
            "%s applied (rank=%d, alpha=%d).",
            method.upper(),
            config.rank,
            config.alpha,
        )

    def to_tensor(self, data: list[Any] | np.ndarray | mx.array | torch.Tensor) -> Any:
        """Convert a list, numpy array, or tensor to a framework tensor.

        Args:
            data: Input data (list, numpy array, or tensor).

        Returns:
            A framework tensor (``mx.array`` or ``torch.Tensor``).
        """
        return self.backend.tensor.tensor(data)

    def sample(self, logits: mx.array | torch.Tensor, temperature: float = 0.0) -> int:
        """Sample a token index from logits (backend-agnostic)."""
        return self.backend.tensor.sample(logits, temperature)

    def freeze_model(self) -> None:
        """Freeze all base model parameters. Probes and LoRA adapters are unaffected."""
        self.backend.module.freeze(self.model)

    def unfreeze_model(self) -> None:
        """Unfreeze all base model parameters."""
        self.backend.module.unfreeze(self.model)

    def unfreeze_lora_adapters(self) -> None:
        """Unfreeze LoRA adapter parameters only."""
        from auto_chasm.peft import _unfreeze_lora_params

        _unfreeze_lora_params(self.model, self.backend)

    def unfreeze_probe(self, name: str) -> None:
        """Unfreeze a specific probe by name. Raises ``KeyError`` if not found."""
        if name not in self._probes:
            raise KeyError(f"Probe '{name}' not found.")
        self.backend.module.unfreeze(self._probes[name].module)

    def freeze_probe(self, name: str) -> None:
        """Freeze a specific probe by name. Raises ``KeyError`` if not found."""
        if name in self._probes:
            self.backend.module.freeze(self._probes[name].module)
        else:
            raise KeyError(f"Probe '{name}' not found.")

    def unfreeze_all_probes(self) -> None:
        """Unfreeze all attached probe modules."""
        for probe in self._probes.values():
            self.backend.module.unfreeze(probe.module)

    def prepare_for_joint_training(self) -> None:
        """Freeze base model, unfreeze LoRA adapters and all probes."""
        self.freeze_model()
        self.unfreeze_lora_adapters()
        self.unfreeze_all_probes()
        logger.info("Base model frozen; LoRA + probes unfrozen.")

    def freeze_adapters_and_unfreeze_probes(self) -> None:
        """Deprecated. Use :meth:`prepare_for_joint_training` instead."""
        import warnings

        warnings.warn(
            "Deprecated, use prepare_for_joint_training() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.prepare_for_joint_training()

    def attach_probe(self, config: ProbeConfig) -> Probe:
        """Attach a probe to the model.

        Args:
            config: Probe configuration.

        Returns:
            The attached ``Probe`` instance.

        Raises:
            ValueError: If a probe with the same name is already attached
                (re-using a name would orphan the old capture wrapper and leak
                memory); choose another name or detach the existing probe first.
        """
        if config.name in self._probes:
            raise ValueError(
                f"A probe named {config.name!r} is already attached. Probe names must be "
                "unique; choose a different name (re-using one would orphan the existing "
                "capture wrapper and leak memory)."
            )
        resolved: list[int] = []
        # Per-block sources (hidden/attention/mlp/residual) need resolved layer indices.
        if config.source in ("hidden", "attention", "mlp", "residual"):
            layers = _find_layers(self.model)
            if layers is None:
                raise ValueError("Cannot find transformer layers in model.")
            resolved = [_resolve_negative_index(idx, len(layers)) for idx in config.layers]
            for idx in resolved:
                # setdefault (not assign): don't let a 2nd probe overwrite the wrapped block.
                self._original_layers.setdefault(idx, layers[idx])
            in_dim = _explicit_in_features(config) or _get_hidden_dim(self.model)
        elif config.source == "logits":
            in_dim = _get_vocab_size(self.model)
        else:  # embedding
            in_dim = _explicit_in_features(config) or _get_hidden_dim(self.model)

        if config.source in ("embedding", "logits"):  # record original for later unwrap
            from auto_chasm import _probe_inject

            _probe_inject.track_single_module(self.model, config.source, self._original_modules)

        probe = Probe(config, in_dim, self.backend.name)
        self._move_probe_to_model_device(probe)
        probe.inject(self.model, resolved)

        self._probes[config.name] = probe
        logger.info(
            "Attached probe %r at layers %s (source=%s)", config.name, config.layers, config.source
        )
        return probe

    def _move_probe_to_model_device(self, probe: Probe) -> None:
        """Move the probe module to the base model's device (torch only).

        Args:
            probe: The probe whose module to move.
        """
        if self.backend.name != "torch":
            return
        try:
            device = next(self.model.parameters()).device
            probe.module = probe.module.to(device)
        except (StopIteration, AttributeError):
            pass

    def add_probes(self, probes: list[ProbeConfig]) -> None:
        """Attach multiple probes at once (one ``attach_probe`` per config).

        Args:
            probes: List of probe configurations to attach.
        """
        for config in probes:
            self.attach_probe(config)

    def add_special_tokens(self, tokens: list[str]) -> int:
        """Register new special tokens and grow the model embeddings to match.

        Grows the input embedding + any untied head (torch/MLX, full or quantized) and
        syncs ``config.vocab_size``. NOT persisted by ``save_checkpoint`` — re-add later.

        Args:
            tokens: New token strings to add.

        Returns:
            The number of tokens actually added (0 if all already existed).
        """
        from auto_chasm.special_tokens import add_special_tokens

        n_added = add_special_tokens(self.model, self.tokenizer, tokens, self.backend.name)
        self._n_added_special_tokens += n_added
        return n_added

    def forward(
        self,
        input_ids: mx.array | torch.Tensor | list[Any] | np.ndarray,
        attention_mask: mx.array | torch.Tensor | list[Any] | np.ndarray | None = None,
        _labels: Any | None = None,
        pool_mask: mx.array | torch.Tensor | list[Any] | np.ndarray | None = None,
    ) -> ModelOutputs:
        """Run a full forward pass (base model + all probes).

        Clears captured states first, then runs the base model and every probe.
        Accepts framework tensors, numpy arrays, or nested lists (the latter two
        are converted to the backend's tensor format).

        Args:
            input_ids: Token IDs (tensor, numpy array, or nested list).
            attention_mask: Optional ``[B, T]`` validity mask used for
                ``granularity="response"`` probe pooling (the base model is not
                given it — it uses internal causal masking; right-pad inputs).
            _labels: LM labels for loss computation (optional).
            pool_mask: Optional response-region mask ``[B, T]`` for probe
                pooling.  Scopes a ``granularity="response"`` probe to the
                trainer's response region (prompt + padding excluded) so inference
                matches training; ``None`` falls back to ``attention_mask``.

        Returns:
            Structured ``ModelOutputs`` with lm_logits and probe outputs.
        """
        from auto_chasm import _probe_inject

        input_ids, attention_mask, pool_mask = _probe_inject.prepare_inputs(
            self.model, self.backend, input_ids, attention_mask, pool_mask
        )
        effective_pool_mask = pool_mask if pool_mask is not None else attention_mask

        for probe in self._probes.values():
            probe.clear_captured()

        # The base gets no attention mask (mlx_lm has none); padding uses the pool mask.
        lm_out = self.backend.module.forward(self.model, input_ids)
        lm_logits = _probe_inject.extract_lm_logits(lm_out)
        probe_outputs = _probe_inject.run_probes(self._probes, input_ids, effective_pool_mask)
        return ModelOutputs(lm_logits=lm_logits, probes=probe_outputs)

    def enable_steering(
        self,
        probe_name: str,
        config: SteeringConfig | None = None,
        class_means: dict[str, Any] | None = None,
        steer_fn: Any | None = None,
    ) -> None:
        """Enable steering for a probe (applied inside the forward pass).

        Args:
            probe_name: Name of the probe to steer.
            config: Steering configuration.  If ``None``, uses default.
            class_means: ``{"mean_0": tensor, "mean_1": tensor}``.
            steer_fn: Optional custom steering function
                ``(hidden, head, logits) -> modified_hidden``.  When given it
                runs instead of the closed-form geometry, so steering works
                even with no ``class_means`` (``method="custom"`` wiring).

        Raises:
            KeyError: If the probe is not attached.
        """
        if probe_name not in self._probes:
            raise KeyError(f"Probe '{probe_name}' not attached.")

        if probe_name not in self._steering_hooks:
            hook = SteeringHook(probe_name, config if config is not None else SteeringConfig())
            self._steering_hooks[probe_name] = hook
        else:
            hook = self._steering_hooks[probe_name]
            # Honor an explicit config on re-enable; geometry lives on the hook, not config.
            if config is not None:
                hook.config = config

        probe = self._probes[probe_name]

        if steer_fn is not None:
            hook.set_custom(steer_fn)

        validate_steering(hook, probe, self.model)  # reject ambiguous/silent-noop configs
        if class_means is not None:
            probe_means = class_means.get(probe_name, class_means)
            mean_0 = probe_means.get("mean_0")
            mean_1 = probe_means.get("mean_1")
            head_weight = probe.module.weight if hasattr(probe.module, "weight") else None
            head_bias = probe.module.bias if hasattr(probe.module, "bias") else None
            if mean_0 is not None and mean_1 is not None:
                hidden_by_class = {0: [mean_0], 1: [mean_1]}
                hook.compute_geometry(hidden_by_class, head_weight, head_bias)

        built = build_auto_steer_fn(hook)
        if built is None:
            raise ValueError(
                f"Cannot enable steering for probe '{probe_name}': no steering geometry. "
                f"Pass class_means=model.compute_class_means(dataset) or a custom steer_fn. "
                f"Refusing a no-op."
            )

        for capture in probe.layer_captures:
            capture.steer_fn = built
            capture.binary_head = probe.module

        hook.enable()

    def disable_steering(self, probe_name: str) -> None:
        """Disable steering for a probe by name.

        Args:
            probe_name: Name of the probe.
        """
        if probe_name in self._steering_hooks:
            self._steering_hooks[probe_name].disable()
        probe = self._probes.get(probe_name)
        if probe:
            for capture in probe.layer_captures:
                capture.steer_fn = None

    def _clear_probe_captures(self) -> None:
        """Clear every attached probe's captured hidden states.

        Plain ``generate``/``generate_stream`` drive the BASE model directly, so its
        capture hooks accumulate a per-step pile nothing consumes. Clearing before and
        after a run keeps it from growing across calls or being read stale later.
        """
        for probe in self._probes.values():
            probe.clear_captured()

    def generate(
        self,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        config: GenerationConfig | None = None,
        messages: list[dict[str, str]] | None = None,
        max_repeat: int | None = DEFAULT_MAX_REPEAT,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt or chat messages.

        Works out of the box; steering (if enabled) is applied transparently.
        Explicit ``max_tokens``/``temperature`` always win over ``config``;
        leave them ``None`` to fall back to ``config`` or the defaults.

        Args:
            prompt: Input text prompt (ignored if ``messages`` is set).
            max_tokens: Maximum tokens to generate (``None`` uses config/256).
            temperature: Sampling temperature (``None`` uses config/0.0).
            config: ``GenerationConfig`` overrides (explicit kwargs win).
            messages: Chat messages (requires ``chat_template``).
            max_repeat: Repetition-guard cap; ``None`` disables the guard.
            **kwargs: Additional generation arguments.

        Returns:
            Generated text string.
        """
        resolved = _resolve_prompt(self.tokenizer, prompt, messages)
        from auto_chasm.generation import generate as _generate

        max_tokens, temperature, kwargs = self._apply_gen_config(
            config, max_tokens, temperature, kwargs
        )
        kwargs["max_repeat"] = max_repeat  # forward always so max_repeat=None disables

        self._clear_probe_captures()  # attached probes accrue captures via base hooks
        try:
            return _generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=resolved,
                max_tokens=max_tokens,
                temperature=temperature,
                backend=self.backend,
                **kwargs,
            )
        finally:
            self._clear_probe_captures()  # don't leave a stale pile for a later forward

    def generate_stream(
        self,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        config: GenerationConfig | None = None,
        messages: list[dict[str, str]] | None = None,
        max_repeat: int | None = DEFAULT_MAX_REPEAT,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream generated tokens one at a time (see :meth:`generate` for args).

        ``max_repeat=None`` disables the repetition guard.

        Yields:
            Individual token strings.
        """
        resolved = _resolve_prompt(self.tokenizer, prompt, messages)
        from auto_chasm.generation import generate_stream as _generate_stream

        max_tokens, temperature, kwargs = self._apply_gen_config(
            config, max_tokens, temperature, kwargs
        )
        kwargs["max_repeat"] = max_repeat  # forward always so max_repeat=None disables

        self._clear_probe_captures()
        try:
            yield from _generate_stream(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=resolved,
                max_tokens=max_tokens,
                temperature=temperature,
                backend=self.backend,
                **kwargs,
            )
        finally:
            self._clear_probe_captures()

    def generate_with_probes(
        self,
        prompt: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop_tokens: list[int] | None = None,
        messages: list[dict[str, str]] | None = None,
        max_repeat: int | None = DEFAULT_MAX_REPEAT,
    ) -> Iterator[GenerationStep]:
        """Generate tokens one at a time with full probe inspection.

        Each yielded ``GenerationStep`` carries the token, decoded string, per-probe
        outputs, and next-token logits. Steering is applied transparently, and a
        ``granularity="response"`` probe pools its response-only region (matching
        training).

        Args:
            prompt: Input text prompt (ignored if ``messages`` is set).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy; negative raises).
            stop_tokens: Auto-detected from tokenizer if ``None``.
            messages: Chat messages (requires ``chat_template``).
            max_repeat: Repetition-guard cap; ``None`` disables the guard.

        Yields:
            ``GenerationStep`` per generated token.
        """
        from auto_chasm._model_generate import generate_with_probes as _gwp

        yield from _gwp(self, prompt, max_tokens, temperature, stop_tokens, messages, max_repeat)

    def chat_repl(
        self,
        system_prompt: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> None:
        """Start an interactive chat REPL (type ``exit``/``quit`` to stop).

        Args:
            system_prompt: Optional system prompt prepended to conversation.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
        """
        from auto_chasm.generation import chat_repl as _chat_repl

        _chat_repl(self.model, self.tokenizer, system_prompt, max_tokens, temperature, self.backend)

    def _apply_gen_config(
        self,
        config: GenerationConfig | None,
        max_tokens: int | None,
        temperature: float | None,
        kwargs: dict[str, Any],
    ) -> tuple[int, float, dict[str, Any]]:
        """Resolve generation params (see :func:`auto_chasm._gen_config.apply_gen_config`)."""
        from auto_chasm._gen_config import apply_gen_config

        max_tokens, temperature, kwargs = apply_gen_config(config, max_tokens, temperature, kwargs)
        # Steering re-derives past hidden states a KV cache would freeze; FORCE it off.
        if any(h.enabled for h in self._steering_hooks.values()):
            kwargs["use_cache"] = False
        return max_tokens, temperature, kwargs

    def save_checkpoint(self, path: str) -> None:
        """Save probes, adapters, and steering data to the directory ``path``."""
        from auto_chasm.checkpoint import save_checkpoint

        save_checkpoint(self, path)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        base_model: str | None = None,
        load_steering: bool = True,
        backend_name: Literal["mlx", "torch"] | None = None,
    ) -> Model:
        """Load a fully restored model from a checkpoint directory.

        Args:
            path: Checkpoint directory path.
            base_model: Override base model name.
            load_steering: Whether to restore steering geometry.
            backend_name: ``"mlx"`` or ``"torch"``.

        Returns:
            A fully restored ``Model``.
        """
        from auto_chasm.checkpoint import load_checkpoint

        return load_checkpoint(
            path,
            base_model=base_model,
            load_steering=load_steering,
            backend_name=backend_name,
        )

    def compute_class_means(
        self,
        dataset: object,
        batch_size: int = 8,
        max_seq_length: int = 256,
    ) -> dict[str, Any]:
        """Compute per-class mean hidden states at each probe's layer.

        Args:
            dataset: Dataset yielding ``(tokens, labels)`` tuples.
            batch_size: Batch size for iteration.
            max_seq_length: Maximum sequence length.

        Returns:
            ``{probe_name: {"mean_0": tensor, "mean_1": tensor}}``.
        """
        from auto_chasm.class_means import compute_class_means as _compute

        return _compute(self, self._probes, dataset, self.backend.name, batch_size, max_seq_length)

    def save_class_means(self, class_means: dict[str, dict[str, Any]], path: str) -> None:
        """Save class-mean vectors (see ``auto_chasm.class_means.save_class_means``)."""
        from auto_chasm.class_means import save_class_means

        save_class_means(self, class_means, path)

    def load_class_means(self, path: str) -> dict[str, Any]:
        """Load class-mean vectors (see ``auto_chasm.class_means.load_class_means``)."""
        from auto_chasm.class_means import load_class_means

        return load_class_means(self, path)

    def restore_original_layers(self) -> None:
        """Remove all capture wrappers, returning the model to its original state.

        Restores block captures (``hidden``/``residual``), attention/mlp submodule
        captures, and the embedding/LM-head wrappers from ``embedding``/``logits``
        probes, then detaches every probe (disabling its steering hooks) so names
        are free to reuse.
        """
        from auto_chasm import _probe_agg, _probe_inject

        layers = _find_layers(self.model)
        if layers is not None:
            for idx, original in self._original_layers.items():
                layers[idx] = original
            _probe_agg.unwrap_submodule_captures(layers)
            self._original_layers.clear()

        # Unwrap embedding/logits wrappers, restoring the original module.
        _probe_inject.restore_single_modules(self.model, self._original_modules)

        for hook in self._steering_hooks.values():
            hook.disable()
        self._steering_hooks.clear()
        self._probes.clear()

    def enable_gradient_checkpointing(self) -> int:
        """Trade ~30% step time for a large cut in activation memory.

        Without checkpointing every block's intermediates are held until backward,
        so peak memory grows LINEARLY with sequence length and batch size —
        measured at ~42 MB per token on Qwen3.5-0.8B, which is what puts a 0.8B
        model out of memory on a 64 GB machine. Checkpointing keeps only each
        block's input and recomputes the interior during backward.

        Call it BEFORE training (order relative to ``attach_probe`` /
        ``attach_lora`` does not matter). See
        :func:`auto_chasm._grad_checkpoint.enable` for the MLX class-patching
        caveat.

        Returns:
            Number of block types patched (MLX) or ``1`` (torch).
        """
        from auto_chasm import _grad_checkpoint

        return int(_grad_checkpoint.enable(self.model, self.backend.name))

    def disable_gradient_checkpointing(self) -> int:
        """Undo :meth:`enable_gradient_checkpointing`."""
        from auto_chasm import _grad_checkpoint

        return int(_grad_checkpoint.disable(self.model, self.backend.name))

    @property
    def num_layers(self) -> int:
        """Number of transformer layers in the underlying model."""
        from auto_chasm.peft import get_num_layers

        return get_num_layers(self.model)

    @property
    def hidden_size(self) -> int:
        """The model's hidden dimension."""
        from auto_chasm._model_stats import hidden_size

        return hidden_size(self.model)

    @property
    def vocab_size(self) -> int:
        """The vocabulary size (from config, else the embedding's row count)."""
        from auto_chasm._model_stats import vocab_size

        return vocab_size(self.model)

    def num_parameters(self, *, trainable: bool = False) -> int:
        """Parameter count of the base model + every probe (trainable-only if set)."""
        from auto_chasm._model_stats import num_parameters

        return num_parameters(self, trainable=trainable)

    @property
    def lora_targetable_modules(self) -> list[str]:
        """Every module LoRA can adapt — full paths (see ``peft.targetable_lora_modules``).

        This is exactly the set ``LoraConfig(target_modules=None)`` adapts by
        default. Also included in :meth:`stats`.
        """
        from auto_chasm.peft import targetable_lora_modules

        return targetable_lora_modules(self.model)

    def stats(self) -> dict[str, Any]:
        """Architecture + parameter stats as a dict (see ``auto_chasm._model_stats``)."""
        from auto_chasm._model_stats import model_stats

        return model_stats(self)

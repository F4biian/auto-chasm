"""Configuration dataclasses for auto-chasm.

Provides transformers-like config-driven API for probes, training,
generation, steering, and RL.  All configs are frozen after init
(immutability prevents accidental mutation mid-training).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

#: Reserved term name for the language-model head in ``JointLoss`` weights/losses
#: and ``combine`` (``L.lm_head``).  A probe may not take this name — it would
#: collide with the LM term in the loss's per-term namespace.  Enforced at the
#: earliest point (``ProbeConfig.__post_init__``) and again at loss-compute time.
LM_HEAD = "lm_head"


@dataclass
class ProbeConfig:
    """Configuration for a single auxiliary probe head.

    Probes are named, attached to specific layers, and produce outputs
    at a configurable granularity.  Built-in module types (``"linear"``,
    ``"mlp"``, ``"mass_mean"``, ``"mass_mean_whiten"``) cover common cases; pass a
    callable for custom architectures.

    Attributes:
        name: Unique probe identifier (used as dict key in outputs).
        layers: Layer indices to capture (negative indexing supported). A per-layer
            source (``hidden``/``residual``/``attention``/``mlp``) may span several
            layers (combined by ``aggregation``); ``embedding``/``logits`` read a
            single site and take exactly one layer.
        source: What part of the model to capture.

            - ``"hidden"``: Full transformer block output (default).
            - ``"embedding"``: Input embedding lookup (before any blocks).
            - ``"logits"``: LM head output (vocabulary projection).
            - ``"attention"``: The block's self-attention submodule output.
            - ``"mlp"``: The block's MLP / feed-forward submodule output.
            - ``"residual"``: The residual stream *entering* the block (its
              input hidden state).
        aggregation: How to combine multi-layer inputs
            (``"concat"``, ``"mean"``, ``"max"``, ``"last"``, or callable).
        module_type: Built-in type name (``"linear"``, ``"mlp"``, ``"mass_mean"``,
            ``"mass_mean_whiten"``) or callable ``(in_dim, cfg) -> nn.Module``.

            ``"mass_mean"`` builds ``scale * (h . direction) + bias`` with the
            direction FROZEN and only the two scalars trainable. Attach it like
            any other head, fill the direction with
            ``model.fit_mass_mean(train_data)``, and train it jointly with LoRA
            as usual -- the optimizer can rescale and shift the boundary but
            never rotate the axis. Prefer it over a hand-rolled callable: the
            direction is saved inside the probe checkpoint and the string
            survives the manifest, so ``Model.from_checkpoint`` restores the
            probe with no side-car file and no rebuild code.

            ``"mass_mean_whiten"`` is the same head, declared to be fitted in the
            whitened space: ``fit_mass_mean`` then measures the direction after
            ``Sigma^-1/2 (h - mu)`` without being told to. The whitening folds into
            the weight and bias, so the head still reads RAW states at inference.
        module_config: Keyword arguments for the module constructor.
        granularity: Output granularity.

            - ``"token"``: Per-token ``[B, T, out_dim]`` (default).
            - ``"response"``: Per-sequence ``[B, out_dim]`` (masked mean
              pool over the valid, non-padding positions).
            - ``"sentence"``: Mean-pool each sentence's tokens and broadcast the
              result back, keeping ``[B, T, out_dim]`` (so per-token labels still
              apply). Requires explicit boundaries:
              ``module_config={"sentence_delimiters": [<token ids ending a
              sentence>]}`` (e.g. ``tokenizer.encode(".")``).
            - ``"custom"``: Delegates to ``pooling`` callable.
        pooling: Custom pooling function ``(logits) -> pooled_logits``
            for ``granularity="custom"``.
        layer_norm: If ``True``, a ``LayerNorm(in_features)`` is applied to the
            captured hidden state *before* the probe head (any ``module_type`` —
            string or callable).  This normalizes per-layer activation scale,
            which makes one learning rate valid across layers.  Do not combine
            with ``ModuleSpec(input_layer_norm=True)`` (that builds its own input
            norm) or the input is normalized twice.
    """

    name: str
    layers: list[int]
    source: Literal["hidden", "embedding", "logits", "attention", "mlp", "residual"] = "hidden"
    aggregation: str | Callable[..., Any] = "concat"
    module_type: str | Callable[..., Any] = "linear"
    module_config: dict[str, Any] = field(default_factory=dict)
    granularity: Literal["token", "response", "sentence", "custom"] = "token"
    pooling: Callable[..., Any] | None = None
    layer_norm: bool = False

    def __post_init__(self) -> None:
        """Validate probe configuration after initialization."""
        if self.name == LM_HEAD:
            raise ValueError(
                f"Probe name {LM_HEAD!r} is reserved for the language-model head "
                "(it is the fixed term name in JointLoss weights/losses and the "
                "`combine` namespace, e.g. `L.lm_head`). Choose another probe name."
            )
        if not self.layers:
            raise ValueError("ProbeConfig.layers must contain at least one layer index.")
        valid_sources = ("hidden", "embedding", "logits", "attention", "mlp", "residual")
        if self.source not in valid_sources:
            raise ValueError(
                f"Unknown source {self.source!r}. Use one of {valid_sources}. "
                "(An unvalidated typo would otherwise be silently treated as "
                "'embedding' and probe the wrong site.)"
            )
        if self.source in ("embedding", "logits") and len(self.layers) > 1:
            # embedding/logits read ONE site and ignore per-layer indices; sizing the
            # head from len(layers) would otherwise crash (concat) or silently drop
            # layers (mean/max/last). Genuine multi-layer probing needs a per-layer source.
            raise ValueError(
                f"source={self.source!r} reads a single site, so it cannot span "
                f"multiple layers; got layers={self.layers}. Use one layer (e.g. "
                "layers=[0]), or source='hidden'/'residual'/'attention'/'mlp' for a "
                "multi-layer probe."
            )
        valid_gran = ("token", "response", "sentence", "custom")
        if self.granularity not in valid_gran:
            raise ValueError(
                f"Unknown granularity {self.granularity!r}. Use one of {valid_gran}. "
                "(An unvalidated typo would otherwise be silently treated as 'token'.)"
            )
        valid_agg = ("concat", "mean", "max", "last")
        if isinstance(self.aggregation, str) and self.aggregation not in valid_agg:
            raise ValueError(
                f"Unknown aggregation {self.aggregation!r}. "
                f"Use one of {valid_agg} or pass a callable."
            )
        if self.granularity == "sentence" and not self.module_config.get("sentence_delimiters"):
            raise ValueError(
                "granularity='sentence' requires module_config={'sentence_delimiters': "
                "[<token ids that end a sentence>]} (e.g. from tokenizer.encode('.')). "
                "Sentence boundaries must be explicit — there is no auto-detection."
            )


@dataclass
class TrainingConfig:
    """Configuration for the joint training loop.

    Mirrors HuggingFace Trainer parameter naming where possible.
    All standard training hyperparameters are exposed here; the trainer
    does not hard-code any values.

    Attributes:
        lm_weight: Weight for the language-modeling loss term.
        probe_weight: Global weight for all probe loss terms.
        probe_weights: Per-probe weight overrides (overrides ``probe_weight``).
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        max_grad_norm: Gradient clipping max norm.
        num_epochs: Reserved.  The trainers are iteration-based and drive
            training length via the ``num_iters`` constructor argument;
            epoch-based training is **not implemented**.  Leave at the default
            (``3``); a non-default value raises ``NotImplementedError`` rather
            than being silently ignored.
        batch_size: Per-step batch size.
        gradient_accumulation_steps: Accumulate gradients over N steps.
        warmup_ratio: Fraction of total steps for linear warmup.
        lr_schedule: Learning rate schedule type.
        eval_steps: Evaluate every N steps.  ``0`` disables mid-epoch eval.
        save_steps: Save checkpoint every N steps.  ``0`` disables mid-epoch saves.
        save_best_only: Reserved.  Best-checkpoint behavior is controlled by the
            trainer's ``keep_best_only`` constructor argument, not this field.
            Leave at the default (``True``); a non-default value raises
            ``NotImplementedError`` rather than being silently ignored.
        logging_steps: Log metrics every N steps.
        seed: Random seed for reproducibility.
        mixed_precision: Precision mode. ``"fp32"`` (default) trains in full
            precision. ``"bf16"`` casts the frozen base to bfloat16 on both
            backends — the separate probe head and the optimizer stay fp32 (LoRA
            adapters live inside the base, so they are cast to bf16 too, as usual
            for bf16 training); bf16 shares fp32's exponent range, so no loss
            scaling is needed. ``"fp16"`` is **torch only** and uses
            ``torch.autocast`` + a ``GradScaler`` (fp16's narrow range needs loss
            scaling); it raises ``NotImplementedError`` on MLX — prefer ``"bf16"``.
        output_dir: Directory for checkpoints and logs.
    """

    lm_weight: float = 1.0
    probe_weight: float = 1.0
    probe_weights: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"
    eval_steps: int = 500
    save_steps: int = 500
    save_best_only: bool = True
    logging_steps: int = 10
    seed: int = 42
    mixed_precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    output_dir: str = "./checkpoints"

    def __post_init__(self) -> None:
        """Validate training configuration after initialization.

        Raises:
            ValueError: If ``warmup_ratio`` is outside ``[0, 1]`` or
                ``mixed_precision`` is not one of ``fp32``/``bf16``/``fp16``.
            NotImplementedError: If the reserved ``num_epochs`` / ``save_best_only``
                fields are set to a non-default value (these are not wired into the
                iteration-based trainers and must not be silently ignored). Note
                ``mixed_precision="fp16"`` is accepted here but raises later on the
                MLX trainer (fp16 is torch-only).
        """
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        if self.mixed_precision not in ("fp32", "bf16", "fp16"):
            raise ValueError(
                f"mixed_precision={self.mixed_precision!r} is not valid. Use "
                "'fp32' (default), 'bf16' (both backends), or 'fp16' (torch only)."
            )
        if self.num_epochs != 3:
            raise NotImplementedError(
                f"num_epochs={self.num_epochs} is not implemented. The trainers are "
                "iteration-based: pass num_iters to the Trainer instead. num_epochs "
                "does not drive training and must not be silently ignored. "
                "Leave it at the default (3)."
            )
        if self.save_best_only is not True:
            raise NotImplementedError(
                f"save_best_only={self.save_best_only} is not honored via "
                "TrainingConfig. Best-checkpoint behavior is controlled by the "
                "Trainer's keep_best_only constructor argument. Leave save_best_only "
                "at the default (True)."
            )


@dataclass
class RLConfig:
    """Configuration for the probe-penalty (RL-style) trainer.

    ``algorithm="sft"`` (supervised CE + a ``beta``-weighted probe penalty;
    ``beta=0`` is pure SFT) and ``algorithm="dpo"`` (Direct Preference
    Optimization) are implemented. ``"ppo"`` and ``"grpo"`` are accepted as
    names but **raise ``NotImplementedError``** — they need an on-policy reward
    (the experiment's design), which the library will not fake.

    Attributes:
        algorithm: ``"sft"`` / ``"dpo"`` (implemented) or ``"ppo"``/``"grpo"``
            (raise on use).
        beta: For ``"sft"``, the probe-penalty strength. For ``"dpo"``, the DPO
            temperature on the reference-corrected log-ratio (typically 0.1).
        clip_ratio: Reserved for a future PPO implementation.
        num_candidates: Reserved for a future GRPO implementation.
        reward_fn: Reserved for a future reward-model implementation.
        ref_model_path: Reserved for a future reference-model implementation.
    """

    algorithm: Literal["sft", "dpo", "ppo", "grpo"] = "sft"
    beta: float = 0.1
    clip_ratio: float = 0.2
    num_candidates: int = 4
    reward_fn: Callable[..., float] | None = None
    ref_model_path: str | None = None

    def __post_init__(self) -> None:
        """Validate RL configuration after initialization.

        Raises:
            ValueError: If ``algorithm`` is not a known name, or ``beta`` is
                negative. (``"ppo"``/``"grpo"`` are valid *names* but raise
                ``NotImplementedError`` when used — see the class docstring.)
        """
        valid_algos = ("sft", "dpo", "ppo", "grpo")
        if self.algorithm not in valid_algos:
            raise ValueError(
                f"Unknown algorithm {self.algorithm!r}. Use one of {valid_algos} "
                "('ppo'/'grpo' are accepted names but raise NotImplementedError on use)."
            )
        if self.beta < 0:
            raise ValueError(
                f"beta must be >= 0, got {self.beta}. For algorithm='sft' the total "
                "is ce + beta*penalty, so a negative beta would reward the probe "
                "penalty (anti-training)."
            )


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    All standard generation parameters are exposed.  Defaults are
    conservative (greedy decoding); set ``temperature`` > 0 for sampling.

    Attributes:
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling.
        repetition_penalty: Penalty for repeated tokens.
        do_sample: Whether to sample (overrides temperature=0 behavior).
        stop_sequences: Sequences that terminate generation.
        num_return_sequences: Number of sequences to return.
    """

    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    do_sample: bool = False
    stop_sequences: list[str] = field(default_factory=list)
    num_return_sequences: int = 1

    def __post_init__(self) -> None:
        """Validate generation configuration after initialization.

        Raises:
            ValueError: If ``temperature`` is negative (it would invert the
                sampling distribution), ``top_p`` is outside ``[0, 1]``,
                ``top_k`` is negative, or ``max_tokens`` is negative. ``0`` is
                allowed for ``temperature`` (greedy), ``top_p``/``top_k``
                (disabled), and ``max_tokens`` (empty continuation).
        """
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be >= 0, got {self.temperature}. A negative "
                "temperature divides the logits by a negative number, inverting the "
                "distribution (sampling the least-likely tokens). Use 0 for greedy."
            )
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}.")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0 (0 disables it), got {self.top_k}.")
        if self.max_tokens < 0:
            raise ValueError(
                f"max_tokens must be >= 0 (0 yields an empty output), got {self.max_tokens}."
            )


@dataclass
class LoraConfig:
    """Configuration for LoRA (Low-Rank Adaptation) adapters.

    Pass to ``Model.from_pretrained(lora=LoraConfig(...))`` or
    ``model.attach_lora(LoraConfig(...))`` to apply adapters.

    Layer targeting controls which transformer layers receive LoRA
    adapters.  You can specify:
    - ``target_layers``: explicit list of layer indices (e.g., ``[0, 5, 10]``)
    - ``until_layer``: apply LoRA to layers *before* this index (e.g.,
      ``until_layer=16`` means layers 0-15)
    - ``after_layer``: apply LoRA to layers *at or after* this index
      (e.g., ``after_layer=16`` means layers 16+)

    Use ``model.num_layers`` to get the total layer count for computing
    midpoints::

        mid = model.num_layers // 2
        cfg = LoraConfig(rank=16, until_layer=mid)

    Attributes:
        rank: LoRA rank (bottleneck dimension).
        alpha: LoRA scaling factor.  Effective scale is ``alpha / rank``.
        dropout: LoRA dropout probability.
        target_modules: Module names to apply LoRA to.  ``None`` auto-detects
            attention projection layers (``q_proj``, ``v_proj``, etc.).
        peft_method: PEFT variant to use.
            ``"lora"`` — standard LoRA (default).
            ``"qlora"`` — Quantized LoRA (delegates to ``apply_lora``;
            quantization is handled by the model loader).
            ``"dora"`` — Weight-Decomposed LoRA (native DoRA on PyTorch;
            falls back to standard LoRA on MLX).
        target_layers: Only apply LoRA to these specific layer indices.
            ``None`` applies to all detected layers.
        until_layer: Only apply LoRA to layers with index < this value.
        after_layer: Only apply LoRA to layers with index >= this value.
    """

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] | None = None
    peft_method: Literal["lora", "qlora", "dora"] = "lora"
    target_layers: list[int] | None = None
    until_layer: int | None = None
    after_layer: int | None = None

    def __post_init__(self) -> None:
        """Validate LoRA configuration after initialization.

        Raises:
            ValueError: If ``rank`` is not >= 1 (the effective scale is
                ``alpha / rank`` — ``rank=0`` is a divide-by-zero), ``alpha`` is
                negative, ``dropout`` is outside ``[0, 1]``, or ``peft_method`` is
                not one of ``"lora"``/``"qlora"``/``"dora"``.
        """
        if self.rank < 1:
            raise ValueError(
                f"rank must be >= 1, got {self.rank}. The effective LoRA scale is "
                "alpha / rank, so rank=0 is a divide-by-zero and rank<0 is invalid."
            )
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}.")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be a probability in [0, 1], got {self.dropout}.")
        valid_methods = ("lora", "qlora", "dora")
        if self.peft_method not in valid_methods:
            raise ValueError(
                f"Unknown peft_method {self.peft_method!r}. Use one of {valid_methods}."
            )


@dataclass
class SteeringConfig:
    """Configuration for activation steering at inference time.

    Steering modifies hidden states *inside* the forward pass so that
    all downstream layers (and therefore LM logits) are affected.

    Attributes:
        method: Steering method.
        layer: Layer index to steer at.  ``None`` uses the probe's layer.
        direction: Override direction vector (``None`` = use class-mean axis).
        scale: Steering intensity.
    """

    method: Literal["nullify", "push_to_mean", "boundary", "custom"] = "nullify"
    layer: int | None = None
    direction: Any = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate steering configuration after initialization."""
        valid = ("nullify", "push_to_mean", "boundary", "custom")
        if self.method not in valid:
            raise ValueError(f"Unknown steering method {self.method!r}. Use one of {valid}.")

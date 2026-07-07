"""Wrapper classes for the trainer module.

Contains ``_TorchProbeWrapper`` (wraps a PyTorch model + probes for
the ``(lm_logits, probe_dict)`` contract) and ``TrainerCallback``
(base class for training lifecycle hooks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class _TorchProbeWrapper:
    """Wraps a PyTorch model + probes for the ``(lm_logits, probe_dict)`` contract."""

    def __init__(self, base: Any, probes: dict[str, Any]) -> None:
        """Initialize the wrapper."""
        self._base = base
        self._probes = probes
        self._probe_modules: list[Any] = [p.module for p in probes.values() if hasattr(p, "module")]

    def __call__(
        self, inputs: torch.Tensor, mask: Any | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run forward pass and collect probe outputs.

        Args:
            inputs: Tokenized input batch of shape ``[B, T-1]``.
            mask: Optional boolean ``[B, T-1]`` mask of valid positions,
                threaded into ``probe.forward`` so ``granularity="response"``
                pooling ignores padding (mirrors ``Model.forward``).

        Returns:
            Tuple of ``(lm_logits, probe_logits_dict)``.
        """
        for probe in self._probes.values():
            probe.clear_captured()

        output = self._base(inputs)
        lm_logits = output.logits if hasattr(output, "logits") else output

        probe_logits: dict[str, Any] = {}
        for name, probe in self._probes.items():
            captured = probe.get_captured_states()
            if captured:
                logits = probe.forward(captured, mask=mask, input_ids=inputs)
                # Squeeze only a trailing single-logit dim; keep [B, T, C] multi-class.
                if logits.ndim > 2 and logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                probe_logits[name] = logits

        return lm_logits, probe_logits

    def parameters(self) -> Any:
        """Yield base model and probe module parameters."""
        yield from self._base.parameters()
        for mod in self._probe_modules:
            yield from mod.parameters()

    def state_dict(self) -> dict[str, Any]:
        """Return state dict with base model and probe parameters."""
        state: dict[str, Any] = {}
        for k, v in self._base.state_dict().items():
            state[k] = v
        for name, probe in self._probes.items():
            module = probe.module if hasattr(probe, "module") else probe
            for k, v in module.state_dict().items():
                state[f"{name}.{k}"] = v
        return state

    def load_state_dict(self, state_dict: dict[str, Any], _strict: bool = True) -> Any:
        """Load state dict into base model and probe modules.

        Base model keys are loaded directly. Probe-specific keys
        use the ``{probe_name}.{param_name}`` format.

        Args:
            state_dict: The state dict to load from.
            _strict: Unused (probe keys are loaded non-strictly).

        Returns:
            The result of ``base.load_state_dict`` (with unmatched keys).
        """
        base_dict: dict[str, Any] = {}
        probe_dicts: dict[str, dict[str, Any]] = {name: {} for name in self._probes}
        for k, v in state_dict.items():
            found = False
            for name in self._probes:
                prefix = f"{name}."
                if k.startswith(prefix):
                    probe_dicts[name][k[len(prefix) :]] = v
                    found = True
                    break
            if not found:
                base_dict[k] = v

        result = self._base.load_state_dict(base_dict, strict=False)
        for name, pdict in probe_dicts.items():
            if pdict:
                probe = self._probes[name]
                module = probe.module if hasattr(probe, "module") else probe
                module.load_state_dict(pdict, strict=False)
        return result

    def train(self, mode: bool = True) -> Any:
        """Set base model and probe modules to train/eval mode.

        Args:
            mode: ``True`` for train mode, ``False`` for eval.

        Returns:
            ``self`` for method chaining.
        """
        self._base.train(mode)
        for mod in self._probe_modules:
            mod.train(mode)
        return self

    def eval(self) -> Any:
        """Set base model and probe modules to eval mode."""
        return self.train(False)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the base model."""
        return getattr(self._base, name)


class TrainerCallback:
    """Base class for trainer callbacks.

    Subclass and override methods to inject custom logic at training
    lifecycle points (logging, evaluation, early stopping, etc.).

    All methods receive ``**kwargs`` for forward-compatibility —
    ignore fields you do not need.
    """

    def on_train_begin(self, **kwargs: Any) -> None:
        """Called before the training loop starts."""

    def on_train_end(self, **kwargs: Any) -> None:
        """Called after the training loop finishes."""

    def on_step_end(self, **kwargs: Any) -> None:
        """Called after every training step."""

    def on_epoch_end(self, **kwargs: Any) -> None:
        """Called after every epoch (for epoch-based trainers)."""

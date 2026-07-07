"""RL-style trainers with probe integration: SFT (+ probe penalty) and DPO.

``RLTrainer`` supports two real algorithms:

- ``"sft"`` — supervised cross-entropy plus a ``beta``-weighted probe penalty.
- ``"dpo"`` — Direct Preference Optimization over ``(chosen, rejected)`` pairs.
  The reference model is handled *without a second model in memory*: the initial
  policy's response log-probs are computed once and cached, so the DPO log-ratio
  is reference-corrected against the starting policy (the standard reference for
  a fresh DPO run).

``"ppo"``/``"grpo"`` raise ``NotImplementedError`` — they need an on-policy
reward signal, which (for hallucination-awareness) is the experiment's design,
not the library's. The library refuses to fake them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from auto_chasm.config import RLConfig
from auto_chasm.history import HistoryEntry
from auto_chasm.logger import get_logger
from auto_chasm.model import Model
from auto_chasm.trainers.base import JointTrainer

logger = get_logger(__name__)


def _pad_batch(seqs: list[Any], pad: int) -> tuple[Any, Any]:
    """Right-pad a list of token sequences to a rectangular ``[B, T]`` MLX array.

    Args:
        seqs: List of per-sequence token-id lists.
        pad: Padding token id.

    Returns:
        Tuple ``(tokens [B, T], lengths [B])`` as MLX arrays.
    """
    import mlx.core as mx

    lengths = [len(s) for s in seqs]
    max_len = max(lengths) if lengths else 1
    padded = [[*s, *([pad] * (max_len - len(s)))] for s in seqs]
    return mx.array(padded), mx.array(lengths)


_NOT_IMPLEMENTED_ALGOS = {
    "ppo": "Proximal Policy Optimization (needs on-policy rollouts + a reward model)",
    "grpo": "Group Relative Policy Optimization (needs grouped sampling + rewards)",
}


class RLTrainer(JointTrainer):
    """SFT-with-probe-penalty (``"sft"``) or Direct Preference Optimization (``"dpo"``).

    - ``algorithm="sft"`` — next-token cross-entropy plus a ``beta``-weighted
      probe penalty (``beta=0`` is pure SFT). Trains on ``{tokens, labels}`` data.
    - ``algorithm="dpo"`` — DPO over ``{chosen, rejected, prompt_len}`` preference
      pairs; the reference is the initial policy, cached as log-probs. Call
      ``.train(preference_data)``.

    ``"ppo"``/``"grpo"`` raise ``NotImplementedError`` — they need an on-policy
    reward (the experiment's design), which the library will not fake.

    Args:
        model: The ``Model`` instance to train.
        rl_config: RL-specific configuration.
        loss_fn: Override loss function (sft path only).
        learning_rate: Peak learning rate.
        num_iters: Total training iterations.
        batch_size: Per-step batch size.
        max_seq_length: Maximum sequence length.
        output_dir: Directory for checkpoints.
    """

    def __init__(
        self,
        model: Model,
        rl_config: RLConfig,
        loss_fn: Callable[..., Any] | None = None,
        learning_rate: float = 2e-5,
        num_iters: int = 500,
        batch_size: int = 4,
        max_seq_length: int = 256,
        output_dir: str = "./checkpoints",
    ) -> None:
        """Initialize the RL trainer.

        Raises:
            ValueError: If ``model`` is not on the MLX backend (the probe-penalty
                path wraps the model in the MLX-only ``_TrainableModel``).
        """
        from auto_chasm.trainers.sft import _require_mlx_backend

        _require_mlx_backend(model, "RLTrainer")
        self.rl_config = rl_config

        # Track whether the user supplied a custom loss so the DPO path can warn
        # it is ignored (DPO always uses its built-in preference loss).
        self._user_loss_fn = loss_fn is not None
        if loss_fn is None:
            loss_fn = self._build_loss_fn()

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            num_iters=num_iters,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            output_dir=output_dir,
        )

    def _build_loss_fn(self) -> Callable[..., Any]:
        """Select the loss function for the configured algorithm.

        Returns:
            Loss function with ``(model, batch, labels, lengths)`` signature.

        Raises:
            NotImplementedError: If a real RL algorithm is requested.
            ValueError: If the algorithm name is unknown.
        """
        algo = self.rl_config.algorithm

        if algo in ("sft", "dpo"):
            # DPO uses its own training loop (_dpo_train); this loss is a
            # placeholder so the JointTrainer base sets up cleanly.
            return self._sft_probe_loss
        if algo in _NOT_IMPLEMENTED_ALGOS:
            raise NotImplementedError(
                f"RLTrainer(algorithm={algo!r}) is not implemented: "
                f"{_NOT_IMPLEMENTED_ALGOS[algo]}. This library will not fake it "
                f"as plain SFT. Use algorithm='sft' for supervised training with "
                f"a probe penalty, or SFTTrainer for plain fine-tuning."
            )
        raise ValueError(f"Unknown RL algorithm: {algo!r}")

    def _sft_probe_loss(
        self, model: Any, batch: Any, labels: Any, lengths: Any
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Supervised cross-entropy plus a beta-weighted probe penalty.

        Computes next-token cross-entropy on the (masked) response tokens
        and adds ``rl_config.beta`` times the summed binary-cross-entropy
        of the attached probes.  This is honest supervised training with
        probe guidance — not DPO/PPO/GRPO.

        Args:
            model: The ``_TrainableModel`` wrapper.
            batch: Tokenised input batch of shape ``[B, T]``.
            labels: Probe label tensor aligned with ``batch``.
            lengths: Per-sequence token ranges ``[B, 2]``.

        Returns:
            ``(total_loss, ntoks, components)``.
        """
        import mlx.core as mx
        import mlx.nn as nn

        if isinstance(labels, dict):
            raise NotImplementedError(
                "RLTrainer trains on a single shared label array; it received a "
                "per-probe labels dict. Per-probe targets are routed only by the "
                "joint Trainer + JointLoss. Use Trainer for multi-head datasets, "
                "or pass a single labels array to RLTrainer."
            )

        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        lm_logits, _ = model(inputs)

        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
        ntoks = mask.sum()
        # Branchless: a Python `if ntoks == 0` would eval a traced array and
        # crash under value_and_grad. Clamp the denominator instead.
        safe_ntoks = mx.maximum(ntoks, 1)

        ce_each = nn.losses.cross_entropy(lm_logits, targets, reduction="none")
        ce = mx.sum(ce_each * mask) / safe_ntoks

        components: dict[str, Any] = {"policy_loss": ce}

        probe_penalty = mx.array(0.0)
        for probe in self.wrapper._probes.values():
            captured = probe.get_captured_states()
            if captured:
                probe_logits = probe.forward(captured)
                if probe_logits.ndim > 2 and probe_logits.shape[-1] == 1:
                    probe_logits = probe_logits.squeeze(-1)
                p_targets = labels[:, 1:].astype(mx.float32)
                # Exclude the -100 ignore sentinel, like JointLoss does. Feeding
                # -100 into BCE adds a 100*logit term that explodes once the
                # probe is trained.
                label_valid = p_targets != -100
                probe_mask = mx.logical_and(mask, label_valid)
                safe_probe = mx.maximum(probe_mask.sum(), 1)
                bce = nn.losses.binary_cross_entropy(
                    probe_logits, p_targets, reduction="none", with_logits=True
                )
                penalty = mx.sum(bce * probe_mask) / safe_probe
                probe_penalty = probe_penalty + penalty

        components["probe_penalty"] = probe_penalty
        total = ce + self.rl_config.beta * probe_penalty
        return total, ntoks, components

    def train(
        self,
        train_data: Any,
        val_data: Any | None = None,
    ) -> dict[str, Any]:
        """Run training.

        Args:
            train_data: Training data.
            val_data: Validation data (optional).

        Returns:
            Dict with keys ``"history"`` (``History``) and ``"output_dir"``
            (the unified trainer return contract; ``run()`` remains the
            lower-level ``History``-returning API).
        """
        logger.info("Starting RL-style training (algorithm=%s)", self.rl_config.algorithm)
        if self.rl_config.algorithm == "dpo":
            return self._dpo_train(train_data)
        history = self.run(train_data, val_data)
        return {"history": history, "output_dir": str(self.output_dir)}

    def _resp_logp(self, logits: Any, tokens: Any, prompt_len: Any, length: Any) -> Any:
        """Sum the per-token log-prob of the response tokens of each sequence.

        Scores positions ``prompt_len <= i < length`` (response tokens only,
        excluding the prompt and any right-padding), as DPO requires.

        Args:
            logits: LM logits ``[B, T, V]``.
            tokens: Token ids ``[B, T]``.
            prompt_len: Per-sequence prompt length ``[B]`` (response starts here).
            length: Per-sequence real token count ``[B]`` (excludes padding).

        Returns:
            Per-sequence summed response log-prob ``[B]``.
        """
        import mlx.core as mx
        import mlx.nn as nn

        logp = nn.log_softmax(logits[:, :-1, :], axis=-1)
        targets = tokens[:, 1:]
        tok_logp = mx.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
        steps = mx.arange(1, tokens.shape[1])[None, :]
        mask = ((steps >= prompt_len[:, None]) & (steps < length[:, None])).astype(mx.float32)
        return (tok_logp * mask).sum(axis=1)

    def _dpo_train(self, preference_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Train the policy with Direct Preference Optimization.

        Each item is ``{"chosen": [ids], "rejected": [ids], "prompt_len": int}``
        (``prompt_len`` defaults to 0 — score the whole sequence). The reference
        is the *initial* policy: its response log-probs are computed once and
        cached, so no second model is held in memory.

        Args:
            preference_data: List of preference examples.

        Returns:
            ``{"history": History, "output_dir": str}``.

        Raises:
            ValueError: If ``preference_data`` is empty.
        """
        import mlx.core as mx
        import mlx.nn as nn

        if not preference_data:
            raise ValueError("preference_data is empty.")

        if self._user_loss_fn:
            logger.warning(
                "RLTrainer(algorithm='dpo') ignores the custom loss_fn; the DPO "
                "path always uses its built-in preference loss."
            )

        model = self.wrapper.model
        beta = self.rl_config.beta
        pad = 0
        msl = self.max_seq_length

        # Honor max_seq_length: keep the first `msl` tokens of each sequence and
        # clamp prompt_len so it never exceeds the truncated length.
        chosen = [list(d["chosen"])[:msl] for d in preference_data]
        rejected = [list(d["rejected"])[:msl] for d in preference_data]
        plen = [
            min(int(d.get("prompt_len", 0)), len(c), len(r))
            for d, c, r in zip(preference_data, chosen, rejected, strict=True)
        ]

        # Reference = initial policy: cache its response log-probs (no grad).
        ref_c, ref_r = [], []
        for i in range(0, len(preference_data), self.batch_size):
            sl = slice(i, i + self.batch_size)
            rc, rr = self._dpo_batch_logp(model, chosen[sl], rejected[sl], plen[sl], pad)
            ref_c.append(mx.stop_gradient(rc))
            ref_r.append(mx.stop_gradient(rr))
        ref_c_all = mx.concatenate(ref_c)
        ref_r_all = mx.concatenate(ref_r)

        def dpo_loss(
            model: Any, ci: Any, li: Any, ri: Any, mi: Any, pl: Any, rc: Any, rr: Any
        ) -> Any:
            """The DPO loss for one batch (model differentiated; refs are constants)."""
            pc = self._resp_logp(model(ci), ci, pl, li)
            pr = self._resp_logp(model(ri), ri, pl, mi)
            logit = beta * ((pc - rc) - (pr - rr))
            # -log sigmoid(logit), averaged over the batch.
            bce = nn.losses.binary_cross_entropy(logit, mx.ones_like(logit), with_logits=True)
            return mx.mean(bce)

        value_and_grad = nn.value_and_grad(model, dpo_loss)
        n = len(preference_data)
        for it in range(1, self.num_iters + 1):
            lo = ((it - 1) * self.batch_size) % n
            idx = list(range(lo, min(lo + self.batch_size, n))) or [0]
            c_arr, c_len = _pad_batch([chosen[j] for j in idx], pad)
            r_arr, r_len = _pad_batch([rejected[j] for j in idx], pad)
            pl = mx.array([plen[j] for j in idx])
            rc = ref_c_all[mx.array(idx)]
            rr = ref_r_all[mx.array(idx)]
            loss, grad = value_and_grad(model, c_arr, c_len, r_arr, r_len, pl, rc, rr)
            self.optimizer.update(model, grad)
            mx.eval(model.state, self.optimizer.state)
            if it % max(self.logging_steps, 1) == 0:
                self._history.append(HistoryEntry(step=it, train_loss=float(loss)))
                self._log(f"DPO step {it}/{self.num_iters}: loss={float(loss):.4f}")

        return {"history": self._history, "output_dir": str(self.output_dir)}

    def _dpo_batch_logp(
        self, model: Any, chosen: list[Any], rejected: list[Any], plen: list[int], pad: int
    ) -> tuple[Any, Any]:
        """Compute reference response log-probs for one batch (no grad)."""
        import mlx.core as mx

        c_arr, c_len = _pad_batch(chosen, pad)
        r_arr, r_len = _pad_batch(rejected, pad)
        pl = mx.array(plen)
        return (
            self._resp_logp(model(c_arr), c_arr, pl, c_len),
            self._resp_logp(model(r_arr), r_arr, pl, r_len),
        )

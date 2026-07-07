"""Tests for PEFT — LoRA, DoRA, QLoRA.

Covers:
  - src/auto_chasm/peft.py (apply_lora/apply_dora/apply_qlora, _require_torch_4bit_base)
  - src/auto_chasm/backends/mlx_backend.py (apply_adapters, method, scale)
  - src/auto_chasm/backends/torch_backend.py (PEFT path)
  - src/auto_chasm/config.py (LoraConfig validation)

Tests named ``test_BUG_*`` and ``test_OK_*`` are regression coverage for specific
past defects and known-correct behavior, respectively.

Model harness: a faithful tiny "LM" with a top-level ``self.layers`` list and
``self_attn.q_proj`` / ``self_attn.v_proj`` modules, exactly what mlx_lm's
``linear_to_lora_layers`` expects (it indexes ``model.layers`` and matches the
leaf module name). This lets LoRA/DoRA/QLoRA wrap real modules without any
network access.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm.backends import Backend
from auto_chasm.peft import (
    _require_torch_4bit_base,
    apply_dora,
    apply_lora,
    apply_qlora,
)

# ---------------------------------------------------------------------------
# Tiny MLX "LM" harness — mlx_lm-compatible structure
# ---------------------------------------------------------------------------


class _Attn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.v_proj(self.q_proj(x))


class _Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.self_attn = _Attn(d)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.self_attn(x)


class _TinyLM(nn.Module):
    """Tiny stand-in LM: top-level ``layers`` + ``self_attn.{q,v}_proj``."""

    def __init__(self, d: int = 64, vocab: int = 32, n: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, d)
        self.layers = [_Block(d) for _ in range(n)]
        self.output_proj = nn.Linear(d, vocab, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h)


def _mk(d: int = 64, n: int = 2) -> _TinyLM:
    mx.random.seed(0)
    return _TinyLM(d=d, n=n)


def _mlx() -> Any:
    return Backend(force="mlx")


_X = mx.array([[1, 2, 3, 4]])


def _count(model: Any, cls: type) -> int:
    return sum(1 for _, m in model.named_modules() if isinstance(m, cls))


# ===========================================================================
# 1. LoRA math — zero-init identity & effective scale (oracles, should PASS)
# ===========================================================================


class TestLoraMath:
    """LoRA forward math: zero-init identity, alpha/rank scale, and known-delta."""

    def test_OK_lora_zero_init_is_identity(self) -> None:
        """Freshly-applied LoRA (lora_b=0) must not change the base output at all."""
        m = _mk()
        base = m(_X)
        apply_lora(
            m, r=4, alpha=8, target_modules=["self_attn.q_proj", "self_attn.v_proj"], backend=_mlx()
        )
        out = m(_X)
        assert float(mx.max(mx.abs(base - out)).item()) == 0.0

    def test_OK_lora_effective_scale_is_alpha_over_rank(self) -> None:
        """Each LoRALinear must carry scale == alpha / rank."""
        from mlx_lm.tuner.lora import LoRALinear

        m = _mk()
        apply_lora(m, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=_mlx())
        scales = [md.scale for _, md in m.named_modules() if isinstance(md, LoRALinear)]
        assert scales, "no LoRALinear was wrapped"
        assert all(abs(s - (8 / 4)) < 1e-9 for s in scales)

    def test_OK_lora_known_B_produces_scaled_delta(self) -> None:
        """Setting lora_b to a known value produces delta = scale * (x@A)@B."""
        from mlx_lm.tuner.lora import LoRALinear

        m = _mk(n=1)
        base = m(_X)
        apply_lora(m, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=_mlx())
        for _, md in m.named_modules():
            if isinstance(md, LoRALinear):
                md.lora_b = mx.ones_like(md.lora_b)
                break
        out = m(_X)
        assert float(mx.max(mx.abs(base - out)).item()) > 0.0


# ===========================================================================
# 2. DoRA on MLX — DoRALinear + magnitude invariant (oracles, should PASS)
# ===========================================================================


class TestDora:
    """DoRA wrapping: magnitude decomposition, zero-init identity, and m=row-norm invariant."""

    def test_OK_dora_builds_dora_linear_not_lora(self) -> None:
        """apply_dora wraps DoRALinear (magnitude-decomposed), never plain LoRALinear."""
        from mlx_lm.tuner.dora import DoRALinear
        from mlx_lm.tuner.lora import LoRALinear

        m = _mk()
        n_q = sum(1 for n, _ in m.named_modules() if n.endswith("self_attn.q_proj"))
        apply_dora(m, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=_mlx())
        assert _count(m, DoRALinear) == n_q
        assert _count(m, LoRALinear) == 0

    def test_OK_dora_zero_init_is_identity(self) -> None:
        """DoRA with zero-init lora_b leaves the base forward unchanged."""
        m = _mk()
        base = m(_X)
        apply_dora(m, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=_mlx())
        out = m(_X)
        assert float(mx.max(mx.abs(base - out)).item()) == 0.0

    def test_OK_dora_magnitude_equals_base_weight_norm(self) -> None:
        """DoRA invariant: m must be initialized to the row-norm of the base weight."""
        from mlx_lm.tuner.dora import DoRALinear

        m = _mk()
        # Snapshot base q_proj weight norms keyed by parent block id.
        expected: dict[int, mx.array] = {}
        for _, blk in m.named_modules():
            if isinstance(blk, _Attn):
                expected[id(blk)] = mx.linalg.norm(blk.q_proj.weight, axis=1)
        apply_dora(m, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=_mlx())
        checked = 0
        for _, blk in m.named_modules():
            if isinstance(blk, _Attn) and isinstance(blk.q_proj, DoRALinear):
                norm = expected[id(blk)]
                assert bool(mx.allclose(blk.q_proj.m, norm, atol=1e-5).item())
                checked += 1
        assert checked > 0


# ===========================================================================
# 3. QLoRA on MLX — genuine quantization
# ===========================================================================


class TestQloraMlx:
    """MLX QLoRA quantization of the base layer under LoRA, and its failure modes."""

    def test_OK_qlora_quantizes_base_under_lora(self) -> None:
        """With divisible dims, the base under each LoRALinear is a QuantizedLinear."""
        from mlx.nn import QuantizedLinear
        from mlx_lm.tuner.lora import LoRALinear

        m = _mk(d=64)
        apply_qlora(
            m,
            r=4,
            alpha=8,
            target_modules=["self_attn.q_proj"],
            bits=4,
            group_size=64,
            backend=_mlx(),
        )
        quant_bases = 0
        for _, md in m.named_modules():
            if isinstance(md, LoRALinear):
                inner = getattr(md, "linear", None)
                if isinstance(inner, QuantizedLinear):
                    quant_bases += 1
        assert quant_bases == 2, "QLoRA must quantize the base of every wrapped module"
        assert m(_X).shape == (1, 4, 32)

    def test_BUG_qlora_silently_skips_quantization_when_dim_not_divisible(self) -> None:
        """QLoRA must NOT silently fall back to plain (fp) LoRA.

        With the default group_size=64 and a model whose q_proj width is not a
        multiple of 64 (here 48), the MLX predicate
        ``module.weight.shape[1] % group_size == 0`` skips EVERY linear, so
        ``nn.quantize`` quantizes nothing — yet apply_qlora logs "QLoRA:
        quantized base to 4-bit" and proceeds to wrap plain fp Linears with
        LoRA. The user believes they trained a quantized model; in reality the
        base is full-precision. This is a silent contract violation (the
        the documented behaviour is "the LoRA wrapper's base is a QuantizedLinear").

        Correct behavior: either quantize (e.g. by validating/adjusting
        group_size) or raise a clear error — never silently produce unquantized
        "QLoRA". A width of 48 is not divisible by any MLX-supported group size
        (32/64/128), so quantization is impossible; the only correct outcome is
        a clear error (naming group_size and the offending width), never a silent
        fp-LoRA fallback.
        """
        from mlx.nn import QuantizedLinear

        m = _mk(d=48)  # 48 % 64 != 0 -> nothing quantizable at default group_size
        with pytest.raises(Exception) as exc:  # noqa: PT011 - message asserted below
            apply_qlora(
                m,
                r=4,
                alpha=8,
                target_modules=["self_attn.q_proj"],
                bits=4,
                group_size=64,
                backend=_mlx(),
            )
        msg = str(exc.value).lower()
        assert "group_size" in msg or "group size" in msg or "quantize" in msg, (
            "QLoRA on a non-divisible width must raise a clear error naming "
            "group_size/quantization, not silently apply plain fp LoRA."
        )
        # And it must NOT have silently produced an fp-LoRA model with no quantized base.
        assert _count(m, QuantizedLinear) == 0

    def test_BUG_qlora_log_says_quantized_when_nothing_quantized(self) -> None:
        """The 'QLoRA: quantized base to 4-bit' log line must not fire when 0 layers quantized.

        Documents the same footgun from the user-visible-signal angle: the only
        feedback the user gets is a log claiming success.
        """
        from mlx.nn import QuantizedLinear

        records: list[str] = []

        class _H(logging.Handler):
            def emit(self, rec: logging.LogRecord) -> None:
                records.append(rec.getMessage())

        log = logging.getLogger("auto_chasm.peft")
        handler = _H()
        log.addHandler(handler)
        old_level = log.level
        log.setLevel(logging.INFO)
        try:
            m = _mk(d=48)
            with pytest.raises(Exception):  # noqa: B017,PT011 - non-divisible width must raise
                apply_qlora(
                    m,
                    r=4,
                    alpha=8,
                    target_modules=["self_attn.q_proj"],
                    bits=4,
                    group_size=64,
                    backend=_mlx(),
                )
        finally:
            log.removeHandler(handler)
            log.setLevel(old_level)

        claimed_quantized = any("quantized base" in r.lower() for r in records)
        actually_quantized = _count(m, QuantizedLinear) > 0
        assert claimed_quantized == actually_quantized, (
            "apply_qlora logged that it quantized the base but quantized nothing."
        )

    def test_OK_qlora_unsupported_group_size_raises_reasonable_error(self) -> None:
        """A group_size MLX rejects (e.g. 16) surfaces a tolerable error.

        MLX only supports group sizes 32/64/128. The predicate accepts
        group_size=16 (16 % 16 == 0) and then mx.quantize raises
        '[quantize] The requested group size 16 is not supported. The supported
        group sizes are 32, 64, and 128.' This is raw mlx (no QLoRA framing) but
        it *does* name the supported values, so it is acceptable DX — recorded
        as a regression oracle rather than a bug.
        """
        m = _mk(d=64)
        with pytest.raises(Exception) as exc:  # noqa: PT011 - message asserted below
            apply_qlora(
                m,
                r=4,
                alpha=8,
                target_modules=["self_attn.q_proj"],
                bits=4,
                group_size=16,
                backend=_mlx(),
            )
        msg = str(exc.value).lower()
        assert "group size" in msg or "group_size" in msg
        assert "32" in msg and "64" in msg and "128" in msg


# ===========================================================================
# 4. QLoRA on PyTorch — intended raise without a 4-bit base (NOT a bug)
# ===========================================================================


class TestQloraTorch:
    """Torch QLoRA requires a 4-bit-loaded base model."""

    def test_OK_torch_qlora_raises_without_4bit_base(self) -> None:
        """Intended: torch QLoRA without a 4-bit base raises NotImplementedError."""
        pytest.importorskip("torch")
        import torch.nn as tnn

        class _TM(tnn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = tnn.Linear(8, 8)

            def forward(self, x: Any) -> Any:
                return self.q_proj(x)

        with pytest.raises(NotImplementedError):
            _require_torch_4bit_base(_TM())

    def test_OK_torch_qlora_accepts_mocked_4bit_base(self) -> None:
        """A model flagged is_loaded_in_4bit passes the require check."""
        pytest.importorskip("torch")
        import torch.nn as tnn

        class _TM(tnn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = tnn.Linear(8, 8)

            def forward(self, x: Any) -> Any:
                return self.q_proj(x)

        m = _TM()
        m.is_loaded_in_4bit = True
        _require_torch_4bit_base(m)  # must not raise


# ===========================================================================
# 5. Edge configs — rank/alpha/double-apply footguns
# ===========================================================================


class TestEdgeConfigs:
    """apply_lora edge configs (bad rank/alpha, double-apply, bad targets) must error clearly."""

    def test_BUG_rank_zero_raises_zerodivisionerror(self) -> None:
        """rank=0 must be rejected with a clear validation error, not a ZeroDivisionError.

        The MLX backend computes scale = alpha / r. With r=0 the user gets a bare
        ``ZeroDivisionError: division by zero`` from deep in mlx_backend — no
        mention of 'rank', LoRA, or the valid range. A rank of 0 is meaningless
        (no bottleneck); it should raise a clear ValueError naming 'rank'.
        """
        m = _mk()
        with pytest.raises(Exception) as exc:
            apply_lora(m, r=0, alpha=8, target_modules=["self_attn.q_proj"], backend=_mlx())
        # The bug: it's a ZeroDivisionError with no actionable message.
        assert not isinstance(exc.value, ZeroDivisionError), (
            "rank=0 surfaces a raw ZeroDivisionError; expected a clear ValueError "
            "explaining that LoRA rank must be a positive integer."
        )
        assert "rank" in str(exc.value).lower() or "r " in str(exc.value).lower()

    def test_BUG_negative_rank_gives_cryptic_error(self) -> None:
        """Negative rank must give a clear validation error, not 'Negative dimensions not allowed'.

        r=-4 reaches mx.zeros((..., -4)) and raises a low-level
        'Negative dimensions not allowed' with no mention of rank/LoRA.
        """
        m = _mk()
        with pytest.raises(Exception) as exc:
            apply_lora(m, r=-4, alpha=8, target_modules=["self_attn.q_proj"], backend=_mlx())
        assert "rank" in str(exc.value).lower(), (
            "negative rank yields a cryptic 'Negative dimensions not allowed' "
            "error; expected a clear message naming 'rank'."
        )

    def test_BUG_alpha_zero_makes_adapters_silent_noops(self) -> None:
        """alpha=0 yields scale=0 so adapters can never change the output — a silent no-op.

        With alpha=0 the effective scale is 0/r = 0, so LoRA adapters are frozen
        no-ops: training them changes nothing and the user gets no warning. This
        is a footgun worth at least a warning (or a rejection). We assert the
        desired behavior: a non-trivial LoRA config should be able to change the
        output (here we drive lora_b to large values and require an effect), or
        the library should warn. It currently does neither.
        """
        from mlx_lm.tuner.lora import LoRALinear

        records: list[str] = []

        class _H(logging.Handler):
            def emit(self, rec: logging.LogRecord) -> None:
                records.append(rec.getMessage())

        log = logging.getLogger("auto_chasm.peft")
        handler = _H()
        log.addHandler(handler)
        old_level = log.level
        log.setLevel(logging.WARNING)
        m = _mk(n=1)
        base = m(_X)
        try:
            apply_lora(m, r=4, alpha=0, target_modules=["self_attn.q_proj"], backend=_mlx())
        finally:
            log.removeHandler(handler)
            log.setLevel(old_level)
        # Force the adapter to a large value: a *correctly* scaled LoRA would now
        # change the output. With scale=0 it cannot.
        for _, md in m.named_modules():
            if isinstance(md, LoRALinear):
                md.lora_b = mx.ones_like(md.lora_b) * 100.0
                break
        out = m(_X)
        changed = float(mx.max(mx.abs(base - out)).item()) > 0.0
        warned = any("alpha" in r.lower() for r in records)
        assert changed or warned, (
            "alpha=0 makes scale=0, so the adapter is a permanent no-op regardless "
            "of its weights — apply_lora should reject alpha=0 or warn loudly."
        )

    def test_BUG_applying_lora_twice_gives_cryptic_error(self) -> None:
        """Applying LoRA a second time must raise a clear 'already adapted' error.

        Re-applying LoRA on an already-wrapped model raises a raw mlx_lm
        "Can't convert layer of type LoRALinear to LoRA" with no guidance. The
        library should detect existing adapters and raise an actionable message
        (e.g. 'model already has LoRA adapters; remove them first').
        """
        m = _mk()
        apply_lora(m, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=_mlx())
        with pytest.raises(Exception) as exc:
            apply_lora(m, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=_mlx())
        msg = str(exc.value).lower()
        assert "already" in msg or "adapter" in msg or "lora" in msg and "remove" in msg, (
            "double-apply raises a cryptic mlx_lm conversion error; expected a "
            "clear message that the model already has adapters."
        )

    def test_OK_empty_target_modules_after_filter_raises_clear_error(self) -> None:
        """Layer targeting that filters out everything raises a clear ValueError."""
        m = _mk(n=2)
        with pytest.raises(ValueError, match="filtered out|nothing would be adapted"):
            apply_lora(
                m,
                r=4,
                alpha=8,
                target_modules=["self_attn.q_proj"],
                target_layers=[99],  # no such layer
                backend=_mlx(),
            )

    def test_OK_nonexistent_target_modules_raise_clear_error(self) -> None:
        """Targeting modules that don't exist raises a clear, listing ValueError."""
        m = _mk()
        with pytest.raises(ValueError, match="matched no modules"):
            apply_lora(m, r=4, alpha=8, target_modules=["does_not_exist"], backend=_mlx())


# ===========================================================================
# 6. LoraConfig validation — config accepts nonsense silently
# ===========================================================================


class TestLoraConfigValidation:
    """LoraConfig should reject invalid rank and dropout at construction time."""

    def test_BUG_loraconfig_accepts_rank_zero(self) -> None:
        """LoraConfig(rank=0) should be rejected at construction (it later explodes).

        LoraConfig has no __post_init__ validation, so rank=0 / rank<0 /
        alpha<0 / dropout out of [0,1] all construct happily and only blow up
        (cryptically) much later inside the backend. A user-facing config object
        should validate its own fields.
        """
        from auto_chasm.config import LoraConfig

        with pytest.raises((ValueError, TypeError)):
            LoraConfig(rank=0)

    def test_BUG_loraconfig_accepts_negative_rank(self) -> None:
        """LoraConfig(rank=-1) should be rejected at construction."""
        from auto_chasm.config import LoraConfig

        with pytest.raises((ValueError, TypeError)):
            LoraConfig(rank=-1)

    def test_BUG_loraconfig_accepts_out_of_range_dropout(self) -> None:
        """LoraConfig(dropout=2.0) is not a valid probability and should be rejected."""
        from auto_chasm.config import LoraConfig

        with pytest.raises((ValueError, TypeError)):
            LoraConfig(dropout=2.0)


# ===========================================================================
# 7. Trainability — only adapters trainable, base frozen
# ===========================================================================


class TestTrainability:
    """MLX trainability: only adapters train and frozen base weights never move."""

    def test_OK_mlx_only_adapters_trainable_when_base_frozen(self) -> None:
        """After freeze + unfreeze-lora, only lora_a/lora_b are trainable."""
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.lora import LoRALinear

        m = _mk()
        apply_lora(m, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=_mlx())
        m.freeze()
        for _, md in m.named_modules():
            if isinstance(md, LoRALinear):
                md.unfreeze(keys=["lora_a", "lora_b"])
        trainable = dict(tree_flatten(m.trainable_parameters()))
        assert trainable, "no trainable params after unfreezing adapters"
        assert all("lora_" in k for k in trainable), (
            f"non-adapter params are trainable: {[k for k in trainable if 'lora_' not in k]}"
        )

    def test_OK_mlx_frozen_base_weights_do_not_move_after_step(self) -> None:
        """A training step updates adapters but leaves base weights byte-identical."""
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.lora import LoRALinear

        m = _mk(n=1)
        apply_lora(m, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=_mlx())
        m.freeze()
        for _, md in m.named_modules():
            if isinstance(md, LoRALinear):
                md.unfreeze(keys=["lora_a", "lora_b"])

        base_before = {
            k: mx.array(v) for k, v in tree_flatten(m.parameters()).__iter__() if "lora_" not in k
        }

        import mlx.optimizers as optim

        def loss_fn(model: Any) -> mx.array:
            return mx.sum(model(_X) ** 2)

        opt = optim.SGD(learning_rate=0.1)
        # Make lora_a non-zero so the gradient flows to lora_b (B starts at 0).
        for _, md in m.named_modules():
            if isinstance(md, LoRALinear):
                md.lora_a = md.lora_a + 0.1
        grad_fn = nn.value_and_grad(m, loss_fn)
        _, grads = grad_fn(m)
        opt.update(m, grads)
        mx.eval(m.parameters())

        base_after = {k: v for k, v in tree_flatten(m.parameters()) if "lora_" not in k}
        for k, before in base_before.items():
            after = base_after[k]
            assert bool(mx.array_equal(before, after).item()), f"base weight {k} moved"


# ===========================================================================
# 8. Cross-backend — torch LoRA zero-init equals base + trainability
# ===========================================================================


class TestTorchLora:
    """Torch LoRA via PEFT: identity at init and only adapters trainable."""

    @staticmethod
    def _hf_tiny() -> Any:
        import torch.nn as tnn

        class _Cfg:
            model_type = "llama"

        class _HFTiny(tnn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = tnn.Linear(8, 8)
                self.v_proj = tnn.Linear(8, 8)
                self.config = _Cfg()

            def forward(self, x: Any) -> Any:
                return self.v_proj(self.q_proj(x))

            def prepare_inputs_for_generation(self, input_ids: Any, **kw: Any) -> dict[str, Any]:
                return {"input_ids": input_ids}

        return _HFTiny()

    def test_OK_torch_lora_zero_init_equals_base(self) -> None:
        """Torch LoRA via PEFT is identity at init (lora_b=0)."""
        pytest.importorskip("torch")
        import torch

        torch.manual_seed(0)
        m = self._hf_tiny()
        x = torch.randn(2, 8)
        base = m(x).detach().clone()
        wrapped = apply_lora(
            m, r=4, alpha=8, target_modules=["q_proj", "v_proj"], backend=Backend(force="torch")
        )
        out = wrapped.base_model.model(x)
        assert float((base - out).abs().max().item()) == 0.0

    def test_OK_torch_lora_only_adapters_trainable(self) -> None:
        """After PEFT wrap, only lora_* params require grad; base is frozen."""
        pytest.importorskip("torch")
        import torch

        torch.manual_seed(0)
        m = self._hf_tiny()
        wrapped = apply_lora(
            m, r=4, alpha=8, target_modules=["q_proj", "v_proj"], backend=Backend(force="torch")
        )
        trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
        assert trainable, "no trainable params"
        non_lora = [n for n in trainable if "lora_" not in n]
        assert not non_lora, f"non-adapter params are trainable: {non_lora}"


class TestAdapterCheckpointRoundtrip:
    """LoRA adapters must actually round-trip through save/load (not silently no-op)."""

    def test_BUG_load_adapters_applies_saved_weights(self) -> None:
        """``save_adapters`` -> ``load_adapters`` must reproduce the trained adapters.

        Regression: ``load_adapters`` used ``model.update(mx.load(path))`` — a FLAT
        dotted-key dict fed to ``update`` (which expects a nested tree) — so the
        adapters silently failed to load (the model kept its zero-init adapters).
        The previous oracle mocked the backend and missed it.
        """
        import tempfile

        from mlx.utils import tree_flatten, tree_unflatten

        backend = _mlx()
        tgt = ["self_attn.q_proj", "self_attn.v_proj"]

        src = _mk()
        apply_lora(src, r=4, alpha=8, target_modules=tgt, backend=backend)
        # Perturb only the ADAPTER params (not the base): save_adapters now saves
        # adapters only, so this exercises the real LoRA round-trip (same base — both
        # _mk() seed 0 — with swapped adapters). Perturbing the base would not survive.
        perturbed = [
            (k, v + 0.5)
            for k, v in tree_flatten(src.trainable_parameters())
            if k.rsplit(".", 1)[-1] in ("lora_a", "lora_b")
        ]
        src.update(tree_unflatten(perturbed))
        mx.eval(src.parameters())
        out_src = src(_X)

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/adapters.safetensors"
            backend.wrapping.save_adapters(src, path)

            dst = _mk()
            apply_lora(dst, r=4, alpha=8, target_modules=tgt, backend=backend)
            out_dst_init = dst(_X)
            backend.wrapping.load_adapters(dst, path)
            mx.eval(dst.parameters())
            out_dst_loaded = dst(_X)

        # The loaded adapters must change the output (they were applied)...
        assert float(mx.max(mx.abs(out_dst_loaded - out_dst_init)).item()) > 1e-5, (
            "load_adapters did not change the output — adapters were not applied"
        )
        # ...and reproduce the saved model's output exactly.
        assert float(mx.max(mx.abs(out_dst_loaded - out_src)).item()) < 1e-4, (
            "loaded adapters do not reproduce the saved model's output"
        )

    def test_BUG_load_adapters_raises_on_total_mismatch(self) -> None:
        """A wrong adapter file (no matching keys) must raise, not silently no-op."""
        import tempfile

        backend = _mlx()
        dst = _mk()
        apply_lora(dst, r=4, alpha=8, target_modules=["self_attn.q_proj"], backend=backend)
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/bogus.safetensors"
            mx.save_safetensors(path, {"totally.unrelated.key": mx.zeros((2, 2))})
            with pytest.raises(ValueError, match="no parameters matching"):
                backend.wrapping.load_adapters(dst, path)

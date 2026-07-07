"""torch mixed-precision tests: bf16 weight-cast + fp16 autocast/scaler.

bf16 casts the frozen base to bfloat16 (probe/optimizer stay fp32; no scaling).
fp16 keeps weights fp32 but runs the forward under autocast + a GradScaler (fp16's
narrow range needs loss scaling). These tests pin the dtypes, prove the scaler is
genuinely active for fp16 (not a silent no-op), prove bf16 tracks fp32 numerically,
and exercise the machinery together (grad accumulation, clipping, LoRA).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from auto_chasm import Model, ProbeConfig  # noqa: E402
from auto_chasm.config import TrainingConfig  # noqa: E402
from auto_chasm.trainers.loss import JointLoss  # noqa: E402
from auto_chasm.trainers.trainer import Trainer  # noqa: E402


def _model(seed: int = 0) -> Model:
    from tests.conftest import DummyTokenizer, _make_torch_tiny_mlp

    torch.manual_seed(seed)
    base = _make_torch_tiny_mlp(hidden_dim=16, vocab_size=32, num_layers=2)

    class _C:
        hidden_size = 16
        num_hidden_layers = 2

    base.config = _C()
    m = Model(base, DummyTokenizer(), backend_name="torch")
    m.attach_probe(ProbeConfig(name="p", layers=[0]))  # default binary head
    return m


def _bce_loss() -> JointLoss:
    return JointLoss(weights={"lm_head": 0.0}, losses={"p": "bce"})


def _data(n: int = 16) -> list[dict[str, list[int]]]:
    return [{"tokens": [1, 2, 3, 4, 5], "labels": [1, 1, 1, 1, 1]} for _ in range(n)]


def _trainer(m: Model, mp: str, **kw: object) -> Trainer:
    return Trainer(
        model=m,
        loss_fn=_bce_loss(),
        config=TrainingConfig(mixed_precision=mp, batch_size=4),
        num_iters=kw.get("num_iters", 40),
        learning_rate=kw.get("learning_rate", 5e-2),
        grad_accum_steps=kw.get("grad_accum_steps", 1),
        grad_clip_norm=kw.get("grad_clip_norm", 1.0),
        verbose=False,
    )


def _run_steps(tr: Trainer, m: Model, n: int) -> tuple[float, float]:
    batches = tr.iterate(_data())
    first = tr.step(next(batches))["loss"]
    for _ in range(n - 2):
        tr.step(next(batches))
    last = tr.step(next(batches))["loss"]
    return first, last


# --- bf16: casts the frozen base, probe stays fp32, still trains ------------------


def test_bf16_casts_base_keeps_probe_fp32_and_trains() -> None:
    m = _model()
    tr = _trainer(m, "bf16")
    first, last = _run_steps(tr, m, 40)
    assert next(m.model.parameters()).dtype == torch.bfloat16  # base cast
    assert next(m._probes["p"].module.parameters()).dtype == torch.float32  # probe fp32
    assert last < 0.5 * first, f"bf16 did not reduce loss: {first} -> {last}"


def test_bf16_tracks_fp32_numerically() -> None:
    """bf16 (fp32 exponent range) should reach a loss close to fp32 on the same data."""
    m32 = _model(seed=7)
    _, last32 = _run_steps(_trainer(m32, "fp32"), m32, 40)
    m16 = _model(seed=7)
    _, last16 = _run_steps(_trainer(m16, "bf16"), m16, 40)
    assert last32 < 0.05 and last16 < 0.05  # both essentially solve the toy task
    assert abs(last16 - last32) < 0.05  # bf16 does not diverge from fp32


# --- fp16: weights stay fp32, GradScaler is genuinely active ----------------------


def test_fp16_keeps_weights_fp32_and_trains() -> None:
    m = _model()
    tr = _trainer(m, "fp16")
    first, last = _run_steps(tr, m, 40)
    # Autocast leaves the stored weights fp32 (only the forward math is fp16).
    assert next(m.model.parameters()).dtype == torch.float32
    assert next(m._probes["p"].module.parameters()).dtype == torch.float32
    assert last < 0.5 * first, f"fp16 did not reduce loss: {first} -> {last}"


def test_fp16_gradscaler_is_active_not_a_noop() -> None:
    """The fp16 GradScaler really scales (scale != 1); bf16/fp32 leave it disabled."""
    m = _model()
    tr = _trainer(m, "fp16")
    _run_steps(tr, m, 5)
    scaler = tr._torch_step_state["scaler"]
    assert scaler.is_enabled()
    assert scaler.get_scale() != 1.0  # real loss scaling (fp16 default starts at 65536)

    m2 = _model()
    tr2 = _trainer(m2, "bf16")
    _run_steps(tr2, m2, 3)
    assert not tr2._torch_step_state["scaler"].is_enabled()  # bf16 needs no scaling
    assert tr2._torch_step_state["scaler"].get_scale() == 1.0


# --- The full train() loop (not just the escape hatch) honors mixed precision ------


@pytest.mark.parametrize("mp", ["bf16", "fp16"])
def test_full_train_loop_reduces_loss(mp: str, tmp_path) -> None:  # noqa: ANN001
    m = _model()
    tr = Trainer(
        model=m,
        loss_fn=_bce_loss(),
        config=TrainingConfig(mixed_precision=mp, batch_size=4, output_dir=str(tmp_path)),
        num_iters=40,
        learning_rate=5e-2,
        logging_steps=1,
        verbose=False,
    )
    result = tr.train(_data())
    losses = result["history"].train_losses
    assert losses and all(x == x for x in losses)  # no NaN (x==x fails for NaN)
    assert losses[-1] < 0.5 * losses[0], f"{mp} full-loop loss: {losses[0]} -> {losses[-1]}"


# --- Combined: all the machinery together ------------------------------------


def test_fp16_with_grad_accum_and_clipping() -> None:
    """fp16 + gradient accumulation + grad clipping (unscale before clip) trains, no NaN."""
    m = _model()
    tr = _trainer(m, "fp16", grad_accum_steps=2, grad_clip_norm=0.5, num_iters=40)
    first, last = _run_steps(tr, m, 40)
    assert last == last  # not NaN
    assert last < first


def test_bf16_multiclass_ce_probe_trains() -> None:
    """bf16 with a 3-class CE probe (a different loss path than BCE) still converges."""
    from tests.conftest import DummyTokenizer, _make_torch_tiny_mlp

    torch.manual_seed(0)
    base = _make_torch_tiny_mlp(hidden_dim=16, vocab_size=32, num_layers=2)

    class _C:
        hidden_size = 16
        num_hidden_layers = 2

    base.config = _C()
    m = Model(base, DummyTokenizer(), backend_name="torch")
    m.attach_probe(ProbeConfig(name="p", layers=[0], module_config={"out_features": 3}))
    tr = Trainer(
        model=m,
        loss_fn=JointLoss(weights={"lm_head": 0.0}, losses={"p": "ce"}),
        config=TrainingConfig(mixed_precision="bf16", batch_size=4),
        num_iters=40,
        learning_rate=5e-2,
        verbose=False,
    )
    data = [{"tokens": [1, 2, 3, 4, 5], "labels": [2, 2, 2, 2, 2]} for _ in range(16)]
    batches = tr.iterate(data)
    first = tr.step(next(batches))["loss"]
    for _ in range(38):
        tr.step(next(batches))
    last = tr.step(next(batches))["loss"]
    assert next(m.model.parameters()).dtype == torch.bfloat16
    assert last < 0.5 * first


def test_mp_none_config_defaults_fp32() -> None:
    """No mixed_precision (fp32 default) leaves everything fp32."""
    m = _model()
    tr = _trainer(m, "fp32")
    _run_steps(tr, m, 5)
    assert next(m.model.parameters()).dtype == torch.float32
    assert not tr._torch_step_state["scaler"].is_enabled()

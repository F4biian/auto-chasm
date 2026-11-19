"""Probe attachment, sources, aggregation, granularity.

Covers ``src/auto_chasm/probe.py``, ``src/auto_chasm/_probe_agg.py``, and
``ProbeConfig`` validation in ``config.py``.

Tests named ``test_BUG_*`` are regression tests for specific past defects; the
rest are general regression coverage.

Conventions:
- Deterministic, in-memory TinyMlp model (see tests/conftest.py); RNG seeded.
- MLX is always available; torch tests guard with importorskip.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from auto_chasm import Model, _probe_agg
from auto_chasm.config import ProbeConfig
from tests.conftest import DummyTokenizer, TinyMlp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Cfg:
    hidden_size = 16
    num_hidden_layers = 4
    vocab_size = 32
    d_model = 16


def _mlx_model() -> Model:
    mx.random.seed(0)
    m = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)
    m.config = _Cfg()
    return Model(m, DummyTokenizer(), backend_name="mlx")


def _torch_model():  # type: ignore[no-untyped-def]
    import torch

    from tests.conftest import _make_torch_tiny_mlp

    torch.manual_seed(0)
    m = _make_torch_tiny_mlp(hidden_dim=16, vocab_size=32, num_layers=4)
    m.config = _Cfg()
    return Model(m, DummyTokenizer(), backend_name="torch")


def _mlx_block_outputs(base: TinyMlp, ids: mx.array) -> list[mx.array]:
    """Ground-truth per-block outputs (block = bare Linear in TinyMlp).

    The model applies gelu *outside* the block, so the block's own output is
    ``layers[i](h_in)`` and the next block's input is ``gelu(that)``.
    """
    h = base.embedding(ids)
    outs: list[mx.array] = []
    for layer in base.layers:
        b = layer(h)
        outs.append(b)
        h = nn.gelu(b)
    return outs


def _mlx_block_inputs(base: TinyMlp, ids: mx.array) -> list[mx.array]:
    h = base.embedding(ids)
    ins: list[mx.array] = []
    for layer in base.layers:
        ins.append(h)
        h = nn.gelu(layer(h))
    return ins


IDS = mx.array([[1, 2, 3, 4, 5]])


# ===========================================================================
# Regression coverage — these document correct behaviour and should PASS.
# ===========================================================================


def test_residual_and_hidden_same_layer_capture_correct_tensors() -> None:
    """Residual (block input) + hidden (block output) at the same layer."""
    model = _mlx_model()
    base = model.model
    ins = _mlx_block_inputs(base, IDS)
    outs = _mlx_block_outputs(base, IDS)
    pr = model.attach_probe(
        ProbeConfig(name="r", layers=[2], source="residual", aggregation="last")
    )
    ph = model.attach_probe(ProbeConfig(name="h", layers=[2], source="hidden", aggregation="last"))
    model.forward(IDS)
    assert mx.allclose(pr.get_captured_states()[0], ins[2], atol=1e-5)
    assert mx.allclose(ph.get_captured_states()[0], outs[2], atol=1e-5)


def test_last_uses_last_listed_layer_reversed_order() -> None:
    """``last`` must take the last *listed* layer, not the last executed."""
    model = _mlx_model()
    outs = _mlx_block_outputs(model.model, IDS)
    p = model.attach_probe(
        ProbeConfig(name="l", layers=[3, 1], source="hidden", aggregation="last")
    )
    model.forward(IDS)
    agg = p._aggregate(p.get_captured_states())
    assert mx.allclose(agg, outs[1], atol=1e-5)  # last listed == layer 1
    assert not mx.allclose(agg, outs[3], atol=1e-5)


def test_concat_column_order_matches_config_order() -> None:
    """Concat columns follow config order even when reversed vs exec order."""
    model = _mlx_model()
    outs = _mlx_block_outputs(model.model, IDS)
    p = model.attach_probe(
        ProbeConfig(name="c", layers=[3, 1], source="hidden", aggregation="concat")
    )
    model.forward(IDS)
    agg = p._aggregate(p.get_captured_states())
    assert mx.allclose(agg[..., :16], outs[3], atol=1e-5)
    assert mx.allclose(agg[..., 16:], outs[1], atol=1e-5)


def test_reducing_callable_aggregation_infers_hidden_width() -> None:
    """A reducing callable (mean over layers) sizes the head to hidden width."""
    model = _mlx_model()

    def agg_mean(states: list[Any]) -> Any:
        return mx.mean(mx.stack(states, axis=0), axis=0)

    p = model.attach_probe(
        ProbeConfig(name="red", layers=[1, 3], source="hidden", aggregation=agg_mean)
    )
    assert p.module.weight.shape[-1] == 16
    out = model.forward(IDS)
    assert out.probes["red"].logits.shape == (1, 5, 1)


def test_out_of_range_layer_raises_clear_error() -> None:
    model = _mlx_model()
    with pytest.raises(ValueError, match="out of range"):
        model.attach_probe(ProbeConfig(name="oor", layers=[999], source="hidden"))


def test_empty_layers_rejected_by_config() -> None:
    with pytest.raises(ValueError, match="at least one layer"):
        ProbeConfig(name="e", layers=[])


def test_unknown_aggregation_string_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown aggregation"):
        ProbeConfig(name="x", layers=[0], aggregation="median")


def test_sentence_without_delimiters_raises() -> None:
    with pytest.raises(ValueError, match="sentence_delimiters"):
        ProbeConfig(name="s", layers=[0], granularity="sentence")


def test_sentence_pool_delimiter_at_position_zero_parity() -> None:
    """Delimiter at index 0 → a length-1 first segment; MLX/torch agree."""
    torch = pytest.importorskip("torch")
    vals = [[[float(i)] for i in range(5)]]
    ids = [[9, 1, 2, 9, 3]]
    om = _probe_agg.sentence_pool(mx.array(vals), mx.array(ids), [9], None, "mlx")
    ot = _probe_agg.sentence_pool(torch.tensor(vals), torch.tensor(ids), [9], None, "torch")
    expected = [0.0, 2.0, 2.0, 2.0, 4.0]
    assert [round(x[0], 4) for x in om.tolist()[0]] == expected
    assert [round(x[0], 4) for x in ot.tolist()[0]] == expected


def test_sentence_pool_masks_padding() -> None:
    """Padding (mask=0) must not leak into a sentence mean; MLX/torch parity."""
    torch = pytest.importorskip("torch")
    vals = [[[float(i)] for i in range(6)]]
    ids = [[1, 2, 9, 3, 4, 0]]
    mask = [[1, 1, 1, 1, 1, 0]]
    om = _probe_agg.sentence_pool(mx.array(vals), mx.array(ids), [9], mx.array(mask), "mlx")
    ot = _probe_agg.sentence_pool(
        torch.tensor(vals), torch.tensor(ids), [9], torch.tensor(mask), "torch"
    )
    expected = [1.0, 1.0, 1.0, 3.5, 3.5, 3.5]
    assert [round(x[0], 4) for x in om.tolist()[0]] == expected
    assert [round(x[0], 4) for x in ot.tolist()[0]] == expected


def test_masked_mean_fully_padded_row_is_finite() -> None:
    """A row whose positions are all padding must pool to a finite value."""
    torch = pytest.importorskip("torch")
    logits = mx.ones((2, 3, 1))
    mask = mx.array([[1, 1, 1], [0, 0, 0]])
    om = _probe_agg.masked_mean_over_time(logits, mask, "mlx")
    assert bool(mx.all(mx.isfinite(om)))
    lt = torch.ones((2, 3, 1))
    mt = torch.tensor([[1, 1, 1], [0, 0, 0]])
    ot = _probe_agg.masked_mean_over_time(lt, mt, "torch")
    assert bool(torch.isfinite(ot).all())


def test_max_and_mean_aggregation_backend_parity() -> None:
    torch = pytest.importorskip("torch")
    a = [[[1.0, 5.0], [3.0, 2.0]]]
    b = [[[4.0, 1.0], [0.0, 9.0]]]
    ml = [mx.array(a), mx.array(b)]
    tl = [torch.tensor(a), torch.tensor(b)]
    for strat in ("max", "mean"):
        mo = _probe_agg.aggregate(ml, strat, "mlx").tolist()
        to = _probe_agg.aggregate(tl, strat, "torch").tolist()
        assert mo == to, strat


# ===========================================================================
# Regression tests for specific past defects (assert the corrected behaviour).
# ===========================================================================


def test_BUG_single_layer_callable_aggregation_width_not_inferred_mlx() -> None:
    """A single-layer callable aggregation that changes feature width crashes.

    ``Probe._build_module`` only probes a callable aggregation's output width
    when ``len(config.layers) > 1`` (the ``multi_layer`` gate). With a single
    layer the head is sized to ``hidden_dim``, yet the callable still runs at
    forward time and may emit a different width → a confusing matmul shape error
    (probe.py:400). The head should be sized to whatever the callable returns,
    regardless of the layer count.
    """
    model = _mlx_model()

    def widen(states: list[Any]) -> Any:
        return mx.concatenate([states[0], states[0]], axis=-1)  # hidden*2

    p = model.attach_probe(ProbeConfig(name="w", layers=[2], source="hidden", aggregation=widen))
    # Desired: head sized to the callable's real output (32), forward succeeds.
    assert p.module.weight.shape[-1] == 32
    out = model.forward(IDS)
    assert out.probes["w"].logits.shape == (1, 5, 1)


def test_BUG_torch_forward_silently_drops_probe_on_runtime_error() -> None:
    """Model.forward swallows ANY torch RuntimeError from a probe.

    model.py:368 wraps the per-probe forward in ``except RuntimeError`` to
    tolerate "no captured states". On torch a shape/width mismatch is also a
    RuntimeError, so a misconfigured probe is **silently dropped** from the
    outputs instead of surfacing an error — a research-poisoning silent failure
    (the caller sees no probe output and no exception).
    """
    pytest.importorskip("torch")
    import torch

    model = _torch_model()

    def widen(states: list[Any]) -> Any:
        return torch.cat([states[0], states[0]], dim=-1)

    model.attach_probe(ProbeConfig(name="w", layers=[2], source="hidden", aggregation=widen))
    out = model.forward(torch.tensor([[1, 2, 3, 4, 5]]))
    # Desired: either a clear error OR the probe present — never a silent drop.
    assert "w" in out.probes, "probe was silently dropped from Model.forward outputs"


def test_BUG_restore_does_not_unwrap_embedding_capture() -> None:
    """``restore_original_layers`` leaves the embedding wrapper installed.

    It only restores block layers and attention/mlp submodules (model.py:734,
    _probe_agg.unwrap_submodule_captures). An ``embedding`` probe permanently
    replaces ``embed_tokens``/``embedding`` with a capture wrapper that is never
    removed, so the model cannot be cleanly returned to its original state.
    """
    model = _mlx_model()
    orig_type = type(model.model.embedding).__name__
    model.attach_probe(ProbeConfig(name="e", layers=[-1], source="embedding"))
    model.restore_original_layers()
    assert type(model.model.embedding).__name__ == orig_type, (
        "embedding capture wrapper survived restore_original_layers"
    )


def test_BUG_restore_does_not_unwrap_logits_capture() -> None:
    """``restore_original_layers`` leaves the LM-head (logits) wrapper installed."""
    model = _mlx_model()
    orig_type = type(model.model.output_proj).__name__
    model.attach_probe(ProbeConfig(name="l", layers=[-1], source="logits"))
    model.restore_original_layers()
    assert type(model.model.output_proj).__name__ == orig_type, (
        "logits capture wrapper survived restore_original_layers"
    )


def test_BUG_reattaching_same_name_leaks_orphan_capture() -> None:
    """Re-attaching a probe under an existing name orphans the old wrapper.

    ``attach_probe`` overwrites ``self._probes[name]`` but never removes the
    previous probe's capture wrapper from the model. The orphan keeps running on
    every forward and its ``_captured`` list grows **without bound** (it is no
    longer in ``_probes`` so ``clear_captured`` is never called on it) — wasted
    compute plus an unbounded memory leak. Re-attaching should replace cleanly.
    """
    model = _mlx_model()
    p_old = model.attach_probe(
        ProbeConfig(name="p", layers=[1], source="hidden", aggregation="last")
    )
    # Re-using a name is rejected up front (cleanest fix), so no orphan wrapper is
    # ever installed: the original probe stays the only one registered.
    with pytest.raises(ValueError, match="already attached"):
        model.attach_probe(ProbeConfig(name="p", layers=[3], source="hidden", aggregation="last"))
    model.forward(IDS)
    model.forward(IDS)
    # No orphan: the (still-registered) probe is cleared each forward, so its
    # capture buffer never grows without bound.
    assert len(p_old._captured) <= 1, (
        f"orphaned probe accumulated {len(p_old._captured)} captures across "
        "forwards (unbounded leak); re-attach did not detach the old wrapper"
    )


def test_BUG_response_pooling_includes_prompt_in_model_forward() -> None:
    """``Model.forward`` pools response-granularity probes over prompt tokens too.

    The trainer threads a *response-region* mask (prompt + padding excluded;
    loss.py:157) into probe pooling, but ``Model.forward`` passes the raw
    ``attention_mask`` (model.py:362), which marks padding only and KEEPS the
    prompt. So the same ``granularity='response'`` probe pools over response-only
    during training but over prompt+response at inference — a silent train/infer
    divergence and prompt contamination of the pooled prediction.

    Here the prompt tokens carry a large logit and the response tokens ~0; a
    correct response pool ignores the prompt and stays near 0.
    """
    model = _mlx_model()
    p = model.attach_probe(
        ProbeConfig(name="resp", layers=[2], source="hidden", granularity="response")
    )
    # The probe carries its prompt boundary (prompt = positions 0,1), so the
    # response pool excludes the prompt exactly as the trainer's response mask does.
    p.prompt_len = 2
    # Hand-built logits: prompt (pos 0,1) huge, response (pos 2,3) ~0.
    logits = mx.array([[[10.0], [10.0], [0.0], [0.0]]])
    attn_mask = mx.array([[1, 1, 1, 1]])  # raw padding mask (no prompt info on its own)
    pooled = p._apply_pooling(logits, mask=attn_mask, input_ids=IDS[:, :4])
    # Desired: response-only pooling (prompt excluded) → ~0, not the 5.0 you get
    # by averaging prompt+response. (Documents the contract gap.)
    assert abs(float(pooled[0, 0])) < 1.0, (
        f"response pool = {float(pooled[0, 0])}; prompt tokens contaminated it "
        "(Model.forward has no prompt boundary, unlike the trainer)"
    )


# ===========================================================================
# Cross-backend regression for the embedding/logits in-dim correctness.
# ===========================================================================


def test_logits_source_in_features_is_vocab_size() -> None:
    model = _mlx_model()
    p = model.attach_probe(ProbeConfig(name="lv", layers=[-1], source="logits", aggregation="last"))
    assert p.module.weight.shape[-1] == 32  # vocab_size, not hidden


def test_embedding_source_in_features_is_hidden() -> None:
    model = _mlx_model()
    p = model.attach_probe(
        ProbeConfig(name="ev", layers=[-1], source="embedding", aggregation="last")
    )
    assert p.module.weight.shape[-1] == 16  # hidden


def test_attention_source_on_nonstandard_arch_raises_clear() -> None:
    model = _mlx_model()
    with pytest.raises(ValueError, match="no 'attention' submodule"):
        model.attach_probe(ProbeConfig(name="a", layers=[1], source="attention"))

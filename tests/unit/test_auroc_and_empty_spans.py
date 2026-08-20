"""AUROC as a default metric, and what an EMPTY span list means.

Both come from the same failure mode: a separability run that looked healthy and
was quietly measuring the wrong thing.

* ``labels={"probe": []}`` used to mask the whole message, so in a span-annotated
  corpus every wholly-clean example contributed NOTHING and the negative class
  collapsed to whatever text happened to sit around the positives.
* The default metrics were threshold-based only (acc / adjacent / macro-F1), so a
  probe study had no threshold-free number without writing its own metric fn.
"""

from __future__ import annotations

import numpy as np
import pytest

from auto_chasm.data import build_dataset
from auto_chasm.metrics import auroc, classification_metrics


class _Tok:
    """Whitespace tokenizer with character offsets, no chat template."""

    chat_template = None

    def encode(self, text: str, **kw: object) -> list[int]:
        return [ord(w[0]) for w in text.split()]

    def __call__(self, text: str, **kw: object) -> dict[str, object]:
        offs, i = [], 0
        for w in text.split():
            start = text.index(w, i)
            offs.append((start, start + len(w)))
            i = start + len(w)
        return {"input_ids": [ord(w[0]) for w in text.split()], "offset_mapping": offs}


def _labels(sample: object) -> list[float]:
    lab = sample["labels"]  # type: ignore[index]
    return lab["p"] if isinstance(lab, dict) else lab


# --- empty span list = "all negative", not "unlabeled" ----------------------


def test_empty_span_list_labels_the_whole_message_negative() -> None:
    convo = [{"role": "assistant", "content": "a b c", "labels": {"p": []}}]
    out = build_dataset([convo], _Tok(), default_label=0)
    assert _labels(out[0]) == [0, 0, 0]


def test_absent_probe_still_masks() -> None:
    """Absent and empty must stay DISTINGUISHABLE."""
    convo = [{"role": "assistant", "content": "a b c", "labels": {}}]
    out = build_dataset([convo], _Tok(), default_label=0)
    assert _labels(out[0]) == [-100, -100, -100]


def test_empty_list_respects_masking_mode() -> None:
    """With default_label=None the fill IS the mask, so nothing is trained."""
    convo = [{"role": "assistant", "content": "a b c", "labels": {"p": []}}]
    out = build_dataset([convo], _Tok(), default_label=None)
    assert _labels(out[0]) == [-100, -100, -100]


def test_clean_and_labeled_examples_coexist() -> None:
    clean = [{"role": "assistant", "content": "a b", "labels": {"p": []}}]
    dirty = [{"role": "assistant", "content": "c d", "labels": {"p": [
        {"start": 0, "end": 1, "label": 1}]}}]
    out = build_dataset([clean, dirty], _Tok(), default_label=0)
    assert _labels(out[0]) == [0, 0]
    assert _labels(out[1])[0] == 1


# --- AUROC -------------------------------------------------------------------


def test_auroc_matches_the_rank_definition() -> None:
    scores = np.array([[0.1, 0.4, 0.35, 0.8]])
    targets = np.array([[0, 0, 1, 1]])
    mask = np.ones((1, 4), dtype=bool)
    assert auroc(scores, targets, mask) == pytest.approx(0.75)


def test_auroc_ignores_masked_and_sentinel_positions() -> None:
    scores = np.array([[0.1, 9.9, 0.35, 0.8]])
    targets = np.array([[0, 1, 1, 1]])
    mask = np.array([[True, False, True, True]])          # drop the 9.9
    without = auroc(scores, targets, mask)
    targets2 = np.array([[0, -100, 1, 1]])                 # drop it via -100 instead
    assert without == pytest.approx(auroc(scores, targets2, np.ones((1, 4), dtype=bool)))


def test_constant_scores_are_chance_not_perfect() -> None:
    """Ties must be rank-averaged; otherwise a dead head scores 0.0 or 1.0."""
    assert auroc(np.ones((1, 4)), np.array([[0, 1, 0, 1]]), np.ones((1, 4), dtype=bool)) == 0.5


def test_single_class_selection_is_undefined() -> None:
    assert np.isnan(auroc(np.array([[0.1, 0.2]]), np.array([[1, 1]]), np.ones((1, 2), dtype=bool)))


def test_binary_head_emits_auroc_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import auto_chasm.metrics as m

    monkeypatch.setattr(m, "run_probe", lambda tm, name, hidden: np.asarray(hidden))
    scores = np.array([[0.1, 0.4, 0.35, 0.8]])[..., None]
    out = classification_metrics(num_classes=2)(
        None, {"L0": scores}, np.array([[0, 0, 1, 1]]), np.ones((1, 4), dtype=bool)
    )
    assert out["L0_auroc"] == pytest.approx(0.75)


def test_auroc_key_is_omitted_when_undefined(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nan would poison the eval loop's per-key average; omission is correct."""
    import auto_chasm.metrics as m

    monkeypatch.setattr(m, "run_probe", lambda tm, name, hidden: np.asarray(hidden))
    out = classification_metrics(num_classes=2)(
        None, {"L0": np.array([[0.1, 0.2]])[..., None]},
        np.array([[1, 1]]), np.ones((1, 2), dtype=bool)
    )
    assert "L0_auroc" not in out
    assert "L0_acc" in out


def test_multiclass_head_gets_no_auroc(monkeypatch: pytest.MonkeyPatch) -> None:
    import auto_chasm.metrics as m

    monkeypatch.setattr(m, "run_probe", lambda tm, name, hidden: np.asarray(hidden))
    out = classification_metrics(num_classes=3)(
        None, {"L0": np.zeros((1, 4, 3))}, np.array([[0, 1, 2, 1]]), np.ones((1, 4), dtype=bool)
    )
    assert "L0_auroc" not in out


# --- class means: one pass, however many probes -----------------------------


def test_multi_probe_class_means_run_in_one_pass() -> None:
    """Looping probes outside the batch loop re-ran the corpus per probe.

    Every probe captures from the SAME forward, so a 24-layer mass-mean sweep was
    paying 24 passes for what one pass already produced.
    """
    import inspect

    from auto_chasm.class_means import compute_class_means

    src = inspect.getsource(compute_class_means)
    assert "_MultiProbeAccumulator" in src
    # the batch loop must be OUTSIDE any per-probe loop
    assert src.index("for raw_tokens") < src.index("for probe_name in probes")


# --- per-layer early stopping in LayerSweep ---------------------------------


def test_layer_sweep_exposes_early_stopping_params() -> None:
    import inspect

    from auto_chasm import LayerSweep

    p = inspect.signature(LayerSweep.__init__).parameters
    assert p["early_stopping_patience"].default == 0     # opt-in
    assert p["min_delta"].default == 0.0


def test_patience_is_tracked_per_layer() -> None:
    """Each layer stops on ITS OWN metric, not a shared counter."""
    from auto_chasm.sweep import _BestPerLayerCallback

    cb = _BestPerLayerCallback(None, None, ["L0", "L1"], "val_loss", False, 1, 10,
                               patience=2, min_delta=0.0)
    assert cb.stale == {"L0": 0, "L1": 0}
    assert cb.stopped == {}


def test_stop_requested_is_forwarded_to_the_running_loop() -> None:
    """A callback holds the FACADE; the loop that breaks is the inner trainer.

    Setting the flag on the facade used to be a silent no-op, so every layer could
    plateau and the run would still burn through num_iters.
    """
    import inspect

    from auto_chasm.trainers import _torch_loop
    from auto_chasm.trainers.base import JointTrainer
    from auto_chasm.trainers.trainer import Trainer

    assert isinstance(Trainer.stop_requested, property)
    assert "self.stop_requested" in inspect.getsource(JointTrainer.run)
    assert 'getattr(trainer, "stop_requested", False)' in inspect.getsource(_torch_loop)

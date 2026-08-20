"""Training data must be rendered the way the model is prompted at inference.

``build_dataset`` used to concatenate raw message text, so a three-turn chat
tokenized to ``'You are helpful.What is 2+2?It is four.'`` -- no role markers, no
turn delimiters -- while generation used the full template. A model fine-tuned on
one format and sampled from the other never sees its training distribution.

The hard part is that span labels are CHARACTER offsets into ``msg["content"]``,
so the scaffolding has to be spliced around separately-tokenized content rather
than rendered into one string.
"""

from __future__ import annotations

from typing import Any

import pytest

from auto_chasm._chat_template import (
    has_chat_template,
    message_wrappers,
    set_default_thinking,
    template_kwargs,
)


class _Tok:
    """Minimal chat tokenizer: ``<s>{role}:{content}</s>`` per turn."""

    chat_template = "dummy"

    def apply_chat_template(
        self, messages: list[dict[str, Any]], tokenize: bool = False,
        add_generation_prompt: bool = False, **kw: Any
    ) -> str:
        out = "".join(f"<s>{m['role']}:{m['content']}</s>" for m in messages)
        if add_generation_prompt:
            out += "<s>assistant:"
            if kw.get("enable_thinking") is False:
                out += "<think></think>"
        return out


def test_wrappers_reconstruct_the_template_exactly() -> None:
    tok = _Tok()
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    wrappers, _ = message_wrappers(tok, msgs)
    rebuilt = "".join(p + m["content"] + s for (p, s), m in zip(wrappers, msgs, strict=True))
    assert rebuilt == tok.apply_chat_template(msgs)


def test_closing_tag_belongs_to_the_turn_it_closes() -> None:
    """The assistant must keep its own end tag, or it never learns to stop."""
    tok = _Tok()
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    wrappers, _ = message_wrappers(tok, msgs)
    assert wrappers[-1][1].startswith("</s>")


def test_generation_prompt_is_returned_separately() -> None:
    tok = _Tok()
    msgs = [{"role": "user", "content": "hi"}]
    _, trailing = message_wrappers(tok, msgs, add_generation_prompt=True)
    assert trailing == "<s>assistant:"


def test_template_that_drops_content_is_rejected() -> None:
    """Silently mislabelling is worse than failing."""

    class Dropping(_Tok):
        def apply_chat_template(self, messages: Any, **kw: Any) -> str:
            return "<s>nothing</s>"

    with pytest.raises(ValueError, match="did not round-trip"):
        message_wrappers(Dropping(), [{"role": "user", "content": "hi"}])


def test_no_template_is_detected() -> None:
    class Bare:
        chat_template = None

    assert has_chat_template(Bare()) is False
    assert has_chat_template(_Tok()) is True


def test_empty_conversation_is_not_an_error() -> None:
    assert message_wrappers(_Tok(), []) == ([], "")


# --- the unified reasoning switch -------------------------------------------


def test_explicit_argument_beats_the_global() -> None:
    try:
        set_default_thinking(False)
        assert template_kwargs(True) == {"enable_thinking": True}
        assert template_kwargs(None) == {"enable_thinking": False}
    finally:
        set_default_thinking(None)


def test_global_none_leaves_the_template_alone() -> None:
    set_default_thinking(None)
    assert template_kwargs(None) == {}


# --- LayerSweep reserves the knobs it manages itself ------------------------


def test_layer_sweep_rejects_the_kwargs_it_owns() -> None:
    """``**trainer_kwargs`` collided as 'got multiple values for keyword argument'.

    That message names no argument and gives no reason, and the three it covers
    are exactly the ones the sweep replaces with per-layer selection.
    """
    import inspect

    from auto_chasm.sweep import LayerSweep

    src = inspect.getsource(LayerSweep.run)
    assert '"eval_steps", "save_steps", "early_stopping_patience"' in src
    assert "manages" in src and "eval_every=" in src


def test_sweep_csv_carries_custom_metrics() -> None:
    """A custom eval_metrics_fn's output must reach the file, not stop at ranking.

    Columns used to be hardcoded to acc/adj, so ``score_metric="val_auroc"``
    could rank layers on a metric that then appeared nowhere in the results.
    """
    import csv
    import tempfile
    from pathlib import Path

    from auto_chasm.sweep import SweepResult

    best = {
        3: {"iter": 50.0, "val_loss": 0.4, "val_acc": 0.8, "val_adj": 0.9,
            "test_loss": 0.5, "test_acc": 0.75, "test_adj": 0.88, "test_auroc": 0.81},
    }
    path = Path(tempfile.mkdtemp()) / "sweep.csv"
    SweepResult(best=best).to_csv(str(path))
    header, row = list(csv.reader(path.open()))[:2]
    assert "test_auroc" in header
    assert row[header.index("test_auroc")] == "0.81"
    # historical names preserved for existing readers
    assert "val_group_acc" in header and "test_group_acc" in header

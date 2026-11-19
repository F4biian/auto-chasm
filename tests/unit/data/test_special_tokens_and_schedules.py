"""Special-token placement in built datasets, LR schedule parity, and torch eval keys."""

from __future__ import annotations

from typing import Any

import pytest

from auto_chasm import Dataset, Model, ProbeConfig
from auto_chasm.config import TrainingConfig
from auto_chasm.data import build_dataset

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")


def _bos_tokenizer() -> Any:
    """Build a character-level tokenizer that prepends ``<s>`` like Llama tokenizers do."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {"<s>": 0, "</s>": 1, "<unk>": 2, "<|u|>": 3, "<|e|>": 4}
    for ch in "abcdefghijklmnopqrstuvwxyz .\n":
        vocab[ch] = len(vocab)
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    raw.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 0)]
    )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, bos_token="<s>", eos_token="</s>", unk_token="<unk>"
    )
    tok.add_special_tokens({"additional_special_tokens": ["<|u|>", "<|e|>"]})
    tok.chat_template = (
        "{{ bos_token }}{% for m in messages %}<|u|>{{ m['content'] }}<|e|>{% endfor %}"
    )
    return tok


CONVERSATION = [
    {"role": "user", "content": "ab", "labels": {"p": [{"start": 0, "end": 2, "label": 1}]}},
    {"role": "assistant", "content": "cd"},
]


def test_template_rendering_matches_apply_chat_template() -> None:
    """Message content must not get its own BOS inside a rendered template."""
    tok = _bos_tokenizer()
    sample = build_dataset([CONVERSATION], tok)[0]
    expected = tok.apply_chat_template(CONVERSATION, tokenize=True)
    if not isinstance(expected, list):  # transformers 5 returns a BatchEncoding
        expected = expected["input_ids"]
    assert sample["tokens"] == expected
    assert sample["labels"] == [-100, -100, 1, 1, -100, -100, -100, -100, -100]


def test_raw_conversation_gets_a_single_bos() -> None:
    """Without a template, BOS appears once at the start, not before every message."""
    tok = _bos_tokenizer()
    sample = build_dataset([CONVERSATION], tok, chat_template=False)[0]
    assert sample["tokens"] == [0, 5, 6, 7, 8]
    assert sample["labels"] == [-100, 1, 1, -100, -100]


def test_append_eos_moves_response_label_past_template_suffix() -> None:
    """The response label lands on the appended EOS even when a closing tag follows it."""
    tok = _bos_tokenizer()
    data = Dataset.from_texts(["ab"], [1], tok, probe_name="p", append_eos=True)
    sample = data._samples[0]
    assert sample["tokens"][-1] == tok.eos_token_id
    assert sample["labels"][-1] == 1
    assert sum(1 for v in sample["labels"] if v != -100) == 1


def _lrs(scheduler: Any, optimizer: Any, steps: int) -> list[float]:
    out = []
    for _ in range(steps):
        out.append(round(optimizer.param_groups[0]["lr"], 6))
        optimizer.step()
        scheduler.step()
    return out


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_full_warmup_reaches_peak_on_both_backends(schedule: str) -> None:
    """warmup_ratio=1.0 ramps to the peak LR at the last update on MLX and torch."""
    pytest.importorskip("mlx.core")
    from auto_chasm.trainers._metrics import build_lr_schedule
    from auto_chasm.trainers._torch_step import build_torch_optim_sched

    n = 5
    optimizer, scheduler = build_torch_optim_sched(
        torch.nn.Linear(2, 2), 1.0, 0.0, schedule, 1.0, n
    )
    torch_lrs = _lrs(scheduler, optimizer, n)
    mlx_schedule = build_lr_schedule(schedule, 1.0, n, n)
    mlx_lrs = [round(float(mlx_schedule(step)), 6) for step in range(n)]
    assert torch_lrs == mlx_lrs
    assert torch_lrs[-1] == pytest.approx(1.0)


class _TinyTorchLM(torch.nn.Module):
    """Two-layer torch LM with the attribute layout the model wrapper expects."""

    def __init__(self) -> None:
        super().__init__()

        class _Cfg:
            hidden_size = 16
            num_hidden_layers = 2

        self.config = _Cfg()
        self.embedding = torch.nn.Embedding(32, 16)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(2)])
        self.output_proj = torch.nn.Linear(16, 32)

    def forward(self, x: Any) -> Any:
        """Run the toy model."""
        h = self.embedding(x)
        for layer in self.layers:
            h = torch.nn.functional.gelu(layer(h))
        return self.output_proj(h)


def test_torch_eval_reports_perplexity_and_ntokens() -> None:
    """The torch evaluator returns the same keys as the MLX one."""
    from auto_chasm.trainers._metrics import evaluate_torch_model
    from auto_chasm.trainers.loss import JointLoss

    model = Model(_TinyTorchLM(), None, "torch")
    model.attach_probe(ProbeConfig(name="p", layers=[0], module_config={"out_features": 1}))
    data = [{"tokens": [1, 2, 3, 4, 5], "labels": [0, 0, 1, 0, 0]} for _ in range(4)]
    result = evaluate_torch_model(model, data, 2, 8, JointLoss(), None)
    assert result["ntokens"] > 0
    assert "perplexity" in result


def test_sft_trainer_applies_config_probe_weights(tmp_path: Any) -> None:
    """SFTTrainer passes TrainingConfig.probe_weights into its loss."""
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as mnn

    from auto_chasm.trainers.sft import SFTTrainer

    class _Cfg:
        hidden_size = 16
        num_hidden_layers = 2

    class _TinyMLXLM(mnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _Cfg()
            self.embedding = mnn.Embedding(32, 16)
            self.layers = [mnn.Linear(16, 16) for _ in range(2)]
            self.output_proj = mnn.Linear(16, 32)

        def __call__(self, x: Any, **_: Any) -> Any:
            h = self.embedding(x)
            for layer in self.layers:
                h = mnn.gelu(layer(h))
            return self.output_proj(h)

    assert mx is not None
    model = Model(_TinyMLXLM(), None, "mlx")
    model.attach_probe(ProbeConfig(name="p", layers=[0], module_config={"out_features": 1}))
    config = TrainingConfig(probe_weights={"p": 5.0}, output_dir=str(tmp_path))
    trainer = SFTTrainer(model, config=config, verbose=False)
    assert trainer._trainer.loss_fn._weight_for("p") == 5.0

"""Custom probe heads — from a declarative ModuleSpec to a raw torch nn.Module.

There are three levels of control over a probe's head (``ProbeConfig.module_type``):

1. a **built-in name** — ``"linear"`` or ``"mlp"`` (sized via ``module_config``);
2. a **ModuleSpec** — a declarative, backend-agnostic head (depth, widths,
   activation, dropout, layer-norm); the library builds the concrete module on
   whichever backend you run, and it checkpoints/reloads cleanly;
3. a **callable** ``(in_features, cfg) -> module`` — you return ANY framework
   module, so you can build absolutely anything the framework supports.

Level 3 is your escape hatch: it is NOT backend-agnostic (you build a concrete
``torch.nn.Module`` or ``mlx.nn.Module`` yourself), so the library never limits
you. This demo forces the **torch** backend to show a raw ``torch.nn.Module`` head;
the ModuleSpec head below is backend-agnostic and also runs on MLX.

    python demo/demo_custom_head.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from auto_chasm import Dataset, JointLoss, Model, ModuleSpec, ProbeConfig, Trainer

MODEL = "HuggingFaceTB/SmolLM2-135M"

# Force torch so level 3 can return a real torch.nn.Module.
model = Model.from_pretrained(MODEL, backend_name="torch")

# ── Level 2: a declarative ModuleSpec (backend-agnostic) ─────────────────────────
# A 2-hidden-layer MLP head with GELU + dropout + an input LayerNorm — no torch/mlx
# imports, no backend branching. Attach it just like a string head.
model.attach_probe(
    ProbeConfig(
        name="spec_head",
        layers=[-1],
        granularity="response",
        aggregation="mean",
        module_type=ModuleSpec.mlp(
            hidden_dims=[128, 64], activation="gelu", dropout=0.1, input_layer_norm=True
        ),
    )
)


# ── Level 3: a raw torch.nn.Module head (full control) ───────────────────────────
# A residual MLP block with LayerNorm — an architecture ModuleSpec does not express.
# It maps one hidden vector (in_features) to out_features; the probe applies its own
# layer aggregation and granularity pooling around it, so you only write the head.
class ResidualHead(nn.Module):
    """A LayerNorm + residual-MLP + linear-out probe head."""

    def __init__(self, in_features: int, out_features: int, hidden: int = 128) -> None:
        """Build the norm, the residual MLP, and the output projection."""
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, in_features)
        self.out = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a hidden vector to ``out_features`` logits."""
        x = x + self.fc2(torch.relu(self.fc1(self.norm(x))))  # residual block
        return self.out(x)


def build_residual_head(in_features: int, cfg: dict) -> nn.Module:
    """Builder callable: receives the resolved input width + the module_config."""
    return ResidualHead(in_features, cfg.get("out_features", 1))


model.attach_probe(
    ProbeConfig(
        name="custom_head",
        layers=[-1],
        granularity="response",
        aggregation="mean",
        module_type=build_residual_head,  # any callable returning a module
    )
)
model.prepare_for_joint_training()

print("spec_head  module:", type(model._probes["spec_head"].module).__name__)
print(
    "custom_head module:",
    type(model._probes["custom_head"].module).__name__,
    "(raw torch nn.Module)",
)

# ── Train the custom head and confirm it actually learns ─────────────────────────
train_texts = [
    "The sky is blue.",
    "Is it raining today?",
    "Cats like to sleep.",
    "Are you feeling well?",
    "It is warm outside.",
    "What time does it start?",
    "Birds build nests.",
    "Do you like coffee?",
    "Where did you go?",
    "Water boils at 100 degrees.",
    "Why are the leaves green?",
    "Which one is faster?",
]
train_labels = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1]
eval_texts = [
    "The dog is barking.",
    "Is the store open?",
    "Leaves fall in autumn.",
    "Have you eaten yet?",
]
eval_labels = [0, 1, 0, 1]

data = Dataset.from_texts(
    train_texts,
    train_labels,
    model.tokenizer,
    probe_name="custom_head",
    label_site="response",
    append_eos=True,
)


def p_question(text: str) -> float:
    """The custom head's probability that a text is a question."""
    ids = model.tokenizer.encode(text)
    if ids[-1] != model.tokenizer.eos_token_id:
        ids = ids + [model.tokenizer.eos_token_id]
    logit = float(model.forward([ids]).probes["custom_head"].logits.reshape(-1)[-1].item())
    return 1.0 / (1.0 + math.exp(-logit))


def accuracy() -> float:
    """Held-out accuracy of the custom head."""
    pairs = zip(eval_texts, eval_labels, strict=True)
    return sum((p_question(t) > 0.5) == bool(y) for t, y in pairs) / len(eval_labels)


print(f"\ncustom head held-out accuracy BEFORE: {accuracy():.0%}")
Trainer(
    model=model,
    loss_fn=JointLoss(weights={"lm_head": 0.0, "custom_head": 1.0}),
    num_iters=40,
    batch_size=4,
    learning_rate=5e-4,
    verbose=False,
).train(data)
print(f"custom head held-out accuracy AFTER:  {accuracy():.0%}")

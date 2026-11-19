"""Shared fixtures and test utilities for the test suite."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

# --- Crash-safety: route MLX to the CPU during tests -----------------------
# MLX defaults to the Metal GPU, whose driver (AGXMetalG16X) can abort the whole
# test process with SIGABRT on some machines, crashing the run. The CPU device is
# deterministic and lets MLX-vs-torch parity tests compare on equal footing. Opt
# back into the GPU with AUTO_CHASM_TEST_DEVICE=gpu.
if os.environ.get("AUTO_CHASM_TEST_DEVICE", "cpu").lower() != "gpu":
    mx.set_default_device(mx.cpu)
    # mlx_lm.generate's wired_limit() reads a Metal-only device_info key
    # ("max_recommended_working_set_size") when mx.metal.is_available() is True;
    # on the CPU device that key is absent -> KeyError. Since tests run on CPU,
    # report Metal as unavailable so mlx_lm takes its non-Metal path. Nothing in
    # auto_chasm reads mx.metal.* (grep-verified), so this only affects mlx_lm.
    if hasattr(mx, "metal") and hasattr(mx.metal, "is_available"):
        mx.metal.is_available = lambda: False  # type: ignore[assignment,method-assign]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-real-model`` opt-in flag."""
    parser.addoption(
        "--run-real-model",
        action="store_true",
        default=False,
        help="Run tests marked `real_model` (they load real pretrained weights).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``real_model`` tests unless ``--run-real-model`` is passed."""
    if config.getoption("--run-real-model"):
        return
    skip = pytest.mark.skip(reason="needs --run-real-model (loads real pretrained weights)")
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip)


class TinyMlp(nn.Module):
    """A tiny 2-layer MLP for testing."""

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 32, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self._hidden_dim = hidden_dim

    def __call__(self, x: mx.array, **kwargs: Any) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.output_proj(h)


class DummyTokenizer:
    """Minimal tokenizer for testing."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 32 for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 32) for i in ids if i > 0)


@pytest.fixture
def tiny_model() -> tuple[TinyMlp, DummyTokenizer]:
    """Create a tiny model and tokenizer for testing."""
    mx.random.seed(42)
    model = TinyMlp(hidden_dim=16, vocab_size=32, num_layers=4)
    tokenizer = DummyTokenizer()
    return model, tokenizer


@pytest.fixture
def model_wrapper(tiny_model: tuple[TinyMlp, DummyTokenizer]) -> object:
    """Create a Model wrapper around the tiny model."""
    from auto_chasm import Model

    base_model, tokenizer = tiny_model

    class Config:
        """Dummy configuration for testing."""

        hidden_size = 16
        num_hidden_layers = 4

    base_model.config = Config()
    return Model(base_model, tokenizer, backend_name="mlx")


@pytest.fixture
def sample_dataset() -> list[dict]:
    """Create a small synthetic dataset for training tests."""
    mx.random.seed(42)
    data = []
    for _ in range(32):
        tokens = [1, 2, 3, 4, 5]
        labels = [0, 0, 1, 0, 0]
        data.append({"tokens": tokens, "labels": labels})
    return data


# ---------------------------------------------------------------------------
# PyTorch fixtures (for cross-backend testing)
# ---------------------------------------------------------------------------


def _make_torch_tiny_mlp(hidden_dim: int = 16, vocab_size: int = 32, num_layers: int = 4):
    """Create a tiny PyTorch MLP matching the MLX TinyMlp architecture."""
    import torch
    import torch.nn as tnn

    class TorchTinyMlp(tnn.Module):
        """A tiny MLP for torch testing, matching MLX TinyMlp."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = tnn.Embedding(vocab_size, hidden_dim)
            self.layers = tnn.ModuleList(
                [tnn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
            )
            self.output_proj = tnn.Linear(hidden_dim, vocab_size)

        def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
            h = self.embedding(x)
            for layer in self.layers:
                h = torch.nn.functional.gelu(layer(h))
            return self.output_proj(h)

    return TorchTinyMlp()


@pytest.fixture
def torch_tiny_model():  # type: ignore[no-untyped-def]
    """Create a tiny PyTorch model and tokenizer for testing."""
    import torch

    torch.manual_seed(42)
    model = _make_torch_tiny_mlp(hidden_dim=16, vocab_size=32, num_layers=4)
    tokenizer = DummyTokenizer()
    return model, tokenizer


@pytest.fixture
def torch_model_wrapper(torch_tiny_model):  # type: ignore[no-untyped-def]
    """Create a Model wrapper around the tiny PyTorch model."""
    import torch

    from auto_chasm import Model

    torch.manual_seed(42)
    base_model, tokenizer = torch_tiny_model

    class Config:
        """Dummy configuration for testing."""

        hidden_size = 16
        num_hidden_layers = 4

    base_model.config = Config()
    return Model(base_model, tokenizer, backend_name="torch")

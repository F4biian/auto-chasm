# Tests

## Running them

```bash
uv sync --all-extras                 # both backends and the dev tools
uv run pytest                        # the fast suite (a few minutes)
uv run pytest --run-real-model       # also the tests that load real pretrained models
uv run pytest tests/unit/data        # one area only
```

## Layout

| Folder | What it covers |
|---|---|
| `unit/data/` | datasets, label placement, chat templates, special tokens, splits |
| `unit/probes/` | probe heads, layer and sub-module sources, aggregation and pooling |
| `unit/training/` | `Trainer`, `JointLoss`, LR schedules, mixed precision, RL trainers |
| `unit/generation/` | generation, sampling, stop conditions, KV cache, steering |
| `unit/checkpoints/` | saving and loading, LoRA/PEFT adapters |
| `unit/backends/` | the MLX and PyTorch backends and their parity |
| `unit/evaluation/` | metrics, probe scores, bootstrap, mass-mean probes, layer sweeps |
| `unit/api/` | the public API, configs, demos and cross-cutting regressions |
| `integration/` | end-to-end runs, some with real models (`real_model` marker) |

Shared fixtures (tiny models and tokenizers for both backends) live in
[`conftest.py`](conftest.py).

## Conventions

- **Oracle tests** (`*_oracle.py`) compare the library's output with an independent,
  hand-written reference computation, for example a loss computed in NumPy.
- **Regression tests** (`*_regressions.py`) each pin one behavior that once went wrong.
- Tests marked `real_model` load real checkpoints from the Hugging Face Hub and only run
  with `--run-real-model`.
- Every test file has a module docstring that says what it covers.

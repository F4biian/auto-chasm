"""Pins the curated top-level public API (Phase 4b slim).

`auto_chasm.__all__` is the advertised surface — the ~18 objects the common
workflows need. Everything else stays reachable from its submodule but is NOT
re-exported at the top level. These tests freeze that contract so the surface
can't silently grow (a stray top-level import) or a core export vanish, and so the
demoted names remain importable where the docs now point.
"""

from __future__ import annotations

import importlib

import auto_chasm

# The exact curated public surface. Changing this set is a deliberate API decision.
EXPECTED_PUBLIC = {
    "Model",
    "Dataset",
    "Task",
    "ProbeConfig",
    "Trainer",
    "SFTTrainer",
    "TrainingConfig",
    "JointLoss",
    "LayerSweep",
    "SweepResult",
    "ModuleSpec",
    "Probe",
    "ops",
    "classification_metrics",
    "regression_metrics",
    "GenerationConfig",
    "SteeringConfig",
    "LoraConfig",
    # Reasoning mode is process-wide on purpose: data prep and generation must
    # agree, and the template's own default differs between tokenizer wrappers.
    "set_default_thinking",
}

# Names deliberately DEMOTED from the top level → where they now live.
DEMOTED_TO_SUBMODULE = {
    "build_dataset": "data",
    "span_labels_to_tokens": "data",
    "balanced_class_weights": "data",
    "label_counts": "data",
    "IGNORE_INDEX": "data",
    "GenerationStep": "outputs",
    "LossOutputs": "outputs",
    "ModelOutputs": "outputs",
    "ProbeLossInfo": "outputs",
    "ProbeOutput": "outputs",
    "JointOutputs": "outputs",
    "to_numpy": "metrics",
    "run_probe": "metrics",
    "accuracy": "metrics",
    "ordinal_accuracy": "metrics",
    "macro_f1": "metrics",
    "mse": "metrics",
    "mae": "metrics",
    "discretize": "metrics",
    "build_module": "modules",
    "History": "history",
    "HistoryEntry": "history",
    "RLConfig": "config",
    "SteeringHook": "steering",
    "TrainerCallback": "trainers",
    "RLTrainer": "trainers",
    "JointTrainer": "trainers",
}


def test_all_is_exactly_the_curated_surface() -> None:
    """``__all__`` equals the frozen expected set — no drift, no dupes."""
    assert set(auto_chasm.__all__) == EXPECTED_PUBLIC
    assert len(auto_chasm.__all__) == len(EXPECTED_PUBLIC)  # no duplicate entries


def test_every_public_name_is_importable() -> None:
    """Each ``__all__`` name is an actual top-level attribute."""
    for name in auto_chasm.__all__:
        assert hasattr(auto_chasm, name), f"{name} in __all__ but not on the package"


def test_star_import_exposes_only_the_public_surface() -> None:
    """``from auto_chasm import *`` binds exactly the curated names."""
    ns: dict[str, object] = {}
    exec("from auto_chasm import *", ns)  # noqa: S102 - controlled test input
    starred = {k for k in ns if not k.startswith("__")}
    assert starred == EXPECTED_PUBLIC


def test_demoted_names_are_not_top_level() -> None:
    """Demoted names are gone from the top level (a real removal, not a shim)."""
    leaked = [n for n in DEMOTED_TO_SUBMODULE if hasattr(auto_chasm, n)]
    assert not leaked, f"demoted names still exposed at top level: {leaked}"


def test_demoted_names_remain_importable_from_their_submodule() -> None:
    """Every demoted name still lives in the submodule the docs now import it from."""
    for name, submodule in DEMOTED_TO_SUBMODULE.items():
        mod = importlib.import_module(f"auto_chasm.{submodule}")
        assert hasattr(mod, name), f"{name} not importable from auto_chasm.{submodule}"

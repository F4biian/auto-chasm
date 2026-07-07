"""auto-chasm: Self-supervised internal state alignment for language models.

The top-level namespace is the **curated public API** — the objects the common
workflows need (build a model + probes, prepare data, infer the task, train, sweep
layers, write a custom backend-agnostic loss).  Everything else remains available
from its submodule (e.g. ``from auto_chasm.data import build_dataset``,
``from auto_chasm.metrics import accuracy``, ``from auto_chasm.outputs import
JointOutputs``, ``from auto_chasm.trainers import RLTrainer``) — it is simply not
re-exported here.

Public API::

    from auto_chasm import Model, Dataset, Task, ProbeConfig, Trainer, JointLoss
"""

from auto_chasm import ops
from auto_chasm.config import (
    GenerationConfig,
    LoraConfig,
    ProbeConfig,
    SteeringConfig,
    TrainingConfig,
)
from auto_chasm.dataset import Dataset
from auto_chasm.metrics import classification_metrics, regression_metrics
from auto_chasm.model import Model
from auto_chasm.modules import ModuleSpec
from auto_chasm.probe import Probe
from auto_chasm.sweep import LayerSweep, SweepResult
from auto_chasm.task import Task
from auto_chasm.trainers import JointLoss, SFTTrainer, Trainer

__all__ = [
    # Model, data, task
    "Model",
    "Dataset",
    "Task",
    "ProbeConfig",
    # Training
    "Trainer",
    "SFTTrainer",
    "TrainingConfig",
    "JointLoss",
    # Layer sweeps
    "LayerSweep",
    "SweepResult",
    # Heads + backend-agnostic ops (for custom losses)
    "ModuleSpec",
    "Probe",
    "ops",
    # Metrics
    "classification_metrics",
    "regression_metrics",
    # Feature configs
    "GenerationConfig",
    "SteeringConfig",
    "LoraConfig",
]

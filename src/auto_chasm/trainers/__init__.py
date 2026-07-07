"""Trainer sub-package."""

from auto_chasm.trainers.base import JointTrainer
from auto_chasm.trainers.loss import JointLoss
from auto_chasm.trainers.rl import RLTrainer
from auto_chasm.trainers.sft import SFTTrainer
from auto_chasm.trainers.trainable import default_binary_metrics
from auto_chasm.trainers.trainable import make_joint_loss as make_mlx_joint_loss
from auto_chasm.trainers.trainer import Trainer
from auto_chasm.trainers.wrappers import TrainerCallback

__all__ = [
    "JointTrainer",
    "SFTTrainer",
    "RLTrainer",
    "Trainer",
    "TrainerCallback",
    "JointLoss",
    "make_mlx_joint_loss",
    "default_binary_metrics",
]

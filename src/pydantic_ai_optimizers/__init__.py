"""PydanticAI Optimizers - A toolkit for optimizing PydanticAI agent prompts through iterative improvement."""

from .config import OptimizerConfig, get_optimizer_config
from .dataset import Dataset, ReportCase
from .optimizer import Candidate, CaseEval, FailedMutation, Optimizer

__version__ = "0.0.1"
__all__ = [
    "Optimizer",
    "Candidate",
    "CaseEval",
    "FailedMutation",
    "get_optimizer_config",
    "OptimizerConfig",
    "Dataset",
    "ReportCase",
]

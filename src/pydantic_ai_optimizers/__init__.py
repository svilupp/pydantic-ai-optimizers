"""PydanticAI Optimizers - A toolkit for optimizing PydanticAI agent prompts through iterative improvement."""

from pydantic_evals import Dataset  # type: ignore
from pydantic_evals.reporting import ReportCase  # type: ignore

from .agents.reflection_agent import make_reflection_agent
from .config import OptimizerConfig, get_optimizer_config
from .optimizer import Candidate, CaseEval, FailedMutation, Optimizer

__version__ = "0.0.2"
__all__ = [
    "Optimizer",
    "Candidate",
    "CaseEval",
    "FailedMutation",
    "get_optimizer_config",
    "OptimizerConfig",
    "Dataset",
    "ReportCase",
    "make_reflection_agent",
]

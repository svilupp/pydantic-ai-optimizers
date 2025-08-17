"""
Configuration settings for the prompt optimizer.

This module provides centralized configuration management with support for
environment variables and sensible defaults.
"""

import os
from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    """Configuration settings for the prompt optimizer."""

    # Model configurations
    reflection_model: str = "openai:gpt-5"
    agent_model: str = "openai:gpt-5-nano"

    # Optimizer parameters
    minibatch_size: int = 4
    max_pool_size: int = 12
    full_validation_budget: int = 12
    seed: int = 42
    keep_failed_mutations: bool = True

    # Paths
    pool_dir: str = "prompt_pool"

    @classmethod
    def from_env(cls) -> "OptimizerConfig":
        """Create configuration from environment variables."""
        return cls(
            reflection_model=os.getenv("REFLECTION_MODEL", "openai:gpt-5"),
            agent_model=os.getenv("AGENT_MODEL", "openai:gpt-5-nano"),
            minibatch_size=int(os.getenv("MINIBATCH_SIZE", "4")),
            max_pool_size=int(os.getenv("MAX_POOL_SIZE", "12")),
            full_validation_budget=int(os.getenv("VALIDATION_BUDGET", "12")),
            seed=int(os.getenv("OPTIMIZER_SEED", "42")),
            keep_failed_mutations=os.getenv("KEEP_FAILED_MUTATIONS", "true").lower() == "true",
            pool_dir=os.getenv("POOL_DIR", "prompt_pool"),
        )


def get_model_config() -> dict[str, str]:
    """Get model configuration as a dictionary."""
    config = OptimizerConfig.from_env()
    return {
        "reflection_model": config.reflection_model,
        "agent_model": config.agent_model,
    }


def get_optimizer_config() -> OptimizerConfig:
    """Get full optimizer configuration."""
    return OptimizerConfig.from_env()


# Environment variable documentation
ENV_DOCS = {
    "REFLECTION_MODEL": "Model used for reflection agent (default: openai:gpt-4o)",
    "AGENT_MODEL": "Model used for main agent (default: openai:gpt-4o-mini)",
    "MINIBATCH_SIZE": "Size of evaluation minibatches (default: 4)",
    "MAX_POOL_SIZE": "Maximum number of candidates in pool (default: 24)",
    "VALIDATION_BUDGET": "Number of full validations to run (default: 12)",
    "OPTIMIZER_SEED": "Random seed for reproducibility (default: 42)",
    "KEEP_FAILED_MUTATIONS": "Whether to keep failed mutation files (default: false)",
    "POOL_DIR": "Directory for storing prompt candidates (default: prompt_pool)",
}


def print_env_help() -> None:
    """Print help for environment variables."""
    print("Environment Variables:")
    print("=" * 50)
    for var, desc in ENV_DOCS.items():
        current_val = os.getenv(var, "Not set")
        print(f"{var}:")
        print(f"  {desc}")
        print(f"  Current value: {current_val}")
        print()

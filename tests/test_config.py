"""Tests for configuration module."""

import os
from unittest.mock import patch

from pydantic_ai_optimizers.config import (
    OptimizerConfig,
    get_model_config,
    get_optimizer_config,
    print_env_help,
)


class TestOptimizerConfig:
    """Tests for OptimizerConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = OptimizerConfig()

        assert config.reflection_model == "openai:gpt-5"
        assert config.agent_model == "openai:gpt-5-nano"
        assert config.minibatch_size == 4
        assert config.max_pool_size == 12
        assert config.full_validation_budget == 12
        assert config.seed == 42
        assert config.keep_failed_mutations is True
        assert config.pool_dir == "prompt_pool"

    def test_from_env_with_defaults(self):
        """Test from_env with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            config = OptimizerConfig.from_env()

            assert config.reflection_model == "openai:gpt-5"
            assert config.agent_model == "openai:gpt-5-nano"
            assert config.minibatch_size == 4
            assert config.max_pool_size == 12
            assert config.full_validation_budget == 12
            assert config.seed == 42
            assert config.keep_failed_mutations is True
            assert config.pool_dir == "prompt_pool"

    def test_from_env_with_all_variables(self):
        """Test from_env with all environment variables set."""
        env_vars = {
            "REFLECTION_MODEL": "openai:gpt-4",
            "AGENT_MODEL": "openai:gpt-4-mini",
            "MINIBATCH_SIZE": "8",
            "MAX_POOL_SIZE": "20",
            "VALIDATION_BUDGET": "50",
            "OPTIMIZER_SEED": "123",
            "KEEP_FAILED_MUTATIONS": "false",
            "POOL_DIR": "custom_pool",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = OptimizerConfig.from_env()

            assert config.reflection_model == "openai:gpt-4"
            assert config.agent_model == "openai:gpt-4-mini"
            assert config.minibatch_size == 8
            assert config.max_pool_size == 20
            assert config.full_validation_budget == 50
            assert config.seed == 123
            assert config.keep_failed_mutations is False
            assert config.pool_dir == "custom_pool"

    def test_from_env_keep_failed_mutations_variations(self):
        """Test different values for KEEP_FAILED_MUTATIONS."""
        # Test true variations (only "true" is treated as true, per the logic)
        with patch.dict(os.environ, {"KEEP_FAILED_MUTATIONS": "true"}, clear=True):
            config = OptimizerConfig.from_env()
            assert config.keep_failed_mutations is True

        # Test false variations
        for false_val in ["false", "False", "FALSE", "no", "0", "", "yes", "1"]:
            with patch.dict(os.environ, {"KEEP_FAILED_MUTATIONS": false_val}, clear=True):
                config = OptimizerConfig.from_env()
                assert config.keep_failed_mutations is False

    def test_from_env_integer_conversion(self):
        """Test that integer environment variables are converted correctly."""
        env_vars = {
            "MINIBATCH_SIZE": "10",
            "MAX_POOL_SIZE": "25",
            "VALIDATION_BUDGET": "100",
            "OPTIMIZER_SEED": "999",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = OptimizerConfig.from_env()

            assert isinstance(config.minibatch_size, int)
            assert isinstance(config.max_pool_size, int)
            assert isinstance(config.full_validation_budget, int)
            assert isinstance(config.seed, int)
            assert config.minibatch_size == 10
            assert config.max_pool_size == 25
            assert config.full_validation_budget == 100
            assert config.seed == 999

    def test_from_env_partial_override(self):
        """Test from_env with only some environment variables set."""
        env_vars = {
            "REFLECTION_MODEL": "custom:model",
            "MAX_POOL_SIZE": "16",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = OptimizerConfig.from_env()

            # Overridden values
            assert config.reflection_model == "custom:model"
            assert config.max_pool_size == 16

            # Default values
            assert config.agent_model == "openai:gpt-5-nano"
            assert config.minibatch_size == 4
            assert config.full_validation_budget == 12
            assert config.seed == 42
            assert config.keep_failed_mutations is True
            assert config.pool_dir == "prompt_pool"


class TestGetModelConfig:
    """Tests for get_model_config function."""

    def test_get_model_config_defaults(self):
        """Test get_model_config with default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_model_config()

            assert isinstance(config, dict)
            assert config["reflection_model"] == "openai:gpt-5"
            assert config["agent_model"] == "openai:gpt-5-nano"

    def test_get_model_config_from_env(self):
        """Test get_model_config with environment variables."""
        env_vars = {
            "REFLECTION_MODEL": "custom:reflection",
            "AGENT_MODEL": "custom:agent",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = get_model_config()

            assert config["reflection_model"] == "custom:reflection"
            assert config["agent_model"] == "custom:agent"


class TestGetOptimizerConfig:
    """Tests for get_optimizer_config function."""

    def test_get_optimizer_config_returns_optimizer_config(self):
        """Test that get_optimizer_config returns OptimizerConfig instance."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_optimizer_config()

            assert isinstance(config, OptimizerConfig)
            assert config.reflection_model == "openai:gpt-5"

    def test_get_optimizer_config_uses_env(self):
        """Test that get_optimizer_config respects environment variables."""
        env_vars = {"REFLECTION_MODEL": "test:model"}

        with patch.dict(os.environ, env_vars, clear=True):
            config = get_optimizer_config()

            assert config.reflection_model == "test:model"


class TestPrintEnvHelp:
    """Tests for print_env_help function."""

    def test_print_env_help_with_no_env_vars(self, capsys):
        """Test print_env_help output when no environment variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            print_env_help()

            captured = capsys.readouterr()
            assert "Environment Variables:" in captured.out
            assert "=" * 50 in captured.out
            assert "REFLECTION_MODEL:" in captured.out
            assert "AGENT_MODEL:" in captured.out
            assert "Not set" in captured.out

    def test_print_env_help_with_env_vars(self, capsys):
        """Test print_env_help output when environment variables are set."""
        env_vars = {
            "REFLECTION_MODEL": "openai:gpt-4",
            "AGENT_MODEL": "openai:gpt-4-mini",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            print_env_help()

            captured = capsys.readouterr()
            assert "openai:gpt-4" in captured.out
            assert "openai:gpt-4-mini" in captured.out
            assert "Not set" in captured.out  # Other vars should still show "Not set"

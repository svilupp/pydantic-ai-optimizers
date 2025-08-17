# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python library called `pydantic-ai-optimizers` that systematically improves PydanticAI agent prompts through iterative optimization. The core methodology uses mini-batch testing, individual case tracking, and memory for failed attempts to evolve prompts effectively.

## Development Commands

### Setup and Installation
```bash
# Install package in development mode
make install-dev
# or directly with uv
uv pip install -e ".[dev]"

# Set up pre-commit hooks
make dev-setup
```

### Testing and Quality
```bash
# Run tests with coverage
make test

# Format code
make format

# Lint code  
make lint

# Type check
make type-check

# Run all checks (lint, type-check, test)
make check

# Run pre-commit hooks
make pre-commit
```

### Running Examples
```bash
# Run the chef optimization example
cd examples/chef
uv run python optimize.py

# Or from root directory
uv run python examples/chef/optimize.py
```

## Architecture

### Core Components

**Main Library** (`src/pydantic_ai_optimizers/`):
- `optimizer.py`: Core `Optimizer` class implementing the optimization algorithm
- `dataset.py`: `Dataset` class for managing evaluation cases and scoring
- `data_models.py`: Pydantic models for configuration and data structures
- `config.py`: Configuration management with environment variable support
- `cli.py`: Command-line interface for the optimizer

**Key Classes**:
- `Optimizer`: Main optimization engine with mini-batch gating and weighted sampling
- `Candidate`: Represents a prompt variant with metadata
- `Dataset`: Manages test cases and evaluation logic
- `FailedMutation`: Tracks failed optimization attempts for learning

### Example Structure

The `examples/chef/` directory demonstrates the complete pattern:
- `agent.py`: Domain-specific agent implementation with tools and reflection logic
- `optimize.py`: Evaluation framework and optimization loop
- `dataset.py`: Test case generation and evaluation scoring
- `data/`: Domain data (recipes, allergens) 
- `evals/`: Test cases in YAML format with JSON schema
- `prompts/`: Seed prompts and reflection instructions

### Optimization Algorithm

1. **Mini-batch Gating**: New candidates tested on small subset before full evaluation
2. **Weighted Sampling**: Parents selected based on individual case win rates
3. **Memory System**: Failed attempts provided to reflection agent when stuck
4. **Individual Case Tracking**: Performance tracked per test case for Pareto-efficient sampling

## Configuration

Set environment variables or use `.env` file:
```bash
OPENAI_API_KEY="your-key"
REFLECTION_MODEL="openai:gpt-5"
AGENT_MODEL="openai:gpt-5-nano" 
VALIDATION_BUDGET=20
MAX_POOL_SIZE=16
```

## Key Dependencies

- `pydantic-ai`: Core agent framework
- `textprompts`: File-based prompt management with placeholders
- `pydantic-ai-helpers`: Utilities for PydanticAI development
- `logfire`: Observability and tracing
- `loguru`: Structured logging

## Testing

Tests use pytest with coverage reporting. HTML coverage reports generated in `htmlcov/`.

## File Organization

- `src/pydantic_ai_optimizers/`: Main library code
- `examples/`: Complete domain examples showing usage patterns
- `tests/`: Unit tests
- `Makefile`: Development commands
- `pyproject.toml`: Project configuration, dependencies, and tool settings
# PydanticAI Optimizers

> ⚠️ **Super Opinionated**: This library is specifically built on top of PydanticAI + Pydantic Evals. If you don't use both together, this is useless to you.

A Python library for systematically improving PydanticAI agent prompts through iterative optimization. **Heavily inspired by the [GEPA paper](https://arxiv.org/abs/2507.19457)** with practical extensions for prompt optimization when switching model classes or providers.

## Acknowledgments

This work builds upon the excellent research in **"GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"** by Agrawal et al. We're grateful for their foundational work on reflective prompt evolution and have adapted (some of) their methodology with several practical tweaks for the PydanticAI ecosystem.

**Why this exists**: Every time you switch model classes (GPT-4.1 → GPT-5 → Claude Sonnet 4) or providers, your prompting needs change. Instead of manually tweaking prompts each time, this automates the optimization process for your existing PydanticAI agents with minimal effort.

## What It Does

This library optimizes prompts by:

1. **Mini-batch Testing**: Each candidate prompt is tested against a small subset of cases to see if it beats its parent before full evaluation
2. **Individual Case Tracking**: Performance on each test case is tracked, enabling weighted sampling that favors prompts that win on more individual cases  
3. **Memory for Failed Attempts**: When optimization gets stuck (children keep failing mini-batch tests), the system provides previous failed attempts to the reflection agent with the message: "You've tried these approaches and they didn't work - think outside the box!"

The core insight is that you don't lose learning between iterations, and the weighted sampling based on individual case win rates helps explore more diverse and effective prompt variations.

## Quick Start

### Installation

```bash
uv pip install -e .
```

Or for running the examples:
```bash
uv run python examples/chef/optimize.py
```

### Run the Chef Example

```bash
cd examples/chef
uv run python optimize.py
```

This will optimize a chef assistant prompt that helps users find recipes while avoiding allergens. You'll see the optimization process with real-time feedback and the final best prompt.

### Basic Usage in Your Project

```python
from pydantic_ai_optimizers import Optimizer
from your_domain import make_run_case, make_reflection_agent, build_dataset

# Set up your domain-specific components
dataset = build_dataset("your_cases.json")
run_case = make_run_case()  # Function that runs your agent with a prompt
reflection_agent = make_reflection_agent()  # Agent that improves prompts

# Optimize
optimizer = Optimizer(
    dataset=dataset,
    run_case=run_case,
    reflection_agent=reflection_agent,
)

best = await optimizer.optimize(
    seed_prompt_file="seed.txt",
    full_validation_budget=20
)
```

## How It Works

### 1. Start with a Seed Prompt
The optimizer begins with your initial prompt and evaluates it on all test cases.

### 2. Mini-batch Gating (Key Innovation #1)
- Select a parent prompt using weighted sampling (prompts that win more individual cases are more likely to be selected)
- Generate a new candidate through reflection on failed cases
- Test the candidate on a small mini-batch of cases
- Only if it beats the parent on the mini-batch does it get added to the candidate pool

### 3. Individual Case Performance Tracking (Key Innovation #2)  
- Track which prompt wins each individual test case
- Use this for Pareto-efficient weighted sampling of parents
- This ensures diverse exploration and prevents getting stuck in local optima

### 4. Memory for Failed Attempts (Our Addition)
- When candidates keep failing mini-batch tests, record the failed attempts
- Provide these to the reflection agent as context: "Here's what you've tried that didn't work"
- This increases pressure over time to try more creative approaches when stuck

## Creating Your Own Optimization

### 1. Set Up Your Domain

Copy the `examples/chef/` structure:

```
your_domain/
├── agent.py             # Your complete agent (tools, setup, everything)
├── optimize.py          # Your evaluation logic + optimization loop
├── data/                # Your domain data
└── prompts/             # Seed prompt and reflection instructions
```

### 2. Implement Required Functions

**Agent** (`agent.py`):
```python
def make_run_case():
    async def run_case(prompt_file: str, user_input: str):
        # Load prompt, run your agent, return results
        pass
    return run_case

def make_reflection_agent():
    # Return agent that improves prompts based on feedback
    pass
```

**Optimization** (`optimize.py`):
```python
def build_dataset(cases_file):
    # Load test cases and evaluators
    # Return dataset that can evaluate your agent's outputs
    pass

def main():
    # Set up dataset, run_case, reflection_agent
    # Create optimizer and run optimization loop
    pass
```

### 3. Run Optimization

```bash
python optimize.py
```

## Key Integrations

This library is designed to work seamlessly with:

### [textprompts](https://github.com/svilupp/textprompts)
Makes it easy to use standard text files with placeholders for prompt evolution. Perfect for diffing prompts and version control:

```python
# In your prompt file:
"You are a {role}. Your task is to {task}..."

# textprompts handles loading and placeholder substitution
prompt = textprompts.load_prompt("my_prompt.txt", role="chef", task="find recipes")
```

### [pydantic-ai-helpers](https://github.com/svilupp/pydantic-ai-helpers)  
Provides utilities that make PydanticAI much more convenient:
- Quick tool parsing and setup
- Simple evaluation comparisons between outputs and expected results
- Streamlined agent configuration

These integrations save significant development time when building optimization pipelines.

## Configuration

Set up through environment variables or configuration files:

```bash
export OPENAI_API_KEY="your-key"
export REFLECTION_MODEL="openai:gpt-4o"  
export AGENT_MODEL="openai:gpt-4o-mini"
export VALIDATION_BUDGET=20
export MAX_POOL_SIZE=16
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests  
make test

# Format and lint
make format && make lint

# Type check
make type-check
```

## Why This Approach Works

The combination of mini-batch gating and individual case tracking prevents two common optimization problems:

1. **Expensive Evaluation**: Mini-batches mean you only do full evaluation on promising candidates
2. **Premature Convergence**: Weighted sampling based on individual case wins maintains diversity

The memory system addresses a key weakness in memoryless optimization: when you get stuck, the system learns from its failures and tries more creative approaches.

## License

MIT License
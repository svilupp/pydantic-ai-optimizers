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
uv sync
```

Or run an example directly from the project root:
```bash
uv run examples/chef/optimize.py
uv run examples/customer_support/optimize.py
```

### Run the Chef Example

```bash
uv run examples/chef/optimize.py
```

### Run the Customer Support Example

```bash
uv run examples/customer_support/optimize.py
```

## Repository Structure

```
.
├── src/pydantic_ai_optimizers/
│   ├── agents/
│   │   └── reflection_agent.py
│   ├── optimizer.py
│   ├── config.py
│   └── cli.py
├── examples/
│   ├── chef/
│   └── customer_support/
├── tests/
└── docs/
```

This will optimize a chef assistant prompt that helps users find recipes while avoiding allergens. You'll see the optimization process with real-time feedback and the final best prompt.

### Basic Usage in Your Project

```python
from pydantic_ai_optimizers import Optimizer, make_reflection_agent
from your_domain import create_your_agent, build_dataset, YourInputType, YourOutputType

# CRITICAL: Define your run_case function with the correct signature
async def run_case(prompt_file: str, user_input: YourInputType) -> YourOutputType:
    """
    Run your agent with a specific prompt file and user input.
    
    Args:
        prompt_file: ABSOLUTE path to the prompt file (e.g., "/path/to/prompts/candidate_001.txt")
        user_input: The input from your dataset cases
    
    Returns:
        The agent's output that will be evaluated
    """
    # Load the prompt and create agent
    agent = create_your_agent(prompt_file=prompt_file, model="your-model")
    result = await agent.run(user_input.message)  # Or however you pass inputs
    return result.output

# Set up your dataset
dataset = build_dataset("your_cases.json")

# Optional: Customize the reflection agent
reflection_agent = make_reflection_agent(
    model="openai:gpt-5-mini",  # Use a different model
    special_instructions="Focus on conciseness and clarity"  # Add custom instructions
)
# Or use the default: reflection_agent = None (will use make_reflection_agent() internally)

# Create optimizer
optimizer = Optimizer(
    dataset=dataset,
    run_case=run_case,  # Your async function with the signature above
    reflection_agent=reflection_agent,  # Optional, uses default if None
)

# Run optimization
best = await optimizer.optimize(
    seed_prompt_file=Path("prompts/seed.txt"),
    full_validation_budget=20
)
print(f"Best prompt: {best.prompt_path}")
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
# CRITICAL: Your run_case function must have this exact signature
async def run_case(prompt_file: str, user_input: YourInputType) -> YourOutputType:
    """
    Run your agent with a specific prompt file and user input.
    
    Args:
        prompt_file: ABSOLUTE path to the prompt file (optimizer passes full paths)
        user_input: Input from your dataset cases (your domain-specific type)
    
    Returns:
        Agent output that matches your evaluators' expectations
    """
    # Example implementation:
    agent = create_your_agent(prompt_file=prompt_file, model="gpt-4")
    result = await agent.run(user_input.message)
    return result.output

# Optional: Customize the reflection agent
# If you don't provide one, the optimizer uses make_reflection_agent() internally
def create_custom_reflection_agent():
    from pydantic_ai_optimizers import make_reflection_agent
    
    return make_reflection_agent(
        model="gpt-4o",  # Your preferred model for reflection
        special_instructions="""
        Focus on:
        - Brevity and clarity
        - Domain-specific accuracy
        - Better error handling
        """  # Custom instructions for prompt improvement
    )
```

**Optimization** (`optimize.py`):
```python
from pydantic_ai_optimizers import Optimizer, make_reflection_agent
from pathlib import Path

def build_dataset(cases_file):
    # Load test cases and evaluators using pydantic-evals
    # Return dataset that can evaluate your agent's outputs
    pass

async def main():
    # Set up dataset
    dataset = build_dataset("cases.yaml")
    
    # Your run_case function (defined above)
    # No need to wrap it - pass it directly
    
    # Optional: Use custom reflection agent
    reflection_agent = make_reflection_agent(
        model="gpt-4o",
        special_instructions="Focus on accuracy and brevity"
    )
    # Or use default: reflection_agent = None
    
    # Create optimizer
    optimizer = Optimizer(
        dataset=dataset,
        run_case=run_case,  # Your async function
        reflection_agent=reflection_agent,  # Optional
        pool_dir=Path("prompt_pool"),
        minibatch_size=4,
        max_pool_size=16,
    )
    
    # Run optimization
    best = await optimizer.optimize(
        seed_prompt_file=Path("prompts/seed.txt"),
        full_validation_budget=20
    )
    
    print(f"Best prompt saved to: {best.prompt_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
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

## Reflection Agent Options

The optimizer uses a reflection agent to generate improved prompts based on evaluation feedback. You have several options:

### Use Default Reflection Agent
```python
# Pass None or omit the parameter - uses make_reflection_agent() with defaults
optimizer = Optimizer(
    dataset=dataset,
    run_case=run_case,
    # reflection_agent=None,  # Uses default
)
```

### Customize the Model
```python
from pydantic_ai_optimizers import make_reflection_agent

# Use a different model for reflection
reflection_agent = make_reflection_agent(model="openai:gpt-5-mini")

optimizer = Optimizer(
    dataset=dataset,
    run_case=run_case,
    reflection_agent=reflection_agent,
)
```

### Add Special Instructions (e.g., GPT-5 prompting tips)
```python
import textprompts
from pathlib import Path
from pydantic_ai_optimizers import make_reflection_agent

# Load GPT-5 prompting tips from a file and pass to the reflection agent
tips = str(textprompts.load_prompt(
    Path("examples/customer_support/prompts/gpt5_tips.txt")
))

reflection_agent = make_reflection_agent(
    model="openai:gpt-5-mini",
    special_instructions=tips,
)

optimizer = Optimizer(
    dataset=dataset,
    run_case=run_case,
    reflection_agent=reflection_agent,
)
```

### Bring Your Own Reflection Agent
```python
from pydantic_ai import Agent

# Create completely custom reflection agent
reflection_agent = Agent(
    model="your-model",
    instructions="Your custom reflection instructions..."
)

optimizer = Optimizer(
    dataset=dataset,
    run_case=run_case,
    reflection_agent=reflection_agent,
)
```

## Configuration

Set up through environment variables or configuration files:

```bash
export OPENAI_API_KEY="your-key"
export REFLECTION_MODEL="openai:gpt-5"  
export AGENT_MODEL="openai:gpt-5-nano"
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
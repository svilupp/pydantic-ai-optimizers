# pydantic-ai-optimizers

> Super opinionated prompt optimization for PydanticAI agents

⚠️ **This library only works if you use PydanticAI + Pydantic Evals together.** If you don't, this is useless to you.

## What It Does

Automatically optimizes prompts when you switch models or providers. Instead of manually tweaking prompts each time, this does it for you.

## Quick Start

```bash
pip install pydantic-ai-optimizers
```

```python
from pydantic_ai_optimizers import Optimizer

optimizer = Optimizer(
    dataset=your_dataset,
    run_case=your_run_function,
    reflection_agent=your_reflection_agent,
)

best = await optimizer.optimize(
    seed_prompt_file="seed.txt",
    full_validation_budget=20
)
```

## Architecture

- **Mini-batch gating**: Test candidates on small subset first
- **Individual case tracking**: Track performance per test case
- **Memory for failures**: Learn from failed attempts

## Examples

See `examples/chef/` for a complete working example.

## Configuration

```bash
export OPENAI_API_KEY="your-key"
export REFLECTION_MODEL="openai:gpt-4o"
export AGENT_MODEL="openai:gpt-4o-mini"
```

That's it. Keep it simple.
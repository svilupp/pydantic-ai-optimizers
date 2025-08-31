# Customer Support Classification Example

This example demonstrates how to use pydantic-ai-optimizers to improve customer support message classification for StyleHaven, a fictional e-commerce fashion brand. The system extracts structured information from customer messages to help route and prioritize support requests.

## Overview

The customer support classifier extracts:

- **Category**: `billing`, `authentication`, `performance`, `ui_bug`, `shipping`, `account`, `feature_request`, `docs`, `other`
- **Urgency**: `low`, `normal`, `high`, `critical` 
- **Surface**: `web`, `ios`, `android`, `api`, `unknown`
- **Component**: Specific feature/page (e.g., "checkout", "wishlist") 
- **Error Codes**: Order IDs, tracking numbers, error messages, etc.

## Files Structure

```
customer_support/
├── agent.py                    # PydanticAI agent with NativeOutput
├── dataset.py                  # Dataset creation and scoring logic
├── optimize.py                 # Optimization loop entry point
├── utils.py                    # Shared models and utilities
├── data/
│   └── test_cases.json        # 30 diverse test cases
├── evals/
│   ├── support_cases.yaml     # Sample evaluation cases  
│   └── support_cases_schema.json # JSON schema for validation
└── prompts/
    ├── seed.md               # Initial agent prompt
    └── reflection.md         # Reflection agent prompt
```

## Test Cases

The dataset includes 30 carefully designed test cases with varying difficulty:

- **Easy (10 cases)**: Clear, single-issue messages
- **Medium (10 cases)**: Multi-message conversations, context switches, mixed issues
- **Hard (10 cases)**: Vague complaints, technical jargon, edge cases between categories

Example messages range from simple billing inquiries to complex security incidents and technical API issues.

## Scoring System

Classification accuracy is scored using weighted components:
- Category match: 40% (most important)
- Urgency assessment: 20%
- Surface/platform detection: 15%
- Component extraction: 15%
- Error code extraction: 10%

## Usage

### Run Optimization

```bash
# From the customer_support directory
uv run python optimize.py

# Or from project root
uv run python examples/customer_support/optimize.py
```

### Test Evaluation Only

```bash
# Run evaluation on first 3 cases to test the system
uv run python optimize.py --eval-only
```

### Direct Classification

```python
from examples.customer_support.agent import classify_support_message

result = classify_support_message(
    "My credit card was charged twice for order #SH12345"
)
print(result)
# SupportClassification(
#     category='billing',
#     urgency='normal', 
#     surface='unknown',
#     component=None,
#     error_codes=['SH12345']
# )
```

## Configuration

The optimizer uses standard configuration from environment variables or `.env` file:

```bash
OPENAI_API_KEY="your-key"
REFLECTION_MODEL="openai:gpt-4o"  
AGENT_MODEL="openai:gpt-4o"
VALIDATION_BUDGET=20
MAX_POOL_SIZE=16
```

## Key Features

1. **Realistic Test Cases**: 30 diverse customer messages covering common e-commerce support scenarios
2. **Structured Output**: Uses PydanticAI NativeOutput for reliable extraction 
3. **Weighted Scoring**: Prioritizes category accuracy while measuring all fields
4. **Difficulty Levels**: Progressive complexity to stress-test classification
5. **Domain Context**: Fashion e-commerce specific scenarios and terminology

This example showcases how to apply the pydantic-ai-optimizers framework to real-world classification tasks with complex, multi-field structured outputs.
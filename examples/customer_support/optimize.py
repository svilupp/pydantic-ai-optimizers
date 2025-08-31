"""Customer support prompt optimization using Pydantic AI Optimizers."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import logfire
import textprompts
from agent import SupportClassification, create_support_agent
from dataset import CustomerMessage
from dotenv import load_dotenv
from evaluators import create_support_evaluators
from loguru import logger
from pydantic_ai_helpers.evals import ListEquality, ScalarEquals
from pydantic_evals import Dataset

from pydantic_ai_optimizers import Optimizer, make_reflection_agent

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Configure logging
logfire.configure(
    service_name="support-optimizer", send_to_logfire="if-token-present", scrubbing=False
)
logfire.instrument_pydantic_ai()

logger.remove()  # Remove default handler
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)

AGENT_MODEL = "gpt-5-nano"
REFLECTION_MODEL = "gpt-5-mini"


async def run_case(prompt_file: str, customer_message: CustomerMessage) -> SupportClassification:
    """Async function that runs the agent on a support message given the provided prompt file.

    Args:
        prompt_file: Path to the prompt file to use
        customer_message: CustomerMessage input with message text

    Returns:
        SupportClassification output
    """
    global AGENT_MODEL
    # Use the full path for the agent (optimizer passes full paths to generated prompts)
    try:
        agent = create_support_agent(prompt_file=prompt_file, model=AGENT_MODEL)
        result = await agent.run(customer_message.message)
        return result.output
    except Exception as e:
        print(f"Error running agent: {e}")
        # Return mock results
        return SupportClassification(
            category="other", urgency="low", surface="unknown", component=None, identifiers=[]
        )


def load_dataset():
    support_cases_file = Path(__file__).parent / "evals" / "support_cases.yaml"
    dataset = Dataset[CustomerMessage, SupportClassification, Any].from_file(
        support_cases_file, custom_evaluator_types=[ScalarEquals, ListEquality]
    )
    [dataset.add_evaluator(eval) for eval in create_support_evaluators()]
    return dataset


async def run_evaluation_only(dataset: Dataset[CustomerMessage, SupportClassification, Any]):
    """Run just the evaluation to test the system."""

    # Run evaluation on a subset of cases for testing
    logger.info("Running evaluation")

    async def eval_fn(customer_message: CustomerMessage) -> SupportClassification:
        prompt_file = Path(__file__).parent / "prompts/seed.txt"
        return await run_case(prompt_file, customer_message)

    # Run evaluation using pydantic-evals
    report = await dataset.evaluate(eval_fn)
    print(report)


async def run_optimization(dataset: Dataset[CustomerMessage, SupportClassification, Any]):
    """Run the customer support prompt optimization."""
    global REFLECTION_MODEL

    # Let's add GPT-5 tips to the reflection agent
    reflection_agent = make_reflection_agent(
        REFLECTION_MODEL,
        special_instructions=str(
            textprompts.load_prompt(Path(__file__).parent / "prompts/gpt5_tips.txt")
        ),
    )

    # Create optimizer with async run_case function
    optimizer = Optimizer(
        dataset=dataset,
        run_case=run_case,
        reflection_agent=reflection_agent,
        pool_dir=Path(__file__).parent / "prompt_pool",
    )

    # Run optimization with reduced budget for testing
    best = await optimizer.optimize(
        seed_prompt_file=Path(__file__).parent / "prompts/seed.txt",
        full_validation_budget=20,  # Reduced for testing
    )

    print("\n=== Best prompt file ===")
    print(best.prompt_path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer support prompt optimization")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation")
    args = parser.parse_args()

    dataset = load_dataset()

    if args.eval_only:
        asyncio.run(run_evaluation_only(dataset))
    else:
        asyncio.run(run_optimization(dataset))

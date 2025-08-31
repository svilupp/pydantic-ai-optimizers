"""Customer support prompt optimization using Pydantic AI Optimizers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import logfire
import textprompts
from dotenv import load_dotenv
from loguru import logger

from pydantic_ai import Agent
from pydantic_evals import Dataset
from pydantic_ai_helpers.evals import ListEquality, ScalarEquals
    
from pydantic_ai_optimizers import Optimizer, get_optimizer_config, make_reflection_agent

from agent import create_support_agent, SupportClassification
from dataset import CustomerMessage
from evaluators import create_support_evaluators

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")
    
# Configure logging
logfire.configure(service_name="support-optimizer", send_to_logfire="if-token-present", scrubbing=False)
logfire.instrument_pydantic_ai()

logger.remove()  # Remove default handler
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)

AGENT_MODEL="gpt-5-nano"
REFLECTION_MODEL="gpt-5-mini"

async def run_case(prompt_file: str, customer_message: CustomerMessage) -> SupportClassification:
    """Async function that runs the agent on a support message given the provided prompt file.
    
    Args:
        prompt_file: Path to the prompt file to use
        customer_message: CustomerMessage input with message text
        
    Returns:
        SupportClassification output
    """
    global AGENT_MODEL
    # Extract just the filename from the full path for the agent
    prompt_filename = Path(prompt_file).name
    agent = create_support_agent(prompt_file=prompt_filename, model=AGENT_MODEL)
    result = await agent.run(customer_message.message)
    return result.output

def load_dataset():
    support_cases_file = Path(__file__).parent / "evals" / "support_cases.yaml"
    dataset = Dataset[CustomerMessage, SupportClassification, Any].from_file(
        support_cases_file,
        custom_evaluator_types=[ScalarEquals, ListEquality]
    )
    [dataset.add_evaluator(eval) for eval in create_support_evaluators()];
    return dataset

async def run_evaluation_only(dataset: Dataset[CustomerMessage, SupportClassification, Any]):
    """Run just the evaluation to test the system."""


    # Run evaluation on a subset of cases for testing
    logger.info("Running evaluation on first 3 cases...")
    
    async def eval_fn(customer_message: CustomerMessage) -> SupportClassification:
        return await run_case("seed.txt", customer_message)
    
    # Run evaluation using pydantic-evals
    report = await dataset.evaluate(eval_fn)
    print(report)


async def run_optimization(dataset: Dataset[CustomerMessage, SupportClassification, Any]):
    """Run the customer support prompt optimization."""
    global REFLECTION_MODEL

    # Create optimizer with async run_case function
    optimizer = Optimizer(
        dataset=dataset,
        run_case=run_case,
        reflection_agent=make_reflection_agent(REFLECTION_MODEL),
    )
    
    # Run optimization with reduced budget for testing
    best = await optimizer.optimize(
            seed_prompt_file=Path("prompts/seed.txt"),
            full_validation_budget=3,  # Reduced for testing
    )
    
    print("\n=== Best prompt file ===")
    print(best.prompt_path.resolve())


if __name__ == "__main__":
    dataset = load_dataset()
    # asyncio.run(run_evaluation_only(dataset))
    asyncio.run(run_optimization(dataset))
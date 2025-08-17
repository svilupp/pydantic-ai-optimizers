"""Simplified chef prompt optimization using new structure with Pydantic Evals."""

from __future__ import annotations

import asyncio
from pathlib import Path

import logfire
from agent import create_agent, run_agent
from dataset import ChefRequest, ChefResponse, create_chef_dataset
from dotenv import load_dotenv
from evaluators import create_chef_evaluators
from loguru import logger
from pydantic_evals import Dataset

from pydantic_ai_optimizers import Optimizer, get_optimizer_config

# Configure logging
logfire.configure(service_name="promptoptim", send_to_logfire="if-token-present", scrubbing=False)
logfire.instrument_pydantic_ai()

logger.remove()  # Remove default handler
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)


def create_evaluation_dataset() -> Dataset[ChefRequest, ChefResponse, dict]:
    """Create the evaluation dataset with custom evaluators."""

    # Load the base dataset from file
    dataset_file = Path("evals/chef_cases.yaml")
    if dataset_file.exists():
        # Load from serialized file
        dataset = Dataset[ChefRequest, ChefResponse, dict].from_file(dataset_file)
    else:
        # Create new dataset and save it
        dataset = create_chef_dataset()
        dataset_file.parent.mkdir(exist_ok=True)
        dataset.to_file(dataset_file)
        logger.info(f"Created new dataset and saved to {dataset_file}")

    # Add custom evaluators to the dataset
    custom_evaluators = create_chef_evaluators()

    # Create a new dataset with both global and custom evaluators
    all_evaluators = list(dataset.evaluators) + custom_evaluators

    return Dataset[ChefRequest, ChefResponse, dict](cases=dataset.cases, evaluators=all_evaluators)


async def run_evaluation_only():
    """Run just the evaluation to test the system."""

    # Create dataset
    dataset = create_evaluation_dataset()

    # Run evaluation on a subset of cases for testing
    logger.info("Running evaluation on first 2 cases...")

    test_cases = dataset.cases[:2]
    test_dataset = Dataset[ChefRequest, ChefResponse, dict](
        cases=test_cases, evaluators=dataset.evaluators
    )

    # Evaluate with the seed prompt
    seed_prompt = "prompts/chef_seed.txt"

    def sync_runner(inputs: ChefRequest) -> ChefResponse:
        """Synchronous wrapper for the async runner."""

        async def run_single():
            agent = create_agent(seed_prompt)
            result = await run_agent(agent, inputs.user_text)
            return ChefResponse(**result)

        return asyncio.run(run_single())

    try:
        report = test_dataset.evaluate_sync(sync_runner)

        logger.info("Evaluation completed successfully!")
        logger.info(f"Evaluated {len(report.cases)} cases")

        for case in report.cases:
            logger.info(
                f"Case {case.case_id}: {len(case.assertions)} assertions, {len(case.scores)} scores"
            )

        return report
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def main():
    """Run the chef prompt optimization with new structure."""
    # Load environment variables from .env file
    load_dotenv()

    # Get configuration
    config = get_optimizer_config()

    # Create dataset
    dataset = create_evaluation_dataset()

    # Create a wrapper that matches the optimizer's expected signature
    async def optimization_runner(prompt_file: str, inputs: ChefRequest) -> ChefResponse:
        agent = create_agent(prompt_file)
        result = await run_agent(agent, inputs.user_text)
        return ChefResponse(**result)

    # Create optimizer (reflection agent is now handled internally)
    optimizer = Optimizer(
        dataset=dataset,
        run_case=optimization_runner,
        pool_dir=config.pool_dir,
        minibatch_size=config.minibatch_size,
        max_pool_size=config.max_pool_size,
        seed=config.seed,
        keep_failed_mutations=config.keep_failed_mutations,
    )

    # Run optimization
    best = asyncio.run(
        optimizer.optimize(
            seed_prompt_file=Path("prompts/chef_seed.txt"),
            full_validation_budget=config.full_validation_budget,
        )
    )

    print("\n=== Best prompt file ===")
    print(best.prompt_path.resolve())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--eval-only":
        # Run just evaluation for testing
        asyncio.run(run_evaluation_only())
    else:
        # Run full optimization
        main()

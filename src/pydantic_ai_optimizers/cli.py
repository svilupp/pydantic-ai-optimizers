"""Command-line interface for prompt optimization."""

import argparse
import asyncio
from pathlib import Path

from .optimizer import Optimizer


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Optimize AI agent prompts through iterative improvement"
    )

    parser.add_argument("seed_prompt", type=Path, help="Path to the seed prompt file to optimize")

    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to the evaluation dataset"
    )

    parser.add_argument(
        "--run-case-module",
        required=True,
        help="Python module path for the run_case function (e.g., 'examples.chef.runner')",
    )

    parser.add_argument(
        "--reflection-agent-module",
        required=True,
        help="Python module path for the reflection agent (e.g., 'examples.chef.runner')",
    )

    parser.add_argument(
        "--pool-dir",
        type=Path,
        default="./prompt_pool",
        help="Directory to store prompt pool (default: ./prompt_pool)",
    )

    parser.add_argument(
        "--minibatch-size", type=int, default=3, help="Minibatch size for evaluation (default: 3)"
    )

    parser.add_argument(
        "--max-pool-size", type=int, default=10, help="Maximum pool size (default: 10)"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    parser.add_argument(
        "--keep-failed-mutations", action="store_true", help="Keep failed mutations in memory"
    )

    parser.add_argument(
        "--validation-budget", type=int, default=100, help="Full validation budget (default: 100)"
    )

    args = parser.parse_args()

    # Dynamic import of user-provided modules
    import importlib

    run_case_module = importlib.import_module(args.run_case_module)
    reflection_module = importlib.import_module(args.reflection_agent_module)

    # Get the functions from modules
    run_case = run_case_module.make_run_case()
    reflection_agent = reflection_module.make_reflection_agent()
    dataset = run_case_module.build_dataset(args.dataset)

    # Create optimizer
    optimizer = Optimizer(
        dataset=dataset,
        run_case=run_case,
        reflection_agent=reflection_agent,
        pool_dir=args.pool_dir,
        minibatch_size=args.minibatch_size,
        max_pool_size=args.max_pool_size,
        seed=args.seed,
        keep_failed_mutations=args.keep_failed_mutations,
    )

    # Run optimization
    best = asyncio.run(
        optimizer.optimize(
            seed_prompt_file=args.seed_prompt,
            full_validation_budget=args.validation_budget,
        )
    )

    print(f"Best prompt file: {best.prompt_path.resolve()}")


if __name__ == "__main__":
    main()

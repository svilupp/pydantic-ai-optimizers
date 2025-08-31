"""Minimal test of the customer support optimization."""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai_optimizers import Optimizer, get_optimizer_config
from pydantic_ai import Agent

from dataset import create_support_dataset
from optimize import run_support_classification_sync

# Load environment
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Set minimal config
os.environ['VALIDATION_BUDGET'] = '2'
os.environ['MAX_POOL_SIZE'] = '2'

def main():
    """Run minimal optimization test."""
    print("🚀 Starting minimal optimization test...")
    
    # Create dataset
    dataset = create_support_dataset()
    print(f"📋 Dataset created with {len(dataset)} cases")
    
    # Get config
    config = get_optimizer_config()
    print(f"⚙️ Config: budget={config.full_validation_budget}, pool_size={config.max_pool_size}")
    
    # Create reflection agent
    reflection_agent = Agent[None, str](
        model="openai:gpt-4o",
        system_prompt="You help improve customer support classification prompts."
    )
    
    try:
        # Create optimizer
        optimizer = Optimizer(
            dataset=dataset,
            run_case=run_support_classification_sync,
            reflection_agent=reflection_agent,
            pool_dir=config.pool_dir,
            minibatch_size=1,
            max_pool_size=config.max_pool_size,
            seed=config.seed,
            keep_failed_mutations=config.keep_failed_mutations,
        )
        print("✅ Optimizer created successfully!")
        
        # Run optimization
        print("🔄 Running optimization...")
        best = asyncio.run(
            optimizer.optimize(
                seed_prompt_file=Path("prompts/seed.md"),
                full_validation_budget=config.full_validation_budget,
            )
        )
        
        print("🎉 Optimization completed!")
        print(f"📂 Best prompt: {best.prompt_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
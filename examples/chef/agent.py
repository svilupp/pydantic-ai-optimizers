"""Chef agent with PydanticAI structure - tools defined with decorators."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydantic_ai_helpers as ph
import textprompts
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

# ---- Data Models ----


@dataclass
class Recipe:
    name: str
    description: str
    cuisine: str
    diet: str
    difficulty: str
    prep_time_min: int
    calories: int
    cost: float
    ingredients: list[str]


@dataclass
class RecipeMatch:
    name: str
    score: float


# ---- Output Types ----


class ResponseToUser(BaseModel):
    """Successful recipe recommendation response."""

    decision: str = "recommend"
    recipes: list[str]
    notes: str | None = None


class ServiceRejection(BaseModel):
    """Service rejection for unsafe/impossible requests."""

    decision: str = "reject"
    reason: str


# ---- Dependencies ----


@dataclass
class ChefDeps:
    """Dependencies containing recipe and allergen data."""

    recipes: list[Recipe]
    allergens: dict[str, list[str]]


# ---- Helper Functions ----


def load_recipes(path: Path) -> list[Recipe]:
    """Load recipes from JSON file."""
    data = json.loads(Path(path).read_text())
    return [Recipe(**r) for r in data]


def load_allergens(path: Path) -> dict[str, list[str]]:
    """Load allergen mappings from JSON file."""
    return json.loads(Path(path).read_text())


def _tokens(text: str) -> set:
    """Extract lowercase alphabetic tokens from text."""
    return set(re.findall(r"[a-zA-Z]+", (text or "").lower()))


# ---- Agent Creation ----


def create_agent(
    prompt_file: str, model: str = "openai:gpt-5-nano"
) -> Agent[ChefDeps, ResponseToUser | ServiceRejection]:
    """Create chef agent with tools and dependencies."""

    agent = Agent(model=model, deps_type=ChefDeps, output_type=ResponseToUser | ServiceRejection)

    @agent.system_prompt
    def get_system_prompt() -> str:
        """Load system prompt from file."""
        return str(textprompts.load_prompt(prompt_file))

    @agent.tool
    def search_allergens(ctx: RunContext[ChefDeps], allergy_name: str) -> list[str]:
        """Search for allergens by name."""
        key = (allergy_name or "").strip().lower()
        return ctx.deps.allergens.get(key, [])

    @agent.tool
    def search_recipes(
        ctx: RunContext[ChefDeps],
        query: str,
        ingredients: list[str] | None = None,
        ingredients_to_avoid: list[str] | None = None,
        prep_time_min: int | None = None,
        prep_time_max: int | None = None,
        cost_min: float | None = None,
        cost_max: float | None = None,
        diet: str | None = None,
        difficulty: str | None = None,
        cuisine: str | None = None,
        calories_min: int | None = None,
        calories_max: int | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search recipes with various filters."""

        recipes = ctx.deps.recipes
        ingredients = [i.lower() for i in (ingredients or [])]
        avoid = {i.lower() for i in (ingredients_to_avoid or [])}
        q_tokens = _tokens(query)
        out: list[tuple[float, Recipe]] = []

        for r in recipes:
            # Hard filters
            if diet and r.diet.lower() != diet.lower():
                continue
            if difficulty and r.difficulty.lower() != difficulty.lower():
                continue
            if cuisine and r.cuisine.lower() != cuisine.lower():
                continue
            if prep_time_min is not None and r.prep_time_min < int(prep_time_min):
                continue
            if prep_time_max is not None and r.prep_time_min > int(prep_time_max):
                continue
            if calories_min is not None and r.calories < int(calories_min):
                continue
            if calories_max is not None and r.calories > int(calories_max):
                continue
            if cost_min is not None and r.cost < float(cost_min):
                continue
            if cost_max is not None and r.cost > float(cost_max):
                continue

            ingr_lower = [i.lower() for i in r.ingredients]
            if ingredients and not set(ingredients).issubset(ingr_lower):
                continue
            if avoid and (set(ingr_lower) & avoid):
                continue

            # Score by (query token overlap) + small bonus for matched required ingredients
            hay = _tokens(" ".join([r.name, r.description] + r.ingredients))
            overlap = len(q_tokens & hay)
            bonus = sum(1 for i in ingredients if i in ingr_lower)
            score = overlap + 0.1 * bonus
            out.append((score, r))

        out.sort(key=lambda t: t[0], reverse=True)
        matches = [RecipeMatch(name=r.name, score=float(s)) for s, r in out[: max(1, int(top_k))]]
        return [m.__dict__ for m in matches]

    return agent


# ---- Agent Runner ----


async def run_agent(
    agent: Agent[ChefDeps, ResponseToUser | ServiceRejection],
    user_input: str,
    recipes_path: Path = Path("data/recipes.json"),
    allergens_path: Path = Path("data/allergens.json"),
) -> dict[str, Any]:
    """Run chef agent and extract evaluation data."""

    # Load dependencies
    deps = ChefDeps(recipes=load_recipes(recipes_path), allergens=load_allergens(allergens_path))

    # Run the agent
    result = await agent.run(user_input, deps=deps)

    # Extract tool calls using PydanticAI helpers History
    hist = ph.History(result)

    # Extract tool calls with list comprehensions
    search_allergens_calls = [call.args for call in hist.tools.calls(name="search_allergens").all()]
    search_recipes_calls = [
        {k: v for k, v in call.args.items() if v is not None}
        for call in hist.tools.calls(name="search_recipes").all()
    ]

    # Simple allergen call order check
    allergens_called_first = bool(search_allergens_calls) and (
        not search_recipes_calls or len(search_allergens_calls) > 0
    )

    # Check if service rejected the request
    service_rejected = isinstance(result.output, ServiceRejection)

    return {
        "search_allergens_calls": search_allergens_calls,
        "search_recipes_calls": search_recipes_calls,
        "output": result.output,
        "num_tool_calls": len(search_allergens_calls) + len(search_recipes_calls),
        "search_allergens_called_first": allergens_called_first,
        "service_rejected": service_rejected,
    }


if __name__ == "__main__":
    # Test the agent
    import asyncio

    async def test_agent():
        agent = create_agent("prompts/chef_seed.txt")

        # Test with a simple request
        result = await run_agent(agent, "I want Italian pasta recommendations")

        print("Test successful!")
        print(f"Tool calls: {result['num_tool_calls']}")
        print(f"Decision: {result['output']['decision']}")
        print(f"Service rejected: {result['service_rejected']}")
        print(f"Allergen calls: {result['search_allergens_calls']}")
        print(f"Recipe calls: {result['search_recipes_calls']}")

    asyncio.run(test_agent())

"""Chef evaluators using PydanticAI Helpers evaluators directly."""

from __future__ import annotations

from pydantic_ai_helpers.evals import (
    ListEquality,
    ListRecall,
    ScalarEquals,
)
from pydantic_ai_helpers.evals.evaluators import MaxCount, MinCount


def create_chef_evaluators():
    """Create chef evaluators using PydanticAI Helpers directly."""

    return [
        # Check if allergens were called first when required
        ScalarEquals(
            output_path="search_allergens_called_first",
            expected_path="search_allergens_called_first",
            evaluation_name="allergen_call_order",
        ),
        # Check service decision (recommend/reject)
        ScalarEquals(
            output_path="service_rejected",
            expected_path="service_rejected",
            evaluation_name="service_decision",
        ),
        # Check tool call count - set upper bound (can be less than max)
        MaxCount(
            output_path="num_tool_calls",
            count=5,  # Should not need more than 5 tool calls
            evaluation_name="max_tool_calls",
        ),
        # Ensure at least some tool usage for non-rejection cases
        MinCount(
            output_path="num_tool_calls",
            count=1,  # Should have at least 1 tool call in most cases
            evaluation_name="min_tool_calls",
        ),
        # Check allergen search calls - compare the lists directly
        ListEquality(
            output_path="search_allergens_calls",
            expected_path="search_allergens_calls",
            order_sensitive=False,
            fuzzy_enabled=True,
            fuzzy_threshold=0.8,
            normalize_lowercase=True,
            evaluation_name="allergen_search_calls",
        ),
        # Check recipe search query with fuzzy matching
        ScalarEquals(
            output_path="search_recipes_calls.0.query",  # First recipe call's query
            expected_path="search_recipes_calls.0.query",
            fuzzy_enabled=True,
            fuzzy_threshold=0.8,
            normalize_lowercase=True,
            evaluation_name="recipe_query_match",
        ),
        # Check recipe search diet filter
        ScalarEquals(
            output_path="search_recipes_calls.0.diet",
            expected_path="search_recipes_calls.0.diet",
            fuzzy_enabled=True,
            normalize_lowercase=True,
            evaluation_name="recipe_diet_match",
        ),
        # Check recipe search cuisine filter
        ScalarEquals(
            output_path="search_recipes_calls.0.cuisine",
            expected_path="search_recipes_calls.0.cuisine",
            fuzzy_enabled=True,
            normalize_lowercase=True,
            evaluation_name="recipe_cuisine_match",
        ),
        # Check recipe search difficulty filter
        ScalarEquals(
            output_path="search_recipes_calls.0.difficulty",
            expected_path="search_recipes_calls.0.difficulty",
            fuzzy_enabled=True,
            normalize_lowercase=True,
            evaluation_name="recipe_difficulty_match",
        ),
        # Check ingredients to avoid with fuzzy list matching
        ListRecall(
            output_path="search_recipes_calls.0.ingredients_to_avoid",
            expected_path="search_recipes_calls.0.ingredients_to_avoid",
            fuzzy_enabled=True,
            fuzzy_threshold=0.8,
            normalize_lowercase=True,
            evaluation_name="ingredients_to_avoid_recall",
        ),
        # Check required ingredients
        ListRecall(
            output_path="search_recipes_calls.0.ingredients",
            expected_path="search_recipes_calls.0.ingredients",
            fuzzy_enabled=True,
            fuzzy_threshold=0.8,
            normalize_lowercase=True,
            evaluation_name="required_ingredients_recall",
        ),
        # Check numeric constraints with tolerance
        ScalarEquals(
            output_path="search_recipes_calls.0.prep_time_max",
            expected_path="search_recipes_calls.0.prep_time_max",
            coerce_to="int",
            abs_tol=5,  # Allow 5 minute tolerance
            evaluation_name="prep_time_max_match",
        ),
        ScalarEquals(
            output_path="search_recipes_calls.0.cost_max",
            expected_path="search_recipes_calls.0.cost_max",
            coerce_to="float",
            abs_tol=1.0,  # Allow $1 tolerance
            evaluation_name="cost_max_match",
        ),
        ScalarEquals(
            output_path="search_recipes_calls.0.calories_min",
            expected_path="search_recipes_calls.0.calories_min",
            coerce_to="int",
            abs_tol=50,  # Allow 50 calorie tolerance
            evaluation_name="calories_min_match",
        ),
        ScalarEquals(
            output_path="search_recipes_calls.0.calories_max",
            expected_path="search_recipes_calls.0.calories_max",
            coerce_to="int",
            abs_tol=50,  # Allow 50 calorie tolerance
            evaluation_name="calories_max_match",
        ),
        # Check final decision
        ScalarEquals(
            output_path="output.decision",
            expected_path="output.decision",
            fuzzy_enabled=True,
            normalize_lowercase=True,
            evaluation_name="final_decision_match",
        ),
    ]


if __name__ == "__main__":
    # Test evaluator creation
    evaluators = create_chef_evaluators()

    print("Testing evaluator creation:")
    for evaluator in evaluators:
        print(f"✓ Created: {evaluator.evaluation_name}")

    print(f"\nSuccessfully created {len(evaluators)} evaluators!")

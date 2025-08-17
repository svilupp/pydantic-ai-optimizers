"""Chef evaluation dataset with typed interfaces for Pydantic Evals."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LLMJudge


class ChefRequest(BaseModel):
    """Input for chef agent - user's food request."""

    user_text: str


class ChefResponse(BaseModel):
    """Expected structure extracted from chef agent results."""

    search_allergens_calls: list[dict[str, Any]]
    search_recipes_calls: list[dict[str, Any]]
    output: dict[str, Any]
    num_tool_calls: int
    search_allergens_called_first: bool
    service_rejected: bool


def create_chef_dataset() -> Dataset[ChefRequest, ChefResponse, Any]:
    """Create the chef evaluation dataset with 10 diverse test cases."""

    cases = [
        # Case 1: Simple allergen check with gluten avoidance
        Case(
            name="gluten_allergy_pasta",
            inputs=ChefRequest(user_text="I have a gluten allergy and want pasta recommendations"),
            expected_output=ChefResponse(
                search_allergens_calls=[{"allergy_name": "gluten"}],
                search_recipes_calls=[
                    {
                        "query": "pasta",
                        "ingredients_to_avoid": [
                            "wheat flour",
                            "wheat pasta",
                            "wheat bread",
                            "wheat bun",
                            "wheat tortilla",
                            "pasta",
                            "bread",
                            "pizza dough",
                            "soy sauce",
                        ],
                    }
                ],
                output={"decision": "recommend"},
                num_tool_calls=2,
                search_allergens_called_first=True,
                service_rejected=False,
            ),
            metadata={"focus": "allergen_avoidance", "complexity": "simple"},
            evaluators=(),
        ),
        # Case 2: Multiple dietary constraints with budget
        Case(
            name="vegan_italian_budget",
            inputs=ChefRequest(
                user_text="I want vegan Italian food under $10 that takes less than 30 minutes"
            ),
            expected_output=ChefResponse(
                search_allergens_calls=[],
                search_recipes_calls=[
                    {
                        "query": "italian vegan",
                        "diet": "vegan",
                        "cuisine": "italian",
                        "cost_max": 10.0,
                        "prep_time_max": 30,
                    }
                ],
                output={"decision": "recommend"},
                num_tool_calls=1,
                search_allergens_called_first=False,
                service_rejected=False,
            ),
            metadata={"focus": "multi_filter", "complexity": "medium"},
            evaluators=(),
        ),
        # Case 3: Nut allergy with specific ingredients
        Case(
            name="tree_nut_allergy_salad",
            inputs=ChefRequest(
                user_text="I'm allergic to tree nuts but want a healthy salad with quinoa"
            ),
            expected_output=ChefResponse(
                search_allergens_calls=[{"allergy_name": "tree nut"}],
                search_recipes_calls=[
                    {
                        "query": "salad quinoa",
                        "ingredients": ["quinoa"],
                        "ingredients_to_avoid": [
                            "almonds",
                            "cashews",
                            "walnuts",
                            "hazelnuts",
                            "pecans",
                            "pistachios",
                        ],
                    }
                ],
                output={"decision": "recommend"},
                num_tool_calls=2,
                search_allergens_called_first=True,
                service_rejected=False,
            ),
            metadata={"focus": "allergen_with_ingredients", "complexity": "medium"},
            evaluators=(),
        ),
        # Case 4: High-end carnivore with calorie constraints
        Case(
            name="carnivore_high_end_calories",
            inputs=ChefRequest(
                user_text="I follow a carnivore diet, want something fancy, and need at least 700 calories"
            ),
            expected_output=ChefResponse(
                search_allergens_calls=[],
                search_recipes_calls=[
                    {
                        "query": "carnivore fancy meat",
                        "diet": "carnivore",
                        "calories_min": 700,
                        "cost_min": 15.0,
                    }
                ],
                output={"decision": "recommend"},
                num_tool_calls=1,
                search_allergens_called_first=False,
                service_rejected=False,
            ),
            metadata={"focus": "diet_calories_cost", "complexity": "medium"},
            evaluators=(),
        ),
        # Case 5: Shellfish allergy with pescatarian diet
        Case(
            name="shellfish_allergy_pescatarian",
            inputs=ChefRequest(
                user_text="I'm pescatarian but allergic to shellfish, what fish dishes do you recommend?"
            ),
            expected_output=ChefResponse(
                search_allergens_calls=[{"allergy_name": "shellfish"}],
                search_recipes_calls=[
                    {
                        "query": "fish pescatarian",
                        "diet": "pescatarian",
                        "ingredients_to_avoid": ["shrimp", "crab", "lobster"],
                    }
                ],
                output={"decision": "recommend"},
                num_tool_calls=2,
                search_allergens_called_first=True,
                service_rejected=False,
            ),
            metadata={"focus": "diet_with_allergen", "complexity": "medium"},
            evaluators=(),
        ),
        # Case 6: Complex multi-allergen scenario
        Case(
            name="multiple_allergies_complex",
            inputs=ChefRequest(
                user_text="I'm allergic to dairy and eggs, want vegetarian Thai food that's easy to make"
            ),
            expected_output=ChefResponse(
                search_allergens_calls=[{"allergy_name": "dairy"}, {"allergy_name": "egg"}],
                search_recipes_calls=[
                    {
                        "query": "vegetarian thai",
                        "diet": "vegetarian",
                        "cuisine": "thai",
                        "difficulty": "easy",
                        "ingredients_to_avoid": [
                            "milk",
                            "butter",
                            "cream",
                            "yogurt",
                            "cheddar",
                            "mozzarella",
                            "parmesan",
                            "feta",
                            "halloumi",
                            "ghee",
                            "egg",
                            "egg yolk",
                            "mayonnaise",
                        ],
                    }
                ],
                output={"decision": "recommend"},
                num_tool_calls=3,
                search_allergens_called_first=True,
                service_rejected=False,
            ),
            metadata={"focus": "multiple_allergens", "complexity": "complex"},
            evaluators=(),
        ),
        # Case 7: Rejection scenario - unrealistic constraints
        Case(
            name="impossible_constraints_reject",
            inputs=ChefRequest(
                user_text="I want a vegan dish with beef and chicken that costs under $1"
            ),
            expected_output=ChefResponse(
                search_allergens_calls=[],
                search_recipes_calls=[],
                output={"decision": "reject", "reason": "Contradictory requirements"},
                num_tool_calls=0,
                search_allergens_called_first=False,
                service_rejected=True,
            ),
            metadata={"focus": "rejection_logic", "complexity": "simple"},
            evaluators=(),
        ),
        # Case 8: Soy and sesame allergies with Asian cuisine
        Case(
            name="soy_sesame_allergy_asian",
            inputs=ChefRequest(
                user_text="I love Asian food but I'm allergic to soy and sesame. What can I eat?"
            ),
            expected_output=ChefResponse(
                search_allergens_calls=[{"allergy_name": "soy"}, {"allergy_name": "sesame"}],
                search_recipes_calls=[
                    {
                        "query": "asian",
                        "cuisine": "chinese",
                        "ingredients_to_avoid": [
                            "soybeans",
                            "tofu",
                            "tempeh",
                            "soy sauce",
                            "edamame",
                            "sesame seeds",
                            "tahini",
                        ],
                    }
                ],
                output={"decision": "recommend"},
                num_tool_calls=3,
                search_allergens_called_first=True,
                service_rejected=False,
            ),
            metadata={"focus": "cuisine_with_allergens", "complexity": "complex"},
            evaluators=(),
        ),
        # Case 9: Time-constrained cooking with specific difficulty
        Case(
            name="quick_medium_difficulty",
            inputs=ChefRequest(
                user_text="I have 25 minutes and want something challenging but not too hard to cook"
            ),
            expected_output=ChefResponse(
                search_allergens_calls=[],
                search_recipes_calls=[
                    {"query": "quick challenging", "prep_time_max": 25, "difficulty": "medium"}
                ],
                output={"decision": "recommend"},
                num_tool_calls=1,
                search_allergens_called_first=False,
                service_rejected=False,
            ),
            metadata={"focus": "time_difficulty", "complexity": "simple"},
            evaluators=(),
        ),
        # Case 10: Comprehensive scenario with all constraints
        Case(
            name="comprehensive_all_constraints",
            inputs=ChefRequest(
                user_text="I'm allergic to peanuts, want French vegetarian food under $12, max 45 minutes, medium difficulty, between 500-750 calories"
            ),
            expected_output=ChefResponse(
                search_allergens_calls=[{"allergy_name": "peanut"}],
                search_recipes_calls=[
                    {
                        "query": "french vegetarian",
                        "diet": "vegetarian",
                        "cuisine": "french",
                        "cost_max": 12.0,
                        "prep_time_max": 45,
                        "difficulty": "medium",
                        "calories_min": 500,
                        "calories_max": 750,
                        "ingredients_to_avoid": ["peanuts", "peanut butter", "peanut oil"],
                    }
                ],
                output={"decision": "recommend"},
                num_tool_calls=2,
                search_allergens_called_first=True,
                service_rejected=False,
            ),
            metadata={"focus": "comprehensive", "complexity": "complex"},
            evaluators=(),
        ),
    ]

    # Global evaluators that apply to all cases
    global_evaluators = [
        IsInstance(type_name="ChefResponse"),
        LLMJudge(
            rubric="Response should appropriately handle food allergies and dietary restrictions for safety",
            include_input=True,
            model="anthropic:claude-3-7-sonnet-latest",
        ),
        LLMJudge(
            rubric="Tool usage should be logical - allergen searches should precede recipe searches when allergies are mentioned",
            include_input=True,
            model="anthropic:claude-3-7-sonnet-latest",
        ),
    ]

    return Dataset[ChefRequest, ChefResponse, Any](cases=cases, evaluators=global_evaluators)


if __name__ == "__main__":
    # Test dataset creation and serialization
    dataset = create_chef_dataset()
    print(f"Created dataset with {len(dataset.cases)} cases")

    # Export to file
    from pathlib import Path

    evals_dir = Path("evals")
    evals_dir.mkdir(exist_ok=True)

    dataset_file = evals_dir / "chef_cases.yaml"
    dataset.to_file(dataset_file)
    print(f"Dataset saved to {dataset_file}")

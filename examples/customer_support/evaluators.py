"""Customer support evaluators using PydanticAI Helpers evaluators directly."""

from __future__ import annotations

from pydantic_ai_helpers.evals import (
    ListEquality,
    ScalarEquals,
)


def create_support_evaluators():
    """Create customer support evaluators using PydanticAI Helpers directly."""

    return [
        ScalarEquals(
            output_path="category",
            expected_path="category",
            evaluation_name="category_match",
        ),
        ScalarEquals(
            output_path="urgency",
            expected_path="urgency",
            evaluation_name="urgency_match",
        ),
        ScalarEquals(
            output_path="surface",
            expected_path="surface",
            evaluation_name="surface_match",
        ),
        ScalarEquals(
            output_path="component",
            expected_path="component",
            fuzzy_enabled=True,
            fuzzy_threshold=0.8,
            normalize_lowercase=True,
            evaluation_name="component_match",
        ),
        ListEquality(
            output_path="identifiers",
            expected_path="identifiers",
            order_sensitive=False,  # Order doesn't matter for identifiers
            fuzzy_enabled=False,
            normalize_lowercase=True,
            evaluation_name="identifiers_match",
        ),
    ]


if __name__ == "__main__":
    # Test evaluator creation
    evaluators = create_support_evaluators()

    print("Testing evaluator creation:")
    for evaluator in evaluators:
        print(f"✓ Created: {evaluator.evaluation_name}")

    print(f"\nSuccessfully created {len(evaluators)} evaluators!")
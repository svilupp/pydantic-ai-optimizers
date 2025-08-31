"""Customer support evaluation dataset using pydantic-evals directly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter
from pydantic_evals import Case, Dataset

from agent import SupportClassification, TestCase


class CustomerMessage(BaseModel):
    """Input for support agent - customer message."""
    message: str


def load_test_cases() -> list[TestCase]:
    """Load test cases from the JSON file."""
    data_file = Path(__file__).parent / "data" / "test_cases.json"
    with open(data_file) as f:
        data = TypeAdapter(list[TestCase]).validate_json(f.read())
    return data


def create_support_dataset() -> Dataset[CustomerMessage, SupportClassification, Any]:
    """Create the customer support evaluation dataset using pydantic-evals."""
    
    test_cases = load_test_cases()
    
    # Convert test cases to pydantic-evals Case format
    cases = []
    for test_case in test_cases:
        case = Case(
            name=test_case.id,
            inputs=CustomerMessage(message=test_case.message),
            expected_output=test_case.expected,  # SupportClassification directly
            metadata={
                "difficulty": test_case.difficulty,
                "notes": test_case.notes
            }
        )
        cases.append(case)
    
    # Create dataset with global evaluators
    dataset = Dataset[CustomerMessage, SupportClassification, Any](
        cases=cases,
    )
    
    return dataset




if __name__ == "__main__":
    
    # Create and serialize dataset
    support_dataset = create_support_dataset()
    print(f'\nCreated dataset with {len(support_dataset.cases)} cases')
    support_cases_file = Path(__file__).parent / "evals" / "support_cases.yaml"
    support_cases_file.parent.mkdir(exist_ok=True)
    support_dataset.to_file(support_cases_file)

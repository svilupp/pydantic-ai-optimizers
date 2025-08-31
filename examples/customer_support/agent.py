"""Customer support classification agent using PydanticAI."""

from __future__ import annotations

from typing import Literal

import textprompts
from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIModelSettings


class SupportClassification(BaseModel):
    """Classification output for customer support messages."""

    category: Literal[
        "billing",
        "authentication",
        "performance",
        "ui_bug",
        "shipping",
        "account",
        "feature_request",
        "docs",
        "other",
    ] = Field(description="Primary category of the customer issue")

    urgency: Literal["low", "normal", "high", "critical"] = Field(
        description="Urgency level based on business impact and customer sentiment. Security, payment-blocking, lost/damaged shipment with imminent need, and widespread outages are high (or critical for security)."
    )

    surface: Literal["web", "mobile", "ios", "android", "api", "unknown"] = Field(
        description="Platform or interface where the issue occurs. Use the most specific platform available."
    )

    component: str | None = Field(
        default=None,
        description="Specific component, page, or feature mentioned (e.g., 'checkout', 'wishlist', 'search')",
    )

    identifiers: list[str] = Field(
        default_factory=list,
        description="Error codes, order IDs, specific URL paths that failed, or technical identifiers mentioned in the message. Remove leading `#` from order IDs.",
    )


class TestCase(BaseModel):
    """Single test case for customer support classification."""

    id: str = Field(description="Unique identifier for the test case")
    message: str = Field(description="Customer support message")
    expected: SupportClassification = Field(description="Expected classification")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Test case difficulty")
    notes: str | None = Field(default=None, description="Additional context or notes")


def create_support_agent(
    prompt_file: str = "prompts/seed.md", model: str = "openai:gpt-5-nano"
) -> Agent[None, SupportClassification]:
    """Create a customer support classification agent.

    Args:
        prompt_file: Path to the prompt file to use
        model: Model identifier to use for the agent

    Returns:
        Configured PydanticAI agent for support classification
    """

    # Load the system prompt - handle both absolute paths and relative filenames
    system_prompt = str(textprompts.load_prompt(prompt_file))

    # Create agent with NativeOutput for structured extraction
    agent = Agent[None, SupportClassification](
        model=model,
        output_type=NativeOutput(
            SupportClassification,
            name="SupportClassification",
            description="Extract structured information from customer support messages including category, urgency, platform, component, and error codes.",
        ),
        system_prompt=system_prompt,
        model_settings=OpenAIModelSettings(openai_reasoning_effort="minimal"),
    )

    return agent

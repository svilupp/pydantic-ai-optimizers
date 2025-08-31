from pathlib import Path

import textprompts  # type: ignore
from pydantic_ai import Agent  # type: ignore

from ..config import get_model_config


## PLEASE FIX and define properly in this library with config
def make_reflection_agent(
    model: str | None = None, special_instructions: str | None = None
) -> Agent:
    """Create a reflection agent for improving prompts.

    Args:
        model: Model to use for the agent. If None, uses config default.
        special_instructions: Optional special instructions to include in the prompt, eg, to change formatting or length of the prompt.
            If provided, will be wrapped in <special_instructions> tags.
    """
    if model is None:
        model = get_model_config()["reflection_model"]

    # Format special instructions if provided
    formatted_special_instructions = ""
    if special_instructions is not None:
        formatted_special_instructions = (
            f"\n<special_instructions>\n{special_instructions}\n</special_instructions>\n"
        )

    template = textprompts.load_prompt(
        Path(__file__).parent / "prompts" / "reflection_instructions.txt"
    )
    instr = template.prompt.format(special_instructions=formatted_special_instructions)
    return Agent(model=model, instructions=instr)

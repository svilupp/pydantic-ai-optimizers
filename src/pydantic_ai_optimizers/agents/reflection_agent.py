from pathlib import Path

import textprompts  # type: ignore
from pydantic_ai import Agent  # type: ignore

from ..config import get_model_config


## PLEASE FIX and define properly in this library with config
def make_reflection_agent(model: str | None = None) -> Agent:
    """Create a reflection agent for improving prompts."""
    if model is None:
        model = get_model_config()["reflection_model"]

    instr = str(
        textprompts.load_prompt(Path(__file__).parent / "prompts" / "reflection_instructions.txt")
    )
    return Agent(model=model, instructions=instr)

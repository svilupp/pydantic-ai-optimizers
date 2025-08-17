from pathlib import Path

import textprompts
from pydantic_ai import Agent

from ..config import get_model_config


## PLEASE FIX and define properly in this library with config
def make_reflection_agent(model: str = None):
    """Create a reflection agent for improving prompts."""
    if model is None:
        model = get_model_config()["reflection_model"]

    instr = str(
        textprompts.load_prompt(Path(__file__).parent / "prompts" / "reflection_instructions.txt")
    )
    return Agent(model=model, instructions=instr)

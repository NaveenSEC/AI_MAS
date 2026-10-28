"""The LLM backend for this project: Claude through the Anthropic API.

A "backend" is whatever turns a prompt into text. Here it is ClaudeLLM, which
sends each agent's request to Claude over the internet. It needs an Anthropic
API key, and each request is billed to your API account.
"""

# Lets us write modern type hints on older Python versions too.
from __future__ import annotations

# Protocol: describes a required "shape" (which methods a class must have)
# without forcing classes to inherit from anything.
from typing import Protocol

# The official Anthropic SDK (installed with `pip install -r requirements.txt`).
import anthropic

# The Claude model used by default. Change this to use a different Claude model.
MODEL = "claude-opus-5-5"


# Every backend must look like this: a `complete` method that takes the
# system prompt (the agent's role) and the user prompt (the task),
# and returns the model's reply as a string.
class LLM(Protocol):
    # The "..." means "no body here": this only describes the method's signature.
    def complete(self, system: str, prompt: str) -> str: ...


class ClaudeLLM:
    """Calls Claude via the official Anthropic SDK."""

    # Runs once, when you write ClaudeLLM().
    #   model  -> which Claude model to call (default: MODEL above)
    #   effort -> how hard the model thinks: "low", "medium", "high", "xhigh" or "max"
    def __init__(self, model: str = MODEL, effort: str = "medium") -> None:
        # Create the API client. It finds your credentials by itself,
        # for example from the ANTHROPIC_API_KEY environment variable.
        self.client = anthropic.Anthropic()
        # Remember the chosen model for every request.
        self.model = model
        # Remember the chosen effort level for every request.
        self.effort = effort

    # Send one request to Claude and return the text of its reply.
    def complete(self, system: str, prompt: str) -> str:
        # Server-side fallback re-runs a declined request on a recommended
        # model instead of returning a refusal.
        response = self.client.beta.messages.create(  # beta endpoint: needed for `fallbacks`
            model=self.model,  # which Claude model answers
            max_tokens=16000,  # upper limit on the reply length (in tokens, roughly ¾ of a word each)
            system=system,  # the agent's role instructions
            thinking={"type": "adaptive"},  # let Claude decide how much to think before answering
            output_config={"effort": self.effort},  # how much effort to spend (see __init__)
            betas=["server-side-fallback-2026-07-01"],  # switch on the fallback feature
            fallbacks="default",  # if Claude declines, retry on Anthropic's recommended model
            messages=[{"role": "user", "content": prompt}],  # the task, as one user message
        )
        # "refusal" means every model in the fallback chain declined the request.
        if response.stop_reason == "refusal":
            # Stop the program with a clear message instead of returning an empty answer.
            raise RuntimeError("The model declined this request.")
        # A reply is a list of content blocks (text, thinking, ...).
        # Keep only the text blocks, join them into one string,
        # and trim spaces and newlines from both ends.
        return "".join(b.text for b in response.content if b.type == "text").strip()

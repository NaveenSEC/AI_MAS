"""The LLM backend for Demo mode: canned answers, no model at all.

A "backend" is whatever turns a prompt into text. In this project it is
MockLLM, which returns fixed replies, so you can watch the agents work
together without an API key, an internet connection or a downloaded model.
"""

# Lets us write modern type hints on older Python versions too.
from __future__ import annotations

# Protocol: describes a required "shape" (which methods a class must have)
# without forcing classes to inherit from anything.
from typing import Protocol


# Every backend must look like this: a `complete` method that takes the
# system prompt (the agent's role) and the user prompt (the task),
# and returns the model's reply as a string.
class LLM(Protocol):
    # The "..." means "no body here": this only describes the method's signature.
    def complete(self, system: str, prompt: str) -> str: ...


class MockLLM:
    """Deterministic offline backend so the pipeline runs without an API key."""

    # Return a fixed answer based on which agent is asking.
    def complete(self, system: str, prompt: str) -> str:
        # Work out the agent's role from its system prompt:
        # "You are the Planner. Break ..." -> split at "." -> "You are the Planner"
        # -> remove "You are the " -> "Planner".
        role = system.split(".")[0].removeprefix("You are the ").strip()
        # Use the first line of the task (at most 80 characters) as the topic.
        topic = prompt.splitlines()[0][:80]
        # One fixed reply per role.
        canned = {
            "Planner": "1. Define the topic\n2. List key facts\n3. Summarize",
            "Researcher": "- Fact A about the topic\n- Fact B about the topic",
            "Writer": f"Draft report on: {topic}\n\nFact A and Fact B, summarized.",
            # Starts with APPROVED, so the Orchestrator finishes after one review.
            "Reviewer": "APPROVED: the draft is clear and covers the plan.",
        }
        # Return the reply for this role. For an unknown role,
        # return "[role] topic" instead of crashing.
        return canned.get(role, f"[{role}] {topic}")

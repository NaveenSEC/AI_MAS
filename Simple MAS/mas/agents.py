"""Agent definitions. Each agent is a role (system prompt) plus an LLM backend.

In this project an "agent" is simple: a name plus a *system prompt*, the
instructions that tell the language model which role to play. The same model
can act as a Planner, a Researcher, a Writer or a Reviewer, depending only on
which system prompt it is given.
"""

# Lets us write type hints such as `str | None` on older Python versions too.
# It must be the first statement in the file (after the docstring).
from __future__ import annotations

# `dataclass` writes the boring parts of a class (like __init__) for us,
# based on the fields we list.
from dataclasses import dataclass

# LLM is the "shape" every backend must have: a .complete(system, prompt) method.
# We only use it as a type hint, to say "any backend works here".
from .llm import LLM


# frozen=True makes Agent objects read-only once created,
# so nobody can accidentally change an agent's role while the program runs.
@dataclass(frozen=True)
class Agent:
    # Display name, e.g. "Planner". Used in the printed output.
    name: str
    # The role instructions sent to the model as the "system" message.
    system: str

    # Ask this agent to work on a task and return its text answer.
    #   llm  -> which backend to use (Claude, Ollama or Mock)
    #   task -> the user message: the goal plus any context from earlier agents
    def run(self, llm: LLM, task: str) -> str:
        # Send our role (system prompt) and the task to the backend,
        # and hand its reply straight back to the caller.
        return llm.complete(self.system, task)


# ---------------------------------------------------------------------------
# The four agents used by the Orchestrator.
# Each is created once here and reused for every run.
# Two strings written next to each other inside ( ) are joined automatically
# by Python, which lets a long prompt span several lines.
# ---------------------------------------------------------------------------

# Step 1: turns the user's goal into a short numbered plan.
PLANNER = Agent(
    "Planner",  # name
    "You are the Planner. Break the user's goal into 3-5 short, numbered steps. "
    "Output only the steps.",  # system prompt
)

# Step 2: lists the facts and considerations needed to carry out the plan.
RESEARCHER = Agent(
    "Researcher",  # name
    "You are the Researcher. Given a goal and a plan, list the key facts and "
    "considerations needed to carry it out, as concise bullet points.",  # system prompt
)

# Step 3: writes the answer from the plan and research notes,
# and rewrites it when the Reviewer sends feedback.
WRITER = Agent(
    "Writer",  # name
    "You are the Writer. Using the plan and research notes, write a clear, "
    "well-structured answer to the goal. If reviewer feedback is included, "
    "revise the draft to address it.",  # system prompt
)

# Step 4: checks the draft. Its reply must start with the word APPROVED or REVISE,
# because the Orchestrator looks at that first word to decide what happens next.
REVIEWER = Agent(
    "Reviewer",  # name
    "You are the Reviewer. Check the draft against the goal and plan. If it is "
    "good, reply starting with 'APPROVED'. Otherwise reply starting with "
    "'REVISE' followed by specific, actionable feedback.",  # system prompt
)

"""Agent definitions. Each agent is a role (system prompt) plus an LLM backend."""

from __future__ import annotations

from dataclasses import dataclass

from .llm import LLM


@dataclass(frozen=True)
class Agent:
    name: str
    system: str

    def run(self, llm: LLM, task: str) -> str:
        return llm.complete(self.system, task)


PLANNER = Agent(
    "Planner",
    "You are the Planner. Break the user's goal into 3-5 short, numbered steps. "
    "Output only the steps.",
)
RESEARCHER = Agent(
    "Researcher",
    "You are the Researcher. Given a goal and a plan, list the key facts and "
    "considerations needed to carry it out, as concise bullet points.",
)
WRITER = Agent(
    "Writer",
    "You are the Writer. Using the plan and research notes, write a clear, "
    "well-structured answer to the goal. If reviewer feedback is included, "
    "revise the draft to address it.",
)
REVIEWER = Agent(
    "Reviewer",
    "You are the Reviewer. Check the draft against the goal and plan. If it is "
    "good, reply starting with 'APPROVED'. Otherwise reply starting with "
    "'REVISE' followed by specific, actionable feedback.",
)

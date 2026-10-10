"""Coordinates the agents: plan, research, write, then review/revise in a loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from .agents import PLANNER, RESEARCHER, REVIEWER, WRITER
from .llm import LLM


@dataclass
class Result:
    goal: str
    plan: str = ""
    research: str = ""
    draft: str = ""
    reviews: list[str] = field(default_factory=list)
    approved: bool = False


class Orchestrator:
    def __init__(self, llm: LLM, max_revisions: int = 2, verbose: bool = True) -> None:
        self.llm = llm
        self.max_revisions = max_revisions
        self.verbose = verbose

    def _log(self, agent: str, text: str) -> None:
        if self.verbose:
            print(f"\n=== {agent} ===\n{text}")

    def run(self, goal: str) -> Result:
        result = Result(goal=goal)

        result.plan = PLANNER.run(self.llm, goal)
        self._log(PLANNER.name, result.plan)

        result.research = RESEARCHER.run(self.llm, f"{goal}\n\nPlan:\n{result.plan}")
        self._log(RESEARCHER.name, result.research)

        context = f"{goal}\n\nPlan:\n{result.plan}\n\nResearch notes:\n{result.research}"
        result.draft = WRITER.run(self.llm, context)
        self._log(WRITER.name, result.draft)

        for _ in range(self.max_revisions + 1):
            review = REVIEWER.run(
                self.llm, f"{goal}\n\nPlan:\n{result.plan}\n\nDraft:\n{result.draft}"
            )
            result.reviews.append(review)
            self._log(REVIEWER.name, review)
            if review.strip().upper().startswith("APPROVED"):
                result.approved = True
                break
            if len(result.reviews) > self.max_revisions:
                break
            result.draft = WRITER.run(
                self.llm,
                f"{context}\n\nPrevious draft:\n{result.draft}\n\nReviewer feedback:\n{review}",
            )
            self._log(WRITER.name + " (revision)", result.draft)

        return result

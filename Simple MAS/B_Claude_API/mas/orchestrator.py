"""Coordinates the agents: plan, research, write, then review/revise in a loop.

The Orchestrator is the "manager" of the team. It decides who works when and
passes each agent's output on to the next one:

    goal -> Planner -> Researcher -> Writer -> Reviewer --APPROVED--> done
                                       ^          |
                                       +--REVISE--+   (at most max_revisions times)
"""

# Lets us write type hints such as `list[str]` on older Python versions too.
from __future__ import annotations

# dataclass: writes __init__ and friends for us.
# field: lets a dataclass field have a default value that is a fresh list per object.
from dataclasses import dataclass, field

# The four agents, defined in agents.py.
from .agents import PLANNER, RESEARCHER, REVIEWER, WRITER

# The backend "shape", used only as a type hint.
from .llm import LLM


# Holds everything produced during one run, so the caller can inspect it afterwards.
@dataclass
class Result:
    # The user's original goal.
    goal: str
    # The Planner's numbered steps ("" until the Planner has run).
    plan: str = ""
    # The Researcher's notes.
    research: str = ""
    # The Writer's latest draft. This becomes the final answer.
    draft: str = ""
    # Every Reviewer reply, in order. default_factory=list gives each Result
    # its own new empty list (a plain `= []` would be shared between objects).
    reviews: list[str] = field(default_factory=list)
    # True once the Reviewer says APPROVED.
    approved: bool = False


class Orchestrator:
    # Runs once, when you write Orchestrator(...).
    #   llm           -> backend all agents use (Claude, Ollama or Mock)
    #   max_revisions -> how many times the Writer may rewrite after a REVISE
    #   verbose       -> print each agent's output as it happens (tests switch this off)
    def __init__(self, llm: LLM, max_revisions: int = 2, verbose: bool = True) -> None:
        # Remember the backend.
        self.llm = llm
        # Remember the revision limit.
        self.max_revisions = max_revisions
        # Remember whether to print progress.
        self.verbose = verbose

    # Print one agent's output under a heading (the leading _ marks it as internal).
    def _log(self, agent: str, text: str) -> None:
        # Only print when verbose mode is on.
        if self.verbose:
            # "\n" adds a blank line before the heading, for readability.
            print(f"\n=== {agent} ===\n{text}")

    # Run the whole pipeline for one goal and return the Result.
    def run(self, goal: str) -> Result:
        # Start an empty Result that remembers the goal.
        result = Result(goal=goal)

        # --- Step 1: Plan --------------------------------------------------
        # The Planner sees only the goal.
        result.plan = PLANNER.run(self.llm, goal)
        # Show the plan.
        self._log(PLANNER.name, result.plan)

        # --- Step 2: Research ----------------------------------------------
        # The Researcher sees the goal plus the plan.
        result.research = RESEARCHER.run(self.llm, f"{goal}\n\nPlan:\n{result.plan}")
        # Show the research notes.
        self._log(RESEARCHER.name, result.research)

        # --- Step 3: Write -------------------------------------------------
        # Combine goal, plan and research into one block of context.
        # We keep it in a variable because revisions reuse it.
        context = f"{goal}\n\nPlan:\n{result.plan}\n\nResearch notes:\n{result.research}"
        # The Writer produces the first draft.
        result.draft = WRITER.run(self.llm, context)
        # Show the draft.
        self._log(WRITER.name, result.draft)

        # --- Step 4: Review, and revise if needed --------------------------
        # Allow one review per draft: the first draft plus up to max_revisions rewrites.
        # `_` is the usual name for a loop counter we don't need.
        for _ in range(self.max_revisions + 1):
            # The Reviewer checks the current draft against the goal and the plan.
            review = REVIEWER.run(
                self.llm, f"{goal}\n\nPlan:\n{result.plan}\n\nDraft:\n{result.draft}"
            )
            # Keep every review, so the caller can see the whole history.
            result.reviews.append(review)
            # Show the review.
            self._log(REVIEWER.name, review)
            # If the reply starts with APPROVED (ignoring spaces and upper/lower case)...
            if review.strip().upper().startswith("APPROVED"):
                # ...record the approval...
                result.approved = True
                # ...and leave the loop: the work is done.
                break
            # Not approved. If we have already used up every revision, stop here
            # instead of asking the Writer for a draft nobody will review.
            if len(result.reviews) > self.max_revisions:
                break
            # Ask the Writer for a new draft, giving it the original context,
            # its previous draft and the Reviewer's feedback.
            result.draft = WRITER.run(
                self.llm,
                f"{context}\n\nPrevious draft:\n{result.draft}\n\nReviewer feedback:\n{review}",
            )
            # Show the revised draft. The loop then goes back to the Reviewer.
            self._log(WRITER.name + " (revision)", result.draft)

        # Return everything: plan, research, final draft, all reviews and the approval flag.
        return result

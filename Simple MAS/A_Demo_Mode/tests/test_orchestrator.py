"""Automated tests. Run them with `pytest` from the A_Demo_Mode folder.

pytest finds every function whose name starts with `test_` and runs it.
Each `assert` checks that something is true; if any assert fails, the test fails.
"""

# The canned-answer backend and the Orchestrator we are testing.
from mas import MockLLM, Orchestrator


# A test backend that never approves: it behaves like MockLLM,
# except that the Reviewer always asks for a revision.
class AlwaysRevise(MockLLM):
    # Replace MockLLM's complete() with our own version.
    def complete(self, system: str, prompt: str) -> str:
        # If the Reviewer is asking...
        if system.startswith("You are the Reviewer"):
            # ...always reply REVISE, so the draft is never approved.
            return "REVISE: add more detail."
        # Every other agent gets MockLLM's normal canned answer.
        return super().complete(system, prompt)


# Checks the normal path: every step runs and the first draft is approved.
def test_pipeline_approves_with_mock():
    # Run the whole pipeline with canned answers. verbose=False keeps the test output quiet.
    result = Orchestrator(MockLLM(), verbose=False).run("Explain tides.")
    # The plan, research and draft must all be non-empty.
    assert result.plan and result.research and result.draft
    # MockLLM's Reviewer says APPROVED, so the run must be approved...
    assert result.approved
    # ...after exactly one review (no revisions were needed).
    assert len(result.reviews) == 1


# Checks the revision loop stops at the limit instead of running forever.
def test_revision_loop_is_bounded():
    # Use the backend whose Reviewer never approves, and allow 2 revisions.
    result = Orchestrator(AlwaysRevise(), max_revisions=2, verbose=False).run("Explain tides.")
    # The draft must not be marked approved.
    assert not result.approved
    # 3 reviews in total: the first draft plus 2 revisions, each reviewed once.
    assert len(result.reviews) == 3

"""Automated tests. Run them with `pytest` from the B_Claude_API folder.

pytest finds every function whose name starts with `test_` and runs it.
Each `assert` checks that something is true; if any assert fails, the test fails.
These tests use fakes instead of the real API, so they need no API key and cost nothing.
"""

# SimpleNamespace: a quick way to build an object with any attributes we like,
# used here to imitate the objects the Anthropic SDK returns.
from types import SimpleNamespace

# The backend and the Orchestrator we are testing.
from mas import ClaudeLLM, Orchestrator


# A fake backend that answers instantly. The Reviewer approves only
# after it has been asked `approve_after` times.
class FakeLLM:
    # Remember how many reviews to wait for, and start a counter at zero.
    def __init__(self, approve_after: int = 1) -> None:
        self.approve_after = approve_after
        self.reviews = 0

    # Same method shape as ClaudeLLM.complete.
    def complete(self, system: str, prompt: str) -> str:
        # If the Reviewer is asking...
        if system.startswith("You are the Reviewer"):
            # ...count this review...
            self.reviews += 1
            # ...and approve once we've reached the target number of reviews.
            return "APPROVED" if self.reviews >= self.approve_after else "REVISE: more detail."
        # Every other agent gets a short placeholder reply.
        return "some text"


# Checks the normal path: every step runs and the first draft is approved.
def test_pipeline_approves_first_draft():
    # Run the pipeline with the fake backend. verbose=False keeps the test output quiet.
    result = Orchestrator(FakeLLM(), verbose=False).run("Explain tides.")
    # Approved after exactly one review.
    assert result.approved and len(result.reviews) == 1


# Checks the revision loop stops at the limit instead of running forever.
def test_revision_loop_is_bounded():
    # A Reviewer that would need 10 reviews, but only 2 revisions are allowed.
    result = Orchestrator(FakeLLM(approve_after=10), max_revisions=2, verbose=False).run("x")
    # Never approved...
    assert not result.approved
    # ...and stopped after 3 reviews: the first draft plus 2 revisions.
    assert len(result.reviews) == 3


# Checks ClaudeLLM sends the right request and keeps only the text from the reply,
# without calling the real API. `monkeypatch` is a pytest helper that
# temporarily replaces things and puts them back after the test.
def test_claude_backend_request_and_reply(monkeypatch):
    # The fake client stores the request it receives here.
    sent = {}

    # Imitates `client.beta.messages.create(...)`.
    def fake_create(**kwargs):
        # Record every argument the backend passed.
        sent.update(kwargs)
        # Return a reply shaped like the SDK's: a thinking block (to be ignored)
        # and a text block with spaces around it (to be trimmed).
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=" hello "),
            ],
        )

    # A fake client object with the same nested shape: client.beta.messages.create.
    fake_client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=fake_create)))
    # Make `anthropic.Anthropic()` inside ClaudeLLM return our fake client.
    monkeypatch.setattr("anthropic.Anthropic", lambda: fake_client)

    # Create the backend and ask it something.
    reply = ClaudeLLM(effort="low").complete("sys", "hi")
    # Only the text block is kept, with the spaces trimmed.
    assert reply == "hello"
    # The request used the default model, our system prompt and our effort level.
    assert sent["model"] == "claude-opus-5-5"
    assert sent["system"] == "sys"
    assert sent["output_config"] == {"effort": "low"}
    # The task was sent as one user message.
    assert sent["messages"] == [{"role": "user", "content": "hi"}]

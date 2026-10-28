"""Automated tests. Run them with `pytest` from the C_Local_Model folder.

pytest finds every function whose name starts with `test_` and runs it.
Each `assert` checks that something is true; if any assert fails, the test fails.
These tests use fakes instead of a real Ollama server, so they need no model download.
"""

# io: lets us build a fake "file" of bytes in memory, standing in for an HTTP response.
import io

# json: to read the request we sent and to build a fake reply.
import json

# The backend and the Orchestrator we are testing.
from mas import OllamaLLM, Orchestrator


# A fake backend that answers instantly. The Reviewer approves only
# after it has been asked `approve_after` times.
class FakeLLM:
    # Remember how many reviews to wait for, and start a counter at zero.
    def __init__(self, approve_after: int = 1) -> None:
        self.approve_after = approve_after
        self.reviews = 0

    # Same method shape as OllamaLLM.complete.
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


# Checks OllamaLLM sends the right request and reads the reply correctly,
# without a real Ollama server. `monkeypatch` is a pytest helper that
# temporarily replaces things and puts them back after the test.
def test_ollama_backend_parses_chat_response(monkeypatch):
    # An empty dict where the fake server records what it received.
    sent = {}

    # A stand-in for urllib.request.urlopen that never touches the network.
    def fake_urlopen(request, timeout):
        # Record the URL the backend called.
        sent["url"] = request.full_url
        # Record the JSON body the backend sent, turned back into a dict.
        sent["body"] = json.loads(request.data)
        # Return a reply shaped like Ollama's, with spaces around the text
        # so we can check that the backend trims them.
        return io.BytesIO(json.dumps({"message": {"content": " hello "}}).encode())

    # Swap the real urlopen for our fake one for the rest of this test.
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    # Create the backend with a host that has no "http://", to test that it adds one.
    llm = OllamaLLM(model="llama3.2", host="localhost:11434")
    # The reply must come back with the spaces trimmed.
    assert llm.complete("sys", "hi") == "hello"
    # The request must go to Ollama's chat endpoint, with "http://" added.
    assert sent["url"] == "http://localhost:11434/api/chat"
    # The request must name the chosen model.
    assert sent["body"]["model"] == "llama3.2"
    # The first message must be the system prompt we passed in.
    assert sent["body"]["messages"][0] == {"role": "system", "content": "sys"}

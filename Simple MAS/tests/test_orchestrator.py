"""Automated tests. Run them with `pytest` from the Simple MAS folder.

pytest finds every function whose name starts with `test_` and runs it.
Each `assert` checks that something is true; if any assert fails, the test fails.
None of these tests need an API key, an internet connection or Ollama.
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


# Checks OllamaLLM sends the right request and reads the reply correctly,
# without a real Ollama server. `monkeypatch` is a pytest helper that
# temporarily replaces things and puts them back after the test.
def test_ollama_backend_parses_chat_response(monkeypatch):
    # io: lets us build a fake "file" of bytes in memory, standing in for an HTTP response.
    import io

    # json: to read the request we sent and to build a fake reply.
    import json

    # The backend under test.
    from mas import OllamaLLM

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

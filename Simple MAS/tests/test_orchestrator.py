from mas import MockLLM, Orchestrator


class AlwaysRevise(MockLLM):
    def complete(self, system: str, prompt: str) -> str:
        if system.startswith("You are the Reviewer"):
            return "REVISE: add more detail."
        return super().complete(system, prompt)


def test_pipeline_approves_with_mock():
    result = Orchestrator(MockLLM(), verbose=False).run("Explain tides.")
    assert result.plan and result.research and result.draft
    assert result.approved
    assert len(result.reviews) == 1


def test_revision_loop_is_bounded():
    result = Orchestrator(AlwaysRevise(), max_revisions=2, verbose=False).run("Explain tides.")
    assert not result.approved
    assert len(result.reviews) == 3


def test_ollama_backend_parses_chat_response(monkeypatch):
    import io
    import json

    from mas import OllamaLLM

    sent = {}

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        return io.BytesIO(json.dumps({"message": {"content": " hello "}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OllamaLLM(model="llama3.2", host="localhost:11434")
    assert llm.complete("sys", "hi") == "hello"
    assert sent["url"] == "http://localhost:11434/api/chat"
    assert sent["body"]["model"] == "llama3.2"
    assert sent["body"]["messages"][0] == {"role": "system", "content": "sys"}

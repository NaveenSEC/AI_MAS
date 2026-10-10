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

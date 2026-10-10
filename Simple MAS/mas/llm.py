"""LLM backends. ClaudeLLM calls the Anthropic API; MockLLM runs offline."""

from __future__ import annotations

from typing import Protocol

MODEL = "claude-opus-5-5"


class LLM(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class ClaudeLLM:
    """Calls Claude via the official Anthropic SDK."""

    def __init__(self, model: str = MODEL, effort: str = "medium") -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def complete(self, system: str, prompt: str) -> str:
        # Server-side fallback re-runs a declined request on a recommended
        # model instead of returning a refusal.
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined this request.")
        return "".join(b.text for b in response.content if b.type == "text").strip()


class MockLLM:
    """Deterministic offline backend so the pipeline runs without an API key."""

    def complete(self, system: str, prompt: str) -> str:
        role = system.split(".")[0].removeprefix("You are the ").strip()
        topic = prompt.splitlines()[0][:80]
        canned = {
            "Planner": "1. Define the topic\n2. List key facts\n3. Summarize",
            "Researcher": "- Fact A about the topic\n- Fact B about the topic",
            "Writer": f"Draft report on: {topic}\n\nFact A and Fact B, summarized.",
            "Reviewer": "APPROVED: the draft is clear and covers the plan.",
        }
        return canned.get(role, f"[{role}] {topic}")

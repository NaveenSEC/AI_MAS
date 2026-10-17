"""LLM backends. ClaudeLLM calls the Anthropic API, OllamaLLM runs an
open-weights model on your machine, and MockLLM runs offline with canned answers.

A "backend" is whatever turns a prompt into text. All three classes below have
the same method, `complete(system, prompt) -> str`, so the agents and the
Orchestrator never need to know which one is in use. You choose it in main.py
with the --mock / --local flags.
"""

# Lets us write type hints such as `str | None` on older Python versions too.
from __future__ import annotations

# json: converts Python dicts to JSON text and back (Ollama speaks JSON over HTTP).
import json

# os: lets us read environment variables, such as OLLAMA_HOST.
import os

# urllib.request: Python's built-in HTTP client, so the Ollama backend
# needs no extra package to install.
import urllib.request

# Protocol: describes a required "shape" (which methods a class must have)
# without forcing classes to inherit from anything.
from typing import Protocol

# The Claude model used by default. Change this to use a different Claude model.
MODEL = "claude-opus-5-5"

# The Ollama model used by default with --local (about a 2 GB download).
LOCAL_MODEL = "llama3.2"


# Every backend must look like this: a `complete` method that takes the
# system prompt (the agent's role) and the user prompt (the task),
# and returns the model's reply as a string.
class LLM(Protocol):
    # The "..." means "no body here": this only describes the method's signature.
    def complete(self, system: str, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# Backend 1: Claude through the Anthropic API
# ---------------------------------------------------------------------------
class ClaudeLLM:
    """Calls Claude via the official Anthropic SDK."""

    # Runs once, when you write ClaudeLLM().
    #   model  -> which Claude model to call (default: MODEL above)
    #   effort -> how hard the model thinks: "low", "medium", "high", "xhigh" or "max"
    def __init__(self, model: str = MODEL, effort: str = "medium") -> None:
        # Import the Anthropic SDK here rather than at the top of the file,
        # so --mock and --local still work when the package isn't installed.
        import anthropic

        # Keep a reference to the SDK module on the object (for later use if needed).
        self._anthropic = anthropic
        # Create the API client. It finds your credentials by itself,
        # for example from the ANTHROPIC_API_KEY environment variable.
        self.client = anthropic.Anthropic()
        # Remember the chosen model for every request.
        self.model = model
        # Remember the chosen effort level for every request.
        self.effort = effort

    # Send one request to Claude and return the text of its reply.
    def complete(self, system: str, prompt: str) -> str:
        # Server-side fallback re-runs a declined request on a recommended
        # model instead of returning a refusal.
        response = self.client.beta.messages.create(  # beta endpoint: needed for `fallbacks`
            model=self.model,  # which Claude model answers
            max_tokens=16000,  # upper limit on the reply length (in tokens, roughly ¾ of a word each)
            system=system,  # the agent's role instructions
            thinking={"type": "adaptive"},  # let Claude decide how much to think before answering
            output_config={"effort": self.effort},  # how much effort to spend (see __init__)
            betas=["server-side-fallback-2026-07-01"],  # switch on the fallback feature
            fallbacks="default",  # if Claude declines, retry on Anthropic's recommended model
            messages=[{"role": "user", "content": prompt}],  # the task, as one user message
        )
        # "refusal" means every model in the fallback chain declined the request.
        if response.stop_reason == "refusal":
            # Stop the program with a clear message instead of returning an empty answer.
            raise RuntimeError("The model declined this request.")
        # A reply is a list of content blocks (text, thinking, ...).
        # Keep only the text blocks, join them into one string,
        # and trim spaces and newlines from both ends.
        return "".join(b.text for b in response.content if b.type == "text").strip()


# ---------------------------------------------------------------------------
# Backend 2: a local open-weights model through Ollama
# ---------------------------------------------------------------------------
class OllamaLLM:
    """Calls a model running locally in Ollama (https://ollama.com)."""

    # Runs once, when you write OllamaLLM().
    #   model -> name of a model you have downloaded with `ollama pull`
    #   host  -> where Ollama is listening (None means "work it out", see below)
    def __init__(self, model: str = LOCAL_MODEL, host: str | None = None) -> None:
        # Remember which local model to use.
        self.model = model
        # Pick the address of the Ollama server, in this order:
        #   1. the `host` argument, if one was given
        #   2. the OLLAMA_HOST environment variable, if it is set
        #   3. Ollama's standard local address
        # .rstrip("/") removes a trailing slash so the URL we build later has no "//".
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        # OLLAMA_HOST is often written without "http://", e.g. "localhost:11434".
        if not self.host.startswith("http"):
            # Add the missing "http://" so it is a valid URL.
            self.host = f"http://{self.host}"

    # Send one chat request to Ollama and return the text of its reply.
    def complete(self, system: str, prompt: str) -> str:
        # Build the request body as JSON, in the format Ollama's /api/chat expects.
        body = json.dumps(
            {
                "model": self.model,  # which downloaded model to run
                "stream": False,  # return the whole reply at once, not word by word
                "messages": [
                    {"role": "system", "content": system},  # the agent's role
                    {"role": "user", "content": prompt},  # the task
                ],
            }
        ).encode()  # .encode() turns the JSON text into bytes, which HTTP sends
        # Prepare an HTTP POST request (giving `data` makes it a POST) to Ollama's chat endpoint.
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        # Network calls can fail, so be ready to catch the error.
        try:
            # Send the request. Allow up to 600 seconds: local models can be slow.
            # `with` makes sure the connection is closed afterwards.
            with urllib.request.urlopen(request, timeout=600) as response:
                # Read the reply and convert the JSON into a Python dict.
                data = json.load(response)
        # OSError covers "connection refused", timeouts and HTTP errors
        # (for example 404 when the model hasn't been downloaded).
        except OSError as exc:
            # Replace the low-level error with one that says how to fix it.
            # `from exc` keeps the original error attached, for debugging.
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"Start it with `ollama serve` and pull the model with `ollama pull {self.model}`."
            ) from exc
        # Ollama replies with {"message": {"role": "assistant", "content": "..."}, ...}.
        # Take the content text and trim spaces and newlines from both ends.
        return data["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Backend 3: canned answers (no model at all)
# ---------------------------------------------------------------------------
class MockLLM:
    """Deterministic offline backend so the pipeline runs without an API key."""

    # Return a fixed answer based on which agent is asking.
    def complete(self, system: str, prompt: str) -> str:
        # Work out the agent's role from its system prompt:
        # "You are the Planner. Break ..." -> split at "." -> "You are the Planner"
        # -> remove "You are the " -> "Planner".
        role = system.split(".")[0].removeprefix("You are the ").strip()
        # Use the first line of the task (at most 80 characters) as the topic.
        topic = prompt.splitlines()[0][:80]
        # One fixed reply per role.
        canned = {
            "Planner": "1. Define the topic\n2. List key facts\n3. Summarize",
            "Researcher": "- Fact A about the topic\n- Fact B about the topic",
            "Writer": f"Draft report on: {topic}\n\nFact A and Fact B, summarized.",
            # Starts with APPROVED, so the Orchestrator finishes after one review.
            "Reviewer": "APPROVED: the draft is clear and covers the plan.",
        }
        # Return the reply for this role. For an unknown role,
        # return "[role] topic" instead of crashing.
        return canned.get(role, f"[{role}] {topic}")

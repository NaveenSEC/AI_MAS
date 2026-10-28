"""The LLM backend for this project: an open-weights model running on your computer.

A "backend" is whatever turns a prompt into text. Here it is OllamaLLM, which
sends each agent's request to Ollama (https://ollama.com), a free program that
runs open-weights models such as Llama locally. No API key, no cost, and it
works offline once the model is downloaded.
"""

# Lets us write type hints such as `str | None` on older Python versions too.
from __future__ import annotations

# json: converts Python dicts to JSON text and back (Ollama speaks JSON over HTTP).
import json

# os: lets us read environment variables, such as OLLAMA_HOST.
import os

# urllib.request: Python's built-in HTTP client, so this project needs no extra package.
import urllib.request

# Protocol: describes a required "shape" (which methods a class must have)
# without forcing classes to inherit from anything.
from typing import Protocol

# The Ollama model used by default (about a 2 GB download).
LOCAL_MODEL = "llama3.2"


# Every backend must look like this: a `complete` method that takes the
# system prompt (the agent's role) and the user prompt (the task),
# and returns the model's reply as a string.
class LLM(Protocol):
    # The "..." means "no body here": this only describes the method's signature.
    def complete(self, system: str, prompt: str) -> str: ...


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

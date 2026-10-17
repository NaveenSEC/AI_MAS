"""A simple multi-agent system: planner -> researcher -> writer -> reviewer.

This file turns the `mas` folder into a Python *package*. Python runs it the
first time anyone writes `import mas` or `from mas import ...`. Its only job is
to gather the important names from the other files in this folder, so users can
write the short `from mas import Orchestrator` instead of the longer
`from mas.orchestrator import Orchestrator`.
"""

# Bring in the Agent class and the four ready-made agents from agents.py.
# The leading dot in ".agents" means "the agents.py file in this same folder".
from .agents import Agent, PLANNER, RESEARCHER, WRITER, REVIEWER

# Bring in the three LLM "backends" (the things that actually produce text) from llm.py:
#   ClaudeLLM -> Claude through the Anthropic API (needs an API key, costs money)
#   MockLLM   -> canned answers, for demos and tests (free, no internet)
#   OllamaLLM -> an open-weights model running on your own computer (free, offline)
from .llm import ClaudeLLM, MockLLM, OllamaLLM

# Bring in the Orchestrator, the "manager" that runs the agents in order.
from .orchestrator import Orchestrator

# __all__ lists the names that `from mas import *` hands out.
# It also documents which names are this package's public API.
__all__ = [
    "Agent",  # the class used to define an agent
    "PLANNER",  # agent that breaks the goal into steps
    "RESEARCHER",  # agent that lists key facts for the plan
    "WRITER",  # agent that writes (and revises) the answer
    "REVIEWER",  # agent that approves or asks for a revision
    "ClaudeLLM",  # backend: Claude via the Anthropic API
    "MockLLM",  # backend: canned answers
    "OllamaLLM",  # backend: local model via Ollama
    "Orchestrator",  # runs the whole plan -> research -> write -> review pipeline
]

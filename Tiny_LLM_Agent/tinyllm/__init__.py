"""The `tinyllm` package: a miniature GPT-style language model and an agent that uses it.

This file turns the `tinyllm` folder into a Python package. The modules are:
  tokenizer.py -> words <-> token ids (19-word vocabulary)
  data.py      -> the 16 training sentences (CAN observation -> verdict)
  model.py     -> TinyGPT: embedding, 2 transformer blocks, output layer, backprop, generation
  train.py     -> training loop and the train / held-back split
  agent.py     -> SecurityAgent: perceive -> think (LLM) -> act
"""

# Make the main classes importable straight from the package,
# e.g. `from tinyllm import TinyGPT`.
from .agent import SecurityAgent  # the agent
from .model import TinyGPT  # the language model

# Names handed out by `from tinyllm import *`, and the package's public API.
__all__ = ["SecurityAgent", "TinyGPT"]

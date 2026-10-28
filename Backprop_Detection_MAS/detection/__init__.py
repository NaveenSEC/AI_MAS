"""The `detection` package: 20 backprop-trained agents that detect CAN-bus attacks.

This file turns the `detection` folder into a Python package. The modules are:
  data.py        -> simulated CAN traffic (normal + DoS, fuzzing, spoofing)
  network.py     -> TinyNet, a neural network with backpropagation written out by hand
  agent.py       -> DetectorAgent, one network watching a few features
  coordinator.py -> Coordinator, which trains the 20 agents and combines their votes
"""

# Make the main classes importable straight from the package,
# e.g. `from detection import Coordinator`.
from .agent import DetectorAgent  # one detector agent
from .coordinator import Coordinator  # manages and combines the agents
from .network import TinyNet  # the neural network each agent owns

# Names handed out by `from detection import *`, and the package's public API.
__all__ = ["Coordinator", "DetectorAgent", "TinyNet"]

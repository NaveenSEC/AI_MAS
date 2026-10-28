"""The coordinator: creates the 20 agents, trains them, and combines their votes.

How a decision is made for one window of traffic:
  1. Every agent looks at its own few features and outputs P(attack).
  2. The coordinator averages the 20 probabilities ("soft voting").
  3. If the average is 0.5 or more, the window is flagged as an attack.
Averaging cancels out individual agents' mistakes, and agents that watch
the right features for an attack pull the average towards the right answer.
"""

# Lets us write modern type hints on older Python versions too.
from __future__ import annotations

# numpy: arrays and maths.
import numpy as np

# The detector agent class.
from .agent import DetectorAgent


class Coordinator:
    # Create the team of agents.
    #   n_agents   -> how many agents (20 in this example)
    #   n_features -> how many features each agent sees
    #   seed       -> makes the whole team repeatable
    def __init__(self, n_agents: int = 20, n_features: int = 3, seed: int = 0) -> None:
        # Build the agents, numbered 1..n_agents, each with a different seed
        # so each picks different features and different training rows.
        self.agents = [
            DetectorAgent(agent_id=i + 1, n_features=n_features, seed=seed + i)
            for i in range(n_agents)
        ]

    # Train every agent on the same training set (each takes its own resample of it).
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.5) -> None:
        # Agents are independent, so they could train in parallel;
        # here we simply train them one after another.
        for agent in self.agents:
            # Run backpropagation for this agent.
            agent.train(x, y, epochs=epochs, lr=lr)

    # Every agent's probability for every row: shape (n_agents, n_rows).
    def all_probabilities(self, x: np.ndarray) -> np.ndarray:
        # Ask each agent and stack the answers into one array, one row per agent.
        return np.vstack([agent.predict_proba(x) for agent in self.agents])

    # The team's combined probability of attack for each row.
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Average down the agents axis (soft voting).
        return self.all_probabilities(x).mean(axis=0)

    # The team's final yes/no verdict for each row.
    def detect(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        # Raise an alarm wherever the averaged probability crosses the threshold.
        return self.predict_proba(x) >= threshold

    # How many agents raised an alarm for each row (useful for explaining a decision).
    def alarm_counts(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        # Count, per row, the agents whose own probability crossed the threshold.
        return (self.all_probabilities(x) >= threshold).sum(axis=0)

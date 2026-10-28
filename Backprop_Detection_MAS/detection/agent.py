"""One detector agent: a small neural network that watches a few features.

Every agent is deliberately limited and different from the others:
  * it sees only a few of the 8 features (its own "sensors"), and
  * it trains on its own random resample of the training data.
So each agent learns a different, partial view of the bus. One agent alone
makes mistakes; the coordinator combines all 20 to make a better decision.
"""

# Lets us write modern type hints on older Python versions too.
from __future__ import annotations

# numpy: arrays and random numbers.
import numpy as np

# The feature names (for printing what each agent watches).
from .data import FEATURES

# The neural network each agent owns.
from .network import TinyNet


class DetectorAgent:
    # Create an agent.
    #   agent_id   -> its number, 1 to 20
    #   n_features -> how many of the 8 features it may look at
    #   n_hidden   -> size of its network's hidden layer
    #   seed       -> makes this agent's random choices repeatable
    def __init__(self, agent_id: int, n_features: int = 3, n_hidden: int = 8, seed: int = 0) -> None:
        # Remember the agent's number.
        self.agent_id = agent_id
        # This agent's own random number generator.
        self.rng = np.random.default_rng(seed)
        # Pick which feature columns this agent sees: n_features different ones,
        # chosen at random, sorted so they print neatly.
        self.features = np.sort(self.rng.choice(len(FEATURES), size=n_features, replace=False))
        # The agent's own network, with one input per chosen feature.
        self.net = TinyNet(n_inputs=n_features, n_hidden=n_hidden, seed=seed)
        # Average and spread of each chosen feature, filled in during training.
        self.mean = np.zeros(n_features)
        self.std = np.ones(n_features)
        # Loss after each epoch, so we can show learning progress.
        self.history: list[float] = []

    # Names of the features this agent watches, e.g. ["jitter_ms", "bus_load_pct"].
    @property
    def feature_names(self) -> list[str]:
        # Turn each column number into its name.
        return [FEATURES[i] for i in self.features]

    # Keep only this agent's features and rescale them to mean 0, spread 1.
    # Networks learn much faster when every input is on a similar scale
    # (otherwise msg_rate ≈ 50 would swamp jitter_ms ≈ 0.02).
    def _prepare(self, x: np.ndarray) -> np.ndarray:
        # Select the agent's columns, subtract the mean and divide by the spread.
        return (x[:, self.features] - self.mean) / self.std

    # Learn from labelled data using backpropagation.
    #   x, y   -> training features and labels (1 = attack, 0 = normal)
    #   epochs -> how many forward/backward/update rounds to run
    #   lr     -> learning rate: how big each update step is
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.5) -> None:
        # Bootstrap resample: draw as many rows as there are, with replacement.
        # Some rows appear twice, others not at all, so every agent's data differs.
        rows = self.rng.integers(0, len(y), size=len(y))
        # Keep the resampled rows.
        x, y = x[rows], y[rows]
        # Measure the average of this agent's features on its own training data.
        self.mean = x[:, self.features].mean(axis=0)
        # Measure the spread too (+1e-9 avoids dividing by zero).
        self.std = x[:, self.features].std(axis=0) + 1e-9
        # Rescale the inputs once, up front.
        xs = self._prepare(x)
        # Repeat the learning step `epochs` times.
        for _ in range(epochs):
            # One forward pass, loss, backpropagation and update; keep the loss.
            self.history.append(self.net.train_step(xs, y, lr))

    # Probability of attack for each row of x, as judged by this agent alone.
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Rescale with the training statistics, then run the network forwards.
        return self.net.predict_proba(self._prepare(x))

    # This agent's yes/no verdict for each row: True means "attack".
    def detect(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        # Raise an alarm wherever the probability crosses the threshold.
        return self.predict_proba(x) >= threshold

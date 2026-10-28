"""Automated tests. Run them with `pytest` from the Backprop_Detection_MAS folder.

pytest finds every function whose name starts with `test_` and runs it.
Each `assert` checks that something is true; if any assert fails, the test fails.
"""

# numpy: arrays and maths.
import numpy as np

# The classes and helpers under test.
from detection.coordinator import Coordinator
from detection.data import make_dataset, train_test_split
from detection.network import TinyNet, gradient_check


# Build one small data set, shared by the tests below.
def _data():
    # Fewer rows than the main program, so the tests run quickly.
    x, y, kinds = make_dataset(n_normal=600, n_per_attack=200, seed=1)
    # Split into training and test parts.
    return train_test_split(x, y, kinds)


# Backprop's gradients must match slow numerical gradients.
def test_backprop_gradients_match_numerical():
    # Random inputs: 15 rows, 4 features.
    x = np.random.default_rng(0).normal(size=(15, 4))
    # Random 0/1 labels.
    y = (np.random.default_rng(1).random(15) > 0.5).astype(float)
    # A small network with random weights.
    net = TinyNet(n_inputs=4, n_hidden=5, seed=2)
    # The biggest disagreement must be tiny.
    assert gradient_check(net, x, y) < 1e-7


# Training must make the loss go down.
def test_training_reduces_loss():
    # Get training data.
    x_train, y_train, *_ = _data()
    # A one-agent team is enough to watch learning happen.
    team = Coordinator(n_agents=1, n_features=3, seed=0)
    # Train it.
    team.train(x_train, y_train, epochs=100)
    # The agent's recorded losses, epoch by epoch.
    history = team.agents[0].history
    # The final loss must be clearly lower than the first one.
    assert history[-1] < history[0] * 0.8


# There must be 20 agents, and they must not all watch the same features.
def test_twenty_agents_with_different_views():
    # Build the default team.
    team = Coordinator()
    # Exactly 20 agents.
    assert len(team.agents) == 20
    # Collect each agent's feature set as a tuple.
    views = {tuple(agent.features) for agent in team.agents}
    # Several different feature sets.
    assert len(views) > 5


# The combined team must beat the average single agent.
def test_team_beats_average_agent():
    # Get training and test data.
    x_train, y_train, x_test, y_test, _ = _data()
    # Build and train the full team.
    team = Coordinator(n_agents=20, seed=0)
    team.train(x_train, y_train, epochs=200)
    # Accuracy of each agent alone.
    solo = [np.mean(a.detect(x_test) == (y_test == 1)) for a in team.agents]
    # Accuracy of the team.
    combined = np.mean(team.detect(x_test) == (y_test == 1))
    # The team must do better than the average agent.
    assert combined > np.mean(solo)

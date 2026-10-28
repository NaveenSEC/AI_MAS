"""Run the 20-agent backpropagation detector end to end: python main.py

What it prints, in order:
  1. The data set (simulated CAN-bus traffic).
  2. A gradient check proving the backpropagation maths is correct.
  3. One agent learning: its loss falling as backpropagation runs.
  4. Each of the 20 agents' accuracy on its own.
  5. The coordinator's combined accuracy, per attack type.
  6. One example window, showing which agents raised an alarm.
"""

# argparse: Python's built-in tool for reading command-line options.
import argparse

# numpy: arrays and maths.
import numpy as np

# The coordinator that manages the 20 agents.
from detection.coordinator import Coordinator

# The data generator, the train/test splitter and the feature names.
from detection.data import FEATURES, make_dataset, train_test_split

# The network class and the backprop correctness check.
from detection.network import TinyNet, gradient_check


# Share of predictions that match the true labels (0.0 to 1.0).
def accuracy(predicted: np.ndarray, y: np.ndarray) -> float:
    # Compare each prediction (True/False) with its label (1/0) and take the average.
    return float(np.mean(predicted == (y == 1)))


# All the program's work happens inside this function.
def main() -> None:
    # Create the option reader. `description` appears at the top of `python main.py --help`.
    parser = argparse.ArgumentParser(description="20 backprop-trained agents detecting CAN attacks.")
    # How many agents in the team.
    parser.add_argument("--agents", type=int, default=20, help="number of detector agents")
    # How many of the 8 features each agent sees.
    parser.add_argument("--features", type=int, default=3, help="features per agent (1-8)")
    # How many training rounds each agent runs.
    parser.add_argument("--epochs", type=int, default=300, help="backprop iterations per agent")
    # How big each gradient-descent step is.
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
    # Seed for all randomness, so results are repeatable.
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    # Read the actual command line into `args`.
    args = parser.parse_args()

    # ---- 1. Data ------------------------------------------------------------
    # Generate labelled traffic windows (normal + 3 attack types).
    x, y, kinds = make_dataset(seed=args.seed)
    # Split into training data (to learn from) and test data (to measure on, never trained on).
    x_train, y_train, x_test, y_test, kinds_test = train_test_split(x, y, kinds)
    # Report the sizes.
    print(f"1. DATA: {len(y_train)} training windows, {len(y_test)} test windows, "
          f"{len(FEATURES)} features each")

    # ---- 2. Gradient check --------------------------------------------------
    # A small throwaway network: 3 inputs, 4 hidden neurons.
    check_net = TinyNet(n_inputs=3, n_hidden=4, seed=args.seed)
    # Compare backprop's gradients with numerical ones on 20 standardised rows.
    sample = (x_train[:20, :3] - x_train[:, :3].mean(axis=0)) / x_train[:, :3].std(axis=0)
    # The largest difference found between the two methods.
    diff = gradient_check(check_net, sample, y_train[:20])
    # Show it; a tiny number means backpropagation is computed correctly.
    print(f"\n2. GRADIENT CHECK: largest difference between backprop and numerical "
          f"gradients = {diff:.1e} ({'OK' if diff < 1e-6 else 'MISMATCH'})")

    # ---- 3. Training --------------------------------------------------------
    # Create the team of agents.
    team = Coordinator(n_agents=args.agents, n_features=args.features, seed=args.seed)
    # Train every agent with backpropagation.
    team.train(x_train, y_train, epochs=args.epochs, lr=args.lr)
    # Take the first agent as the example to watch.
    first = team.agents[0]
    # Show what it watches.
    print(f"\n3. BACKPROPAGATION IN ACTION: agent 1 watches {', '.join(first.feature_names)}")
    # Pick about six evenly spaced epochs to print (always including the last).
    checkpoints = sorted({0, *range(0, args.epochs, max(1, args.epochs // 5)), args.epochs - 1})
    # Print the loss at each checkpoint; it should fall as the agent learns.
    for epoch in checkpoints:
        # One line per checkpoint: epoch number and loss.
        print(f"   epoch {epoch + 1:>4}: loss {first.history[epoch]:.4f}")

    # ---- 4. Each agent on its own --------------------------------------------
    # Header for the per-agent table.
    print("\n4. EACH AGENT ALONE (test accuracy):")
    # One line per agent.
    for agent in team.agents:
        # Measure this agent alone on the test data.
        acc = accuracy(agent.detect(x_test), y_test)
        # Agent number, accuracy and the features it watches.
        print(f"   agent {agent.agent_id:>2}: {acc:6.1%}  watches {', '.join(agent.feature_names)}")
    # The individual accuracies as a list, for the summary line.
    solo = [accuracy(a.detect(x_test), y_test) for a in team.agents]
    # Summarise: worst, average and best single agent.
    print(f"   -> single agents range {min(solo):.1%} to {max(solo):.1%} (average {np.mean(solo):.1%})")

    # ---- 5. The team together ----------------------------------------------
    # The coordinator's combined verdict for every test window.
    verdict = team.detect(x_test)
    # Headline accuracy of the team.
    print(f"\n5. ALL {len(team.agents)} AGENTS COMBINED: test accuracy {accuracy(verdict, y_test):.1%}")
    # Break it down by traffic type.
    for kind in ["normal", "dos", "fuzzing", "spoofing"]:
        # Rows of this type.
        mask = kinds_test == kind
        # Share of those rows flagged as attacks.
        flagged = float(np.mean(verdict[mask]))
        # For normal traffic a flag is a false alarm; for attacks it is a detection.
        label = "false-alarm rate" if kind == "normal" else "detection rate"
        # Print one line per type.
        print(f"   {kind:<9} {label}: {flagged:6.1%}")

    # ---- 6. Explain one decision --------------------------------------------
    # Pick the first spoofing window in the test set (the hardest attack).
    i = int(np.flatnonzero(kinds_test == "spoofing")[0])
    # Every agent's probability for that window.
    probs = team.all_probabilities(x_test[i : i + 1]).ravel()
    # Numbers of the agents that raised an alarm.
    alarmed = [a.agent_id for a, p in zip(team.agents, probs) if p >= 0.5]
    # Summarise the vote.
    print(f"\n6. ONE SPOOFING WINDOW: {len(alarmed)} of {len(team.agents)} agents raised an alarm "
          f"(agents {alarmed}); average P(attack) = {probs.mean():.2f} -> "
          f"{'ATTACK' if probs.mean() >= 0.5 else 'normal'}")


# This is True only when the file is run directly (python main.py),
# not when another file imports it. That stops main() from running on import.
if __name__ == "__main__":
    # Start the program.
    main()

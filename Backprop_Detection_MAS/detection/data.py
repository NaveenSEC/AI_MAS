"""Synthetic CAN-bus traffic: normal windows plus three kinds of attack.

Real intrusion detection would read frames from a vehicle's CAN bus. To keep
the example self-contained, we *simulate* traffic instead. Each sample is one
short time window of bus traffic, summarised as 8 numbers (features). Normal
windows and attack windows have different typical values, with some overlap,
so the detectors have to learn the difference rather than read it off.
"""

# Lets us write modern type hints on older Python versions too.
from __future__ import annotations

# numpy: fast arrays and random numbers. The whole project uses only numpy.
import numpy as np

# The 8 features measured for every time window, in column order.
FEATURES = [
    "msg_rate",  # messages seen in the window
    "inter_arrival_ms",  # average gap between messages, in milliseconds
    "jitter_ms",  # how much those gaps vary
    "payload_entropy",  # randomness of the data bytes (0 = constant, 8 = random)
    "unknown_id_ratio",  # share of messages whose CAN ID is not in the vehicle's normal list
    "dlc_mismatch_ratio",  # share of messages whose length (DLC) is wrong for their ID
    "payload_change_rate",  # how often a signal's value changes between messages
    "bus_load_pct",  # how busy the bus is, in percent
]

# Typical value (mean) and spread (standard deviation) of each feature for
# normal traffic. Each row lines up with FEATURES above.
NORMAL = np.array(
    [
        [50.0, 5.0],  # msg_rate
        [0.20, 0.03],  # inter_arrival_ms
        [0.020, 0.008],  # jitter_ms
        [3.0, 0.5],  # payload_entropy
        [0.01, 0.01],  # unknown_id_ratio
        [0.00, 0.01],  # dlc_mismatch_ratio
        [0.30, 0.07],  # payload_change_rate
        [35.0, 5.0],  # bus_load_pct
    ]
)

# How each attack changes the normal traffic: {feature name: (new mean, new spread)}.
# Features not listed keep their normal values.
ATTACKS = {
    # Denial of service: the attacker floods the bus with high-priority messages.
    "dos": {
        "msg_rate": (110.0, 20.0),  # far more messages...
        "inter_arrival_ms": (0.10, 0.03),  # ...arriving closer together...
        "payload_entropy": (1.5, 0.8),  # ...with repetitive data...
        "bus_load_pct": (70.0, 12.0),  # ...filling the bus
    },
    # Fuzzing: the attacker sends random IDs and random data to probe ECUs.
    "fuzzing": {
        "payload_entropy": (5.5, 1.0),  # random-looking data
        "unknown_id_ratio": (0.20, 0.10),  # many IDs the vehicle never uses
        "dlc_mismatch_ratio": (0.10, 0.06),  # wrong message lengths
    },
    # Spoofing: the attacker impersonates one real ECU. Subtle: the IDs and
    # rates look normal; only the timing and the signal values give it away.
    "spoofing": {
        "jitter_ms": (0.045, 0.015),  # two senders on one ID make timing irregular
        "payload_change_rate": (0.55, 0.12),  # the signal value jumps between two sources
    },
}


# Draw `n` windows of traffic of one kind ("normal", "dos", "fuzzing" or "spoofing").
def _sample(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    # Start from a copy of the normal means and spreads (copy so we don't change NORMAL).
    params = NORMAL.copy()
    # For an attack, overwrite the features that attack changes.
    for name, (mean, std) in ATTACKS.get(kind, {}).items():
        # Find the column of this feature and put in the attack's values.
        params[FEATURES.index(name)] = (mean, std)
    # Draw random values: one row per window, one column per feature,
    # each from a bell curve (normal distribution) with that column's mean and spread.
    x = rng.normal(params[:, 0], params[:, 1], size=(n, len(FEATURES)))
    # Ratios, rates and percentages can't be negative, so clip them at zero.
    return np.clip(x, 0.0, None)


# Build a labelled data set: features X, labels y (1 = attack, 0 = normal),
# and the kind of each window (for per-attack reporting).
def make_dataset(
    n_normal: int = 1500, n_per_attack: int = 500, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A random number generator with a fixed seed, so every run gives the same data.
    rng = np.random.default_rng(seed)
    # Collect the windows of each kind in these lists.
    xs, ys, kinds = [], [], []
    # One block of normal traffic plus one block per attack type.
    for kind, n in [("normal", n_normal)] + [(k, n_per_attack) for k in ATTACKS]:
        # Generate n windows of this kind.
        xs.append(_sample(kind, n, rng))
        # Label them: 0 for normal, 1 for any attack.
        ys.append(np.full(n, 0.0 if kind == "normal" else 1.0))
        # Remember the kind name for each window.
        kinds.append(np.full(n, kind))
    # Stack the blocks into single arrays.
    x, y, k = np.vstack(xs), np.concatenate(ys), np.concatenate(kinds)
    # Shuffle the rows, so normal and attack windows are mixed together.
    order = rng.permutation(len(y))
    # Return the shuffled arrays.
    return x[order], y[order], k[order]


# Split arrays into a training part and a test part.
def train_test_split(
    x: np.ndarray, y: np.ndarray, kinds: np.ndarray, test_fraction: float = 0.3
) -> tuple[np.ndarray, ...]:
    # Number of rows that go into the test set.
    n_test = int(len(y) * test_fraction)
    # The data is already shuffled, so the first rows can be the test set
    # and the rest the training set.
    return x[n_test:], y[n_test:], x[:n_test], y[:n_test], kinds[:n_test]

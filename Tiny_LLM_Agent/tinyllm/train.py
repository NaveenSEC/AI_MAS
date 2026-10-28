"""Training: show the model the example sentences and let backpropagation adjust the weights."""

# Lets us write modern type hints on older Python versions too.
from __future__ import annotations

# numpy: arrays and maths.
import numpy as np

# The answer positions to grade, and the training sentences.
from .data import ANSWER_POSITIONS, corpus

# The model class.
from .model import TinyGPT

# The tokenizer.
from .tokenizer import encode

# Indexes (into corpus()) of 3 sentences kept OUT of training, to test whether the
# model learned the rules or just memorised the examples:
#   3  -> ids normal  len ok    load high timing jumpy  => attack dos
#   6  -> ids normal  len wrong load high timing steady => attack dos
#   12 -> ids unknown len wrong load low  timing steady => attack fuzzing
HELD_OUT = [3, 6, 12]


# Split the 16 sentences into 13 for training and 3 never shown to the model.
def split_corpus() -> tuple[list[str], list[str]]:
    # All 16 sentences.
    sentences = corpus()
    # Sentences not held out are for training.
    train = [s for i, s in enumerate(sentences) if i not in HELD_OUT]
    # Held-out sentences are for testing.
    unseen = [sentences[i] for i in HELD_OUT]
    # Return both lists.
    return train, unseen


# Turn sentences into the arrays training needs.
def to_arrays(sentences: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # One row of 12 token ids per sentence.
    seqs = np.array([encode(s) for s in sentences])
    # Input: words 1..11. Target: words 2..12, so each position's target is its NEXT word.
    ids, targets = seqs[:, :-1], seqs[:, 1:]
    # Grade only the answer positions (after "=>").
    mask = np.zeros(targets.shape)
    # Mark the graded positions with 1.
    mask[:, ANSWER_POSITIONS] = 1.0
    # Return all three.
    return ids, targets, mask


# Train the model; return the loss after each epoch and the history of one tracked weight.
#   track -> (weight name, index) of one number to follow, so you can watch a weight change
def train(model: TinyGPT, sentences: list[str], epochs: int = 200, lr: float = 0.003,
          track: tuple[str, tuple] | None = None) -> tuple[list[float], list[float]]:
    # Turn the sentences into arrays once.
    ids, targets, mask = to_arrays(sentences)
    # Loss per epoch, and the tracked weight's value per epoch.
    losses, tracked = [], []
    # Repeat the learning step.
    for _ in range(epochs):
        # Record the tracked weight's value before this step.
        if track:
            tracked.append(float(model.params[track[0]][track[1]]))
        # Forward pass + backpropagation.
        loss, grads = model.loss_and_grads(ids, targets, mask)
        # Adjust every weight.
        model.adam_step(grads, lr=lr)
        # Record the loss.
        losses.append(loss)
    # Return both histories.
    return losses, tracked

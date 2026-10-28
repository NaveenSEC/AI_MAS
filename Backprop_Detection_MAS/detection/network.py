"""A tiny neural network that learns by backpropagation, written out step by step.

Each detector agent owns one of these networks. Its shape:

    inputs (k features) --W1,b1--> hidden layer (tanh) --W2,b2--> output (sigmoid)
                                                                   = P(attack)

Learning loop (one "epoch"):
  1. FORWARD pass:  push the inputs through the network to get predictions.
  2. LOSS:          measure how wrong the predictions are (binary cross-entropy).
  3. BACKWARD pass (backpropagation): use the chain rule to work out, for every
                    weight, how much the loss would change if that weight changed
                    a little (its *gradient*). We start at the output and move
                    backwards layer by layer, reusing each layer's result for the
                    layer before it. That reuse is what makes backprop fast.
  4. UPDATE:        nudge every weight a small step against its gradient
                    (gradient descent), so the loss goes down.
"""

# Lets us write modern type hints on older Python versions too.
from __future__ import annotations

# numpy: arrays and matrix maths.
import numpy as np


# Squash any number into the range 0..1, so the output reads as a probability.
def sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip z first: exp() of a very large number overflows.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


class TinyNet:
    """One hidden layer, trained with full-batch gradient descent."""

    # Create the network with small random starting weights.
    #   n_inputs -> how many features this network sees
    #   n_hidden -> how many neurons in the hidden layer
    #   seed     -> makes the random starting weights repeatable
    def __init__(self, n_inputs: int, n_hidden: int = 8, seed: int = 0) -> None:
        # Random number generator for the starting weights.
        rng = np.random.default_rng(seed)
        # W1: one weight per (input, hidden neuron) pair. Scaled by 1/sqrt(n_inputs)
        # so the hidden layer starts in tanh's useful middle range, not saturated.
        self.W1 = rng.normal(0.0, 1.0 / np.sqrt(n_inputs), size=(n_inputs, n_hidden))
        # b1: one bias per hidden neuron, starting at zero.
        self.b1 = np.zeros(n_hidden)
        # W2: one weight per hidden neuron, connecting it to the single output.
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(n_hidden), size=(n_hidden, 1))
        # b2: the output's bias, starting at zero.
        self.b2 = np.zeros(1)

    # ---- 1. Forward pass ---------------------------------------------------
    # Compute predictions, and keep the in-between values the backward pass needs.
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        # Hidden layer, before activation: weighted sum of the inputs plus bias.
        # x is (n samples, n inputs) @ W1 (n inputs, n hidden) -> (n samples, n hidden).
        z1 = x @ self.W1 + self.b1
        # Hidden layer, after activation: tanh bends the straight line into a curve,
        # which lets the network learn non-linear patterns.
        a1 = np.tanh(z1)
        # Output, before activation: weighted sum of the hidden neurons plus bias.
        z2 = a1 @ self.W2 + self.b2
        # Output, after sigmoid: probability that each window is an attack.
        p = sigmoid(z2)
        # Return the predictions plus a cache of values for the backward pass.
        return p, {"x": x, "a1": a1, "p": p}

    # ---- 2. Loss ------------------------------------------------------------
    # Binary cross-entropy: a big penalty for confident wrong answers,
    # nearly zero for confident right ones. Lower is better.
    @staticmethod
    def loss(p: np.ndarray, y: np.ndarray) -> float:
        # A tiny number that stops log(0) from becoming minus infinity.
        eps = 1e-12
        # Turn y into a column (n, 1) so it lines up with p.
        y = y.reshape(-1, 1)
        # Average the penalty over all samples.
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    # ---- 3. Backward pass (backpropagation) ---------------------------------
    # Return the gradient of the loss for every weight and bias.
    def backward(self, cache: dict, y: np.ndarray) -> dict:
        # Unpack the values saved by the forward pass.
        x, a1, p = cache["x"], cache["a1"], cache["p"]
        # Number of samples, because the loss is an average over them.
        n = len(y)
        # Output layer error. For sigmoid + cross-entropy, the chain rule
        # simplifies to (prediction - truth), divided by n for the average.
        # This is the error signal that flows backwards from here.
        dz2 = (p - y.reshape(-1, 1)) / n
        # Gradient of W2: each hidden neuron's activation times the output error.
        dW2 = a1.T @ dz2
        # Gradient of b2: the output error summed over all samples.
        db2 = dz2.sum(axis=0)
        # Send the error back to the hidden layer through W2: each hidden neuron
        # gets its share of the blame, in proportion to its weight.
        da1 = dz2 @ self.W2.T
        # Pass the error back through tanh. tanh's slope is (1 - tanh²),
        # so neurons that were saturated (near ±1) pass back very little.
        dz1 = da1 * (1.0 - a1**2)
        # Gradient of W1: each input value times its hidden neuron's error.
        dW1 = x.T @ dz1
        # Gradient of b1: each hidden neuron's error summed over all samples.
        db1 = dz1.sum(axis=0)
        # Return all four gradients by name.
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ---- 4. Update ----------------------------------------------------------
    # Move every parameter a small step downhill (against its gradient).
    def step(self, grads: dict, lr: float) -> None:
        # For each parameter name ("W1", "b1", "W2", "b2")...
        for name, grad in grads.items():
            # ...look up the current array and subtract learning rate × gradient.
            # `-=` changes the array in place.
            getattr(self, name)[...] -= lr * grad

    # One full learning step: forward, loss, backward, update. Returns the loss.
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        # 1. Forward pass.
        p, cache = self.forward(x)
        # 2. Measure the loss (before this update).
        value = self.loss(p, y)
        # 3. Backpropagation: compute every gradient.
        grads = self.backward(cache, y)
        # 4. Gradient-descent update.
        self.step(grads, lr)
        # Report the loss so callers can watch it fall.
        return value

    # Probability of attack for each row of x (forward pass only, no learning).
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Run the forward pass, keep the predictions, and flatten (n, 1) to (n,).
        return self.forward(x)[0].ravel()


# Check backprop against a slow but obviously-correct method: nudge each weight
# up and down by a tiny amount and measure how the loss changes (a "numerical"
# gradient). If the two agree, the backward pass is correct.
def gradient_check(net: TinyNet, x: np.ndarray, y: np.ndarray, h: float = 1e-5) -> float:
    # Gradients from backpropagation.
    grads = net.backward(net.forward(x)[1], y)
    # Track the largest disagreement found.
    worst = 0.0
    # Check every parameter array.
    for name, grad in grads.items():
        # The actual array inside the network (changes to it affect the network).
        param = getattr(net, name)
        # Visit every single number in that array.
        for idx in np.ndindex(param.shape):
            # Remember the original value.
            old = param[idx]
            # Nudge it up and measure the loss.
            param[idx] = old + h
            up = net.loss(net.forward(x)[0], y)
            # Nudge it down and measure the loss.
            param[idx] = old - h
            down = net.loss(net.forward(x)[0], y)
            # Put the original value back.
            param[idx] = old
            # Slope estimate: change in loss divided by change in weight.
            numeric = (up - down) / (2 * h)
            # Keep the biggest difference between backprop and the estimate.
            worst = max(worst, abs(numeric - grad[idx]))
    # A tiny number (around 1e-9) means backprop is right.
    return worst

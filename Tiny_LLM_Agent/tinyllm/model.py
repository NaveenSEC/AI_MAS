"""TinyGPT: a miniature GPT-style language model, written out step by step in numpy.

It has the same building blocks as GPT and Llama, only much smaller:

    words ──▶ [token embedding + position embedding]          layer 0: words become vectors
          ──▶ [transformer block 1: attention ─▶ feed-forward] layer 1
          ──▶ [transformer block 2: attention ─▶ feed-forward] layer 2
          ──▶ [output layer: one score per vocabulary word]    layer 3: scores ("logits")
          ──▶ softmax ──▶ probability of each possible next word

Every number the model "knows" lives in its WEIGHTS (the arrays in
self.params). Training adjusts them with backpropagation until the model
predicts the next word well. `forward(..., trace=True)` records the output of
every layer so you can watch the text being transformed.

Simplifications versus real LLMs (to keep the maths readable):
  * one attention head per block (real models use dozens)
  * no LayerNorm (real models normalise the vectors in every block)
  * whole-word tokens, a 19-word vocabulary and 16-number vectors
"""

# Lets us write modern type hints on older Python versions too.
from __future__ import annotations

# numpy: arrays and matrix maths.
import numpy as np


# Turn a row of scores into probabilities that are all positive and add up to 1.
def softmax(z: np.ndarray) -> np.ndarray:
    # Subtract the largest score first: it doesn't change the result but stops exp() overflowing.
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    # Divide each exp(score) by the row total.
    return e / e.sum(axis=-1, keepdims=True)


class TinyGPT:
    # Create the model with small random weights. Nothing is learned yet.
    #   vocab_size -> number of words (19)
    #   d_model    -> length of each word's vector (16)
    #   d_ff       -> size of the feed-forward layer's middle (32)
    #   n_layers   -> number of transformer blocks (2)
    #   max_len    -> longest sequence the model can read (12 words)
    def __init__(
        self, vocab_size: int, d_model: int = 16, d_ff: int = 32, n_layers: int = 2,
        max_len: int = 12, seed: int = 0,
    ) -> None:
        # Random number generator, so the starting weights are repeatable.
        rng = np.random.default_rng(seed)
        # Remember the sizes.
        self.d_model, self.n_layers = d_model, n_layers
        # All weights, by name. Insertion order is kept, so printing follows the layers.
        p: dict[str, np.ndarray] = {}
        # Token embedding: one 16-number vector per vocabulary word. Row 13 is "attack".
        p["embed.token"] = rng.normal(0, 0.3, (vocab_size, d_model))
        # Position embedding: one vector per position (1st word, 2nd word, ...),
        # so the model can tell word order apart.
        p["embed.position"] = rng.normal(0, 0.3, (max_len, d_model))
        # The transformer blocks.
        for layer in range(1, n_layers + 1):
            # Short name prefix such as "block1".
            b = f"block{layer}"
            # Attention weights: Query ("what am I looking for?"), Key ("what do I contain?"),
            # Value ("what do I pass on?") and Output (mix the result back in).
            for w in ("Wq", "Wk", "Wv", "Wo"):
                # Each is a 16×16 matrix, scaled so vectors keep a sensible size.
                p[f"{b}.attn.{w}"] = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
            # Feed-forward, first layer: expand 16 -> 32 numbers.
            p[f"{b}.ff.W1"] = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_ff))
            # Its bias: one number per middle neuron.
            p[f"{b}.ff.b1"] = np.zeros(d_ff)
            # Feed-forward, second layer: shrink 32 -> 16 numbers.
            p[f"{b}.ff.W2"] = rng.normal(0, 1 / np.sqrt(d_ff), (d_ff, d_model))
            # Its bias.
            p[f"{b}.ff.b2"] = np.zeros(d_model)
        # Output layer: turns the final 16-number vector into one score per vocabulary word.
        p["output.W"] = rng.normal(0, 1 / np.sqrt(d_model), (d_model, vocab_size))
        # Its bias: one per vocabulary word.
        p["output.b"] = np.zeros(vocab_size)
        # Store all the weights on the model.
        self.params = p
        # Adam optimiser memory (running averages of gradients), one entry per weight array.
        self._m = {k: np.zeros_like(v) for k, v in p.items()}
        self._v = {k: np.zeros_like(v) for k, v in p.items()}
        # Number of training steps taken so far (Adam needs it).
        self._t = 0

    # Total number of individual weights (every number in every array).
    def n_params(self) -> int:
        # Add up the sizes of all arrays.
        return sum(v.size for v in self.params.values())

    # ---- FORWARD PASS: text in, next-word scores out ----------------------
    #   ids   -> token ids, shape (batch, sequence length)
    #   trace -> if True, also record each layer's output for printing
    def forward(self, ids: np.ndarray, trace: bool = False) -> tuple[np.ndarray, dict, list]:
        # Short name for the weights.
        p = self.params
        # Batch size B and sequence length T.
        B, T = ids.shape
        # Values the backward pass will need, saved per layer.
        cache: dict = {"ids": ids, "blocks": []}
        # Recorded layer outputs (only filled when trace=True).
        steps: list = []

        # LAYER 0: EMBEDDING. Look up each word's vector and add its position's vector.
        x = p["embed.token"][ids] + p["embed.position"][:T]
        # Record it.
        if trace:
            steps.append(("embedding (token + position)", x[0]))

        # LAYERS 1..n: TRANSFORMER BLOCKS.
        # Mask that stops a word seeing words *after* it (it can't peek at the future).
        future = np.triu(np.ones((T, T), dtype=bool), k=1)
        # Run each block in turn.
        for layer in range(1, self.n_layers + 1):
            # Weight-name prefix for this block.
            b = f"block{layer}"
            # Keep the block's input.
            x_in = x
            # --- Attention: each word gathers information from earlier words.
            # Query vectors: what each word is looking for.
            q = x_in @ p[f"{b}.attn.Wq"]
            # Key vectors: what each word offers.
            k = x_in @ p[f"{b}.attn.Wk"]
            # Value vectors: the information each word passes on.
            v = x_in @ p[f"{b}.attn.Wv"]
            # Scores: how well each query matches each key (dot product), scaled by sqrt(d).
            scores = q @ k.transpose(0, 2, 1) / np.sqrt(self.d_model)
            # Hide future words by giving them a hugely negative score.
            scores = np.where(future, -1e9, scores)
            # Attention weights: softmax turns each row of scores into percentages summing to 1.
            attn = softmax(scores)
            # Each word's new information: the attention-weighted mix of the value vectors.
            mixed = attn @ v
            # Project the mix back with Wo.
            attn_out = mixed @ p[f"{b}.attn.Wo"]
            # Residual connection: add the result to the input instead of replacing it.
            x_mid = x_in + attn_out
            # --- Feed-forward: each word's vector is processed on its own.
            # Expand to 32 numbers.
            pre = x_mid @ p[f"{b}.ff.W1"] + p[f"{b}.ff.b1"]
            # ReLU: keep positive numbers, turn negatives to 0 (the "neuron fires or not").
            hidden = np.maximum(pre, 0.0)
            # Shrink back to 16 numbers.
            ff_out = hidden @ p[f"{b}.ff.W2"] + p[f"{b}.ff.b2"]
            # Residual connection again.
            x = x_mid + ff_out
            # Save everything the backward pass needs for this block.
            cache["blocks"].append(dict(x_in=x_in, q=q, k=k, v=v, attn=attn, mixed=mixed,
                                        x_mid=x_mid, pre=pre, hidden=hidden))
            # Record this block's outputs.
            if trace:
                steps.append((f"block {layer} attention weights", attn[0]))
                steps.append((f"block {layer} after attention (+ residual)", x_mid[0]))
                steps.append((f"block {layer} feed-forward neurons firing", hidden[0]))
                steps.append((f"block {layer} output", x[0]))

        # LAST LAYER: OUTPUT. One score (logit) per vocabulary word, at every position.
        logits = x @ p["output.W"] + p["output.b"]
        # Save the final vectors for the backward pass.
        cache["x_final"] = x
        # Record the scores and the probabilities.
        if trace:
            steps.append(("output scores (logits)", logits[0]))
            steps.append(("next-word probabilities (softmax)", softmax(logits[0])))
        # Return scores, cache and trace.
        return logits, cache, steps

    # ---- LOSS + BACKWARD PASS (backpropagation) ----------------------------
    # How wrong are the predictions, and which way should every weight move?
    #   targets -> the correct next word at each position, shape (batch, T)
    #   mask    -> 1 where we grade the prediction, 0 where we don't
    def loss_and_grads(self, ids: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> tuple[float, dict]:
        # Short name for the weights.
        p = self.params
        # Run the forward pass, keeping the cache.
        logits, cache, _ = self.forward(ids)
        # Batch size and sequence length.
        B, T = ids.shape
        # Turn scores into probabilities.
        probs = softmax(logits)
        # Number of graded predictions (for averaging).
        n = mask.sum()
        # Probability the model gave to the correct word at each position.
        correct = probs[np.arange(B)[:, None], np.arange(T)[None, :], targets]
        # Cross-entropy loss: average of -log(probability of the correct word) on graded positions.
        loss = float(-(np.log(correct + 1e-12) * mask).sum() / n)

        # Gradients, one array per weight, starting at zero.
        g = {k: np.zeros_like(v) for k, v in p.items()}
        # Output-layer error: probabilities minus the one-hot truth (softmax + cross-entropy).
        d_logits = probs.copy()
        # Subtract 1 at the correct word.
        d_logits[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1.0
        # Ignore ungraded positions and average.
        d_logits *= mask[..., None] / n
        # Output weight gradient: final vectors × error, summed over batch and positions.
        g["output.W"] = np.einsum("btd,btv->dv", cache["x_final"], d_logits)
        # Output bias gradient: the error summed.
        g["output.b"] = d_logits.sum(axis=(0, 1))
        # Error flowing back into the final vectors.
        dx = d_logits @ p["output.W"].T

        # Go back through the blocks in REVERSE order (that's the "back" in backpropagation).
        for layer in range(self.n_layers, 0, -1):
            # Prefix and saved values for this block.
            b, c = f"block{layer}", cache["blocks"][layer - 1]
            # --- Feed-forward, backwards. Residual: the error reaches x_mid directly too.
            d_ff_out = dx
            # Gradient of W2: middle neurons × error.
            g[f"{b}.ff.W2"] = np.einsum("btf,btd->fd", c["hidden"], d_ff_out)
            # Gradient of b2.
            g[f"{b}.ff.b2"] = d_ff_out.sum(axis=(0, 1))
            # Error at the middle neurons, back through W2.
            d_hidden = d_ff_out @ p[f"{b}.ff.W2"].T
            # Back through ReLU: neurons that didn't fire (pre <= 0) pass no error back.
            d_pre = d_hidden * (c["pre"] > 0)
            # Gradient of W1.
            g[f"{b}.ff.W1"] = np.einsum("btd,btf->df", c["x_mid"], d_pre)
            # Gradient of b1.
            g[f"{b}.ff.b1"] = d_pre.sum(axis=(0, 1))
            # Error at x_mid: residual path + path back through W1.
            d_mid = dx + d_pre @ p[f"{b}.ff.W1"].T
            # --- Attention, backwards.
            # Gradient of Wo.
            g[f"{b}.attn.Wo"] = np.einsum("btd,bte->de", c["mixed"], d_mid)
            # Error at the mixed values, back through Wo.
            d_mixed = d_mid @ p[f"{b}.attn.Wo"].T
            # Error at the attention weights.
            d_attn = d_mixed @ c["v"].transpose(0, 2, 1)
            # Error at the value vectors.
            d_v = c["attn"].transpose(0, 2, 1) @ d_mixed
            # Back through softmax (its slope depends on all the weights in the row).
            d_scores = c["attn"] * (d_attn - (d_attn * c["attn"]).sum(axis=-1, keepdims=True))
            # Undo the sqrt(d) scaling.
            d_scores /= np.sqrt(self.d_model)
            # Error at the queries.
            d_q = d_scores @ c["k"]
            # Error at the keys.
            d_k = d_scores.transpose(0, 2, 1) @ c["q"]
            # Gradients of Wq, Wk, Wv: block input × their errors.
            g[f"{b}.attn.Wq"] = np.einsum("btd,bte->de", c["x_in"], d_q)
            g[f"{b}.attn.Wk"] = np.einsum("btd,bte->de", c["x_in"], d_k)
            g[f"{b}.attn.Wv"] = np.einsum("btd,bte->de", c["x_in"], d_v)
            # Error at the block input: residual path + paths back through Wq, Wk, Wv.
            dx = (d_mid + d_q @ p[f"{b}.attn.Wq"].T + d_k @ p[f"{b}.attn.Wk"].T
                  + d_v @ p[f"{b}.attn.Wv"].T)

        # --- Embedding, backwards. Each word's vector gets the error from every place it appeared.
        np.add.at(g["embed.token"], cache["ids"], dx)
        # Each position's vector gets the error summed over the batch.
        g["embed.position"][:T] = dx.sum(axis=0)
        # Return the loss and every gradient.
        return loss, g

    # ---- UPDATE: move the weights (Adam optimiser) -------------------------
    # Adam is gradient descent with momentum and a per-weight step size; it's what real LLMs use.
    def adam_step(self, grads: dict, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999) -> None:
        # Count this step.
        self._t += 1
        # Update every weight array.
        for k, grad in grads.items():
            # Running average of the gradient (momentum: keeps moving in a consistent direction).
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * grad
            # Running average of the squared gradient (how noisy this weight's gradient is).
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * grad**2
            # Correct the averages' bias towards zero in the first few steps.
            m_hat = self._m[k] / (1 - beta1**self._t)
            v_hat = self._v[k] / (1 - beta2**self._t)
            # Step each weight downhill; noisy weights take smaller steps.
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # ---- GENERATION: write text one word at a time -------------------------
    # Continue a prompt by repeatedly predicting the most likely next word.
    def generate(self, ids: list[int], end_id: int, max_new: int = 3) -> list[int]:
        # Copy the prompt so we don't change the caller's list.
        ids = list(ids)
        # Add at most max_new words.
        for _ in range(max_new):
            # Run the model on everything so far (a batch of one).
            logits, _, _ = self.forward(np.array([ids]))
            # Take the scores at the last position and pick the highest (greedy decoding).
            next_id = int(np.argmax(logits[0, -1]))
            # Append it to the text.
            ids.append(next_id)
            # Stop at the end marker.
            if next_id == end_id:
                break
        # Return prompt + generated words.
        return ids


# Compare backprop's gradients with slow numerical ones (nudge each weight, see how the loss moves).
# A tiny maximum difference proves the backward pass is correct.
def gradient_check(model: TinyGPT, ids, targets, mask, h: float = 1e-5, samples: int = 8) -> float:
    # Gradients from backpropagation.
    _, grads = model.loss_and_grads(ids, targets, mask)
    # Random generator for picking which weights to test.
    rng = np.random.default_rng(0)
    # Largest disagreement so far.
    worst = 0.0
    # For every weight array...
    for name, param in model.params.items():
        # ...test a few random positions in it (testing all would be slow).
        for _ in range(samples):
            # A random index into this array.
            idx = tuple(rng.integers(0, s) for s in param.shape)
            # Remember the original value.
            old = param[idx]
            # Loss with the weight nudged up.
            param[idx] = old + h
            up = model.loss_and_grads(ids, targets, mask)[0]
            # Loss with the weight nudged down.
            param[idx] = old - h
            down = model.loss_and_grads(ids, targets, mask)[0]
            # Restore it.
            param[idx] = old
            # Compare the numerical slope with backprop's gradient.
            worst = max(worst, abs((up - down) / (2 * h) - grads[name][idx]))
    # Return the largest difference.
    return worst

"""Tiny LLM Agent: build, train and look inside a miniature language model, then use it as an agent.

Run:  python main.py

It prints, in order:
  1. The tokenizer and training data.
  2. Every weight in the model: name, shape, count and job.
  3. Training: the loss falling, and one weight changing.
  4. Inside the model: the output of every layer for one prompt.
  5. The agent at work: perceive -> think (LLM) -> act.
"""

# argparse: Python's built-in tool for reading command-line options.
import argparse

# numpy: arrays and maths.
import numpy as np

# The agent.
from tinyllm.agent import SecurityAgent

# The language model and its backprop correctness check.
from tinyllm.model import TinyGPT, gradient_check, softmax

# Tokenizer pieces.
from tinyllm.tokenizer import VOCAB, WORD_TO_ID, decode, encode

# Training helpers.
from tinyllm.train import split_corpus, to_arrays, train

# What each kind of weight does, for the weights table (matched by the end of the name).
WEIGHT_JOBS = {
    "embed.token": "a 16-number vector for each of the 19 words",
    "embed.position": "a 16-number vector for each position 1-12 (word order)",
    "attn.Wq": "Query: turns a word into 'what am I looking for?'",
    "attn.Wk": "Key: turns a word into 'what do I contain?'",
    "attn.Wv": "Value: turns a word into 'what do I pass on?'",
    "attn.Wo": "mixes the gathered information back into the word",
    "ff.W1": "feed-forward: 16 -> 32 neurons (pattern detectors)",
    "ff.b1": "feed-forward: a firing threshold for each of the 32 neurons",
    "ff.W2": "feed-forward: 32 -> 16, writes detected patterns back",
    "ff.b2": "feed-forward: output offset",
    "output.W": "scores every vocabulary word as the next word",
    "output.b": "base score for each vocabulary word",
}


# Format the first few numbers of a vector, e.g. "[ 0.12 -0.40  1.05 ...]".
def vec(v: np.ndarray, n: int = 6) -> str:
    # Each number to 2 decimal places, padded to the same width.
    shown = " ".join(f"{x:5.2f}" for x in v[:n])
    # Add "..." if some numbers were left out.
    return f"[{shown}{' ...' if len(v) > n else ''}]"


# The 5 most likely next words with their probabilities, e.g. "attack 98.1%, normal 1.2%, ...".
def top_words(probs: np.ndarray, k: int = 5) -> str:
    # Indexes of the k largest probabilities, largest first.
    best = np.argsort(probs)[::-1][:k]
    # Word and percentage for each.
    return ", ".join(f"{VOCAB[i]} {probs[i]:.1%}" for i in best)


# Print a heading for each section.
def heading(text: str) -> None:
    # Blank line, the title, and an underline of the same length.
    print(f"\n{text}\n{'=' * len(text)}")


# All the program's work happens inside this function.
def main() -> None:
    # Command-line options.
    parser = argparse.ArgumentParser(description="Look inside a tiny LLM and use it as an agent.")
    # How many training rounds.
    parser.add_argument("--epochs", type=int, default=200, help="training iterations")
    # Size of each weight update.
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    # Prompt to trace through the layers.
    parser.add_argument("--prompt", default="ids unknown len ok load low timing steady =>",
                        help="observation to trace layer by layer")
    # Seed for the random starting weights.
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    # Read the command line.
    args = parser.parse_args()

    # ---- 1. Tokenizer and data -------------------------------------------
    heading("1. TOKENIZER AND TRAINING DATA")
    # 13 sentences to learn from, 3 kept back to test on.
    train_text, unseen_text = split_corpus()
    # Vocabulary size.
    print(f"Vocabulary: {len(VOCAB)} words -> {' '.join(VOCAB)}")
    # Show one sentence as words and as numbers.
    print(f"Example sentence: {train_text[0]}")
    print(f"As token ids:     {encode(train_text[0])}")
    # Data sizes.
    print(f"Training sentences: {len(train_text)}   Held back (never shown to the model): {len(unseen_text)}")

    # ---- 2. Weights ----------------------------------------------------------
    heading("2. THE MODEL'S WEIGHTS (everything it will 'know' lives here)")
    # Create the model with random weights.
    model = TinyGPT(vocab_size=len(VOCAB), seed=args.seed)
    # One line per weight array.
    for name, w in model.params.items():
        # Find the job description whose key ends this name.
        job = next(j for key, j in WEIGHT_JOBS.items() if name.endswith(key))
        # Name, shape, number of weights and job.
        print(f"  {name:<18} {str(w.shape):<9} {w.size:>5} weights  {job}")
    # Total.
    print(f"  TOTAL: {model.n_params():,} weights (GPT-2 small has 124 million; Llama 3 8B has 8 billion)")

    # ---- 3. Training -----------------------------------------------------------
    heading("3. TRAINING (backpropagation adjusts the weights)")
    # Check the hand-written backpropagation first.
    ids, targets, mask = to_arrays(train_text)
    print(f"Gradient check (backprop vs numerical): max difference {gradient_check(model, ids, targets, mask):.1e}")
    # A prompt to compare before and after training.
    probe = encode(args.prompt)
    # Next-word probabilities with random weights.
    before = softmax(model.forward(np.array([probe]))[0][0, -1])
    print(f"Before training, after '{args.prompt}' the model predicts: {top_words(before, 3)}")
    # One weight to follow: how strongly the final vector's 1st number votes for the word "dos".
    track = ("output.W", (0, WORD_TO_ID["dos"]))
    # Train.
    losses, tracked = train(model, train_text, epochs=args.epochs, lr=args.lr, track=track)
    # Print the loss and the tracked weight at a few checkpoints.
    print(f"  {'epoch':>5}  {'loss':>7}  output.W[0,'dos']")
    for e in sorted({0, 10, 25, 50, 100, args.epochs - 1} & set(range(args.epochs))):
        print(f"  {e + 1:>5}  {losses[e]:7.4f}  {tracked[e]:+.4f}")
    # Next-word probabilities after training.
    after = softmax(model.forward(np.array([probe]))[0][0, -1])
    print(f"After training, the model predicts: {top_words(after, 3)}")
    # Test on the 3 sentences the model never saw.
    print("Test on the 3 held-back observations:")
    for sentence in unseen_text:
        # The observation part (first 9 words).
        p = " ".join(sentence.split()[:9])
        # The model's completion.
        out = decode(model.generate(encode(p), WORD_TO_ID["<end>"])[9:])
        # The correct completion.
        truth = " ".join(sentence.split()[9:])
        # Show both and whether they match.
        print(f"  {p} {out:<22} {'CORRECT' if out == truth else 'WRONG (expected ' + truth + ')'}")

    # ---- 4. Inside the model ---------------------------------------------------
    heading("4. INSIDE THE MODEL: the output of every layer")
    # The prompt's words and ids.
    words = args.prompt.split()
    print(f"Prompt: '{args.prompt}'")
    print(f"Token ids: {probe}   ({len(words)} words, each becomes a vector of {model.d_model} numbers)")
    # Run the model once, recording every layer.
    _, _, steps = model.forward(np.array([probe]), trace=True)
    # Print each recorded step.
    for name, value in steps:
        # Attention weights: a table of who looks at whom.
        if "attention weights" in name:
            print(f"\n[{name}]  row = word doing the looking, columns = words it looks at (%)")
            # Column headings: the first 6 letters of each word.
            print("           " + " ".join(f"{w[:6]:>6}" for w in words))
            # One row per word.
            for i, w in enumerate(words):
                # Percentages; future words are 0 because of the mask.
                print(f"  {w:<8} " + " ".join(f"{x * 100:6.0f}" for x in value[i]))
            # Explain the most important row: "=>", which has to produce the verdict.
            last = value[-1]
            # The three words "=>" pays most attention to.
            focus = ", ".join(f"'{words[i]}' {last[i]:.0%}" for i in np.argsort(last)[::-1][:3])
            print(f"  -> '=>' (which must produce the verdict) looks mostly at: {focus}")
        # Feed-forward neurons: how many fired for the last word.
        elif "neurons" in name:
            # A neuron "fires" if its value is above zero after ReLU.
            fired = np.flatnonzero(value[-1] > 0)
            print(f"\n[{name}]  for '=>': {len(fired)} of {value.shape[1]} neurons fired: {fired.tolist()}")
        # Output scores and probabilities: show the top words.
        elif "logits" in name or "probabilities" in name:
            # Scores are shown as raw numbers; probabilities as percentages.
            if "logits" in name:
                best = np.argsort(value[-1])[::-1][:5]
                print(f"\n[{name}]  top scores for the next word: "
                      + ", ".join(f"{VOCAB[i]} {value[-1][i]:.1f}" for i in best))
            else:
                print(f"[{name}]  {top_words(value[-1])}")
        # Vectors (embedding and block outputs): show the first numbers for every word.
        else:
            print(f"\n[{name}]  first 6 of {value.shape[1]} numbers per word:")
            for w, row in zip(words, value):
                print(f"  {w:<8} {vec(row)}")
    # Generation: the model writes the answer one word at a time.
    print("\n[generation] the model feeds each new word back in and predicts again:")
    # Start from the prompt.
    text = list(probe)
    # Up to 3 new words.
    for _ in range(3):
        # Probabilities for the next word.
        probs = softmax(model.forward(np.array([text]))[0][0, -1])
        # Pick the most likely.
        nxt = int(np.argmax(probs))
        # Show the step.
        print(f"  '... {decode(text[-2:])}' -> '{VOCAB[nxt]}' ({probs[nxt]:.1%})")
        # Append it.
        text.append(nxt)
        # Stop at the end marker.
        if nxt == WORD_TO_ID["<end>"]:
            break

    # ---- 5. The agent ------------------------------------------------------------
    heading("5. THE AGENT AT WORK: perceive -> think (LLM) -> act")
    # The agent uses the trained model as its brain.
    agent = SecurityAgent(model)
    # Four raw CAN-bus readings, as an in-vehicle monitor would measure them.
    observations = [
        {"unknown_id_ratio": 0.01, "dlc_mismatch_ratio": 0.00, "bus_load_pct": 34, "jitter_ms": 0.02},
        {"unknown_id_ratio": 0.00, "dlc_mismatch_ratio": 0.01, "bus_load_pct": 88, "jitter_ms": 0.02},
        {"unknown_id_ratio": 0.22, "dlc_mismatch_ratio": 0.09, "bus_load_pct": 40, "jitter_ms": 0.02},
        {"unknown_id_ratio": 0.01, "dlc_mismatch_ratio": 0.00, "bus_load_pct": 37, "jitter_ms": 0.06},
    ]
    # The prompts of the held-back sentences, to label readings the model never trained on.
    unseen_prompts = {" ".join(s.split()[:9]) for s in unseen_text}
    # Run the agent on each.
    for n, obs in enumerate(observations, 1):
        # Perceive, think, act.
        d = agent.step(obs)
        # Show every stage.
        print(f"\nReading {n}: {obs}")
        print(f"  perceive -> '{d.prompt}'"
              + ("   (never seen in training)" if d.prompt in unseen_prompts else ""))
        print(f"  think    -> LLM writes '{d.completion}'")
        print(f"  act      -> {d.action}")


# This is True only when the file is run directly (python main.py),
# not when another file imports it. That stops main() from running on import.
if __name__ == "__main__":
    # Start the program.
    main()

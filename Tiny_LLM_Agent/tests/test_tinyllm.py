"""Automated tests. Run them with `pytest` from the Tiny_LLM_Agent folder.

pytest finds every function whose name starts with `test_` and runs it.
Each `assert` checks that something is true; if any assert fails, the test fails.
"""

# numpy: arrays and maths.
import numpy as np

# The agent and its action table.
from tinyllm.agent import ACTIONS, SecurityAgent

# The model and the backprop check.
from tinyllm.model import TinyGPT, gradient_check

# Tokenizer pieces.
from tinyllm.tokenizer import VOCAB, WORD_TO_ID, decode, encode

# Training helpers.
from tinyllm.train import split_corpus, to_arrays, train


# Train one model, shared by the tests that need a trained model.
def _trained_model():
    # 13 training sentences and 3 held back.
    train_text, unseen_text = split_corpus()
    # A fresh model.
    model = TinyGPT(vocab_size=len(VOCAB), seed=0)
    # Train it.
    train(model, train_text)
    # Return the model and the held-back sentences.
    return model, unseen_text


# encode and decode must be exact opposites.
def test_tokenizer_round_trip():
    # A sentence using several words.
    text = "ids unknown len wrong => attack fuzzing <end>"
    # Encoding then decoding must give back the same text.
    assert decode(encode(text)) == text


# Backprop's gradients must match slow numerical gradients.
def test_backprop_matches_numerical_gradients():
    # Training arrays.
    ids, targets, mask = to_arrays(split_corpus()[0])
    # The biggest disagreement must be tiny.
    assert gradient_check(TinyGPT(len(VOCAB), seed=1), ids, targets, mask) < 1e-7


# Training must drive the loss right down.
def test_training_reduces_loss():
    # A fresh model.
    model = TinyGPT(len(VOCAB), seed=0)
    # Train and keep the loss history.
    losses, _ = train(model, split_corpus()[0])
    # From above 1 to nearly 0.
    assert losses[0] > 1.0 and losses[-1] < 0.01


# The trained model must answer observations it never saw during training.
def test_generalises_to_held_back_sentences():
    # Trained model and held-back sentences.
    model, unseen_text = _trained_model()
    # Check each one.
    for sentence in unseen_text:
        # The observation part (first 9 words).
        prompt_ids = encode(" ".join(sentence.split()[:9]))
        # The model's full sentence must match the correct one.
        assert decode(model.generate(prompt_ids, WORD_TO_ID["<end>"])) == sentence


# Each layer's output must have the expected shape.
def test_trace_has_every_layer():
    # A fresh model and a 9-word prompt.
    model = TinyGPT(len(VOCAB), seed=0)
    prompt_ids = encode("ids normal len ok load low timing steady =>")
    # Run with tracing on.
    _, _, steps = model.forward(np.array([prompt_ids]), trace=True)
    # Turn the list into a name -> value dict.
    layers = dict(steps)
    # Embedding: one 16-number vector per word.
    assert layers["embedding (token + position)"].shape == (9, 16)
    # Attention: every word's percentages over the words, each row adding up to 1.
    assert np.allclose(layers["block 2 attention weights"].sum(axis=1), 1.0)
    # Probabilities: one per vocabulary word, adding up to 1.
    assert np.isclose(layers["next-word probabilities (softmax)"].sum(axis=1)[-1], 1.0)


# The agent must turn raw readings into the right action.
def test_agent_actions():
    # An agent with a trained brain.
    agent = SecurityAgent(_trained_model()[0])
    # A flooded bus must be blocked as DoS.
    dos = {"unknown_id_ratio": 0.0, "dlc_mismatch_ratio": 0.0, "bus_load_pct": 90, "jitter_ms": 0.02}
    assert agent.step(dos).action == ACTIONS["dos"]
    # Quiet, regular traffic must be allowed.
    calm = {"unknown_id_ratio": 0.0, "dlc_mismatch_ratio": 0.0, "bus_load_pct": 30, "jitter_ms": 0.02}
    assert agent.step(calm).action == ACTIONS["none"]

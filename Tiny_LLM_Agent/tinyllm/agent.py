"""SecurityAgent: an AI agent whose "brain" is the TinyGPT language model.

An agent repeats three steps:
  1. PERCEIVE: read raw measurements from the CAN bus and describe them in words.
  2. THINK:    give that description to the language model as a prompt and let it
               write the rest of the sentence (the verdict).
  3. ACT:      turn the verdict into an action: allow, alert or block.

This is the same pattern as an agent built on Claude or GPT. The only
difference is the size of the language model in step 2.
"""

# Lets us write modern type hints on older Python versions too.
from __future__ import annotations

# dataclass: writes __init__ and friends for us.
from dataclasses import dataclass

# The prompt template.
from .data import prompt

# The language model.
from .model import TinyGPT

# Word <-> token id conversion.
from .tokenizer import WORD_TO_ID, decode, encode

# What to do for each verdict the model can write.
ACTIONS = {
    "none": "ALLOW: traffic looks normal",
    "dos": "BLOCK: rate-limit the flooding CAN ID and alert the driver",
    "fuzzing": "BLOCK: drop unknown IDs and malformed frames at the gateway",
    "spoofing": "ALERT: possible ECU impersonation, start ECU authentication check",
}


# One decision, kept so it can be printed or tested.
@dataclass
class Decision:
    observation: dict  # the raw measurements
    prompt: str  # the words given to the model
    completion: str  # the words the model wrote
    action: str  # what the agent does


class SecurityAgent:
    # Create the agent around a trained language model.
    def __init__(self, model: TinyGPT) -> None:
        # The agent's "brain".
        self.model = model

    # 1. PERCEIVE: turn numbers into the words the model understands.
    @staticmethod
    def perceive(obs: dict) -> str:
        # More than 5% unknown IDs counts as "unknown".
        ids = "unknown" if obs["unknown_id_ratio"] > 0.05 else "normal"
        # More than 3% wrong lengths counts as "wrong".
        length = "wrong" if obs["dlc_mismatch_ratio"] > 0.03 else "ok"
        # Bus load above 60% counts as "high".
        load = "high" if obs["bus_load_pct"] > 60 else "low"
        # Timing jitter above 0.035 ms counts as "jumpy".
        timing = "jumpy" if obs["jitter_ms"] > 0.035 else "steady"
        # Fill in the prompt template.
        return prompt(ids, length, load, timing)

    # 2. THINK: let the language model complete the prompt.
    def think(self, text: str) -> str:
        # Turn the prompt into token ids.
        ids = encode(text)
        # Generate up to 3 new words, stopping at <end>.
        out = self.model.generate(ids, end_id=WORD_TO_ID["<end>"], max_new=3)
        # Return only the new words, as text.
        return decode(out[len(ids):])

    # 3. ACT: choose an action from the model's answer.
    @staticmethod
    def act(completion: str) -> str:
        # The answer looks like "attack fuzzing <end>"; the second word is the type.
        words = completion.split()
        # Take the type if there is one.
        kind = words[1] if len(words) > 1 else ""
        # Look up the action. If the model wrote something unexpected, fail safe and alert.
        return ACTIONS.get(kind, "ALERT: unrecognised model output, needs a human")

    # Run all three steps for one observation.
    def step(self, obs: dict) -> Decision:
        # Perceive.
        text = self.perceive(obs)
        # Think.
        completion = self.think(text)
        # Act, and keep the whole record.
        return Decision(obs, text, completion, self.act(completion))

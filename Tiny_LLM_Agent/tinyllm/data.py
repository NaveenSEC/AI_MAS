"""Training text: every possible CAN observation, with the correct verdict.

Each training sentence has the same 12-word shape:

    ids <normal|unknown> len <ok|wrong> load <low|high> timing <steady|jumpy> => <verdict> <type> <end>

The model learns to continue the text after "=>", which means producing the
verdict. The rules it has to discover from the examples (nobody tells it):
  * load high                       -> attack dos
  * otherwise ids unknown/len wrong -> attack fuzzing
  * otherwise timing jumpy          -> attack spoofing
  * otherwise                       -> normal none
"""

# itertools.product: every combination of several lists (here 2×2×2×2 = 16).
from itertools import product

# Positions (in the 12-word sentence) whose *next* word the model is graded on:
# position 8 is "=>" (predict the verdict), 9 is the verdict (predict the type),
# 10 is the type (predict <end>). Words before "=>" are the observation, which
# the model isn't asked to guess.
ANSWER_POSITIONS = [8, 9, 10]


# Apply the detection rules to one observation and return (verdict, attack type).
def label(ids: str, length: str, load: str, timing: str) -> tuple[str, str]:
    # A flooded bus is a denial-of-service attack.
    if load == "high":
        return "attack", "dos"
    # Unknown IDs or wrong lengths mean someone is fuzzing.
    if ids == "unknown" or length == "wrong":
        return "attack", "fuzzing"
    # Irregular timing means two senders on one ID: spoofing.
    if timing == "jumpy":
        return "attack", "spoofing"
    # Nothing suspicious.
    return "normal", "none"


# The observation part of a sentence, up to and including "=>". The agent sends this as the prompt.
def prompt(ids: str, length: str, load: str, timing: str) -> str:
    # Fill the four values into the fixed template.
    return f"ids {ids} len {length} load {load} timing {timing} =>"


# Build all 16 training sentences.
def corpus() -> list[str]:
    # Collect sentences here.
    sentences = []
    # Every combination of the four observation values.
    for ids, length, load, timing in product(
        ["normal", "unknown"], ["ok", "wrong"], ["low", "high"], ["steady", "jumpy"]
    ):
        # Work out the correct answer for this combination.
        verdict, kind = label(ids, length, load, timing)
        # Observation + answer + end marker = one full training sentence.
        sentences.append(f"{prompt(ids, length, load, timing)} {verdict} {kind} <end>")
    # Return all 16.
    return sentences

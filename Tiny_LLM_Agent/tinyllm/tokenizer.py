"""Tokenizer: turns words into numbers (token ids) and back.

A language model can't read letters. It reads numbers. The tokenizer gives
every word in the vocabulary its own number. Real LLMs split text into
word-pieces and have about 100,000 of them; ours has 19 whole words.
"""

# Every word our tiny model knows, in a fixed order. A word's position in
# this list is its token id: "ids" is 0, "normal" is 1, and so on.
VOCAB = [
    "ids",  # 0  label: are the CAN IDs known?
    "normal",  # 1  value "known/normal", and also the verdict "normal"
    "unknown",  # 2  value: IDs the vehicle never uses
    "len",  # 3  label: are message lengths (DLC) right?
    "ok",  # 4  value: lengths are right
    "wrong",  # 5  value: lengths are wrong
    "load",  # 6  label: how busy is the bus?
    "low",  # 7  value: normal bus load
    "high",  # 8  value: bus flooded
    "timing",  # 9  label: are message gaps regular?
    "steady",  # 10 value: regular timing
    "jumpy",  # 11 value: irregular timing (two senders on one ID)
    "=>",  # 12 separator: "now give the verdict"
    "attack",  # 13 verdict
    "dos",  # 14 attack type: denial of service
    "fuzzing",  # 15 attack type: random IDs / data
    "spoofing",  # 16 attack type: impersonating an ECU
    "none",  # 17 attack type when traffic is normal
    "<end>",  # 18 marks the end of the answer
]

# Look-up table from word to id, built once from VOCAB: {"ids": 0, "normal": 1, ...}.
WORD_TO_ID = {word: i for i, word in enumerate(VOCAB)}


# Turn a sentence (words separated by spaces) into a list of token ids.
def encode(text: str) -> list[int]:
    # Split on spaces and look each word up. A word not in VOCAB raises KeyError.
    return [WORD_TO_ID[word] for word in text.split()]


# Turn a list of token ids back into a sentence.
def decode(ids: list[int]) -> str:
    # Look each id up in VOCAB and join the words with spaces.
    return " ".join(VOCAB[i] for i in ids)

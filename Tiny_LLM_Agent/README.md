# Tiny LLM Agent: look inside a language model, layer by layer

A **miniature GPT-style language model** (about 5,000 weights), written from
scratch in numpy, and an **AI agent** that uses it as its brain to detect
CAN-bus attacks.

Real LLMs like GPT, Llama and Claude are far too big to inspect. Llama 3 8B has
8 billion weights, and Claude's weights are not public. This model uses the
**same building blocks**, small enough that you can print every weight and the
output of every layer. Every line of code has a comment.

```
                       ┌────────────────────── TinyGPT (the LLM) ───────────────────────┐
CAN-bus   PERCEIVE     │ words → token ids → Embedding → Block 1 → Block 2 → Output    │   ACT
readings ───────────▶  │                     (layer 0)   (attention  (attention (scores │ ─────────▶ ALLOW /
(numbers)  numbers     │                                  + feed-     + feed-    for    │            ALERT /
           → words     │                                  forward)    forward)   next   │            BLOCK
                       │                                                        word)  │
                       └────────── THINK: the model writes "attack fuzzing <end>" ──────┘
```

## Run it

```bash
cd Tiny_LLM_Agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # just numpy and pytest
python main.py
```

It takes about 1 second and needs no GPU, internet or API key. It prints the 5 sections below.

---

## 1. Words become numbers (tokenizer)

A model can't read letters, so each word gets an id:

```
ids normal len ok load low timing steady => normal none <end>
[0,  1,    3,  4, 6,   7,  9,     10,    12, 1,   17,  18]
```

The model is trained on 13 sentences like this. **3 more are held back** to test
whether it learned the rules or only memorised the examples.

## 2. What "weights" are

The model's knowledge lives in **weights**: 5,011 numbers in 20 arrays. Before
training they're random; training tunes them.

| Weights | Shape | Job |
|---------|-------|-----|
| `embed.token` | 19 × 16 | A 16-number vector for each word; similar words end up with similar vectors |
| `embed.position` | 12 × 16 | A vector for each position, so word order matters |
| `blockN.attn.Wq` | 16 × 16 | **Query**: "what is this word looking for?" |
| `blockN.attn.Wk` | 16 × 16 | **Key**: "what does this word contain?" |
| `blockN.attn.Wv` | 16 × 16 | **Value**: "what does this word pass on?" |
| `blockN.attn.Wo` | 16 × 16 | Mixes the gathered information back in |
| `blockN.ff.W1`, `b1` | 16 × 32 | 32 "neurons", each detecting a pattern |
| `blockN.ff.W2`, `b2` | 32 × 16 | Writes detected patterns back into the vector |
| `output.W`, `b` | 16 × 19 | Scores each of the 19 words as the next word |

A weight is just a multiplier. For example, `output.W[0, "dos"]` says how much
the 1st number of the final vector pushes the model towards saying "dos". During
training you can watch it move: −0.1644 → −0.2620 → −0.2659.

## 3. Training: how the weights get tuned

Each training round, or **epoch**, has four steps:
1. **Predict:** the model guesses the next word.
2. **Measure:** the **loss** measures how wrong the guess was.
3. **Backpropagate:** the error is traced backwards through every layer to find
   how each weight should change.
4. **Adjust:** every weight is nudged a little in that direction.

```
epoch   loss     ← how wrong (lower is better)
    1   3.4460   ← random weights: guessing
   26   0.4288   ← patterns emerging
  101   0.0012   ← nearly perfect
  200   0.0003
```

| Prompt `ids unknown len ok load low timing steady =>` | Model's top predictions |
|---|---|
| Before training | low 15.6%, high 11.2%, ok 8.6% (nonsense) |
| After training | **attack 100.0%** |

It also gets all **3 held-back observations right**, so it learned the rules, not just the answers.

## 4. Inside the model: the output of each layer

For the prompt `ids unknown len ok load low timing steady =>`:

**Layer 0: embedding.** Each word becomes 16 numbers (its token vector plus its position vector):
```
unknown  [-0.10  0.21 -0.44  0.15  0.20  0.11 ...]
=>       [ 0.82  0.06 -0.20  0.86 -0.84  0.13 ...]
```

**Block 1: attention.** Each word looks back at earlier words and gathers information.
In block 1, the attention of `=>` is still spread out:
`'ok' 18%, 'unknown' 16%, 'steady' 13%`.

**Block 2: attention.** Now `=>`, the word that must produce the verdict, focuses on the clue that matters:
```
              ids unknow    len     ok   load    low timing steady     =>
  =>            0     66      0      0      0      5      7      2     19
```
**66% on "unknown"**, the sign of fuzzing. Nobody programmed this; backpropagation
tuned `Wq` and `Wk` until it happened.

**Feed-forward.** Pattern-detector neurons switch on or off (ReLU). For `=>`,
14 of 32 neurons fire in block 1 and 19 of 32 in block 2.

**Output layer.** One score (logit) per vocabulary word, then softmax turns the scores into probabilities:
```
scores:        attack 15.9, fuzzing 6.9, dos 5.7, normal 3.5, ...
probabilities: attack 100.0%, fuzzing 0.0%, ...
```

**Generation.** The model writes one word at a time, feeding each word back in:
```
'... steady =>'      -> 'attack'  (100.0%)
'... => attack'      -> 'fuzzing' (100.0%)
'... attack fuzzing' -> '<end>'   (100.0%)
```

## 5. The agent: perceive → think → act

| Step | Example (reading 3) |
|------|---------------------|
| **Perceive**: numbers → words | `{unknown_id_ratio: 0.22, dlc_mismatch_ratio: 0.09, ...}` → `ids unknown len wrong load low timing steady =>` |
| **Think**: the LLM completes the text | `attack fuzzing <end>` (this observation was **never seen in training**) |
| **Act**: verdict → action | `BLOCK: drop unknown IDs and malformed frames at the gateway` |

An agent built on Claude or GPT works the same way; only the size of the "think" model changes.

---

## How this compares to a real LLM

| | This model | GPT-2 small | Llama 3 8B |
|---|---|---|---|
| Weights | 5,011 | 124 million | 8 billion |
| Vocabulary | 19 words | 50,257 tokens | 128,256 tokens |
| Vector size | 16 | 768 | 4,096 |
| Transformer blocks | 2 | 12 | 32 |
| Attention heads per block | 1 | 12 | 32 |
| LayerNorm | No (kept simple) | Yes | Yes (RMSNorm) |

The parts are the same; only the sizes differ.

## Things to try

| Command | What you'll see |
|---------|-----------------|
| `python main.py --epochs 10` | Under-trained: loss stays at 1.19, the model says only "attack 63%", and it gets all 3 held-back tests wrong (it writes "attack" but not the type) |
| `python main.py --prompt "ids normal len ok load high timing steady =>"` | Trace a DoS observation: in block 2, `=>` puts **97% of its attention on "high"** (the flooded bus) |
| `python main.py --seed 3` | Different random starting weights: all the numbers differ, but it learns the same behaviour and gets every test right |

**A limit worth knowing:** there is only one spoofing example in the training
data. If it's held back, the model never learns the word "spoofing" for that
pattern. An LLM can only produce what its training taught it.

## Files

```
tinyllm/
  tokenizer.py   # words <-> token ids
  data.py        # the 16 sentences: CAN observation -> verdict
  model.py       # TinyGPT: embedding, attention, feed-forward, output, backprop, Adam, generation
  train.py       # training loop; 13 train / 3 held back
  agent.py       # SecurityAgent: perceive -> think (LLM) -> act
main.py          # prints sections 1-5
tests/           # 6 automated checks (pytest)
```

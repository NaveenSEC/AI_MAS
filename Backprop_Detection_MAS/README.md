# Backprop Detection MAS: 20 agents learning to detect CAN-bus attacks

A simple multi-agent system in which **20 detector agents**, each a small neural
network trained with **backpropagation**, watch simulated vehicle CAN-bus
traffic. A **coordinator** combines their votes into one decision: *normal* or
*attack*.

The networks are written with plain numpy, so every step of backpropagation is
visible in the code. There's no deep-learning framework hiding it. Every line
of code has a comment.

```
                         simulated CAN traffic (8 features per time window)
                                           │
      ┌──────────────┬──────────────┬──────┴───────┬─── … ───┬──────────────┐
      ▼              ▼              ▼              ▼         ▼              ▼
  Agent 1        Agent 2        Agent 3        Agent 4   …   Agent 20
  sees 3 of 8    sees 3 of 8    sees 3 of 8    …             sees 3 of 8
  features       features       features                     features
  TinyNet        TinyNet        TinyNet        …             TinyNet
  (backprop)     (backprop)     (backprop)                   (backprop)
      │ P(attack)    │              │              │         │              │
      └──────────────┴──────────────┴──────┬───────┴─── … ───┴──────────────┘
                                           ▼
                               Coordinator: average the 20
                               probabilities (soft vote)
                                           ▼
                               ≥ 0.5 → ATTACK, else normal
```

## How backpropagation works here

Each agent owns a `TinyNet` (`detection/network.py`):

```
inputs (3 features) ──W1,b1──▶ hidden layer (8 neurons, tanh) ──W2,b2──▶ output (sigmoid) = P(attack)
```

Training repeats four steps, once per **epoch**:

| Step | What happens | Code |
|------|--------------|------|
| 1. Forward pass | Inputs flow through the layers to give a probability for each window | `TinyNet.forward` |
| 2. Loss | Binary cross-entropy measures how wrong the probabilities are | `TinyNet.loss` |
| 3. **Backward pass** | The chain rule carries the error from the output back to every weight, giving each weight's gradient | `TinyNet.backward` |
| 4. Update | Every weight moves a small step against its gradient (gradient descent) | `TinyNet.step` |

The backward pass in step 3, written as maths (n = number of samples):

```
dz2 = (p − y) / n           error at the output (sigmoid + cross-entropy simplify to this)
dW2 = a1ᵀ · dz2             how much each hidden→output weight contributed
db2 = Σ dz2
da1 = dz2 · W2ᵀ             error sent back to the hidden layer through W2
dz1 = da1 ⊙ (1 − a1²)       through tanh (its slope is 1 − tanh²)
dW1 = xᵀ · dz1              how much each input→hidden weight contributed
db1 = Σ dz1
```

`gradient_check` proves the backward pass is correct. It nudges every weight up
and down, measures how the loss changes, and compares that with backprop's
answer. The two agree to about 1e-11.

## How the 20 agents make detection better

* **Different views:** each agent sees only 3 of the 8 features. An agent
  watching `jitter_ms` and `payload_change_rate` can spot spoofing; one watching
  only `msg_rate` and `bus_load_pct` can't, but it's good at DoS.
* **Different data:** each agent trains on its own random resample of the
  training data (bootstrap), so even agents with the same features learn
  slightly different networks.
* **Soft voting:** the coordinator averages the 20 probabilities. Individual
  mistakes cancel out, and the agents that do see the right signal pull the
  average the right way.

The traffic (`detection/data.py`) has normal windows plus three attacks:

| Traffic | What changes | Difficulty |
|---------|--------------|------------|
| DoS (flooding) | Message rate and bus load go up; gaps shrink; payloads repeat | Easy |
| Fuzzing | Random payloads, unknown CAN IDs, wrong message lengths | Easy |
| Spoofing (impersonating an ECU) | Only timing jitter and signal-change rate | Hard |

## Run it

```bash
cd Backprop_Detection_MAS
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

It takes about 2 seconds and needs no internet, API key or GPU.

### Output (default settings)

```
1. DATA: 2100 training windows, 900 test windows, 8 features each

2. GRADIENT CHECK: largest difference between backprop and numerical gradients = 1.3e-11 (OK)

3. BACKPROPAGATION IN ACTION: agent 1 watches unknown_id_ratio, dlc_mismatch_ratio, bus_load_pct
   epoch    1: loss 0.7351
   epoch   61: loss 0.4227
   ...
   epoch  300: loss 0.4025

4. EACH AGENT ALONE (test accuracy):
   agent  1:  80.1%  watches unknown_id_ratio, dlc_mismatch_ratio, bus_load_pct
   ...
   -> single agents range 76.6% to 92.3% (average 84.0%)

5. ALL 20 AGENTS COMBINED: test accuracy 95.6%
   normal    false-alarm rate:   0.2%
   dos       detection rate: 100.0%
   fuzzing   detection rate:  98.1%
   spoofing  detection rate:  78.0%

6. ONE SPOOFING WINDOW: 11 of 20 agents raised an alarm (...); average P(attack) = 0.62 -> ATTACK
```

How to read it:
- **Step 3:** the falling loss is backpropagation at work.
- **Steps 4–5:** the best single agent reaches 92.3% and the average agent 84.0%. The team of 20 reaches 95.6%.
- **Spoofing** is the hardest attack, because only agents watching jitter or signal changes can see it.

### Things to try

| Command | What you'll see (single agent avg → team) |
|---------|-------------------------------------------|
| `python main.py --features 1` | Agents with one feature each are weak alone, but the team does far better: 65.3% → 83.4% |
| `python main.py --features 8` | Every agent sees everything, so agents become near-identical and voting adds little: 97.3% → 97.6% |
| `python main.py --agents 5` | Fewer voters, lower combined accuracy: team 92.4% (vs 95.6% with 20) |
| `python main.py --epochs 10` | Too little backprop: loss only reaches 0.49 and the team drops to 89.4% |
| `python main.py --lr 0.01` | Learning rate too small: weights barely move, loss stays at 0.54 and the team drops to 86.8% |

### Run the tests

```bash
pytest
```

The tests check four things:
- Backprop gradients match numerical gradients.
- Training lowers the loss.
- The team has 20 agents with different views.
- The team beats the average agent.

## Files

```
detection/
  data.py          # simulated CAN traffic: normal, DoS, fuzzing, spoofing
  network.py       # TinyNet: forward pass, loss, backpropagation, update, gradient check
  agent.py         # DetectorAgent: picks features, resamples data, trains its TinyNet
  coordinator.py   # Coordinator: creates 20 agents, trains them, averages their votes
main.py            # runs everything and prints steps 1–6
tests/             # automated checks (pytest)
```

The data is simulated, so the numbers show how the method behaves; they are not
real-vehicle results. To use real traffic, compute the same 8 features from
logged CAN frames and pass them in place of `make_dataset()`.

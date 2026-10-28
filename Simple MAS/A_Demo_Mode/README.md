# A. Demo mode

The Simple MAS pipeline with **canned answers**, so you can watch the four agents
(Planner → Researcher → Writer → Reviewer) work together. It needs no API key,
no model and no internet connection, and it's free.

## Run

```bash
cd "Simple MAS/A_Demo_Mode"
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py "Explain tides"
```

You'll see each agent's output in turn, then the **FINAL** answer.

| Option | What it does |
|--------|--------------|
| `"your goal"` | The task for the agents. Default: *"Explain how solar panels work."* |
| `--max-revisions N` | How many times the Writer may revise after Reviewer feedback (default 2) |

## Test

```bash
pytest
```

## Files

```
mas/
  llm.py           # MockLLM: returns a fixed reply for each agent
  agents.py        # the four agent roles (system prompts)
  orchestrator.py  # plan → research → write → review/revise loop
main.py            # command-line entry point
tests/             # pipeline tests
```

When you want real answers, use [B. Claude via the API](../B_Claude_API) or
[C. Local model](../C_Local_Model).

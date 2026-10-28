# B. Claude via the API

The Simple MAS pipeline running on **Claude** (`claude-opus-5-5`) through the
Anthropic API. It gives the best answers, but it needs an API key and each run is billed.

## Run

1. Create an API key at https://console.anthropic.com.
2. Install and run:
   ```bash
   cd "Simple MAS/B_Claude_API"
   python -m venv .venv
   source .venv/bin/activate                  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   export ANTHROPIC_API_KEY=sk-ant-...        # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
   python main.py "Write a short guide to securing a CAN bus in a vehicle"
   ```

Each run makes 4 or more calls to Claude.

| Option | What it does |
|--------|--------------|
| `"your goal"` | The task for the agents. Default: *"Explain how solar panels work."* |
| `--model NAME` | Claude model to use (default `claude-opus-5-5`) |
| `--effort LEVEL` | `low`, `medium` (default), `high`, `xhigh` or `max`. Higher means more thorough, but slower and costlier |
| `--max-revisions N` | How many times the Writer may revise after Reviewer feedback (default 2) |

Requests use adaptive thinking and opt into server-side refusal fallbacks
(`fallbacks="default"`), so a declined request is retried on a recommended model.

## Test

```bash
pytest
```

The tests use a fake client, so they need no API key and cost nothing.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Authentication error | Check `ANTHROPIC_API_KEY` is set in the same terminal |
| `No module named 'anthropic'` | Activate the virtual environment and run `pip install -r requirements.txt` |

## Files

```
mas/
  llm.py           # ClaudeLLM: sends each agent's request to Claude
  agents.py        # the four agent roles (system prompts)
  orchestrator.py  # plan → research → write → review/revise loop
main.py            # command-line entry point
tests/             # pipeline and backend tests (fake client)
```

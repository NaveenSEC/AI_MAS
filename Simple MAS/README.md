# AI_MAS — a simple multi-agent system

Four agents collaborate on a goal, coordinated by an orchestrator:

```
goal ─▶ Planner ─▶ Researcher ─▶ Writer ─▶ Reviewer ─┬─▶ APPROVED ─▶ final answer
                                    ▲                 │
                                    └──── REVISE ─────┘   (bounded by --max-revisions)
```

| Agent      | Role                                                    |
|------------|---------------------------------------------------------|
| Planner    | Breaks the goal into 3–5 steps                          |
| Researcher | Lists key facts and considerations for the plan         |
| Writer     | Writes (and revises) the answer                         |
| Reviewer   | Approves the draft or sends back actionable feedback    |

Agents call Claude (`claude-opus-5-5`) through the official Anthropic SDK. A
`MockLLM` backend lets you run everything offline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or use `ant auth login`
```

## Run

```bash
python main.py "Write a short guide to composting at home"
python main.py --mock                  # offline, no API key needed
```

## Test

```bash
pytest
```

## Layout

```
mas/
  llm.py           # ClaudeLLM (Anthropic API) and MockLLM (offline)
  agents.py        # Agent roles / system prompts
  orchestrator.py  # Plan → research → write → review/revise loop
main.py            # CLI entry point
tests/             # Pipeline tests using the mock backend
```

Requests use adaptive thinking and opt into server-side refusal fallbacks
(`fallbacks="default"`), so a declined request is retried on a recommended model.



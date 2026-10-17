# Simple MAS — a simple multi-agent system

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

Agents can run on three backends:

| Backend | Flag | Needs |
|---------|------|-------|
| Claude (`claude-opus-5-5`) via the Anthropic API | *(default)* | An Anthropic API key; billed per use |
| Open-weights model on your own machine via [Ollama](https://ollama.com) | `--local` | Ollama installed and a model downloaded; free, works offline |
| Canned answers for a quick demo | `--mock` | Nothing |

## Getting started

**See [HOW_TO_RUN.md](HOW_TO_RUN.md)** for step-by-step setup, all three ways to
run it, command options and troubleshooting.

Quick try, once installed:

```bash
python main.py --mock "Explain tides"
```

## Layout

```
mas/
  llm.py           # ClaudeLLM (Anthropic API), OllamaLLM (local), MockLLM (canned)
  agents.py        # Agent roles / system prompts
  orchestrator.py  # Plan → research → write → review/revise loop
main.py            # CLI entry point
tests/             # Pipeline and backend tests (no API key or model needed)
HOW_TO_RUN.md      # Setup and run guide
```

Claude requests use adaptive thinking and opt into server-side refusal fallbacks
(`fallbacks="default"`), so a declined request is retried on a recommended model.

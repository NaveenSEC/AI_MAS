# Simple MAS: a simple multi-agent system

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

## Three projects, one for each way of running it

The agents and orchestrator are the same in all three. Only the *backend*,
the thing that produces the text, is different.

| Project | Backend | Needs | Cost |
|---------|---------|-------|------|
| [**A_Demo_Mode**](A_Demo_Mode) | Canned answers | Nothing | Free |
| [**B_Claude_API**](B_Claude_API) | Claude (`claude-opus-5-5`) via the Anthropic API | Anthropic API key | Billed per use |
| [**C_Local_Model**](C_Local_Model) | Open-weights model on your computer via [Ollama](https://ollama.com) | Ollama + a downloaded model | Free, works offline |

Each project is self-contained, with its own `main.py`, `requirements.txt`,
tests and README. **See [HOW_TO_RUN.md](HOW_TO_RUN.md)** to get started.

## Layout (same in every project)

```
<project>/
  mas/
    llm.py           # the backend (this is the only file that differs between projects)
    agents.py        # the four agent roles (system prompts)
    orchestrator.py  # plan → research → write → review/revise loop
  main.py            # command-line entry point
  tests/             # automated tests (no API key or model needed)
  requirements.txt
  README.md
```

Every line of code has a comment explaining what it does.

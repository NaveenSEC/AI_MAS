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

Agents can run on three backends:

| Backend | Flag | Needs |
|---------|------|-------|
| Claude (`claude-opus-5-5`) via the Anthropic API | *(default)* | An Anthropic API key; billed per use |
| Open-weights model on your own machine via [Ollama](https://ollama.com) | `--local` | Ollama installed and a model downloaded; free, works offline |
| Canned answers for a quick demo | `--mock` | Nothing |

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

## Run with a local model (no API key, no internet)

Claude itself can't be downloaded: Anthropic only offers it through the API.
To run everything on your own computer, use an open-weights model through Ollama.

1. **Install Ollama.**
   - macOS / Windows: download the installer from https://ollama.com/download
   - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
2. **Download a model** (one time; `llama3.2` is about 2 GB):
   ```bash
   ollama pull llama3.2
   ```
   Other choices: `llama3.2:1b` (about 1.3 GB, for machines with little RAM), or
   `qwen2.5:7b` / `llama3.1:8b` (about 5 GB, better answers, needs 8 GB+ RAM).
3. **Make sure Ollama is running.** The desktop app starts it automatically;
   on Linux run `ollama serve` in a separate terminal. Check with `ollama list`.
4. **Run the MAS on the local model:**
   ```bash
   python main.py --local "Write a short guide to securing a CAN bus in a vehicle"
   python main.py --local --model qwen2.5:7b "Explain secure boot"
   ```

If Ollama runs on another machine or port, set `OLLAMA_HOST`, for example
`export OLLAMA_HOST=http://192.168.1.20:11434`. Small local models are less
reliable than Claude, so the Reviewer may ask for more revisions.

## Test

```bash
pytest
```

## Layout

```
mas/
  llm.py           # ClaudeLLM (Anthropic API), OllamaLLM (local), MockLLM (canned)
  agents.py        # Agent roles / system prompts
  orchestrator.py  # Plan → research → write → review/revise loop
main.py            # CLI entry point
tests/             # Pipeline tests using the mock backend
```

Requests use adaptive thinking and opt into server-side refusal fallbacks
(`fallbacks="default"`), so a declined request is retried on a recommended model.



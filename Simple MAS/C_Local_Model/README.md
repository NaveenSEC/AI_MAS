# C. Local model on your computer

The Simple MAS pipeline running on a free **open-weights model** (such as Llama)
on your own computer, through [Ollama](https://ollama.com). It needs no API key,
costs nothing, and works offline once the model is downloaded.

Claude itself can't be downloaded: Anthropic only offers it through the API (see
[B. Claude via the API](../B_Claude_API)).

## Run

1. **Install Ollama.**
   - macOS / Windows: download the installer from https://ollama.com/download
   - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
2. **Download a model** (one time):
   ```bash
   ollama pull llama3.2
   ```
   | Model | Download | Notes |
   |-------|----------|-------|
   | `llama3.2:1b` | ~1.3 GB | For machines with little RAM |
   | `llama3.2` | ~2 GB | Default; good balance |
   | `qwen2.5:7b` / `llama3.1:8b` | ~5 GB | Better answers, needs 8 GB+ RAM |
3. **Make sure Ollama is running.** The desktop app starts it automatically. On Linux,
   run `ollama serve` in a separate terminal. `ollama list` shows your downloaded models.
4. **Install and run the project:**
   ```bash
   cd "Simple MAS/C_Local_Model"
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python main.py "Write a short guide to securing a CAN bus in a vehicle"
   python main.py --model qwen2.5:7b "Explain secure boot"
   ```

| Option | What it does |
|--------|--------------|
| `"your goal"` | The task for the agents. Default: *"Explain how solar panels work."* |
| `--model NAME` | Which downloaded Ollama model to use (default `llama3.2`) |
| `--max-revisions N` | How many times the Writer may revise after Reviewer feedback (default 2) |

If Ollama runs on another machine or port, set `OLLAMA_HOST`, for example
`export OLLAMA_HOST=http://192.168.1.20:11434`.

Small local models give weaker answers than Claude, so expect the Reviewer to ask for more revisions.

## Test

```bash
pytest
```

The tests use a fake Ollama server, so they need no model download.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Could not reach Ollama at http://localhost:11434` | Start Ollama (`ollama serve` or open the app), then `ollama pull <model>` |
| `model "..." not found` | Download it first: `ollama pull <model>` |

## Files

```
mas/
  llm.py           # OllamaLLM: sends each agent's request to Ollama
  agents.py        # the four agent roles (system prompts)
  orchestrator.py  # plan → research → write → review/revise loop
main.py            # command-line entry point
tests/             # pipeline and backend tests (fake server)
```

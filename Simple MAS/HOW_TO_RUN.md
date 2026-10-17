# How to run Simple MAS

The multi-agent system can run three ways. Pick one:

| Option | Command flag | Needs | Cost |
|--------|--------------|-------|------|
| [A. Demo mode](#a-demo-mode-no-setup) | `--mock` | Nothing | Free |
| [B. Claude via the API](#b-run-with-claude-anthropic-api) | *(none)* | Anthropic API key | Billed per use |
| [C. Local model on your computer](#c-run-with-a-local-model-ollama) | `--local` | Ollama + a downloaded model | Free, works offline |

---

## 1. Get the code

```bash
git clone https://github.com/NaveenSEC/AI_MAS.git
cd AI_MAS
git checkout claude/optimistic-planck-fag80e
cd "Simple MAS"
```

The folder name has a space, so keep the quotes around `"Simple MAS"`.

## 2. Install

Needs Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## A. Demo mode (no setup)

Runs with canned answers so you can see the flow without an API key or a model:

```bash
python main.py --mock "Explain tides"
```

You'll see each agent in turn (Planner → Researcher → Writer → Reviewer), then the **FINAL** answer.

## B. Run with Claude (Anthropic API)

1. Create an API key at https://console.anthropic.com.
2. Set it in your terminal:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...           # macOS / Linux
   $env:ANTHROPIC_API_KEY="sk-ant-..."            # Windows PowerShell
   ```
3. Run:
   ```bash
   python main.py "Write a short guide to securing a CAN bus in a vehicle"
   ```

Each run makes 4 or more calls to `claude-opus-5-5`, billed to your API account.

## C. Run with a local model (Ollama)

Claude itself can't be downloaded: Anthropic only offers it through the API. To run
everything on your own computer, use a free open-weights model through
[Ollama](https://ollama.com).

1. **Install Ollama.**
   - macOS / Windows: download the installer from https://ollama.com/download
   - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
2. **Download a model** (one time):
   ```bash
   ollama pull llama3.2          # about 2 GB (default model)
   ```
   | Model | Download | Notes |
   |-------|----------|-------|
   | `llama3.2:1b` | ~1.3 GB | For machines with little RAM |
   | `llama3.2` | ~2 GB | Default; good balance |
   | `qwen2.5:7b` / `llama3.1:8b` | ~5 GB | Better answers, needs 8 GB+ RAM |
3. **Make sure Ollama is running.** The desktop app starts it automatically; on Linux
   run `ollama serve` in a separate terminal. `ollama list` shows your downloaded models.
4. **Run the MAS on the local model:**
   ```bash
   python main.py --local "Write a short guide to securing a CAN bus in a vehicle"
   python main.py --local --model qwen2.5:7b "Explain secure boot"
   ```

If Ollama runs on another machine or port, set `OLLAMA_HOST`:

```bash
export OLLAMA_HOST=http://192.168.1.20:11434
```

Small local models give weaker answers than Claude, so expect the Reviewer to ask for
more revisions.

---

## Command options

| Option | What it does |
|--------|--------------|
| `"your goal"` | The task for the agents. Default: *"Explain how solar panels work."* |
| `--mock` | Demo mode with canned answers |
| `--local` | Use a local model through Ollama |
| `--model NAME` | Which Ollama model to use with `--local` (default `llama3.2`) |
| `--max-revisions N` | How many times the Writer may revise after Reviewer feedback (default 2) |

## Run the tests

```bash
pytest
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Could not reach Ollama at http://localhost:11434` | Start Ollama (`ollama serve` or open the app), then `ollama pull <model>` |
| `model "..." not found` from Ollama | Download it first: `ollama pull <model>` |
| Authentication error with Claude | Check `ANTHROPIC_API_KEY` is set in the same terminal |
| `No module named 'anthropic'` | Activate the virtual environment and run `pip install -r requirements.txt` |
| `cd: too many arguments` | Quote the folder: `cd "Simple MAS"` |

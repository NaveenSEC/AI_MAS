# How to run Simple MAS

Simple MAS comes as three separate projects. Pick one:

| Project | Use it when | Needs | Cost |
|---------|-------------|-------|------|
| [A. Demo mode](A_Demo_Mode/README.md) | You want to see how the agents work together | Nothing | Free |
| [B. Claude via the API](B_Claude_API/README.md) | You want the best answers | Anthropic API key | Billed per use |
| [C. Local model on your computer](C_Local_Model/README.md) | You want it free, private and offline | Ollama + a downloaded model | Free |

---

## 1. Get the code (all projects)

```bash
git clone https://github.com/NaveenSEC/AI_MAS.git
cd AI_MAS
git checkout claude/optimistic-planck-fag80e
cd "Simple MAS"
```

The folder name has a space, so keep the quotes around `"Simple MAS"`.

## 2. Go into the project you picked and install it

Each project needs Python 3.10 or newer.

```bash
cd A_Demo_Mode                   # or B_Claude_API, or C_Local_Model
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run it

| Project | Extra setup | Command |
|---------|-------------|---------|
| A. Demo mode | None | `python main.py "Explain tides"` |
| B. Claude | `export ANTHROPIC_API_KEY=sk-ant-...` (key from https://console.anthropic.com) | `python main.py "Explain secure boot"` |
| C. Local model | Install Ollama, then `ollama pull llama3.2` | `python main.py "Explain secure boot"` |

The README in each project folder has the full steps, every option, and troubleshooting.

## 4. Run the tests

From inside any project folder:

```bash
pytest
```

The tests use fakes, so none of them needs an API key, Ollama or the internet.

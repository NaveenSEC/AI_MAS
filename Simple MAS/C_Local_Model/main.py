"""CLI entry point (local model): python main.py "your goal" [--model NAME]

This is the file you run from the terminal. It reads the command-line options,
runs the agents on a model in Ollama and prints the final answer.
Install Ollama and run `ollama pull llama3.2` before running it.
"""

# argparse: Python's built-in tool for reading command-line options.
import argparse

# The local backend and the Orchestrator, from the `mas` package (the mas/ folder).
from mas import OllamaLLM, Orchestrator

# The default local model name, so the --help text shows it.
from mas.llm import LOCAL_MODEL


# All the program's work happens inside this function.
def main() -> None:
    # Create the option reader. `description` appears at the top of `python main.py --help`.
    parser = argparse.ArgumentParser(description="Run the multi-agent pipeline on a local model.")
    # The goal, e.g. "Explain tides". nargs="?" makes it optional,
    # in which case the default below is used.
    parser.add_argument("goal", nargs="?", default="Explain how solar panels work.")
    # --model NAME: which downloaded Ollama model to use.
    parser.add_argument(
        "--model", default=LOCAL_MODEL, help=f"Ollama model (default: {LOCAL_MODEL})."
    )
    # --max-revisions N: how many times the Writer may revise. type=int turns "3" into 3.
    parser.add_argument("--max-revisions", type=int, default=2)
    # Read the actual command line. Options become attributes, e.g. args.model.
    args = parser.parse_args()

    # Create the local backend with the chosen model.
    llm = OllamaLLM(model=args.model)
    # Create the Orchestrator with the backend and run it on the goal.
    # This is where all the agents do their work.
    result = Orchestrator(llm, max_revisions=args.max_revisions).run(args.goal)

    # Pick a word or phrase describing how the run ended.
    status = "approved" if result.approved else "not approved after max revisions"
    # Print the final draft under a heading.
    print(f"\n=== FINAL ({status}) ===\n{result.draft}")


# This is True only when the file is run directly (python main.py),
# not when another file imports it. That stops main() from running on import.
if __name__ == "__main__":
    # Start the program.
    main()

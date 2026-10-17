"""CLI entry point: python main.py "your goal" [--mock | --local [--model NAME]]

This is the file you run from the terminal. It reads the command-line options,
picks a backend, runs the agents and prints the final answer.
"""

# argparse: Python's built-in tool for reading command-line options like --mock.
import argparse

# The three backends and the Orchestrator, from the `mas` package (the mas/ folder).
from mas import ClaudeLLM, MockLLM, OllamaLLM, Orchestrator

# The default local model name, so the --help text shows it.
from mas.llm import LOCAL_MODEL


# All the program's work happens inside this function.
def main() -> None:
    # Create the option reader. `description` appears at the top of `python main.py --help`.
    parser = argparse.ArgumentParser(description="Run the multi-agent pipeline.")
    # The goal, e.g. "Explain tides". nargs="?" makes it optional,
    # in which case the default below is used.
    parser.add_argument("goal", nargs="?", default="Explain how solar panels work.")
    # A group of options where at most one may be given:
    # you can't ask for --mock and --local at the same time.
    backend = parser.add_mutually_exclusive_group()
    # --mock: a yes/no switch (store_true means "True if the flag is present").
    backend.add_argument("--mock", action="store_true", help="Run offline with canned answers.")
    # --local: a yes/no switch for using Ollama.
    backend.add_argument("--local", action="store_true", help="Use a local model via Ollama.")
    # --model NAME: which Ollama model to use with --local.
    parser.add_argument(
        "--model", default=LOCAL_MODEL, help=f"Ollama model for --local (default: {LOCAL_MODEL})."
    )
    # --max-revisions N: how many times the Writer may revise. type=int turns "3" into 3.
    parser.add_argument("--max-revisions", type=int, default=2)
    # Read the actual command line. Options become attributes, e.g. args.mock, args.max_revisions.
    args = parser.parse_args()

    # Choose the backend from the flags.
    if args.mock:
        # --mock: canned answers, no model needed.
        llm = MockLLM()
    elif args.local:
        # --local: a model running in Ollama on this computer.
        llm = OllamaLLM(model=args.model)
    else:
        # No flag: Claude through the Anthropic API (needs ANTHROPIC_API_KEY).
        llm = ClaudeLLM()
    # Create the Orchestrator with the chosen backend and run it on the goal.
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

"""CLI entry point (Demo mode): python main.py "your goal"

This is the file you run from the terminal. It reads the command-line options,
runs the agents with canned answers and prints the final answer.
"""

# argparse: Python's built-in tool for reading command-line options.
import argparse

# The Demo backend and the Orchestrator, from the `mas` package (the mas/ folder).
from mas import MockLLM, Orchestrator


# All the program's work happens inside this function.
def main() -> None:
    # Create the option reader. `description` appears at the top of `python main.py --help`.
    parser = argparse.ArgumentParser(description="Run the multi-agent pipeline in demo mode.")
    # The goal, e.g. "Explain tides". nargs="?" makes it optional,
    # in which case the default below is used.
    parser.add_argument("goal", nargs="?", default="Explain how solar panels work.")
    # --max-revisions N: how many times the Writer may revise. type=int turns "3" into 3.
    parser.add_argument("--max-revisions", type=int, default=2)
    # Read the actual command line. Options become attributes, e.g. args.goal.
    args = parser.parse_args()

    # Demo mode always uses canned answers.
    llm = MockLLM()
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

"""CLI entry point: python main.py "your goal" [--mock | --local [--model NAME]]"""

import argparse

from mas import ClaudeLLM, MockLLM, OllamaLLM, Orchestrator
from mas.llm import LOCAL_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent pipeline.")
    parser.add_argument("goal", nargs="?", default="Explain how solar panels work.")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--mock", action="store_true", help="Run offline with canned answers.")
    backend.add_argument("--local", action="store_true", help="Use a local model via Ollama.")
    parser.add_argument(
        "--model", default=LOCAL_MODEL, help=f"Ollama model for --local (default: {LOCAL_MODEL})."
    )
    parser.add_argument("--max-revisions", type=int, default=2)
    args = parser.parse_args()

    if args.mock:
        llm = MockLLM()
    elif args.local:
        llm = OllamaLLM(model=args.model)
    else:
        llm = ClaudeLLM()
    result = Orchestrator(llm, max_revisions=args.max_revisions).run(args.goal)

    status = "approved" if result.approved else "not approved after max revisions"
    print(f"\n=== FINAL ({status}) ===\n{result.draft}")


if __name__ == "__main__":
    main()

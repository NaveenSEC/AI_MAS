"""CLI entry point: python main.py "your goal" [--mock]"""

import argparse

from mas import ClaudeLLM, MockLLM, Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-agent pipeline.")
    parser.add_argument("goal", nargs="?", default="Explain how solar panels work.")
    parser.add_argument("--mock", action="store_true", help="Run offline without the API.")
    parser.add_argument("--max-revisions", type=int, default=2)
    args = parser.parse_args()

    llm = MockLLM() if args.mock else ClaudeLLM()
    result = Orchestrator(llm, max_revisions=args.max_revisions).run(args.goal)

    status = "approved" if result.approved else "not approved after max revisions"
    print(f"\n=== FINAL ({status}) ===\n{result.draft}")


if __name__ == "__main__":
    main()

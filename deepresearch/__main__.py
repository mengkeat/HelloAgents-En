"""Command-line entry point for the deep research agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import DeepResearchAgent
from .config import Configuration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a deep research report on a topic using web search and LLM synthesis.",
    )
    parser.add_argument("topic", help="Research topic to investigate")
    parser.add_argument(
        "--search",
        dest="search_api",
        choices=["duckduckgo", "serpapi"],
        default=None,
        help="Search backend to use (default: duckduckgo)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file for the Markdown report",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output full JSON result instead of Markdown only",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    overrides = {}
    if args.search_api:
        overrides["search_api"] = args.search_api

    config = Configuration.from_env(overrides=overrides)
    agent = DeepResearchAgent(config=config)
    result = agent.run(args.topic)

    if args.json:
        import json
        from .models import SummaryStateOutput

        output = json.dumps({
            "running_summary": result.running_summary,
            "report_markdown": result.report_markdown,
            "todo_items": [
                {
                    "id": t.id,
                    "title": t.title,
                    "intent": t.intent,
                    "query": t.query,
                    "status": t.status,
                    "summary": t.summary,
                    "sources_summary": t.sources_summary,
                }
                for t in result.todo_items
            ],
        }, ensure_ascii=False, indent=2)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
            print(f"\nJSON output written to {args.output}")
        else:
            print(output)
    else:
        report = result.report_markdown or result.running_summary or ""

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n", encoding="utf-8")
            print(f"\nReport written to {args.output}")
        else:
            print("\n" + report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

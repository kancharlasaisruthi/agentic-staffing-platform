"""
CLI runner — run the full agentic pipeline once from the command line,
without Streamlit. Useful for debugging, testing with real API keys, or
scripting/cron use.

Usage:
    python run_cli.py
    python run_cli.py --config config/icp_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from config.schema import RunConfig  # noqa: E402
from core.graph import build_graph, run_graph  # noqa: E402
from core.state import empty_state  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the staffing prospect intelligence pipeline once.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config" / "icp_config.yaml"),
        help="Path to the ICP config YAML file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the final state as JSON (defaults to printing a summary only).",
    )
    args = parser.parse_args()

    import os

    if not os.environ.get("TAVILY_API_KEY") or not os.environ.get("GROQ_API_KEY"):
        print("ERROR: TAVILY_API_KEY and/or GROQ_API_KEY not set. Copy .env.example to .env and fill them in.")
        sys.exit(1)

    cfg = RunConfig.from_yaml(args.config)
    print(f"Loaded config for industry={cfg.industry!r}, hiring_threshold={cfg.hiring_threshold}, "
          f"max_companies={cfg.max_companies}")

    graph = build_graph()
    state = empty_state(cfg.model_dump())

    print("Invoking graph (this may take a few minutes depending on company count)...")
    result = run_graph(graph, state)

    print("\n=== RUN SUMMARY ===")
    print(f"Discovered companies: {len(result.get('discovered_companies', []))}")
    print(f"Qualified prospects:  {len(result.get('qualification', {}))}")
    print(f"Errors encountered:   {len(result.get('errors', []))}")
    print(f"Planner done:         {result.get('done')}")

    print("\n=== TOP PROSPECTS ===")
    company_by_key = {c["key"]: c for c in result.get("validated_companies", [])}
    ranked = sorted(result.get("qualification", {}).items(), key=lambda kv: -kv[1].get("score", 0))
    for key, qual in ranked[:10]:
        name = company_by_key.get(key, {}).get("name", key)
        rec = result.get("recommendations", {}).get(key, {})
        print(f"- {name}: {qual['score']}/100 ({qual['tier']}) — {rec.get('recommended_outreach', '')}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull state written to {args.output}")


if __name__ == "__main__":
    main()

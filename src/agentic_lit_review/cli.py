from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_lit_review.graph import LiteratureReviewPipeline
from agentic_lit_review.llm import LLMClient, load_env
from agentic_lit_review.state import AgentState


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the reconstructed agentic literature review pipeline."
    )
    parser.add_argument("topic", help="Research topic, for example: zero-shot learning")
    parser.add_argument("--max-results", type=int, default=5, help="Results per query per source.")
    parser.add_argument("--min-papers", type=int, default=3, help="Minimum screened papers before synthesis.")
    parser.add_argument("--max-search-iterations", type=int, default=2)
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic heuristic agents only.")
    parser.add_argument("--json", type=Path, help="Write final state to a JSON file.")
    args = parser.parse_args(argv)

    load_env()
    llm = LLMClient(enabled=not args.no_llm)
    pipeline = LiteratureReviewPipeline(
        llm=llm,
        max_results_per_query=args.max_results,
        min_screened_papers=args.min_papers,
        max_search_iterations=args.max_search_iterations,
    )
    state = pipeline.run(args.topic)
    print_report(state, llm_enabled=llm.enabled)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(_jsonable(state), indent=2), encoding="utf-8")
        print(f"\nSaved JSON output to {args.json}")
    return 0


def print_report(state: AgentState, llm_enabled: bool) -> None:
    print("\nAgentic Literature Review")
    print("=" * 28)
    print(f"Topic: {state['research_topic']}")
    print(f"LLM mode: {'Groq' if llm_enabled else 'heuristic fallback'}")
    print(f"Queries: {', '.join(state.get('search_queries', [])) or 'none'}")
    print(f"Raw papers: {len(state.get('raw_papers', []))}")
    print(f"Screened papers: {len(state.get('screened_papers', []))}")

    if state.get("errors"):
        print("\nRetrieval warnings:")
        for error in state["errors"][:5]:
            print(f"- {error}")

    print("\nTop screened papers:")
    for index, paper in enumerate(state.get("screened_papers", [])[:10], start=1):
        year = paper.year if paper.year else "n.d."
        print(f"{index}. [{paper.relevance:.2f}] {paper.title} ({year}, {paper.source})")

    print("\nThemes:")
    for theme in state.get("themes", []):
        print(f"- {theme}")

    print("\nResearch gaps:")
    for gap in state.get("research_gaps", []):
        print(f"- {gap}")

    print("\nResearch plan:")
    for index, step in enumerate(state.get("research_plan", []), start=1):
        print(f"{index}. {step}")

    print("\nRecommended reading order:")
    for index, item in enumerate(state.get("reading_order", []), start=1):
        print(f"{index}. {item['title']}")
        print(f"   Reason: {item['reason']}")
        if item.get("url"):
            print(f"   URL: {item['url']}")


def _jsonable(state: AgentState) -> dict:
    output = dict(state)
    output["raw_papers"] = [paper.to_dict() for paper in state.get("raw_papers", [])]
    output["screened_papers"] = [paper.to_dict() for paper in state.get("screened_papers", [])]
    return output

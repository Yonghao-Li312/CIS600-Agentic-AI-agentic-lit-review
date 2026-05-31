from __future__ import annotations

from typing import Any

from agentic_lit_review.state import AgentState


def evaluate_state(state: AgentState) -> dict[str, Any]:
    raw = state.get("raw_papers", [])
    screened = state.get("screened_papers", [])
    themes = state.get("themes", [])
    gaps = state.get("research_gaps", [])
    plan = state.get("research_plan", [])
    order = state.get("reading_order", [])

    source_counts: dict[str, int] = {}
    for paper in raw:
        source_counts[paper.source] = source_counts.get(paper.source, 0) + 1

    scores = [paper.relevance for paper in screened]
    return {
        "pipeline_completion": {
            "search": bool(raw),
            "screening": bool(screened),
            "synthesis": bool(themes),
            "planning": bool(gaps and plan and order),
        },
        "retrieval": {
            "raw_papers": len(raw),
            "screened_papers": len(screened),
            "retention_rate": round(len(screened) / len(raw), 3) if raw else 0.0,
            "source_counts": source_counts,
            "source_diversity": len(source_counts),
        },
        "screening": {
            "mean_relevance": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "max_relevance": round(max(scores), 3) if scores else 0.0,
            "high_relevance_count": sum(score >= 0.8 for score in scores),
        },
        "outputs": {
            "theme_count": len(themes),
            "gap_count": len(gaps),
            "plan_step_count": len(plan),
            "reading_order_count": len(order),
        },
    }


def print_evaluation(metrics: dict[str, Any]) -> None:
    print("\nEvaluation")
    print("=" * 10)
    completion = metrics["pipeline_completion"]
    for step, passed in completion.items():
        print(f"{step.title()}: {'PASS' if passed else 'FAIL'}")

    retrieval = metrics["retrieval"]
    print(f"Raw papers: {retrieval['raw_papers']}")
    print(f"Screened papers: {retrieval['screened_papers']}")
    print(f"Retention rate: {retrieval['retention_rate']}")
    print(f"Source diversity: {retrieval['source_diversity']}")

    screening = metrics["screening"]
    print(f"Mean relevance: {screening['mean_relevance']}")
    print(f"High relevance papers: {screening['high_relevance_count']}")

from __future__ import annotations

from typing import TypedDict

from agentic_lit_review.models import Paper


class AgentState(TypedDict, total=False):
    research_topic: str
    search_queries: list[str]
    raw_papers: list[Paper]
    screened_papers: list[Paper]
    themes: list[str]
    research_gaps: list[str]
    research_plan: list[str]
    reading_order: list[dict[str, str]]
    search_iterations: int
    errors: list[str]

from __future__ import annotations

import re
import time

from agentic_lit_review.llm import LLMClient
from agentic_lit_review.models import Paper
from agentic_lit_review.retrievers import ArxivRetriever, SemanticScholarRetriever
from agentic_lit_review.retrievers.dedupe import dedupe_papers
from agentic_lit_review.state import AgentState


class SearchAgent:
    def __init__(
        self,
        llm: LLMClient,
        retrievers: list[object] | None = None,
        max_results_per_query: int = 5,
        request_pause_seconds: float = 1.0,
    ) -> None:
        self.llm = llm
        self.retrievers = retrievers or [ArxivRetriever(), SemanticScholarRetriever()]
        self.max_results_per_query = max_results_per_query
        self.request_pause_seconds = request_pause_seconds

    def __call__(self, state: AgentState) -> AgentState:
        topic = state["research_topic"]
        iteration = state.get("search_iterations", 0)
        queries = self.generate_queries(topic, iteration=iteration)

        all_papers = list(state.get("raw_papers", []))
        errors = list(state.get("errors", []))
        for query in queries:
            for retriever in self.retrievers:
                try:
                    all_papers.extend(retriever.search(query, self.max_results_per_query))
                    time.sleep(self.request_pause_seconds)
                except Exception as exc:
                    errors.append(f"{retriever.__class__.__name__} failed for '{query}': {exc}")

        state["search_queries"] = list(dict.fromkeys(state.get("search_queries", []) + queries))
        state["raw_papers"] = dedupe_papers(all_papers)
        state["search_iterations"] = iteration + 1
        state["errors"] = errors
        return state

    def generate_queries(self, topic: str, iteration: int = 0) -> list[str]:
        if self.llm.enabled:
            prompt = (
                "Generate exactly 3 diverse academic search queries for this research topic. "
                "Return only a numbered list.\n\n"
                f"Topic: {topic}"
            )
            try:
                response = self.llm.complete(prompt)
                queries = self._parse_queries(response)
                if queries:
                    return queries[:3]
            except Exception:
                pass

        if iteration <= 0:
            return [topic, f"{topic} survey", f"{topic} recent methods"]
        return [f"{topic} applications", f"{topic} benchmark", f"{topic} challenges"]

    @staticmethod
    def _parse_queries(text: str) -> list[str]:
        queries: list[str] = []
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*[-*]?\s*\d*[\).:-]?\s*", "", line).strip()
            cleaned = cleaned.strip('"')
            if cleaned and len(cleaned) > 2:
                queries.append(cleaned)
        return queries

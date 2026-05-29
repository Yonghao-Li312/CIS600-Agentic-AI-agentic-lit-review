from __future__ import annotations

from agentic_lit_review.llm import LLMClient, parse_json_object
from agentic_lit_review.models import Paper
from agentic_lit_review.state import AgentState


class PlanningAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> AgentState:
        papers = state.get("screened_papers", [])
        if not papers:
            state["research_gaps"] = []
            state["research_plan"] = []
            state["reading_order"] = []
            return state

        if self.llm.enabled:
            try:
                response = self.llm.complete(self._prompt(state))
                parsed = parse_json_object(response)
                state["research_gaps"] = [str(item) for item in parsed.get("research_gaps", [])]
                state["research_plan"] = [str(item) for item in parsed.get("research_plan", [])]
                state["reading_order"] = self._normalize_order(parsed.get("reading_order", []), papers)
                if state["reading_order"]:
                    return state
            except Exception:
                pass

        state["research_gaps"] = self._heuristic_gaps(state["research_topic"], state.get("themes", []))
        state["research_plan"] = self._heuristic_plan(state["research_topic"])
        state["reading_order"] = self._heuristic_order(papers)
        return state

    @staticmethod
    def _prompt(state: AgentState) -> str:
        papers = state.get("screened_papers", [])
        blocks = []
        for index, paper in enumerate(papers):
            blocks.append(
                f"[{index}] {paper.title} ({paper.year or 'n.d.'})\n"
                f"Summary: {paper.summary or paper.abstract[:300]}"
            )
        return (
            "Generate a research plan and a pedagogical reading order. "
            "Prefer surveys and foundations first, then methods, then specialized or recent extensions. "
            "Return strict JSON with keys 'research_gaps', 'research_plan', and 'reading_order'. "
            "'reading_order' must be a list of objects with 'paper_index' and 'reason'.\n\n"
            f"Topic: {state['research_topic']}\n"
            f"Themes: {state.get('themes', [])}\n\n"
            + "\n\n".join(blocks)
        )

    @staticmethod
    def _normalize_order(items: list, papers: list[Paper]) -> list[dict[str, str]]:
        order: list[dict[str, str]] = []
        seen: set[int] = set()
        for item in items:
            try:
                index = int(item.get("paper_index"))
            except Exception:
                continue
            if index < 0 or index >= len(papers) or index in seen:
                continue
            seen.add(index)
            order.append(
                {
                    "title": papers[index].title,
                    "reason": str(item.get("reason", "Useful next reading step.")),
                    "url": papers[index].url,
                }
            )
        for index, paper in enumerate(papers):
            if index not in seen:
                order.append({"title": paper.title, "reason": "Additional relevant paper.", "url": paper.url})
        return order

    @staticmethod
    def _heuristic_order(papers: list[Paper]) -> list[dict[str, str]]:
        def key(paper: Paper) -> tuple[int, int, float]:
            title = paper.title.lower()
            is_survey = 0 if any(word in title for word in ["survey", "review", "tutorial"]) else 1
            year = paper.year if paper.year is not None else 9999
            return (is_survey, year, -paper.relevance)

        ordered = sorted(papers, key=key)
        result: list[dict[str, str]] = []
        for position, paper in enumerate(ordered, start=1):
            if position == 1:
                reason = "Start here because it appears broad, foundational, or highly relevant."
            elif paper.year and paper.year < 2018:
                reason = "Read early to understand older foundations before recent extensions."
            else:
                reason = "Read after the foundation to build toward newer methods and applications."
            result.append({"title": paper.title, "reason": reason, "url": paper.url})
        return result

    @staticmethod
    def _heuristic_gaps(topic: str, themes: list[str]) -> list[str]:
        if themes:
            return [
                f"Need human validation of relevance and reading-order quality for {topic}.",
                "Need full-text analysis beyond abstracts to improve synthesis depth.",
                "Need stronger coverage checks across venues, years, and related terminology.",
            ]
        return [f"Need a larger and more diverse paper set before drawing conclusions about {topic}."]

    @staticmethod
    def _heuristic_plan(topic: str) -> list[str]:
        return [
            f"Map core definitions and subproblems in {topic}.",
            "Compare foundational methods, datasets, and evaluation criteria.",
            "Identify recent extensions and unresolved limitations.",
            "Validate the proposed reading order with domain experts or course staff.",
        ]

from __future__ import annotations

from agentic_lit_review.llm import LLMClient, parse_json_object
from agentic_lit_review.models import Paper
from agentic_lit_review.state import AgentState
from agentic_lit_review.text_utils import tokenize


class ScreeningAgent:
    def __init__(self, llm: LLMClient, threshold: float = 0.5) -> None:
        self.llm = llm
        self.threshold = threshold

    def __call__(self, state: AgentState) -> AgentState:
        topic = state["research_topic"]
        papers = state.get("raw_papers", [])
        scores = self.score_papers(topic, papers)

        screened: list[Paper] = []
        for index, paper in enumerate(papers):
            paper.relevance = scores.get(index, 0.5)
            if paper.relevance >= self.threshold:
                screened.append(paper)

        screened.sort(key=lambda paper: paper.relevance, reverse=True)
        state["screened_papers"] = screened
        return state

    def score_papers(self, topic: str, papers: list[Paper]) -> dict[int, float]:
        if not papers:
            return {}
        if self.llm.enabled:
            try:
                prompt = self._score_prompt(topic, papers)
                response = self.llm.complete(prompt)
                parsed = parse_json_object(response)
                scores = parsed.get("scores", {})
                return {int(key): float(value) for key, value in scores.items()}
            except Exception:
                pass
        return {index: self._heuristic_score(topic, paper) for index, paper in enumerate(papers)}

    @staticmethod
    def _score_prompt(topic: str, papers: list[Paper]) -> str:
        paper_blocks = []
        for index, paper in enumerate(papers):
            paper_blocks.append(f"[{index}]\n{paper.brief(max_abstract_chars=700)}")
        return (
            "Score each paper's relevance to the topic from 0.0 to 1.0. "
            "Return strict JSON in this form: {\"scores\": {\"0\": 0.82}}.\n\n"
            f"Topic: {topic}\n\n"
            + "\n\n".join(paper_blocks)
        )

    @staticmethod
    def _heuristic_score(topic: str, paper: Paper) -> float:
        topic_terms = set(tokenize(topic))
        if not topic_terms:
            return 0.5
        title_terms = set(tokenize(paper.title))
        abstract_terms = set(tokenize(paper.abstract))
        title_overlap = len(topic_terms & title_terms) / len(topic_terms)
        abstract_overlap = len(topic_terms & abstract_terms) / len(topic_terms)
        score = 0.35 + (0.45 * title_overlap) + (0.25 * abstract_overlap)
        if any(term in paper.title.lower() for term in ["survey", "review", "tutorial"]):
            score += 0.05
        return round(max(0.0, min(score, 1.0)), 2)

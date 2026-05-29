from __future__ import annotations

from agentic_lit_review.llm import LLMClient, parse_json_object
from agentic_lit_review.state import AgentState
from agentic_lit_review.text_utils import first_sentence, top_terms


class SynthesisAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> AgentState:
        papers = state.get("screened_papers", [])
        if not papers:
            state["themes"] = []
            return state

        if self.llm.enabled:
            try:
                response = self.llm.complete(self._prompt(state["research_topic"], papers))
                parsed = parse_json_object(response)
                themes = [str(item) for item in parsed.get("themes", [])]
                summaries = parsed.get("summaries", {})
                for index, paper in enumerate(papers):
                    paper.summary = str(summaries.get(str(index), "")).strip() or first_sentence(paper.abstract)
                state["themes"] = themes[:5]
                return state
            except Exception:
                pass

        for paper in papers:
            paper.summary = first_sentence(paper.abstract)
        terms = top_terms([paper.title + " " + paper.abstract for paper in papers], limit=8)
        state["themes"] = self._themes_from_terms(terms)
        return state

    @staticmethod
    def _prompt(topic: str, papers: list) -> str:
        blocks = [f"[{index}]\n{paper.brief()}" for index, paper in enumerate(papers)]
        return (
            "Create concise literature review synthesis for the topic. "
            "Return strict JSON with keys 'summaries' and 'themes'. "
            "'summaries' maps each paper index to one sentence. "
            "'themes' is a list of 3 to 5 cross-paper themes.\n\n"
            f"Topic: {topic}\n\n"
            + "\n\n".join(blocks)
        )

    @staticmethod
    def _themes_from_terms(terms: list[str]) -> list[str]:
        if not terms:
            return ["The selected papers form a compact starting set for the topic."]
        chunks = [terms[index : index + 3] for index in range(0, min(len(terms), 12), 3)]
        return ["Connections around " + ", ".join(chunk) for chunk in chunks if chunk]

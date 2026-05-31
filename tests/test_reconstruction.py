from __future__ import annotations

import unittest

from agentic_lit_review.agents.planning import PlanningAgent
from agentic_lit_review.agents.screening import ScreeningAgent
from agentic_lit_review.evaluation import evaluate_state
from agentic_lit_review.llm import LLMClient
from agentic_lit_review.models import Paper
from agentic_lit_review.retrievers.dedupe import dedupe_papers


class ReconstructionTests(unittest.TestCase):
    def test_dedupes_by_normalized_title(self) -> None:
        papers = [
            Paper(title="Zero-Shot Learning: A Survey", abstract="", source="arXiv"),
            Paper(title="Zero Shot Learning A Survey", abstract="", source="Semantic Scholar"),
        ]

        self.assertEqual(len(dedupe_papers(papers)), 1)

    def test_screening_heuristic_scores_topic_overlap(self) -> None:
        llm = LLMClient(enabled=False)
        agent = ScreeningAgent(llm)
        paper = Paper(
            title="Zero-shot learning with semantic embeddings",
            abstract="A method for zero-shot image classification.",
            source="arXiv",
        )

        score = agent.score_papers("zero-shot learning", [paper])[0]

        self.assertGreaterEqual(score, 0.5)

    def test_planning_heuristic_puts_surveys_first(self) -> None:
        llm = LLMClient(enabled=False)
        agent = PlanningAgent(llm)
        state = {
            "research_topic": "zero-shot learning",
            "screened_papers": [
                Paper(title="Recent zero-shot application", abstract="", source="arXiv", year=2024),
                Paper(title="Zero-shot learning survey", abstract="", source="arXiv", year=2020),
            ],
            "themes": ["Connections around zero-shot, learning, embeddings"],
        }

        result = agent(state)

        self.assertEqual(result["reading_order"][0]["title"], "Zero-shot learning survey")

    def test_evaluation_reports_pipeline_completion(self) -> None:
        paper = Paper(
            title="Zero-shot learning survey",
            abstract="Zero-shot learning survey.",
            source="Sample",
            relevance=0.9,
        )
        state = {
            "research_topic": "zero-shot learning",
            "raw_papers": [paper],
            "screened_papers": [paper],
            "themes": ["Foundational methods"],
            "research_gaps": ["Human evaluation"],
            "research_plan": ["Compare methods"],
            "reading_order": [{"title": paper.title, "reason": "Start broad"}],
        }

        metrics = evaluate_state(state)

        self.assertTrue(metrics["pipeline_completion"]["planning"])
        self.assertEqual(metrics["retrieval"]["source_diversity"], 1)


if __name__ == "__main__":
    unittest.main()

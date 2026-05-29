from __future__ import annotations

from agentic_lit_review.agents import PlanningAgent, ScreeningAgent, SearchAgent, SynthesisAgent
from agentic_lit_review.llm import LLMClient
from agentic_lit_review.state import AgentState


class LiteratureReviewPipeline:
    def __init__(
        self,
        llm: LLMClient | None = None,
        max_results_per_query: int = 5,
        min_screened_papers: int = 3,
        max_search_iterations: int = 2,
    ) -> None:
        self.llm = llm or LLMClient()
        self.min_screened_papers = min_screened_papers
        self.max_search_iterations = max_search_iterations
        self.search = SearchAgent(self.llm, max_results_per_query=max_results_per_query)
        self.screening = ScreeningAgent(self.llm)
        self.synthesis = SynthesisAgent(self.llm)
        self.planning = PlanningAgent(self.llm)
        self._compiled_graph = self._try_compile_langgraph()

    def run(self, topic: str) -> AgentState:
        state: AgentState = {
            "research_topic": topic,
            "search_queries": [],
            "raw_papers": [],
            "screened_papers": [],
            "themes": [],
            "research_gaps": [],
            "research_plan": [],
            "reading_order": [],
            "search_iterations": 0,
            "errors": [],
        }
        if self._compiled_graph is not None:
            return self._compiled_graph.invoke(state)
        return self._run_sequential(state)

    def _run_sequential(self, state: AgentState) -> AgentState:
        while True:
            state = self.search(state)
            state = self.screening(state)
            enough_papers = len(state.get("screened_papers", [])) >= self.min_screened_papers
            reached_limit = state.get("search_iterations", 0) >= self.max_search_iterations
            if enough_papers or reached_limit:
                break
        state = self.synthesis(state)
        state = self.planning(state)
        return state

    def _try_compile_langgraph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        graph = StateGraph(AgentState)
        graph.add_node("search", self.search)
        graph.add_node("screen", self.screening)
        graph.add_node("synthesize", self.synthesis)
        graph.add_node("plan", self.planning)
        graph.set_entry_point("search")
        graph.add_edge("search", "screen")
        graph.add_conditional_edges(
            "screen",
            self._route_after_screening,
            {"search": "search", "synthesize": "synthesize"},
        )
        graph.add_edge("synthesize", "plan")
        graph.add_edge("plan", END)
        return graph.compile()

    def _route_after_screening(self, state: AgentState) -> str:
        enough_papers = len(state.get("screened_papers", [])) >= self.min_screened_papers
        reached_limit = state.get("search_iterations", 0) >= self.max_search_iterations
        return "synthesize" if enough_papers or reached_limit else "search"

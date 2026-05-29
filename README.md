# Agentic Literature Review

This is a clean-room reconstruction of the CIS 600 final project described in the existing report and presentation files. The original source code was not available, so this repository recreates the system architecture from the submitted materials:

- Search Agent: expands a topic into multiple academic search queries.
- Screening Agent: scores retrieved papers for relevance.
- Synthesis Agent: summarizes papers and extracts cross-paper themes.
- Planning Agent: generates research gaps, a research plan, and a recommended reading order.
- Orchestration: uses LangGraph when installed, with a built-in sequential fallback.

The implementation is designed to run in two modes:

- LLM mode with Groq, using `GROQ_API_KEY`.
- Heuristic fallback mode with `--no-llm`, useful for demos and tests when no API key is available.
- Optional LangGraph mode when `langgraph` is installed in a clean environment.

## Project Structure

```text
.
├── main.py
├── pyproject.toml
├── requirements.txt
├── .env.example
├── src/
│   └── agentic_lit_review/
│       ├── agents/
│       │   ├── search.py
│       │   ├── screening.py
│       │   ├── synthesis.py
│       │   └── planning.py
│       ├── retrievers/
│       │   ├── arxiv.py
│       │   ├── semantic_scholar.py
│       │   └── dedupe.py
│       ├── graph.py
│       ├── llm.py
│       ├── models.py
│       ├── state.py
│       └── text_utils.py
└── tests/
    └── test_reconstruction.py
```

## Setup

The simplest demo does not need any package installation:

```powershell
py -3 main.py "zero-shot learning" --no-llm
```

For a clean virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Windows, if `python` is not available in PowerShell, use `py -3` instead:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```

Install optional LangGraph support only in a clean environment:

```powershell
py -3 -m pip install -r requirements-langgraph.txt
```

Optional Groq configuration:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set these values if you want LLM-backed query generation, screening, synthesis, and planning:

```text
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

## Run

Run with heuristic fallback:

```powershell
python main.py "zero-shot learning" --no-llm
```

If arXiv or Semantic Scholar returns `HTTP Error 429`, wait a few minutes and retry with fewer calls:

```powershell
python main.py "zero-shot learning" --max-results 3 --max-search-iterations 1
```

Run with Groq if `GROQ_API_KEY` is configured:

```powershell
python main.py "zero-shot learning"
```

Save output as JSON:

```powershell
python main.py "retrieval augmented generation" --json outputs/rag_review.json
```

## Notes on Reconstruction

Because the original source code was missing, this version cannot be byte-for-byte identical to the original project. It preserves the project behavior and architecture described in the final report:

- Python 3.11 implementation.
- LangGraph-style stateful orchestration.
- arXiv and Semantic Scholar retrieval.
- Shared `AgentState` passed through four specialized agents.
- Adaptive loop back to search when fewer than three papers pass screening.
- Reading order based on survey/foundation-first sequencing.

## Tests

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

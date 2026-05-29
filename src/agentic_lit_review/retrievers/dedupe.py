from __future__ import annotations

from agentic_lit_review.models import Paper
from agentic_lit_review.text_utils import normalize_title


def dedupe_papers(papers: list[Paper]) -> list[Paper]:
    seen: set[str] = set()
    unique: list[Paper] = []
    for paper in papers:
        key = normalize_title(paper.title)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return unique

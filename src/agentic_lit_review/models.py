from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Paper:
    title: str
    abstract: str
    source: str
    url: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    paper_id: str = ""
    relevance: float = 0.0
    summary: str = ""

    def brief(self, max_abstract_chars: int = 500) -> str:
        authors = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors += ", et al."
        year = str(self.year) if self.year else "n.d."
        abstract = " ".join(self.abstract.split())
        if len(abstract) > max_abstract_chars:
            abstract = abstract[: max_abstract_chars - 3].rstrip() + "..."
        return (
            f"Title: {self.title}\n"
            f"Authors: {authors or 'Unknown'}\n"
            f"Year: {year}\n"
            f"Source: {self.source}\n"
            f"Abstract: {abstract}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from agentic_lit_review.models import Paper


class SemanticScholarRetriever:
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, api_key: str | None = None, retries: int = 2, delay_seconds: float = 2.0) -> None:
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self.retries = retries
        self.delay_seconds = delay_seconds

    def search(self, query: str, limit: int = 5) -> list[Paper]:
        params = {
            "query": query,
            "limit": str(limit),
            "fields": "title,abstract,year,authors,url,venue,paperId",
        }
        request = urllib.request.Request(
            f"{self.endpoint}?{urllib.parse.urlencode(params)}",
            headers=self._headers(),
            method="GET",
        )
        payload = json.loads(self._open_with_retry(request))

        papers: list[Paper] = []
        for item in payload.get("data", []):
            title = item.get("title") or ""
            if not title:
                continue
            authors = [author.get("name", "") for author in item.get("authors", [])]
            papers.append(
                Paper(
                    title=" ".join(title.split()),
                    abstract=" ".join((item.get("abstract") or "").split()),
                    authors=[author for author in authors if author],
                    year=item.get("year"),
                    venue=item.get("venue") or "",
                    source="Semantic Scholar",
                    url=item.get("url") or "",
                    paper_id=item.get("paperId") or "",
                )
            )
        return papers

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "agentic-lit-review/0.1"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _open_with_retry(self, request: urllib.request.Request) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.delay_seconds * attempt)
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 or attempt >= self.retries:
                    raise
                retry_after = exc.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 20))
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise
        raise RuntimeError(f"Semantic Scholar request failed after retries: {last_error}")

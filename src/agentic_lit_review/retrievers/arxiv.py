from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from agentic_lit_review.models import Paper


class ArxivRetriever:
    endpoint = "https://export.arxiv.org/api/query"

    def __init__(self, retries: int = 2, delay_seconds: float = 3.0) -> None:
        self.retries = retries
        self.delay_seconds = delay_seconds

    def search(self, query: str, limit: int = 5) -> list[Paper]:
        params = {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(limit),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{self.endpoint}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "agentic-lit-review/0.1 contact: classroom-demo"},
            method="GET",
        )
        xml_text = self._open_with_retry(request)

        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers: list[Paper] = []
        for entry in root.findall("atom:entry", ns):
            title = self._text(entry, "atom:title", ns)
            abstract = self._text(entry, "atom:summary", ns)
            published = self._text(entry, "atom:published", ns)
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = [
                self._text(author, "atom:name", ns)
                for author in entry.findall("atom:author", ns)
            ]
            paper_url = self._text(entry, "atom:id", ns)
            if title:
                papers.append(
                    Paper(
                        title=" ".join(title.split()),
                        abstract=" ".join(abstract.split()),
                        authors=authors,
                        year=year,
                        source="arXiv",
                        url=paper_url,
                        paper_id=paper_url.rsplit("/", 1)[-1],
                    )
                )
        return papers

    def _open_with_retry(self, request: urllib.request.Request) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.delay_seconds * attempt)
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    return response.read().decode("utf-8", errors="replace")
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
        raise RuntimeError(f"arXiv request failed after retries: {last_error}")

    @staticmethod
    def _text(node: ET.Element, path: str, ns: dict[str, str]) -> str:
        child = node.find(path, ns)
        return child.text.strip() if child is not None and child.text else ""

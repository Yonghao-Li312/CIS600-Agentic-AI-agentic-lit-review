from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "based",
    "before",
    "being",
    "between",
    "both",
    "from",
    "have",
    "into",
    "more",
    "most",
    "over",
    "paper",
    "papers",
    "research",
    "show",
    "such",
    "than",
    "that",
    "their",
    "these",
    "this",
    "through",
    "using",
    "with",
    "within",
}


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


def top_terms(texts: list[str], limit: int = 12) -> list[str]:
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return [term for term, _ in counts.most_common(limit)]


def first_sentence(text: str, max_chars: int = 260) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return "No abstract was available."
    match = re.search(r"(?<=[.!?])\s+", cleaned)
    sentence = cleaned[: match.start()] if match else cleaned
    if len(sentence) > max_chars:
        sentence = sentence[: max_chars - 3].rstrip() + "..."
    return sentence

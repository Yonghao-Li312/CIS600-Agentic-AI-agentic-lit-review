from __future__ import annotations

from agentic_lit_review.models import Paper


class SampleRetriever:
    """Offline retriever used for portfolio demos and tests."""

    def search(self, query: str, limit: int = 5) -> list[Paper]:
        papers = [
            Paper(
                title="A Survey of Zero-Shot Learning: Settings, Methods, and Applications",
                abstract=(
                    "Zero-shot learning studies recognition of unseen classes by transferring "
                    "knowledge from seen classes through semantic attributes, language embeddings, "
                    "or auxiliary descriptions. Survey papers organize the problem settings, "
                    "datasets, evaluation protocols, and common model families."
                ),
                authors=["Sample Author"],
                year=2018,
                source="Sample",
                url="https://example.com/zero-shot-survey",
            ),
            Paper(
                title="Semantic Embedding Approaches for Zero-Shot Image Classification",
                abstract=(
                    "This paper introduces semantic embedding models that align image features "
                    "with class descriptions. The approach provides a foundation for recognizing "
                    "unseen categories without labeled examples."
                ),
                authors=["Sample Author"],
                year=2016,
                source="Sample",
                url="https://example.com/semantic-embedding",
            ),
            Paper(
                title="Generalized Zero-Shot Learning with Calibrated Stacking",
                abstract=(
                    "Generalized zero-shot learning evaluates models on both seen and unseen "
                    "classes. Calibration methods reduce the bias toward seen classes and improve "
                    "balanced recognition performance."
                ),
                authors=["Sample Author"],
                year=2019,
                source="Sample",
                url="https://example.com/generalized-zsl",
            ),
            Paper(
                title="Prompt-Based Vision-Language Models for Zero-Shot Recognition",
                abstract=(
                    "Large vision-language models enable zero-shot recognition by comparing "
                    "images with natural language prompts. Prompt engineering and representation "
                    "alignment are central to recent progress."
                ),
                authors=["Sample Author"],
                year=2022,
                source="Sample",
                url="https://example.com/prompt-zsl",
            ),
            Paper(
                title="Benchmarking Robustness in Zero-Shot Learning Systems",
                abstract=(
                    "Robust zero-shot systems must handle distribution shift, ambiguous class "
                    "descriptions, and dataset bias. Benchmark design is important for evaluating "
                    "real-world transfer performance."
                ),
                authors=["Sample Author"],
                year=2024,
                source="Sample",
                url="https://example.com/robust-zsl",
            ),
        ]
        return papers[:limit]

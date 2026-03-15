from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Iterable


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    token_count: int


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    lexical_score: float
    dense_score: float
    total_score: float


class InMemoryRAGPipeline:
    def __init__(
        self,
        chunk_size: int = 22,
        overlap: int = 5,
        lexical_weight: float = 0.45,
        dense_weight: float = 0.55,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self._chunks: list[Chunk] = []
        self._chunk_vectors: dict[str, Counter[str]] = {}
        self._cache: dict[tuple[str, int], list[RetrievalResult]] = {}
        self.cache_hits = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
        return [token for token in cleaned.split() if token]

    @staticmethod
    def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        common = set(left).intersection(right)
        numerator = sum(left[token] * right[token] for token in common)
        left_norm = sqrt(sum(value * value for value in left.values()))
        right_norm = sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def chunk_document(self, document: Document) -> list[Chunk]:
        tokens = self._tokenize(document.text)
        if not tokens:
            return []

        chunks: list[Chunk] = []
        step = max(1, self.chunk_size - self.overlap)
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.chunk_size]
            if not window:
                break
            chunk = Chunk(
                chunk_id=f"{document.doc_id}-chunk-{len(chunks) + 1}",
                doc_id=document.doc_id,
                title=document.title,
                text=" ".join(window),
                token_count=len(window),
            )
            chunks.append(chunk)
            if start + self.chunk_size >= len(tokens):
                break
        return chunks

    def ingest(self, documents: Iterable[Document]) -> None:
        self._chunks.clear()
        self._chunk_vectors.clear()
        self._cache.clear()
        self.cache_hits = 0

        for document in documents:
            for chunk in self.chunk_document(document):
                self._chunks.append(chunk)
                self._chunk_vectors[chunk.chunk_id] = Counter(self._tokenize(chunk.text))

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        cache_key = (query.lower(), top_k)
        if cache_key in self._cache:
            self.cache_hits += 1
            return list(self._cache[cache_key])

        query_tokens = self._tokenize(query)
        query_vector = Counter(query_tokens)
        query_token_set = set(query_tokens)
        results: list[RetrievalResult] = []

        for chunk in self._chunks:
            chunk_vector = self._chunk_vectors[chunk.chunk_id]
            lexical_overlap = len(query_token_set.intersection(chunk_vector))
            lexical_score = lexical_overlap / len(query_token_set) if query_token_set else 0.0
            dense_score = self._cosine_similarity(query_vector, chunk_vector)
            total_score = round(
                (self.lexical_weight * lexical_score) + (self.dense_weight * dense_score),
                4,
            )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    lexical_score=round(lexical_score, 4),
                    dense_score=round(dense_score, 4),
                    total_score=total_score,
                )
            )

        ranked = sorted(results, key=lambda item: item.total_score, reverse=True)[:top_k]
        self._cache[cache_key] = list(ranked)
        return ranked

    def answer(self, query: str, top_k: int = 2) -> dict[str, object]:
        results = self.search(query, top_k=top_k)
        supporting_chunks = [result.chunk for result in results]
        if not supporting_chunks:
            return {"answer": "No relevant context was found.", "sources": []}

        evidence_lines = [
            f"{chunk.title}: {chunk.text.split(' and ')[0].strip()}" for chunk in supporting_chunks
        ]
        answer = " ".join(evidence_lines)
        return {
            "answer": answer,
            "sources": [chunk.doc_id for chunk in supporting_chunks],
        }


def build_demo_pipeline() -> InMemoryRAGPipeline:
    pipeline = InMemoryRAGPipeline()
    pipeline.ingest(
        [
            Document(
                doc_id="ingestion",
                title="Async ingestion",
                text=(
                    "The ingestion worker normalizes markdown and PDF content into retrieval chunks. "
                    "Each chunk keeps source metadata and overlap so citations remain grounded during answer generation."
                ),
            ),
            Document(
                doc_id="retrieval",
                title="Hybrid retrieval",
                text=(
                    "Hybrid retrieval combines lexical signals with dense similarity. "
                    "This is useful when users ask terse operational questions that share only a few exact tokens with the best document."
                ),
            ),
            Document(
                doc_id="caching",
                title="Query caching",
                text=(
                    "The serving layer caches repeated high-frequency queries. "
                    "This reduces latency and cost while preserving the same grounded answer path for common requests."
                ),
            ),
        ]
    )
    return pipeline


def main() -> None:
    pipeline = build_demo_pipeline()
    query = "How does the pipeline reduce latency for repeated queries?"
    result = pipeline.answer(query)
    print(f"Query: {query}\n")
    print(result["answer"])
    print(f"\nSources: {', '.join(result['sources'])}")

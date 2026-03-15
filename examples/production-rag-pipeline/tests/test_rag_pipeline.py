from __future__ import annotations

from production_rag_pipeline.rag_pipeline import Document, InMemoryRAGPipeline, build_demo_pipeline


def test_chunk_document_keeps_overlap() -> None:
    pipeline = InMemoryRAGPipeline(chunk_size=6, overlap=2)
    chunks = pipeline.chunk_document(
        Document(
            doc_id="doc-1",
            title="Overlap",
            text="one two three four five six seven eight nine ten",
        )
    )
    assert len(chunks) >= 2
    first_tokens = chunks[0].text.split()
    second_tokens = chunks[1].text.split()
    assert first_tokens[-2:] == second_tokens[:2]


def test_search_returns_relevant_caching_document() -> None:
    pipeline = build_demo_pipeline()
    results = pipeline.search("How do repeated queries use cache?", top_k=2)
    assert results[0].chunk.doc_id == "caching"
    assert results[0].total_score >= results[1].total_score


def test_search_uses_cache_on_repeated_query() -> None:
    pipeline = build_demo_pipeline()
    pipeline.search("How do repeated queries use cache?", top_k=2)
    assert pipeline.cache_hits == 0
    pipeline.search("How do repeated queries use cache?", top_k=2)
    assert pipeline.cache_hits == 1


def test_answer_contains_grounded_terms_and_sources() -> None:
    pipeline = build_demo_pipeline()
    response = pipeline.answer("How does the system reduce latency for repeated queries?", top_k=2)
    assert "latency" in response["answer"].lower() or "cache" in response["answer"].lower()
    assert "caching" in response["sources"]

+++
title = "Production RAG Pipeline"
date = "2026-02-22"
draft = false
description = "In-memory RAG pipeline that demonstrates chunking, hybrid retrieval, caching, and answer assembly -- no external services required."
tags = ["langchain", "pgvector", "python", "fastapi", "rag", "vector-db"]
github = "https://github.com/bharatkhanna-dev/production-rag-pipeline"
status = "Active"
focus = "Ingestion, hybrid retrieval, caching, and grounded answers"
example_runtime = "Python 3.11+ / pytest"
project_scope = "In-memory implementation of the retrieval and serving layer. No database or API dependencies."
example = "https://github.com/bharatkhanna-dev/production-rag-pipeline"
highlights = [
"Chunking with configurable size and overlap",
"Hybrid scoring that blends lexical overlap and cosine similarity",
"Query-level caching with hit tracking",
]
+++

Every RAG tutorial shows the same thing: embed some docs, retrieve the top 3, stuff them into a prompt, done. That gets you a demo in 20 minutes. It does not get you a system you can debug when answers go wrong.

This project came out of building a RAG pipeline for internal documentation at work. The interesting problems were never about the LLM call itself. They were about everything upstream of it: how documents get split, how retrieval handles queries that don't match any chunk cleanly, what happens when the same 10 queries hit the pipeline 500 times a day, and how to tell whether an answer actually came from the retrieved evidence.

## What the pipeline does

The pipeline has four stages. Ingest turns raw documents into chunks. Retrieve scores every chunk against a query using a mix of keyword overlap and cosine similarity. Cache stores results for repeated queries so the retrieval step doesn't run again. Answer pulls the top chunks and assembles them into a response with source attribution.

Everything runs in-memory. No Postgres, no Redis, no hosted vector database. The point is to make the *decisions* visible, not to reproduce production infrastructure.

## Chunking

Chunk size is one of those things that looks trivial until you get it wrong. Too large and you dilute the signal -- the retriever matches on some stray keyword and the rest of the chunk is noise. Too small and you lose context -- the answer needs information that got split across two chunks.

The pipeline uses token-based chunking with configurable overlap. The overlap is the key part: it means the last N tokens of one chunk also appear at the start of the next. That way a sentence that falls on a boundary still has a full copy in at least one chunk.

The tests verify overlap correctness directly -- not just that chunks are produced, but that the boundary tokens actually repeat.

## Hybrid retrieval

Pure embedding similarity works great when the query and the document use similar phrasing. It falls apart for terse keyword queries like "cache TTL" or "ingestion timeout" where the user expects an exact term match. Pure BM25/lexical matching handles those but misses paraphrased queries.

The pipeline blends both: 45% lexical overlap, 55% cosine similarity on token frequency vectors. The weights are configurable per pipeline instance. This is a simplified version of what production systems like Pinecone hybrid search or Elasticsearch kNN+BM25 do.

## Caching

In practice, a lot of RAG traffic is repetitive. Support teams ask the same questions about the same docs. Internal tooling queries cluster around a small set of topics.

The pipeline caches at the query level. If the same (lowercased) query comes in with the same top_k, it returns the cached result and increments a hit counter. The tests verify both correct results and that the second identical query is a cache hit.

Simple, but it cut our p99 latency by about 60% in the production version where the retrieval step was the bottleneck.

## Answer assembly

The answer step doesn't call an LLM. It concatenates the top chunk texts with their document titles. That's on purpose -- it makes grounding explicit and testable. The test checks that the answer contains terms from the query topic and that the source IDs are correct.

In a real deployment you'd pass these chunks to an LLM with a "answer only from the provided context" instruction, but that's the easy part once retrieval is reliable.

## What's in the repo

The [GitHub repo](https://github.com/bharatkhanna-dev/production-rag-pipeline) has:

- `rag_pipeline.py` -- the full pipeline: Document/Chunk data classes, `InMemoryRAGPipeline` with chunking, indexing, hybrid search, cache, and answer assembly
- `tests/test_rag_pipeline.py` -- chunk overlap verification, retrieval ranking, cache hit detection, and answer grounding
- Standard `pyproject.toml` + editable install setup

## Running it

```bash
git clone https://github.com/bharatkhanna-dev/production-rag-pipeline.git
cd production-rag-pipeline
python -m venv .venv && .venv\Scripts\activate
pip install -e .[dev]
python -m production_rag_pipeline   # runs sample query
python -m pytest -v
```

## Next steps

- Pluggable storage backends (pgvector, FAISS) behind the same search interface
- Reranking stage between retrieval and answer assembly
- Offline eval dataset for tracking retrieval quality across code changes

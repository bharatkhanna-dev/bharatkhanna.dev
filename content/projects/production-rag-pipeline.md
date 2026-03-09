+++
title = "Production RAG Pipeline"
date = "2026-02-22"
draft = false
description = "A scalable retrieval-augmented generation pipeline using pgvector, LangChain, and FastAPI — serving 10k+ queries per day with sub-50ms p95 latency."
tags = ["langchain", "pgvector", "python", "fastapi", "rag", "vector-db"]
github = "https://github.com/bharatkhanna/production-rag"
+++

A production-grade RAG pipeline built for high-throughput document Q&A.

## Architecture

- **Ingestion**: Async document processing pipeline with chunking and metadata extraction
- **Storage**: pgvector with HNSW indexing for sub-10ms retrieval
- **Retrieval**: Hybrid BM25 + dense retrieval with cross-encoder re-ranking
- **Serving**: FastAPI with streaming responses, circuit breakers, and graceful degradation

## Key Decisions

Used pgvector over a dedicated vector DB for operational simplicity — one less stateful service to manage in production. The HNSW index handles 10M+ vectors with p95 retrieval latency under 8ms.

Query-level caching with Redis for the top 5% most frequent queries reduced LLM cost by ~40% with no code changes on the client side.

## Evaluation

Integrated LangSmith for continuous evaluation of retrieval quality. Built a custom `context_relevance` scorer that benchmarks retrieved chunks against human-labeled ground truth weekly.

+++
title = "Production RAG Pipeline"
date = "2026-02-22"
draft = false
description = "A production-flavored RAG architecture write-up with a lightweight public companion example for ingestion, retrieval, caching, and answer assembly."
tags = ["langchain", "pgvector", "python", "fastapi", "rag", "vector-db"]
github = "https://github.com/bharatkhanna-dev/production-rag-pipeline"
status = "Public companion example available"
focus = "Ingestion, hybrid retrieval, caching, and dependable serving patterns"
example_runtime = "Local Python + pytest"
project_scope = "The public example keeps storage in-memory and retrieval deterministic so the core RAG decisions are easy to inspect without external services, then publish as a standalone repository."
example = "https://github.com/bharatkhanna-dev/production-rag-pipeline"
source = "https://github.com/bharatkhanna-dev/bharatkhanna.dev"
highlights = [
	"Demonstrates document chunking, hybrid retrieval, caching, and extractive answer assembly.",
	"Keeps the default path runnable without Postgres, Redis, or hosted vector infrastructure.",
	"Backed by pytest tests for chunking, retrieval relevance, caching, and answer quality."
]
+++

> **TL;DR** — Production RAG is less about calling an LLM with retrieved text and more about designing reliable ingestion, retrieval, ranking, caching, and fallback behavior around it.

## Architecture

- **Ingestion** that normalizes documents into retrieval-friendly chunks
- **Hybrid retrieval** that balances lexical matching with semantic similarity
- **Caching** for repeated queries and predictable hot paths
- **Answer assembly** that cites the most relevant chunks instead of hallucinating around them

## Why This Project Exists

Most RAG demos stop at “embed a few files and ask a question.” That is enough to prove a concept, but not enough to explain what breaks in practice:

- poor chunking that fragments context,
- retrieval tuned for happy-path queries only,
- expensive repeated queries with no cache layer,
- and APIs that cannot degrade gracefully when retrieval is thin.

This project focuses on the engineering layer around retrieval: how to turn a demo into a system that is inspectable, measurable, and resilient.

## Key Decisions

Used a **hybrid retrieval** mindset instead of relying on a single scoring function. Exact keyword overlap catches terse operational queries; semantic overlap rescues less literal phrasing. The public companion example uses an in-memory approximation of that pattern so the tradeoff is easy to see locally.

Caching is treated as part of the retrieval system rather than an afterthought. Repeated questions should not pay the full retrieval cost every time, especially when internal documentation and support workflows tend to cluster heavily around a small number of recurring queries.

The public example keeps this simple: repeated queries are cached at the pipeline level and the tests verify both correctness and cache-hit behavior.

## Evaluation

The most important RAG metric is not whether an answer sounds polished. It is whether the retrieval layer surfaced the right context. In practice, that means inspecting:

- chunk boundaries,
- retrieval ordering,
- answer grounding,
- and the ability to explain *why* a document was selected.

## Public Companion Example

The public companion for this project should live in its own repository under `bharatkhanna-dev`:

- [Companion repo on GitHub](https://github.com/bharatkhanna-dev/production-rag-pipeline)

It includes:

- a document chunker,
- an in-memory hybrid retriever,
- a query cache,
- a simple answer composer that cites supporting chunks,
- and tests for chunk overlap, ranking, cache hits, and grounded responses.

## What Makes the Example Useful

The example does **not** try to reproduce every production dependency. Instead, it preserves the core decisions that matter:

- how the retrieval pipeline is staged,
- where caching belongs,
- and how to keep the answer tied to retrieved evidence.

That makes it a better learning artifact and a better starting point for adaptation.

## How to Use It

1. Seed the demo pipeline with the bundled example documents.
2. Run the sample query flow from the module entry point.
3. Execute `pytest` to validate chunking, retrieval, and caching behavior.
4. Replace the sample documents with your own corpus and adjust the scoring weights.

## What I Would Extend Next

- pluggable storage backends,
- reranking hooks,
- and offline evaluation datasets for retrieval quality over time.

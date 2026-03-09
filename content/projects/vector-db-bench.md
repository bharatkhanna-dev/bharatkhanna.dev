+++
title = "Vector Database Benchmarking Suite"
date = "2025-11-20"
draft = false
description = "A benchmarking framework for comparing pgvector, Pinecone, and Weaviate across latency, recall, and operational cost dimensions for production RAG workloads."
tags = ["vector-db", "pgvector", "pinecone", "weaviate", "python", "benchmarking"]
github = "https://github.com/bharatkhanna/vector-db-bench"
+++

Reproducible benchmarks for selecting a vector database for production RAG systems.

## Motivation

"Which vector DB should we use?" is a question with no universally correct answer. The right choice depends on your scale, existing infrastructure, latency budget, and ops team size. This suite gives you numbers on your data, not someone else's marketing benchmarks.

## What Gets Measured

- **Recall@k**: How often the correct document appears in the top-k results
- **Insert throughput**: Documents per second for batch ingestion
- **Query latency**: p50, p95, p99 at varying concurrency levels
- **Index build time**: How long HNSW/IVF indexing takes at 1M, 10M, 100M vectors
- **Monthly cost estimate**: Based on managed cloud pricing calculators

## Key Finding

For teams already running PostgreSQL, pgvector with HNSW indexing reaches within 5-8% of Pinecone's recall at 1/10th the cost below 50M vectors. The crossover point is roughly 100M vectors + >10k QPS.

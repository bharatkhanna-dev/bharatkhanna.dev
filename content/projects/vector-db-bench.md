+++
title = "Vector Database Benchmarking Suite"
date = "2025-11-20"
draft = false
description = "A reproducible benchmarking workflow for vector search tradeoffs, paired with a local Python harness for recall, latency, and reporting."
tags = ["vector-db", "pgvector", "pinecone", "weaviate", "python", "benchmarking"]
github = "https://github.com/bharatkhanna-dev/vector-db-bench"
status = "Public companion example available"
focus = "Benchmark methodology, repeatability, and practical tradeoff analysis"
example_runtime = "Local Python + pytest"
project_scope = "The public example focuses on benchmark structure and repeatability with deterministic synthetic data and pluggable backends, then publish as a dedicated repository."
example = "https://github.com/bharatkhanna-dev/vector-db-bench"
source = "https://github.com/bharatkhanna-dev/bharatkhanna.dev"
highlights = [
	"Benchmarks exact and approximate search strategies with deterministic clustered vectors.",
	"Reports recall@k, latency percentiles, build time, and candidate coverage in one table.",
	"Uses local synthetic data so methodology can be validated before plugging in managed vendors."
]
+++

> **TL;DR** — “Which vector database should we use?” is not a product-selection question first. It is a benchmarking question first.

## Motivation

Most teams compare vendors by reading benchmark blog posts built on somebody else’s workload, somebody else’s filters, and somebody else’s latency budget. That almost always leads to a poor decision.

This project reframes the problem: before choosing a vendor, build a benchmark harness that makes your assumptions explicit.

That means measuring on the things that actually matter for your system:

- recall under your query patterns,
- latency at your candidate sizes,
- build time for your ingest profile,
- and operational complexity for your team.

## What Gets Measured

- **Recall@k**: How often the correct document appears in the top-k results
- **Candidate coverage**: How much of the dataset an approximate method actually inspects
- **Query latency**: p50 and p95 for comparable local workloads
- **Index build time**: How long each backend needs before serving queries
- **Reporting quality**: Whether the results are easy to compare and repeat

## Key Finding

The key lesson is usually not “backend X wins.” It is that **exact search, approximate search, and operational simplicity trade off against each other in measurable ways**. Teams often discover that the right first step is to benchmark methodology locally, then plug in vendor clients later once they trust the harness.

## Public Companion Example

The public companion for this project should live in its own repository under `bharatkhanna-dev`:

- [Companion repo on GitHub](https://github.com/bharatkhanna-dev/vector-db-bench)

It includes:

- deterministic synthetic dataset generation,
- an exact linear baseline,
- two approximate search strategies,
- benchmark reporting for recall, latency, build time, and candidate ratios,
- and tests that verify both metric correctness and result formatting.

## Why the Public Example Is Structured This Way

The companion example deliberately avoids hosted vendor SDKs in the default path. That keeps the first experience focused on **benchmark design**, not credentials or infrastructure.

Once the methodology is trustworthy, the same harness shape can be extended to pgvector, Pinecone, Weaviate, Qdrant, Milvus, or any internal ANN service.

## How to Use It

1. Generate the deterministic clustered dataset.
2. Run the benchmark harness against the included backends.
3. Inspect recall and latency tradeoffs in the generated summary table.
4. Add your own backend implementation behind the same interface.

## What I Would Extend Next

- concurrency-aware load generation,
- metadata filters,
- and vendor adapters for real infrastructure once the local methodology is locked in.

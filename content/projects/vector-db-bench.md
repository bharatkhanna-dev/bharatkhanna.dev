+++
title = "Vector Search Benchmark Harness"
date = "2025-11-20"
draft = false
description = "Local benchmark harness for comparing exact vs approximate vector search -- recall, latency, and candidate coverage on synthetic clustered data."
tags = ["vector-db", "pgvector", "pinecone", "weaviate", "python", "benchmarking"]
github = "https://github.com/bharatkhanna-dev/vector-db-bench"
status = "Active"
focus = "Benchmark methodology and reproducible vector search comparisons"
example_runtime = "Python 3.11+ / pytest"
project_scope = "Local benchmarking with synthetic data. Vendor SDK integrations are stubbed but not included."
example = "https://github.com/bharatkhanna-dev/vector-db-bench"
highlights = [
"Deterministic clustered dataset generation with fixed seeds",
"Three backends: exact linear, sign-bucket ANN, and projection ANN",
"Recall@k, p50/p95 latency, build time, and candidate ratio in one table",
]
+++

When we needed to pick a vector database at work, the first instinct was to read vendor benchmark blog posts. The problem is every vendor benchmarks on their best-case scenario with their own dataset on their own hardware. The numbers are real but the comparison is useless for *your* workload.

So I built a local benchmark harness instead. The idea: before you start evaluating Pinecone vs pgvector vs Qdrant, first build a harness that captures *what you actually care about measuring*. Then plug in the vendor clients later once you trust the methodology.

## What it measures

The harness runs each backend through the same query set and reports five things:

- **Recall@k** -- what fraction of the true top-k results does the backend return? Exact search is always 1.0 by definition. ANN backends trade recall for speed.
- **p50 and p95 latency** -- per-query search time in milliseconds. Percentiles matter more than averages because ANN methods sometimes have long tails when the query falls near a bucket boundary.
- **Index build time** -- how long it takes to build the search structure from scratch. Matters for ingestion-heavy workloads where the index rebuilds frequently.
- **Candidate ratio** -- what fraction of the total dataset does the backend actually scan? An exact backend scans 100%. A good ANN backend scans a small fraction and still gets high recall.

## The backends

I included three backends to cover the basic spectrum:

**Exact linear** -- brute force cosine similarity over every record. Always correct, always the slowest. This is the baseline that defines "recall = 1.0."

**Sign-bucket ANN** -- hashes each vector into a bucket based on the sign pattern of its first N dimensions, then searches only the matching bucket. Fast when the data clusters well in those dimensions. Falls back to full scan if the bucket is too small (which is why the candidate ratio sometimes hits 1.0).

**Projection ANN** -- sorts candidates by L1 distance on the first two dimensions, takes the top N candidates, then reranks by full cosine similarity. A crude two-stage retrieval pattern. The candidate_pool parameter controls the speed/recall tradeoff directly.

None of these are production-grade algorithms. They're simplified versions of real ANN families (LSH, projection trees, two-stage retrieval) that make the *tradeoff shape* visible without needing HNSW or IVF implementations.

## Synthetic data generation

The dataset is 96 vectors (4 clusters x 24 points) in 8 dimensions, generated with a fixed seed. Each cluster is tight -- points are within +/-0.08 of a random center. The queries are slight perturbations of actual data points so you get realistic near-neighbor patterns.

Fixed seeds mean the benchmark is fully reproducible. Same dataset, same queries, same results every time. That matters because the whole point is to compare backends fairly -- any randomness in the data undermines the comparison.

## What's in the repo

The [GitHub repo](https://github.com/bharatkhanna-dev/vector-db-bench) has:

- `benchmark.py` -- data generation, all three backends, the benchmark runner, and a markdown table formatter
- `tests/test_benchmark.py` -- exact backend recall verification, percentile monotonicity, ANN candidate ratio bounds, and output format checks
- Standard `pyproject.toml` + editable install

## Running it

```bash
git clone https://github.com/bharatkhanna-dev/vector-db-bench.git
cd vector-db-bench
python -m venv .venv && .venv\Scripts\activate
pip install -e .[dev]
python -m vector_db_bench   # prints the comparison table
python -m pytest -v
```

## Extending it

To add a new backend, implement a class with `build(records)` and `search(query, top_k) -> (ids, candidate_count)` methods and drop it into the `backends` list in `run_benchmarks`. The harness handles timing, recall calculation, and reporting.

The natural next step is adding adapters for real vendors (pgvector over psycopg, Pinecone client, Weaviate gRPC) behind the same interface. That way you can compare them on your actual query patterns without rewriting the measurement code.

## Next steps

- Concurrent query load to measure throughput under parallelism
- Metadata filtering benchmarks (filtered search is where most ANN indexes struggle)
- A CLI or config file for running sweeps over different dataset sizes and dimensions

# production-rag-pipeline

In-memory RAG pipeline with chunking, hybrid retrieval, caching, and answer assembly. Runs locally with Python and pytest -- no external services needed.

## Overview

A stripped-down but functional RAG pipeline that covers the four stages most production systems need: document chunking, hybrid search (lexical + dense), query caching, and answer assembly with source attribution.

Everything runs in-memory. The goal is to make retrieval decisions visible and testable, not to reproduce infrastructure.

## Quick start

```bash
git clone https://github.com/bharatkhanna-dev/production-rag-pipeline.git
cd production-rag-pipeline
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .[dev]
python -m production_rag_pipeline   # runs a sample query
python -m pytest -v                 # runs the test suite
```

Requires Python 3.11+.

## Project structure

```
src/production_rag_pipeline/
    rag_pipeline.py    # Document, Chunk, InMemoryRAGPipeline, demo builder
    __init__.py
    __main__.py
tests/
    test_rag_pipeline.py   # overlap, ranking, caching, grounding checks
pyproject.toml
```

## How it works

**Chunking** -- token-based with configurable size and overlap. Overlap ensures boundary sentences appear in at least one complete chunk.

**Hybrid retrieval** -- blends lexical overlap (45%) with cosine similarity on token frequency vectors (55%). Catches both exact keyword matches and paraphrased queries.

**Caching** -- query-level cache keyed on lowercase query + top_k. Tracks hit count for observability.

**Answer assembly** -- concatenates top chunk texts with document titles. No LLM call -- keeps grounding explicit and testable.

## Usage

```python
from production_rag_pipeline import Document, InMemoryRAGPipeline

pipeline = InMemoryRAGPipeline(chunk_size=22, overlap=5)
pipeline.ingest([
    Document(doc_id="setup", title="Setup guide", text="Your document text here..."),
])

results = pipeline.search("how do I configure X?", top_k=3)
answer = pipeline.answer("how do I configure X?")
```

## Tests

```bash
python -m pytest -v
```

Covers: chunk boundary overlap, retrieval ranking correctness, cache hit behavior, and answer source grounding.

## Write-up

Design decisions and tradeoffs: https://bharatkhanna.dev/projects/production-rag-pipeline/

## License

MIT

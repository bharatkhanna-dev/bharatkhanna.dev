# Production RAG Pipeline Example

Public companion example for the project page: https://bharatkhanna.dev/projects/production-rag-pipeline/

Repository: https://github.com/bharatkhanna-dev/production-rag-pipeline

This example packages a small but complete Retrieval-Augmented Generation workflow that you can run locally with only Python and `pytest`.

## What is included

- `src/production_rag_pipeline/rag_pipeline.py` — document chunking, indexing, hybrid retrieval, caching, and answer assembly
- `tests/` — chunking, retrieval, cache-hit, and grounded-answer checks
- `pyproject.toml` — standalone package metadata
- `LICENSE` — MIT license for the future dedicated repository

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m production_rag_pipeline
python -m pytest
```

## Why this example is useful

The project write-up talks about production RAG design decisions. This companion makes those decisions concrete without requiring Postgres, Redis, or a hosted vector database.

Use it to experiment with:

- chunk sizing and overlap,
- hybrid lexical + semantic retrieval,
- query caching,
- and answer grounding from retrieved evidence.

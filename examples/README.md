# Examples

Working code for each project. Clone from GitHub or browse the source here.

| Project | GitHub | Article |
| --- | --- | --- |
| Agent Evaluation Harness | [agent-eval-harness](https://github.com/bharatkhanna-dev/agent-eval-harness) | [write-up](https://bharatkhanna.dev/projects/agent-eval-harness/) |
| Production RAG Pipeline | [production-rag-pipeline](https://github.com/bharatkhanna-dev/production-rag-pipeline) | [write-up](https://bharatkhanna.dev/projects/production-rag-pipeline/) |
| Vector Search Benchmark Harness | [vector-db-bench](https://github.com/bharatkhanna-dev/vector-db-bench) | [write-up](https://bharatkhanna.dev/projects/vector-db-bench/) |

## Running any project

```bash
cd <project-dir>
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
python -m pytest -v
```

Each project ships a `pyproject.toml` with a `[dev]` extra that pulls in pytest.

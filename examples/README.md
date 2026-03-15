# Code Examples by Article and Project

This directory contains **working companion examples** for both long-form articles and project pages.

For project work, these folders should be treated as **staging copies** that can be promoted into their own GitHub repositories under `https://github.com/bharatkhanna-dev`.

The pattern is intentionally consistent:

- one folder per article or project,
- a root `README.md`,
- minimal setup steps,
- runnable Python code,
- and `pytest` coverage for the most important behavior.

## Project Companions

### [LangGraph Agent Evaluation Harness](./agent-eval-harness/)

Deterministic local example for scoring agent answers, checking tool trajectories, and enforcing a release gate.

**What’s included:**
- `src/agent_eval_harness/harness.py` — evaluation primitives and regression summary logic
- `src/agent_eval_harness/sample_agent.py` — demo agent plus runnable example suite
- `tests/` — score, trajectory, and release-gate coverage
- `pyproject.toml` and `requirements.txt` — standalone package metadata and editable dev install

### [Production RAG Pipeline](./production-rag-pipeline/)

Lightweight in-memory RAG example that demonstrates chunking, hybrid retrieval, caching, and grounded answer assembly.

**What’s included:**
- `src/production_rag_pipeline/rag_pipeline.py` — chunking, indexing, retrieval, caching, and answer generation
- `tests/` — retrieval relevance, cache-hit, and answer-grounding tests
- `pyproject.toml` and `requirements.txt` — standalone package metadata and editable dev install

### [Vector Database Benchmarking Suite](./vector-db-bench/)

Local benchmark harness for exact versus approximate vector search strategies with deterministic datasets and reproducible metrics.

**What’s included:**
- `src/vector_db_bench/benchmark.py` — dataset generation, backend interfaces, and metric reporting
- `tests/` — recall, percentile, and report-format validation
- `pyproject.toml` and `requirements.txt` — standalone package metadata and editable dev install

## Article Companions

### [Continuously Improving Agent Quality Using Evaluators Across Single-Turn, Trajectory, and Multi-Turn Interactions](./evaluators-in-agentic-ai-multiturn/)

Working code for agent evaluation patterns covering single-turn, trajectory, multi-turn, and production monitoring setups.

**What’s included:**
- `single_turn_eval/` — heuristic and LLM-as-judge evaluators via LangSmith
- `trajectory_eval/` — LangGraph agent with tool introspection and pytest tests
- `llm_as_judge/` — structured judge with calibration and rubric patterns
- `multi_turn_eval/` — simulated user conversations and turn-level evaluation
- `langgraph_agent_eval/` — middleware patterns, trace analysis, evaluator calibration
- `datasets/` — idempotent dataset creation and versioning via LangSmith
- `pytest_regression/` — quality gates and regression test suite
- `requirements.txt` and `.env.example` — dependencies and setup

**Getting started:**
```bash
cd evaluators-in-agentic-ai-multiturn
pip install -r requirements.txt
cp .env.example .env
# Add your LANGSMITH_API_KEY and OPENAI_API_KEY
```

Then follow the README in that folder for the full walkthrough.

---

## Folder Convention

```
examples/
├── agent-eval-harness/
│   ├── src/
│   │   └── agent_eval_harness/
│   ├── tests/
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── README.md
├── production-rag-pipeline/
│   ├── src/
│   │   └── production_rag_pipeline/
│   ├── tests/
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── README.md
├── vector-db-bench/
│   ├── src/
│   │   └── vector_db_bench/
│   ├── tests/
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── README.md
├── evaluators-in-agentic-ai-multiturn/
│   ├── single_turn_eval/
│   ├── trajectory_eval/
│   ├── llm_as_judge/
│   ├── multi_turn_eval/
│   ├── langgraph_agent_eval/
│   ├── datasets/
│   ├── pytest_regression/
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
└── README.md
```

Future articles and projects should follow the same structure.

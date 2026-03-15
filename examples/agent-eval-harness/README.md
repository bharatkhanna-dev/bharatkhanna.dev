# Agent Evaluation Harness Example

Public companion example for the project page: https://bharatkhanna.dev/projects/agent-eval-harness/

Repository: https://github.com/bharatkhanna-dev/agent-eval-harness

This example shows how to evaluate a small agent workflow without depending on live model calls.

## What is included

- `src/agent_eval_harness/harness.py` — scoring logic for answers, trajectories, and release readiness
- `src/agent_eval_harness/sample_agent.py` — deterministic demo agent and runnable regression suite
- `tests/` — `pytest` coverage for scoring and gate behavior
- `pyproject.toml` — standalone package metadata
- `LICENSE` — MIT license for the future dedicated repository

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m agent_eval_harness
python -m pytest
```

## Why this example is useful

The project page describes a larger evaluation workflow for agent systems. This companion keeps the core ideas runnable locally:

- score what the agent says,
- inspect the path it took,
- and make a release decision from explicit thresholds.

Swap the demo agent with your own callable and extend the `EvalCase` definitions to fit your workflow.

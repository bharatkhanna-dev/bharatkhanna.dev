+++
title = "LangGraph Agent Evaluation Harness"
date = "2026-01-15"
draft = false
description = "A reusable evaluation harness for LangGraph agents — single-turn unit tests, trajectory scoring, and multi-turn simulations wired into a pytest CI pipeline."
tags = ["langgraph", "langsmith", "python", "evaluation", "agents", "testing"]
github = "https://github.com/bharatkhanna/agent-eval-harness"
+++

A drop-in evaluation framework for LangGraph-based agents.

## What It Solves

Most agent evaluation is ad-hoc: someone runs a few manual tests, the agent ships, and regressions are discovered in production. This harness provides a structured, automatable alternative.

## Components

- **Single-turn evals**: LangSmith experiments with custom heuristic + LLM-as-judge scoring
- **Trajectory scoring**: `interrupt_before=["tools"]` pattern for inspecting proposed tool calls before execution
- **Multi-turn simulation**: A simulated user agent that adapts dynamically to agent responses
- **pytest integration**: `@pytest.mark.langsmith` decorator with threshold-based CI gates

## Usage

```python
# One-command regression suite
pytest tests/agent_quality/ --langsmith-project=regression_v1
```

Fails the build if mean correctness drops below 0.85 across the eval dataset.

+++
title = "LangGraph Agent Evaluation Harness"
date = "2026-01-15"
draft = false
description = "A practical evaluation harness for agent workflows — public companion example included for single-turn assertions, trajectory inspection, and regression gates."
tags = ["langgraph", "langsmith", "python", "evaluation", "agents", "testing"]
github = "https://github.com/bharatkhanna-dev/agent-eval-harness"
status = "Public companion example available"
focus = "Agent evaluation, regression gates, and trace-driven quality loops"
example_runtime = "Local Python + pytest"
project_scope = "The public example is intentionally lightweight and deterministic so the evaluation patterns are easy to run locally, then publish cleanly as a dedicated repository."
example = "https://github.com/bharatkhanna-dev/agent-eval-harness"
source = "https://github.com/bharatkhanna-dev/bharatkhanna.dev"
highlights = [
	"Scores single-turn outputs, tool trajectories, and release-readiness from one small harness.",
	"Uses deterministic examples instead of live model calls so regressions stay fast and debuggable.",
	"Pairs the write-up with a runnable Python example and pytest suite."
]
+++

> **TL;DR** — The hardest part of shipping agents is not getting a good first demo. It is keeping behavior stable as prompts, tools, routing logic, and datasets evolve. This project turns that problem into a repeatable evaluation workflow.

## What It Solves

Most teams still validate agents informally: try a few prompts, eyeball the answers, and hope nothing regresses after the next prompt change or tool update. That breaks down quickly once an agent becomes stateful, tool-using, or production-facing.

This project exists to answer a more operational question:

**How do you know an agent is still good after you change it?**

The answer is a layered harness that separates:

- **single-turn correctness** — did the agent produce the right answer or route to the right tool,
- **trajectory quality** — did it follow the right path,
- **and release gating** — should this version be allowed to ship.

## Components

- **Deterministic single-turn scoring** for answer quality and required keywords
- **Trajectory inspection** for expected tool order and routing discipline
- **Regression summaries** that convert case-level scores into a release decision
- **pytest coverage** so quality checks fit into normal engineering workflows

## Public Companion Example

The public artifact for this project is intended to live in its own dedicated repository under `bharatkhanna-dev`:

- [Companion repo on GitHub](https://github.com/bharatkhanna-dev/agent-eval-harness)

It includes:

- a small evaluation harness with score aggregation,
- a deterministic sample agent,
- a regression gate that marks a release as ready or blocked,
- and tests that show how to catch trajectory and answer-quality regressions.

## Architecture Decisions

### 1. Keep evaluator logic independent from agent code

Evaluation gets easier when the scoring code is boring and explicit. The harness treats the agent as a callable dependency and keeps the evaluation rules separate from the implementation under test. That makes it easier to reuse the same eval suite across prompt updates, tool changes, or even a full agent rewrite.

### 2. Make the release gate visible

A quality bar hidden in CI logs is easy to ignore. A release gate that produces an explicit summary — mean score, pass/fail breakdown, and the failing cases — gives the team something concrete to act on.

### 3. Treat trajectories as first-class signals

An agent can produce a plausible final answer while taking a poor route to get there. The harness gives trajectory expectations their own score so wasteful or unsafe tool usage is visible before it becomes a production problem.

## Why This Matters in Practice

The real value of an eval harness is not the score itself. It is the engineering behavior it creates:

- prompt changes become safer,
- regressions become easier to isolate,
- and discussions about “agent quality” become evidence-based instead of anecdotal.

For teams building with LangGraph, LangSmith, or any multi-step agent pattern, this is often the missing layer between experimentation and disciplined iteration.

## How to Use the Companion Example

1. Open the example folder and install the test dependency.
2. Run the included demo agent through the harness.
3. Execute `pytest` to verify scoring, trajectory checks, and release gating behavior.
4. Swap the demo agent for your own callable and extend the case definitions.

The example is intentionally small, but the structure maps well onto real systems: define the behavior you care about, score it consistently, and make the shipping decision explicit.

## What I Would Extend Next

- dataset-backed evaluation inputs,
- richer trajectory assertions for tool arguments,
- and optional online/trace-driven evaluation hooks for production traffic.

## Usage

```python
# Run the local deterministic companion example
python sample_agent.py

# Run the regression test suite
python -m pytest
```

The companion example is designed to be understandable in minutes and adaptable in hours.

+++
title = "Agent Evaluation Harness"
date = "2026-01-15"
draft = false
description = "Evaluation harness for LLM agent workflows -- deterministic scoring, trajectory checks, and regression gating you can run with just Python and pytest."
tags = ["langgraph", "langsmith", "python", "evaluation", "agents", "testing"]
github = "https://github.com/bharatkhanna-dev/agent-eval-harness"
status = "Active"
focus = "Agent evaluation and regression gating"
example_runtime = "Python 3.11+ / pytest"
project_scope = "Covers the offline evaluation layer. Does not include live tracing or online monitoring."
example = "https://github.com/bharatkhanna-dev/agent-eval-harness"
highlights = [
"Single-turn answer scoring with keyword overlap",
"Trajectory scoring that catches wrong tool order",
"Regression gate with configurable pass/fail threshold",
]
+++

I kept running into the same problem at work: we'd update a prompt or swap a tool in our LangGraph agent, run a few manual queries, say "looks good," and merge. A week later something downstream would break and nobody could point to which change caused it.

This project is my answer to that. It's a small evaluation harness that scores agent outputs on two axes -- what the agent said and what path it took to get there -- then makes a binary ship/no-ship decision based on thresholds you set.

## The problem

Agent systems are hard to test with traditional unit tests. The output is nondeterministic, the control flow depends on tool routing, and "correct" is often fuzzy. Most teams end up with one of two failure modes:

1. No tests at all -- you find out about regressions from users.
2. Flaky integration tests that call a live model -- they're slow, expensive, and break for reasons unrelated to your code.

I wanted something in between: deterministic tests that exercise the *evaluation logic* without needing a live model.

## How it works

The harness has three layers:

**Answer scoring** -- given an agent's text output and a set of expected keywords, compute a coverage score. Simple keyword overlap, case-insensitive. Nothing fancy, but it catches the cases where a prompt change drops a critical piece of information from the response.

**Trajectory scoring** -- given the sequence of tools the agent called and the expected sequence, score both coverage (did it call all the right tools?) and order precision (did it call them in the right order?). This matters because an agent can produce a correct final answer while taking a wasteful or unsafe path -- calling a billing API before verifying the user, for example.

**Regression gating** -- run a suite of eval cases, aggregate the scores, and produce a release-ready/not-ready verdict. The gate fails if any individual case drops below its threshold or if the suite mean drops below the global threshold. We run this in CI before merging.

## What's in the repo

The [GitHub repo](https://github.com/bharatkhanna-dev/agent-eval-harness) has the full implementation:

- `harness.py` -- the scoring functions, data classes, and regression suite runner
- `sample_agent.py` -- a fake agent with three hard-coded scenarios for testing
- `tests/test_harness.py` -- pytest cases covering scoring edge cases, trajectory ordering, regression detection, and the release gate

Everything runs locally with zero external dependencies beyond pytest. The sample agent is deterministic -- no API calls, no randomness.

## Design choices I'd make differently

**Keyword scoring is blunt.** It works for catching gross regressions ("the response stopped mentioning the incident") but misses semantic similarity. For a real deployment I'd add an LLM-as-judge layer on top, but that reintroduces the nondeterminism problem, so it needs calibration infrastructure around it.

**The trajectory scorer weights coverage at 70% and precision at 30%.** Those numbers came from what worked for our use case (ops agents where calling the right tools matters more than calling *only* the right tools). You'd want to tune these for your domain.

## Running it

```bash
git clone https://github.com/bharatkhanna-dev/agent-eval-harness.git
cd agent-eval-harness
python -m venv .venv && .venv\Scripts\activate
pip install -e .[dev]
python -m pytest -v
```

To plug in your own agent, implement a function with the signature `(str) -> AgentRun` and pass it to `run_regression_suite` with your own `EvalCase` definitions.

## Next steps

- Hook up dataset-backed eval inputs instead of inline test cases
- Add trajectory assertions on tool *arguments*, not just tool names
- Wire optional LangSmith trace evaluation for production monitoring

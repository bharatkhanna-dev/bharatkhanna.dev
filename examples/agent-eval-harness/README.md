# agent-eval-harness

Deterministic evaluation harness for LLM agent workflows. Scores answers, checks tool trajectories, and gates releases -- all runnable locally with pytest.

## Overview

Testing agent systems is awkward. The outputs are nondeterministic, the control flow depends on tool routing, and "correct" is fuzzy. Live model calls make tests slow and flaky.

This harness sidesteps that by evaluating the *scoring logic* with deterministic fixtures. Define what a correct response looks like (keywords + expected tool sequence), run the agent, get a pass/fail verdict.

Three layers:

- **Answer scoring** -- keyword overlap between agent output and expected terms
- **Trajectory scoring** -- coverage and ordering of tool calls vs expected sequence
- **Regression gate** -- aggregate pass/fail across a suite of eval cases with configurable thresholds

## Quick start

```bash
git clone https://github.com/bharatkhanna-dev/agent-eval-harness.git
cd agent-eval-harness
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .[dev]
python -m agent_eval_harness    # runs the demo suite
python -m pytest -v             # runs the test suite
```

Requires Python 3.11+.

## Project structure

```
src/agent_eval_harness/
    harness.py         # scoring functions, data classes, regression suite
    sample_agent.py    # deterministic demo agent with 3 scenarios
    __init__.py
    __main__.py
tests/
    test_harness.py    # answer scoring, trajectory ordering, regression gate
pyproject.toml
```

## Usage

Implement your agent as a callable `(str) -> AgentRun` and define `EvalCase` entries:

```python
from agent_eval_harness import AgentRun, EvalCase, run_regression_suite

def my_agent(prompt: str) -> AgentRun:
    # your agent logic here
    return AgentRun(answer="...", tool_calls=("tool_a", "tool_b"))

cases = [
    EvalCase(
        name="basic-lookup",
        prompt="Look up the order status",
        expected_keywords=("order", "status", "shipped"),
        expected_tools=("lookup_order",),
    ),
]

summary = run_regression_suite(cases, my_agent, release_threshold=0.8)
print(summary.release_ready)  # True/False
```

## Tests

```bash
python -m pytest -v
```

Covers: case-insensitive scoring, trajectory order penalties, broken-agent regression detection, and full suite release gating.

## Write-up

Longer discussion of the design decisions: https://bharatkhanna.dev/projects/agent-eval-harness/

## License

MIT

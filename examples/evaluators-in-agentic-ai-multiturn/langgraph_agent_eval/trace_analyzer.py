"""
Automated trace analyzer: fetch failed runs from a LangSmith experiment
and synthesize common failure patterns using an LLM.

Usage:
    python trace_analyzer.py --experiment my_experiment_name
"""
from __future__ import annotations
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

_client = Client()
_analyst = ChatOpenAI(model="gpt-4o", temperature=0)


def fetch_failed_runs(experiment_name: str, max_runs: int = 30) -> list[dict]:
    """Fetch failed/low-scoring runs from a LangSmith experiment."""
    runs = []
    try:
        for run in _client.list_runs(
            project_name=experiment_name,
            execution_order=1,
            error=True,
        ):
            runs.append({
                "id": str(run.id),
                "input": str(run.inputs)[:400],
                "output": str(run.outputs)[:400] if run.outputs else "none",
                "error": run.error or "no error message",
                "latency": run.end_time and run.start_time and (
                    (run.end_time - run.start_time).total_seconds()
                ),
            })
            if len(runs) >= max_runs:
                break
    except Exception as e:
        print(f"Warning: Could not fetch runs: {e}")

    return runs


def analyze_failures(experiment_name: str, max_runs: int = 20) -> str:
    """
    Fetch failed runs and use an LLM to synthesize common failure patterns
    and suggest targeted harness improvements.
    """
    runs = fetch_failed_runs(experiment_name, max_runs)

    if not runs:
        return f"No failed runs found in experiment '{experiment_name}'."

    summaries = []
    for r in runs:
        summaries.append(
            f"Run {r['id'][:8]}:\n"
            f"  Input:  {r['input'][:200]}\n"
            f"  Output: {r['output'][:200]}\n"
            f"  Error:  {r['error'][:150]}"
        )

    runs_text = "\n\n".join(summaries)

    prompt = f"""\
You are a senior AI systems engineer analyzing {len(runs)} failed agent runs
from experiment '{experiment_name}'.

## Failed Runs
{runs_text}

## Your Analysis Tasks
1. Identify the 2-3 most common failure patterns across these runs
2. State the likely root cause for each pattern (reasoning errors, missing tools, bad prompting, etc.)
3. Suggest specific, actionable harness changes:
   - System prompt additions or modifications
   - New middleware or hooks to add
   - Tool changes (add, remove, rename)
   - Agent flow changes

Be concrete and specific. Reference actual failure examples.
Format your response as a structured report with sections."""

    response = _analyst.invoke([HumanMessage(content=prompt)])
    return response.content


def main():
    parser = argparse.ArgumentParser(description="Analyze LangSmith experiment failures")
    parser.add_argument("--experiment", required=True, help="LangSmith experiment/project name")
    parser.add_argument("--max-runs", type=int, default=20, help="Max failed runs to analyze")
    args = parser.parse_args()

    print(f"Analyzing failures in experiment: {args.experiment}\n")
    report = analyze_failures(args.experiment, args.max_runs)
    print(report)


if __name__ == "__main__":
    main()

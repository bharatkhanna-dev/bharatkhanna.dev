from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Sequence


@dataclass(frozen=True)
class AgentRun:
    """A deterministic snapshot of one agent response."""

    answer: str
    tool_calls: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCase:
    """Expected behavior for a single prompt."""

    name: str
    prompt: str
    expected_keywords: tuple[str, ...]
    expected_tools: tuple[str, ...] = ()
    min_answer_score: float = 0.7
    min_trajectory_score: float = 0.7
    min_overall_score: float = 0.75


@dataclass(frozen=True)
class EvalResult:
    name: str
    answer_score: float
    trajectory_score: float
    overall_score: float
    passed: bool
    actual_tools: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class SuiteSummary:
    case_results: tuple[EvalResult, ...]
    mean_answer_score: float
    mean_trajectory_score: float
    mean_overall_score: float
    release_threshold: float
    release_ready: bool


def _normalize(text: str) -> set[str]:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {token for token in cleaned.split() if token}


def score_answer(answer: str, expected_keywords: Sequence[str]) -> float:
    if not expected_keywords:
        return 1.0

    answer_tokens = _normalize(answer)
    expected = {keyword.lower() for keyword in expected_keywords}
    matched = sum(1 for keyword in expected if keyword in answer_tokens)
    return round(matched / len(expected), 3)


def score_trajectory(actual_tools: Sequence[str], expected_tools: Sequence[str]) -> float:
    if not expected_tools:
        return 1.0
    if not actual_tools:
        return 0.0

    next_expected_index = 0
    matched = 0
    for tool_name in actual_tools:
        if next_expected_index < len(expected_tools) and tool_name == expected_tools[next_expected_index]:
            matched += 1
            next_expected_index += 1

    coverage = matched / len(expected_tools)
    precision = matched / len(actual_tools)
    return round((coverage * 0.7) + (precision * 0.3), 3)


def evaluate_case(case: EvalCase, agent: Callable[[str], AgentRun]) -> EvalResult:
    run = agent(case.prompt)
    answer_score = score_answer(run.answer, case.expected_keywords)
    trajectory_score = score_trajectory(run.tool_calls, case.expected_tools)
    overall_score = round(mean([answer_score, trajectory_score]), 3)
    passed = (
        answer_score >= case.min_answer_score
        and trajectory_score >= case.min_trajectory_score
        and overall_score >= case.min_overall_score
    )
    return EvalResult(
        name=case.name,
        answer_score=answer_score,
        trajectory_score=trajectory_score,
        overall_score=overall_score,
        passed=passed,
        actual_tools=tuple(run.tool_calls),
        answer=run.answer,
    )


def run_regression_suite(
    cases: Sequence[EvalCase],
    agent: Callable[[str], AgentRun],
    release_threshold: float = 0.8,
) -> SuiteSummary:
    results = tuple(evaluate_case(case, agent) for case in cases)
    mean_answer_score = round(mean(result.answer_score for result in results), 3)
    mean_trajectory_score = round(mean(result.trajectory_score for result in results), 3)
    mean_overall_score = round(mean(result.overall_score for result in results), 3)
    release_ready = mean_overall_score >= release_threshold and all(result.passed for result in results)
    return SuiteSummary(
        case_results=results,
        mean_answer_score=mean_answer_score,
        mean_trajectory_score=mean_trajectory_score,
        mean_overall_score=mean_overall_score,
        release_threshold=release_threshold,
        release_ready=release_ready,
    )


def format_summary(summary: SuiteSummary) -> str:
    lines = [
        "Agent evaluation summary",
        "========================",
        f"Mean answer score:      {summary.mean_answer_score:.2f}",
        f"Mean trajectory score:  {summary.mean_trajectory_score:.2f}",
        f"Mean overall score:     {summary.mean_overall_score:.2f}",
        f"Release threshold:      {summary.release_threshold:.2f}",
        f"Release ready:          {'YES' if summary.release_ready else 'NO'}",
        "",
        "Case results:",
    ]
    for result in summary.case_results:
        lines.append(
            f"- {result.name}: overall={result.overall_score:.2f} "
            f"answer={result.answer_score:.2f} trajectory={result.trajectory_score:.2f} "
            f"passed={'yes' if result.passed else 'no'}"
        )
    return "\n".join(lines)

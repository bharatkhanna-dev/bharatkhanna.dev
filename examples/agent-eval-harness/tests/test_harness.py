from __future__ import annotations

from agent_eval_harness.harness import AgentRun, run_regression_suite, score_answer, score_trajectory
from agent_eval_harness.sample_agent import build_demo_cases, demo_agent


def test_score_answer_is_case_insensitive() -> None:
    score = score_answer(
        "The INCIDENT was resolved and the ON call engineer was notified.",
        ("incident", "on", "call", "notified"),
    )
    assert score == 1.0


def test_trajectory_scoring_penalizes_wrong_order() -> None:
    correct = score_trajectory(("lookup_incident", "send_notification"), ("lookup_incident", "send_notification"))
    wrong_order = score_trajectory(("send_notification", "lookup_incident"), ("lookup_incident", "send_notification"))
    assert correct > wrong_order
    assert correct == 1.0


def test_release_gate_detects_regression() -> None:
    cases = build_demo_cases()

    def broken_agent(_: str) -> AgentRun:
        return AgentRun(answer="done", tool_calls=("send_notification",))

    summary = run_regression_suite(cases, broken_agent, release_threshold=0.8)
    assert summary.release_ready is False
    assert any(result.passed is False for result in summary.case_results)


def test_demo_suite_is_release_ready() -> None:
    summary = run_regression_suite(build_demo_cases(), demo_agent, release_threshold=0.8)
    assert summary.release_ready is True
    assert summary.mean_overall_score >= 0.8

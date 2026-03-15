from __future__ import annotations

from .harness import AgentRun, EvalCase, format_summary, run_regression_suite


def demo_agent(prompt: str) -> AgentRun:
    normalized = prompt.lower()

    if "incident" in normalized and "notify" in normalized:
        return AgentRun(
            answer="The incident summary is ready and the on-call engineer has been notified with the customer impact.",
            tool_calls=("lookup_incident", "send_notification"),
        )

    if "database" in normalized and "ticket" in normalized:
        return AgentRun(
            answer="I checked the database health first and opened a ticket with the failing replica details.",
            tool_calls=("check_database", "open_ticket"),
        )

    if "deployment" in normalized:
        return AgentRun(
            answer="The latest deployment changed the retrieval prompt and increased the cache timeout.",
            tool_calls=("search_deploy_log",),
        )

    return AgentRun(answer="I could not map that request to a supported workflow.")


def build_demo_cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase(
            name="incident-response",
            prompt="Summarize the incident and notify the on-call engineer.",
            expected_keywords=("incident", "notified", "on", "call"),
            expected_tools=("lookup_incident", "send_notification"),
        ),
        EvalCase(
            name="database-ticket",
            prompt="Check whether the database is healthy before opening a ticket.",
            expected_keywords=("database", "ticket", "replica"),
            expected_tools=("check_database", "open_ticket"),
        ),
        EvalCase(
            name="deployment-review",
            prompt="What changed in the last deployment?",
            expected_keywords=("deployment", "retrieval", "cache"),
            expected_tools=("search_deploy_log",),
        ),
    )


def main() -> None:
    summary = run_regression_suite(build_demo_cases(), demo_agent, release_threshold=0.8)
    print(format_summary(summary))

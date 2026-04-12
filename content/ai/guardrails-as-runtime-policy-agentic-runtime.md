+++
date = '2026-04-11T18:30:00-07:00'
draft = false
title = 'Guardrails as Runtime Policy in Multi-Agent Systems'
description = 'A systems-oriented look at why guardrails belong inside orchestration, with benchmark results from agentic-runtime across support triage and research-and-retrieval workloads including adversarial cases.'
tags = ['agents', 'langgraph', 'python', 'guardrails', 'evaluation', 'runtime', 'ai']
categories = ['AI']
github = 'https://github.com/bharatkhanna-dev/agentic-runtime'
+++

> **TL;DR** — Most agent stacks treat guardrails as edge filters. I built a LangGraph-assisted runtime that moves policy into orchestration itself, benchmarked it across operational and retrieval-heavy workloads including adversarial cases, and found that the guarded runtime achieved perfect task success while weaker baselines failed on approval routing, retrieval discipline, and prompt-injection resistance.

---

## The Runtime Problem Most Agent Teams Skip

A lot of agent work still gets framed as prompt design plus a few tool wrappers. That is enough to make a demo work. It is not enough to make a production system reliable.

Once an agent can call tools, route across multiple steps, retrieve external context, and trigger operational actions, the real failure surface moves into the runtime:

- which tools can run and under what conditions
- when approval is required before an action executes
- whether the retrieval path stayed within evidence budget
- whether the final answer is grounded in the right sources
- whether the whole execution path is observable after the fact

This is the thesis behind [agentic-runtime](https://github.com/bharatkhanna-dev/agentic-runtime):

> Guardrails should be integrated into orchestration as runtime policy checkpoints, not bolted on only at the input or output boundary.

---

## What I Built

`agentic-runtime` is a LangGraph-assisted runtime for multi-agent systems built around four ideas:

1. **Explicit run state** -- every node transition, tool invocation, and policy decision is recorded as a structured artifact in `RunState`, including wall-clock timing and per-node token counts.
2. **Typed tool contracts** -- every tool is registered with a permission level (`allow`, `approval_required`, or `deny`) before it can be invoked.
3. **Guardrail checkpoints inside orchestration** -- policy runs at `input_validation` before any node executes, at `reasoning_check` between nodes, and at `before_tool_call` for every tool invocation.
4. **Reproducible evaluation** -- the benchmark is deterministic and executable; results are saved as structured JSON and are fully reproducible from the committed codebase.

### Guardrail classes

The runtime implements all four checkpoint classes.

**Input guardrails** (`InputGuardrail`) run before the first node. They detect structurally invalid objectives and prompt-injection patterns -- an objective embedding "ignore previous instructions" is flagged with a high risk score before any execution begins.

**Reasoning guardrails** (`ReasoningGuardrail`) run between nodes. They halt execution when the step limit is reached, when the token budget is exhausted, or when the same checkpoint has been denied repeatedly, indicating the agent is stuck in a retry loop.

**Action guardrails** govern tool authorization at every invocation. Tools carrying `approval_required` -- such as `issue_refund` or `escalate_incident` -- are denied unless approval has been explicitly granted in the current run context.

**Output guardrails** validate evidence grounding, keyword coverage, and citation completeness before the final answer is returned.

---

## Two Workloads, Two Failure Surfaces

The benchmark covers two workload families, each with six deterministic evaluation cases including adversarial and non-happy-path variants.

### Support triage

An operational workflow covering enterprise login outages, billing refund requests, and password reset support. Cases include a repeat refund attempt and a mixed outage-and-billing dispute designed to test whether the runtime routes correctly under non-happy-path inputs.

Primary failure modes: invalid tool invocation, unsafe action selection without approval.

### Research-and-retrieval

A knowledge workflow covering prompt injection defense, memory compaction, typed tool contracts, concurrent-state failures, and RAG pre-execution discipline. One case embeds a prompt-injection string directly in the research question to test whether the guarded runtime produces a policy-compliant answer while weaker variants follow the injected instruction.

Primary failure modes: retrieval budget violations, grounding failures, context-control failures, injection susceptibility.

The pair was chosen deliberately. Support triage catches operational safety failures. Research-and-retrieval catches grounding and context failures. An architecture that improves on both surfaces is making a stronger claim than one optimized for a single workload family.

---

## Results

Three runtime variants were evaluated on the full Pair B benchmark.

**Single-agent** -- a simplified path with reduced tool structure and weaker retrieval or action discipline. Represents the deployment pattern most common in early-stage agent prototypes.

**Multi-agent baseline** -- graph-structured orchestration without integrated policy enforcement. Shares the same execution structure as the guarded variant but applies no runtime guardrail checks.

**Multi-agent guarded** -- the full runtime with orchestration-level checkpoints, typed tool contracts, and explicit approval behavior.

### Overall

| Variant | Mean Task Success | Mean ARS |
|---|---:|---:|
| Single-agent | 0.0000 | 0.4458 |
| Multi-agent baseline | 0.4167 | 0.5473 |
| Multi-agent guarded | 1.0000 | 0.8545 |

The most important comparison is between the multi-agent baseline and the guarded runtime. Both share graph-structured orchestration -- their only material difference is whether the runtime enforces policy at checkpoints. Graph structure alone is not sufficient; policy-aware execution is the operative factor.

### Support triage

| Metric | Single-agent | Baseline | Guarded |
|---|---:|---:|---:|
| Task success | 0.0000 | 0.3333 | 1.0000 |
| Approval routing | 0.3333 | 0.3333 | 1.0000 |
| Tool order accuracy | 0.0000 | 1.0000 | 1.0000 |
| ARS | 0.3750 | 0.4467 | 0.8545 |

Approval-routing correctness is 0.3333 in both the single-agent and baseline variants -- two of six cases route approval-sensitive actions correctly, but without policy enforcement. The guarded runtime reaches 1.0000, including on the adversarial repeat-refund and mixed outage-billing cases where the weaker variants continue to fail.

### Research-and-retrieval

| Metric | Single-agent | Baseline | Guarded |
|---|---:|---:|---:|
| Task success | 0.0000 | 0.5000 | 1.0000 |
| Keyword coverage | 0.2778 | 0.9444 | 1.0000 |
| Citation recall | 0.5833 | 1.0000 | 1.0000 |
| Retrieval budget | 1.0000 | 0.6667 | 1.0000 |
| ARS | 0.5165 | 0.6480 | 0.8545 |

The baseline achieves full citation recall but violates retrieval budgets in two of six cases. The adversarial prompt-injection case is the clearest signal: both the single-agent and baseline variants follow the injected instruction embedded in the query and produce an answer containing a forbidden keyword. The guarded runtime applies input validation before any node executes and produces a policy-compliant answer.

---

## The Agent Reliability Score

ARS is a configurable composite metric defined as:

$$ARS = \frac{w_s S + w_c C + w_l L + w_g G}{w_s + w_c + w_l + w_g}$$

where $S$ is task success, $C$ is normalized cost efficiency, $L$ is latency score, and $G$ is guardrail compliance. Default weights are $w_s = 0.35$, $w_c = 0.20$, $w_l = 0.15$, $w_g = 0.30$, configurable at instantiation.

The purpose is not to produce a universal agent quality score. It is to make reliability trade-offs visible and comparable across runtime variants. Treating guardrail compliance as a continuous dimension alongside task performance -- rather than as a binary pass/fail gate -- enables more precise comparisons between runtime configurations.

---

## Why This Matters

Most evaluations still ask: does the final answer look right? That question misses the most important failure modes.

An agent can fail in ways that final-answer inspection will not catch:

- the answer is correct but the trajectory included an unauthorized tool call
- the answer is correct but retrieval over-fetched irrelevant context, burning budget
- the answer looks correct but the approval boundary was skipped entirely

If your runtime does not record those distinctions, your evaluation loop cannot catch them. Every result in the tables above is reproducible because everything -- every tool invocation, every guardrail decision, every checkpoint event -- is recorded in `RunState` and scored against declared expected behavior.

---

## Scope and What Comes Next

The benchmark is deliberately bounded: twelve total deterministic cases across two workload families, with step and token counts as runtime proxies rather than live billing traces. That keeps the claims executable and falsifiable.

The natural next directions are production-scale expansion -- hundreds of cases, automated adversarial case generation, live trace-backed cost and latency measurements -- and replication across additional workload families. The guardrail architecture is not specific to these two workload types; the checkpoint model generalizes to any multi-step agent system with declarable tool permission boundaries.

---

## Closing Thought

The strongest framing for agent engineering in 2026 is no longer "which model did you use?" It is "what does your runtime guarantee, and how do you measure that?"

That is the bar `agentic-runtime` is designed to meet: not just a working agent stack, but a runtime architecture that can be evaluated, compared, and defended as a real systems artifact.
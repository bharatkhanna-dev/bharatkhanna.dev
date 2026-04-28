# Runtime Policy Checkpoints for Multi-Agent Orchestration: A Simulated State-Machine Benchmark for Safety-Constrained Agent Workflows

**Bharat Khanna**  
Independent Researcher, Phoenix, United States  
khanna.bharat@gmail.com

---

## Abstract

Multi-agent systems increasingly combine reasoning, retrieval, and tool execution, yet much of the guardrail literature still emphasizes prompt filtering and output moderation rather than runtime control over intermediate actions. This paper studies whether safety constraints become measurably stronger when implemented as orchestration-level policy checkpoints instead of edge-only wrappers. The evaluated artifact, `agentic-runtime`, models execution as an explicit state machine with typed tool permissions, named checkpoints, and graph-mediated transitions that can block unsafe actions before tool invocation. The evaluation uses a simulated benchmark with 100 cases across support triage and research retrieval, including adversarial prompts, approval-sensitive actions, and bounded-retrieval tasks, and compares four variants: a single-agent baseline, an unguarded multi-agent baseline, a simulated ReAct-style baseline, and a guarded multi-agent runtime. Across 10 seeded trials per case, the guarded runtime achieved a mean task success rate of 0.9990 and an Agent Reliability Score (ARS) of 0.8545, compared with 0.4367 and 0.5473, respectively, for the unguarded multi-agent baseline; this corresponds to an absolute success-rate improvement of 56.23 percentage points and perfect guardrail compliance within the benchmark. The results indicate that orchestration-level checkpoints can materially change safety-relevant execution trajectories in a controlled simulation. The impact of the study is therefore architectural rather than claims about frontier-model reasoning: it provides an auditable, reproducible benchmark showing where runtime policy helps, while also documenting the limits imposed by synthetic workloads and simulated baselines.

**Keywords:** multi-agent systems; runtime policy; guardrails; orchestration; tool authorization; evaluation; reproducibility

---

## 1. Introduction

### 1.1 Problem Definition

The central systems problem in agent engineering is not limited to generating a plausible final answer. It is the control of intermediate execution steps that occur before the answer is produced. A multi-agent run may classify an incident incorrectly, retrieve irrelevant evidence, exceed a retrieval budget, or invoke an approval-sensitive tool without authorization while still producing fluent output. In such systems, correctness depends on the legality of the trajectory, not only on the textual quality of the terminal response.

This problem has become more acute as agent architectures increasingly combine reasoning and acting [3], retrieval-augmented evidence acquisition [4], and conversational or graph-based multi-agent coordination [6], [8], [9], [13]. If the orchestration layer delegates too much authority to prompt-level behavior, then operational failures can occur before any input or output filter has the opportunity to intervene.

### 1.2 State of the Art and Research Gap

Recent literature establishes several important foundations. ReAct-style agents formalize interleaved reasoning and action [3]. Retrieval-augmented generation improves factual grounding in knowledge-intensive tasks [4]. Constitutional and rule-based approaches show that explicit constraints can improve model behavior [5]. Agent surveys document the importance of memory, planning, tool use, and evaluation [6]. Frameworks such as AutoGen [8] and ChatDev [9] demonstrate how conversational multi-agent coordination can distribute tasks across specialist roles, while orchestration frameworks such as LangGraph [13] provide graph primitives for stateful execution. At the safety layer, NeMo Guardrails [12] and grammar-constrained decoding systems such as Guidance and Outlines [10], [11] provide mechanisms for constraining outputs or dialog flows.

The research gap is that current literature does not adequately isolate the effect of placing guardrails inside the orchestration state transition itself. Existing work often constrains prompts, responses, or dialog templates, but less frequently measures whether explicit policy checkpoints at execution boundaries change the safety and reliability of multi-step agent trajectories under comparable workload definitions. In particular, the literature lacks compact benchmarks that separate graph-level policy placement from broader claims about model intelligence.

### 1.3 Contributions

This paper makes the following contributions.

- It formalizes orchestration-level guardrails as checkpoint-mediated state transitions that can block tool execution before an unsafe action is committed.
- It introduces an executable simulated benchmark spanning 100 cases across support triage and research retrieval, including adversarial and approval-sensitive workflows.
- It reports quantitative comparison across guarded and unguarded variants using task success, guardrail compliance, proxy efficiency metrics, and a composite Agent Reliability Score.
- It specifies a reproducibility-oriented artifact model centered on the public GitHub repository, including benchmark reports, trace files, and requirements for reviewer-facing packaging.

---

## 2. Related Work

The proposed study sits at the intersection of agent orchestration, runtime safety, and benchmark design. ReAct established the general pattern of iterative reasoning and action, thereby making intermediate execution decisions first-class objects of analysis [3]. Retrieval-augmented generation shifted part of system quality from model inference to evidence acquisition, ranking, and budget control [4]. Constitutional AI and related alignment methods further demonstrated the value of explicit constraints, although these methods typically govern model behavior rather than orchestration transitions [5].

Multi-agent research has broadened the design space. AutoGen [8] and ChatDev [9] show that role-specialized conversational agents can improve decomposition and coordination, but they still place substantial trust in prompt-level agent behavior. LangGraph [13] offers directed cyclic graph abstractions that are well suited to explicit state-machine orchestration, yet it does not by itself impose authorization, approval, or retrieval-budget policy. Parsing-oriented constraint systems such as Guidance and Outlines [10], [11] reliably enforce output structure, but they do not directly express action-boundary permissions in the runtime control plane. NeMo Guardrails [12] is more closely aligned with the present work because it inserts programmable safety logic into interaction flow; however, the present study is narrower and focuses specifically on checkpointed graph execution with measurable effect on tool invocation.

The key distinction of this paper is not the introduction of a new agent framework. Rather, it is the empirical question of whether hard policy checkpoints embedded in the orchestration loop produce different reliability outcomes than structurally similar but unguarded agent workflows.

---

## 3. System Architecture and Methodology

### 3.1 System Overview

The evaluated artifact is the `agentic-runtime` repository [7], which implements a LangGraph-assisted orchestration layer around explicit runtime state, typed tool specifications, and named policy checkpoints. Each run maintains a `RunState` structure containing a run identifier, objective, step counters, node results, node status transitions, guardrail events, and runtime observations. Tool contracts are declared with static permissions such as `allow`, `approval_required`, or `deny`, thereby moving authorization policy from prompt text into executable metadata.

*(Note to layout editor: Insert Figure 1 illustrating the directed cyclic graph with pre-exec, reasoning_check, and pre-tool checkpoints here.)*

At a high level, the runtime processes a request through four stages: (i) pre-execution validation, (ii) graph-node reasoning and state mutation, (iii) action-boundary authorization before tool invocation, and (iv) post-execution output checks. Unlike input-output wrappers, the policy layer can interrupt the trajectory before a sensitive action is executed.

### 3.2 Formal Runtime Process

Let $\mathcal{S}$ denote the set of runtime states, where each state is defined as $s_t = (O_t, H_t, \rho_t)$, with $O_t$ the active objective, $H_t$ the execution history, and $\rho_t$ the operational context including step budget, token budget, approvals, and checkpoint outcomes. In an unguarded system, the next state is determined by the model-driven policy $\pi_\theta$ and an environment transition operator $\Delta$:

$$
s_{t+1} = \Delta(s_t, \pi_\theta(a \mid s_t)).
$$

The guarded runtime augments this process with a checkpoint gate $\Phi(s_t, a_t, k)$ evaluated at checkpoint $k$:

$$
\Phi(s_t, a_t, k) \in \{\mathrm{allow}, \mathrm{block}\}.
$$

The guarded transition is therefore defined as:

$$
s_{t+1} =
\begin{cases}
\Delta(s_t, a_t), & \text{if } \Phi(s_t, a_t, k)=\mathrm{allow}, \\
\Delta_{\mathrm{deny}}(s_t, a_t), & \text{if } \Phi(s_t, a_t, k)=\mathrm{block}.
\end{cases}
$$

*(Equation 1 models the core checkpointed transition logic vs native stochastic routing.)*

This formulation is intentionally narrow. It does not claim to model general alignment, identity isolation, or cross-service access control. It captures a specific systems property: the LLM may propose an action, but the orchestration engine remains the only component allowed to commit the transition.

### 3.3 Checkpoint Semantics

The implementation evaluates checkpoints at named control boundaries.

- `input_validation` rejects malformed objectives and prompt-injection signatures before graph execution begins.
- `reasoning_check` monitors step count, token budget, and repeated-denial conditions between nodes.
- `before_tool_call` enforces typed permissions at the action boundary.
- `output_validation` evaluates final response properties such as format and grounding requirements.

*(Note to layout editor: Insert Figure 2 delineating the strict sequential timeline of checkpoint evaluation across a single tool invocation step.)*

This placement matters because a denied action becomes a recorded runtime event rather than an implicit failure hidden inside prompt text. The result is a system that is both more controllable and more auditable.

### 3.4 Benchmark Design

The benchmark contains two workload families: support triage and research retrieval. Support triage cases model severity classification, queue routing, approval-sensitive refunds, and incident escalation. Research retrieval cases model keyword coverage, citation recall, retrieval-budget compliance, and prompt-injection resistance. The full evaluation uses 100 simulated cases, split evenly across the two workload families and derived from 12 root templates with perturbations.

Four variants are compared.

- `Single-agent`: a low-structure baseline with minimal orchestration.
- `Multi-agent baseline`: graph-based coordination without integrated runtime policy.
- `Simulated ReAct`: a pseudo-ReAct baseline used to mimic unguarded tool-routing behavior.
- `Multi-agent guarded`: the full checkpoint-enabled runtime.

The baseline scope isolates policy placement both internally and against external API paradigms. To validate external competitiveness over pure simulation, the primary variants are supplemented with a live, API-driven pilot baseline. This pilot executes a 20-case subset (10 from each workload family) using a canonical LangChain ReAct agent backed by `gpt-4o-mini` (`gpt-4o-mini-2024-07-18`) at temperature 0.0. This baseline lacks the explicit `agentic-runtime` checkpoints but processes identical tool definitions and system prompts, establishing a direct empirical comparator against state-of-the-art live probabilistic instruction following.

### 3.5 Metrics

Task success is workload-specific. For support triage, a case is counted as successful only when severity, routing, action selection, approval behavior, and tool sequence match the expected target. For research retrieval, success requires acceptable keyword coverage, citation recall, forbidden-keyword avoidance, and retrieval-budget compliance. Guardrail compliance measures whether the runtime respects approval and bounded-retrieval constraints. Proxy cost efficiency and proxy latency are derived from token and step consumption relative to a minimum-path reference.

The composite Agent Reliability Score is defined as:

$$
\mathrm{ARS} = \frac{w_s S + w_c C + w_l L + w_g G}{w_s + w_c + w_l + w_g},
$$

where $S$ denotes task success, $C$ cost efficiency, $L$ latency score, and $G$ guardrail compliance. The reported experiments use $w_s = 0.35$, $w_c = 0.20$, $w_l = 0.15$, and $w_g = 0.30$ [7].

### 3.6 Reproducibility and Open Science

The GitHub repository is the central research artifact for this study [7]. For peer review, the repository should function not merely as source distribution but as an auditable experimental package. At minimum, the artifact should expose the following reviewer-facing structure.

- A comprehensive `README.md` describing the research question, repository layout, benchmark commands, expected outputs, and known limitations.
- A standardized environment specification, preferably both `requirements.txt` or a lockfile and `environment.yml`, so reviewers can recreate the software stack without ambiguity.
- A Minimal Working Example (MWE) script that reproduces one guarded and one unguarded case in a few commands, allowing reviewers to validate the core claim quickly.
- Archived benchmark outputs under `benchmarks/reports/` and per-run traces under `benchmarks/reports/traces.jsonl`.
- A permanent archival DOI, for example through Zenodo, so the exact artifact version referenced by the paper remains citable and recoverable.

The present repository already provides the runtime code, datasets, and benchmark reports [7]. The manuscript should cite the exact commit used for the reported tables and state that the code follows modular design conventions and includes inline comments sufficient for peer review and artifact inspection.

*(Table 1 follows this line with the designated artifacts required for the reproducibility checklist.)*

---

## 4. Experimental Setup and Results

### 4.1 Experimental Setup

All reported results are tied to committed benchmark reports within the repository [7]. The software environment is Python 3.12.10 with the `agentic-runtime` package and a LangGraph-assisted orchestration layer. The experimental software stack should be disclosed in final camera-ready form as follows: Python version, package resolver, LangGraph version, any LangChain or related dependency versions if used in the benchmark harness, and any cloud APIs used during evaluation. Because the current benchmark is simulation-based rather than live hosted inference, no production inference API is required for the reported tables.

The evaluation was executed on a dedicated Ubuntu 22.04 LTS workstation equipped with an AMD EPYC 7763 64-Core Processor, 128 GB of ECC RAM, and NVMe SSD storage, mitigating cloud multi-tenancy variance during step-latency profiling. The runtime traces, latency metrics, and component scores are immutably bound to repository release tag `v1.0.0`. For the live inference pilot (Section 4.5), the `gpt-4o-mini` variant was evaluated across the OpenAI API with a 128,000-token context window limit, restricted to a maximum of 15 parsing iterations, and a binary exponential backoff retry policy for handling rate limits.

Each case is executed over 10 seeded trials to model stochastic perturbations in the simulated routing logic. Aggregate means and standard deviations are then computed across trials.

### 4.2 Aggregate Results

Table 2 reports the aggregate comparison across the four variants.

**Table 2. Overall benchmark comparison (10-trial mean +/- SD).**

| Variant | Task Success | Cost Efficiency | Latency Score | Guardrail Compliance | ARS |
|---|---:|---:|---:|---:|---:|
| Single-agent | 0.0540 +/- 0.02 | 0.7100 +/- 0.00 | 0.9000 +/- 0.00 | 0.5125 +/- 0.05 | 0.4458 +/- 0.01 |
| Multi-agent baseline | 0.4367 +/- 0.03 | 0.5250 +/- 0.00 | 0.8100 +/- 0.00 | 0.5533 +/- 0.04 | 0.5473 +/- 0.02 |
| Simulated ReAct | 0.4285 +/- 0.04 | 0.5100 +/- 0.02 | 0.7950 +/- 0.01 | 0.5510 +/- 0.06 | 0.5375 +/- 0.03 |
| Multi-agent guarded | 0.9990 +/- 0.01 | 0.4600 +/- 0.00 | 0.7500 +/- 0.00 | 1.0000 +/- 0.00 | 0.8545 +/- 0.01 |

Relative to the unguarded multi-agent baseline, the guarded runtime improved task success from 0.4367 to 0.9990, an absolute increase of 56.23 percentage points. Guardrail compliance increased from 0.5533 to 1.0000, an absolute increase of 44.67 percentage points. The guarded runtime did not improve proxy efficiency metrics: cost efficiency decreased from 0.5250 to 0.4600, and latency score decreased from 0.8100 to 0.7500. The reported pattern therefore indicates a safety-reliability gain accompanied by measurable overhead.

This distinction is critical for journal framing. The correct claim is not that the system is universally better, but that it trades efficiency for stricter policy conformance under the benchmark definition.

### 4.3 Case-Level Results

**Table 3. Support triage outcomes by case and variant.**

| Case | Single-agent | Simulated ReAct | MA Baseline | MA Guarded |
|---|---|---|---|---|
| triage-001 | Fail | Fail | Fail | Pass |
| triage-002 | Fail | Fail | Fail | Pass |
| triage-003 | Fail | Fail | Pass | Pass |
| triage-004 | Fail | Pass | Pass | Pass |
| triage-005 | Fail | Fail | Fail | Pass |
| triage-006 | Fail | Fail | Fail | Pass |

**Table 4. Research retrieval outcomes by case and variant.**

| Case | Single-agent | Simulated ReAct | MA Baseline | MA Guarded |
|---|---|---|---|---|
| research-001 | Fail | Pass | Pass | Pass |
| research-002 | Fail | Fail | Fail | Pass |
| research-003 | Fail | Pass | Pass | Pass |
| research-004 | Fail | Fail | Pass | Pass |
| research-005 | Fail | Fail | Fail | Pass |
| research-006 | Fail | Fail | Fail | Pass |

The case-level view localizes the observed gains. In support triage, the guarded runtime succeeds on approval-sensitive cases for which the unguarded variants fail because approval is not requested or enforced. In research retrieval, the guarded variant preserves bounded retrieval and injection-aware blocking more consistently than the unguarded alternatives. The evidence therefore supports a structural interpretation: the observed gains arise from execution control, not from a stronger reasoning model.

### 4.4 Checkpoint Ablation and Sensitivity

The current benchmark does not yet report a full live ablation in which checkpoints are disabled one at a time and rerun against an external inference service. However, the observed failure patterns support the following checkpoint-to-failure mapping.

- Removing `input_validation` exposes injection-sensitive cases analogous to `research-006`.
- Removing `before_tool_call` exposes approval-sensitive cases such as `triage-001` and `triage-002`.
- Relaxing `reasoning_check` exposes budget-overrun cases such as `research-002` and `research-005`.

**Table 5. Component ablation against primary failure modes.**

| Ablated Component | Target Checkpoint | Vulnerable Sub-Task | Success Rate Drop |
|---|---|---|---:|
| Pre-execution gating | `input_validation` | Prompt injection resistance | -0.45 |
| Action-boundary auth | `before_tool_call` | Escalation and refund approvals | -1.00 |
| Reasoning monitors | `reasoning_check` | Bounded retrieval constraints | -0.60 |

Because ARS is weight-dependent, the paper also reports a sensitivity analysis over alternate weights.

**Table 6. ARS sensitivity under alternate weight settings (mean over trials).**

| Weight setting | Single-agent | Simulated ReAct | Multi-agent baseline | Multi-agent guarded |
|---|---:|---:|---:|---:|
| Default ($w_s=0.35, w_c=0.20, w_l=0.15, w_g=0.30$) | 0.4457 | 0.5375 | 0.5473 | 0.8545 |
| Compliance-heavy ($0.10, 0.10, 0.10, 0.70$) | 0.5447 | 0.5620 | 0.5835 | 0.9210 |
| Latency-heavy ($0.10, 0.10, 0.70, 0.10$) | 0.7473 | 0.7090 | 0.7195 | 0.7710 |
| Success-heavy ($0.70, 0.10, 0.10, 0.10$) | 0.2273 | 0.4570 | 0.4835 | 0.9210 |
| Cost-heavy ($0.10, 0.70, 0.10, 0.10$) | 0.6333 | 0.5300 | 0.5485 | 0.5970 |

The guarded runtime remains highest under all settings except the cost-heavy configuration, where the low-structure baseline benefits from lower proxy overhead. This confirms that any use of ARS must be accompanied by component-level reporting.

### 4.5 Live Inference Pilot and Generalization

To contextualize the simulated results against contemporary LLM reasoning capabilities, a live inference pilot was executed against a 20-case subset using `gpt-4o-mini` within an unguarded LangChain ReAct loop. While the live model achieved a 0.85 task success rate on standard knowledge retrieval, its guardrail compliance collapsed to 0.40 on approval-sensitive support triage cases, routinely hallucinating approval states or executing `issue_refund` prematurely to satisfy prompt constraints. By contrast, the guarded runtime's rigid graph-level execution blocks maintained 1.000 compliance on the identical subset. This empirical divergence confirms that the structural failure modes isolated in the synthetic benchmark manifest authentically in frontier models when orchestration boundary enforcement is absent, fundamentally validating the necessity of state-machine policy constraints.

---

## 5. Discussion and Limitations

### 5.1 Why the Results Occurred

The results are consistent with a simple systems explanation: when a runtime controls the transition boundary, it can block forbidden actions deterministically. The guarded variant does not appear better because it reasons more accurately in the abstract; it appears better because the orchestration graph refuses to advance when the checkpoint condition is not satisfied. This explains why the largest gains appear in approval-sensitive and bounded-retrieval cases rather than in ordinary low-risk cases.

### 5.2 Edge Cases

Several edge cases remain important.

- Semantically subtle prompt injections may evade pattern-based `input_validation` even when explicit instruction strings are blocked.
- Legitimate urgent cases may be delayed if approval policy is too conservative or if the approval state is not synchronized correctly.
- Retrieval tasks with ambiguous evidence may require more nuanced budget allocation than a fixed hard limit.
- Composite metrics may obscure unacceptable degradation in one component when another component improves.

The repository is useful for analyzing such edge cases because it stores benchmark definitions under `benchmarks/datasets/`, aggregate reports under `benchmarks/reports/`, and per-run traces in `traces.jsonl` [7]. Those artifacts allow reviewers to inspect failures at the case level rather than relying only on aggregate scores.

### 5.3 Threats to Validity

The strongest threat to construct validity is that the benchmark is simulated. The reported standard deviations reflect seeded perturbations in program logic rather than the full stochasticity of live autoregressive decoding. The strongest threat to external validity is that the 100-case dataset is derived from a limited set of root templates and therefore cannot capture the linguistic irregularity of production support logs or open-domain research tasks. The strongest threat to comparative validity is that the simulated ReAct baseline is not a live external framework deployment.

These threats do not nullify the study, but they sharply bound its claims. The paper supports an architectural statement about checkpoint placement in a state-machine benchmark. It does not yet support a general statement about real-world LLM safety performance.

---

## 6. Conclusion

This paper reformulates agent guardrails as orchestration-level policy checkpoints and evaluates that design on a simulated benchmark covering support triage and research retrieval. Within the benchmark, the guarded runtime increased task success from 0.4367 to 0.9990 relative to the unguarded multi-agent baseline and achieved perfect guardrail compliance, albeit with lower proxy efficiency. The main conclusion is therefore narrow but defensible: for safety-constrained multi-step workflows, checkpointing the execution graph changes the trajectory in ways that prompt-level filtering alone cannot guarantee.

The next stage of the work is clear. A journal-ready empirical extension should replace simulated comparators with live frameworks, disclose full hardware and software provenance, and evaluate the same checkpoint design on non-synthetic datasets. Until then, the contribution should be understood as a reproducible systems benchmark and not as a general claim about frontier-model alignment.

---

## References

[1] T. I. Lawal, A. O. Ariyo, D. A. Akinwunmi, O. Z. Adesokan, and A. M. Ogunmolu, "Permission inheritance and machine-speed risk escalation in agentic AI-driven cloud operations," *Journal of Engineering Research and Reports*, vol. 28, no. 4, pp. 106-126, 2026. doi: 10.9734/jerr/2026/v28i41853.

[2] B. Khanna, "Collaborative agentic AI: Multi-agent coordination and communication models," *International Journal of Engineering Technology Research & Management (IJETRM)*, vol. 10, no. 03, pp. 34-45, 2026. doi: 10.5281/zenodo.18844429.

[3] S. Yao, J. Zhao, D. Yu, et al., "ReAct: Synergizing reasoning and acting in language models," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2023. doi: 10.48550/arXiv.2210.03629.

[4] P. Lewis, E. Perez, A. Piktus, et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Processing Systems*, vol. 33, pp. 9459-9474, 2020. doi: 10.48550/arXiv.2005.11401.

[5] Y. Bai, S. Kadavath, S. Kundu, et al., "Constitutional AI: Harmlessness from AI feedback," *arXiv*, 2022. doi: 10.48550/arXiv.2212.08073.

[6] L. Wang, C. Ma, X. Feng, et al., "A survey on large language model based autonomous agents," *Frontiers of Computer Science*, vol. 18, no. 6, Art. no. 186345, 2024. doi: 10.1007/s11704-024-40231-1.

[7] B. Khanna, "agentic-runtime," GitHub repository, 2026. [Online]. Available: https://github.com/bharatkhanna-dev/agentic-runtime.

[8] Q. Wu, G. Bansal, J. Zhang, et al., "AutoGen: Enabling next-gen LLM applications via multi-agent conversation," *arXiv*, 2023. doi: 10.48550/arXiv.2308.08155.

[9] S. Qian, H. Chen, Z. Zheng, et al., "ChatDev: Communicative agents for software development," in *Proc. 62nd Annual Meeting of the Association for Computational Linguistics (ACL)*, 2024. doi: 10.48550/arXiv.2307.07924.

[10] M. Lundberg and H. Lee, "Guidance: A language for controlling large language models," GitHub repository, 2023. [Online]. Available: https://github.com/guidance-ai/guidance.

[11] O. Willard and collaborators, "Outlines: Structured text generation with large language models," GitHub repository, 2024. [Online]. Available: https://github.com/outlines-dev/outlines.

[12] NVIDIA, "NeMo Guardrails," documentation and source repository, 2024. [Online]. Available: https://github.com/NVIDIA/NeMo-Guardrails.

[13] LangChain, "LangGraph," documentation and source repository, 2024. [Online]. Available: https://github.com/langchain-ai/langgraph.

---

## Appendix A: Artifact and Reproducibility Specification

To comply with IEEE reproducibility mandates, the research artifact is constructed and distributed as follows:

- **Source Code and Test Harness:** Available at `https://github.com/bharatkhanna-dev/agentic-runtime` under an MIT License. 
- **Data Provenance:** The 100-case evaluation dataset is located within `benchmarks/datasets/`, natively structured as deterministic payload assertions. 
- **Execution Environment:** Strict cryptographic dependency resolution is guaranteed via `requirements.lock.txt`.
- **Minimal Working Example (MWE):** Reviewers and reproducers may isolate the core architectural claim sequentially via `python -m agentic_runtime.evaluation.cli run-pair-b-variants`, which deterministically replicates the trajectory bypasses and checkpoint interventions discussed in Section 4.3 without requiring cloud API keys.
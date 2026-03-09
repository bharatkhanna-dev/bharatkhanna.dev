# Code Examples by Article

This directory contains working code examples organized by article. Each article has its own folder with complete, runnable examples demonstrating key concepts.

## Articles

### [Continuously Improving Agent Quality Using Evaluators Across Single-Turn, Trajectory, and Multi-Turn Interactions](./evaluators-in-agentic-ai-multiturn/)

Working code for agent evaluation patterns covering single-turn, trajectory, multi-turn, and production monitoring setups.

**What's Included:**
- `single_turn_eval/` — Heuristic and LLM-as-judge evaluators via LangSmith
- `trajectory_eval/` — LangGraph agent with tool introspection and pytest tests
- `llm_as_judge/` — Structured judge with calibration and rubric patterns
- `multi_turn_eval/` — Simulated user conversations and turn-level evaluation
- `langgraph_agent_eval/` — Middleware patterns, trace analysis, evaluator calibration
- `datasets/` — Idempotent dataset creation and versioning via LangSmith
- `pytest_regression/` — Quality gates and regression test suite
- `requirements.txt` and `.env.example` — Dependencies and setup

**Getting Started:**
```bash
cd evaluators-in-agentic-ai-multiturn
pip install -r requirements.txt
cp .env.example .env
# Add your LANGSMITH_API_KEY and OPENAI_API_KEY
```

Then follow the README in that folder for detailed walkthroughs.

---

## Future Articles

New articles will have their own folders following this same pattern.

```
examples/
├── evaluators-in-agentic-ai-multiturn/
│   ├── single_turn_eval/
│   ├── trajectory_eval/
│   ├── llm_as_judge/
│   ├── multi_turn_eval/
│   ├── langgraph_agent_eval/
│   ├── datasets/
│   ├── pytest_regression/
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
├── [future-article-name]/
│   ├── folder1/
│   ├── folder2/
│   ├── requirements.txt
│   └── README.md
└── README.md (this file)
```

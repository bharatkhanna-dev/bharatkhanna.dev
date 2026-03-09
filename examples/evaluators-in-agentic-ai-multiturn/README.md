# Agent Evaluation Examples

Companion code for the article *Continuously Improving Agent Quality Using Evaluators Across Single-Turn, Trajectory, and Multi-Turn Interactions*.

## Structure

```
evaluators-in-agentic-ai-multiturn/
├── requirements.txt              # All dependencies
├── .env.example                  # Environment variable template
├── single_turn_eval/             # Layer 1: unit evals
├── trajectory_eval/              # Layer 2: tool call trajectory evals
├── llm_as_judge/                 # LLM-as-judge patterns
├── multi_turn_eval/              # Layer 3: multi-turn simulation
├── langgraph_agent_eval/         # LangGraph-native eval patterns
├── datasets/                     # Dataset creation and experiment running
└── pytest_regression/            # pytest regression harness
```

## Quick Start

```powershell
# 1. Create virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env and fill in your API keys

# 4. Create datasets in LangSmith
python datasets/create_and_push.py

# 5. Run single-turn eval
python single_turn_eval/eval.py

# 6. Run multi-turn simulation
python multi_turn_eval/run_eval.py

# 7. Run regression suite
pytest pytest_regression/ -v
```

## Prerequisites

- Python 3.11+
- LangSmith account (free at https://smith.langchain.com)
- OpenAI API key

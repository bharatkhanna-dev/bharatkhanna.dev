# LangGraph Agent Evaluation

Advanced evaluation patterns for LangGraph agents:
- Middleware patterns (loop detection, pre-completion checklist)
- Automated trace analysis for harness improvement
- Evaluator calibration testing

## Run

```powershell
# Run middleware agent demo
python langgraph_agent_eval/agent_with_middleware.py

# Run evaluator calibration check
python langgraph_agent_eval/evaluator_calibration.py

# Analyze experiment failures (requires a LangSmith experiment name)
python langgraph_agent_eval/trace_analyzer.py --experiment my-experiment-name
```

## Key files

| File | Description |
|---|---|
| `agent_with_middleware.py` | LangGraph agent with harness middleware |
| `trace_analyzer.py` | Fetch failing runs and synthesize improvement suggestions |
| `evaluator_calibration.py` | Test judge alignment against human-labeled anchors |

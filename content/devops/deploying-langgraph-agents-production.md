+++
title = "Deploying LangGraph Agents to Production: A Practical Infrastructure Guide"
date = "2026-01-10"
draft = false
description = "Everything you need to go from a working LangGraph agent to a production deployment: containerization, async serving with FastAPI, observability, circuit breakers, and zero-downtime deploys."
tags = ["langgraph", "fastapi", "docker", "kubernetes", "devops", "agents", "python"]
categories = ["DevOps"]
+++

> **TL;DR** — A LangGraph agent is just a stateful graph. Deploying one is just serving that graph behind an API. The complexity isn't in the graph — it's in the nine things you forget to handle until they fail in production.

---

## The Gap Between "Working" and "Production"

Your LangGraph agent runs perfectly in a Jupyter notebook. Five of your friends have tested it. The demo was flawless.

Then you deploy it and everything that seemed fine becomes a problem:

- The LLM API is slow today. Requests hang for 45 seconds before users give up.
- A user sends an unusually long document and your agent runs 20 tool calls in a loop.
- Your pod restarts and you lose all in-flight conversation state.
- You can't tell whether the agent is degraded or working normally because you have no metrics.

This article covers the infrastructure layer that handles all of this.

---

## Containerization

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install only production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Use `python:3.11-slim` (not `latest`) — you want reproducible builds, not surprise updates.

---

## Async API Design

LangGraph agents are I/O-bound (LLM calls, tool calls). Use FastAPI with `async` handlers:

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph
import asyncio

app = FastAPI()

@app.post("/agent/stream")
async def stream_agent(request: AgentRequest):
    async def generate():
        config = {"configurable": {"thread_id": request.thread_id}}
        async for chunk in agent.astream(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        ):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## Timeout and Circuit Breaker Pattern

LLM APIs have variable latency. Without timeouts, a slow model response hangs your entire API:

```python
import asyncio
from functools import wraps

def with_timeout(seconds: float):
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(fn(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise HTTPException(503, detail="Agent timed out. Try again.")
        return wrapper
    return decorator

@app.post("/agent/invoke")
@with_timeout(30.0)
async def invoke_agent(request: AgentRequest):
    ...
```

---

## Observability

Without metrics, you're flying blind. Minimum viable observability stack:

```python
from prometheus_client import Counter, Histogram, generate_latest
import time

agent_requests_total = Counter("agent_requests_total", "Total agent invocations", ["status"])
agent_latency = Histogram("agent_latency_seconds", "Agent response latency")

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    agent_latency.observe(latency)
    agent_requests_total.labels(status=response.status_code).inc()
    return response
```

Export to Prometheus, visualize in Grafana. Alert when p95 latency exceeds your SLA.

---

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api
spec:
  replicas: 3
  strategy:
    rollingUpdate:
      maxUnavailable: 1      # keep 2/3 running during deploy
      maxSurge: 1
  template:
    spec:
      containers:
        - name: agent
          image: your-registry/agent-api:v1.2.3  # pin to exact tag, never :latest
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
          envFrom:
            - secretRef:
                name: agent-secrets   # OPENAI_API_KEY, LANGCHAIN_API_KEY, etc.
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
```

The `rollingUpdate` strategy ensures zero-downtime deploys. Pinning `image` to an exact tag (`v1.2.3`) prevents accidental rollouts from `latest` changing under you.

---

## The One Thing Most Teams Skip

**Health checks that actually test the agent.** Most deployments add a `/health` endpoint that returns `{"status": "ok"}` unconditionally. This is useless — it confirms the API process is alive, not that the agent works.

A proper health check should:
1. Make one real LLM call with a known input
2. Assert the output matches an expected pattern
3. Return 503 if it doesn't

Yes, this costs one LLM call every 30 seconds. That cost is trivial compared to serving a degraded agent to users for hours without knowing.

# pytest Regression Suite

Quality gate tests for agent correctness, safety, and response quality.

## Run

```powershell
# From examples/ with .venv activated

# Run all tests
pytest pytest_regression/ -v

# Run with LangSmith tracking and quality gate
$env:LANGCHAIN_PROJECT = "pytest_regression_v1"
pytest pytest_regression/ -v

# Run only safety tests
pytest pytest_regression/ -k "Safety"

# Run and fail fast on first failure
pytest pytest_regression/ -x
```

## Test classes

| Class | Tests |
|---|---|
| `TestFactualCorrectness` | Python year, Australia capital, WWW inventor, binary search |
| `TestSafetyBehavior` | Prompt injection, system prompt leak, empty input |
| `TestResponseQuality` | Substantive responses, code generatio |

## Quality gate

When `LANGCHAIN_PROJECT` is set, `conftest.py` fetches mean scores from LangSmith
after the run and fails CI if any metric falls below its threshold.
See `conftest.py` → `SCORE_THRESHOLDS` to adjust thresholds.

# 03 — Dependency Modernization

| Field | Value |
| --- | --- |
| Spec ID | 03 |
| Status | Not started |
| Depends on | 02 |
| Blocks | 04, 06, 07, 08 |
| Est. effort | 2–4 days |
| Risk | High (version migrations can break imports) |

## Objective
Bring the stack to current, supported versions and replace deprecated APIs (G4). Migrate both LangChain agents off `initialize_agent` to **LCEL/LangGraph**, move to the modern **Google Gen AI SDK** with a current Gemini model, remove the pre-1.0 `openai` pin, and update FastAPI/Pydantic/torch/dgl/mlflow. This is the highest-risk spec; proceed incrementally and keep tests green at each step.

## Background & current state (`requirements.txt`)
```
torch==2.0.1          torchvision==0.15.2     dgl==1.1.2      torch-geometric==2.3.1
fastapi==0.103.1      pydantic==2.3.0         uvicorn==0.23.2
langchain==0.0.284    langchain-community==0.0.5
google-generativeai==0.2.2   openai==0.28.1
mlflow==2.7.1         neo4j==5.12.0
```
- `initialize_agent(... AgentType.CONVERSATIONAL_REACT_DESCRIPTION ...)` at `src/explainability/agent.py:393-401`.
- `initialize_agent(... AgentType.ZERO_SHOT_REACT_DESCRIPTION ...)` at `src/threat_discovery/research_agent.py:238-244`.
- `ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")` via `config.GEMINI_MODEL_NAME` (`config.py:74`) and hardcoded `gemini-1.5-flash` (`research_agent.py:229`, `working_explanations.py:79`).
- `BaseTool` subclasses use class attributes `name`/`description` (`agent.py:57`, `research_agent.py:54`).

## Prerequisites
- Spec `02` complete (so you can run `make check` after each change).
- Read LangChain's migration notes for `initialize_agent → create_react_agent`/LangGraph.

## Out of scope
- Pydantic schema code changes (that is spec `04`, but bump the version here).
- Rewriting agent *logic/structure* beyond what the API migration requires (deep agent rework is `07`/`08`).

## Implementation steps

### Step 1 — Pin the new dependency set
Replace `requirements.txt` with current, mutually-compatible versions. Use this target (verify latest patch at implementation time; keep majors as shown):

```text
# Core DS
pandas==2.2.*
numpy==1.26.*
scikit-learn==1.5.*

# Dataset
kagglehub
kaggle

# Deep learning & GNN  (install torch/dgl from their official indexes — see Step 2)
torch==2.3.*
dgl==2.2.*
torch-geometric==2.5.*

# Graph DB
neo4j==5.23.*

# API
fastapi==0.115.*
uvicorn[standard]==0.30.*
pydantic==2.9.*
pydantic-settings==2.5.*
python-multipart==0.0.*

# LLM / agents (modern stack)
langchain==0.3.*
langchain-core==0.3.*
langchain-community==0.3.*
langchain-google-genai==2.*
langgraph==0.2.*
google-genai==1.*          # new unified Google Gen AI SDK (replaces google-generativeai)

# MLOps
mlflow==2.16.*
dvc==3.*

# Config
python-dotenv==1.*
pyyaml==6.*

# Testing/dev (keep aligned with spec 02)
pytest==8.*
pytest-asyncio==0.24.*
pytest-cov==5.*
httpx==0.27.*
black==24.*
flake8==7.*
isort==5.13.*
mypy==1.11.*
```

> **Remove** `openai==0.28.1` and `py2neo` (unused; `neo4j` driver is used directly). If a later need for OpenAI arises, add `openai>=1.0` and use the v1 client.
> **Gemini SDK choice:** `langchain-google-genai` is used by the agents and is sufficient. `google-genai` (new SDK) replaces the deprecated `google-generativeai` for any direct `genai.GenerativeModel(...)` calls (e.g., `research_agent.py:352`, `working_explanations.py:79`). Migrate those direct calls per Step 4.

### Step 2 — Install torch/dgl correctly
`torch` and `dgl` need platform-specific wheels. Document the exact commands used in `PROGRESS.md`. Typical CPU install:
```bash
pip install torch==2.3.* --index-url https://download.pytorch.org/whl/cpu
pip install dgl==2.2.* -f https://data.dgl.ai/wheels/torch-2.3/repo.html
pip install -r requirements.txt
```
> If a working `dgl` build is unavailable for the platform, record the blocker. The model code depends on `dgl` (`src/gnn_model/model.py:16`), so this must resolve before spec `06`.

### Step 3 — Modernize Gemini model name in config
In `src/config.py:74`, update the default model. Use a current model id at implementation time, e.g.:
```python
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
```
Add `GEMINI_MODEL_NAME` to `.env.example`. Replace **all** hardcoded `"gemini-1.5-flash"` / `"gemini-1.5-pro-latest"` strings with `config.GEMINI_MODEL_NAME` (`research_agent.py:229,352,555`, `working_explanations.py:79`).

### Step 4 — Migrate direct `google.generativeai` calls to the new SDK
Old (`research_agent.py:352`, `working_explanations.py:76-79`):
```python
import google.generativeai as genai
genai.configure(api_key=...)
model = genai.GenerativeModel('gemini-1.5-flash')
resp = model.generate_content(prompt)
text = resp.text
```
New (`google-genai`):
```python
from google import genai
client = genai.Client(api_key=config.GEMINI_API_KEY)
resp = client.models.generate_content(model=config.GEMINI_MODEL_NAME, contents=prompt)
text = resp.text
```
> Keep behavior identical; only swap the client. Preserve the existing try/except fallbacks.

### Step 5 — Migrate the explainability agent off `initialize_agent`
In `src/explainability/agent.py`:
- Replace the deprecated `BaseTool` class-attribute style with the `@tool` decorator or a `StructuredTool`, and bind the Neo4j driver via closure. Example:
```python
from langchain_core.tools import tool

def make_transaction_context_tool(driver):
    @tool
    def get_transaction_context(user_id: str) -> str:
        """Retrieve transaction context (profile, recent txns, neighbors, risk, graph metrics) as JSON."""
        # ... existing query logic from Neo4jTransactionTool._run ...
        return json_str
    return get_transaction_context
```
- Replace `initialize_agent(...)` (`:393-401`) with a LangGraph ReAct agent:
```python
from langgraph.prebuilt import create_react_agent
self.agent = create_react_agent(self.llm, tools=[make_transaction_context_tool(self.neo4j_driver)])
```
- Update `explain_transaction()` (`:478-534`) to invoke the new agent. The graph returns messages; extract the final message content:
```python
result = await self.agent.ainvoke({"messages": [("user", prompt)]})
output_text = result["messages"][-1].content
```
> The brittle parser `_parse_agent_response` (`:536-575`) is **replaced** in spec `07` with structured output. In this spec, keep it working but feed it `output_text`.

### Step 6 — Migrate the threat-discovery agent off `initialize_agent`
In `src/threat_discovery/research_agent.py`, mirror Step 5: convert `WebResearchTool` to a `@tool` function and build the agent with `create_react_agent`. Keep the simulated logic for now (spec `08` replaces it). Update `_research_threat_topic` (`:295-326`) to use `agent.invoke({"messages": [...]})`.

### Step 7 — Bump FastAPI/Pydantic compatibility
After bumping `fastapi`/`pydantic`, the app may emit Pydantic v2 deprecation warnings but should still run. The **full** Pydantic v2 idiom migration is spec `04`. Confirm the app imports and `/health` responds.

## Contract / data changes
- Removed dependencies: `openai`, `py2neo`, `google-generativeai`, `torchvision` (unless used — grep first).
- New env var: `GEMINI_MODEL_NAME`.

## Acceptance criteria
- [ ] `pip install -r requirements.txt` (plus torch/dgl index steps) completes in a clean venv; commands recorded in `PROGRESS.md`.
- [ ] No remaining imports of `langchain.agents.initialize_agent` (`grep -rn "initialize_agent" src/` returns nothing).
- [ ] No remaining `import google.generativeai` (`grep -rn "google.generativeai" src/` returns nothing).
- [ ] No hardcoded `gemini-1.5-*` strings in `src/` (`grep -rn "gemini-1.5" src/` returns nothing).
- [ ] `python -c "import src.api.main"` imports without error.
- [ ] `uvicorn src.api.main:app` starts and `GET /health` returns 200.
- [ ] `make check` (lint+type+test) passes; smoke tests still green.

## Test plan
- Extend `tests/unit/test_smoke.py`: assert `src.explainability.agent` and `src.threat_discovery.research_agent` import without error.
- Add `tests/unit/test_imports.py` that imports every module in `src/` (catches breakage from version bumps). Mark LLM-constructing imports safe by not instantiating agents at import time.

## Validation
```bash
grep -rn "initialize_agent" src/ ; echo "exit:$?"
grep -rn "google.generativeai" src/ ; echo "exit:$?"
grep -rn "gemini-1.5" src/ ; echo "exit:$?"
python -c "import src.api.main; print('import ok')"
python -m uvicorn src.api.main:app --port 8001 &
sleep 8 && curl -fs http://localhost:8001/health && kill %1
make check
```

## Rollback / fallback
Keep the old `requirements.txt` as `requirements.legacy.txt` until the spec is validated. If a migration step blocks (e.g., `dgl` wheel), revert `requirements.txt` and record the blocker; do **not** half-migrate the agents (leave them on the version that matches the installed LangChain).

## Definition of Done
All acceptance criteria checked; agents construct and the app boots on the modern stack; commit on `feat/spec-03-deps-modernize`.

## References
- `requirements.txt`, `src/explainability/agent.py:17-24,57-69,393-401,478-534`, `src/threat_discovery/research_agent.py:24-29,54-65,222-244,295-326,352,555`, `src/working_explanations.py:38-46,76-104`, `src/config.py:74`.
- LangChain docs: migrating `initialize_agent` to `langgraph.prebuilt.create_react_agent`.
- Google Gen AI SDK (`google-genai`) quickstart.

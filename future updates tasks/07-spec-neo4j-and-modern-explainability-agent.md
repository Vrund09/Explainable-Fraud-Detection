# 07 — Neo4j Setup & Modern Explainability Agent

| Field | Value |
| --- | --- |
| Spec ID | 07 |
| Status | Not started |
| Depends on | 03 (LangGraph stack), 06 (real predictions to explain) |
| Blocks | 09 (eval needs the real agent) |
| Est. effort | 2–4 days |
| Risk | Medium |

## Objective
Make the explanation layer real and robust: provide a **runnable Neo4j** (docker-compose) with a **seed script**, rebuild `AIInvestigator` on **LangGraph** with **structured output** (replacing brittle string parsing, G7), verify the Cypher queries return data, and consolidate the half-built mock path (`working_explanations.py`, `mock_neo4j.py`) into one honest implementation (G12).

## Background & current state
- `Neo4jTransactionTool` runs five Cypher queries (`agent.py:119-271`) against `User`/`TRANSACTION` graph.
- Agent built via `initialize_agent(CONVERSATIONAL_REACT_DESCRIPTION)` (`agent.py:393-401`) — migrated to LangGraph in spec `03`.
- `_parse_agent_response()` (`agent.py:536-575`) scans text + regex; `confidence` hardcoded `0.7` (G7).
- `_create_fallback_explanation()` (`agent.py:577-631`) is the no-LLM path.
- `working_explanations.py` reads nonexistent `data/mock_neo4j/*.json` (G12); `src/mock_neo4j.py` is empty.
- `/explain` endpoint (`main.py:464-524`) calls `investigator.explain_transaction(...)`.

## Prerequisites
- Spec `03` (LangGraph + `langchain-google-genai`), spec `06` (predictions exist to explain).

## Out of scope
- Threat-discovery agent (`08`); deep eval harness (`09`).

## Implementation steps

### Step 1 — Provide a runnable Neo4j
Add a `docker-compose.neo4j.yml` (or a service in the full compose from spec `13`):
```yaml
services:
  neo4j:
    image: neo4j:5.23
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD:-testpassword}
      NEO4J_PLUGINS: '["apoc"]'
    ports: ["7474:7474", "7687:7687"]
    volumes: ["neo4j_data:/data"]
volumes: { neo4j_data: {} }
```

### Step 2 — Seed script from processed graph
Add `scripts/seed_neo4j.py` that ingests `data/processed/graph_nodes.csv` + `graph_edges.csv` (produced by `graph_constructor.py`) into Neo4j with the `User` label and `TRANSACTION` relationships, plus the user aggregate properties the Cypher expects (`total_transactions`, `total_amount_sent`, `fraud_rate`, etc. — see `agent.py:121-129`). Reuse `GraphConstructor.ingest_to_neo4j` if present; otherwise write `UNWIND`-batched `MERGE` statements and create indexes on `User.user_id`.
> If the full dataset is too big, seed a **representative subset** consistent with the inference graph from spec `05/06` so explanations line up with predictions. Document the subset.

### Step 3 — Define structured-output schemas
Replace the brittle parser with a Pydantic model the LLM must fill. Add to `agent.py` (or a new `explainability/schemas.py`):
```python
from pydantic import BaseModel, Field
from typing import List, Literal

class FraudExplanation(BaseModel):
    summary: str = Field(description="2-5 sentence analyst-readable explanation")
    key_factors: List[str] = Field(description="specific risk factors observed", max_length=6)
    recommendation: Literal["approve", "review", "block"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(description="concrete data points cited from the graph context", default_factory=list)
```

### Step 4 — Rebuild the agent with LangGraph + structured output
- Convert `Neo4jTransactionTool` to a `@tool` function bound to the driver (as in spec `03` Step 5).
- Build the agent with `create_react_agent(self.llm, tools=[...])`.
- For the **final** answer, call the LLM with `with_structured_output(FraudExplanation)` so the result is a validated object, not scanned text. Pattern:
```python
self.structured_llm = self.llm.with_structured_output(FraudExplanation)
# after the agent gathers context (tool output), produce the final explanation:
explanation: FraudExplanation = await self.structured_llm.ainvoke(final_prompt_with_context)
```
- Replace `_parse_agent_response()` usage in `explain_transaction()` (`agent.py:512`) with the structured object → `explanation.model_dump()`. Delete `_parse_agent_response` (or keep only as a legacy fallback for non-structured models, clearly marked).

### Step 5 — Ground explanations in the real prediction
`/explain` should receive (or look up) the prediction context for the transaction so the explanation is consistent with the model's score. Update `ExplanationRequest`/endpoint (`main.py:464-489`) to accept the `prediction_context` (already supported) and ensure the API passes the real `predict_fraud_served` result (from spec `06`) when the client requests explanation alongside prediction.

### Step 6 — Consolidate the mock path
- Either delete `src/working_explanations.py` and the empty `src/mock_neo4j.py` (if unused after this spec) **or** repurpose `working_explanations.py` as the documented **offline fallback** that reads a committed fixture `tests/fixtures/neo4j_sample.json` (not a nonexistent path). Pick one; document the choice. Remove the import-time global side effect (`working_explanations.py:124`) — instantiate lazily.
- Ensure `grep -rn "data/mock_neo4j" src/` returns nothing.

### Step 7 — Harden fallbacks
Keep `_create_fallback_explanation()` for when Neo4j/LLM are down, but make it return a valid `FraudExplanation`-shaped dict so the API response schema is uniform regardless of path. Add a `data_sources`/`degraded` flag.

## Contract / data changes
- `/explain` response now always matches `FraudExplanation` fields (`summary`, `key_factors`, `recommendation`, `confidence`, `evidence`). Update `ExplanationOutput` (`schemas.py`) to match.
- New script `scripts/seed_neo4j.py`; new `docker-compose.neo4j.yml`.

## Acceptance criteria
- [ ] `docker compose -f docker-compose.neo4j.yml up -d` starts Neo4j; `scripts/seed_neo4j.py` loads users + transactions; a manual Cypher `MATCH (u:User) RETURN count(u)` returns > 0.
- [ ] `AIInvestigator` constructs with the LangGraph agent; no `initialize_agent` references remain.
- [ ] `explain_transaction()` returns a validated `FraudExplanation` (assert via schema), not regex-scanned text; `confidence` is model-provided, not hardcoded `0.7`.
- [ ] With Neo4j seeded and an API key set, `/explain` for a real flagged transaction returns a coherent explanation that cites at least one graph data point in `evidence`.
- [ ] With Neo4j/LLM unavailable, `/explain` still returns a schema-valid degraded explanation (no 500).
- [ ] `grep -rn "data/mock_neo4j" src/` and `grep -rn "_parse_agent_response" src/` reflect the consolidation (removed or clearly-labeled legacy).

## Test plan
Create `tests/unit/test_explainability_agent.py`:
- **Fake LLM:** a stub implementing `with_structured_output(...).ainvoke()` returning a fixed `FraudExplanation`; assert `explain_transaction` maps it through unchanged.
- **Fake Neo4j driver:** a stub session returning canned records; assert the tool builds the expected context JSON (covers `_get_user_profile`, `_get_recent_transactions`, etc.).
- **Fallback:** with `llm=None`, assert `_create_fallback_explanation` returns schema-valid output.
Mark any test that needs a live Gemini/Neo4j with `@pytest.mark.requires_secrets`/`requires_neo4j`.

Create `tests/integration/test_neo4j_tool.py` (marked `requires_neo4j`): against the compose Neo4j seeded with a tiny fixture, assert the five queries run and return expected shapes.

## Validation
```bash
docker compose -f docker-compose.neo4j.yml up -d
python scripts/seed_neo4j.py --subset 5000
RUN_NEO4J_TESTS=1 pytest tests/integration/test_neo4j_tool.py -q
pytest tests/unit/test_explainability_agent.py -q
grep -rn "initialize_agent\|data/mock_neo4j" src/ ; echo exit:$?
make check
```

## Rollback / fallback
The structured-output change is isolated to `explain_transaction`'s tail. If a chosen Gemini model doesn't support structured output well, use `with_structured_output(..., method="json_mode")` or a JSON-schema prompt + `FraudExplanation.model_validate_json`. Keep the degraded fallback always available.

## Definition of Done
Real Neo4j seedable; agent on LangGraph with validated structured explanations grounded in graph evidence; mock path consolidated; commit on `feat/spec-07-explainability`.

## References
- `src/explainability/agent.py:49-271,294-407,478-575,577-653`, `src/api/main.py:464-524`, `src/api/schemas.py` (`ExplanationRequest`,`ExplanationOutput`), `src/working_explanations.py`, `src/mock_neo4j.py`, `src/data_processing/graph_constructor.py`.

# 09 — Agents: Deep Analysis & Evaluation Harness

| Field | Value |
| --- | --- |
| Spec ID | 09 |
| Status | Not started |
| Depends on | 07, 08 |
| Blocks | — (informs roadmap; gates agent quality) |
| Est. effort | 3–5 days |
| Risk | Medium |

## Objective
Deliver the **deep agent span** requested: a rigorous review of both agents (gaps, mistakes, failure modes, security, cost/latency), a runnable **evaluation harness** with golden cases and groundedness/faithfulness checks, **tracing/observability**, and a prioritized **future-enhancements roadmap**. This spec produces both an analysis document and working eval code/tests.

## Background & current state
- **Investigator** (`src/explainability/agent.py`): ReAct over one Neo4j tool; after spec `07` uses LangGraph + structured output. `max_iterations=3`, `temperature=0.3`, `max_output_tokens=1000` (`agent.py:362-401`). Memory `ConversationBufferWindowMemory(k=5)` (`:386-390`).
- **Threat agent** (`src/threat_discovery/research_agent.py`): ReAct over a web-research tool; after spec `08` uses LangGraph + structured output + real retrieval.
- No evaluation, no tracing, no cost accounting today.

## Prerequisites
- Specs `07` and `08` complete (agents are on the modern stack and produce structured output).

## Out of scope
- Re-implementing the agents (done in `07`/`08`). This spec measures and hardens them.

## Implementation steps

### Step 1 — Author the deep-analysis document
Create `docs/agents_deep_analysis.md` covering, for **each** agent:

1. **Architecture & control flow** — tools, prompt, iteration cap, memory, model, output schema. Include a sequence diagram (Mermaid).
2. **Identified gaps & mistakes** (be specific, cite lines), e.g.:
   - Investigator: explanations historically not grounded (pre-`07` regex parsing, hardcoded confidence `0.7`); `max_iterations=3` may truncate multi-hop reasoning; no guard that cited numbers exist in the tool output (hallucination risk); single tool limits evidence (no direct GNN feature-attribution tool).
   - Threat agent: pre-`08` fully simulated; no source verification; risk-level inference keyword-based; no dedup; results never affected detection.
3. **Failure modes** — Neo4j down, LLM timeout/quota, tool error, empty graph context, prompt injection via tool output, non-JSON structured-output failures. For each: current behavior + desired behavior.
4. **Security** — prompt-injection surface (Neo4j data and web snippets flow into the prompt), secret handling, PII in transaction data, output sanitization. Mitigations: input/tool-output delimiting, instruction-hierarchy prompts, allowlist of tools, never echoing secrets, redaction of user ids in logs.
5. **Cost & latency** — token budget per call, expected calls per request, p50/p95 latency targets, caching opportunities (context caching, memoizing Neo4j context per user for a TTL).
6. **Future enhancements roadmap** (prioritized): GNN feature-attribution tool (GNNExplainer/Integrated Gradients) exposed to the investigator; multi-tool reasoning; retrieval-augmented threat memory; feedback loop where analyst accept/reject labels fine-tune prompts; incorporating discovered threats as model features (retrain).

### Step 2 — Build the evaluation harness
Create `src/evaluation/agent_eval.py` with a small, dependency-light evaluator:
- **Golden dataset:** `tests/fixtures/agent_golden/explanations.jsonl` — each line: `{transaction_id, context, expected_recommendation, must_mention: [...], must_not_mention: [...]}`. Start with ~15 hand-written cases spanning LOW→CRITICAL.
- **Metrics:**
  - *Schema validity*: output parses into `FraudExplanation` (100% required).
  - *Groundedness*: every item in `evidence` appears in (or is entailed by) the provided tool/context (string-containment + optional LLM-judge). Report % grounded.
  - *Recommendation accuracy*: predicted vs `expected_recommendation`.
  - *Faithfulness (optional LLM-judge)*: a separate LLM rates whether the explanation is supported by the context (0–1).
  - *Coverage*: fraction of `must_mention` present; zero `must_not_mention`.
- **Runner:** `python -m src.evaluation.agent_eval --suite explanations --report reports/agent_eval.json`. Uses a **fake/deterministic LLM** by default (records prompts, returns canned structured output) so it runs in CI; `--live` uses real Gemini (gated by secret).

### Step 3 — Threat-agent evaluation
Add a second suite `tests/fixtures/agent_golden/threats.jsonl` and evaluate: schema validity of `ThreatIntel`, presence of real `sources` in live mode, dedup correctness, risk-level sanity (e.g., "ransomware" ⇒ HIGH/CRITICAL).

### Step 4 — Tracing & observability
- Integrate **LangSmith** (optional, env-gated): set `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` in `.env.example` (commented). When unset, no tracing and no failure.
- Add lightweight local tracing regardless: a callback/context manager that logs per-call `{tool_calls, tokens (if available), latency_ms, outcome}` to `monitoring/data/agent_traces.jsonl`. Wire into spec `10`'s monitoring.

### Step 5 — Guardrails derived from the analysis
Implement the top mitigations identified in Step 1:
- **Tool-output delimiting**: wrap Neo4j/web content in clear delimiters with an instruction that content between them is data, not instructions (prompt-injection mitigation).
- **Groundedness check at runtime**: after structured output, drop any `evidence` item not found in the gathered context; if `confidence` is high but groundedness is low, downgrade `recommendation` to `review` and flag `degraded`.
- **Timeouts & retries**: wrap LLM/tool calls with a timeout and one retry; on exhaustion use the schema-valid fallback.

## Contract / data changes
- New module `src/evaluation/agent_eval.py`; reports under `reports/`.
- New (commented) env vars for LangSmith; `agent_traces.jsonl` output.

## Acceptance criteria
- [ ] `docs/agents_deep_analysis.md` exists and covers both agents across all six sections, with line-cited gaps and a prioritized roadmap.
- [ ] `python -m src.evaluation.agent_eval --suite explanations` runs offline (fake LLM) and writes `reports/agent_eval.json` with schema-validity, groundedness, recommendation-accuracy, and coverage metrics.
- [ ] Threat suite runs and reports schema validity + dedup + risk sanity.
- [ ] Runtime groundedness guard is active: a unit test shows an ungrounded `evidence` item is dropped and an over-confident+ungrounded case is downgraded to `review`.
- [ ] Tool-output delimiting present in both agents' prompts (grep for the delimiter markers).
- [ ] LangSmith is optional and absence causes no errors.
- [ ] `make check` passes.

## Test plan
Create `tests/eval/test_agent_eval.py`:
- runs the explanation eval suite with the fake LLM; asserts schema validity == 100% and that metrics are computed.
- `test_groundedness_guard_drops_unsupported_evidence`.
- `test_overconfident_ungrounded_downgraded_to_review`.
- `test_threat_dedup_metric`.
- `test_tracing_writes_jsonl` (uses a temp dir).

## Validation
```bash
python -m src.evaluation.agent_eval --suite explanations --report reports/agent_eval.json
python -c "import json; r=json.load(open('reports/agent_eval.json')); print(r['summary'])"
pytest tests/eval/test_agent_eval.py -q
grep -rn "BEGIN DATA\|END DATA" src/explainability src/threat_discovery   # delimiter markers present
make check
```

## Rollback / fallback
The eval harness and tracing are additive. The runtime groundedness guard changes agent output only in failure cases; if it proves too aggressive, make the downgrade threshold configurable (`AGENT_GROUNDEDNESS_MIN`) rather than removing it.

## Definition of Done
Deep-analysis doc + runnable offline eval harness for both agents + runtime groundedness/injection guards + optional tracing; commit on `feat/spec-09-agents-eval`.

## References
- `src/explainability/agent.py:294-407,478-575`, `src/threat_discovery/research_agent.py:209-371`, specs `07` and `08`, LangSmith tracing docs, GNNExplainer (PyG) for the roadmap.

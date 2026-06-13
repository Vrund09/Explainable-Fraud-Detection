# 08 — Threat-Discovery Agent: Make It Real

| Field | Value |
| --- | --- |
| Spec ID | 08 |
| Status | Not started |
| Depends on | 03 (LangGraph stack) |
| Blocks | 09 (eval covers this agent) |
| Est. effort | 2–4 days |
| Risk | Medium (external API + cost) |

## Objective
Replace the **simulated** web research (G6) with genuine retrieval, produce **structured, deduplicated** threat intelligence, persist it, and define a concrete **integration path** so discovered threats actually influence detection (rules/feature flags), not just generate text. Modernize the agent to LangGraph (started in spec `03`).

## Background & current state
- `WebResearchTool._run()` returns hardcoded dicts via substring matching (`research_agent.py:67-106`); "patterns"/"methods"/"risk" come from static maps (`:112-206`). The comment admits it's fake (`:73`).
- `ThreatDiscoveryAgent` uses `initialize_agent(ZERO_SHOT_REACT_DESCRIPTION)` (`:238-244`) and direct `genai.GenerativeModel('gemini-1.5-flash')` calls (`:352,555`).
- Results saved to `data/threat_intelligence/*.json` + `all_threats.json` (`:481-524`).
- `ThreatIntelligence` dataclass (`:38-48`) defines the schema.
- A `demonstrate_threat_discovery()` CLI exists (`:602-648`).

## Prerequisites
- Spec `03` (LangGraph, `google-genai`, model name from config).
- A web-search/retrieval provider API key (choose one): **Tavily** (`tavily-python`), SerpAPI, or Bing Web Search. Tavily is recommended (LLM-oriented, simple). Add the key to `.env.example` (e.g., `TAVILY_API_KEY`).

## Out of scope
- The investigator agent (`07`); evaluation (`09`).

## Implementation steps

### Step 1 — Real retrieval tool
Replace the simulated `WebResearchTool` with a real one. Using Tavily:
```python
from langchain_core.tools import tool
from tavily import TavilyClient
from ..config import config

def make_web_research_tool():
    client = TavilyClient(api_key=config.TAVILY_API_KEY)

    @tool
    def web_research_fraud_threats(search_query: str) -> str:
        """Search the web for current fraud techniques/threat trends. Returns JSON with titled results, urls, and snippets."""
        resp = client.search(query=search_query, max_results=6, search_depth="advanced",
                             include_answer=True, topic="news")
        results = [{"title": r["title"], "url": r["url"], "snippet": r["content"], "score": r.get("score")}
                   for r in resp.get("results", [])]
        return json.dumps({"query": search_query, "answer": resp.get("answer"), "results": results})
    return web_research_fraud_threats
```
> If no provider key is configured, the tool must **clearly** return a labeled `"mode": "offline_stub"` payload (reusing the old static maps) so it never silently pretends to be live. Add `THREAT_RESEARCH_MODE` (`live`/`offline`) to config; default `offline` so CI is deterministic.

### Step 2 — Structured threat extraction
Replace the regex/keyword extractors (`_extract_techniques_from_analysis` `:427-445`, `_extract_risk_level` `:447-459`, `_extract_indicators` `:461-479`) with LLM **structured output**:
```python
from pydantic import BaseModel, Field
from typing import List, Literal

class ThreatIntel(BaseModel):
    threat_name: str
    description: str = Field(max_length=600)
    fraud_techniques: List[str]
    risk_level: Literal["LOW","MEDIUM","HIGH","CRITICAL"]
    detection_indicators: List[str]
    confidence_score: float = Field(ge=0, le=1)
    sources: List[str] = Field(default_factory=list, description="source URLs")
```
Use `llm.with_structured_output(ThreatIntel)` in `_parse_research_result()` (`:328-371`) so extraction is validated, with real `sources` from the retrieval results.

### Step 3 — Agent on LangGraph
Finish the migration from spec `03`: build with `create_react_agent(self.llm, tools=[make_web_research_tool()])`; `_research_threat_topic()` (`:295-326`) invokes the graph and then runs structured extraction.

### Step 4 — Deduplication & persistence
- Before saving, dedupe against `data/threat_intelligence/all_threats.json` by normalized `threat_name` + technique overlap (e.g., Jaccard > 0.6 ⇒ merge/update instead of append).
- Keep the timestamped snapshot + rolling `all_threats.json` (existing behavior `:501-522`) but store `sources` and a content hash.

### Step 5 — Integration path into detection (the point of the agent)
Define how discovered threats affect the system. Implement at least one concrete hook:
- **Rules feed:** write a `data/threat_intelligence/active_rules.json` mapping detection indicators → lightweight rule checks (e.g., "rapid sequence after device change") that the API can surface as **additional risk factors** alongside the GNN score (display-only, clearly separate from the model score).
- Expose a read endpoint `GET /threats` (optional, behind auth) returning the current threat list for the frontend's "Threat Feed" page (spec `12`).
> Be explicit in docs that these are **advisory signals**, not model inputs, unless/until retraining incorporates them (future enhancement, see spec `09`).

### Step 6 — Scheduling (optional, documented)
Provide a CLI `python -m src.threat_discovery.research_agent --topics "crypto fraud 2025" ...` and document running it on a schedule (cron/GitHub Action). Do not add a background scheduler to the API process.

## Contract / data changes
- New env: `TAVILY_API_KEY` (or chosen provider), `THREAT_RESEARCH_MODE`.
- New artifacts: `data/threat_intelligence/active_rules.json`, threat records gain `sources` + content hash.
- Optional new endpoint `GET /threats`.

## Acceptance criteria
- [ ] `grep -rn "Simulate web research\|in practice, would use actual web scraping" src/` returns nothing (the fake path is gone or explicitly labeled `offline_stub`).
- [ ] With `THREAT_RESEARCH_MODE=live` and a provider key, `discover_new_threats(["crypto fraud 2025"])` returns `ThreatIntel` objects whose `sources` contain real URLs.
- [ ] With `THREAT_RESEARCH_MODE=offline`, the tool returns a clearly-labeled stub and the pipeline still produces schema-valid `ThreatIntel` (deterministic, for CI).
- [ ] Threat extraction is via `with_structured_output` (no regex/keyword extractors remain as the primary path).
- [ ] Dedup prevents duplicate `all_threats.json` entries on re-run.
- [ ] `active_rules.json` is produced and documented as advisory-only.
- [ ] `make check` passes.

## Test plan
Create `tests/unit/test_threat_agent.py`:
- **Fake retrieval tool + fake structured LLM:** assert `_research_threat_topic` yields a valid `ThreatIntel` with sources passed through.
- **Dedup:** seed `all_threats.json` with one record; run again with a near-duplicate; assert no duplicate appended.
- **Offline mode:** assert the stub is labeled and deterministic.
Mark live-provider tests `@pytest.mark.requires_secrets`.

## Validation
```bash
grep -rn "actual web scraping\|Simulate web research" src/ ; echo exit:$?
THREAT_RESEARCH_MODE=offline python -c "from src.threat_discovery.research_agent import ThreatDiscoveryAgent as A; \
import json; t=A().discover_new_threats(['payment fraud']); print([x.threat_name for x in t])"
pytest tests/unit/test_threat_agent.py -q
make check
```

## Rollback / fallback
`THREAT_RESEARCH_MODE=offline` is the always-safe path and is the CI default. If the provider API errors or quota is hit, the tool degrades to the labeled offline stub rather than failing the pipeline.

## Definition of Done
Real retrieval-backed, structured, deduplicated threat intelligence with a concrete advisory integration hook and honest offline mode; commit on `feat/spec-08-threat-agent`.

## References
- `src/threat_discovery/research_agent.py` (full file; key lines `38-48,51-106,112-206,222-244,295-371,427-524,602-648`), `src/config.py` (add provider keys), Tavily/SerpAPI docs.

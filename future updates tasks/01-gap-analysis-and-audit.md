# 01 — Gap Analysis & Audit

This is the diagnostic behind every spec. It documents what is wrong (or risky, or outdated) in the current codebase, with exact locations, root cause, impact, and the spec that fixes it. Read this to understand **why** before you change **what**.

Severity scale: **Critical** (headline feature is fake/broken), **High** (misleading or blocks a clean run), **Medium** (correctness/quality debt), **Low** (polish/hardening).

---

## Summary table

| ID | Sev | Title | Location | Fixed by |
| --- | --- | --- | --- | --- |
| G1 | Critical | API never uses the GNN (always heuristic) | `src/api/main.py:317`, `src/gnn_model/predict.py:283-312` | 06 |
| G2 | Critical | Trained artifact orphaned; API loads from nonexistent MLflow registry | `src/gnn_model/predict.py:475-508`; `paysim_94_fraud_model_final` (root) | 05, 06 |
| G3 | High | Demo UI fully mocked (`Math.random()`), not API-connected | `fraud_detection_demo.html:984-1041` | 12 |
| G4 | High | Severely outdated deps + deprecated APIs | `requirements.txt`; `initialize_agent` in both agents | 03 |
| G5 | High | `tests/` empty but CI expects coverage + lint to pass | `tests/`, `.github/workflows/ci.yml:64-92` | 02, 11 |
| G6 | High | Threat-discovery "web research" is simulated | `src/threat_discovery/research_agent.py:67-106` | 08 |
| G7 | Medium | Brittle LLM output parsing (string scan + regex) | `src/explainability/agent.py:536-575` | 07 |
| G8 | Medium | `os` used without import → `NameError` | `src/health_monitoring.py:105` | 10 |
| G9 | Medium | Monitoring modules orphaned, import-time side effects, hardcoded 94.0 | `src/health_monitoring.py:112`, `src/metrics_system.py:99` | 10 |
| G10 | Medium | Pydantic v1 idioms (`validator`, `constr`, `.dict()`) | `src/api/schemas.py:13-14`, `src/api/main.py:175` | 04 |
| G11 | Medium | Feature-dim mismatch: config 10 vs documented 17 | `src/config.py:89`; `project_ssot.md:27` | 05 |
| G12 | Medium | `working_explanations.py` reads nonexistent `data/mock_neo4j`; `mock_neo4j.py` empty | `src/working_explanations.py:17`, `src/mock_neo4j.py` | 07 |
| G13 | Low | CORS/TrustedHost `*`, no auth, default dev secrets | `src/api/main.py:111-122`, `src/config.py:182-183` | 04 |
| G14 | Low | Honesty: "94% F1" stated as fact; real metrics `[INSERT METRIC]` | `project_ssot.md:23,29`; `src/metrics_system.py:24` | 05, 14 |
| G15 | Low | Empty/placeholder files create confusion | `GCP_GraphSAGE_Training.ipynb` (1 byte), empty dirs `data/ docs/ notebooks?/` | 14 |

---

## Detailed findings

### G1 — The API never uses the GNN (Critical)
**What:** `/predict` calls `predictor.predict_fraud(transaction_dict, ...)` at `src/api/main.py:317` **without** the `full_graph` argument. In `src/gnn_model/predict.py`, `predict_fraud()` only runs the GNN when `full_graph is not None` (line 259). With no graph, it enters the `else` branch (lines 283-312) and computes a **rule-based heuristic** (`base 0.1 + 0.3 if amount high + 0.2 if CASH_OUT/TRANSFER`).
**Root cause:** The serving path was never wired to load the transaction graph and run subgraph inference. The SSOT itself acknowledges this (`project_ssot.md:16`).
**Impact:** The headline "GraphSAGE 94% F1" model is **not used at inference**. Every prediction the API returns is a toy heuristic. This is the single most important fix.
**Fixed by:** Spec `06` (with artifacts from `05`).

### G2 — Orphaned trained artifact; broken model loading (Critical)
**What:** A 5.9 MB file `paysim_94_fraud_model_final` sits at the repo root and is **never referenced** anywhere in `src/`. Meanwhile `load_production_model()` (`predict.py:475-508`) loads from the MLflow registry (`models:/fraud-detection-model/Production`), which **does not exist** on a fresh clone, then falls back to "None" stage versions that also don't exist → raises.
**Root cause:** Training happened in `PaySim_Hypertuned_Training.ipynb` (101 KB) and the model was saved as a loose file, never registered or wired.
**Impact:** On a clean environment the API cannot load a real model; only the lifespan's broad `except` keeps it limping toward the heuristic path. The "real" model is dead weight.
**Fixed by:** Spec `05` (inspect/validate/register the artifact, fix architecture/feature mismatch) and `06` (load it for inference).

### G3 — Demo UI is fully mocked (High)
**What:** `fraud_detection_demo.html` computes risk **client-side** with a hand-rolled heuristic and `Math.random()` for "confidence" (`:984-1041`), and ships scenario presets (`:1115-1130`). It never calls the FastAPI backend.
**Impact:** The "live demo" demonstrates nothing about the actual model or API. For a portfolio this is misleading.
**Fixed by:** Spec `12` replaces it with a React SPA wired to real endpoints.

### G4 — Outdated dependencies & deprecated APIs (High)
**What (from `requirements.txt`):** `langchain==0.0.284`, `langchain-community==0.0.5`, `google-generativeai==0.2.2`, `openai==0.28.1`, `fastapi==0.103.1`, `pydantic==2.3.0`, `torch==2.0.1`, `dgl==1.1.2`, `mlflow==2.7.1`. Both agents use `langchain.agents.initialize_agent` (`agent.py:393`, `research_agent.py:238`), which is deprecated. `BaseTool` subclasses set class-level `name`/`description` (old Pydantic-v1 style) (`agent.py:57-69`, `research_agent.py:54-65`).
**Impact:** Pre-LCEL LangChain, pre-1.0 OpenAI, and an ancient Gemini SDK. Demonstrates stale skills and risks install/runtime breakage. `initialize_agent` patterns won't survive a modern LangChain upgrade.
**Fixed by:** Spec `03`.

### G5 — No tests, but CI assumes them (High)
**What:** `tests/` contains only `__init__.py`. CI (`.github/workflows/ci.yml`) runs `black --check`, `isort --check-only`, `flake8`, `mypy`, and `pytest --cov` (`:64-92`). The format/lint checks will fail on current code, and coverage is meaningless with zero tests.
**Impact:** CI is decorative; "4 CI/CD jobs" in the SSOT are not actually protecting anything.
**Fixed by:** Spec `02` (make tooling pass + scaffold) and `11` (real suite + gates).

### G6 — Simulated threat research (High)
**What:** `WebResearchTool._run()` (`research_agent.py:67-106`) returns **hardcoded dictionaries** keyed by substring matches; the comment even says "in practice, would use actual web scraping" (`:73`). Risk levels and detection methods come from static maps (`:112-206`).
**Impact:** The "proactive threat discovery agent" does no research. It's a templated text generator.
**Fixed by:** Spec `08` (real retrieval + structured output + integration path).

### G7 — Brittle LLM output parsing (Medium)
**What:** `_parse_agent_response()` (`agent.py:536-575`) splits text on newlines, scans for the substrings "risk factor"/"recommend", and extracts the first regex number as an "analysis_score". `confidence` is hardcoded to `0.7`.
**Impact:** Fragile, non-deterministic, and not grounded. No schema guarantees.
**Fixed by:** Spec `07` (structured output via `with_structured_output()` + Pydantic).

### G8 — `os` NameError in health monitoring (Medium)
**What:** `src/health_monitoring.py:105` calls `os.makedirs(...)` but the module imports only `time, json, datetime, Path` (`:9-13`). At runtime `_save_metrics()` raises `NameError`, swallowed by the bare `except` (`:108`).
**Impact:** Health metrics never persist; the failure is silent.
**Fixed by:** Spec `10`.

### G9 — Orphaned monitoring with side effects (Medium)
**What:** `health_monitoring.py` and `metrics_system.py` define **module-level singletons** that run on import (`HealthMonitor()` `:112` prints; `MetricsCollector()` `:99` prints **and writes a file** in `__init__` `:44`). Neither is imported by `src/api/main.py`. `model_accuracy` is hardcoded to `94.0` (`health_monitoring.py:26`, `metrics_system.py:24`).
**Impact:** Importing these modules has surprising side effects; the API's real `/health` (`main.py:235`) is disconnected from these metrics; "accuracy" is a literal, not measured.
**Fixed by:** Spec `10`.

### G10 — Pydantic v1 idioms (Medium)
**What:** `schemas.py` imports `validator, root_validator` and `constr, confloat, conint` from `pydantic.types` (`:13-14`) and uses `Field(..., example=...)`. `main.py` calls `.dict()` on response models (`:175,190,207`).
**Impact:** These are deprecated in Pydantic v2 (`@field_validator`/`@model_validator`, `Annotated[... , Field(...)]`, `.model_dump()`, `json_schema_extra`). Emits warnings now, breaks later.
**Fixed by:** Spec `04`.

### G11 — Feature dimension mismatch (Medium)
**What:** `config.GNN_INPUT_DIM = 10` (`config.py:89`), but the SSOT documents "17 user-level features and 15 edge attributes" (`project_ssot.md:27`). The saved model's true input dim is unknown until inspected.
**Impact:** If the artifact was trained on 17 features but the config/model assume 10, inference will shape-mismatch. Must be resolved before wiring (`06`).
**Fixed by:** Spec `05` (inspect artifact, reconcile config + feature pipeline).

### G12 — Dead mock data path (Medium)
**What:** `working_explanations.py:17` reads `data/mock_neo4j/{users,network}.json` which do not exist; `_load_mock_data` swallows the error and returns `{}` (`:33-34`). `src/mock_neo4j.py` is an empty 0-byte file. A global `WorkingExplanationSystem()` instantiates on import (`:124`) and prints.
**Impact:** An alternate, half-built explanation path that silently degrades and confuses the architecture.
**Fixed by:** Spec `07` (consolidate explanation paths; either provide real seed data + Neo4j or a clearly-labeled, real fixture).

### G13 — Security defaults (Low, but flag for prod)
**What:** `CORSMiddleware(allow_origins=["*"])` and `TrustedHostMiddleware(allowed_hosts=["*"])` (`main.py:111-122`); no authentication on any endpoint despite `API_KEY_HEADER` defined (`config.py:182`); default secret `dev-secret-key-change-in-production` (`config.py:183`).
**Impact:** Fine for a local demo, but a reviewer will flag it. The SPA also needs a known CORS origin.
**Fixed by:** Spec `04`.

### G14 — Unverifiable performance claims (Low)
**What:** "94% F1" is stated as achieved (`project_ssot.md:23,29`) and hardcoded in metrics (`metrics_system.py:24`), while business metrics are literally `[INSERT METRIC]` (`project_ssot.md:23,32,33`).
**Impact:** Portfolio honesty risk. Claims must be reproducible (tie to an MLflow run / eval report) or softened.
**Fixed by:** Spec `05` (produce a real eval report) and `14` (honest framing + provenance).

### G15 — Placeholder clutter (Low)
**What:** `GCP_GraphSAGE_Training.ipynb` is 1 byte (empty); `data/`, `docs/`, `notebooks/` (per README) show as empty in the worktree; `mock_neo4j.py` empty.
**Impact:** Confuses readers about what's real.
**Fixed by:** Spec `14` (clean up or populate, document intent).

---

## Cross-cutting themes

1. **"Real vs. mocked" gap.** The three most prominent features — model inference (G1/G2), the demo (G3), and threat research (G6) — are all simulated. The full-real-fix path closes all three.
2. **Modernization debt.** Deps (G4) and Pydantic idioms (G10) are 1–2 major versions behind.
3. **Verification debt.** No tests (G5), broken/orphaned monitoring (G8/G9), unverifiable claims (G14). The eval/test specs (`09`, `11`) close this.

Proceed to spec `02`.

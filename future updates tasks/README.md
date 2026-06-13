# Future Updates Tasks — Spec Index

This folder contains **spec-driven task files** that upgrade the *Explainable AI for Graph-Based Fraud Detection* project from a partially-mocked prototype into a genuine, modern, portfolio-grade system.

Each numbered file is a **self-contained specification**: read it, implement exactly what it says, run its tests and validation commands, tick its checklist, then move to the next one. The specs are written for an implementer who has **no prior context** on this repo — every change names exact files, lines, and commands.

> **Repo root** referenced throughout: the directory that contains `src/`, `requirements.txt`, `Dockerfile`, and `fraud_detection_demo.html`. All paths like `src/api/main.py` are relative to that root unless stated otherwise.

---

## How to use this folder

1. Read **`00-conventions-and-workflow.md`** first. It defines the loop you must follow for every spec, the branching/commit rules, environment setup, and guardrails.
2. Read **`01-gap-analysis-and-audit.md`** to understand *why* each change exists (the diagnosis behind the specs).
3. Execute the numbered specs **in order** (see the dependency graph below). Do not start a spec whose prerequisites are unchecked.
4. Each spec ends with **Acceptance Criteria**, a **Test Plan**, **Validation commands**, and a **Definition of Done**. A spec is "done" only when all of these pass.

---

## Decisions locked for this effort

| Topic | Decision |
| --- | --- |
| UI | Modern **React + Vite + TypeScript + Tailwind + shadcn/ui**, light Claude-like minimal aesthetic, wired to the live FastAPI |
| Backend depth | **Full real fix** — real GNN inference, real model artifact, real Neo4j, retrain/export pipeline |
| Dependencies | **Modernize all** — current LangChain (LCEL/LangGraph), Gemini 2.x SDK, Pydantic v2 idioms, current FastAPI/torch/dgl |
| Spec location | This folder: `future updates tasks/` at the repo root |

---

## Spec catalogue

| # | File | Purpose |
| --- | --- | --- |
| — | `README.md` | This index |
| — | `00-conventions-and-workflow.md` | Mandatory workflow, env setup, testing philosophy, guardrails |
| — | `01-gap-analysis-and-audit.md` | Full diagnostic of every gap with `file:line`, severity, root cause |
| 02 | `02-spec-foundation-tooling-and-test-bootstrap.md` | Make CI green-able: lint/format/type config, pytest scaffold, pre-commit, Makefile |
| 03 | `03-spec-dependency-modernization.md` | Upgrade all deps; migrate agents to LCEL/LangGraph; Gemini 2.x SDK |
| 04 | `04-spec-pydantic-v2-and-api-hardening.md` | Pydantic v2 migration; API-key auth; CORS for the SPA; rate limiting; OpenAPI export |
| 05 | `05-spec-model-artifact-recovery-and-retraining.md` | Recover/validate the trained artifact; fix feature-dim mismatch; reproducible retrain/export |
| 06 | `06-spec-real-gnn-inference-wiring.md` | **Core fix:** real GNN inference in `/predict`; graph load at startup; cold-start handling |
| 07 | `07-spec-neo4j-and-modern-explainability-agent.md` | docker-compose Neo4j + seed; rebuild investigator on LangGraph w/ structured output |
| 08 | `08-spec-threat-discovery-agent-real.md` | Replace simulated web research with real retrieval; structured threats; integration path |
| 09 | `09-spec-agents-deep-analysis-and-eval-harness.md` | Deep agent review + evaluation harness + tracing + roadmap |
| 10 | `10-spec-monitoring-and-observability.md` | Fix monitoring bugs; wire metrics into API; remove import side effects; structured logging |
| 11 | `11-spec-testing-and-ci-hardening.md` | Full test suite; coverage gates; MLflow eval gate; all CI jobs green |
| 12 | `12-spec-frontend-react-claude-ui.md` | The React SPA + Claude-like design system + typed API client + tests |
| 13 | `13-spec-deployment-and-compose.md` | Full-stack docker-compose; deploy targets; secrets/env management |
| 14 | `14-spec-docs-and-portfolio-polish.md` | Honest README rewrite; diagrams; screenshots; demo script; SSOT bullets |

---

## Requirement coverage

| Original ask | Satisfied by |
| --- | --- |
| 1) Find gaps + how to fix | `01` (diagnosis) + specs `02`–`14` (fixes) |
| 2) Upgrades / portfolio shine | `03`, `05`, `12`, `13`, `14` |
| 3) Claude-like minimal UI | `12` |
| 4) Deep agent span (gaps, mistakes, future) | `09` (+ fixes in `07`, `08`) |
| 5) Spec-driven MD with testing/validation | this folder + `00` + Test Plan/Validation/DoD in every spec |

---

## Execution order & dependency graph

```
02  Foundation & test bootstrap
│
03  Dependency modernization
│
04  Pydantic v2 + API hardening
│
05  Model artifact recovery & retraining
│
06  Real GNN inference wiring        ← depends on 05 (artifacts) + 03/04 (stack/contract)
│
├── 07  Neo4j + explainability agent  ← depends on 03 (LangGraph)
├── 08  Threat-discovery agent        ← depends on 03 (LangGraph)   [07 & 08 parallel]
│
09  Agents deep analysis + eval       ← depends on 07, 08
│
10  Monitoring & observability
│
11  Testing & CI hardening            ← validates 02–10
│
12  Frontend React Claude UI          ← depends on 04 (contract), ideally 06 (real preds)
│
13  Deployment & compose              ← depends on 06, 12
│
14  Docs & portfolio polish           ← depends on everything (final pass)
```

**Minimum path to a genuine, demonstrable system:** `02 → 03 → 04 → 05 → 06 → 12`. Specs `07`–`11`, `13`, `14` harden and complete it.

---

## Global Definition of Done (applies to every spec)

A spec is complete only when **all** of the following hold:

- [ ] Every item in that spec's **Acceptance Criteria** is satisfied.
- [ ] Every test in that spec's **Test Plan** exists and passes.
- [ ] Every command in **Validation** runs successfully and its output is captured.
- [ ] No previously-passing test was deleted, skipped, or weakened.
- [ ] `black --check`, `isort --check-only`, and `flake8` pass on changed Python files (after spec `02`).
- [ ] The change is committed on its own branch with the commit message format from `00-conventions`.
- [ ] If anything blocked completion, it is written to `future updates tasks/PROGRESS.md` (create it on first use) and surfaced, rather than worked around.

---

## Glossary

| Term | Meaning |
| --- | --- |
| **GNN** | Graph Neural Network (here, GraphSAGE) — `src/gnn_model/` |
| **Artifact** | A saved model + preprocessing files (scaler, feature names, node mapping, graph) |
| **Heuristic fallback** | The rule-based scorer in `predict_fraud()` used when no graph is supplied |
| **LCEL** | LangChain Expression Language (the modern composition API) |
| **LangGraph** | LangChain's stateful agent/graph runtime that replaces `initialize_agent` |
| **Structured output** | LLM responses parsed into a Pydantic model via `with_structured_output()` |
| **SPA** | Single-Page Application (the React frontend) |
| **SSOT** | `project_ssot.md` — the project's source-of-truth summary/resume doc |
| **Cold start** | Predicting for a user/node not present in the training graph |

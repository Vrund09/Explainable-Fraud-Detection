# 11 — Testing Suite & CI Hardening

| Field | Value |
| --- | --- |
| Spec ID | 11 |
| Status | Not started |
| Depends on | 02–10 (tests cover the work they produced) |
| Blocks | 13 (deploy should be gated on green CI) |
| Est. effort | 3–5 days |
| Risk | Medium |

## Objective
Turn the decorative CI (G5) into a real safety net: a comprehensive test suite (unit/integration/eval), a coverage gate, a model **evaluation gate**, and a CI pipeline that actually exercises the app. Consolidate the per-spec tests into a coherent suite and make all jobs green.

## Background & current state
- `tests/` had only `__init__.py`; specs `02`–`10` each added targeted tests. This spec ensures completeness, raises the coverage gate, and fixes CI.
- CI (`.github/workflows/ci.yml`) has 8 jobs: test, security, build, integration-test, deploy-staging, deploy-production, release, cleanup. The `test` job runs black/isort/flake8/mypy/pytest with a Neo4j service (`:22-110`). Integration tests build the Docker image and curl `/health` + `/model/status` (`:226-287`).

## Prerequisites
- Specs `02`–`10` (the code under test exists and is modernized).

## Out of scope
- Frontend tests (spec `12` owns Vitest/Playwright).

## Implementation steps

### Step 1 — Inventory & fill test gaps
Ensure these exist and pass (most created in earlier specs — verify and complete):
- `tests/unit/test_schemas.py` (04), `test_model.py` (05), `test_predict.py` (06), `test_explainability_agent.py` (07), `test_threat_agent.py` (08), `test_monitoring.py` (10), `test_imports.py` (03), `test_smoke.py` (02).
- `tests/integration/test_api_predict.py` (04/06), `test_api_explain.py` (07), `test_neo4j_tool.py` (07, `requires_neo4j`).
- `tests/eval/test_agent_eval.py` (09).
Add any missing coverage for: `data_processing/graph_constructor.py` (feature engineering on a tiny synthetic CSV), `gnn_model/training.py` (one mini training step on a toy graph, `slow`).

### Step 2 — Shared fixtures
Expand `tests/conftest.py` with:
- `app_client` — `httpx.AsyncClient(transport=ASGITransport(app=app))`.
- `mini_artifacts` — writes a tiny `model.pt`/`graph.bin`/`scaler.pkl`/`feature_names.json`/`node_mapping.json` to a tmp dir and points `config.MODELS_DIR` at it (so `/predict` runs the real path on a toy model). Reused by `06`/`10` tests.
- `fake_llm` — deterministic structured-output stub.
- `fake_neo4j_driver` — canned session/records.

### Step 3 — Coverage gate
- Add `--cov=src --cov-fail-under=70` to the pytest invocation (raise from 0). Record the actual % in `PROGRESS.md`. Do not lower later.
- Exclude untestable/visualization-only code via `[tool.coverage.run] omit` (e.g., `training.py` plotting) only where justified.

### Step 4 — Model evaluation gate
Add `scripts/eval_gate.py` that loads `models/production/metrics.json` and fails (exit 1) if headline metrics fall below a floor (e.g., `pr_auc < 0.8` or `f1 < 0.85` — set floors from the real measured values in spec `05`, not the aspirational 94%). CI runs this so a regressed model blocks deploy.

### Step 5 — Fix & extend CI (`.github/workflows/ci.yml`)
- Ensure `requirements.txt` installs in CI (torch/dgl CPU index — add the index URLs from spec `03` Step 2 to the install step).
- Set env for deterministic agent tests: `THREAT_RESEARCH_MODE=offline`, no live keys; LLM/Neo4j tests are skipped unless `RUN_*` flags set.
- Add the eval gate step (`python scripts/eval_gate.py`) to the `test` job (skip gracefully if `metrics.json` absent on PRs from forks).
- Keep the Neo4j service; add `RUN_NEO4J_TESTS=1` only in a dedicated integration job (not the unit job).
- Upload coverage + the eval report as artifacts.
- Confirm `black/isort/flake8/mypy` use the spec `02` configs and line length 100.

### Step 6 — Make it green
Run the full suite locally, fix failures at the **root cause** (never weaken assertions). Achieve a clean `make check` and `pytest --cov --cov-fail-under=70`.

## Contract / data changes
- New `scripts/eval_gate.py`. CI updated. Coverage gate at 70%.

## Acceptance criteria
- [ ] `pytest -q` passes locally with all unit + integration (non-gated) + eval tests.
- [ ] `pytest --cov=src --cov-fail-under=70` passes (record actual %).
- [ ] `python scripts/eval_gate.py` passes against the real `metrics.json` and **fails** when fed a metrics file below the floor (prove with a temp file in a test).
- [ ] CI `test` job is green on a clean checkout (lint, type, tests, coverage, eval gate).
- [ ] No test is skipped except those marked `requires_secrets`/`requires_neo4j`/`slow` (documented).
- [ ] No previously-passing test was deleted or weakened.

## Test plan
- This spec *is* the test plan; additionally add `tests/unit/test_eval_gate.py` proving the gate fails below floor and passes above.
- Add `tests/unit/test_graph_constructor.py` (feature engineering on synthetic data) and `tests/unit/test_training_step.py` (`slow`, one optimizer step reduces loss on a toy graph).

## Validation
```bash
make check
pytest --cov=src --cov-report=term-missing --cov-fail-under=70
python scripts/eval_gate.py
# simulate CI install of torch/dgl CPU:
pip install torch==2.3.* --index-url https://download.pytorch.org/whl/cpu
```

## Rollback / fallback
If 70% coverage is not reachable within the spec, set the gate to the highest achieved value (rounded down to a 5% step) and record a TODO to raise it — but never set it to 0. Gated tests (Neo4j/secrets) must remain skippable so CI stays deterministic.

## Definition of Done
Comprehensive, deterministic suite; coverage + eval gates enforced; all non-deploy CI jobs green; commit on `feat/spec-11-testing-ci`.

## References
- `.github/workflows/ci.yml` (full), `tests/` (all specs), `models/production/metrics.json` (05), `00-conventions §4`.

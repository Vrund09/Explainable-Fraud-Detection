# 10 — Monitoring & Observability

| Field | Value |
| --- | --- |
| Spec ID | 10 |
| Status | Not started |
| Depends on | 06 (real predictions to measure) |
| Blocks | — |
| Est. effort | 1–2 days |
| Risk | Low–Medium |

## Objective
Fix the broken/orphaned monitoring (G8, G9), wire real metrics into the API request lifecycle, remove import-time side effects, stop reporting a hardcoded "94%", and add structured logging + request tracing so the system is observable.

## Background & current state
- `src/health_monitoring.py:105` calls `os.makedirs` but never imports `os` → `NameError` (swallowed) (G8).
- `health_monitoring.py` and `metrics_system.py` define **module-level singletons** with side effects on import (`HealthMonitor()` `:112` prints; `MetricsCollector()` `:99` prints + writes a file in `__init__` `:44`) and are **not imported** by `src/api/main.py` (G9).
- `model_accuracy` hardcoded `94.0` (`health_monitoring.py:26`, `metrics_system.py:24`).
- The API's `/health` (`main.py:235-253`) and `/model/status` (`main.py:256-281`) compute their own info independent of these modules.
- `monitoring/` already has `performance_dashboard.html` and `monitoring_summary.json` referencing `monitoring/data/`.

## Prerequisites
- Spec `06` (so metrics reflect real GNN predictions, including `scoring_path`).

## Out of scope
- Agent tracing (handled in spec `09`); full Prometheus/Grafana stack (optional add-on noted below).

## Implementation steps

### Step 1 — Fix the `os` import bug
Add `import os` to `src/health_monitoring.py` (top of file, with the other stdlib imports `:9-13`). Confirm `_save_metrics()` (`:102-109`) now creates `monitoring/data/` and writes without error.

### Step 2 — Remove import-time side effects
- Delete the module-level singletons `health_monitor = HealthMonitor()` (`health_monitoring.py:112`) and `metrics_collector = MetricsCollector()` (`metrics_system.py:99`) **or** convert to lazy accessors:
```python
_monitor = None
def get_health_monitor() -> "HealthMonitor":
    global _monitor
    if _monitor is None:
        _monitor = HealthMonitor()
    return _monitor
```
- Move the file-write out of `MetricsCollector.__init__` (`:44`); write only on `record_*`/explicit `save_metrics()`.
- Remove `print(...)` calls from `__init__` (use `logger.info`).

### Step 3 — Consolidate to one monitoring module
`health_monitoring.py` and `metrics_system.py` overlap heavily. Merge into a single `src/monitoring.py` (or keep `metrics_system.py` and delete `health_monitoring.py`) exposing:
- `record_prediction(result, response_time_ms)`, `record_explanation()`, `record_error(kind)`
- `get_health_status()`, `get_performance_metrics()`
Update any references. Document the chosen module in `PROGRESS.md`.

### Step 4 — Replace hardcoded accuracy with real provenance
Read model metrics from `models/production/metrics.json` (spec `05`) instead of literal `94.0`. If the file is absent, report `model_f1: null` / `"unknown"` rather than a fabricated number (ties to G14 honesty).

### Step 5 — Wire metrics into the API
In `src/api/main.py`:
- In the request-tracking middleware (`:215-228`) or the existing background tasks (`log_prediction_async` `:588-598`), call `record_prediction(prediction_result, process_time_ms)` and `record_explanation()` in `/explain`.
- Update `/health` (`:235-253`) to merge `get_health_status()` (uptime, totals, avg latency, fraud rate) so the endpoint reflects real activity.
- Add `GET /metrics/summary` returning `get_performance_metrics()` (behind optional auth) for the frontend dashboard.

### Step 6 — Structured logging
- Configure JSON logging (or keep the existing format but add a `request_id` field) so each prediction/explanation log line includes `request_id`, `scoring_path`, `fraud_probability`, `latency_ms`. The `request_id` already exists in middleware (`main.py:216-219`).
- Redact user ids in logs if `config.LOG_REDACT_PII` is true (add the flag; default false for local dev).

### Step 7 — Optional Prometheus (documented, not required)
Note in `docs/` how to add `prometheus-fastapi-instrumentator` for `/metrics` (Prometheus format) and a Grafana dashboard. Implement only if time allows; otherwise leave the existing `monitoring/performance_dashboard.html` consuming `monitoring/data/*.json`.

## Contract / data changes
- New endpoint `GET /metrics/summary`.
- `/health` enriched with live counters.
- New config: `LOG_REDACT_PII` (default false).
- Monitoring consolidated to one module; `monitoring/data/*.json` written reliably.

## Acceptance criteria
- [ ] `grep -n "^import os" src/health_monitoring.py` (or the consolidated module) confirms `os` is imported; `_save_metrics()` writes `monitoring/data/metrics.json` without error.
- [ ] Importing the monitoring module has **no** side effects (no print, no file write at import) — verified by a test that imports it and asserts no file created.
- [ ] After N predictions via the API, `/health` and `GET /metrics/summary` report `total_predictions==N`, a real average latency, and a fraud rate computed from results.
- [ ] No hardcoded `94.0` remains as a reported metric (`grep -rn "94.0" src/` only in tests/fixtures if anywhere); accuracy comes from `metrics.json` or reports "unknown".
- [ ] Prediction log lines include `request_id`, `scoring_path`, `latency_ms`.
- [ ] `make check` passes.

## Test plan
Create `tests/unit/test_monitoring.py`:
- `test_import_has_no_side_effects` (monkeypatch cwd to tmp; import module; assert no file written, capture no stdout).
- `test_record_prediction_updates_counters_and_latency`.
- `test_save_metrics_creates_dir` (uses tmp dir; proves the `os` fix).
- `test_accuracy_from_metrics_json_or_unknown`.
Extend `tests/integration/test_api_predict.py`: after two `/predict` calls, assert `/metrics/summary` shows `total_predictions == 2`.

## Validation
```bash
python -c "import src.monitoring as m; print('import side-effect-free')"   # or consolidated name
pytest tests/unit/test_monitoring.py -q
python -m uvicorn src.api.main:app --port 8003 &
sleep 8
curl -s -X POST localhost:8003/predict -H 'content-type: application/json' -d '{"sender_id":"a","receiver_id":"b","amount":100,"type":"PAYMENT"}' >/dev/null
curl -s localhost:8003/metrics/summary | python -m json.tool
kill %1
make check
```

## Rollback / fallback
Changes are localized to the monitoring module(s) and a few API hooks. If consolidation is risky mid-stream, first land the `os` fix + side-effect removal (smallest safe change), then consolidate in a follow-up commit within this spec.

## Definition of Done
Monitoring imports cleanly, persists reliably, reflects real predictions, and reports honest accuracy; `/health` + `/metrics/summary` are live; commit on `feat/spec-10-monitoring`.

## References
- `src/health_monitoring.py:9-13,26,44,102-112`, `src/metrics_system.py:18-45,90-99`, `src/api/main.py:215-253,256-281,588-624`, `models/production/metrics.json` (spec 05), `monitoring/`.

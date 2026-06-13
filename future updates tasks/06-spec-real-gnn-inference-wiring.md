# 06 — Real GNN Inference Wiring (CORE FIX)

| Field | Value |
| --- | --- |
| Spec ID | 06 |
| Status | Not started |
| Depends on | 05 (verified artifacts), 04 (clean contract), 03 (stack) |
| Blocks | 12 (frontend should show real predictions), 13 |
| Est. effort | 3–5 days |
| Risk | High (the headline fix; correctness-critical) |

## Objective
Make `/predict` and `/predict/batch` actually run the **GraphSAGE model** instead of the rule-based heuristic (G1, G2). Load the inference graph + artifacts at app startup, implement real **subgraph inference** for a transaction's participants, handle **cold-start** users not in the graph, and clearly separate (and honestly label) any fallback. After this spec, a prediction reflects the trained model.

## Background & current state
- `/predict` calls `predictor.predict_fraud(transaction_dict, ...)` **without** `full_graph` (`src/api/main.py:317`).
- `predict_fraud()` runs the GNN only when `full_graph is not None` (`predict.py:259-281`); otherwise it uses the heuristic (`predict.py:283-312`).
- `create_subgraph_for_transaction()` already exists (`predict.py:183-221`) and uses `self.node_mapping` + `dgl.khop_subgraph`.
- `load_model()` / `load_preprocessing_artifacts()` exist (`predict.py:74-152`) but are never called with the local artifacts.
- `load_production_model()` only tries MLflow (`predict.py:475-508`).
- App lifespan loads the model via `load_production_model()` (`main.py:60-78`) and never loads a graph.
- Artifacts from spec `05` live in `models/production/` (`model.pt`, `scaler.pkl`, `feature_names.json`, `node_mapping.json`, `graph.bin`).

## Prerequisites
- Spec `05` artifacts verified (model loads, dims consistent).

## Out of scope
- Explanations (`07`), monitoring wiring (`10`).

## Implementation steps

### Step 1 — Add a robust local loader to `predict.py`
Add a method to load everything from `models/production/` without requiring MLflow:
```python
def load_from_local_artifacts(self, artifacts_dir: Path = config.MODELS_DIR / "production") -> None:
    import dgl, pickle, json
    ckpt = torch.load(artifacts_dir / "model.pt", map_location=self.device)
    self.model = GraphSAGEClassifier(**ckpt["model_config"])
    self.model.load_state_dict(ckpt["model_state_dict"])
    self.model.to(self.device).eval()
    with open(artifacts_dir / "scaler.pkl", "rb") as f:
        self.scaler = pickle.load(f)
    self.feature_names = json.loads((artifacts_dir / "feature_names.json").read_text())
    self.node_mapping = json.loads((artifacts_dir / "node_mapping.json").read_text())
    self.full_graph = dgl.load_graphs(str(artifacts_dir / "graph.bin"))[0][0]
    logger.info("Loaded local inference artifacts: %d nodes, %d features",
                self.full_graph.num_nodes(), len(self.feature_names))
```
Add `self.full_graph: Optional[dgl.DGLGraph] = None` to `__init__` (`predict.py:59-66`).

### Step 2 — Make `load_production_model()` prefer local artifacts
Update `load_production_model()` (`predict.py:475-508`) to try, in order:
1. `models/production/` local artifacts (Step 1) — **primary**.
2. MLflow registry (existing logic) — secondary.
3. Raise a clear error if neither is available (do **not** silently return a model-less predictor).
```python
def load_production_model() -> FraudPredictor:
    predictor = FraudPredictor(device="cuda" if torch.cuda.is_available() else "cpu")
    local_dir = config.MODELS_DIR / "production"
    if (local_dir / "model.pt").exists():
        predictor.load_from_local_artifacts(local_dir)
        return predictor
    try:
        predictor.load_model()  # MLflow
        predictor.load_preprocessing_artifacts(...)  # if available
        return predictor
    except Exception as e:
        raise RuntimeError(f"No model artifacts found (local or MLflow): {e}")
```

### Step 3 — Implement the real serving prediction path
The current `predict_fraud()` requires the caller to pass `full_graph`. Add a serving wrapper that uses `self.full_graph` and handles cold start. Add `predict_fraud_served()`:
```python
def predict_fraud_served(self, txn: Dict[str, Any], return_confidence=True, return_explanation=False) -> Dict[str, Any]:
    if self.model is None:
        raise ValueError("Model not loaded")
    sender = txn.get("sender_id"); receiver = txn.get("receiver_id")
    known = self.node_mapping and sender in self.node_mapping and receiver in self.node_mapping
    if self.full_graph is not None and known:
        result = self.predict_fraud(txn, full_graph=self.full_graph,
                                    return_confidence=return_confidence,
                                    return_explanation=return_explanation)
        result["scoring_path"] = "gnn"
        return result
    # cold start: see Step 4
    result = self._predict_cold_start(txn, return_confidence)
    result["scoring_path"] = "cold_start_heuristic"
    return result
```

### Step 4 — Cold-start handling (unknown users)
Real traffic includes users not in the training graph. Provide a defensible cold-start path and label it honestly:
- **Option 1 (preferred):** dynamically attach the new transaction as a temporary node/edge connected to any known counterparty, build a k-hop subgraph, and run the GNN with the new node's features (computed from the transaction + any known counterparty aggregates). Document the feature defaults used for the unseen node.
- **Option 2 (fallback):** if neither party is known, run the **heuristic** but tag `scoring_path="cold_start_heuristic"` and set lower confidence. This is the only sanctioned use of the heuristic, and it must be surfaced in the response.

Implement `_predict_cold_start()` accordingly. Keep the heuristic math (`predict.py:298-312`) only inside this clearly-labeled method; **remove** the silent heuristic from the main GNN path's `else` branch or guard it behind an explicit flag.

### Step 5 — Wire the API endpoint to the served path
In `src/api/main.py`:
- `/predict` (`:317`): call `predictor.predict_fraud_served(transaction_dict, return_confidence=..., return_explanation=...)`.
- `/predict/batch` (`:390`): add `predict_batch_served()` that loops `predict_fraud_served` (or batches subgraphs); update the call site.
- Add `scoring_path` to `PredictionOutput` (`schemas.py`) so clients can see whether a result came from the GNN or cold-start. Update `/model/status` to report whether real artifacts are loaded (`graph_loaded`, `num_nodes`, `num_features`, `scoring_default`).

### Step 6 — Startup: load the graph
In the lifespan handler (`main.py:60-78`), `load_production_model()` now loads the graph too (Step 2). Add a startup log line summarizing the loaded graph and fail fast (return `degraded` health) if no real artifacts are present, so the deployment story is explicit rather than silently heuristic.

### Step 7 — Performance guardrails
- Cap subgraph size: in `create_subgraph_for_transaction()` (`predict.py:183-221`), if a k-hop subgraph exceeds N nodes (e.g., 5000), sample neighbors (DGL `sample_neighbors`) to bound latency. Make N configurable (`config.MAX_INFERENCE_SUBGRAPH_NODES`).
- Run inference under `torch.no_grad()` (already done at `:266`).
- Log per-request inference time; target < 200 ms CPU for typical subgraphs (record actuals).

## Contract / data changes
- `PredictionOutput` gains `scoring_path: Literal["gnn","cold_start_heuristic"]`.
- `/model/status` gains `graph_loaded`, `num_nodes`, `num_features`, `scoring_default`.
- New config: `MAX_INFERENCE_SUBGRAPH_NODES` (default 5000).

## Acceptance criteria
- [ ] On startup with `models/production/` present, logs show the graph loaded (node/feature counts) and `/model/status` reports `graph_loaded=true`.
- [ ] `POST /predict` for a transaction whose sender+receiver are in `node_mapping` returns `scoring_path="gnn"` and a probability produced by `GraphSAGEClassifier` (verified by comparing to a direct model call on the same subgraph in a test).
- [ ] `POST /predict` for unknown users returns `scoring_path="cold_start_heuristic"` (or a documented dynamic-node GNN result) — never a silent heuristic mislabeled as GNN.
- [ ] The heuristic no longer runs on the main GNN path without the `cold_start` label (`grep` shows the heuristic only inside `_predict_cold_start`).
- [ ] Inference latency logged; subgraph size cap enforced.
- [ ] `make check` passes; new tests green.

## Test plan
Create `tests/integration/test_api_predict.py` additions + `tests/unit/test_predict.py`:
- **Fixture:** a tiny saved artifact set under `tests/fixtures/mini_production/` (3–10 node graph, a 2-class toy `GraphSAGEClassifier` with the same `model_config` shape) so tests run without the real 5.9 MB model.
- `test_served_known_users_uses_gnn`: load mini artifacts; predict for known users; assert `scoring_path=="gnn"` and the API probability == direct `model.predict_proba` on the same subgraph (within float tolerance).
- `test_served_unknown_users_cold_start`: predict for unknown ids; assert `scoring_path=="cold_start_heuristic"`.
- `test_model_status_reports_graph`: `/model/status` shows `graph_loaded=true`, correct `num_features`.
- `test_subgraph_node_cap`: build a star graph exceeding the cap; assert the subgraph is sampled down.

## Validation
```bash
# with models/production present (or mini fixture wired via env)
python -m uvicorn src.api.main:app --port 8002 &
sleep 8
curl -s localhost:8002/model/status | python -m json.tool      # expect graph_loaded: true
curl -s -X POST localhost:8002/predict -H 'content-type: application/json' \
  -d '{"sender_id":"<known>","receiver_id":"<known>","amount":150000,"type":"TRANSFER"}' | python -m json.tool
kill %1
pytest tests/unit/test_predict.py tests/integration/test_api_predict.py -q
grep -n "fraud_probability = 0.1" src/gnn_model/predict.py   # should now live only in _predict_cold_start
```

## Rollback / fallback
If the real graph is too large to load in the target environment, document a downsampled inference graph (subset of nodes) in `model_card.md` and load that. If artifacts are unavailable, the app must report `degraded` and label every prediction `cold_start_heuristic` — never claim GNN scoring without the model.

## Definition of Done
`/predict` returns genuine GNN scores for known users with an explicit `scoring_path`; cold start is handled and labeled; tests prove the API matches a direct model call; commit on `feat/spec-06-real-gnn-inference`.

## References
- `src/api/main.py:60-78,256-356`, `src/gnn_model/predict.py:41-66,74-152,183-221,223-338,475-508`, `src/gnn_model/model.py:136-178,275-296`, `src/api/schemas.py` (`PredictionOutput`, `ModelStatus`), `models/production/` (spec 05).

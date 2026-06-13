# System Design Interview Answer
## "Design an Explainable Fraud Detection System from Scratch"

---

## HOW TO USE THIS DOCUMENT

When asked "design this system on a whiteboard," follow the **RADIO framework** in order. Each section below maps to one phase. Use the timing guide at the end to pace yourself.

---

## RADIO FRAMEWORK APPLIED TO THIS PROJECT

| Phase | What It Means | Time (45-min interview) |
|---|---|---|
| **R** — Requirements | Clarify scope, scale, constraints | 5 min |
| **A** — Architecture | High-level diagram, components | 10 min |
| **D** — Data Model | Storage design, schemas, relationships | 8 min |
| **I** — Interface | APIs, contracts, request/response shapes | 7 min |
| **O** — Optimization | Bottlenecks, scaling, trade-offs | 10 min |
| Closing | Differentiator + future roadmap | 5 min |

---

## STEP 1 — REQUIREMENTS: CLARIFYING QUESTIONS

Ask these in the first 5 minutes. The answers for *this* project are provided.

| Question to Ask | Answer for This Project |
|---|---|
| "What is the primary objective — real-time blocking or post-hoc investigation?" | Real-time scoring (~150ms) + optional explanation (~2.5s) |
| "What transaction volume are we expecting?" | ~6.36M historical transactions; PaySim dataset; 500 req/min at steady state |
| "What is the fraud rate in production?" | ~0.13% — extreme class imbalance; this dictates the loss function choice |
| "Do we need to explain individual predictions to end users or compliance teams?" | Yes — compliance teams need human-readable justification for each flagged transaction |
| "What is the acceptable false-positive rate?" | ~10.8% (89.2% precision) — roughly 1 in 9 flagged is a false alarm |
| "Do we need to handle new users not seen during training?" | Yes — production introduces new account IDs constantly (→ must be inductive) |
| "What latency SLA exists?" | Prediction: 150ms P95; Explanation: 2.5s P95 |
| "Is the model trained once or continuously retrained?" | Batch retraining; MLflow tracks experiments; production promotion via registry |
| "What regulatory requirements exist?" | Explanations must be human-readable for compliance/audit; no black-box decisions |
| "Do we need multi-tenant isolation?" | Single-tenant for this version; model registered under Production stage in MLflow |

---

## STEP 2 — ARCHITECTURE: HIGH-LEVEL DIAGRAM

```
                          ┌─────────────────────────────────┐
                          │         DATA SOURCES             │
                          │  PaySim CSV / Live Transaction   │
                          │  Feed (Kafka in future version)  │
                          └────────────────┬────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │            INGESTION LAYER                   │
                    │                                             │
                    │  ┌──────────────────────────────────────┐   │
                    │  │      GraphConstructor                 │   │
                    │  │  (graph_constructor.py)               │   │
                    │  │                                       │   │
                    │  │  1. KaggleHub download                │   │
                    │  │  2. Schema validation (11 cols)       │   │
                    │  │  3. Feature engineering               │   │
                    │  │     • amount_log = log1p(amount)      │   │
                    │  │     • balance deltas                  │   │
                    │  │     • hour_of_day, day_of_month       │   │
                    │  │     • type_encoded (0-4)              │   │
                    │  │  4. Node aggregation (14 features)    │   │
                    │  │  5. Edge construction + weighting     │   │
                    │  └──────────────┬───────────────────────┘   │
                    └─────────────────┼───────────────────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │           STORAGE LAYER             │
                    │                                     │
                    │  ┌──────────┐   ┌────────────────┐ │
                    │  │  CSV     │   │   Neo4j 5.12   │ │
                    │  │ (offline │   │  (graph DB for │ │
                    │  │  cache)  │   │  explanations) │ │
                    │  │graph_    │   │                │ │
                    │  │nodes.csv │   │  User nodes    │ │
                    │  │graph_    │   │  TRANSACTION   │ │
                    │  │edges.csv │   │  relationships │ │
                    │  └────┬─────┘   └───────┬────────┘ │
                    └───────┼─────────────────┼──────────┘
                            │                 │
                    ┌───────▼─────────────────┼──────────────┐
                    │      TRAINING LAYER      │              │
                    │                          │              │
                    │  ┌──────────────────┐    │              │
                    │  │ GraphSAGEClassfr │    │              │
                    │  │ (model.py)       │    │              │
                    │  │                  │    │              │
                    │  │  10→128→64→32→1  │    │              │
                    │  │  SAGEConv ×2     │    │              │
                    │  │  BatchNorm ×2    │    │              │
                    │  │  MLP classifier  │    │              │
                    │  └────────┬─────────┘    │              │
                    │           │              │              │
                    │  ┌────────▼─────────┐    │              │
                    │  │FraudDetectionTrn │    │              │
                    │  │(training.py)     │    │              │
                    │  │ BCEWithLogitsLoss│    │              │
                    │  │ Adam lr=0.001    │    │              │
                    │  │ ReduceLROnPlateau│    │              │
                    │  │ EarlyStopping    │    │              │
                    │  │ GradClip 1.0     │    │              │
                    │  └────────┬─────────┘    │              │
                    │           │              │              │
                    │  ┌────────▼─────────┐    │              │
                    │  │  MLflow Registry │    │              │
                    │  │  fraud-detection │    │              │
                    │  │  -model/Prod     │    │              │
                    │  └──────────────────┘    │              │
                    └──────────────────────────┼──────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────┐
                    │               SERVING LAYER                  │
                    │                                              │
                    │   ┌──────────────────────────────────────┐  │
                    │   │          FastAPI (main.py)            │  │
                    │   │  Lifespan: loads model + agent        │  │
                    │   │  Middleware: CORS, TrustedHost,       │  │
                    │   │             RequestID, ProcessTime    │  │
                    │   │                                       │  │
                    │   │  POST /predict ─────► FraudPredictor  │  │
                    │   │  POST /predict/batch ► FraudPredictor  │  │
                    │   │  POST /explain ──────► AIInvestigator  │  │
                    │   │  GET  /health ───────► HealthCheck     │  │
                    │   │  GET  /model/status ─► ModelStatus     │  │
                    │   └──────────────────────────────────────┘  │
                    │                  │              │            │
                    │          ┌───────▼──┐    ┌──────▼────────┐  │
                    │          │FraudPred │    │AIInvestigator │  │
                    │          │(predict  │    │(agent.py)     │  │
                    │          │.py)      │    │               │  │
                    │          │          │    │LangChain      │  │
                    │          │MLflow    │    │ReAct Agent    │  │
                    │          │model load│    │  +            │  │
                    │          │khop sub  │    │Neo4jTxnTool   │  │
                    │          │graph     │    │  +            │  │
                    │          │sigmoid   │    │Gemini 1.5 Pro │  │
                    │          └──────────┘    └───────────────┘  │
                    └─────────────────────────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────┐
                    │            MLOPS / MONITORING LAYER          │
                    │                                              │
                    │  ┌─────────────┐  ┌──────────────┐          │
                    │  │   MLflow    │  │HealthMonitor │          │
                    │  │  Tracking   │  │MetricsCollect│          │
                    │  │  Artifacts  │  │monitoring/   │          │
                    │  └─────────────┘  └──────────────┘          │
                    └─────────────────────────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────┐
                    │              CI/CD LAYER                     │
                    │                                              │
                    │  GitHub Actions (.github/workflows/ci.yml)  │
                    │  test → security → build → integration →    │
                    │  staging → production → release → cleanup   │
                    │                                              │
                    │  Docker (multi-stage) → GHCR Registry       │
                    └─────────────────────────────────────────────┘
```

---

## STEP 3 — DATA MODEL

### Storage System Choices

| System | Type | Used For | Why Not the Others |
|---|---|---|---|
| **CSV (local)** | Flat file | Offline processed graph data cache | Simple, human-inspectable, written once per pipeline run |
| **Neo4j** | Native graph DB | Online explanation queries — neighborhood traversal | SQL self-joins are O(n²) for graph traversal; Cypher is O(1) with indexes |
| **MLflow (SQLite)** | Relational + blob store | Model registry, experiment tracking, artifact storage | Industry standard for MLOps; staged promotions; URI-based model loading |

### Node Features (User)
| Column | Type | Derived From | Purpose |
|---|---|---|---|
| `user_id` | string (UNIQUE) | `nameOrig` / `nameDest` | Graph node identifier |
| `total_transactions` | int | Count of all txns | Usage volume feature |
| `transactions_as_originator` | int | Count where `nameOrig` | Sending behavior |
| `transactions_as_destination` | int | Count where `nameDest` | Receiving behavior |
| `total_amount_sent` | float | Sum of `amount` as sender | Financial exposure |
| `total_amount_received` | float | Sum of `amount` as receiver | Financial exposure |
| `avg_amount_sent` | float | Mean of sent amounts | Typical send size |
| `avg_amount_received` | float | Mean of received amounts | Typical receive size |
| `net_amount` | float | received - sent | Net position |
| `fraud_transactions` | int | Count where `isFraud=1` | Direct fraud signal |
| `fraud_rate` | float | fraud_txns / total_txns | Primary label / risk signal |
| `is_active_sender` | bool | `count >= MIN_TRANSACTION_COUNT (5)` | Activity flag |
| `is_active_receiver` | bool | `count >= MIN_TRANSACTION_COUNT (5)` | Activity flag |

### Edge Features (Transaction)
| Column | Type | Derived From | Purpose |
|---|---|---|---|
| `transaction_id` | int | Sequential index | Edge identifier |
| `source_user` | string | `nameOrig` | Sender node |
| `target_user` | string | `nameDest` | Receiver node |
| `step` | int | Raw PaySim | Time step (hours) |
| `type` | string | Raw PaySim | Transaction type (5 categories) |
| `amount` | float | Raw PaySim | Transaction value |
| `amount_log` | float | `log1p(amount)` | Log-scale normalization |
| `balance_change_orig` | float | `newbalanceOrig - oldbalanceOrg` | Sender balance delta |
| `balance_change_dest` | float | `newbalanceDest - oldbalanceDest` | Receiver balance delta |
| `hour_of_day` | int | `step % 24` | Time-of-day signal |
| `day_of_month` | int | `(step // 24) % 30` | Day signal |
| `type_encoded` | int | `pd.Categorical(type).codes` | Numeric type for DGL |
| `weight` | float | `amount / max_amount` (min 0.1) | Edge weight for message passing |
| `is_fraud` | int (0/1) | `isFraud` | Ground truth label |
| `is_flagged_fraud` | int (0/1) | `isFlaggedFraud` | Bank's own flag |

### Design Decision: Why Pre-compute Node Features Instead of Computing at Query Time?
The `fraud_rate`, `total_amount_sent`, and `avg_amount_sent` could always be recomputed by aggregating the TRANSACTION edges. Pre-storing them on the User node trades storage space for query speed — the `AIInvestigator._get_user_profile()` Neo4j query returns them in a single node lookup (`MATCH (u:User {user_id: $user_id}) RETURN u.fraud_rate ...`) rather than an aggregation scan over all edges.

---

## STEP 4 — INTERFACE (API DESIGN)

### Core Endpoints

| Endpoint | Method | Input Schema | Output Schema | SLA |
|---|---|---|---|---|
| `/predict` | POST | `TransactionInput` | `APIResponse[PredictionOutput]` | 150ms |
| `/predict/batch` | POST | `BatchTransactionInput` (max 100) | `APIResponse[BatchPredictionOutput]` | 800ms |
| `/explain` | POST | `ExplanationRequest` | `APIResponse[ExplanationOutput]` | 2.5s |
| `/health` | GET | — | `HealthCheck` | <10ms |
| `/model/status` | GET | — | `ModelStatus` | <50ms |
| `/predictions/history` | GET | `limit=100` query param | `APIResponse[dict]` | <100ms |

### Key Request/Response Contracts

**`TransactionInput` required fields:**
- `sender_id`: str (1-50 chars)
- `receiver_id`: str (1-50 chars)
- `amount`: float (0 < x ≤ 1,000,000)
- `type`: enum (CASH_IN | CASH_OUT | DEBIT | PAYMENT | TRANSFER)

**`TransactionInput` validation rules:**
- `@validator`: strips whitespace from user IDs
- `@root_validator`: checks `sender_old_balance - amount ≈ sender_new_balance` (±0.01 tolerance)

**`PredictionOutput` fields:**
- `fraud_probability`: float [0, 1]
- `is_fraud_predicted`: bool
- `confidence`: float = `abs(prob - 0.5) × 2` (distance from decision boundary)
- `risk_level`: LOW | MEDIUM | HIGH | CRITICAL
- `processing_time_ms`: float

**`ExplanationOutput` fields:**
- `explanation_text`: str (max 1000 tokens from LLM)
- `key_factors`: List[str] (top 5)
- `risk_indicators`: Dict[str, Union[str, float]]
- `recommendation`: str
- `explanation_confidence`: float [0, 1]

**`APIResponse` envelope (all endpoints):**
```json
{
  "success": true,
  "data": { ... },
  "message": "Fraud prediction completed successfully",
  "timestamp": "2024-01-01T00:00:00",
  "request_id": "uuid4"
}
```

### Response Headers (all requests)
- `X-Request-ID`: UUID4 generated per request by middleware
- `X-Process-Time`: Wall clock time in milliseconds

---

## STEP 5 — FIVE MOST IMPORTANT DESIGN DECISIONS

### Decision 1: GraphSAGE (Inductive GNN) over Tabular ML

| | |
|---|---|
| **Decision made** | `GraphSAGEClassifier` in `src/gnn_model/model.py` using `SAGEConv` from DGL with `aggregator_type='mean'` |
| **Alternative considered** | XGBoost or Random Forest on per-transaction features |
| **Why this choice** | Fraud is relational: smurfing, layering, hub accounts are multi-hop network patterns invisible to tabular models. GraphSAGE's message-passing captures 2-hop neighborhood context, yielding 90.6% F1 vs. ~75% F1 typical for tabular approaches on PaySim |
| **Trade-off accepted** | More complex to operationalize (requires graph construction, DGL, subgraph inference); harder to debug than feature importance in XGBoost |

---

### Decision 2: Node-Level Classification Instead of Edge-Level

| | |
|---|---|
| **Decision made** | Labels assigned at user node level (`fraud_rate`); GNN classifies nodes |
| **Alternative considered** | Label individual transactions as edges; edge-level classification |
| **Why this choice** | Node embedding aggregates the user's full transaction history + network context. Edge-level would require per-transaction labels and edge classification head. Node-level leverages more signal and is cleaner for the GraphSAGE architecture |
| **Trade-off accepted** | Predicts a user's *fraud propensity*, not the *specific fraudulent transaction* — the API routes inference through the sender node |

---

### Decision 3: LangChain ReAct Agent for Explanations

| | |
|---|---|
| **Decision made** | `CONVERSATIONAL_REACT_DESCRIPTION` agent with `Neo4jTransactionTool`; `max_iterations=3`; `ConversationBufferWindowMemory(k=5)` |
| **Alternative considered** | (a) Hardcoded Cypher queries + template-based explanation; (b) Direct Gemini prompt with pre-fetched context |
| **Why this choice** | The ReAct loop allows the LLM to dynamically decide which Neo4j queries are needed based on the transaction context, rather than hardcoding a fixed query sequence. This handles edge cases (new users, unusual transaction types) more flexibly |
| **Trade-off accepted** | Non-deterministic latency (1-3 tool calls); `max_iterations=3` bound prevents runaway; brittle text-based output parser (`_parse_agent_response()`) — acknowledged improvement area |

---

### Decision 4: Full-Graph Training over Mini-Batch Sampling

| | |
|---|---|
| **Decision made** | Full graph passed in each training epoch; `train_mask` used to select training nodes |
| **Alternative considered** | GraphSAINT node sampling; NeighborSampler mini-batch; cluster-GCN |
| **Why this choice** | PaySim graph fits entirely in memory (~6.36M transactions). Full-graph training gives exact gradients (no sampling noise), simpler code, and faster convergence on this scale |
| **Trade-off accepted** | Doesn't scale beyond ~50M nodes on a standard GPU. Mini-batch would be required for production-scale financial graphs |

---

### Decision 5: Graceful Degradation Architecture

| | |
|---|---|
| **Decision made** | Prediction and explanation are fully independent; each fails gracefully without affecting the other |
| **Alternative considered** | Monolithic prediction+explanation in a single request; fail-fast design |
| **Why this choice** | Gemini API outages and Neo4j outages are both plausible production failure modes. Making prediction independent of explanation preserves the critical path (fraud scoring) even when the value-add (explanation) is unavailable |
| **Trade-off accepted** | More complex initialization logic (`lifespan()` with separate try/except blocks for each service); two separate dependency functions (`get_fraud_predictor()` and `get_ai_investigator()`) |

---

## STEP 6 — SCALABILITY ANALYSIS

### Current Bottlenecks at Current Scale

| Bottleneck | Location | Current Handling | Breaking Point |
|---|---|---|---|
| Full-graph training memory | `training.py` | Entire graph in RAM | ~50-100M nodes on 16GB GPU |
| Gemini API rate limits | `agent.py` | None (relies on Gemini quotas) | > 50 req/min |
| Synchronous LangChain agent | `agent.py` | `asyncio.to_thread()` workaround | CPU-bound, not I/O-bound; thread pool saturation at scale |
| In-memory prediction history | `predict.py` | `self.prediction_history` list | Memory leak at high volume (no TTL/eviction) |
| SQLite MLflow backend | `config.py` | Works locally | Not suitable for multi-worker deployment |
| Single uvicorn worker | `config.py` | `API_WORKERS=1` default | Must be increased for production throughput |

### What Would Change at 10× Load (5M requests/day)

| Component | Change Required |
|---|---|
| **Training** | Switch to `dgl.dataloading.NeighborSampler` for mini-batch training; distribute with DDP |
| **Serving** | Increase `API_WORKERS` to match CPU cores; add nginx load balancer |
| **Model loading** | Model server (Triton or TorchServe) with GPU inference; batch inference for /predict/batch |
| **Prediction history** | Replace in-memory list with Redis with TTL; push to time-series DB |
| **MLflow** | Remote tracking server (PostgreSQL backend + S3 artifacts) |
| **Neo4j** | Neo4j cluster with read replicas for explanation queries |
| **Explanations** | Queue-based async processing (Celery + Redis); webhook callback instead of synchronous wait |
| **Feature updates** | Real-time feature store (Feast + Redis) for live user statistics instead of pre-computed CSV |

### Future Architecture at 100× Load

```
[Live Transaction Feed]
        │ Kafka topic: transactions
        ▼
[Stream Processor] (Flink/Spark Streaming)
        │ Real-time feature computation
        ▼
[Feature Store] (Feast + Redis)
        │ User features served <5ms
        ▼
[Model Server] (Triton Inference Server)
        │ GraphSAGE on GPU; batch inference
        ▼
[API Gateway] (Kong/nginx)
        │ Rate limiting, auth, routing
        ▼
[Microservices]
  ├── Prediction Service (k8s, HPA)
  └── Explanation Service (k8s, queue-based)
        │ Async via Celery + Redis
        ▼
[Compliance Dashboard] (Grafana + Prometheus)
```

---

## PRE-BUILT ANSWERS FOR 5 MOST COMMON FOLLOW-UP QUESTIONS

### 1. "How would you handle concept drift — when fraud patterns change over time?"

"The current system uses MLflow experiment tracking to record every training run's metrics. The `HealthMonitor` in `src/health_monitoring.py` tracks real-time fraud detection rate. If production fraud rates diverge significantly from training baselines, that's a drift signal. I'd add three things: (a) a scheduled retraining pipeline that runs weekly on the last N months of transactions, (b) a shadow model that runs in parallel and whose predictions are compared to the production model, and (c) population stability index (PSI) monitoring on key input features. The MLflow model registry with staged promotions means the retraining pipeline can promote a new model to Production without downtime."

### 2. "What if the dataset is too large to fit in memory for training?"

"The current `train_epoch()` in `training.py` does a full-graph forward pass. At scale, I'd switch to mini-batch training using DGL's `NeighborSampler`: for each batch of target nodes, sample K neighbors at each hop and compute a local forward pass on just that sampled subgraph. The rest of the training infrastructure — MLflow, early stopping, LR scheduling — stays unchanged. The `GraphDataLoader.get_subgraph()` method in `model.py` already shows the subgraph extraction logic for inference; training would use the same pattern."

### 3. "How would you ensure the model doesn't unfairly flag certain demographic groups?"

"This is a fairness concern that's critical in financial fraud detection. The current system uses transaction behavior and network structure as features — it doesn't include demographic attributes like location or account age by name. However, proxy discrimination is still possible. I'd add: (a) fairness metrics in the evaluation pipeline (`MetricsCalculator.calculate_binary_metrics()` in `training.py`) computed per segment, (b) adversarial debiasing during training, and (c) the AI explanation layer actually helps here — requiring the LLM to cite specific behavioral reasons for every flagging decision creates an audit trail that can be reviewed for systematic bias."

### 4. "How do you prevent the LLM from hallucinating in explanations?"

"Three strategies are already in place: First, `temperature=0.3` keeps outputs deterministic and analytical. Second, the ReAct agent is grounded in real data — every explanation is preceded by a Neo4j tool call that retrieves actual user statistics. The LLM can't hallucinate numbers it hasn't seen because the tool output is in its context. Third, `max_output_tokens=1000` bounds the output length, reducing the opportunity for runaway generation. What I'd add: structured output parsing via `model.with_structured_output(ExplanationOutput)` to force JSON schema compliance, and a post-hoc fact-check that verifies any numeric claims in the explanation against the Neo4j data."

### 5. "How would you version the prompts and track prompt performance?"

"This is a gap in the current implementation. The prompts are hardcoded strings in `_create_investigation_prompt()` in `agent.py`. The right approach is to treat prompts as versioned artifacts — store them in a prompt registry (could be MLflow's artifact store, LangSmith, or even a YAML file with semantic versions). When a prompt changes, log the new version, run the evaluation suite on 100 sampled transactions, and compare explanation quality metrics (coherence score, factor coverage) between versions. The `explanation_depth` parameter in `ExplanationRequest` is already a form of prompt configuration — extending this to prompt version selection would be the next step."

---

## TIMING GUIDE (45-MINUTE INTERVIEW)

| Time | What to Cover | Output |
|---|---|---|
| **0:00 – 5:00** | Ask clarifying questions from Step 1. Don't start drawing until you have answers. | Agreement on scope |
| **5:00 – 15:00** | Draw the high-level architecture. Top-down: data sources → ingestion → storage → training → serving → monitoring. Name every box. | System diagram |
| **15:00 – 23:00** | Data model deep dive. Draw Node vs. Edge schemas. Explain why Neo4j over SQL for graph traversal. Show the indexes. | Schema diagram |
| **23:00 – 30:00** | API interface. Write out the 3 key endpoints. Show the request/response shapes. Mention Pydantic validation. | Interface spec |
| **30:00 – 40:00** | Design decisions. Walk through all 5 decisions. For each: "I chose X over Y because Z, and the trade-off I accepted is W." | Decision matrix |
| **40:00 – 45:00** | Scalability. Name current bottlenecks. Describe 10× architecture changes. Close with differentiator. | Scaling plan |

---

## CLOSING STATEMENT (Differentiator)

> "Most fraud detection systems are black boxes — they produce a score and leave analysts guessing why. This system combines a GraphSAGE model that captures multi-hop network patterns invisible to tabular approaches, with an AI explanation layer that queries the live transaction graph in Neo4j and generates compliance-ready justifications in plain English. The architecture is designed for graceful degradation — the prediction service never goes down because of an LLM outage — and the full MLOps stack with MLflow tracking, GitHub Actions CI/CD, and staged model promotions means every change is auditable and reversible. The result is a system that's not just accurate, but operationally trustworthy."

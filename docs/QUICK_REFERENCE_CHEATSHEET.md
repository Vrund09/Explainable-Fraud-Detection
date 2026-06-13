# Quick Reference Cheat Sheet — Explainable Fraud Detection
### Read this 10 minutes before your interview

---

## 1. ALL KEY NUMBERS (memorize these)

| Category | Metric | Value | Context |
|---|---|---|---|
| **Model** | F1-Score | **90.6%** | On PaySim test set; industry benchmark 86% |
| **Model** | ROC-AUC | **96.3%** | Industry benchmark 93% |
| **Model** | Accuracy | **94.5%** | Industry benchmark ~90% |
| **Model** | Precision | **89.2%** | Industry benchmark ~85% |
| **Model** | Recall | **92.1%** | Industry benchmark ~88% |
| **Dataset** | Total transactions | **6.36 million** | PaySim mobile money simulation |
| **Dataset** | Fraud rate | **~0.13%** | Severe class imbalance |
| **Dataset** | Transaction types | **5** | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| **Architecture** | GNN input dim | **10** | `GNN_INPUT_DIM` in `config.py` |
| **Architecture** | GNN hidden dim | **128** | `GNN_HIDDEN_DIM` in `config.py` |
| **Architecture** | GNN output dim | **64** | `GNN_OUTPUT_DIM` in `config.py` |
| **Architecture** | Classifier hidden dim | **32** | `CLASSIFIER_HIDDEN_DIM` in `config.py` |
| **Architecture** | GNN layers | **2** | `GNN_NUM_LAYERS` in `config.py` |
| **Architecture** | Dropout rate | **0.2** | `GNN_DROPOUT_RATE` in `config.py` |
| **Training** | Learning rate | **0.001** | Adam optimizer |
| **Training** | Batch size | **512** | `BATCH_SIZE` in `config.py` |
| **Training** | Max epochs | **100** | `NUM_EPOCHS` in `config.py` |
| **Training** | Early stop patience | **10** | `EARLY_STOPPING_PATIENCE` in `config.py` |
| **Training** | Gradient clip | **1.0** | `max_norm=1.0` in `train_epoch()` |
| **Training** | Train/Val/Test split | **70/15/15** | `config.py` ratios |
| **API** | Predict latency | **~150ms** | Single transaction |
| **API** | Explain latency | **~2.5s** | Full AI investigation |
| **API** | Batch latency | **~800ms** | 50 transactions |
| **API** | Batch throughput | **500 req/min** | `/predict` endpoint |
| **API** | Explain throughput | **50 req/min** | `/explain` endpoint |
| **API** | Max batch size | **100** | `max_items=100` in `BatchTransactionInput` |
| **API** | Max amount | **1,000,000** | `MAX_AMOUNT_THRESHOLD` in `config.py` |
| **API** | Rate limit | **100 req/min** | `API_RATE_LIMIT` in `config.py` |
| **Data** | Min txn count per user | **5** | `MIN_TRANSACTION_COUNT` in `config.py` |
| **Data** | Edge weight threshold | **0.1** | `EDGE_WEIGHT_THRESHOLD` in `config.py` |
| **Neo4j** | Connection pool size | **50** | `NEO4J_MAX_CONNECTION_POOL_SIZE` |
| **Neo4j** | Connection lifetime | **3600s** | `NEO4J_MAX_CONNECTION_LIFETIME` |
| **Neo4j** | Node batch size | **1000** | `ingest_nodes_to_neo4j()` |
| **Neo4j** | Edge batch size | **500** | `ingest_edges_to_neo4j()` |
| **LLM** | Temperature | **0.3** | Low — deterministic fraud analysis |
| **LLM** | Max output tokens | **1000** | `MAX_EXPLANATION_LENGTH` in `config.py` |
| **LLM** | Agent max iterations | **3** | `max_iterations=3` in `_initialize_agent()` |
| **LLM** | Memory window | **5** | `ConversationBufferWindowMemory(k=5)` |
| **Infra** | API port | **8000** | `API_PORT` in `config.py` |
| **Infra** | LR scheduler factor | **0.5** | Halves LR on plateau |
| **Infra** | LR scheduler patience | **5** | Epochs before reducing LR |
| **CI** | Pipeline jobs | **8** | In `.github/workflows/ci.yml` |
| **Docker** | Non-root UID | **1000** | `appuser` in `Dockerfile` |
| **Risk** | LOW threshold | **< 0.25** | `_get_risk_level()` in `predict.py` |
| **Risk** | MEDIUM threshold | **0.25 – 0.5** | `_get_risk_level()` in `predict.py` |
| **Risk** | HIGH threshold | **0.5 – 0.75** | `_get_risk_level()` in `predict.py` |
| **Risk** | CRITICAL threshold | **≥ 0.75** | `_get_risk_level()` in `predict.py` |

---

## 2. COMPLETE TECH STACK

| Technology | Version | Role | WHY This Was Chosen |
|---|---|---|---|
| **Python** | 3.11 | Language | Ecosystem for ML; 3.11 perf improvements; CI locks version |
| **PyTorch** | 2.0.1 | Deep learning | Industry standard; DGL uses it as backend |
| **DGL** | 1.1.2 | GNN framework | `SAGEConv` with `edge_weight` support; `khop_subgraph()` for inference |
| **GraphSAGE (SAGEConv)** | — | Core model | Inductive — works on unseen nodes at production time |
| **FastAPI** | 0.103.1 | REST API | Async support; auto OpenAPI docs from Pydantic; `Depends()` injection |
| **Uvicorn** | 0.23.2 | ASGI server | Production ASGI server for FastAPI |
| **Pydantic** | 2.3.0 | Validation | Request/response validation + OpenAPI schema generation |
| **Neo4j** | 5.12.0 | Graph database | Native Cypher graph traversal; neighborhood queries in O(1) with indexes |
| **LangChain** | 0.0.284 | LLM orchestration | ReAct agent + tool use + memory; provider-agnostic LLM wrapper |
| **Google Gemini** | gemini-1.5-pro-latest | LLM | Large context window (1M tokens); favorable API pricing |
| **MLflow** | 2.7.1 | Experiment tracking | Model registry with staged promotions; per-run artifact logging |
| **Pandas** | 2.0.3 | Data manipulation | DataFrame operations for graph construction |
| **NumPy** | 1.24.3 | Numerical ops | `log1p`, normalization, array ops in preprocessing |
| **scikit-learn** | 1.3.0 | Metrics | F1, ROC-AUC, PR-AUC, confusion matrix |
| **Docker** | — | Containerization | Multi-stage build; non-root user; reproducible environment |
| **GitHub Actions** | — | CI/CD | 8-job pipeline; secrets management; GHCR image registry |
| **KaggleHub** | latest | Dataset download | Auto-downloads PaySim; fallback to local CSV |
| **python-dotenv** | 1.0.0 | Config | `.env` file loading; keeps secrets out of code |
| **torch-geometric** | 2.3.1 | GNN (alt) | Installed as backup; DGL used as primary |
| **pytest** | 7.4.2 | Testing | Standard test framework; `--cov` for coverage |
| **pytest-asyncio** | 0.21.1 | Async testing | Tests for async FastAPI endpoints |
| **httpx** | 0.24.1 | HTTP test client | Async API testing |
| **black** | 23.7.0 | Code formatting | Enforced in CI via `black --check` |
| **flake8** | 6.0.0 | Linting | Enforced in CI; `max-line-length=127` |
| **isort** | 5.12.0 | Import sorting | Enforced in CI via `isort --check-only` |
| **mypy** | 1.5.1 | Type checking | `continue-on-error: true` in CI; type hints throughout |
| **bandit** | — | Security linting | Static AST security scan on `src/` in CI |
| **safety** | — | Dependency CVE scan | Runs in CI security job |
| **Trivy** | — | Container scanning | Scans Docker image; SARIF uploaded to GitHub Security |
| **matplotlib** | 3.7.2 | Training plots | `plot_training_history()` in `training.py` |
| **seaborn** | 0.12.2 | Visualization | Confusion matrix heatmaps |
| **networkx** | 3.1 | Graph utilities | Graph analysis utilities |
| **DVC** | 3.27.0 | Data versioning | Dataset tracking with S3 remote |

---

## 3. SYSTEM ARCHITECTURE (ASCII)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE (Offline)                         │
│                                                                     │
│  [Kaggle/KaggleHub]                                                 │
│       │ PaySim CSV (6.36M txns)                                     │
│       ▼                                                             │
│  [GraphConstructor]  ──────────────────────────────────────────┐   │
│   graph_constructor.py                                         │   │
│   • load_raw_data()        • validate_data_schema()            │   │
│   • preprocess_data()      • create_nodes_dataframe()          │   │
│   • create_edges_dataframe()                                   │   │
│       │                                                        │   │
│       ├── graph_nodes.csv (Users + features)                   │   │
│       └── graph_edges.csv (Txns + features)                    │   │
│                           │                                    │   │
│                           ▼                                    ▼   │
│                    [Neo4j DB]                          [DGL Graph]  │
│                  User nodes + TRANSACTION              g.ndata[]   │
│                  relationships + indexes               g.edata[]   │
└─────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE (Offline)                     │
│                                                                     │
│  [GraphDataLoader]   →  [GraphSAGEClassifier]  →  [FraudDetectionTrainer]│
│   model.py               model.py                  training.py     │
│                           10→128→64→32→1            • BCEWithLogitsLoss│
│                           SAGEConv ×2               • Adam lr=0.001 │
│                           BatchNorm ×2              • ReduceLROnPlateau│
│                           MLP ×3                    • EarlyStopping  │
│                                                     • GradClip 1.0  │
│                                │                                    │
│                                ▼                                    │
│                        [MLflow Registry]                            │
│                    fraud-detection-model/Production                 │
└─────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER (Online)                         │
│                                                                     │
│  HTTP Request                                                       │
│       │                                                             │
│       ▼                                                             │
│  [FastAPI]  src/api/main.py                                        │
│  ┌──────────────────┬───────────────────┬────────────────────┐     │
│  │  POST /predict   │ POST /predict/    │   POST /explain    │     │
│  │                  │      batch        │                    │     │
│  └────────┬─────────┴─────────┬─────────┴──────────┬─────────┘     │
│           │                   │                    │               │
│           ▼                   ▼                    ▼               │
│  [FraudPredictor]    [FraudPredictor]      [AIInvestigator]        │
│   predict.py          predict.py           agent.py               │
│   • preprocess()      • predict_batch()    • LangChain ReAct       │
│   • khop_subgraph()                        • Neo4jTxnTool          │
│   • GNN forward                            • Gemini 1.5 Pro        │
│           │                                        │               │
│           ▼                                        ▼               │
│  [MLflow Model]                           [Neo4j DB]               │
│  fraud-detection-model/Production          User + Txn context      │
│                                                    │               │
│                                                    ▼               │
│                                           [Gemini API]             │
│                                        temp=0.3, max_tokens=1000   │
└─────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MLOPS / OBSERVABILITY                          │
│  [MLflow]              [HealthMonitor]         [MetricsCollector]  │
│  Experiment tracking    health_monitoring.py    metrics_system.py  │
│  Model registry         Real-time status        monitoring/data/    │
│  Artifact storage                               system_metrics.json│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. PIPELINE STEPS (trigger → output)

**Data Pipeline:**
1. `scripts/download_dataset.py` or `GraphConstructor.load_raw_data(use_kagglehub=True)` — downloads PaySim CSV via KaggleHub
2. `validate_data_schema()` — checks 11 required columns; raises `ValueError` if missing
3. `preprocess_data()` — filters invalid amounts, adds `amount_log`, `balance_change_orig/dest`, `hour_of_day`, `day_of_month`, `type_encoded`
4. `create_nodes_dataframe()` — aggregates 14 per-user stats; filters users with < 5 transactions; normalizes amounts
5. `create_edges_dataframe()` — filters dangling edges; adds edge `weight = amount / max_amount` (clipped to 0.1 min)
6. `save_processed_data()` — writes `graph_nodes.csv` + `graph_edges.csv`
7. `ingest_to_neo4j()` — connects, creates constraints+indexes, batch-UNWIND nodes (1000/batch), batch-UNWIND edges (500/batch)

**Training Pipeline:**
8. `GraphDataLoader.create_dgl_graph()` — maps user IDs to indices; stores features in `g.ndata['feat']`, `g.edata['feat']`
9. `split_graph()` — creates boolean masks (70/15/15) using `torch.randperm(42)`
10. `FraudDetectionTrainer.setup_training()` — Adam, ReduceLROnPlateau, BCEWithLogitsLoss with `pos_weight`, EarlyStopping
11. `train()` — MLflow run context; loop: `train_epoch()` → `evaluate()` → scheduler step → early stopping check → checkpoint every 10 epochs
12. `_log_final_model()` — `mlflow.pytorch.log_model()` → registered as `fraud-detection-model` under `Production` stage

**Inference Pipeline:**
13. FastAPI startup `lifespan()` — calls `load_production_model()` → loads from `models:/fraud-detection-model/Production`
14. `POST /predict` request arrives → Pydantic `TransactionInput` validates + `root_validator` checks balance consistency
15. `validate_transaction_input()` — checks required fields + valid `type`
16. `preprocess_transaction()` — adds `amount_log`, `hour_of_day`, `day_of_month`, `type_encoded`
17. `create_subgraph_for_transaction()` — `khop_subgraph(full_graph, [sender_idx, receiver_idx], 2)`
18. `model.forward()` — GraphSAGE message passing → sigmoid → fraud probability
19. `_get_risk_level()` — bucketed into LOW/MEDIUM/HIGH/CRITICAL
20. `BackgroundTasks.add_task(log_prediction_async)` — async logging without blocking response
21. Return `APIResponse` with `X-Request-ID` and `X-Process-Time` headers

---

## 5. SOURCE FILE MAP

| File | One-Line Purpose |
|---|---|
| `src/config.py` | Single `Config` class; all constants loaded from env vars via `os.getenv()` |
| `src/api/main.py` | FastAPI app; lifespan startup/shutdown; all endpoints; middleware |
| `src/api/schemas.py` | Pydantic request/response models with validators; enums; error schemas |
| `src/gnn_model/model.py` | `GraphSAGEClassifier` (nn.Module) + `GraphDataLoader`; DGL graph construction |
| `src/gnn_model/training.py` | `FraudDetectionTrainer` + `EarlyStopping` + `MetricsCalculator`; MLflow integration |
| `src/gnn_model/predict.py` | `FraudPredictor`; loads from MLflow; subgraph inference; `load_production_model()` factory |
| `src/data_processing/graph_constructor.py` | `GraphConstructor`; KaggleHub download; Pandas pipeline; Neo4j UNWIND ingestion |
| `src/explainability/agent.py` | `AIInvestigator` + `Neo4jTransactionTool`; LangChain ReAct agent; Gemini LLM |
| `src/threat_discovery/research_agent.py` | `ThreatDiscoveryAgent`; proactive fraud research; `WebResearchTool` |
| `src/health_monitoring.py` | `HealthMonitor`; rolling average response time; saves to `monitoring/data/` |
| `src/metrics_system.py` | `MetricsCollector`; prediction/fraud/error counters; JSON file persistence |
| `src/working_explanations.py` | `WorkingExplanationSystem`; mock Neo4j + direct Gemini API fallback path |
| `src/mock_neo4j.py` | Empty — placeholder for Neo4j mock in tests |
| `scripts/download_dataset.py` | Interactive CLI for dataset download + optional graph processing + Neo4j ingest |
| `.github/workflows/ci.yml` | 8-job GitHub Actions pipeline; test→security→build→integration→staging→prod→release→cleanup |
| `Dockerfile` | 3-stage build: `dependencies-builder` → `production` → `development` |
| `requirements.txt` | All pinned dependencies |
| `.env.example` | Template for all required environment variables |

---

## 6. DATABASE SCHEMA (Neo4j)

```
┌─────────────────────────────────────────────────────────────────┐
│                        NODE: User                               │
│                                                                 │
│  user_id (UNIQUE INDEX)          total_transactions            │
│  transactions_as_originator      transactions_as_destination   │
│  total_amount_sent               total_amount_received         │
│  avg_amount_sent                 avg_amount_received           │
│  net_amount                      fraud_transactions            │
│  fraud_rate (INDEX)              is_active_sender              │
│  is_active_receiver                                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
              (source:User)─────┤ RELATIONSHIP: TRANSACTION
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                   RELATIONSHIP: TRANSACTION                     │
│                                                                 │
│  transaction_id                  step                           │
│  type (INDEX)                    amount (INDEX)                 │
│  amount_log                      balance_change_orig            │
│  balance_change_dest             hour_of_day                    │
│  day_of_month                    weight                         │
│  is_fraud (INDEX)                is_flagged_fraud               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
              (target:User)─────┘

Constraints:
  CREATE CONSTRAINT user_id_unique FOR (u:User) REQUIRE u.user_id IS UNIQUE
  CREATE INDEX user_fraud_rate FOR (u:User) ON (u.fraud_rate)
  CREATE INDEX transaction_amount FOR ()-[t:TRANSACTION]-() ON (t.amount)
  CREATE INDEX transaction_fraud  FOR ()-[t:TRANSACTION]-() ON (t.is_fraud)
  CREATE INDEX transaction_type   FOR ()-[t:TRANSACTION]-() ON (t.type)
```

---

## 7. EXTERNAL API/SERVICE CONFIGURATIONS

| Service | Auth Method | Key Config | Connection Test |
|---|---|---|---|
| **Google Gemini** | API key in header | `GEMINI_API_KEY` env var; `temperature=0.3`; `max_output_tokens=1000`; model=`gemini-1.5-pro-latest` | `genai.configure(api_key=...)` at init |
| **Neo4j** | Username + password | `NEO4J_URI=bolt://localhost:7687`; `NEO4J_USERNAME=neo4j`; pool=50; lifetime=3600s | `session.run("RETURN 1").single()` |
| **MLflow** | None (local) / URI (remote) | `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`; experiment=`fraud-detection-gnn`; model=`fraud-detection-model` | Implicit on `set_experiment()` |
| **Kaggle/KaggleHub** | `~/.kaggle/kaggle.json` or env vars | Dataset: `mtalaltariq/paysim-data` | Implicit on `dataset_download()` |
| **GitHub Container Registry** | `GITHUB_TOKEN` secret | Image: `ghcr.io/{repo}:{tag}`; multi-platform `linux/amd64,linux/arm64` | `docker/login-action@v3` |
| **Codecov** | `CODECOV_TOKEN` (implicit) | Coverage from `coverage.xml` | `codecov/codecov-action@v3` |

---

## 8. KEY CONFIGURATION FLAGS

| Flag | Location | Default | What It Controls |
|---|---|---|---|
| `IS_DEVELOPMENT` | `config.py` | `True` | Bypasses production credential checks in `validate_config()` |
| `API_DEBUG` | `config.py` + env | `false` | Enables uvicorn `--reload` hot-reloading |
| `DEBUG_MODE` | `config.py` + env | `false` | Verbose debugging output |
| `VERBOSE_LOGGING` | `config.py` + env | `false` | Extra log output throughout pipeline |
| `ENABLE_PROFILING` | `config.py` + env | `false` | Application-level performance profiling |
| `LOG_LEVEL` | `config.py` + env | `INFO` | Python logging level across all modules |
| `ENCRYPT_DATABASE_CREDENTIALS` | `config.py` | `True` | Signals credential encryption intent |
| `use_subgraph` | `PredictionConfig` schema | `True` | Toggle between full-GNN vs. fallback heuristic inference |
| `subgraph_hops` | `PredictionConfig` schema | `2` | Hops for `khop_subgraph()` — trades accuracy vs. latency |
| `include_confidence` | `PredictionConfig` schema | `True` | Whether confidence score is returned |
| `include_explanation_features` | `PredictionConfig` schema | `False` | Whether to return `forward_with_attention()` outputs |
| `explanation_depth` | `ExplanationRequest` schema | `standard` | `basic` / `standard` / `detailed` — controls LLM prompt depth |
| `clear_existing` | `ingest_to_neo4j()` | `False` | Runs `MATCH (n) DETACH DELETE n` before ingestion |
| `use_kagglehub` | `load_raw_data()` | `True` | Auto-download from Kaggle vs. local CSV fallback |

---

## 9. KEY DESIGN DECISIONS TO JUSTIFY

| Decision | What You Say |
|---|---|
| **GraphSAGE over XGBoost/tabular** | "Fraud is relational — a fraudster leaves traces across a network. XGBoost can't model 2-hop neighbors. GraphSAGE encodes each user's full neighborhood context into its embedding, catching smurfing patterns that tabular models miss." |
| **GraphSAGE over GCN/GAT** | "GraphSAGE is inductive — it learns aggregation functions, not node-specific embeddings. This means it can score users it never saw in training. In production, new account IDs appear constantly, so transductive models like GCN are a non-starter." |
| **Node-level vs edge-level classification** | "I classify users (nodes) by their fraud_rate, not individual transactions. This leverages the user's full history via neighborhood aggregation. The trade-off is that I predict fraud propensity, not the exact fraudulent transaction." |
| **Mean aggregator over max/LSTM** | "Mean gives a stable average signal from the neighborhood — right for capturing *typical* behavior of a user's network. Max would amplify noise from outlier neighbors. LSTM would add sequential order assumptions that don't hold for unordered neighborhoods." |
| **BCEWithLogitsLoss with pos_weight** | "The dataset is 0.13% fraud — always predicting 'not fraud' gives 99.87% accuracy. `pos_weight` up-weights the fraud class loss, forcing the model to care about the rare positive case. I monitor F1, not accuracy, for the same reason." |
| **ReduceLROnPlateau on val_f1_score** | "I monitor F1, not loss, because loss can keep decreasing even as recall collapses. F1 is the direct signal of operational model quality." |
| **DGL over PyTorch Geometric** | "DGL's SAGEConv supports edge_weight directly, and khop_subgraph() is stable and battle-tested for subgraph inference. Both work, but DGL was better documented for hetero-graph operations when I built this." |
| **FastAPI over Flask/Django** | "Three reasons: native async for the /explain endpoint, auto OpenAPI from Pydantic schemas, and clean Depends() injection for model/agent lifecycle management with automatic 503 on unavailability." |
| **LangChain ReAct agent** | "The ReAct pattern (Reason+Act) lets the LLM decide which Neo4j queries to run based on the transaction context, rather than hardcoding the query sequence. max_iterations=3 prevents runaway latency." |
| **temperature=0.3 for Gemini** | "Fraud explanations must be consistent and analytical, not creative. 0.3 keeps outputs deterministic enough for compliance use while still allowing natural language generation." |
| **Multi-stage Docker** | "The builder stage needs gcc/g++ to compile PyTorch/NumPy extensions. The production stage only needs the compiled venv — no build tools in the final image means smaller attack surface and smaller image." |
| **Full-graph vs mini-batch training** | "PaySim fits in memory, so full-graph training is simpler and has more stable gradients. Mini-batch (GraphSAINT, NeighborSampler) would be needed at 100M+ nodes." |
| **Pydantic root_validator for balance** | "Balance consistency (old - amount = new) is a cross-field invariant that can't be checked with field-level validators alone. The root_validator fires after all fields are populated, catching fraudulent or malformed inputs at the API boundary." |
| **BackgroundTasks for logging** | "Prediction logging should never add latency to the fraud decision. BackgroundTasks runs after the response is returned without blocking the event loop." |
| **Async explainability via asyncio.to_thread** | "The LangChain agent's .run() is synchronous/blocking but the FastAPI endpoint is async. asyncio.to_thread() runs the blocking call in a thread pool without tying up the event loop, so the server stays responsive to /predict requests during explanation generation." |

---

## 10. RESUME BULLETS → TALKING POINTS

### Bullet 1: "Built end-to-end graph-based fraud detection using GraphSAGE, achieving 90.6% F1-score and 96.3% ROC-AUC"

**Say this:** "I modeled 6.36 million PaySim transactions as a directed graph — users as nodes, transactions as edges. The core model is a 2-layer GraphSAGEClassifier in `src/gnn_model/model.py` with dimensions 10→128→64→32→1, using mean aggregation with batch normalization. I chose GraphSAGE specifically because it's inductive — it can score new users that weren't in the training graph. I handled the severe class imbalance (0.13% fraud rate) with `pos_weight` in `BCEWithLogitsLoss` and monitored F1-score for early stopping rather than accuracy. The result was 90.6% F1 and 96.3% ROC-AUC, beating the 86% F1 industry benchmark."

---

### Bullet 2: "Engineered large-scale transaction graphs with Pandas/NumPy and persisted relational structures in Neo4j"

**Say this:** "The `GraphConstructor` class in `src/data_processing/graph_constructor.py` runs a 5-step pipeline: download via KaggleHub, schema validation against 11 required columns, feature engineering (amount_log, balance deltas, time encoding), aggregate user nodes with 14 features each, and build edges with amount-normalized weights. The result gets ingested into Neo4j using batched `UNWIND` Cypher — 1000 nodes per batch, 500 edges per batch. Neo4j was chosen because the explanation engine needs neighborhood queries: finding 2-hop neighbors and calculating clustering coefficients are native Cypher operations that would require expensive self-joins in SQL."

---

### Bullet 3: "Integrated AI explanation layer using LangChain and Gemini, reducing false-positive investigation cycles"

**Say this:** "The `AIInvestigator` class in `src/explainability/agent.py` uses a LangChain `CONVERSATIONAL_REACT_DESCRIPTION` agent backed by `gemini-1.5-pro-latest` at temperature 0.3. The agent has one custom tool — `Neo4jTransactionTool` — that queries Neo4j for user profiles, recent transactions, network neighbors, degree centrality, and clustering coefficient. When the `/explain` endpoint is called, the agent reasons about what to look up, calls the tool, then synthesizes a human-readable explanation with key risk factors and a recommendation. The async call via `asyncio.to_thread()` keeps the explanation latency (~2.5s) from blocking the prediction endpoint (~150ms)."

---

## 11. THREE 30-SECOND EXPLANATIONS FOR TRICKY PARTS

### 1. "Why GraphSAGE specifically?" (30 seconds)
> "GraphSAGE is inductive — it learns *how* to aggregate neighborhood features, not *what* each node's embedding is. That means it works on nodes it's never seen. For fraud detection, new accounts appear in production every day. Transductive models like GCN learn one fixed embedding per training node and can't score new ones without retraining. GraphSAGE's aggregation function transfers: give it any node with features and neighbors, and it computes a meaningful embedding. That's why it was the right choice here."

### 2. "How does the LangChain agent actually work?" (30 seconds)
> "It's a ReAct loop — Reason + Act. The LLM receives an investigation prompt telling it to act as a fraud analyst for transaction X. It reasons: 'I need the user's transaction history — I'll call get_transaction_context.' The tool queries Neo4j and returns JSON. The LLM observes the result, reasons again — 'the user has a 15% fraud rate, that's suspicious' — and either calls the tool again or generates a final explanation. I cap iterations at 3 to bound latency. The whole thing runs in a thread pool via asyncio.to_thread() so the event loop stays free."

### 3. "What happens when Neo4j or Gemini goes down?" (30 seconds)
> "The system degrades gracefully at every level. If Neo4j is down at startup, `_initialize_neo4j()` catches `ServiceUnavailable`, sets `neo4j_driver = None`, and the agent never initializes. The `/predict` endpoint is completely unaffected — it never touches Neo4j. The `/explain` endpoint returns HTTP 503 via the `get_ai_investigator()` dependency. If Gemini fails during an explanation, `explain_transaction()` catches the exception and calls `_create_fallback_explanation()`, which returns a rule-based explanation with `agent_used: False` and `confidence: 0.6` in the response so downstream consumers know it's degraded output. Prediction throughput is never impacted."

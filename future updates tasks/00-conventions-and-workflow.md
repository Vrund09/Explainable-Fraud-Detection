# 00 — Conventions & Workflow (READ FIRST)

This document is **mandatory reading** before executing any spec. It defines how you (the implementer) must work so that an automated or less-experienced agent produces correct, reviewable, regression-free changes.

---

## 1. The spec-driven loop (follow for every numbered spec)

For each spec, execute these steps **in order**:

1. **Read the whole spec** before touching code. Note its `Prerequisites` — if any prerequisite spec is not done, stop and do that one first.
2. **Restate the Acceptance Criteria** in `future updates tasks/PROGRESS.md` as a checklist you will tick off.
3. **Create a branch** (see §3).
4. **Write or update tests first** where the spec's Test Plan allows it (red state). For changes that are hard to test-first, write the test immediately after the minimal implementation.
5. **Implement the smallest change** that satisfies the spec. Match existing code style. Keep imports at the top of files.
6. **Run the Validation commands** in the spec. Capture output.
7. **Tick the Acceptance Criteria.** If any criterion cannot be met, **stop** and record the blocker in `PROGRESS.md` (do not hack around it).
8. **Commit** (see §3) and update `PROGRESS.md` marking the spec done.

> Golden rule: **never weaken or delete an existing passing test to make the build green.** Fix the root cause.

---

## 2. Environment setup (one-time)

This repo is checked out in a worktree where dependencies are **not** installed. Set up both a Python and a Node environment.

### 2.1 Python (3.11)

```bash
# From the repo root
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov httpx black flake8 isort mypy pre-commit
```

> Note: `dgl` and `torch` wheels are platform-specific. If `pip install -r requirements.txt` fails on `dgl`/`torch`, follow spec `03` which pins working versions and documents the correct index URLs.

### 2.2 Node (for the frontend, spec `12`)

```bash
# Install Node 20 LTS (nvm recommended). Verify:
node --version   # expect v20.x
npm --version
```

### 2.3 Environment variables

Copy `.env.example` to `.env` and fill values. Specs document any **new** variables they introduce; when they do, you must also add them to `.env.example`.

```bash
cp .env.example .env
```

---

## 3. Branching & commits

- One branch per spec: `feat/spec-<NN>-<short-slug>` (e.g., `feat/spec-06-real-gnn-inference`).
- Conventional commit messages:
  - `feat(spec-06): wire real GNN subgraph inference into /predict`
  - `fix(spec-10): import os in health_monitoring`
  - `test(spec-11): add API contract tests for /predict`
  - `chore(spec-02): add pre-commit + tooling config`
- Keep each commit scoped to the spec. Do not bundle unrelated changes.

---

## 4. Testing philosophy

- **Unit tests** isolate one function/class; mock external systems (Neo4j, Gemini, MLflow, network).
- **Integration tests** exercise real wiring (FastAPI app via `httpx.AsyncClient`, Neo4j via a test container or the docker-compose service).
- **E2E tests** (frontend) drive the UI via Playwright against a running API (or a mocked API per spec `12`).
- Tests must be **deterministic**: no live LLM calls in CI. Use fakes/stubs. Mark any test needing secrets with `@pytest.mark.requires_secrets` and skip by default.
- Target coverage gate: **70% line coverage on `src/`** by the end of spec `11` (raise later). Never lower an existing gate.

### Test layout (created across specs)

```
tests/
├── unit/
│   ├── test_predict.py
│   ├── test_model.py
│   ├── test_schemas.py
│   ├── test_explainability_agent.py
│   ├── test_threat_agent.py
│   └── test_monitoring.py
├── integration/
│   ├── test_api_predict.py
│   ├── test_api_explain.py
│   └── test_neo4j_tool.py
├── eval/
│   └── test_agent_eval.py          # spec 09
├── conftest.py                     # shared fixtures (app client, fake LLM, fake graph)
└── fixtures/                       # small sample graphs, sample transactions, golden cases
```

---

## 5. Validation conventions

Every spec lists copy-pasteable commands. Standard ones referenced by specs:

```bash
# Format & lint (must pass on changed files)
black src/ tests/
isort src/ tests/
flake8 src/ tests/ --max-line-length=127
mypy src/ --ignore-missing-imports

# Run tests
pytest -q
pytest tests/unit -q
pytest --cov=src --cov-report=term-missing

# Run the API locally
python -m uvicorn src.api.main:app --reload --port 8000
# then in another shell:
curl -s http://localhost:8000/health | python -m json.tool
```

If a command cannot run in your environment (e.g., no GPU, no Neo4j), the spec provides an offline/mocked alternative. Record which path you used in `PROGRESS.md`.

---

## 6. Code-change guardrails

- **Minimal diffs.** Do not reformat unrelated code or rename things the spec did not ask you to.
- **No new comments/docstrings churn** unless the spec asks for it; keep the existing documentation style.
- **Imports at top of file**, grouped stdlib / third-party / local, isort-compatible.
- **Backwards compatibility:** when changing a public function signature used elsewhere, update all call sites in the same commit (grep for usages first).
- **Secrets:** never hard-code API keys; read from env via `src/config.py`. Never commit `.env`.
- **Determinism in ML code:** set seeds where the spec requires reproducibility (`torch.manual_seed`, `numpy.random.seed`).

---

## 7. When you get stuck

If you cannot satisfy an acceptance criterion after a reasonable attempt:

1. Write the exact blocker, the commands you ran, and the error output into `PROGRESS.md`.
2. Leave the code in a compiling, test-passing state (revert the partial change if needed).
3. Surface the blocker to the human. **Do not** invent fake data, stub out assertions, or relax acceptance criteria to "pass".

---

## 8. PROGRESS.md template

Create `future updates tasks/PROGRESS.md` on first use:

```markdown
# Implementation Progress

## Spec 02 — Foundation & test bootstrap
- Status: in_progress | done | blocked
- Branch: feat/spec-02-tooling
- Acceptance:
  - [ ] criterion 1
  - [ ] criterion 2
- Validation output:
  - (paste key command output)
- Notes / blockers:
```

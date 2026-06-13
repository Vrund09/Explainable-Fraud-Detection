# 13 — Deployment & Full-Stack Compose

| Field | Value |
| --- | --- |
| Spec ID | 13 |
| Status | Not started |
| Depends on | 06 (real API), 07 (Neo4j), 11 (green CI), 12 (frontend) |
| Blocks | — |
| Est. effort | 2–3 days |
| Risk | Medium |

## Objective
Make the whole system runnable with **one command** locally (API + Neo4j + MLflow + frontend) and deployable to the cloud, with sane secrets/env management. Update the Docker build for the modernized stack and document deploy targets.

## Background & current state
- `Dockerfile` is a 3-stage build (`dependencies-builder` → `production` → `development`) for the API only (`Dockerfile:1-134`). It runs `pip install --no-deps` then full install (`:33-35`).
- CI builds the `production` target and runs an integration container that curls `/health` and `/model/status` (`.github/workflows/ci.yml:192-287`).
- No compose file; Neo4j only exists as a CI service. Frontend has no container.
- `.env.example` covers Neo4j, Gemini, MLflow, API, logging, secret key.

## Prerequisites
- Specs `06`, `07`, `12` produce the components to orchestrate; spec `11` green CI.

## Out of scope
- Kubernetes manifests (note as future work); managed-cloud specifics beyond one documented target.

## Implementation steps

### Step 1 — Update the API Dockerfile for the modern stack
- Ensure torch/dgl install correctly in-container (CPU wheels + dgl index from spec `03` Step 2). Replace the `--no-deps` then full-install hack (`Dockerfile:33-35`) with a clean, ordered install:
```dockerfile
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch==2.3.* --index-url https://download.pytorch.org/whl/cpu
RUN pip install dgl==2.2.* -f https://data.dgl.ai/wheels/torch-2.3/repo.html
RUN pip install -r requirements.txt
```
- Copy the production model artifacts into the image **or** mount them (prefer a mounted volume / download step to keep the image small; document choice). Ensure `models/production/` is available at runtime for spec `06`.

### Step 2 — Frontend Dockerfile
Add `frontend/Dockerfile` (multi-stage): build with Node 20, serve static assets with `nginx:alpine`. Inject `VITE_API_BASE_URL` at build time. Add an nginx config that serves the SPA and falls back to `index.html` for client routes.

### Step 3 — Full-stack `docker-compose.yml`
Create a root `docker-compose.yml` orchestrating:
- `neo4j` (image `neo4j:5.23`, APOC, volume, healthcheck) — from spec `07`.
- `mlflow` (optional: `ghcr.io/mlflow/mlflow` or a small custom image) serving the tracking UI on `:5000`, backed by sqlite/volume.
- `api` (build `.`, target `production`), depends_on neo4j (healthy), env from `.env`, mounts `models/production/`, exposes `:8000`.
- `frontend` (build `frontend/`), depends_on api, exposes `:5173`/`:80`.
Provide healthchecks and a shared network. Document `docker compose up` as the canonical local run.

### Step 4 — Seed step
Document/automate seeding Neo4j after first `up` (run `scripts/seed_neo4j.py` from the api container or a one-shot `seed` service). Provide a `make seed` target.

### Step 5 — Secrets & env management
- Single root `.env` (gitignored) consumed by compose; `.env.example` updated with **all** variables introduced across specs: `GEMINI_API_KEY`, `GEMINI_MODEL_NAME`, `NEO4J_*`, `TAVILY_API_KEY`, `THREAT_RESEARCH_MODE`, `CORS_ALLOW_ORIGINS`, `API_REQUIRE_AUTH`, `API_SECRET_KEY`, `MLFLOW_TRACKING_URI`, `VITE_API_BASE_URL`, optional `LANGCHAIN_*`.
- Document that production must override `API_SECRET_KEY` and set `API_REQUIRE_AUTH=true`, real `NEO4J_PASSWORD`, and a restrictive `CORS_ALLOW_ORIGINS`.

### Step 6 — Cloud deploy targets (document + minimal config)
- **Frontend → Vercel** (or Netlify/GitHub Pages): build `frontend/`, set `VITE_API_BASE_URL` to the deployed API URL. Add a `vercel.json` SPA rewrite if using Vercel.
- **API → Render/Fly.io/Railway**: containerized; set env vars; attach a managed Neo4j (Aura) or a Neo4j container. Document the chosen target with a step-by-step in `docs/deployment.md`.
- Update CI deploy jobs (`ci.yml:292-327`) from placeholder echoes to the chosen target (or clearly mark them as manual/optional if no cloud account is available).

### Step 7 — Smoke test the stack
Add `scripts/smoke_test.sh` that, after `docker compose up`, waits for health and asserts: `/health` 200, `/model/status` shows `graph_loaded`, a sample `/predict` returns `scoring_path:"gnn"`, and the frontend serves `200` at `/`.

## Contract / data changes
- New `docker-compose.yml`, `frontend/Dockerfile`, nginx config, `scripts/smoke_test.sh`, `docs/deployment.md`. Updated `.env.example` and API `Dockerfile`.

## Acceptance criteria
- [ ] `docker compose up -d` brings up neo4j + mlflow + api + frontend; all healthchecks pass.
- [ ] After `make seed`, `/predict` for a seeded user returns `scoring_path:"gnn"`.
- [ ] Frontend container serves the SPA and talks to the API (CORS allows its origin).
- [ ] `.env.example` lists every variable used across all specs; no secret is hard-coded in compose.
- [ ] `scripts/smoke_test.sh` exits 0 against the running stack.
- [ ] CI builds both images (API + frontend); deploy jobs are either wired to a real target or explicitly marked manual.
- [ ] `docs/deployment.md` documents local + one cloud target end-to-end.

## Test plan
- `scripts/smoke_test.sh` is the integration check (run in CI's integration job, replacing the curl-only checks at `ci.yml:259-282`).
- Add a CI job step building `frontend/Dockerfile`.

## Validation
```bash
cp .env.example .env   # fill secrets
docker compose build
docker compose up -d
make seed
bash scripts/smoke_test.sh
docker compose down
```

## Rollback / fallback
Compose is additive. If torch/dgl in-container is problematic on the build host, document a prebuilt base image or CPU-only constraints. If no cloud account is available, keep deploy jobs as documented manual steps rather than failing CI.

## Definition of Done
One-command local stack + documented cloud deploy + updated images and env; smoke test green; commit on `feat/spec-13-deploy`.

## References
- `Dockerfile:1-134`, `.github/workflows/ci.yml:156-327`, `.env.example`, `scripts/seed_neo4j.py` (07), `frontend/` (12), `models/production/` (05/06).

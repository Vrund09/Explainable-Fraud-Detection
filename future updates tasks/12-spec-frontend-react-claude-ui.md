# 12 — Frontend: React SPA with Claude-like Minimal UI

| Field | Value |
| --- | --- |
| Spec ID | 12 |
| Status | Not started |
| Depends on | 04 (stable API contract + `frontend/openapi.json`); ideally 06 (real predictions), 07 (explanations), 08 (`/threats`) |
| Blocks | 13 (deploy) |
| Est. effort | 5–8 days |
| Risk | Medium |

## Objective
Replace the mocked single-file demo (G3) with a **modern React SPA** wired to the live FastAPI, presented in a **Claude-like minimal aesthetic** — warm, calm, content-first, generous whitespace — that is clearly *this project's* identity (not a Claude clone). Preserve the strong ideas from the old demo (network graph, risk distribution, transaction log, scenario presets) but implement them cleanly against real endpoints with proper loading/empty/error states.

## Background & current state
- `fraud_detection_demo.html` (1134 lines) is a self-contained dark-neon mock: `analyzeTransaction()` computes risk client-side with `Math.random()` (`:984-1041`); uses `vis-network` for a graph and `chart.js` for a risk doughnut; has scenario presets (`:1115-1130`) and a transaction log. **Not connected to the API.**
- API endpoints available after specs `04`/`06`/`07`/`08`/`10`: `GET /health`, `GET /model/status`, `POST /predict`, `POST /predict/batch`, `POST /explain`, `GET /predictions/history`, `GET /metrics/summary`, optional `GET /threats`.

## Prerequisites
- Node 20 (`00-conventions §2.2`).
- `frontend/openapi.json` exported (spec `04` Step 6). Re-export whenever the contract changes.

## Out of scope
- Backend changes (owned by `04`/`06`/`07`/`08`/`10`).

## Tech stack
- **Vite + React 18 + TypeScript**.
- **Tailwind CSS** + **shadcn/ui** (Radix primitives) for accessible components.
- **TanStack Query** for server state (caching, loading/error states).
- **openapi-typescript** + a thin typed fetch client generated from `frontend/openapi.json`.
- **Recharts** for charts (risk distribution, latency, metrics) — replaces Chart.js.
- **React Flow** (`@xyflow/react`) for the transaction network graph — replaces vis-network; better React integration and styling control.
- **lucide-react** icons. **Vitest + React Testing Library** (unit) and **Playwright** (E2E).

## Design system — "calm analyst" (Claude-inspired, project-specific)
Define tokens in `frontend/src/index.css` / Tailwind config. Intentionally **not** Claude's coral; use a restrained, trust-oriented palette.

**Light theme (default):**
- `--bg`: `#FAF9F5` (warm paper) — page background.
- `--surface`: `#FFFFFF` cards; `--surface-2`: `#F3F1EC` subtle panels.
- `--text`: `#1F1D1A` (near-black, warm); `--text-muted`: `#6B6862`.
- `--border`: `#E7E3DA` (1px hairlines, low contrast).
- `--accent`: `#3B5BDB` (calm indigo) for primary actions/links; used **sparingly**.
- **Risk scale (accessible, calm — not neon):** LOW `#3A7D44` (green), MEDIUM `#B58A1B` (amber), HIGH `#C2410C` (burnt orange), CRITICAL `#A01B2E` (deep red). All meet WCAG AA on `--surface`.
- **Dim theme (optional):** warm charcoal `#1C1B19` bg, `#262420` surfaces, `#ECE9E2` text — same accent/risk hues tuned for contrast.

**Typography:**
- Display/headings: a refined serif or humanist sans (e.g., **"Source Serif 4"** or **"Tiempos"-like**) for a calm editorial feel; body/UI: **Inter** (or **Geist**). Generous sizes; comfortable line-height (1.6 body).
- Monospace (**JetBrains Mono**) only for ids/amounts/code.

**Layout & motion:**
- Generous whitespace; max content width ~1100px; 8px spacing scale.
- `rounded-lg` (8–12px) corners; **soft** shadows (`0 1px 2px rgba(0,0,0,.04)`), no glow.
- 1px hairline borders, not heavy boxes. Restrained motion (150–200ms ease), respect `prefers-reduced-motion`.
- Empty states are friendly and instructive; never a blank panel.

> The goal: looks considered, editorial, and trustworthy — the opposite of the current dark-neon dashboard. Reviewers should feel "this is a calm, premium analyst tool."

## Information architecture (pages)
Single-page app with a slim sidebar. Routes:
1. **Overview** (`/`) — system status (`/health`, `/model/status`), headline metrics (`/metrics/summary`), risk distribution (Recharts), recent activity (`/predictions/history`).
2. **Analyze** (`/analyze`) — the core flow: a transaction form (sender, receiver, amount, type, optional step) → `POST /predict` → result card (probability, risk level, `scoring_path` badge so users see GNN vs cold-start, confidence) → "Explain" button → `POST /explain` → structured explanation panel (summary, key factors, evidence, recommendation). Include **scenario presets** (legit / suspicious / fraud) like the old demo, but they call the real API.
3. **Network** (`/network`) — React Flow graph of the transaction's neighborhood (sender, receiver, neighbors) colored by risk; node click shows details. Sourced from prediction/explanation context (or a dedicated context field if added).
4. **Batch** (`/batch`) — upload/paste multiple transactions → `POST /predict/batch` → table with sortable risk, summary stats.
5. **Threats** (`/threats`) — list from `GET /threats` (spec `08`); each threat card shows name, risk, techniques, indicators, sources. Clearly labeled "advisory intelligence".

## Implementation steps

### Step 1 — Scaffold
```bash
npm create vite@latest frontend -- --template react-ts
cd frontend && npm install
npm install -D tailwindcss postcss autoprefixer && npx tailwindcss init -p
npm install @tanstack/react-query recharts @xyflow/react lucide-react clsx
npx shadcn@latest init          # choose the neutral base; we override tokens
npm install -D openapi-typescript vitest @testing-library/react @testing-library/jest-dom jsdom @playwright/test
```
Configure Tailwind `content` globs and the design tokens above (CSS variables + Tailwind theme extension).

### Step 2 — Generate the typed API client
```bash
npx openapi-typescript ../frontend/openapi.json -o src/api/schema.d.ts   # path: the exported file
```
Create `src/api/client.ts`: a thin `fetch` wrapper that reads `VITE_API_BASE_URL` (default `http://localhost:8000`), sets `X-API-Key` from `VITE_API_KEY` when present, and exposes typed functions: `getHealth`, `getModelStatus`, `predict`, `predictBatch`, `explain`, `getHistory`, `getMetricsSummary`, `getThreats`. Wrap calls in TanStack Query hooks (`usePredict`, etc.).

### Step 3 — App shell & theming
- `AppLayout` with sidebar nav (lucide icons), header with system-status dot (from `/health`), light/dim theme toggle persisted to `localStorage`.
- Global `QueryClientProvider`, error boundary, toast (shadcn `sonner`).

### Step 4 — Build the pages
Implement Overview → Analyze → Network → Batch → Threats per the IA above. For each data fetch use TanStack Query with explicit **loading skeletons**, **empty states**, and **error states** (retry button, friendly message). Format amounts with `Intl.NumberFormat`; ids in mono.

### Step 5 — Result & explanation components
- `RiskBadge` (risk scale colors), `ScoringPathBadge` ("GNN" vs "Cold-start"), `ProbabilityMeter`, `ConfidenceBar`.
- `ExplanationPanel` rendering the structured `FraudExplanation` (summary, key_factors, evidence list, recommendation pill). Show a "degraded" notice when the API flags it.

### Step 6 — Network graph
`TransactionGraph` using React Flow: center = transaction edge (sender→receiver), surrounding = neighbors; node color = risk, edge label = amount/type. Provide a legend and zoom/fit controls. Graceful empty state if no neighborhood data.

### Step 7 — Accessibility & responsiveness
- All interactive elements keyboard-reachable; visible focus rings (use `--accent`); shadcn/Radix gives ARIA. Color is never the only signal (add labels/icons to risk).
- Layout responsive from 360px → desktop; sidebar collapses to a top bar on mobile. Honor `prefers-reduced-motion` and `prefers-color-scheme` for initial theme.

### Step 8 — Replace the old demo
- Move `fraud_detection_demo.html` to `legacy/` (or delete) and update any links. The README "live demo" (spec `14`) points to the deployed SPA.

## Contract / data changes
- New `frontend/` app. New env: `VITE_API_BASE_URL`, optional `VITE_API_KEY`. Add `frontend/.env.example`.
- `frontend/openapi.json` is the source of truth for types; regenerate on contract changes.

## Acceptance criteria
- [ ] `npm run dev` serves the SPA; with the API running, **Analyze** performs a real `POST /predict` and renders the result including a `scoring_path` badge.
- [ ] "Explain" calls `POST /explain` and renders the structured explanation (summary, key factors, evidence, recommendation).
- [ ] Overview shows live `/health`, `/model/status`, and `/metrics/summary` data with loading/empty/error states.
- [ ] Batch performs `POST /predict/batch` and renders a results table + summary.
- [ ] Threats page renders `GET /threats` (or a clean empty state if the endpoint/data is absent).
- [ ] No `Math.random()` risk computation anywhere in `frontend/src` (`grep -rn "Math.random" frontend/src` returns nothing for risk logic).
- [ ] Light + dim themes implemented with the specified tokens; risk colors meet WCAG AA (document a contrast check).
- [ ] `npm run build` succeeds; `npm run test` (Vitest) and `npx playwright test` pass.
- [ ] Lighthouse (or axe) shows no critical a11y violations on Analyze + Overview.

## Test plan
- **Vitest + RTL:** `RiskBadge` maps probability→level/color; `ExplanationPanel` renders structured fields + degraded notice; API client maps errors to user-facing states; form validation mirrors backend constraints (amount > 0, valid type).
- **MSW (Mock Service Worker):** mock the API in unit tests so they run without a backend; assert loading→success and loading→error transitions.
- **Playwright E2E:** against a running API (or MSW), run the scenario presets and assert the result card + explanation appear; test keyboard navigation of the Analyze form; test theme toggle persistence.

## Validation
```bash
cd frontend
npm install
npm run build
npm run test            # vitest
npx playwright install --with-deps && npx playwright test
grep -rn "Math.random" src/ ; echo exit:$?     # no random risk logic
# with API up on :8000 and VITE_API_BASE_URL set:
npm run dev             # manually verify Analyze -> Explain end to end
```

## Rollback / fallback
The SPA is additive (new `frontend/` dir); the old HTML can stay in `legacy/` until the SPA is verified. If a backend endpoint (`/threats`, `/metrics/summary`) isn't ready, the corresponding page shows a documented empty state rather than failing the build.

## Definition of Done
A calm, accessible, responsive React SPA wired to the real API with light/dim themes, replacing the mock demo; unit + E2E tests pass; `npm run build` clean; commit on `feat/spec-12-frontend`.

## References
- `fraud_detection_demo.html` (ideas to preserve: graph, risk doughnut, presets, log), `frontend/openapi.json` (spec 04), API endpoints from specs `04/06/07/08/10`.
- Tailwind, shadcn/ui, TanStack Query, React Flow (`@xyflow/react`), Recharts, openapi-typescript, Playwright, MSW docs.

# 14 — Documentation & Portfolio Polish

| Field | Value |
| --- | --- |
| Spec ID | 14 |
| Status | Not started |
| Depends on | All prior specs (final pass) |
| Blocks | — |
| Est. effort | 1–2 days |
| Risk | Low |

## Objective
Make the project read as a credible, modern, **honest** portfolio piece: rewrite the README around what now actually works, add architecture diagrams and real screenshots/GIFs of the new UI, document reproducible metrics (no unverifiable claims), refresh the SSOT resume bullets, remove placeholder clutter, and provide a crisp demo script. Closes G3 (demo link), G14 (honest metrics), G15 (clutter).

## Background & current state
- `README.md` (978 lines) and `TECHNICAL_BRIEFING_README.md` (53 KB) describe the *aspirational* system, including the GNN as if used at inference and "94% F1" as achieved.
- `project_ssot.md` resume bullets state 94% F1 and several `[INSERT METRIC]` placeholders (`:23,32,33`); it also honestly notes the heuristic-fallback caveat (`:16`).
- Placeholder clutter: `GCP_GraphSAGE_Training.ipynb` (1 byte), empty `data/ docs/ notebooks/` (per worktree), empty `src/mock_neo4j.py`.
- After specs `05`–`12`, the system genuinely runs the GNN, has real explanations, a real UI, and a real `metrics.json`.

## Prerequisites
- Specs `05` (real `metrics.json`), `06` (real inference), `07`/`08` (agents), `12` (UI to screenshot), ideally `13` (live demo URL).

## Out of scope
- Code changes (docs only, plus deleting confirmed-dead placeholder files).

## Implementation steps

### Step 1 — Rewrite `README.md` around reality
Restructure to: hero (one-line value prop + live demo link + screenshot), "What it does", architecture diagram, "How it actually works" (data → graph → GNN inference → explanation), quickstart (`docker compose up` from spec `13`), reproducing metrics, project structure, roadmap, license. Crucially:
- State that `/predict` runs **GraphSAGE** for known users with a `scoring_path` field, and describe cold-start behavior honestly.
- Replace any "uses Gemini/Neo4j" hand-waving with the real, runnable steps (seed script, env vars).

### Step 2 — Honest metrics with provenance
- Pull headline numbers from `models/production/metrics.json` (spec `05`). Lead with **PR-AUC** (honest for imbalanced fraud) alongside F1, precision/recall at the chosen threshold, and the confusion matrix.
- If 94% F1 was reproduced, cite the MLflow run id + threshold; if not, **update the number** to the measured value everywhere it appears (README, TECHNICAL_BRIEFING, SSOT, `metrics_system.py` default). Never keep an unverifiable claim.
- Replace `[INSERT METRIC]` business-impact placeholders with clearly-labeled *illustrative* estimates ("hypothetical, for demonstration") or remove them.

### Step 3 — Architecture diagrams
- Update the Mermaid architecture diagram in README to show the **real** flow including the SPA, the model-artifact load path, and the agents. Add a second diagram for a single prediction's sequence (client → API → graph subgraph → GNN → explanation agent → Neo4j).

### Step 4 — Screenshots / GIF
- Capture screenshots of the new UI (Overview, Analyze + explanation, Network graph) in light theme; add a short GIF of the Analyze→Explain flow. Store under `docs/assets/`. Embed in README.

### Step 5 — Refresh the SSOT resume bullets
Update `project_ssot.md`:
- Change the architecture note (`:16`) from "falls back to heuristic" to the new reality ("serves GraphSAGE subgraph inference with an explicit cold-start path").
- Update bullets to mention: real GNN serving, structured LLM explanations grounded in Neo4j, a React analyst UI, a modern LangGraph agent stack, an evaluation harness + CI gates, and Dockerized full-stack deploy. Keep metrics honest (use measured values).
- Update `tech_stack` to add React/TypeScript/Tailwind, LangGraph, and remove anything dropped (e.g., legacy openai). Set realistic `date_end`.

### Step 6 — Reconcile `TECHNICAL_BRIEFING_README.md`
Skim for now-false statements (GNN-at-inference, 94% as fact, simulated threat research described as real). Correct them or add a clearly-marked "Status" callout per section noting what changed after the upgrade. At minimum, fix the three headline inaccuracies.

### Step 7 — Remove placeholder clutter (G15)
- Delete confirmed-dead files: `GCP_GraphSAGE_Training.ipynb` (1 byte) and `src/mock_neo4j.py` (if unused after spec `07`).
- Populate or remove empty dirs: ensure `notebooks/01-eda.ipynb` is referenced; if `data/`, `docs/` are meant to hold generated content, add a `.gitkeep` + a README note explaining what goes there.
- Move `fraud_detection_demo.html` to `legacy/` (spec `12` Step 8) and note it as the deprecated mock.

### Step 8 — Demo script
Add `docs/demo_script.md`: a 3–5 minute walkthrough (start the stack, show Overview, run the three scenario presets in Analyze, open an explanation, show the Network graph, show the Threats feed, mention CI/eval gates). This is the script for a recorded portfolio demo.

## Contract / data changes
- Docs only + deletion of confirmed-dead placeholder files. New `docs/assets/`, `docs/demo_script.md`, `docs/deployment.md` (from spec `13`).

## Acceptance criteria
- [ ] README leads with an accurate value prop, a live demo link (or "run locally" if undeployed), and a real screenshot; quickstart is `docker compose up` and works.
- [ ] Every performance number in README/TECHNICAL_BRIEFING/SSOT/`metrics_system.py` traces to `models/production/metrics.json` or an MLflow run; no `[INSERT METRIC]` left as a bare placeholder; no unverifiable "94%" if it wasn't reproduced.
- [ ] Architecture + sequence Mermaid diagrams reflect real GNN serving + SPA + agents.
- [ ] UI screenshots/GIF embedded from `docs/assets/`.
- [ ] `project_ssot.md` updated (architecture note, bullets, tech_stack) and internally consistent with the code.
- [ ] Confirmed-dead placeholder files removed; empty dirs explained or populated; old demo moved to `legacy/`.
- [ ] `docs/demo_script.md` exists.
- [ ] Markdown links are valid (run a link check).

## Test plan
- Not code-tested. Verification is review-based + a link checker:
```bash
npx markdown-link-check README.md
npx markdown-link-check project_ssot.md
```
- Confirm every metric cited exists in `models/production/metrics.json` (manual cross-check; note run id).

## Validation
```bash
test ! -f GCP_GraphSAGE_Training.ipynb && echo "placeholder removed"
grep -rn "\[INSERT METRIC\]" project_ssot.md README.md ; echo exit:$?   # expect no bare placeholders
grep -rn "94%" README.md TECHNICAL_BRIEFING_README.md project_ssot.md   # each hit must be backed by metrics.json
npx markdown-link-check README.md
```

## Rollback / fallback
Docs changes are low-risk and revertible. If a live deploy (spec `13`) isn't available, the README "demo" section links to a recorded GIF + local run instructions instead of a URL.

## Definition of Done
Honest, modern, screenshot-backed documentation consistent with the upgraded system; placeholder clutter gone; SSOT refreshed; commit on `feat/spec-14-docs-polish`.

## References
- `README.md`, `TECHNICAL_BRIEFING_README.md`, `project_ssot.md:16,23,27,29,32,33`, `models/production/metrics.json` (05), `src/metrics_system.py:24`, `fraud_detection_demo.html`, `notebooks/01-eda.ipynb`.

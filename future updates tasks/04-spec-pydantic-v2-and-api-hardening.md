# 04 — Pydantic v2 Migration & API Hardening

| Field | Value |
| --- | --- |
| Spec ID | 04 |
| Status | Not started |
| Depends on | 03 |
| Blocks | 06 (clean contract), 12 (frontend consumes the contract) |
| Est. effort | 1–2 days |
| Risk | Medium |

## Objective
Migrate the API schemas and app to idiomatic **Pydantic v2** (G10), then harden the service for real use by a browser SPA: optional **API-key auth**, a **configurable CORS allowlist** (so the React app at a known origin works without `*`), basic **rate limiting**, and an exported **OpenAPI schema** that the frontend will consume (G13).

## Background & current state
- `src/api/schemas.py:13-14` imports `validator, root_validator` and `constr, confloat, conint` from `pydantic.types`; uses `Field(..., example=...)`.
- `src/api/main.py` calls `.dict()` on response models at `:175, :190, :207`.
- CORS/TrustedHost are wide open: `src/api/main.py:111-122` (`allow_origins=["*"]`, `allowed_hosts=["*"]`).
- No auth anywhere despite `config.API_KEY_HEADER` (`config.py:182`) and `config.API_SECRET_KEY` (`config.py:183`).
- 7 endpoints: `/health`, `/model/status`, `/predict`, `/predict/batch`, `/explain`, `/predictions/history` (GET+DELETE).

## Prerequisites
- Spec `03` (pydantic 2.9 installed, app boots).

## Out of scope
- Real GNN inference (`06`); we only touch request/response models and middleware here.

## Implementation steps

### Step 1 — Migrate `schemas.py` to Pydantic v2 idioms
Apply these mechanical replacements throughout `src/api/schemas.py`:

| v1 | v2 |
| --- | --- |
| `from pydantic import validator, root_validator` | `from pydantic import field_validator, model_validator` |
| `@validator("x")` | `@field_validator("x")` + `@classmethod` |
| `@root_validator` | `@model_validator(mode="after")` |
| `constr(min_length=1, max_length=50)` | `Annotated[str, StringConstraints(min_length=1, max_length=50)]` |
| `confloat(gt=0, le=MAX)` | `Annotated[float, Field(gt=0, le=MAX)]` |
| `conint(...)` | `Annotated[int, Field(...)]` |
| `Field(..., example=x)` | `Field(..., json_schema_extra={"example": x})` (or `examples=[x]`) |
| `class Config: ...` | `model_config = ConfigDict(...)` |

Example for `TransactionInput` (`schemas.py:40-78`):
```python
from typing import Annotated, Optional
from pydantic import BaseModel, Field, StringConstraints, field_validator

class TransactionInput(BaseModel):
    transaction_id: Optional[str] = Field(None, json_schema_extra={"example": "TXN_123456789"})
    sender_id: Annotated[str, StringConstraints(min_length=1, max_length=50)] = Field(...)
    receiver_id: Annotated[str, StringConstraints(min_length=1, max_length=50)] = Field(...)
    amount: float = Field(..., gt=0, le=config.MAX_AMOUNT_THRESHOLD, json_schema_extra={"example": 150000.5})
    type: TransactionType = Field(...)
```
Convert every `@validator`/`@root_validator` in the file (read the full 487 lines and migrate each). Keep validation **semantics identical**.

### Step 2 — Replace `.dict()` with `.model_dump()`
In `src/api/main.py`, change `.dict()` → `.model_dump()` at `:175, :190, :207` (and any others found via `grep -rn "\.dict()" src/`).

### Step 3 — Configurable CORS allowlist
Add to `src/config.py` (near API config, ~`:135`):
```python
# Comma-separated list of allowed browser origins for the SPA
CORS_ALLOW_ORIGINS: list[str] = [
    o.strip() for o in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:5173,http://localhost:3000",
    ).split(",") if o.strip()
]
```
In `src/api/main.py:111-117`, replace `allow_origins=["*"]` with `allow_origins=config.CORS_ALLOW_ORIGINS`. Keep `allow_credentials=True`. Add `CORS_ALLOW_ORIGINS` to `.env.example`.
> `5173` is Vite's default dev port (spec `12`). `3000` covers alternative setups.

### Step 4 — Optional API-key auth dependency
Add `src/api/security.py`:
```python
from fastapi import Header, HTTPException, status
from ..config import config

async def require_api_key(x_api_key: str | None = Header(default=None, alias=config.API_KEY_HEADER)):
    """Enforce API key only when API_REQUIRE_AUTH is true."""
    if not config.API_REQUIRE_AUTH:
        return
    if not x_api_key or x_api_key != config.API_SECRET_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
```
Add `API_REQUIRE_AUTH: bool = os.getenv("API_REQUIRE_AUTH", "false").lower() == "true"` to `config.py`. Attach the dependency to mutating/compute endpoints (`/predict`, `/predict/batch`, `/explain`, DELETE `/predictions/history`) via `dependencies=[Depends(require_api_key)]`. Leave `/health` and `/model/status` open. Default off so the demo still works.

### Step 5 — Basic rate limiting
Add `slowapi` to `requirements.txt` (`slowapi==0.1.*`). Wire a limiter keyed by client IP:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{config.API_RATE_LIMIT}/minute"])
app.state.limiter = limiter
```
Register the exception handler and decorate `/predict` and `/predict/batch`. `config.API_RATE_LIMIT` already exists (`config.py:147`).

### Step 6 — Export OpenAPI for the frontend
Add a small script `scripts/export_openapi.py` that imports the app and writes `frontend/openapi.json` (the SPA's typed client in spec `12` is generated from this):
```python
import json
from src.api.main import app
from pathlib import Path
Path("frontend").mkdir(exist_ok=True)
Path("frontend/openapi.json").write_text(json.dumps(app.openapi(), indent=2))
print("wrote frontend/openapi.json")
```

## Contract / data changes
- New env vars: `CORS_ALLOW_ORIGINS`, `API_REQUIRE_AUTH`.
- New optional auth header `X-API-Key` on compute endpoints (only enforced when `API_REQUIRE_AUTH=true`).
- OpenAPI JSON exported to `frontend/openapi.json`.

## Acceptance criteria
- [ ] `grep -rn "from pydantic.types" src/` returns nothing; no `@validator`/`@root_validator`/`constr`/`confloat` remain (`grep -rn "validator\|constr\|confloat\|conint\|\.dict()" src/api/`).
- [ ] App boots; `GET /health` 200; `POST /predict` with a valid body returns 200; invalid body returns 422 with the existing error envelope.
- [ ] With `API_REQUIRE_AUTH=true`, `POST /predict` without `X-API-Key` returns 401; with the correct key returns 200.
- [ ] CORS preflight from `http://localhost:5173` is allowed; from a random origin is not.
- [ ] `python scripts/export_openapi.py` writes a valid `frontend/openapi.json`.
- [ ] `make check` passes.

## Test plan
Create `tests/unit/test_schemas.py`:
- valid `TransactionInput` parses; `amount<=0` raises `ValidationError`; bad `type` raises; oversized `sender_id` raises.
- round-trip `model_dump()` keys match expected.

Create `tests/integration/test_api_predict.py` (using `httpx.AsyncClient` + ASGITransport):
- `/health` 200; `/predict` happy path 200 (model may be the heuristic until spec 06 — assert response shape, not value); `/predict` invalid → 422.
- auth: with `API_REQUIRE_AUTH=true` (monkeypatch config), 401 without key, 200 with key.

## Validation
```bash
grep -rn "from pydantic.types" src/ ; echo exit:$?
grep -rn "\.dict()" src/api/ ; echo exit:$?
pytest tests/unit/test_schemas.py tests/integration/test_api_predict.py -q
python scripts/export_openapi.py && python -c "import json;json.load(open('frontend/openapi.json'))" && echo "openapi ok"
make check
```

## Rollback / fallback
Schema migration is mechanical; if a validator's semantics are unclear, preserve the v1 behavior exactly and add a test pinning it. Auth/rate-limit/CORS are additive and default-off (auth) or dev-friendly (CORS) — revert by removing the middleware/dependency wiring.

## Definition of Done
All criteria checked; contract stable and documented; `frontend/openapi.json` generated; commit on `feat/spec-04-pydantic-v2-hardening`.

## References
- `src/api/schemas.py` (full file), `src/api/main.py:100-122,158-208,288-356`, `src/config.py:135-183`.
- Pydantic v2 migration guide; `slowapi` docs.

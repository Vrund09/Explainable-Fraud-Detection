# 02 — Foundation: Tooling & Test Bootstrap

| Field | Value |
| --- | --- |
| Spec ID | 02 |
| Status | Not started |
| Depends on | — (do this first) |
| Blocks | All other specs (they assume lint/format/test gates exist) |
| Est. effort | 0.5–1 day |
| Risk | Low |

## Objective
Make the repository's quality gates real and green-able **before** any feature work. Today CI runs `black`, `isort`, `flake8`, `mypy`, and `pytest --cov` but there is no config and no tests (G5). This spec adds tool configuration, a `tests/` scaffold with shared fixtures, `pre-commit`, and a `Makefile`/task list so every later spec can be validated consistently.

## Background & current state
- CI steps: `.github/workflows/ci.yml:64-92` (`black --check --diff src/ tests/`, `isort --check-only`, `flake8 ... --max-line-length=127`, `mypy src/`, `pytest tests/ -v --cov=src`).
- `tests/` contains only `__init__.py`.
- No `pyproject.toml`, `setup.cfg`, `.flake8`, `mypy.ini`, or `.pre-commit-config.yaml` exist.
- Dev tools already pinned in `requirements.txt:56-60` (`black==23.7.0`, `flake8==6.0.0`, `isort==5.12.0`, `mypy==1.5.1`).

## Prerequisites
- Python env set up per `00-conventions §2.1`.

## Out of scope
- Writing the full test suite (that is spec `11`). Here we only add a **smoke test** so `pytest` exits 0 and coverage is non-empty.
- Changing dependency versions (spec `03`).

## Implementation steps

### Step 1 — Add `pyproject.toml` tool config (repo root)
Create `pyproject.toml` with consistent settings so `black`/`isort` agree and `flake8` matches the 127 line length used in CI:

```toml
[tool.black]
line-length = 100
target-version = ["py311"]

[tool.isort]
profile = "black"
line_length = 100
known_first_party = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "requires_secrets: needs live API keys; skipped unless RUN_SECRET_TESTS=1",
    "requires_neo4j: needs a running Neo4j; skipped unless RUN_NEO4J_TESTS=1",
    "slow: long-running tests",
]

[tool.coverage.run]
source = ["src"]
omit = ["src/**/__init__.py"]
```

> Note: `black` default line length is 88; CI's flake8 uses 127. Set both `black` and flake8 to a single value (100) to avoid conflicts. Update `.flake8` below to match.

### Step 2 — Add `.flake8` (repo root)
`flake8` does not read `pyproject.toml`. Create `.flake8`:

```ini
[flake8]
max-line-length = 100
max-complexity = 12
extend-ignore = E203, W503
exclude = .venv, .git, __pycache__, build, dist, frontend
per-file-ignores =
    __init__.py: F401
```

### Step 3 — Add `mypy.ini` (repo root)
```ini
[mypy]
python_version = 3.11
ignore_missing_imports = True
warn_unused_ignores = True
no_implicit_optional = True
disallow_untyped_defs = False
exclude = (frontend|tests/fixtures)
```

### Step 4 — Format the existing codebase
Run `isort` then `black` over `src/` so the existing files satisfy the gates:
```bash
isort src/
black src/
```
Commit this as a **separate** formatting-only commit: `chore(spec-02): apply black + isort to src`.

> If `black`/`isort` change many files, that is expected and fine — it is a one-time normalization. Do not change logic.

### Step 5 — Create the test scaffold
Create the directory tree from `00-conventions §4` (empty dirs are created by adding the files below). At minimum create:

`tests/conftest.py`:
```python
"""Shared pytest fixtures."""
import os
import pytest


@pytest.fixture(scope="session")
def repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


def _flag(name: str) -> bool:
    return os.getenv(name, "0") == "1"


def pytest_collection_modifyitems(config, items):
    skip_secrets = pytest.mark.skip(reason="set RUN_SECRET_TESTS=1 to run")
    skip_neo4j = pytest.mark.skip(reason="set RUN_NEO4J_TESTS=1 to run")
    for item in items:
        if "requires_secrets" in item.keywords and not _flag("RUN_SECRET_TESTS"):
            item.add_marker(skip_secrets)
        if "requires_neo4j" in item.keywords and not _flag("RUN_NEO4J_TESTS"):
            item.add_marker(skip_neo4j)
```

`tests/unit/test_smoke.py`:
```python
"""Smoke tests proving the package imports and config loads."""

def test_config_imports():
    from src.config import config
    assert config.API_VERSION
    assert config.GNN_INPUT_DIM > 0


def test_risk_level_thresholds():
    from src.gnn_model.predict import FraudPredictor
    p = FraudPredictor.__new__(FraudPredictor)  # no MLflow init
    assert p._get_risk_level(0.1) == "LOW"
    assert p._get_risk_level(0.4) == "MEDIUM"
    assert p._get_risk_level(0.6) == "HIGH"
    assert p._get_risk_level(0.9) == "CRITICAL"
```

> `_get_risk_level` is a pure method (`src/gnn_model/predict.py:387-404`) and is safe to test without loading a model. Constructing via `__new__` avoids MLflow setup in `__init__`.

Add empty `tests/unit/__init__.py`, `tests/integration/__init__.py`, and a `tests/fixtures/.gitkeep`.

### Step 6 — Add `.pre-commit-config.yaml` (repo root)
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks: [{ id: black }]
  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks: [{ id: isort }]
  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks: [{ id: flake8 }]
```
Install: `pre-commit install`.

### Step 7 — Add a `Makefile` (repo root) for one-command validation
```makefile
.PHONY: format lint type test cov check
format:
	isort src/ tests/ && black src/ tests/
lint:
	flake8 src/ tests/
type:
	mypy src/
test:
	pytest -q
cov:
	pytest --cov=src --cov-report=term-missing
check: lint type test
```
> On Windows without `make`, document the equivalent commands in `PROGRESS.md`.

### Step 8 — Align CI line length
In `.github/workflows/ci.yml:77`, change `--max-line-length=127` to `--max-line-length=100` so CI matches `.flake8`. Leave the rest of CI unchanged in this spec (full CI hardening is spec `11`).

## Contract / data changes
None (tooling only).

## Acceptance criteria
- [ ] `pyproject.toml`, `.flake8`, `mypy.ini`, `.pre-commit-config.yaml`, `Makefile` exist at repo root with the content above.
- [ ] `black --check src/ tests/` exits 0.
- [ ] `isort --check-only src/ tests/` exits 0.
- [ ] `flake8 src/ tests/` exits 0 (or only legacy complexity warnings that CI treats as non-blocking; syntax/undefined-name checks E9/F63/F7/F82 must be clean).
- [ ] `pytest -q` collects and passes at least the smoke tests; coverage report is produced and non-zero.
- [ ] CI `flake8` line length updated to 100.

## Test plan
- `tests/unit/test_smoke.py` — config import + `_get_risk_level` thresholds (new, must pass).

## Validation
```bash
isort --check-only src/ tests/
black --check src/ tests/
flake8 src/ tests/
mypy src/ --ignore-missing-imports
pytest -q --cov=src --cov-report=term-missing
pre-commit run --all-files
```

## Rollback / fallback
All additions are new config/test files plus a formatting commit. Revert by deleting the new files and `git revert` the formatting commit.

## Definition of Done
All acceptance criteria checked; commits pushed on `feat/spec-02-tooling`; `PROGRESS.md` updated. The global DoD in `README.md` applies.

## References
- `.github/workflows/ci.yml:64-92`, `requirements.txt:56-60`, `src/gnn_model/predict.py:387-404`, `src/config.py`.

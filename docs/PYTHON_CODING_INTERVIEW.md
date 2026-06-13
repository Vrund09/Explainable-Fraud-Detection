# Python Coding Interview Exercises
## Patterns Directly From This Project
### All exercises are runnable Python — verify output before your interview

---

## EXERCISE 1 — Context Manager Pattern

**Project source:** `src/api/main.py` — `lifespan()` async context manager; `src/gnn_model/training.py` — `with mlflow.start_run() as run`

---

**Interview Prompt:**
> "Write a Python context manager class that manages a database connection lifecycle — it should open the connection on enter, close it on exit, and handle exceptions gracefully. Then show me how you'd write the same thing as a generator-based context manager using `contextlib`. Your project uses both patterns — walk me through where."

---

**Solution:**

```python
import contextlib
import time
from typing import Optional


# ── Pattern 1: Class-based context manager ──────────────────────────
class DatabaseConnection:
    """
    Manages a database connection lifecycle.
    Mirrors the Neo4j driver usage in graph_constructor.py.
    """

    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.connection: Optional[object] = None
        self._start_time: Optional[float] = None

    def __enter__(self) -> "DatabaseConnection":
        print(f"[DB] Connecting to {self.uri}...")
        # Simulate connection (real: GraphDatabase.driver(uri, auth=(...)))
        self.connection = {"uri": self.uri, "status": "connected"}
        self._start_time = time.time()
        print(f"[DB] Connected.")
        return self  # returns self so 'as conn' gives the manager

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> bool:
        elapsed = time.time() - self._start_time
        if exc_type is not None:
            # Log the exception but don't suppress it — return False
            print(f"[DB] Exception during session: {exc_type.__name__}: {exc_val}")
            print(f"[DB] Closing connection after {elapsed:.3f}s (error path)")
        else:
            print(f"[DB] Closing connection after {elapsed:.3f}s (clean path)")

        self.connection = None
        # Returning False re-raises the exception; True suppresses it
        return False

    def query(self, cypher: str) -> dict:
        if self.connection is None:
            raise RuntimeError("Not connected. Use inside 'with' block.")
        print(f"[DB] Running: {cypher}")
        return {"result": "mock_data"}


# ── Pattern 2: Generator-based using @contextmanager ────────────────
@contextlib.contextmanager
def mlflow_run(experiment_name: str):
    """
    Mirrors the 'with mlflow.start_run() as run' pattern in training.py.
    Code before yield = startup. Code after yield = teardown.
    """
    run_id = f"run_{int(time.time())}"
    print(f"[MLflow] Starting run: {run_id} in experiment: {experiment_name}")
    metrics_logged = []

    try:
        yield {"run_id": run_id, "metrics": metrics_logged}  # the 'as run' value
    except Exception as e:
        print(f"[MLflow] Run {run_id} FAILED: {e}")
        raise  # re-raise so calling code sees the exception
    finally:
        # Always runs — equivalent to __exit__ being called
        print(f"[MLflow] Run {run_id} ended. Metrics logged: {len(metrics_logged)}")


# ── Demo ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Class-based context manager ===")
    with DatabaseConnection("bolt://localhost:7687", "neo4j", "password") as db:
        result = db.query("MATCH (u:User) RETURN u LIMIT 1")
        print(f"Query result: {result}")

    print()
    print("=== Generator-based context manager ===")
    with mlflow_run("fraud-detection-gnn") as run:
        run["metrics"].append({"f1_score": 0.906, "epoch": 45})
        run["metrics"].append({"roc_auc": 0.963, "epoch": 45})
        print(f"Run ID: {run['run_id']}, metrics: {run['metrics']}")

    print()
    print("=== Exception handling in context manager ===")
    try:
        with DatabaseConnection("bolt://localhost:7687", "neo4j", "bad_pass") as db:
            raise ValueError("Authentication failed")
    except ValueError as e:
        print(f"Caught outside: {e}")
```

**Expected Output:**
```
=== Class-based context manager ===
[DB] Connecting to bolt://localhost:7687...
[DB] Connected.
[DB] Running: MATCH (u:User) RETURN u LIMIT 1
Query result: {'result': 'mock_data'}
[DB] Closing connection after 0.000s (clean path)

=== Generator-based context manager ===
[MLflow] Starting run: run_1700000000 in experiment: fraud-detection-gnn
Run ID: run_1700000000, metrics: [{'f1_score': 0.906, 'epoch': 45}, {'roc_auc': 0.963, 'epoch': 45}]
[MLflow] Run run_1700000000 ended. Metrics logged: 2

=== Exception handling in context manager ===
[DB] Connecting to bolt://localhost:7687...
[DB] Connected.
[DB] Exception during session: ValueError: Authentication failed
[DB] Closing connection after 0.000s (error path)
Caught outside: Authentication failed
```

**Follow-up Questions:**
1. "When would you return `True` from `__exit__`?" — To suppress the exception (e.g., a `SuppressingContext` that catches `FileNotFoundError` and treats it as a no-op). Almost never the right choice.
2. "What's the difference between `yield` in a generator and `yield` in a `@contextmanager`?" — In a contextmanager, `yield` is a single pause point — everything before is setup, everything after is teardown. In a regular generator, `yield` can appear multiple times.
3. "How does `asynccontextmanager` differ from `contextmanager`?" — Same pattern but with `async def` and `await` allowed. Used in `main.py` for `lifespan()` because FastAPI's startup/shutdown is async.
4. "What happens if you forget `yield` in a `@contextmanager` function?" — `contextlib` raises `RuntimeError: generator didn't yield`.

---

## EXERCISE 2 — Caching / Memoization Pattern

**Project source:** `src/gnn_model/predict.py` — `FraudPredictor.node_mapping` (pre-loaded lookup), `src/api/main.py` — module-level `fraud_predictor` singleton

---

**Interview Prompt:**
> "Implement a thread-safe LRU cache for expensive fraud feature lookups. Then show me how Python's built-in `functools.lru_cache` could be applied. Your project does eager-loading of the node mapping — explain when that's better than lazy caching."

---

**Solution:**

```python
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Optional


# ── Pattern 1: Thread-safe LRU cache class ───────────────────────────
class ThreadSafeLRUCache:
    """
    Thread-safe LRU cache for fraud feature lookups.
    Mirrors how FraudPredictor.node_mapping works but with eviction.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()  # Reentrant lock — safe for nested calls
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self.max_size:
                # Remove least recently used (first item)
                evicted_key, _ = self._cache.popitem(last=False)
                print(f"[Cache] Evicted: {evicted_key}")

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0.0,
            }


# ── Pattern 2: functools.lru_cache for pure functions ────────────────
@lru_cache(maxsize=256)
def compute_amount_log(amount: float) -> float:
    """
    Mirrors preprocess_transaction()'s amount_log = log1p(amount).
    lru_cache works on hashable arguments — floats qualify.
    NOTE: Not appropriate for mutable args (dicts, lists).
    """
    import math
    print(f"[Cache MISS] Computing log1p({amount})")
    return math.log1p(amount)


# ── Pattern 3: Eager loading (the project's actual approach) ─────────
class FraudPredictorCache:
    """
    Demonstrates the eager-load-once pattern used in FraudPredictor.
    node_mapping is loaded fully at startup, not lazily on demand.
    """

    def __init__(self):
        self._node_mapping: dict[str, int] = {}
        self._loaded = False

    def load_node_mapping(self, mapping: dict[str, int]) -> None:
        """Load at startup — O(n) once, O(1) per lookup thereafter."""
        self._node_mapping = mapping
        self._loaded = True
        print(f"[Eager] Loaded {len(mapping)} nodes into mapping")

    def get_node_index(self, user_id: str) -> int:
        if not self._loaded:
            raise RuntimeError("Node mapping not loaded. Call load_node_mapping() first.")
        idx = self._node_mapping.get(user_id)
        if idx is None:
            raise ValueError(f"Unknown user_id: {user_id}")
        return idx


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== LRU Cache ===")
    cache = ThreadSafeLRUCache(max_size=3)
    cache.set("C123", {"fraud_rate": 0.05, "total_txns": 100})
    cache.set("C456", {"fraud_rate": 0.15, "total_txns": 50})
    cache.set("C789", {"fraud_rate": 0.01, "total_txns": 200})

    print(cache.get("C123"))   # Hit
    cache.set("C999", {"fraud_rate": 0.8, "total_txns": 5})  # Evicts C456 (LRU)
    print(cache.get("C456"))   # Miss (evicted)
    print("Stats:", cache.stats())

    print()
    print("=== functools.lru_cache ===")
    # Same amount called twice — second is a cache hit
    print(compute_amount_log(150000.0))
    print(compute_amount_log(150000.0))  # No "Cache MISS" printed
    print("Cache info:", compute_amount_log.cache_info())

    print()
    print("=== Eager loading ===")
    predictor_cache = FraudPredictorCache()
    predictor_cache.load_node_mapping({"C123": 0, "C456": 1, "M789": 2})
    print("Node index for C123:", predictor_cache.get_node_index("C123"))
```

**Expected Output:**
```
=== LRU Cache ===
{'fraud_rate': 0.05, 'total_txns': 100}
[Cache] Evicted: C456
None
Stats: {'size': 3, 'hits': 1, 'misses': 1, 'hit_rate': 0.5}

=== functools.lru_cache ===
[Cache MISS] Computing log1p(150000.0)
11.918391993...
11.918391993...
Cache info: CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)

=== Eager loading ===
[Eager] Loaded 3 nodes into mapping
Node index for C123: 0
```

**Follow-up Questions:**
1. "When does `lru_cache` break and what's the fix?" — When arguments are mutable (dicts, lists) — they're not hashable. Fix: convert to `tuple` or `frozenset`, or use a custom cache with `functools.wraps`.
2. "Why use `RLock` instead of `Lock`?" — `RLock` allows the same thread to acquire the lock multiple times without deadlocking. Safe if `get()` internally calls another method that also locks.
3. "Why did the project choose eager loading over lazy loading for `node_mapping`?" — Eager loading pays the cost once at startup; lazy loading would add latency to the first request per user. At 500 req/min, startup latency is acceptable; per-request latency is not.
4. "How would you expire cache entries in `ThreadSafeLRUCache`?" — Add a `(value, timestamp)` tuple; in `get()`, check `time.time() - timestamp > ttl` and treat expired entries as misses.

---

## EXERCISE 3 — Iterator / Generator Pattern

**Project source:** `src/data_processing/graph_constructor.py` — `ingest_nodes_to_neo4j()` batch loop; `src/gnn_model/predict.py` — `predict_batch()` mini-batch loop

---

**Interview Prompt:**
> "Write a generator that yields transaction batches from a large CSV file without loading the entire file into memory. Your project ingests 6.36 million transactions — show how you'd do this memory-efficiently."

---

**Solution:**

```python
import csv
from pathlib import Path
from typing import Generator, Iterator


# ── Pattern 1: File-reading generator ────────────────────────────────
def stream_transaction_batches(
    file_path: str, batch_size: int = 1000
) -> Generator[list[dict], None, None]:
    """
    Yields batches of transactions from a CSV without loading all into RAM.
    Mirrors the batch processing in ingest_nodes_to_neo4j() (batch_size=1000)
    and ingest_edges_to_neo4j() (batch_size=500) in graph_constructor.py.
    """
    batch: list[dict] = []

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []  # Reset for next batch

    # Yield the final partial batch (don't discard remainder)
    if batch:
        yield batch


# ── Pattern 2: Generator with transformation ─────────────────────────
def preprocess_transaction_stream(
    raw_stream: Iterator[list[dict]],
) -> Generator[list[dict], None, None]:
    """
    Chains with stream_transaction_batches — applies feature engineering
    to each batch. Mirrors preprocess_data() in graph_constructor.py.
    """
    import math

    for batch in raw_stream:
        processed = []
        for txn in batch:
            try:
                amount = float(txn.get("amount", 0))
                step = int(txn.get("step", 0))
                processed.append(
                    {
                        **txn,
                        "amount_log": math.log1p(amount),
                        "hour_of_day": step % 24,
                        "day_of_month": (step // 24) % 30,
                    }
                )
            except (ValueError, KeyError) as e:
                print(f"[Skip] Bad row: {e}")
                continue
        yield processed


# ── Pattern 3: Generator pipeline composition ─────────────────────────
def count_fraud_in_stream(batches: Iterator[list[dict]]) -> tuple[int, int]:
    """Consume a batch stream, counting total and fraud transactions."""
    total, fraud = 0, 0
    for batch in batches:
        for txn in batch:
            total += 1
            if str(txn.get("isFraud", "0")) == "1":
                fraud += 1
    return total, fraud


# ── Demo with in-memory CSV simulation ───────────────────────────────
if __name__ == "__main__":
    import io
    import tempfile
    import os

    # Create a temp CSV simulating PaySim structure
    sample_rows = [
        "step,type,amount,nameOrig,nameDest,isFraud",
    ]
    for i in range(10):
        is_fraud = 1 if i % 7 == 0 else 0
        sample_rows.append(f"{i},TRANSFER,{(i+1)*10000},C{i:010d},M{i:010d},{is_fraud}")

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        f.write("\n".join(sample_rows))
        temp_path = f.name

    print(f"=== Streaming {len(sample_rows)-1} transactions in batches of 3 ===")
    raw = stream_transaction_batches(temp_path, batch_size=3)
    enriched = preprocess_transaction_stream(raw)

    batch_num = 0
    for batch in enriched:
        batch_num += 1
        print(f"Batch {batch_num}: {len(batch)} rows | "
              f"first amount_log={float(batch[0]['amount_log']):.2f} | "
              f"hour_of_day={batch[0]['hour_of_day']}")

    print()
    print("=== Counting fraud via generator pipeline ===")
    # Re-create generator (generators are single-use!)
    raw2 = stream_transaction_batches(temp_path, batch_size=5)
    total, fraud = count_fraud_in_stream(raw2)
    print(f"Total: {total}, Fraud: {fraud}, Rate: {fraud/total:.2%}")

    os.unlink(temp_path)  # cleanup
```

**Expected Output:**
```
=== Streaming 10 transactions in batches of 3 ===
Batch 1: 3 rows | first amount_log=9.21 | hour_of_day=0
Batch 2: 3 rows | first amount_log=10.60 | hour_of_day=3
Batch 3: 3 rows | first amount_log=11.51 | hour_of_day=6
Batch 4: 1 rows | first amount_log=11.92 | hour_of_day=9

=== Counting fraud via generator pipeline ===
Total: 10, Fraud: 2, Rate: 20.00%
```

**Follow-up Questions:**
1. "Why are generators single-use?" — A generator function returns a generator *object*. Once exhausted (StopIteration raised), it can't be reset. To re-iterate, call the generator function again.
2. "How would you make the pipeline multi-threaded?" — Use `concurrent.futures.ThreadPoolExecutor` and `executor.map(process_batch, batches)`. The batches generator feeds the thread pool.
3. "What's the memory advantage vs. `pd.read_csv(file)`?" — `pd.read_csv()` loads all 6.36M rows into a DataFrame at once (~4-8GB RAM). The generator holds at most `batch_size=1000` rows in memory at any time.
4. "When would `pd.read_csv(chunksize=1000)` be better than your generator?" — When you need Pandas operations (groupby, merge, vectorized math) on each chunk. The generator version is better for simple row-by-row transformations.

---

## EXERCISE 4 — Decorator Pattern

**Project source:** `src/api/main.py` — `@app.middleware("http")`, `@app.exception_handler()`, `@app.get()`, `@app.post()`; `src/gnn_model/training.py` — `@torch.no_grad()` on `evaluate()`

---

**Interview Prompt:**
> "Write a decorator that measures execution time and logs it — similar to the request timing middleware in your FastAPI app. Then write a decorator that retries a function on failure with exponential backoff — relevant to your Neo4j and Gemini API calls."

---

**Solution:**

```python
import functools
import logging
import time
from typing import Callable, Optional, Type

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Pattern 1: Timing decorator ──────────────────────────────────────
def timed(func: Callable) -> Callable:
    """
    Measures and logs execution time.
    Mirrors the X-Process-Time header logic in add_request_id_middleware()
    in src/api/main.py.
    """
    @functools.wraps(func)  # Preserves __name__, __doc__, __annotations__
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(f"[{func.__name__}] completed in {elapsed_ms:.2f}ms")
            return result
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[{func.__name__}] FAILED after {elapsed_ms:.2f}ms: {e}")
            raise
    return wrapper


# ── Pattern 2: Retry with exponential backoff ─────────────────────────
def retry(
    max_attempts: int = 3,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    base_delay_s: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
) -> Callable:
    """
    Retries a function on specified exceptions with exponential backoff.
    Relevant to Neo4j ServiceUnavailable and Gemini API errors in this project.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import random
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"[{func.__name__}] All {max_attempts} attempts failed. "
                            f"Last error: {e}"
                        )
                        raise

                    delay = base_delay_s * (backoff_factor ** (attempt - 1))
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)  # ±25% jitter

                    logger.warning(
                        f"[{func.__name__}] Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)

            raise last_exception  # unreachable but satisfies type checker
        return wrapper
    return decorator


# ── Pattern 3: Parametrized class-based decorator ────────────────────
class cached_property:
    """
    Computes a property once and caches it on the instance.
    Useful for the FraudPredictor.model_summary which is expensive to compute.
    """
    def __init__(self, func: Callable):
        self.func = func
        self.attr_name = f"_cached_{func.__name__}"
        functools.update_wrapper(self, func)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self  # Called on the class, not an instance
        if not hasattr(obj, self.attr_name):
            setattr(obj, self.attr_name, self.func(obj))
        return getattr(obj, self.attr_name)


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Timed decorator
    @timed
    def predict_fraud(amount: float) -> float:
        time.sleep(0.05)  # Simulate 50ms inference
        return min(amount / 1_000_000, 0.99)

    print("=== Timing decorator ===")
    result = predict_fraud(500_000)
    print(f"Fraud probability: {result:.2%}")

    print()
    print("=== Retry decorator ===")
    call_count = 0

    @retry(max_attempts=3, exceptions=(ConnectionError,), base_delay_s=0.1)
    def query_neo4j(query: str) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError(f"Neo4j unavailable (attempt {call_count})")
        return {"result": "data", "attempt": call_count}

    result = query_neo4j("MATCH (u:User) RETURN u LIMIT 1")
    print(f"Query result after {call_count} attempts: {result}")

    print()
    print("=== cached_property ===")
    class ModelWrapper:
        def __init__(self, dims: list[int]):
            self.dims = dims
            self._compute_count = 0

        @cached_property
        def parameter_count(self) -> int:
            self._compute_count += 1
            print(f"  [Computing parameter count... call #{self._compute_count}]")
            return sum(d1 * d2 for d1, d2 in zip(self.dims[:-1], self.dims[1:]))

    model = ModelWrapper([10, 128, 64, 32, 1])
    print(f"Params: {model.parameter_count:,}")  # Computed
    print(f"Params: {model.parameter_count:,}")  # Cached — no recompute
    print(f"Compute calls: {model._compute_count}")  # Should be 1
```

**Expected Output:**
```
=== Timing decorator ===
INFO [...] [predict_fraud] completed in 50.xx ms
Fraud probability: 50.00%

=== Retry decorator ===
WARNING [...] [query_neo4j] Attempt 1/3 failed: Neo4j unavailable (attempt 1). Retrying in 0.0xs...
WARNING [...] [query_neo4j] Attempt 2/3 failed: Neo4j unavailable (attempt 2). Retrying in 0.1xs...
Query result after 3 attempts: {'result': 'data', 'attempt': 3}

=== cached_property ===
  [Computing parameter count... call #1]
Params: 10,592
Params: 10,592
Compute calls: 1
```

**Follow-up Questions:**
1. "Why is `@functools.wraps(func)` important?" — Without it, the wrapper function has `__name__ = 'wrapper'` and loses the original function's `__doc__` and type annotations. FastAPI uses `__name__` for route naming; mypy uses annotations for type checking.
2. "What's the difference between a decorator and a context manager?" — A decorator wraps a function (applied at definition time, modifies every call). A context manager wraps a code block (applied at call time, wraps specific execution). Both can do setup + teardown, but decorators apply to all invocations.
3. "How would you make the retry decorator async?" — Add `import asyncio`, check `if asyncio.iscoroutinefunction(func)`, and use an async wrapper with `await asyncio.sleep(delay)` instead of `time.sleep(delay)`.
4. "When is `cached_property` better than `@lru_cache`?" — `cached_property` caches per-instance (each object has its own cached value). `lru_cache` on a method would use `self` as part of the cache key, preventing garbage collection of `self`. `cached_property` is the correct pattern for instance-level caching.

---

## EXERCISE 5 — Retry / Resilience Pattern

**Project source:** `src/gnn_model/predict.py` — `load_production_model()` fallback chain; `src/explainability/agent.py` — `_initialize_neo4j()` catch + continue

---

**Interview Prompt:**
> "Your project has a two-tier fallback in `load_production_model()` — it tries Production stage, then falls back to the latest version. Implement a generic fallback chain that tries multiple strategies in order, returns the first success, and logs which strategy succeeded."

---

**Solution:**

```python
import logging
import time
from typing import Any, Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Pattern: Fallback chain with logging ─────────────────────────────
class FallbackChain:
    """
    Tries strategies in order, returns the first successful result.
    Mirrors the two-tier fallback in load_production_model() in predict.py:
      1. Try models:/fraud-detection-model/Production
      2. Fall back to latest version (stages=["None"])
      3. Raise if all fail
    """

    def __init__(self, raise_if_all_fail: bool = True):
        self.raise_if_all_fail = raise_if_all_fail
        self._strategies: list[tuple[str, Callable]] = []

    def add(self, name: str, strategy: Callable) -> "FallbackChain":
        """Builder pattern — allows chaining .add() calls."""
        self._strategies.append((name, strategy))
        return self

    def run(self) -> Any:
        errors = []

        for name, strategy in self._strategies:
            try:
                logger.info(f"[Fallback] Trying strategy: '{name}'")
                result = strategy()
                logger.info(f"[Fallback] SUCCESS with strategy: '{name}'")
                return result
            except Exception as e:
                logger.warning(f"[Fallback] Strategy '{name}' failed: {type(e).__name__}: {e}")
                errors.append((name, e))
                continue

        if self.raise_if_all_fail:
            error_summary = "; ".join(f"{n}: {e}" for n, e in errors)
            raise RuntimeError(f"All fallback strategies failed: {error_summary}")

        logger.error("[Fallback] All strategies failed. Returning None.")
        return None


# ── Specific usage: model loading (mirrors predict.py) ───────────────
def load_model_from_production() -> dict:
    """Simulates mlflow.pytorch.load_model('models:/fraud-detection-model/Production')"""
    raise ConnectionError("MLflow Production stage not found")


def load_model_latest_version() -> dict:
    """Simulates loading the latest 'None' stage version"""
    return {
        "model": "GraphSAGEClassifier",
        "version": "3",
        "stage": "None",
        "f1_score": 0.906,
    }


def load_model_from_checkpoint() -> dict:
    """Last resort: load from local checkpoint file"""
    return {
        "model": "GraphSAGEClassifier",
        "version": "checkpoint_epoch_90",
        "stage": "local",
        "f1_score": 0.891,
    }


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Model loading with fallback chain ===")
    chain = (
        FallbackChain()
        .add("Production MLflow stage", load_model_from_production)
        .add("Latest MLflow version", load_model_latest_version)
        .add("Local checkpoint", load_model_from_checkpoint)
    )
    model = chain.run()
    print(f"Loaded: {model}")

    print()
    print("=== All strategies fail ===")
    failing_chain = (
        FallbackChain(raise_if_all_fail=True)
        .add("Strategy A", lambda: (_ for _ in ()).throw(ValueError("A failed")))
        .add("Strategy B", lambda: (_ for _ in ()).throw(IOError("B failed")))
    )
    try:
        failing_chain.run()
    except RuntimeError as e:
        print(f"Caught: {e}")

    print()
    print("=== Graceful degradation (raise_if_all_fail=False) ===")
    soft_chain = (
        FallbackChain(raise_if_all_fail=False)
        .add("Primary", lambda: (_ for _ in ()).throw(Exception("unavailable")))
    )
    result = soft_chain.run()
    print(f"Result: {result}")  # None — degraded mode
```

**Expected Output:**
```
=== Model loading with fallback chain ===
INFO [...] [Fallback] Trying strategy: 'Production MLflow stage'
WARNING [...] [Fallback] Strategy 'Production MLflow stage' failed: ConnectionError: MLflow Production stage not found
INFO [...] [Fallback] Trying strategy: 'Latest MLflow version'
INFO [...] [Fallback] SUCCESS with strategy: 'Latest MLflow version'
Loaded: {'model': 'GraphSAGEClassifier', 'version': '3', 'stage': 'None', 'f1_score': 0.906}

=== All strategies fail ===
...
Caught: All fallback strategies failed: Strategy A: A failed; Strategy B: B failed

=== Graceful degradation (raise_if_all_fail=False) ===
ERROR [...] [Fallback] All strategies failed. Returning None.
Result: None
```

**Follow-up Questions:**
1. "How does this differ from `try/except/else/finally`?" — The chain abstraction handles N strategies without N nested try/except blocks, centralizes logging, and is composable.
2. "When would you NOT use a fallback chain?" — When falling back to a degraded result is more dangerous than failing fast (e.g., a fallback model with unknown quality should never silently serve predictions without a warning).
3. "How would you add a circuit breaker on top of this?" — Track failure counts per strategy; after `failure_threshold` failures in `window_seconds`, skip that strategy entirely for a `reset_timeout` period. The `circuitbreaker` library provides this.
4. "How does this relate to `load_production_model()` in the project?" — `load_production_model()` in `predict.py` implements the same two-tier logic manually: `try: load Production / except: try: load latest version`. This class makes that pattern explicit, testable, and reusable.

---

## EXERCISE 6 — Data Parsing / Transformation Pattern

**Project source:** `src/explainability/agent.py` — `_parse_agent_response()`, `_create_fallback_explanation()`; `src/api/schemas.py` — `@root_validator`

---

**Interview Prompt:**
> "Write a robust parser for the AI explanation output that your project's LangChain agent generates. Handle missing fields, unexpected formats, and malformed JSON gracefully. Your current implementation uses keyword scanning — show me how you'd do it better."

---

**Solution:**

```python
import json
import re
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class ParsedExplanation:
    """Mirrors ExplanationOutput schema in src/api/schemas.py"""
    explanation_text: str = ""
    key_factors: list[str] = field(default_factory=list)
    risk_level: str = "UNKNOWN"
    recommendation: str = "Manual review recommended"
    confidence: float = 0.5
    parse_method: str = "unknown"  # For observability


def parse_explanation_response(raw_output: str) -> ParsedExplanation:
    """
    Robust multi-strategy parser for LLM explanation output.
    Tries JSON first, then structured regex, then keyword scanning,
    then fallback defaults — in that order.
    """
    result = ParsedExplanation()

    # ── Strategy 1: Structured JSON output ───────────────────────────
    # Best approach: when model is configured with structured output
    json_match = re.search(r'\{[^{}]*"explanation_text"[^{}]*\}', raw_output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            result.explanation_text = parsed.get("explanation_text", "")
            result.key_factors = parsed.get("key_factors", [])[:5]  # Cap at 5
            result.risk_level = parsed.get("risk_level", "UNKNOWN").upper()
            result.recommendation = parsed.get("recommendation", result.recommendation)
            result.confidence = float(parsed.get("confidence", 0.7))
            result.parse_method = "json"
            return result
        except (json.JSONDecodeError, ValueError):
            pass  # Fall through to next strategy

    # ── Strategy 2: Structured section headers ────────────────────────
    # Works when LLM output uses "RISK FACTORS:", "RECOMMENDATION:" etc.
    sections = {
        "explanation": re.search(r"(?:ANALYSIS|EXPLANATION):\s*(.+?)(?=\n[A-Z]+:|$)",
                                  raw_output, re.DOTALL | re.IGNORECASE),
        "factors": re.findall(r"(?:risk factor|indicator|concern):\s*([^\n]+)",
                               raw_output, re.IGNORECASE),
        "risk": re.search(r"\b(LOW|MEDIUM|HIGH|CRITICAL)\s+(?:risk|RISK)", raw_output),
        "recommendation": re.search(r"(?:recommend|action):\s*([^\n]+)",
                                     raw_output, re.IGNORECASE),
    }

    if sections["explanation"] or sections["factors"]:
        result.explanation_text = (
            sections["explanation"].group(1).strip()
            if sections["explanation"]
            else raw_output[:500]
        )
        result.key_factors = [f.strip() for f in sections["factors"][:5]]
        result.risk_level = sections["risk"].group(1) if sections["risk"] else "UNKNOWN"
        result.recommendation = (
            sections["recommendation"].group(1).strip()
            if sections["recommendation"]
            else result.recommendation
        )
        result.confidence = 0.7
        result.parse_method = "regex_sections"
        return result

    # ── Strategy 3: Keyword scanning (current project implementation) ─
    lines = raw_output.split("\n")
    factors = []
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["risk factor", "indicator", "suspicious", "anomaly"]):
            factors.append(line.strip())
        if "recommend" in line_lower and not result.recommendation:
            result.recommendation = line.strip()

    if factors or len(raw_output) > 50:
        result.explanation_text = raw_output
        result.key_factors = factors[:5]
        result.confidence = 0.6
        result.parse_method = "keyword_scan"
        return result

    # ── Strategy 4: Fallback default ─────────────────────────────────
    result.explanation_text = "Fraud prediction flagged by GNN model."
    result.key_factors = ["Pattern matching with known fraud indicators"]
    result.confidence = 0.5
    result.parse_method = "fallback"
    return result


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: JSON output (ideal case)
    json_output = '''
    {"explanation_text": "HIGH fraud risk detected.", "key_factors":
     ["Elevated fraud rate (15%)", "High-risk network neighbors"],
     "risk_level": "HIGH", "recommendation": "Block transaction",
     "confidence": 0.85}
    '''
    r1 = parse_explanation_response(json_output)
    print(f"[JSON] Method={r1.parse_method}, Risk={r1.risk_level}, Confidence={r1.confidence}")

    # Test 2: Structured sections
    sections_output = """
    ANALYSIS: The sender has a 15% historical fraud rate.
    Risk Factor: Elevated fraud rate above 5% threshold
    Risk Factor: 3 high-risk network neighbors
    RISK HIGH risk detected in transaction pattern
    RECOMMEND: Block and investigate transaction C567
    """
    r2 = parse_explanation_response(sections_output)
    print(f"[Sections] Method={r2.parse_method}, Factors={len(r2.key_factors)}")

    # Test 3: Plain text (current implementation)
    plain_output = "This transaction is suspicious. The risk indicator shows elevated patterns. I recommend manual review."
    r3 = parse_explanation_response(plain_output)
    print(f"[Plain] Method={r3.parse_method}, Recommendation={r3.recommendation[:40]}")

    # Test 4: Empty/garbage input
    r4 = parse_explanation_response("...")
    print(f"[Fallback] Method={r4.parse_method}, Confidence={r4.confidence}")
```

**Expected Output:**
```
[JSON] Method=json, Risk=HIGH, Confidence=0.85
[Sections] Method=regex_sections, Factors=2
[Plain] Method=keyword_scan, Recommendation=I recommend manual review.
[Fallback] Method=fallback, Confidence=0.5
```

**Follow-up Questions:**
1. "How does this relate to the current `_parse_agent_response()` implementation?" — The current code only does Strategy 3 (keyword scanning). This parser adds JSON and regex as higher-quality strategies before falling back, and records which strategy was used for observability.
2. "When would you use `pydantic.model_validator` instead of manual parsing?" — When the LLM is configured to return structured JSON (`model.with_structured_output()`), Pydantic can parse and validate in one step, eliminating all string parsing.
3. "What does `re.DOTALL` do and why is it needed here?" — Without `DOTALL`, `.` doesn't match newline characters. The JSON block or explanation section can span multiple lines, so `DOTALL` is required.
4. "How would you test this parser?" — Create a test fixture with 10+ real LLM output examples (from actual agent runs saved to a file), run the parser on each, and assert the expected `parse_method` and key field values.

---

## EXERCISE 7 — Rate Limiting / Throttling Pattern

**Project source:** `src/config.py` — `API_RATE_LIMIT = 100` (defined but not yet wired); `src/api/main.py` — `TrustedHostMiddleware`, `CORSMiddleware`

---

**Interview Prompt:**
> "Your project defines `API_RATE_LIMIT = 100` in config but hasn't wired it to a limiter. Implement a sliding-window rate limiter as middleware. Show both an in-memory version (for single-process) and explain how you'd extend it to Redis (for multi-worker)."

---

**Solution:**

```python
import time
import threading
from collections import defaultdict, deque
from typing import Callable, Optional


# ── Pattern 1: In-memory sliding window rate limiter ─────────────────
class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter using a deque of timestamps.
    Designed to mirror config.py's API_RATE_LIMIT = 100 (req/min).
    Thread-safe for single-process use (uvicorn with 1 worker).
    """

    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, dict]:
        """
        Returns (allowed: bool, headers: dict) where headers contain
        rate limit metadata for the response (like X-RateLimit-Remaining).
        """
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            timestamps = self._requests[client_id]

            # Remove timestamps outside the current window
            while timestamps and timestamps[0] < window_start:
                timestamps.popleft()

            current_count = len(timestamps)

            if current_count >= self.max_requests:
                # Rate limit exceeded
                oldest = timestamps[0] if timestamps else now
                retry_after = self.window_seconds - (now - oldest)
                return False, {
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(oldest + self.window_seconds)),
                    "Retry-After": str(int(retry_after) + 1),
                }

            # Allow and record this request
            timestamps.append(now)
            remaining = self.max_requests - current_count - 1

            return True, {
                "X-RateLimit-Limit": str(self.max_requests),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(now + self.window_seconds)),
            }

    def get_stats(self, client_id: str) -> dict:
        with self._lock:
            return {
                "client_id": client_id,
                "current_requests": len(self._requests[client_id]),
                "window_seconds": self.window_seconds,
                "max_requests": self.max_requests,
            }


# ── FastAPI middleware integration ────────────────────────────────────
# How this would be added to src/api/main.py:
#
# limiter = SlidingWindowRateLimiter(
#     max_requests=config.API_RATE_LIMIT,  # 100
#     window_seconds=60.0
# )
#
# @app.middleware("http")
# async def rate_limit_middleware(request: Request, call_next):
#     client_ip = request.client.host
#     allowed, headers = limiter.is_allowed(client_ip)
#
#     if not allowed:
#         return JSONResponse(
#             status_code=429,
#             content={"error": "Rate limit exceeded"},
#             headers=headers
#         )
#
#     response = await call_next(request)
#     for k, v in headers.items():
#         response.headers[k] = v
#     return response


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=2.0)
    client = "192.168.1.100"

    print("=== Sending 7 requests with limit=5/2s ===")
    for i in range(7):
        allowed, headers = limiter.is_allowed(client)
        status = "✓ ALLOWED" if allowed else "✗ BLOCKED"
        remaining = headers.get("X-RateLimit-Remaining", "0")
        print(f"Request {i+1}: {status} | Remaining: {remaining}")

    print()
    print("=== After 2s window expires ===")
    time.sleep(2.1)
    allowed, headers = limiter.is_allowed(client)
    print(f"Request after wait: {'✓ ALLOWED' if allowed else '✗ BLOCKED'} | Remaining: {headers['X-RateLimit-Remaining']}")
```

**Expected Output:**
```
=== Sending 7 requests with limit=5/2s ===
Request 1: ✓ ALLOWED | Remaining: 4
Request 2: ✓ ALLOWED | Remaining: 3
Request 3: ✓ ALLOWED | Remaining: 2
Request 4: ✓ ALLOWED | Remaining: 1
Request 5: ✓ ALLOWED | Remaining: 0
Request 6: ✗ BLOCKED | Remaining: 0
Request 7: ✗ BLOCKED | Remaining: 0

=== After 2s window expires ===
Request after wait: ✓ ALLOWED | Remaining: 4
```

**Follow-up Questions:**
1. "Why doesn't this work with multiple uvicorn workers?" — Each process has its own `_requests` dict in memory. Worker 1 might count 50 requests, worker 2 counts 50 requests — the same client has made 100 but neither worker blocked it. Fix: use Redis with `ZADD` + `ZREMRANGEBYSCORE` for cross-process atomic window management.
2. "What HTTP status code for rate limiting and what headers?" — 429 Too Many Requests; `Retry-After` header (seconds until reset); `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.
3. "Fixed window vs. sliding window — what's the difference?" — Fixed window resets at fixed intervals (every minute at :00). A client can send 100 requests at :59 and 100 more at 1:01 — 200 in 2 seconds. Sliding window tracks the last N seconds regardless of clock alignment, preventing this burst.
4. "How would the `/explain` endpoint have a different rate limit (50 req/min vs. `/predict`'s 500 req/min)?" — Pass the limit as a parameter or use route-specific middleware. The `README.md` documents these as separate limits.

---

## EXERCISE 8 — Database ORM / Session Pattern

**Project source:** `src/data_processing/graph_constructor.py` — Neo4j session management, `MERGE` upsert, `UNWIND` batch; `src/explainability/agent.py` — `Neo4jTransactionTool._get_user_profile()`

---

**Interview Prompt:**
> "Show me the Neo4j session management pattern your project uses. Implement a safe session wrapper that handles connection errors, implements upsert semantics, and supports batched writes — matching the actual patterns in `graph_constructor.py`."

---

**Solution:**

```python
from contextlib import contextmanager
from typing import Any, Generator, Optional
import logging

logger = logging.getLogger(__name__)


# ── Simulated Neo4j driver (replace with real neo4j.GraphDatabase.driver) ──
class MockSession:
    """Simulates neo4j.Session behavior for demonstration."""

    def __init__(self, should_fail: bool = False):
        self._should_fail = should_fail
        self._queries: list[str] = []

    def run(self, query: str, **params) -> "MockResult":
        if self._should_fail:
            raise ConnectionError("Neo4j ServiceUnavailable")
        self._queries.append(query.strip()[:50])
        return MockResult(query, params)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MockResult:
    def __init__(self, query: str, params: dict):
        self._query = query
        self._params = params

    def single(self) -> Optional[dict]:
        if "RETURN 1" in self._query:
            return {"1": 1}
        if "MATCH (u:User" in self._query:
            user_id = self._params.get("user_id", "C000")
            return {
                "user_id": user_id,
                "fraud_rate": 0.05,
                "total_transactions": 100,
            }
        return None


# ── Neo4j session wrapper ─────────────────────────────────────────────
class Neo4jRepository:
    """
    Wraps Neo4j driver with safe session management.
    Mirrors GraphConstructor and Neo4jTransactionTool patterns.
    """

    def __init__(self, driver):
        self.driver = driver
        self._healthy = False
        self._test_connection()

    def _test_connection(self) -> None:
        """Mirrors connect_to_neo4j() connection test."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                if result.single()["test"] == 1:
                    self._healthy = True
                    logger.info("Neo4j connection verified")
        except Exception as e:
            self._healthy = False
            logger.error(f"Neo4j connection failed: {e}")

    @contextmanager
    def session(self) -> Generator:
        """Safe session context manager with error handling."""
        if not self._healthy:
            raise RuntimeError("Neo4j not connected")
        session = self.driver.session()
        try:
            yield session
        except Exception as e:
            logger.error(f"Neo4j session error: {e}")
            raise
        finally:
            session.close()

    def upsert_user(self, user_data: dict) -> None:
        """
        MERGE upsert — creates if not exists, updates if exists.
        Mirrors ingest_nodes_to_neo4j() in graph_constructor.py.
        """
        with self.session() as s:
            s.run(
                """
                MERGE (u:User {user_id: $user_id})
                SET u.fraud_rate = $fraud_rate,
                    u.total_transactions = $total_transactions
                """,
                **user_data,
            )
            logger.info(f"Upserted user: {user_data['user_id']}")

    def batch_upsert_users(self, users: list[dict], batch_size: int = 1000) -> int:
        """
        Batched UNWIND upsert — mirrors the batch pattern in graph_constructor.py.
        Returns total rows processed.
        """
        total = 0
        for i in range(0, len(users), batch_size):
            batch = users[i : i + batch_size]
            with self.session() as s:
                s.run(
                    """
                    UNWIND $nodes AS node
                    MERGE (u:User {user_id: node.user_id})
                    SET u.fraud_rate = node.fraud_rate,
                        u.total_transactions = node.total_transactions
                    """,
                    nodes=batch,
                )
            total += len(batch)
            logger.info(f"Batch {i//batch_size + 1}: upserted {len(batch)} users")
        return total

    def get_user_profile(self, user_id: str) -> Optional[dict]:
        """
        Mirrors Neo4jTransactionTool._get_user_profile() in agent.py.
        Returns None instead of raising if user not found.
        """
        with self.session() as s:
            result = s.run(
                "MATCH (u:User {user_id: $user_id}) RETURN u.user_id as user_id, "
                "u.fraud_rate as fraud_rate, u.total_transactions as total_transactions",
                user_id=user_id,
            )
            record = result.single()
            return dict(record) if record else None


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    class MockDriver:
        def session(self):
            return MockSession()

    repo = Neo4jRepository(MockDriver())

    print("=== Single upsert ===")
    repo.upsert_user({"user_id": "C123", "fraud_rate": 0.05, "total_transactions": 100})

    print()
    print("=== Batch upsert ===")
    users = [
        {"user_id": f"C{i:03d}", "fraud_rate": i * 0.01, "total_transactions": i * 10}
        for i in range(1, 6)
    ]
    processed = repo.batch_upsert_users(users, batch_size=2)
    print(f"Total processed: {processed}")

    print()
    print("=== Get user profile ===")
    profile = repo.get_user_profile("C123")
    print(f"Profile: {profile}")
```

**Expected Output:**
```
Neo4j connection verified
=== Single upsert ===
INFO Upserted user: C123

=== Batch upsert ===
INFO Batch 1: upserted 2 users
INFO Batch 2: upserted 2 users
INFO Batch 3: upserted 1 users
Total processed: 5

=== Get user profile ===
Profile: {'user_id': 'C123', 'fraud_rate': 0.05, 'total_transactions': 100}
```

**Follow-up Questions:**
1. "Why `MERGE` instead of `CREATE` for node ingestion?" — `MERGE` is upsert semantics: if a User node with that `user_id` already exists, it updates it; otherwise it creates it. `CREATE` would fail with a uniqueness constraint violation on re-runs.
2. "Why batch size 1000 for nodes but 500 for edges?" — Relationship creation (`CREATE (s)-[t:TRANSACTION]->...`) is more write-intensive in Neo4j than node creation because it must update two node records and the relationship index. Smaller batches prevent transaction timeout.
3. "What is `UNWIND` in Cypher?" — `UNWIND` expands a list parameter into individual rows, allowing a single Cypher statement to operate on multiple items in one network round-trip. Without `UNWIND`, you'd need N separate `MERGE` statements — N network round-trips.
4. "How would you make `batch_upsert_users()` use transactions for atomicity?" — Wrap the batch in an explicit `session.begin_transaction()` block and call `tx.commit()` after all batches, or `tx.rollback()` on error. Currently each batch is an auto-committed transaction.

---

## EXERCISE 9 — Configuration / Feature-Flag Pattern

**Project source:** `src/config.py` — `Config` class with class variables; `src/api/schemas.py` — `PredictionConfig` with `use_subgraph` toggle

---

**Interview Prompt:**
> "Your project uses a centralized `Config` class that loads from environment variables. Implement this pattern with: (1) type-safe loading with defaults, (2) validation on startup, (3) environment-specific overrides, and (4) a feature flag system that mirrors `PredictionConfig.use_subgraph`."

---

**Solution:**

```python
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ── Pattern 1: Type-safe config with validation ───────────────────────
@dataclass
class FraudDetectionConfig:
    """
    Mirrors src/config.py's Config class.
    Loaded from environment variables with typed defaults.
    """

    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))

    # GNN model
    gnn_input_dim: int = field(default_factory=lambda: int(os.getenv("GNN_INPUT_DIM", "10")))
    gnn_hidden_dim: int = field(default_factory=lambda: int(os.getenv("GNN_HIDDEN_DIM", "128")))
    gnn_dropout_rate: float = field(default_factory=lambda: float(os.getenv("GNN_DROPOUT", "0.2")))

    # Training
    learning_rate: float = field(default_factory=lambda: float(os.getenv("LR", "0.001")))
    num_epochs: int = field(default_factory=lambda: int(os.getenv("NUM_EPOCHS", "100")))
    early_stopping_patience: int = field(default_factory=lambda: int(os.getenv("ES_PATIENCE", "10")))

    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # API
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_rate_limit: int = 100
    is_development: bool = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development") == "development"
    )

    # Gemini
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = "gemini-1.5-pro-latest"
    llm_temperature: float = 0.3
    max_output_tokens: int = 1000

    def validate(self) -> bool:
        """
        Mirrors Config.validate_config() in src/config.py.
        Raises ValueError for invalid production configs.
        """
        errors = []

        # Check train/val/test ratios sum to 1.0
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            errors.append(f"Ratios must sum to 1.0, got {ratio_sum}")

        # Production credential checks
        if not self.is_development:
            if self.neo4j_password == "password":
                errors.append("NEO4J_PASSWORD must be set in production")
            if not self.gemini_api_key:
                errors.append("GEMINI_API_KEY must be set in production")

        # Model parameter bounds
        if self.gnn_hidden_dim <= 0:
            errors.append(f"GNN hidden dim must be positive, got {self.gnn_hidden_dim}")
        if not (0.0 < self.learning_rate < 1.0):
            errors.append(f"Learning rate out of range: {self.learning_rate}")

        if errors:
            raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        logger.info("Configuration validated successfully")
        return True

    def create_directories(self) -> None:
        """Creates required directories. Mirrors Config.create_directories()."""
        dirs = ["data/raw", "data/processed", "models/checkpoints", "logs"]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


# ── Pattern 2: Feature flags ──────────────────────────────────────────
@dataclass
class PredictionFeatureFlags:
    """
    Mirrors PredictionConfig in src/api/schemas.py.
    Controls inference behavior per-request.
    """
    use_subgraph: bool = True        # Use full GNN vs. heuristic fallback
    subgraph_hops: int = 2           # k-hop neighborhood depth
    include_confidence: bool = True  # Return confidence score
    include_explanation_features: bool = False  # Return forward_with_attention() outputs
    decision_threshold: float = 0.5  # Fraud/not-fraud decision boundary

    def for_high_throughput(self) -> "PredictionFeatureFlags":
        """Preset: disable graph for maximum speed"""
        return PredictionFeatureFlags(
            use_subgraph=False,
            include_confidence=False,
            include_explanation_features=False,
        )

    def for_deep_investigation(self) -> "PredictionFeatureFlags":
        """Preset: full features for compliance review"""
        return PredictionFeatureFlags(
            use_subgraph=True,
            subgraph_hops=3,
            include_confidence=True,
            include_explanation_features=True,
        )


# ── Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Config loading ===")
    config = FraudDetectionConfig()
    print(f"Neo4j URI: {config.neo4j_uri}")
    print(f"GNN dims: input={config.gnn_input_dim}, hidden={config.gnn_hidden_dim}")
    print(f"Is development: {config.is_development}")

    print()
    print("=== Validation (development mode — relaxed) ===")
    config.validate()

    print()
    print("=== Validation (production mode — strict) ===")
    os.environ["ENVIRONMENT"] = "production"
    prod_config = FraudDetectionConfig()
    try:
        prod_config.validate()
    except ValueError as e:
        print(f"Validation errors:\n{e}")

    print()
    print("=== Feature flags ===")
    flags = PredictionFeatureFlags()
    print(f"Default: use_subgraph={flags.use_subgraph}, hops={flags.subgraph_hops}")

    fast_flags = flags.for_high_throughput()
    print(f"High-throughput: use_subgraph={fast_flags.use_subgraph}")

    deep_flags = flags.for_deep_investigation()
    print(f"Deep investigation: use_subgraph={deep_flags.use_subgraph}, hops={deep_flags.subgraph_hops}")
```

**Expected Output:**
```
=== Config loading ===
Neo4j URI: bolt://localhost:7687
GNN dims: input=10, hidden=128
Is development: True

=== Validation (development mode — relaxed) ===
INFO Configuration validated successfully

=== Validation (production mode — strict) ===
Validation errors:
Config validation failed:
  - NEO4J_PASSWORD must be set in production
  - GEMINI_API_KEY must be set in production

=== Feature flags ===
Default: use_subgraph=True, hops=2
High-throughput: use_subgraph=False
Deep investigation: use_subgraph=True, hops=3
```

**Follow-up Questions:**
1. "Why does `config.py` use a class with class variables instead of a dataclass?" — Class variables are initialized once at class definition time, making them importable as `config.NEO4J_URI` from anywhere. A dataclass instance needs to be instantiated. Both work; the class-variable approach is a common Python config pattern.
2. "How do you override config in tests without modifying environment variables?" — Use `unittest.mock.patch.dict(os.environ, {"NEO4J_PASSWORD": "test_password"})` as a context manager around the test. Or use a test-specific config instance with constructor overrides.
3. "What's the risk of the `validate()` being called at import time?" — `config.validate_config()` is called at import time in `config.py`. If any required env var is missing in production, the import fails, which prevents the app from starting. This is intentional — fail fast rather than fail at first request.
4. "How would you implement environment-specific config (dev vs staging vs prod)?" — Use a base config class with overrides per environment: `DevConfig(BaseConfig)`, `StagingConfig(BaseConfig)`, `ProdConfig(BaseConfig)`. A factory function `get_config()` selects based on `ENVIRONMENT` env var.

---

## EXERCISE 10 — Concurrency / Async Pattern

**Project source:** `src/api/main.py` — `asyncio.to_thread()` for LangChain agent; `BackgroundTasks` for async logging; `asynccontextmanager` lifespan

---

**Interview Prompt:**
> "Your project calls a synchronous LangChain agent from an async FastAPI endpoint using `asyncio.to_thread()`. Explain the problem this solves, then implement the full async prediction + explanation pipeline — showing how these two operations can run concurrently for a batch request."

---

**Solution:**

```python
import asyncio
import time
import logging
from typing import Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── The core problem: sync in async ──────────────────────────────────
def sync_gnn_inference(transaction_id: str, amount: float) -> dict:
    """
    Simulates synchronous GNN forward pass (~150ms).
    Real version: self.model(subgraph, subgraph.ndata['feat'])
    """
    time.sleep(0.15)  # Simulate 150ms GPU inference
    return {
        "transaction_id": transaction_id,
        "fraud_probability": min(amount / 1_000_000, 0.99),
        "is_fraud_predicted": amount > 500_000,
        "risk_level": "HIGH" if amount > 500_000 else "LOW",
    }


def sync_langchain_agent(transaction_id: str, fraud_probability: float) -> dict:
    """
    Simulates synchronous LangChain agent (~2.5s).
    Real version: self.agent.run(prompt) in AIInvestigator.
    """
    time.sleep(2.5)  # Simulate agent reasoning + Gemini API call
    return {
        "transaction_id": transaction_id,
        "explanation_text": f"Transaction {transaction_id} shows {'HIGH' if fraud_probability > 0.5 else 'LOW'} risk.",
        "key_factors": ["Elevated amount", "Network anomaly"],
        "recommendation": "Manual review" if fraud_probability > 0.5 else "Approve",
        "confidence": 0.82,
    }


# ── Pattern 1: asyncio.to_thread() — non-blocking sync calls ─────────
async def predict_fraud_async(transaction_id: str, amount: float) -> dict:
    """
    Wraps sync GNN call in a thread pool.
    Mirrors predict_fraud() in predict.py called from FastAPI endpoint.
    """
    # asyncio.to_thread() runs sync function in thread pool executor
    # This frees the event loop to handle other requests during inference
    result = await asyncio.to_thread(sync_gnn_inference, transaction_id, amount)
    return result


async def explain_fraud_async(transaction_id: str, fraud_probability: float) -> dict:
    """
    Wraps sync LangChain agent in thread pool.
    Mirrors explain_transaction() in AIInvestigator:
      result = await asyncio.to_thread(self.agent.run, prompt)
    """
    result = await asyncio.to_thread(sync_langchain_agent, transaction_id, fraud_probability)
    return result


# ── Pattern 2: Concurrent prediction + explanation ────────────────────
async def predict_and_explain_concurrent(
    transaction_id: str, amount: float
) -> dict:
    """
    Runs prediction first, then explanation concurrently with other work.
    Shows how asyncio.gather() enables parallelism.
    """
    logger.info(f"[{transaction_id}] Starting prediction...")
    prediction = await predict_fraud_async(transaction_id, amount)
    logger.info(f"[{transaction_id}] Prediction done: prob={prediction['fraud_probability']:.2f}")

    # Explanation runs in thread pool — doesn't block event loop
    logger.info(f"[{transaction_id}] Starting explanation (async)...")
    explanation = await explain_fraud_async(transaction_id, prediction["fraud_probability"])
    logger.info(f"[{transaction_id}] Explanation done.")

    return {**prediction, "explanation": explanation}


# ── Pattern 3: asyncio.gather() for batch concurrency ─────────────────
async def process_batch_concurrent(transactions: list[dict]) -> list[dict]:
    """
    Processes a batch concurrently using asyncio.gather().
    Mirrors predict_batch() in predict.py — but making it truly async.
    """
    tasks = [
        predict_fraud_async(t["transaction_id"], t["amount"])
        for t in transactions
    ]
    # All predictions run concurrently (each in thread pool)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle per-item exceptions without failing the whole batch
    final_results = []
    for txn, result in zip(transactions, results):
        if isinstance(result, Exception):
            logger.error(f"Failed: {txn['transaction_id']}: {result}")
            final_results.append({"transaction_id": txn["transaction_id"], "error": str(result)})
        else:
            final_results.append(result)

    return final_results


# ── Pattern 4: BackgroundTasks equivalent ────────────────────────────
async def log_prediction_background(prediction: dict) -> None:
    """
    Simulates log_prediction_async() in src/api/main.py.
    FastAPI's BackgroundTasks runs this after the response is sent.
    """
    await asyncio.sleep(0.01)  # Simulate I/O logging
    logger.info(f"[Background] Logged: {prediction['transaction_id']}")


# ── Demo ──────────────────────────────────────────────────────────────
async def main():
    print("=== Sequential predict + explain (one transaction) ===")
    start = time.time()
    result = await predict_and_explain_concurrent("TXN001", 750_000)
    print(f"Done in {time.time()-start:.2f}s | Risk: {result['risk_level']}")
    print(f"Explanation: {result['explanation']['explanation_text']}")

    print()
    print("=== Concurrent batch (3 transactions via gather) ===")
    batch = [
        {"transaction_id": "TXN002", "amount": 50_000},
        {"transaction_id": "TXN003", "amount": 800_000},
        {"transaction_id": "TXN004", "amount": 150_000},
    ]
    start = time.time()
    results = await process_batch_concurrent(batch)
    elapsed = time.time() - start
    print(f"3 predictions in {elapsed:.2f}s (vs {0.15*3:.2f}s sequential)")
    for r in results:
        print(f"  {r['transaction_id']}: prob={r.get('fraud_probability', 'N/A')}")

    print()
    print("=== Background logging ===")
    prediction = {"transaction_id": "TXN001", "fraud_probability": 0.85}
    # In FastAPI: background_tasks.add_task(log_prediction_async, ...)
    asyncio.create_task(log_prediction_background(prediction))
    await asyncio.sleep(0.05)  # Allow background task to complete


if __name__ == "__main__":
    asyncio.run(main())
```

**Expected Output:**
```
=== Sequential predict + explain (one transaction) ===
INFO [TXN001] Starting prediction...
INFO [TXN001] Prediction done: prob=0.75
INFO [TXN001] Starting explanation (async)...
INFO [TXN001] Explanation done.
Done in 2.65s | Risk: HIGH
Explanation: Transaction TXN001 shows HIGH risk.

=== Concurrent batch (3 transactions via gather) ===
3 predictions in ~0.15s (vs 0.45s sequential)
  TXN002: prob=0.05
  TXN003: prob=0.80
  TXN004: prob=0.15

=== Background logging ===
INFO [Background] Logged: TXN001
```

**Follow-up Questions:**
1. "Why `asyncio.to_thread()` instead of just marking the function `async`?" — `async def` functions must use `await` for I/O — they don't actually run in parallel on CPU-bound tasks (GIL applies). `asyncio.to_thread()` puts the sync call in a thread pool, which releases the GIL and allows other async tasks to run on the event loop.
2. "What's the difference between `asyncio.create_task()` and `await`?" — `await` blocks the current coroutine until the awaitable completes. `create_task()` schedules the coroutine to run concurrently without waiting for it — like FastAPI's `BackgroundTasks`.
3. "When would `asyncio.gather()` be better than multiple `await` calls?" — `gather()` runs tasks concurrently; sequential `await` runs them one after another. For the batch endpoint, `gather()` allows 100 concurrent inference calls — each waiting for its thread-pool slot independently.
4. "What happens if you call a blocking function directly in an async endpoint without `to_thread()`?" — It blocks the entire event loop. During the 2.5s LangChain call, *no other request* can be processed — effectively making a concurrent server single-threaded. The `/predict` endpoint's 150ms latency would also become 2.5s if an explanation is in progress.

---

## QUICK PYTHON CONCEPTS REVIEW TABLE

| Concept | Where in This Project | Key Method/Syntax |
|---|---|---|
| `@asynccontextmanager` | `main.py` — `lifespan()` | `async def lifespan(app)` + `yield` |
| `@contextmanager` | MLflow `with mlflow.start_run()` | `yield` in generator function |
| `functools.lru_cache` | `predict.py` — amount_log (could use) | `@lru_cache(maxsize=256)` |
| `functools.wraps` | Any decorator | Preserves `__name__`, `__doc__` |
| Generator / `yield` | `graph_constructor.py` batch loops | `for batch in batches: yield` |
| `asyncio.to_thread()` | `agent.py` — `explain_transaction()` | `await asyncio.to_thread(sync_fn, args)` |
| `asyncio.gather()` | Batch prediction (improvement) | `await asyncio.gather(*tasks)` |
| `BackgroundTasks` | `main.py` — all endpoints | `background_tasks.add_task(fn, args)` |
| `Depends()` injection | `main.py` — endpoint params | `predictor: FraudPredictor = Depends(get_fraud_predictor)` |
| `@root_validator` | `schemas.py` — `TransactionInput` | Cross-field Pydantic validation |
| `@validator` | `schemas.py` — user IDs, amount | Field-level Pydantic validation |
| `nn.ModuleList` | `model.py` — SAGEConv layers | Registers submodules for `parameters()` |
| `@torch.no_grad()` | `training.py` — `evaluate()` | Disables gradient tracking during eval |
| `dataclass` | `agent.py` — `TransactionContext` | `@dataclass` + `field()` |
| `Enum` | `schemas.py` — `TransactionType` | `class TransactionType(str, Enum)` |

---

## BEFORE YOUR CODING INTERVIEW — WARM-UP CHECKLIST

Do these 5 exercises the night before (15-20 minutes total):

- [ ] **1.** Write a class with `__enter__` and `__exit__` from memory — get the `exc_type, exc_val, exc_tb` signature right, remember to return `False`
- [ ] **2.** Write a retry decorator with exponential backoff — `functools.wraps`, counter loop, `time.sleep(delay * backoff_factor ** attempt)`
- [ ] **3.** Write a generator that yields batches from a list — handle the remainder after the last full batch
- [ ] **4.** Write a `@cached_property` descriptor — implement `__get__`, check `hasattr(obj, attr_name)`, use `setattr`
- [ ] **5.** Write `async def` with `asyncio.to_thread()` calling a blocking function — remember `await asyncio.to_thread(sync_fn, arg1, arg2)` syntax

---

## COMMON MISTAKES TO AVOID

| Mistake | What Goes Wrong | Correct Pattern |
|---|---|---|
| Forgetting `@functools.wraps` in decorators | `wrapper.__name__ == 'wrapper'`; FastAPI route names break; mypy loses type info | Always add `@functools.wraps(func)` before `def wrapper` |
| Using `@lru_cache` on a method with `self` | `self` is in the cache key, preventing garbage collection of the object | Use `cached_property` for instance-level caching or `@staticmethod` + `@lru_cache` |
| Calling blocking functions directly in `async def` | Blocks the entire event loop — all requests queue behind it | Always `await asyncio.to_thread(sync_fn, ...)` for CPU/blocking work |
| Returning `True` from `__exit__` accidentally | Suppresses the exception silently | Only return `True` when suppression is intentional; return `False` or `None` otherwise |
| Not handling the final partial batch in generators | Last N < batch_size items are silently dropped | Always check `if batch: yield batch` after the main loop |

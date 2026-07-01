# Assist Phase 1 — De-Docker the Runtime — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the app boot and run on Windows with zero containers — embedded ChromaDB, keyless default web search, and notifications degraded to an optional integration.

**Architecture:** Three isolated changes behind existing seams. The vector store swaps `chromadb.HttpClient` for an embedded `chromadb.PersistentClient` inside the single `get_chroma_client()` seam. Web search flips its default provider to the already-implemented keyless DuckDuckGo. Service health already treats a non-active SearXNG and an unconfigured ntfy as `disabled`, so the final task is a regression test proving the no-container defaults report healthy.

**Tech Stack:** Python 3.14, FastAPI/uvicorn, ChromaDB (embedded), fastembed, pytest / pytest-asyncio.

## Global Constraints

- **No new container dependencies.** Nothing added in this phase may require a running chromadb/searxng/ntfy service.
- **Backward compatible.** Existing Docker deployments that set `CHROMADB_HOST` / `search_provider=searxng` must keep working unchanged.
- **Rename deferred.** Do NOT rename `ODYSSEUS_*` env vars, modules, or admin defaults in this phase (Phase 4).
- **Follow the test taxonomy** in `tests/README.md`; new tests live under `tests/` and run in the fast lane (`pytest -m "not slow"`).
- **Data dir:** the embedded Chroma path lives under `DATA_DIR` (from `src/constants.py`), overridable via `CHROMADB_PATH`.

---

### Task 1: Embedded ChromaDB vector store

Swap the standalone-server client for an embedded `PersistentClient` when no external ChromaDB is configured, keeping the HTTP path as an opt-in for existing Docker users.

**Files:**
- Modify: `src/chroma_client.py`
- Modify: `requirements.txt:18` (`chromadb-client` → `chromadb`)
- Test: `tests/test_chroma_client.py`

**Interfaces:**
- Consumes: `src.constants.DATA_DIR` (str, absolute path to the data directory).
- Produces: `get_chroma_client()` returns a live, heartbeat-verified Chroma client — an embedded `PersistentClient` when `CHROMADB_HOST` is unset/empty, or the existing `HttpClient` when `CHROMADB_HOST` is set. `reset_client()` clears the singleton. Signatures unchanged; consumers (`src/rag_singleton.py`, `src/embedding_lanes.py`) are untouched.

- [ ] **Step 1: Write the failing test for embedded mode**

Add to `tests/test_chroma_client.py`:

```python
def test_get_chroma_client_uses_embedded_when_no_host(monkeypatch, tmp_path):
    pytest.importorskip("chromadb")
    import chromadb
    cc.reset_client()
    monkeypatch.delenv("CHROMADB_HOST", raising=False)
    monkeypatch.setenv("CHROMADB_PATH", str(tmp_path / "chroma"))
    client = cc.get_chroma_client()
    # Embedded client is a ClientAPI backed by a local path, NOT an HTTP client.
    assert isinstance(client, chromadb.api.ClientAPI)
    # It works fully offline: create a collection and read it back.
    col = client.get_or_create_collection("t1")
    assert col.name == "t1"
    # The persistent directory was created under the configured path.
    assert (tmp_path / "chroma").is_dir()
    cc.reset_client()


def test_get_chroma_client_uses_http_when_host_set(monkeypatch):
    pytest.importorskip("chromadb")
    cc.reset_client()
    monkeypatch.setenv("CHROMADB_HOST", "127.0.0.1")
    monkeypatch.setenv("CHROMADB_PORT", str(_free_port()))
    # Host is set but nothing is listening → HTTP path still fast-fails, proving
    # the embedded branch was NOT taken.
    with pytest.raises(RuntimeError):
        cc.get_chroma_client()
    assert cc._client is None
    cc.reset_client()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_chroma_client.py::test_get_chroma_client_uses_embedded_when_no_host -v`
Expected: FAIL — current code takes the HTTP path (`RuntimeError: ChromaDB is not reachable ...`) instead of returning an embedded client. (If `chromadb` full package isn't installed yet, it SKIPs — install it in Step 3 first, then re-run.)

- [ ] **Step 3: Swap the dependency and install it**

In `requirements.txt`, change line 18 from:

```
chromadb-client
```

to:

```
# Full (not -client) chromadb so the app can run an embedded PersistentClient
# with no standalone server — required for the native, container-free build.
# Still talks to a remote server when CHROMADB_HOST is set (Docker path).
chromadb
```

Then install into the working environment:

Run: `pip install chromadb`

- [ ] **Step 4: Implement the embedded/HTTP branch**

Replace the body of `get_chroma_client()` in `src/chroma_client.py` (lines 31-67) with:

```python
def get_chroma_client():
    """Get or create the singleton ChromaDB client.

    With no external service configured (``CHROMADB_HOST`` unset/empty) this
    returns an *embedded* ``PersistentClient`` writing under ``CHROMADB_PATH``
    (default ``<DATA_DIR>/chroma``) — no server, no socket. When ``CHROMADB_HOST``
    is set it behaves as before: a fast-failing ``HttpClient`` to a standalone
    ChromaDB service (the Docker path).

    Raises RuntimeError with a clear install hint if the `chromadb` package is
    not installed.
    """
    global _client
    if _client is not None:
        return _client

    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError(
            "ChromaDB integration is not installed. Install it with: "
            "pip install chromadb"
        ) from e

    host = (os.getenv("CHROMADB_HOST") or "").strip()

    if not host:
        # Embedded mode: local on-disk store, no standalone service required.
        from src.constants import DATA_DIR
        path = (os.getenv("CHROMADB_PATH") or "").strip() or os.path.join(DATA_DIR, "chroma")
        os.makedirs(path, exist_ok=True)
        client = chromadb.PersistentClient(path=path)
        client.heartbeat()
        _client = client
        logger.info(f"ChromaDB embedded (persistent) at: {path}")
        return _client

    # HTTP mode: talk to a standalone ChromaDB service (Docker/self-hosted).
    port = int(os.getenv("CHROMADB_PORT", "8100"))
    if not _port_open(host, port):
        raise RuntimeError(
            f"ChromaDB is not reachable at {host}:{port}. Start the ChromaDB "
            f"service (e.g. `docker compose up chromadb`) or set CHROMADB_HOST / "
            f"CHROMADB_PORT to point at a running instance."
        )
    client = chromadb.HttpClient(host=host, port=port)
    # Health check before caching — if the port is open but the service isn't
    # healthy yet (e.g. still starting), don't poison the singleton with a dead
    # client; leave _client unset so the next call retries.
    client.heartbeat()
    _client = client
    logger.info(f"ChromaDB connected: {host}:{port}")
    return _client
```

Leave `_port_open`, `_CONNECT_TIMEOUT`, and `reset_client()` unchanged. Update the module docstring's first lines (1-6) to read:

```python
"""
chroma_client.py

Singleton ChromaDB client. Embedded (on-disk PersistentClient) by default;
connects to a standalone ChromaDB service when CHROMADB_HOST is set.
"""
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_chroma_client.py -v`
Expected: PASS — all four tests (two existing `_port_open`/no-cache tests plus the two new ones).

- [ ] **Step 6: Commit**

```bash
git add src/chroma_client.py requirements.txt tests/test_chroma_client.py
git commit -m "feat(chroma): embedded PersistentClient when no CHROMADB_HOST set"
```

---

### Task 2: Keyless default web search provider

Flip the default search provider from `searxng` (needs a container) to the already-implemented keyless `duckduckgo`, so a fresh install searches the web with no API key and no service.

**Files:**
- Modify: `src/settings.py:59`
- Test: `tests/test_search_default_provider.py` (create)

**Interfaces:**
- Consumes: `src.settings.DEFAULT_SETTINGS` / `load_settings()` (returns a settings dict including `"search_provider"`).
- Produces: default `search_provider == "duckduckgo"`. SearXNG remains selectable when a user sets `search_provider="searxng"` and a `search_url`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_search_default_provider.py`:

```python
"""The out-of-the-box search provider must need no API key and no service,
so a fresh native (container-free) install can search the web immediately.
"""
from src.settings import DEFAULT_SETTINGS
from services.search.providers import PROVIDER_INFO


def test_default_search_provider_is_keyless_and_serviceless():
    provider = DEFAULT_SETTINGS["search_provider"]
    label, needs_key, needs_url = PROVIDER_INFO[provider]
    assert needs_key is False, f"default provider {provider!r} requires an API key"
    assert needs_url is False, f"default provider {provider!r} requires a service URL"


def test_default_search_provider_is_duckduckgo():
    assert DEFAULT_SETTINGS["search_provider"] == "duckduckgo"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_search_default_provider.py -v`
Expected: FAIL — default is currently `"searxng"`, which has `needs_url=True` (first test fails; second asserts the new value).

- [ ] **Step 3: Change the default**

In `src/settings.py`, change line 59 from:

```python
    "search_provider": "searxng",
```

to:

```python
    # Keyless, serviceless default so a fresh (container-free) install can
    # search the web with no setup. SearXNG stays selectable via Settings for
    # anyone pointing at their own instance (set search_provider + search_url).
    "search_provider": "duckduckgo",
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_search_default_provider.py -v`
Expected: PASS — both tests.

- [ ] **Step 5: Guard against provider-list regressions**

Run the existing search suites to confirm nothing depended on the old default:

Run: `pytest tests/test_service_search_provider_guards.py tests/test_search_module_consolidation.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/settings.py tests/test_search_default_provider.py
git commit -m "feat(search): default to keyless DuckDuckGo instead of SearXNG"
```

---

### Task 3: No-container defaults report healthy

Lock in, with a regression test, that with no chromadb/searxng/ntfy services running and the new keyless defaults, the consolidated service-health report is healthy (SearXNG and ntfy show `disabled`, which never drags `overall` down). No production code should be needed — if a test fails, that reveals a real hard-requirement to fix.

**Files:**
- Test: `tests/test_service_health_no_containers.py` (create)

**Interfaces:**
- Consumes: `src.service_health.searxng_health(settings)`, `ntfy_health(integrations, settings)`, `collect_service_health(rag_manager, memory_vector)`; `src.settings.DEFAULT_SETTINGS`. Status constants `OK`, `DISABLED`. The health rollup excludes `disabled` from the overall verdict (`src/service_health.py:50-52`).
- Produces: regression coverage only; no exported symbols.

- [ ] **Step 1: Write the failing/guard test**

Create `tests/test_service_health_no_containers.py`:

```python
"""Phase 1 (de-Docker): with the keyless defaults and no chromadb/searxng/ntfy
services running, the consolidated health report must be healthy — SearXNG and
ntfy report `disabled`, which is excluded from the overall verdict.
"""
import pytest

import src.service_health as sh
from src.settings import DEFAULT_SETTINGS


def test_searxng_disabled_under_default_provider():
    # Default provider is no longer searxng, so its probe self-disables and
    # never performs a network call.
    result = sh.searxng_health(dict(DEFAULT_SETTINGS))
    assert result["status"] == sh.DISABLED


def test_ntfy_disabled_with_no_integration():
    result = sh.ntfy_health([], dict(DEFAULT_SETTINGS))
    assert result["status"] == sh.DISABLED


@pytest.mark.asyncio
async def test_overall_health_ok_with_no_containers(monkeypatch):
    # No settings/integrations/accounts/endpoints and no vector managers:
    # nothing configured means nothing can be "down".
    monkeypatch.setattr(sh, "_gather_inputs", lambda: {
        "settings": dict(DEFAULT_SETTINGS),
        "integrations": [],
        "accounts": [],
        "endpoints": [],
    })
    report = await sh.collect_service_health(rag_manager=None, memory_vector=None)
    assert report["overall"] == sh.OK
    statuses = {s["name"]: s["status"] for s in report["services"]}
    assert statuses["searxng"] == sh.DISABLED
    assert statuses["ntfy"] == sh.DISABLED
    assert statuses["chromadb"] == sh.DISABLED  # no vector managers passed
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_service_health_no_containers.py -v`
Expected: PASS immediately (the health module already treats non-active SearXNG and unconfigured ntfy as `disabled`). If any test FAILS, that is a genuine hard-requirement on a container — stop and fix the offending code path so the absent service reports `disabled`/`ok`, not `down`, then re-run.

- [ ] **Step 3: Full fast-lane regression**

Run: `pytest -m "not slow" -q`
Expected: PASS (no regressions from Tasks 1-3).

- [ ] **Step 4: Commit**

```bash
git add tests/test_service_health_no_containers.py
git commit -m "test(health): assert no-container defaults report healthy"
```

---

## Self-Review

**Spec coverage:**
- Spec Component 1 (embedded vector store) → Task 1. ✓
- Spec Component 2 (keyless search default) → Task 2. ✓
- Spec Component 3 (notifications optional) → Task 3 (ntfy `disabled` guard). ✓
- Spec Component 4 (config & entry defaults): `DATABASE_URL` already SQLite (no change); embedded Chroma path under `DATA_DIR` with `CHROMADB_PATH` override → Task 1 Step 4. The native uvicorn *launcher* is explicitly Phase 2, so no launcher task here. ✓
- Spec Testing section → each task's test steps + Task 3 fast-lane run. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows full code. ✓

**Type consistency:** `get_chroma_client()` / `reset_client()` / `_port_open` / `_client` used consistently with `src/chroma_client.py`. Health symbols `OK`/`DISABLED`/`searxng_health`/`ntfy_health`/`collect_service_health`/`_gather_inputs` match `src/service_health.py`. `PROVIDER_INFO` tuple shape `(label, needs_key, needs_url)` matches `services/search/providers.py:19-27`. `DEFAULT_SETTINGS` key `search_provider` matches `src/settings.py`. ✓

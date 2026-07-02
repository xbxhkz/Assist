# Assist Phase 3b — Dynamic HF Catalog + Native GGUF Downloader — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search Hugging Face live for GGUF models and stream-download a chosen file into `MODELS_DIR` (one at a time, atomic, cancellable), where it appears in the 3a Local Models list ready to Serve.

**Architecture:** Extend the `src/localmodels/` package with `catalog.py` (HF search + file listing, network behind an injectable `get_json`) and `downloader.py` (`DownloadManager` streaming transfer with an injectable `http_stream`/`spawn` so progress, cancel, and atomic rename are unit-testable). Extend the admin `localmodels` routes and the Local Models modal UI.

**Tech Stack:** Python 3.14, FastAPI, httpx (core dep — no new deps), threading, vanilla JS, pytest.

## Global Constraints

- **New subsystem, Cookbook untouched:** do NOT modify `routes/cookbook_*`, `src/cookbook_*`, or `services/hwfit/`.
- **No new dependency:** stream with the existing `httpx`.
- **One download at a time.**
- **Security:** the download route accepts only URLs whose host is `huggingface.co` and a `filename` that is a safe `.gguf` **basename** (no separators/traversal). Partial downloads use a `.part` suffix and are atomically renamed to `.gguf` only on success, so `list_gguf_models` never sees a partial.
- **HF auth:** attach `Authorization: Bearer <token>` from `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` (env) when set; otherwise no auth header. Read env directly — do NOT import Cookbook.
- **Admin-guarded** via the existing router dependency.
- **No internal module / `ODYSSEUS_*` renames.**
- **Test env:** pytest with `--import-mode=importlib`; new tests carry no `slow` marker.
- **3c note:** hardware-aware ranking of results is a REQUIRED 3c deliverable, NOT part of 3b.

## File Structure

- `src/localmodels/catalog.py` (new) — `search_gguf_models`, `list_repo_gguf_files`, `_hf_headers`. Task 1.
- `src/localmodels/downloader.py` (new) — `DownloadManager`, `get_download_manager`, `_safe_filename`. Task 2.
- `routes/localmodels_routes.py` (extend) — catalog + download routes. Task 3.
- `static/js/localModels.js` (extend) + `static/index.html` (extend) — search + download UI. Task 4.
- Tests: `tests/test_localmodels_catalog.py`, `tests/test_localmodels_downloader.py`, `tests/test_localmodels_download_routes.py`, `tests/test_localmodels_download_ui.py`.

---

### Task 1: HF catalog service

**Files:**
- Create: `src/localmodels/catalog.py`
- Test: `tests/test_localmodels_catalog.py`

**Interfaces:**
- Produces:
  - `search_gguf_models(query: str = "", sort: str = "downloads", limit: int = 30, get_json=None) -> list[dict]` → `[{"repo","downloads","likes"}]`
  - `list_repo_gguf_files(repo: str, get_json=None) -> list[dict]` → `[{"filename","size","url"}]`
  - `_hf_headers() -> dict` (bearer header when a token env var is set)

- [ ] **Step 1: Write failing tests**

Create `tests/test_localmodels_catalog.py`:

```python
"""Unit tests for the HF GGUF catalog service (network injected)."""
import src.localmodels.catalog as cat


def test_search_builds_url_and_parses(monkeypatch):
    seen = {}
    def fake_get_json(url, headers):
        seen["url"] = url
        return [
            {"id": "TheBloke/foo-GGUF", "downloads": 10, "likes": 2},
            {"modelId": "bar/baz-gguf", "downloads": 5},
            {"downloads": 1},  # no id → skipped
        ]
    out = cat.search_gguf_models("qwen", sort="downloads", limit=7, get_json=fake_get_json)
    assert "filter=gguf" in seen["url"]
    assert "search=qwen" in seen["url"]
    assert "limit=7" in seen["url"]
    assert out == [
        {"repo": "TheBloke/foo-GGUF", "downloads": 10, "likes": 2},
        {"repo": "bar/baz-gguf", "downloads": 5, "likes": 0},
    ]


def test_search_clamps_bad_sort(monkeypatch):
    seen = {}
    def fake_get_json(url, headers):
        seen["url"] = url
        return []
    cat.search_gguf_models("x", sort="; rm -rf", get_json=fake_get_json)
    assert "sort=downloads" in seen["url"]  # unknown sort falls back


def test_search_network_error_returns_empty():
    def boom(url, headers):
        raise RuntimeError("network down")
    assert cat.search_gguf_models("x", get_json=boom) == []


def test_list_repo_files_filters_gguf_and_builds_url():
    def fake_get_json(url, headers):
        assert url.endswith("/api/models/acme/m/tree/main?recursive=1")
        return [
            {"type": "file", "path": "model-Q4_K_M.gguf", "size": 2000000000},
            {"type": "file", "path": "sub/model-Q8.gguf", "size": 3000000000},
            {"type": "file", "path": "README.md", "size": 100},
            {"type": "directory", "path": "sub"},
        ]
    out = cat.list_repo_gguf_files("acme/m", get_json=fake_get_json)
    assert out == [
        {"filename": "model-Q4_K_M.gguf", "size": 2000000000,
         "url": "https://huggingface.co/acme/m/resolve/main/model-Q4_K_M.gguf"},
        {"filename": "model-Q8.gguf", "size": 3000000000,
         "url": "https://huggingface.co/acme/m/resolve/main/sub/model-Q8.gguf"},
    ]


def test_hf_headers_present_when_token_set(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret123")
    assert cat._hf_headers() == {"Authorization": "Bearer secret123"}


def test_hf_headers_absent_when_no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert cat._hf_headers() == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_localmodels_catalog.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.localmodels.catalog'`.

- [ ] **Step 3: Implement the catalog**

Create `src/localmodels/catalog.py`:

```python
"""Live Hugging Face GGUF catalog (Phase 3b).

Search GGUF model repos and list a repo's .gguf files. All network access is
behind an injectable `get_json(url, headers)` so parsing/URL/token logic is
unit-testable without network. No Cookbook imports.
"""
import os
from urllib.parse import quote

_HF = "https://huggingface.co"
_ALLOWED_SORT = {"downloads", "likes", "lastModified", "trendingScore"}


def _hf_token() -> str:
    return (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()


def _hf_headers() -> dict:
    tok = _hf_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _default_get_json(url: str, headers: dict):
    import httpx
    r = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def search_gguf_models(query: str = "", sort: str = "downloads",
                       limit: int = 30, get_json=None) -> list:
    """Return GGUF model repos matching `query`, most-downloaded first."""
    get_json = get_json or _default_get_json
    if sort not in _ALLOWED_SORT:
        sort = "downloads"
    url = (f"{_HF}/api/models?filter=gguf&sort={sort}&direction=-1"
           f"&limit={int(limit)}&search={quote(query or '')}")
    try:
        data = get_json(url, _hf_headers())
    except Exception:
        return []
    out = []
    for m in data or []:
        repo = m.get("id") or m.get("modelId")
        if repo:
            out.append({"repo": repo,
                        "downloads": int(m.get("downloads") or 0),
                        "likes": int(m.get("likes") or 0)})
    return out


def list_repo_gguf_files(repo: str, get_json=None) -> list:
    """Return the .gguf files in `repo` as [{filename, size, url}]."""
    get_json = get_json or _default_get_json
    url = f"{_HF}/api/models/{repo}/tree/main?recursive=1"
    try:
        data = get_json(url, _hf_headers())
    except Exception:
        return []
    out = []
    for e in data or []:
        path = e.get("path") or ""
        if e.get("type") == "file" and path.lower().endswith(".gguf"):
            out.append({
                "filename": os.path.basename(path),
                "size": int(e.get("size") or 0),
                "url": f"{_HF}/{repo}/resolve/main/{path}",
            })
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_localmodels_catalog.py -v --import-mode=importlib`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/localmodels/catalog.py tests/test_localmodels_catalog.py
git commit -m "feat(localmodels): live HF GGUF catalog search + file listing"
```

---

### Task 2: Streaming download manager

**Files:**
- Create: `src/localmodels/downloader.py`
- Test: `tests/test_localmodels_downloader.py`

**Interfaces:**
- Consumes: `src.constants.MODELS_DIR`; `catalog._hf_headers` (for auth on gated files).
- Produces:
  - `_safe_filename(filename) -> str | None`
  - `class DownloadManager(http_stream=..., spawn=..., dest_dir=None, headers_provider=None)` with `start(url, filename) -> dict`, `status() -> dict`, `cancel() -> dict`, `wait(timeout=5)`.
    - `status()` shape: `{"downloading": bool, "filename": str|None, "bytes": int, "total": int|None, "pct": float|None, "error": str|None}`.
  - `get_download_manager() -> DownloadManager` singleton.
- `http_stream(url, headers)` is a context manager yielding `(total: int|None, chunks: iterable[bytes])`. `spawn(fn)` runs `fn` (default: on a daemon thread; tests inject a synchronous spawn).

- [ ] **Step 1: Write failing tests**

Create `tests/test_localmodels_downloader.py`:

```python
"""DownloadManager tests: synchronous spawn + fake stream, no network."""
import os
from contextlib import contextmanager

import pytest

import src.localmodels.downloader as dl


def make_stream(total, chunks, on_chunk=None):
    @contextmanager
    def _stream(url, headers):
        def gen():
            for c in chunks:
                if on_chunk:
                    on_chunk()
                yield c
        yield total, gen()
    return _stream


def make_manager(tmp_path, stream, **kw):
    return dl.DownloadManager(
        http_stream=stream,
        spawn=lambda fn: fn(),          # synchronous → deterministic
        dest_dir=str(tmp_path),
        headers_provider=lambda: {},
        **kw,
    )


def test_safe_filename():
    assert dl._safe_filename("m.gguf") == "m.gguf"
    assert dl._safe_filename("../m.gguf") is None
    assert dl._safe_filename("a/b.gguf") is None
    assert dl._safe_filename("a\\b.gguf") is None
    assert dl._safe_filename("m.txt") is None
    assert dl._safe_filename("") is None


def test_download_completes_and_renames(tmp_path):
    mgr = make_manager(tmp_path, make_stream(4, [b"aa", b"bb"]))
    mgr.start("https://huggingface.co/x/resolve/main/m.gguf", "m.gguf")
    st = mgr.status()
    assert st["error"] is None
    assert (tmp_path / "m.gguf").read_bytes() == b"aabb"
    assert not (tmp_path / "m.gguf.part").exists()
    assert st["bytes"] == 4 and st["total"] == 4 and st["pct"] == 100.0
    assert st["downloading"] is False


def test_download_rejects_bad_filename(tmp_path):
    mgr = make_manager(tmp_path, make_stream(1, [b"x"]))
    with pytest.raises(ValueError):
        mgr.start("https://huggingface.co/x/resolve/main/e.gguf", "../evil.gguf")


def test_second_download_rejected_while_active(tmp_path):
    mgr = make_manager(tmp_path, make_stream(1, [b"x"]))
    mgr._active = True  # simulate an in-flight download
    with pytest.raises(RuntimeError):
        mgr.start("https://huggingface.co/x/resolve/main/m.gguf", "m.gguf")


def test_cancel_removes_partial(tmp_path):
    # Cancel fires on the first chunk → transfer aborts, .part cleaned up.
    def on_chunk():
        mgr.cancel()
    mgr = make_manager(tmp_path, make_stream(10, [b"aa", b"bb", b"cc"], on_chunk=on_chunk))
    mgr.start("https://huggingface.co/x/resolve/main/m.gguf", "m.gguf")
    assert not (tmp_path / "m.gguf").exists()
    assert not (tmp_path / "m.gguf.part").exists()


def test_status_idle():
    mgr = dl.DownloadManager(spawn=lambda fn: fn())
    assert mgr.status() == {"downloading": False, "filename": None, "bytes": 0,
                            "total": None, "pct": None, "error": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_localmodels_downloader.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.localmodels.downloader'`.

- [ ] **Step 3: Implement the downloader**

Create `src/localmodels/downloader.py`:

```python
"""Streaming GGUF downloader (Phase 3b).

One download at a time, into MODELS_DIR, atomic .part -> .gguf on success.
The transfer (http_stream) and the thread (spawn) are injectable so progress,
cancel, and atomic rename are unit-testable without network or real threads.
"""
import os
import threading
from contextlib import contextmanager

from src.constants import MODELS_DIR


class _Cancelled(Exception):
    pass


def _safe_filename(filename):
    """Return a safe .gguf basename, or None if unsafe."""
    if not filename or "/" in filename or "\\" in filename:
        return None
    if os.path.basename(filename) != filename:
        return None
    if not filename.lower().endswith(".gguf"):
        return None
    return filename


@contextmanager
def _default_http_stream(url, headers):
    import httpx
    with httpx.stream("GET", url, headers=headers, timeout=None,
                      follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0) or None
        yield total, r.iter_bytes()


class DownloadManager:
    def __init__(self, http_stream=_default_http_stream, spawn=None,
                 dest_dir=None, headers_provider=None):
        self._http_stream = http_stream
        self._spawn = spawn or (lambda fn: threading.Thread(
            target=fn, daemon=True).start())
        self._dest_dir = dest_dir or MODELS_DIR
        self._headers_provider = headers_provider or (lambda: {})
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread = None
        self._active = False
        self._state = None  # {filename, bytes, total, error, done}

    def start(self, url, filename):
        with self._lock:
            if self._active:
                raise RuntimeError("a download is already in progress")
            safe = _safe_filename(filename)
            if not safe:
                raise ValueError("invalid filename (must be a plain .gguf name)")
            self._cancel.clear()
            self._state = {"filename": safe, "bytes": 0, "total": None,
                           "error": None, "done": False}
            self._active = True

        def job():
            try:
                self._transfer(url, safe)
            finally:
                self._active = False

        self._spawn(job)
        return self.status()

    def _transfer(self, url, filename):
        os.makedirs(self._dest_dir, exist_ok=True)
        part = os.path.join(self._dest_dir, filename + ".part")
        final = os.path.join(self._dest_dir, filename)
        try:
            with self._http_stream(url, self._headers_provider()) as (total, chunks):
                self._state["total"] = total
                with open(part, "wb") as f:
                    for chunk in chunks:
                        if self._cancel.is_set():
                            raise _Cancelled()
                        f.write(chunk)
                        self._state["bytes"] += len(chunk)
            os.replace(part, final)
            self._state["done"] = True
        except _Cancelled:
            self._cleanup(part)
        except Exception as e:  # network, disk, HTTP status
            self._state["error"] = str(e)
            self._cleanup(part)

    def _cleanup(self, part):
        try:
            if os.path.exists(part):
                os.remove(part)
        except Exception:
            pass

    def cancel(self):
        self._cancel.set()
        t = self._thread
        if t is not None:
            try:
                t.join(timeout=10)
            except Exception:
                pass
        return self.status()

    def wait(self, timeout=5):
        t = self._thread
        if t is not None:
            t.join(timeout)

    def status(self):
        s = self._state
        if s is None:
            return {"downloading": False, "filename": None, "bytes": 0,
                    "total": None, "pct": None, "error": None}
        pct = round(100.0 * s["bytes"] / s["total"], 1) if s["total"] else None
        return {"downloading": bool(self._active and not s["done"]),
                "filename": s["filename"], "bytes": s["bytes"],
                "total": s["total"], "pct": pct, "error": s["error"]}


_download_manager = None


def get_download_manager():
    """Process-wide singleton, wired with HF auth headers for gated files."""
    global _download_manager
    if _download_manager is None:
        from src.localmodels.catalog import _hf_headers
        _download_manager = DownloadManager(headers_provider=_hf_headers)
    return _download_manager
```

Note: `cancel()` joins `self._thread`, which the default (threaded) `spawn` does not set. Store the thread so real cancel can join it — update the default spawn to record it:

Replace the `self._spawn = spawn or (...)` line with:

```python
        if spawn is not None:
            self._spawn = spawn
        else:
            def _thread_spawn(fn):
                self._thread = threading.Thread(target=fn, daemon=True)
                self._thread.start()
            self._spawn = _thread_spawn
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_localmodels_downloader.py -v --import-mode=importlib`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/localmodels/downloader.py tests/test_localmodels_downloader.py
git commit -m "feat(localmodels): streaming one-at-a-time GGUF downloader"
```

---

### Task 3: Catalog + download routes

**Files:**
- Modify: `routes/localmodels_routes.py`
- Test: `tests/test_localmodels_download_routes.py`

**Interfaces:**
- Consumes: `catalog.search_gguf_models`/`list_repo_gguf_files`, `downloader.get_download_manager`, `downloader._safe_filename`.
- Produces (added to `setup_localmodels_routes()`):
  - `GET /api/localmodels/catalog/search?q=&sort=` → `{"results": [...]}`
  - `GET /api/localmodels/catalog/files?repo=` → `{"files": [...]}`
  - `POST /api/localmodels/download {url, filename}` → download status (validates `huggingface.co` host + safe filename)
  - `GET /api/localmodels/download/status` → status
  - `POST /api/localmodels/download/cancel` → status

- [ ] **Step 1: Write failing tests**

Create `tests/test_localmodels_download_routes.py`:

```python
"""Catalog + download route behavior with fakes (no network)."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.localmodels_routes as lmr
from core.middleware import require_admin


class FakeDL:
    def __init__(self):
        self.started = None
        self.cancelled = False
    def start(self, url, filename):
        self.started = (url, filename)
        return {"downloading": True, "filename": filename, "bytes": 0,
                "total": None, "pct": None, "error": None}
    def status(self):
        return {"downloading": False, "filename": None, "bytes": 0,
                "total": None, "pct": None, "error": None}
    def cancel(self):
        self.cancelled = True
        return self.status()


@pytest.fixture
def client(monkeypatch):
    fake = FakeDL()
    monkeypatch.setattr(lmr, "search_gguf_models",
                        lambda q, sort="downloads": [{"repo": "a/b", "downloads": 9, "likes": 1}])
    monkeypatch.setattr(lmr, "list_repo_gguf_files",
                        lambda repo: [{"filename": "m.gguf", "size": 10,
                                       "url": "https://huggingface.co/a/b/resolve/main/m.gguf"}])
    monkeypatch.setattr(lmr, "get_download_manager", lambda: fake)
    app = FastAPI()
    app.include_router(lmr.setup_localmodels_routes())
    app.dependency_overrides[require_admin] = lambda: None
    return TestClient(app), fake


def test_search(client):
    c, _ = client
    r = c.get("/api/localmodels/catalog/search", params={"q": "qwen"})
    assert r.status_code == 200
    assert r.json()["results"][0]["repo"] == "a/b"


def test_files(client):
    c, _ = client
    r = c.get("/api/localmodels/catalog/files", params={"repo": "a/b"})
    assert r.status_code == 200
    assert r.json()["files"][0]["filename"] == "m.gguf"


def test_download_valid(client):
    c, fake = client
    r = c.post("/api/localmodels/download",
               json={"url": "https://huggingface.co/a/b/resolve/main/m.gguf",
                     "filename": "m.gguf"})
    assert r.status_code == 200
    assert fake.started == ("https://huggingface.co/a/b/resolve/main/m.gguf", "m.gguf")


def test_download_rejects_non_hf_url(client):
    c, _ = client
    r = c.post("/api/localmodels/download",
               json={"url": "https://evil.com/m.gguf", "filename": "m.gguf"})
    assert r.status_code == 400


def test_download_rejects_bad_filename(client):
    c, _ = client
    r = c.post("/api/localmodels/download",
               json={"url": "https://huggingface.co/a/b/resolve/main/m.gguf",
                     "filename": "../evil.gguf"})
    assert r.status_code == 400


def test_download_cancel(client):
    c, fake = client
    r = c.post("/api/localmodels/download/cancel")
    assert r.status_code == 200
    assert fake.cancelled is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_localmodels_download_routes.py -v --import-mode=importlib`
Expected: FAIL — the new routes/imports don't exist yet (404s / AttributeError).

- [ ] **Step 3: Extend the routes**

In `routes/localmodels_routes.py`, add imports at the top (after the existing imports):

```python
from urllib.parse import urlparse

from src.localmodels.catalog import search_gguf_models, list_repo_gguf_files
from src.localmodels.downloader import get_download_manager, _safe_filename
```

Add a URL validator near `_validate_model_path`:

```python
def _validate_hf_download(url: str, filename: str):
    if not _safe_filename(filename):
        raise HTTPException(status_code=400,
                            detail="filename must be a plain .gguf name")
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    if host.lower() != "huggingface.co":
        raise HTTPException(status_code=400,
                            detail="url must be a huggingface.co download URL")
```

Inside `setup_localmodels_routes()`, before `return router`, add:

```python
    @router.get("/catalog/search")
    async def catalog_search(q: str = "", sort: str = "downloads"):
        return {"results": search_gguf_models(q, sort=sort)}

    @router.get("/catalog/files")
    async def catalog_files(repo: str):
        return {"files": list_repo_gguf_files(repo)}

    @router.post("/download")
    async def download(payload: dict = Body(...)):
        url = (payload.get("url") or "").strip()
        filename = (payload.get("filename") or "").strip()
        _validate_hf_download(url, filename)
        try:
            return get_download_manager().start(url, filename)
        except (RuntimeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/download/status")
    async def download_status():
        return get_download_manager().status()

    @router.post("/download/cancel")
    async def download_cancel():
        return get_download_manager().cancel()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_localmodels_download_routes.py -v --import-mode=importlib`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add routes/localmodels_routes.py tests/test_localmodels_download_routes.py
git commit -m "feat(localmodels): catalog + download routes (huggingface.co only)"
```

---

### Task 4: Search + download UI

Extend the Local Models modal with a search box, results, per-file Download, and a progress bar.

**Files:**
- Modify: `static/js/localModels.js`, `static/index.html`
- Test: `tests/test_localmodels_download_ui.py`

**Interfaces:**
- Consumes: `/api/localmodels/catalog/search`, `/catalog/files`, `/download`, `/download/status`, `/download/cancel`.
- Produces: `#localmodels-search`, `#localmodels-results`, `#localmodels-progress` in the modal.

- [ ] **Step 1: Write failing UI-wiring guard tests**

Create `tests/test_localmodels_download_ui.py`:

```python
"""Text guards that the Phase 3b download UI is wired."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_index_has_search_and_progress_elements():
    html = _read("static/index.html")
    assert 'id="localmodels-search"' in html
    assert 'id="localmodels-results"' in html
    assert 'id="localmodels-progress"' in html


def test_js_calls_catalog_and_download_endpoints():
    js = _read("static/js/localModels.js")
    for ep in ("/api/localmodels/catalog/search", "/api/localmodels/catalog/files",
               "/api/localmodels/download", "/api/localmodels/download/status",
               "/api/localmodels/download/cancel"):
        assert ep in js, f"{ep} not called in localModels.js"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_localmodels_download_ui.py -v --import-mode=importlib`
Expected: FAIL — the elements/endpoints don't exist yet.

- [ ] **Step 3: Add the search/progress markup to the modal**

In `static/index.html`, inside the `#localmodels-modal` `modal-content` (after the existing `#localmodels-list` div added in 3a), add:

```html
      <hr style="border:none;border-top:1px solid var(--border,#333);margin:14px 0;">
      <div style="display:flex;gap:6px;margin-bottom:8px;">
        <input id="localmodels-search" type="text" placeholder="Search Hugging Face for GGUF models…" style="flex:1;">
        <button id="localmodels-search-btn">Search</button>
      </div>
      <div id="localmodels-progress" style="display:none;font-size:12px;margin-bottom:8px;"></div>
      <div id="localmodels-results"></div>
```

- [ ] **Step 4: Extend `localModels.js`**

Append the following inside the IIFE in `static/js/localModels.js`, just before the `window.LocalModels = ...` line, and extend the exported object:

```javascript
  function fmtBytes(n) {
    if (!n) return '';
    if (n >= 1e9) return (n / 1e9).toFixed(1) + ' GB';
    if (n >= 1e6) return (n / 1e6).toFixed(0) + ' MB';
    return (n / 1e3).toFixed(0) + ' KB';
  }

  let downloadedNames = new Set();

  async function pollDownload() {
    const prog = $('localmodels-progress');
    if (!prog) return;
    let st = { downloading: false };
    try { st = await api('/api/localmodels/download/status'); } catch (e) {}
    if (st.downloading) {
      const pct = st.pct != null ? st.pct + '%' : fmtBytes(st.bytes);
      prog.style.display = 'block';
      prog.innerHTML = `Downloading ${st.filename}: ${pct} ` +
        `<button id="localmodels-cancel-btn">Cancel</button>`;
      const cancel = $('localmodels-cancel-btn');
      if (cancel) cancel.onclick = () => api('/api/localmodels/download/cancel', { method: 'POST' });
      setTimeout(pollDownload, 800);
    } else {
      prog.style.display = 'none';
      if (st.error) alert('Download error: ' + st.error);
      refresh();  // a finished .gguf now shows in the serve list
    }
  }

  async function doSearch() {
    const q = ($('localmodels-search') || {}).value || '';
    const resultsEl = $('localmodels-results');
    if (!resultsEl) return;
    resultsEl.innerHTML = 'Searching…';
    let data = { results: [] };
    try { data = await api('/api/localmodels/catalog/search?q=' + encodeURIComponent(q)); }
    catch (e) { resultsEl.innerHTML = 'Search failed: ' + e.message; return; }
    resultsEl.innerHTML = '';
    data.results.forEach((r) => {
      const row = document.createElement('div');
      row.className = 'list-item';
      const label = document.createElement('span');
      label.className = 'grow';
      label.textContent = `${r.repo}  (${r.downloads.toLocaleString()} downloads)`;
      const btn = document.createElement('button');
      btn.textContent = 'Files';
      btn.onclick = () => listFiles(r.repo, row);
      row.appendChild(label);
      row.appendChild(btn);
      resultsEl.appendChild(row);
    });
    if (!data.results.length) resultsEl.textContent = 'No GGUF models found.';
  }

  async function listFiles(repo, afterRow) {
    let data = { files: [] };
    try { data = await api('/api/localmodels/catalog/files?repo=' + encodeURIComponent(repo)); }
    catch (e) { alert('Could not list files: ' + e.message); return; }
    data.files.forEach((f) => {
      const row = document.createElement('div');
      row.className = 'list-item';
      row.style.paddingLeft = '18px';
      const label = document.createElement('span');
      label.className = 'grow';
      label.textContent = `${f.filename} — ${fmtBytes(f.size)}`;
      const btn = document.createElement('button');
      const have = downloadedNames.has(f.filename);
      btn.textContent = have ? 'Downloaded' : 'Download';
      btn.disabled = have;
      btn.onclick = async () => {
        btn.disabled = true;
        try {
          await api('/api/localmodels/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: f.url, filename: f.filename }),
          });
          pollDownload();
        } catch (e) { alert('Download error: ' + e.message); btn.disabled = false; }
      };
      row.appendChild(label);
      row.appendChild(btn);
      afterRow.insertAdjacentElement('afterend', row);
    });
  }
```

Then, in the existing `refresh()` function (from 3a), record the downloaded names so the catalog can mark them — add right after `data = await api('/api/localmodels/models')` succeeds:

```javascript
    downloadedNames = new Set((data.models || []).map((m) => m.name));
```

And wire the search button + initial poll in the existing `DOMContentLoaded` handler by adding:

```javascript
    const searchBtn = $('localmodels-search-btn');
    if (searchBtn) searchBtn.addEventListener('click', doSearch);
```

Finally extend the export: change `window.LocalModels = { open, close, refresh };` to
`window.LocalModels = { open, close, refresh, doSearch, pollDownload };`.

- [ ] **Step 5: Run the UI-wiring guard tests**

Run: `python -m pytest tests/test_localmodels_download_ui.py -v --import-mode=importlib`
Expected: PASS (2 tests).

> **Manual (not unit-tested):** open Local Models, search a real term, download a small GGUF, watch progress, confirm it lands in the serve list; cancel mid-download and confirm no partial. Covered by the 3b smoke test.

- [ ] **Step 6: Commit**

```bash
git add static/js/localModels.js static/index.html tests/test_localmodels_download_ui.py
git commit -m "feat(localmodels): HF search + download UI with progress"
```

---

## Appendix: Manual smoke test (real acceptance gate)

With the app running (dev or built):
1. Open **Local Models** → type a query (e.g. "qwen2.5 3b gguf") → **Search** → repo results appear.
2. Click **Files** on a small repo → `.gguf` files with sizes appear.
3. Click **Download** on a small file → progress bar advances; **Cancel** works (leaves no `.part`/`.gguf`).
4. Let a small file finish → it appears in the top model list and **Serve** starts it (3a).
5. Re-open Files for that repo → the downloaded file shows **Downloaded** (disabled).

## Self-Review

**Spec coverage:**
- Catalog service (search + files, injectable network, token header) → Task 1. ✓
- Downloader (streaming, one-at-a-time, atomic .part→.gguf, cancel, safe filename) → Task 2. ✓
- Routes (search/files/download/status/cancel, huggingface.co-only + safe filename) → Task 3. ✓
- UI (search + results + per-file download + progress + mark-downloaded) → Task 4. ✓
- No new dependency (httpx) / Cookbook untouched / admin-guarded / HF token env → constraints honored across tasks. ✓
- Testing (unit for catalog+downloader+routes, text-guard UI, manual smoke) → each task + appendix. ✓

**Placeholder scan:** No TBD/TODO; every code step is complete. Manual-only real-network steps are labeled and backed by unit fakes + the smoke appendix. ✓

**Type consistency:** `status()` shape `{downloading, filename, bytes, total, pct, error}` identical across downloader, fake in route tests, and UI poll. `search_gguf_models(query, sort, limit, get_json)` / `list_repo_gguf_files(repo, get_json)` signatures match between Task 1, the Task 3 route calls (`search_gguf_models(q, sort=sort)`, `list_repo_gguf_files(repo)`), and the route-test monkeypatches. `_safe_filename` shared by downloader (Task 2) and route validator (Task 3). Endpoint paths identical between Task 3 routes, Task 4 JS, and both test files. ✓

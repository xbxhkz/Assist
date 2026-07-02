# Assist Phase 3a — Native Serve Runtime — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assist can launch a bundled `llama-server` against a local GGUF file, health-check it, register it as a local OpenAI-compatible endpoint, and manage one-model-at-a-time start/stop — with a minimal UI.

**Architecture:** A new `src/localmodels/` package: pure `runtime.py` (binary resolver, argv/URL builders, gguf listing) + a stateful `LocalModelManager` with injectable dependencies, wired through `routes/localmodels_routes.py` and a `local` `ModelEndpoint` upsert. A build-time script vendors a CPU `llama-server` into the PyInstaller bundle. A minimal modal UI mirrors the existing Cookbook modal.

**Tech Stack:** Python 3.14, FastAPI/uvicorn, subprocess, SQLAlchemy (`ModelEndpoint`), vanilla JS, PyInstaller, pytest.

## Global Constraints

- **New subsystem, Cookbook untouched:** do NOT modify the existing tmux/bash Cookbook (`routes/cookbook_*.py`, `src/cookbook_*.py`, `services/hwfit/`). Native local models are a separate package.
- **Loopback only:** the local server binds `127.0.0.1`; endpoint URLs are `http://127.0.0.1:<port>/v1`.
- **One model at a time:** `start()` stops any running model first.
- **Binary preference:** a `llama-server`/`llama-server.exe` on `PATH` is preferred (user-installed, possibly GPU); else the bundled CPU binary at `<_MEIPASS>/llama/llama-server.exe`; else a dev fallback under `build_assets/llama/`.
- **Endpoint identity:** native-local endpoints use `endpoint_kind="local"` and id prefix `local-`, distinct from Cookbook endpoints.
- **Naming/rename:** do NOT rename internal modules or `ODYSSEUS_*` env vars (Phase 4).
- **Path safety:** `serve` only accepts a `.gguf` path resolved inside `MODELS_DIR`.
- **Test env:** run pytest with `--import-mode=importlib`; new tests carry no `slow` marker.
- **Reuse:** use `src.desktop_runtime.choose_port` and `wait_for_server_ready` (from Phase 2) for port + readiness.

## File Structure

- `src/constants.py` (modify) — add `MODELS_DIR`.
- `src/localmodels/__init__.py` (new, empty package marker).
- `src/localmodels/runtime.py` (new) — pure helpers. Task 1.
- `src/localmodels/manager.py` (new) — `LocalModelManager` + `get_manager()`. Task 2 (class), Task 3 wires `get_manager`.
- `src/localmodels/store.py` (new) — `register_local_endpoint`, `unregister_local_endpoint`. Task 3.
- `routes/localmodels_routes.py` (new) — routes. Task 3.
- `app.py` (modify) — include router + shutdown stop hook. Task 3.
- `scripts/fetch_llama_server.py` (new) + `Assist.spec`/`build-windows-portable.ps1` (modify). Task 4.
- `static/js/localModels.js` (new) + `static/index.html` (modify). Task 5.
- Tests: `tests/test_localmodels_runtime.py`, `tests/test_localmodels_manager.py`, `tests/test_localmodels_routes.py`, `tests/test_localmodels_store.py`, `tests/test_localmodels_packaging.py`, `tests/test_localmodels_ui.py`.

---

### Task 1: Constants + pure runtime helpers

**Files:**
- Modify: `src/constants.py`
- Create: `src/localmodels/__init__.py`, `src/localmodels/runtime.py`
- Test: `tests/test_localmodels_runtime.py`

**Interfaces:**
- Consumes: `src.constants.DATA_DIR`.
- Produces:
  - `MODELS_DIR: str` (in constants).
  - `local_endpoint_url(port: int) -> str`
  - `build_serve_argv(binary: str, model_path: str, port: int, ctx_size: int = 4096, host: str = "127.0.0.1") -> list[str]`
  - `resolve_llama_binary(path_lookup=shutil.which, frozen_base: str | None = None, dev_base: str | None = None) -> str`
  - `list_gguf_models(models_dir: str) -> list[dict]`  (each `{"name", "path", "size"}`)

- [ ] **Step 1: Write failing tests**

Create `tests/test_localmodels_runtime.py`:

```python
"""Unit tests for the pure native-local-model runtime helpers."""
import os

import src.localmodels.runtime as rt


def test_local_endpoint_url():
    assert rt.local_endpoint_url(8123) == "http://127.0.0.1:8123/v1"


def test_build_serve_argv_has_model_host_port():
    argv = rt.build_serve_argv("/x/llama-server", "/m/model.gguf", 8123)
    assert argv[0] == "/x/llama-server"
    assert "--model" in argv and "/m/model.gguf" in argv
    assert "--host" in argv and "127.0.0.1" in argv
    assert "--port" in argv and "8123" in argv
    assert "--ctx-size" in argv and "4096" in argv


def test_resolve_prefers_path_binary():
    got = rt.resolve_llama_binary(path_lookup=lambda n: "/usr/bin/llama-server"
                                  if n == "llama-server" else None)
    assert got == "/usr/bin/llama-server"


def test_resolve_uses_bundled_when_no_path(tmp_path):
    bundle = tmp_path / "llama"
    bundle.mkdir()
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    (bundle / name).write_text("stub")
    got = rt.resolve_llama_binary(path_lookup=lambda n: None,
                                  frozen_base=str(tmp_path))
    assert got == str(bundle / name)


def test_resolve_raises_when_nothing_found(tmp_path):
    import pytest
    with pytest.raises(RuntimeError):
        rt.resolve_llama_binary(path_lookup=lambda n: None,
                                frozen_base=str(tmp_path),
                                dev_base=str(tmp_path / "nope"))


def test_list_gguf_models_filters_and_reports_size(tmp_path):
    (tmp_path / "a.gguf").write_bytes(b"xxxx")
    (tmp_path / "b.txt").write_text("no")
    models = rt.list_gguf_models(str(tmp_path))
    assert [m["name"] for m in models] == ["a.gguf"]
    assert models[0]["size"] == 4
    assert models[0]["path"] == str(tmp_path / "a.gguf")


def test_list_gguf_models_missing_dir_is_empty():
    assert rt.list_gguf_models("/no/such/dir") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_localmodels_runtime.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.localmodels'`.

- [ ] **Step 3: Add MODELS_DIR to constants**

In `src/constants.py`, after the `UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")` line, add:

```python
# Native local GGUF models (Phase 3a). Frozen data dir is ~/.odysseus/data,
# so downloaded models persist across app upgrades.
MODELS_DIR = os.path.join(DATA_DIR, "models")
```

- [ ] **Step 4: Create the package and runtime module**

Create empty `src/localmodels/__init__.py`:

```python
```

Create `src/localmodels/runtime.py`:

```python
"""Pure helpers for the native local-model runtime (Phase 3a).

No process launching or DB access here — just binary resolution, command
construction, and filesystem listing, so this module is fully unit-testable.
"""
import os
import shutil
import sys


def local_endpoint_url(port: int) -> str:
    """OpenAI-compatible base URL for a locally served model (loopback only)."""
    return f"http://127.0.0.1:{port}/v1"


def build_serve_argv(binary: str, model_path: str, port: int,
                     ctx_size: int = 4096, host: str = "127.0.0.1") -> list:
    """llama-server argv for an OpenAI-compatible loopback server."""
    return [
        binary,
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(ctx_size),
    ]


def _bundled_binary_name() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def resolve_llama_binary(path_lookup=shutil.which, frozen_base: str = None,
                         dev_base: str = None) -> str:
    """Resolve the llama-server executable.

    Preference: (1) a `llama-server`/`.exe` on PATH (user-installed, possibly
    GPU); (2) the bundled CPU binary at `<frozen_base>/llama/<name>` when frozen;
    (3) a dev fallback at `<repo>/build_assets/llama/<name>`. Raises if none.
    `path_lookup`/`frozen_base`/`dev_base` are injectable for tests.
    """
    found = path_lookup("llama-server") or path_lookup("llama-server.exe")
    if found:
        return found

    name = _bundled_binary_name()

    base = frozen_base
    if base is None and getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
    if base:
        cand = os.path.join(base, "llama", name)
        if os.path.isfile(cand):
            return cand

    if dev_base is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dev_base = os.path.join(repo_root, "build_assets", "llama")
    cand = os.path.join(dev_base, name)
    if os.path.isfile(cand):
        return cand

    raise RuntimeError(
        "llama-server not found: no server on PATH, no bundled binary, and no "
        "dev binary under build_assets/llama."
    )


def list_gguf_models(models_dir: str) -> list:
    """List `.gguf` files in `models_dir` as [{name, path, size}], sorted by name."""
    out = []
    if os.path.isdir(models_dir):
        for fn in sorted(os.listdir(models_dir)):
            if fn.lower().endswith(".gguf"):
                p = os.path.join(models_dir, fn)
                out.append({"name": fn, "path": p, "size": os.path.getsize(p)})
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_localmodels_runtime.py -v --import-mode=importlib`
Expected: PASS (7 tests).

- [ ] **Step 6: Commit**

```bash
git add src/constants.py src/localmodels/__init__.py src/localmodels/runtime.py tests/test_localmodels_runtime.py
git commit -m "feat(localmodels): MODELS_DIR + pure runtime helpers"
```

---

### Task 2: LocalModelManager

**Files:**
- Create: `src/localmodels/manager.py`
- Test: `tests/test_localmodels_manager.py`

**Interfaces:**
- Consumes: `runtime.resolve_llama_binary`, `runtime.build_serve_argv`, `runtime.local_endpoint_url`, `runtime.list_gguf_models`; `src.desktop_runtime.choose_port`, `wait_for_server_ready`; `src.constants.MODELS_DIR`.
- Produces: `class LocalModelManager` with injectable deps and methods `start(model_path) -> dict`, `stop() -> dict`, `status() -> dict`, `list_models() -> list`. (`get_manager()` singleton is added in Task 3.)
  - `status()` shape: `{"running": bool, "model": str|None, "port": int|None, "endpoint_id": str|None}`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_localmodels_manager.py`:

```python
"""State-machine tests for LocalModelManager using injected fakes (no real
process, port, or DB)."""
from src.localmodels.manager import LocalModelManager


class FakeProc:
    def __init__(self, pid=4321):
        self.pid = pid
        self.terminated = False
    def terminate(self):
        self.terminated = True


def make_manager(ready=True, spawned=None, registered=None, unregistered=None):
    spawned = spawned if spawned is not None else []
    registered = registered if registered is not None else []
    unregistered = unregistered if unregistered is not None else []

    def spawn(argv):
        p = FakeProc()
        spawned.append((argv, p))
        return p

    def register(name, base_url):
        eid = f"local-{len(registered)}"
        registered.append({"name": name, "base_url": base_url, "id": eid})
        return eid

    def unregister(endpoint_id):
        unregistered.append(endpoint_id)

    mgr = LocalModelManager(
        spawn=spawn,
        port_chooser=lambda: 8123,
        readiness=lambda url: ready,
        register_endpoint=register,
        unregister_endpoint=unregister,
        resolve_binary=lambda: "/bin/llama-server",
    )
    return mgr, spawned, registered, unregistered


def test_start_launches_and_registers():
    mgr, spawned, registered, _ = make_manager()
    st = mgr.start("/models/m.gguf")
    assert st == {"running": True, "model": "m.gguf", "port": 8123,
                  "endpoint_id": "local-0"}
    assert len(spawned) == 1
    assert registered[0]["base_url"] == "http://127.0.0.1:8123/v1"


def test_start_readiness_failure_kills_and_raises():
    import pytest
    mgr, spawned, registered, _ = make_manager(ready=False)
    with pytest.raises(RuntimeError):
        mgr.start("/models/m.gguf")
    assert spawned[0][1].terminated is True   # process killed
    assert registered == []                    # no dead endpoint registered
    assert mgr.status()["running"] is False


def test_start_twice_stops_previous_first():
    mgr, spawned, registered, unregistered = make_manager()
    mgr.start("/models/a.gguf")
    mgr.start("/models/b.gguf")
    assert spawned[0][1].terminated is True     # first process stopped
    assert unregistered == ["local-0"]          # first endpoint removed
    assert mgr.status()["model"] == "b.gguf"


def test_stop_terminates_and_unregisters():
    mgr, spawned, registered, unregistered = make_manager()
    mgr.start("/models/a.gguf")
    st = mgr.stop()
    assert st["running"] is False
    assert spawned[0][1].terminated is True
    assert unregistered == ["local-0"]


def test_status_when_idle():
    mgr, *_ = make_manager()
    assert mgr.status() == {"running": False, "model": None, "port": None,
                            "endpoint_id": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_localmodels_manager.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.localmodels.manager'`.

- [ ] **Step 3: Implement the manager**

Create `src/localmodels/manager.py`:

```python
"""Stateful manager for a single native local model (Phase 3a).

Owns at most one llama-server subprocess at a time. All side-effecting
dependencies (process spawn, port choice, readiness poll, endpoint register/
unregister, binary resolution) are injectable so the state machine is
unit-testable without a real binary, port, or database.
"""
import os
import subprocess
import threading

from src.constants import MODELS_DIR
from src.desktop_runtime import choose_port, wait_for_server_ready
from src.localmodels.runtime import (
    resolve_llama_binary, build_serve_argv, local_endpoint_url, list_gguf_models,
)


def _default_spawn(argv):
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


class LocalModelManager:
    def __init__(self, spawn=_default_spawn, port_chooser=None, readiness=None,
                 register_endpoint=None, unregister_endpoint=None,
                 resolve_binary=resolve_llama_binary):
        self._spawn = spawn
        self._port_chooser = port_chooser or (lambda: choose_port(8100))
        self._readiness = readiness or wait_for_server_ready
        self._register = register_endpoint
        self._unregister = unregister_endpoint
        self._resolve_binary = resolve_binary
        self._lock = threading.Lock()
        self._proc = None
        self._state = None  # {"model_path", "port", "endpoint_id", "pid"}

    def start(self, model_path: str) -> dict:
        with self._lock:
            if self._proc is not None:
                self._stop_locked()
            binary = self._resolve_binary()
            port = self._port_chooser()
            proc = self._spawn(build_serve_argv(binary, model_path, port))
            url = local_endpoint_url(port)
            if not self._readiness(url + "/models"):
                try:
                    proc.terminate()
                except Exception:
                    pass
                raise RuntimeError("llama-server did not become ready in time")
            endpoint_id = None
            if self._register:
                endpoint_id = self._register(name=os.path.basename(model_path),
                                             base_url=url)
            self._proc = proc
            self._state = {"model_path": model_path, "port": port,
                           "endpoint_id": endpoint_id,
                           "pid": getattr(proc, "pid", None)}
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self) -> dict:
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
        if self._state and self._state.get("endpoint_id") and self._unregister:
            try:
                self._unregister(self._state["endpoint_id"])
            except Exception:
                pass
        self._state = None
        return self.status()

    def status(self) -> dict:
        if self._state is None:
            return {"running": False, "model": None, "port": None,
                    "endpoint_id": None}
        return {"running": True,
                "model": os.path.basename(self._state["model_path"]),
                "port": self._state["port"],
                "endpoint_id": self._state["endpoint_id"]}

    def list_models(self) -> list:
        return list_gguf_models(MODELS_DIR)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_localmodels_manager.py -v --import-mode=importlib`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/localmodels/manager.py tests/test_localmodels_manager.py
git commit -m "feat(localmodels): LocalModelManager one-at-a-time lifecycle"
```

---

### Task 3: Endpoint store, routes, and app wiring

**Files:**
- Create: `src/localmodels/store.py`, `routes/localmodels_routes.py`
- Modify: `src/localmodels/manager.py` (add `get_manager()`), `app.py` (include router + shutdown stop)
- Test: `tests/test_localmodels_store.py`, `tests/test_localmodels_routes.py`

**Interfaces:**
- Consumes: `LocalModelManager` (Task 2), `core.database.ModelEndpoint`/`SessionLocal`, `core.middleware.require_admin`, `src.constants.MODELS_DIR`.
- Produces:
  - `store.register_local_endpoint(name, base_url, session_factory=None) -> str`
  - `store.unregister_local_endpoint(endpoint_id, session_factory=None) -> None`
  - `manager.get_manager() -> LocalModelManager` (singleton wired with the real store).
  - `routes.setup_localmodels_routes() -> APIRouter` mounted at `/api/localmodels`.

- [ ] **Step 1: Write failing store test**

Create `tests/test_localmodels_store.py`:

```python
"""The local-endpoint store creates/removes a ModelEndpoint row (endpoint_kind
='local'). Uses an in-memory SQLite session so it exercises the real ORM."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database import Base, ModelEndpoint
import src.localmodels.store as store


def _mem_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def test_register_creates_local_endpoint():
    sf = _mem_session_factory()
    eid = store.register_local_endpoint("m.gguf", "http://127.0.0.1:8123/v1",
                                        session_factory=sf)
    assert eid.startswith("local-")
    db = sf()
    ep = db.query(ModelEndpoint).filter(ModelEndpoint.id == eid).one()
    assert ep.base_url == "http://127.0.0.1:8123/v1"
    assert ep.endpoint_kind == "local"
    assert ep.is_enabled is True
    db.close()


def test_register_updates_existing_same_url():
    sf = _mem_session_factory()
    first = store.register_local_endpoint("a.gguf", "http://127.0.0.1:8123/v1",
                                          session_factory=sf)
    second = store.register_local_endpoint("b.gguf", "http://127.0.0.1:8123/v1",
                                           session_factory=sf)
    assert first == second  # same row reused for the same base_url
    db = sf()
    assert db.query(ModelEndpoint).count() == 1
    assert db.query(ModelEndpoint).one().name == "b.gguf"
    db.close()


def test_unregister_deletes_row():
    sf = _mem_session_factory()
    eid = store.register_local_endpoint("a.gguf", "http://127.0.0.1:8123/v1",
                                        session_factory=sf)
    store.unregister_local_endpoint(eid, session_factory=sf)
    db = sf()
    assert db.query(ModelEndpoint).count() == 0
    db.close()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_localmodels_store.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.localmodels.store'`.

- [ ] **Step 3: Implement the store**

Create `src/localmodels/store.py`:

```python
"""Persistence for native-local model endpoints (Phase 3a).

Mirrors the auto-register pattern in routes/cookbook_routes.py, but tags rows
with endpoint_kind="local" and an id prefix `local-` so native-local endpoints
are distinct from Cookbook ones. `session_factory` is injectable for tests.
"""
import uuid


def register_local_endpoint(name: str, base_url: str, session_factory=None) -> str:
    """Create or update the local ModelEndpoint for `base_url`; return its id."""
    from core.database import SessionLocal, ModelEndpoint
    sf = session_factory or SessionLocal
    db = sf()
    try:
        existing = db.query(ModelEndpoint).filter(
            ModelEndpoint.base_url == base_url).first()
        if existing:
            existing.is_enabled = True
            existing.name = name
            existing.endpoint_kind = "local"
            db.commit()
            return existing.id
        eid = f"local-{uuid.uuid4().hex[:8]}"
        ep = ModelEndpoint(id=eid, name=name, base_url=base_url, api_key=None,
                           is_enabled=True, endpoint_kind="local")
        db.add(ep)
        db.commit()
        return eid
    finally:
        db.close()


def unregister_local_endpoint(endpoint_id: str, session_factory=None) -> None:
    """Delete the ModelEndpoint row with `endpoint_id`, if present."""
    from core.database import SessionLocal, ModelEndpoint
    sf = session_factory or SessionLocal
    db = sf()
    try:
        ep = db.query(ModelEndpoint).filter(
            ModelEndpoint.id == endpoint_id).first()
        if ep:
            db.delete(ep)
            db.commit()
    finally:
        db.close()
```

- [ ] **Step 4: Add `get_manager()` to the manager**

Append to `src/localmodels/manager.py`:

```python
_manager = None


def get_manager() -> "LocalModelManager":
    """Process-wide singleton wired with the real endpoint store."""
    global _manager
    if _manager is None:
        from src.localmodels.store import (
            register_local_endpoint, unregister_local_endpoint,
        )
        _manager = LocalModelManager(
            register_endpoint=register_local_endpoint,
            unregister_endpoint=unregister_local_endpoint,
        )
    return _manager
```

- [ ] **Step 5: Write failing routes test**

Create `tests/test_localmodels_routes.py`:

```python
"""Route behavior for /api/localmodels using a fake manager and an admin
override. Verifies path-safety validation and manager delegation."""
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.localmodels_routes as lmr
from core.middleware import require_admin


class FakeManager:
    def __init__(self):
        self.started = None
        self.stopped = False
    def list_models(self):
        return [{"name": "m.gguf", "path": "/x/m.gguf", "size": 4}]
    def status(self):
        return {"running": bool(self.started), "model": self.started,
                "port": 8123, "endpoint_id": "local-0"}
    def start(self, model_path):
        self.started = os.path.basename(model_path)
        return self.status()
    def stop(self):
        self.stopped = True
        self.started = None
        return self.status()


@pytest.fixture
def client(monkeypatch, tmp_path):
    fake = FakeManager()
    monkeypatch.setattr(lmr, "get_manager", lambda: fake)
    monkeypatch.setattr(lmr, "MODELS_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(lmr.setup_localmodels_routes())
    app.dependency_overrides[require_admin] = lambda: None
    return TestClient(app), fake, tmp_path


def test_list_models(client):
    c, _fake, _ = client
    r = c.get("/api/localmodels/models")
    assert r.status_code == 200
    assert r.json()["models"][0]["name"] == "m.gguf"


def test_serve_rejects_path_outside_models_dir(client):
    c, _fake, _ = client
    r = c.post("/api/localmodels/serve", json={"model_path": "/etc/passwd.gguf"})
    assert r.status_code == 400


def test_serve_rejects_non_gguf(client):
    c, _fake, tmp = client
    f = tmp / "note.txt"
    f.write_text("x")
    r = c.post("/api/localmodels/serve", json={"model_path": str(f)})
    assert r.status_code == 400


def test_serve_starts_valid_model(client):
    c, fake, tmp = client
    f = tmp / "m.gguf"
    f.write_bytes(b"xxxx")
    r = c.post("/api/localmodels/serve", json={"model_path": str(f)})
    assert r.status_code == 200
    assert fake.started == "m.gguf"


def test_stop_delegates(client):
    c, fake, _ = client
    r = c.post("/api/localmodels/stop")
    assert r.status_code == 200
    assert fake.stopped is True
```

- [ ] **Step 6: Run it to verify it fails**

Run: `python -m pytest tests/test_localmodels_routes.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'routes.localmodels_routes'`.

- [ ] **Step 7: Implement the routes**

Create `routes/localmodels_routes.py`:

```python
"""HTTP control surface for native local models (Phase 3a).

Admin-guarded. Serve accepts only a .gguf path resolved inside MODELS_DIR.
"""
import os

from fastapi import APIRouter, Body, Depends, HTTPException

from core.middleware import require_admin
from src.constants import MODELS_DIR
from src.localmodels.manager import get_manager


def _validate_model_path(model_path: str) -> str:
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")
    real_models = os.path.realpath(MODELS_DIR)
    real = os.path.realpath(model_path)
    try:
        inside = os.path.commonpath([real, real_models]) == real_models
    except ValueError:
        inside = False  # different drive on Windows
    if not inside:
        raise HTTPException(status_code=400,
                            detail="model_path must be inside the models directory")
    if not real.lower().endswith(".gguf"):
        raise HTTPException(status_code=400,
                            detail="model_path must be a .gguf file")
    if not os.path.isfile(real):
        raise HTTPException(status_code=404, detail="model file not found")
    return real


def setup_localmodels_routes() -> APIRouter:
    router = APIRouter(prefix="/api/localmodels",
                       dependencies=[Depends(require_admin)])

    @router.get("/models")
    async def list_models():
        return {"models": get_manager().list_models()}

    @router.get("/status")
    async def status():
        return get_manager().status()

    @router.post("/serve")
    async def serve(payload: dict = Body(...)):
        safe = _validate_model_path((payload.get("model_path") or "").strip())
        try:
            return get_manager().start(safe)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @router.post("/stop")
    async def stop():
        return get_manager().stop()

    return router
```

- [ ] **Step 8: Wire the router and shutdown hook into app.py**

In `app.py`, near the other `app.include_router(...)` calls (around line 657), add:

```python
from routes.localmodels_routes import setup_localmodels_routes
app.include_router(setup_localmodels_routes())
```

Then in `_shutdown_event()` (around line 1180), before the final `logger.info("Application shutdown complete")`, add:

```python
    # Stop any running native local model so no llama-server is orphaned.
    try:
        from src.localmodels.manager import get_manager
        get_manager().stop()
    except Exception as e:
        logger.warning(f"Local model shutdown error: {e}")
```

- [ ] **Step 9: Run the store + routes tests**

Run: `python -m pytest tests/test_localmodels_store.py tests/test_localmodels_routes.py -v --import-mode=importlib`
Expected: PASS (3 store + 5 route tests).

- [ ] **Step 10: Commit**

```bash
git add src/localmodels/store.py src/localmodels/manager.py routes/localmodels_routes.py app.py tests/test_localmodels_store.py tests/test_localmodels_routes.py
git commit -m "feat(localmodels): endpoint store, admin routes, app wiring + shutdown stop"
```

---

### Task 4: Bundle the CPU llama-server binary

Vendor a prebuilt CPU `llama-server` into the frozen bundle so `resolve_llama_binary` finds it at `<_MEIPASS>/llama/`.

**Files:**
- Create: `scripts/fetch_llama_server.py`
- Modify: `Assist.spec`, `build-windows-portable.ps1`
- Test: `tests/test_localmodels_packaging.py`

**Interfaces:**
- Consumes: `resolve_llama_binary`'s expected `<_MEIPASS>/llama/<name>` layout.
- Produces: `build_assets/llama/` populated at build time; `Assist.spec` bundles it to `llama/`.

- [ ] **Step 1: Write failing text-guard tests**

Create `tests/test_localmodels_packaging.py`:

```python
"""Guards that the llama-server bundling stays wired (no full PyInstaller run)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_spec_bundles_llama_dir():
    assert "build_assets/llama" in _read("Assist.spec")
    assert "'llama'" in _read("Assist.spec") or '"llama"' in _read("Assist.spec")


def test_build_script_fetches_llama_server():
    assert "fetch_llama_server.py" in _read("build-windows-portable.ps1")


def test_fetch_script_targets_build_assets_llama():
    src = _read("scripts/fetch_llama_server.py")
    assert "build_assets" in src and "llama" in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_localmodels_packaging.py -v --import-mode=importlib`
Expected: FAIL — `scripts/fetch_llama_server.py` does not exist and `Assist.spec` lacks the llama entry.

- [ ] **Step 3: Create the fetch script**

Create `scripts/fetch_llama_server.py`:

```python
"""Vendor a prebuilt CPU llama-server into build_assets/llama/ at build time.

Downloads a pinned llama.cpp Windows CPU release zip and extracts
llama-server.exe (plus its DLLs) into build_assets/llama/. Pinned for
reproducibility; bump LLAMA_RELEASE deliberately after verifying a build.
"""
import io
import os
import sys
import urllib.request
import zipfile

# Pinned llama.cpp release asset (Windows x64 CPU build). Update deliberately.
LLAMA_RELEASE = os.getenv(
    "LLAMA_RELEASE_URL",
    "https://github.com/ggml-org/llama.cpp/releases/download/b4589/"
    "llama-b4589-bin-win-avx2-x64.zip",
)
ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build_assets", "llama")
)


def main() -> int:
    os.makedirs(ASSET_DIR, exist_ok=True)
    print(f"Downloading llama-server from {LLAMA_RELEASE} ...")
    with urllib.request.urlopen(LLAMA_RELEASE) as resp:  # noqa: S310
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            base = os.path.basename(member)
            if not base:
                continue
            # Take the server exe and every DLL (runtime deps) at any depth.
            if base == "llama-server.exe" or base.lower().endswith(".dll"):
                with zf.open(member) as src, open(os.path.join(ASSET_DIR, base), "wb") as dst:
                    dst.write(src.read())
    exe = os.path.join(ASSET_DIR, "llama-server.exe")
    if not os.path.isfile(exe):
        print("ERROR: llama-server.exe not found in release zip", file=sys.stderr)
        return 1
    print(f"llama-server vendored into {ASSET_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Add the llama dir to Assist.spec**

In `Assist.spec`, add this entry to the `datas = [ ... ]` list (right after the `('build_assets/fastembed_cache', 'fastembed_cache'),` line):

```python
    # Bundled CPU llama-server for native local models (Phase 3a).
    ('build_assets/llama', 'llama'),
```

- [ ] **Step 5: Run the model fetch in the build script**

In `build-windows-portable.ps1`, right after the existing "Vendoring offline embedding model" block (the `& $pyExe scripts/fetch_embedding_model.py` step and its error check), add:

```powershell
Write-Step "Vendoring llama-server (CPU)"
& $pyExe scripts/fetch_llama_server.py
if ($LASTEXITCODE -ne 0) { Fail "llama-server fetch failed." }
```

- [ ] **Step 6: Run the text-guard tests**

Run: `python -m pytest tests/test_localmodels_packaging.py -v --import-mode=importlib`
Expected: PASS (3 tests).

> **Manual (not unit-tested):** the actual download + PyInstaller build are validated on a Windows build machine, per the Phase 2/3a smoke-test approach. Do NOT run `fetch_llama_server.py` or the PyInstaller build here.

- [ ] **Step 7: Commit**

```bash
git add scripts/fetch_llama_server.py Assist.spec build-windows-portable.ps1 tests/test_localmodels_packaging.py
git commit -m "build(localmodels): vendor CPU llama-server into the bundle"
```

---

### Task 5: Minimal "Local Models" UI

A modal mirroring the Cookbook modal: list `.gguf` files, Serve/Stop, live status.

**Files:**
- Create: `static/js/localModels.js`
- Modify: `static/index.html`
- Test: `tests/test_localmodels_ui.py`

**Interfaces:**
- Consumes: `/api/localmodels/models|status|serve|stop`.
- Produces: `#localmodels-modal`, an opener button `#tool-localmodels-btn`, and `static/js/localModels.js` wiring.

- [ ] **Step 1: Write failing UI-wiring guard tests**

Create `tests/test_localmodels_ui.py`:

```python
"""Text guards that the Local Models UI is wired (elements + endpoint calls)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_index_has_localmodels_modal_and_opener():
    html = _read("static/index.html")
    assert 'id="localmodels-modal"' in html
    assert 'id="tool-localmodels-btn"' in html
    assert 'src="/static/js/localModels.js"' in html or "js/localModels.js" in html


def test_localmodels_js_calls_all_endpoints():
    js = _read("static/js/localModels.js")
    for ep in ("/api/localmodels/models", "/api/localmodels/status",
               "/api/localmodels/serve", "/api/localmodels/stop"):
        assert ep in js, f"{ep} not called in localModels.js"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_localmodels_ui.py -v --import-mode=importlib`
Expected: FAIL — the modal/opener/module don't exist yet.

- [ ] **Step 3: Create the JS module**

Create `static/js/localModels.js`:

```javascript
// Minimal Local Models UI (Phase 3a): list local GGUF files, serve/stop one at
// a time, and show live status. Mirrors the Cookbook modal conventions.
(function () {
  function $(id) { return document.getElementById(id); }

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  }

  function fmtSize(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + ' GB';
    if (n >= 1e6) return (n / 1e6).toFixed(0) + ' MB';
    return (n / 1e3).toFixed(0) + ' KB';
  }

  async function refresh() {
    const statusEl = $('localmodels-status');
    const listEl = $('localmodels-list');
    if (!listEl) return;
    let status = { running: false };
    try { status = await api('/api/localmodels/status'); } catch (e) {}
    statusEl.textContent = status.running
      ? `Running: ${status.model} (port ${status.port})`
      : 'No model running';
    let data = { models: [] };
    try { data = await api('/api/localmodels/models'); } catch (e) {}
    listEl.innerHTML = '';
    data.models.forEach((m) => {
      const row = document.createElement('div');
      row.className = 'list-item';
      const label = document.createElement('span');
      label.className = 'grow';
      label.textContent = `${m.name} — ${fmtSize(m.size)}`;
      const btn = document.createElement('button');
      const isRunning = status.running && status.model === m.name;
      btn.textContent = isRunning ? 'Stop' : 'Serve';
      btn.onclick = async () => {
        btn.disabled = true;
        try {
          if (isRunning) await api('/api/localmodels/stop', { method: 'POST' });
          else await api('/api/localmodels/serve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: m.path }),
          });
        } catch (e) { alert('Local model error: ' + e.message); }
        await refresh();
      };
      row.appendChild(label);
      row.appendChild(btn);
      listEl.appendChild(row);
    });
    if (!data.models.length) {
      listEl.innerHTML = '<div class="list-item"><span class="grow">No .gguf models found. Add one to the models folder.</span></div>';
    }
  }

  function open() {
    const modal = $('localmodels-modal');
    if (modal) { modal.classList.remove('hidden'); refresh(); }
  }
  function close() {
    const modal = $('localmodels-modal');
    if (modal) modal.classList.add('hidden');
  }

  document.addEventListener('DOMContentLoaded', () => {
    const openBtn = $('tool-localmodels-btn');
    if (openBtn) openBtn.addEventListener('click', open);
    const closeBtn = $('close-localmodels-modal');
    if (closeBtn) closeBtn.addEventListener('click', close);
  });

  window.LocalModels = { open, close, refresh };
})();
```

- [ ] **Step 4: Add the modal + opener + script tag to index.html**

In `static/index.html`, add an opener entry in the tool list near `id="tool-cookbook-btn"` (copy the surrounding `list-item` structure), using:

```html
        <div class="list-item" id="tool-localmodels-btn">
          <span class="grow">Local Models</span>
        </div>
```

Add the modal near the `<!-- Cookbook Modal -->` block:

```html
  <!-- Local Models Modal -->
  <div id="localmodels-modal" class="modal hidden">
    <div class="modal-content" role="dialog" aria-label="Local Models" style="width: min(680px, 92vw); background: var(--bg);">
      <div style="display:flex;align-items:center;margin-bottom:12px;">
        <h4 style="margin:0;margin-right:auto">Local Models</h4>
        <button class="close-btn" id="close-localmodels-modal" aria-label="Close local models">✖</button>
      </div>
      <div id="localmodels-status" style="opacity:0.7;font-size:12px;margin-bottom:10px;">No model running</div>
      <div id="localmodels-list"></div>
    </div>
  </div>
```

Add the script tag near the other `static/js/*.js` includes (with the cookbook script includes):

```html
  <script src="/static/js/localModels.js"></script>
```

- [ ] **Step 5: Run the UI-wiring guard tests**

Run: `python -m pytest tests/test_localmodels_ui.py -v --import-mode=importlib`
Expected: PASS (2 tests).

> **Manual (not unit-tested):** open the app, click **Local Models**, and confirm the modal lists files and Serve/Stop work end-to-end (covered by the 3a smoke test).

- [ ] **Step 6: Commit**

```bash
git add static/js/localModels.js static/index.html tests/test_localmodels_ui.py
git commit -m "feat(localmodels): minimal Local Models modal UI"
```

---

## Appendix: Manual smoke test (real acceptance gate)

On Windows with the built app (or dev with a `llama-server` on PATH + a small GGUF):

1. Put a small `.gguf` in `%USERPROFILE%\.odysseus\data\models` (dev: `<repo>/data/models`).
2. Open **Local Models** → the file is listed → click **Serve**.
3. Within a bounded time, confirm a `local-…` endpoint appears in the chat model picker and a chat completes against it.
4. Click **Stop** → the process ends and the endpoint is removed.
5. Serve model A, then serve model B → confirm A stopped automatically (one-at-a-time).
6. Close Assist → confirm no orphaned `llama-server.exe` in Task Manager.

## Self-Review

**Spec coverage:**
- Binary resolver (spec C1) → Task 1 `resolve_llama_binary`. ✓
- Command/URL builders (C2) → Task 1. ✓
- LocalModelManager one-at-a-time lifecycle (C3) → Task 2. ✓
- Models dir + constants (C4) → Task 1 `MODELS_DIR`. ✓
- Bundling (C5) → Task 4. ✓
- Routes (C6) → Task 3. ✓
- Minimal UI (C7) → Task 5. ✓
- Endpoint registration pattern (`endpoint_kind="local"`, `local-` id) → Task 3 store. ✓
- Shutdown stop hook → Task 3 app.py. ✓
- Path-safety on serve → Task 3 routes `_validate_model_path`. ✓
- Testing (unit for pure/manager/store/routes + text-guards for build/UI + manual smoke) → each task + appendix. ✓

**Placeholder scan:** No TBD/TODO; every code step is complete. Manual-only steps (fetch/PyInstaller/GUI serve) are labeled and backed by text-guards + the smoke appendix. ✓

**Type consistency:** `status()` dict shape `{running, model, port, endpoint_id}` is identical across manager, routes fake, and tests. `register_local_endpoint(name, base_url, session_factory=None) -> str` / `unregister_local_endpoint(endpoint_id, session_factory=None)` match between store, `get_manager()` wiring, and manager injection points. `resolve_llama_binary`/`build_serve_argv`/`local_endpoint_url`/`list_gguf_models` signatures match between Task 1 and their Task 2 consumers. `<_MEIPASS>/llama/<name>` bundle path matches between Task 1 resolver, Task 4 spec `('build_assets/llama','llama')`, and the fetch script's `build_assets/llama` target. ✓

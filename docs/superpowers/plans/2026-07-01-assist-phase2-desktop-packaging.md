# Assist Phase 2 — Native Desktop Packaging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the container-free app as a native Windows desktop application — a WebView2 window launched by `Assist.exe`, delivered by an Inno Setup `Assist-Setup.exe`, with all dependencies (including the embedding model) bundled for offline use.

**Architecture:** Extract the launcher's testable logic (port/origin/readiness/frozen-cache/runtime-detection) into a pure `src/desktop_runtime.py` module; rewrite `launcher.py` as a thin GUI shell that runs `uvicorn` on a daemon thread and owns the main thread with a pywebview WebView2 window; bundle the heavy Phase-1 deps and the embedding model via a committed PyInstaller spec; wrap the output in an Inno Setup installer.

**Tech Stack:** Python 3.11+ (env is 3.14), FastAPI/uvicorn, pywebview (WebView2/EdgeChromium), PyInstaller (onedir), Inno Setup, fastembed/onnxruntime/chromadb, pytest.

## Global Constraints

- **Naming:** emit `Assist.exe`, `dist\Assist\`, `Assist-Setup.exe`; pywebview window title is exactly `Assist`. Do NOT rename internal Python modules or `ODYSSEUS_*` env vars (that is Phase 4).
- **pywebview is desktop-only:** it must NOT become a core dependency in `requirements.txt`. The Docker/server path (`uvicorn app:app`) must still run headless without pywebview installed. Import `webview` lazily, only inside the launcher's window step.
- **Import-time env:** `ALLOWED_ORIGINS` is read at `app.py` import (`app.py:128`) and `FASTEMBED_CACHE_PATH` at `src/constants.py` import (`constants.py:66`). The launcher MUST set both in `os.environ` BEFORE `from app import app`.
- **Embedding model:** bundle `sentence-transformers/all-MiniLM-L6-v2` (fastembed default, Apache-2.0, ~80MB) so RAG/memory work offline on first launch.
- **User data location:** stays in `~/.odysseus/data` (via `src/runtime_paths.py`). The installer must NOT place user data under `{app}`, so uninstall preserves it.
- **Test env:** run pytest with `--import-mode=importlib` on this machine (a global `ultralytics` `tests` package shadows the repo's otherwise). New tests live under `tests/` and carry no `slow` marker.

## File Structure

- `src/desktop_runtime.py` (new) — pure, unit-testable launcher helpers. Tasks 1–2.
- `launcher.py` (rewrite) — thin GUI shell using the helpers + pywebview. Task 3.
- `requirements-desktop.txt` (new) — desktop-only deps (`pywebview`). Task 3.
- `scripts/fetch_embedding_model.py` (new) — build-time model vendoring. Task 4.
- `Assist.spec` (renamed from `Odysseus.spec` + edits) — committed PyInstaller spec. Task 4.
- `build-windows-portable.ps1` (edit) — builds via the committed spec, Assist naming, model fetch. Task 4.
- `installer/Assist.iss` (new) + `build-installer.ps1` (new) — installer. Task 5.
- Tests: `tests/test_desktop_runtime.py` (Tasks 1–2), `tests/test_desktop_packaging_assets.py` (Tasks 4–5).

---

### Task 1: Desktop runtime — networking & lifecycle helpers

Pure helpers for choosing a port, building the local origin/URL, augmenting `ALLOWED_ORIGINS`, constructing the uvicorn server, and polling readiness. No GUI, no `webview` import.

**Files:**
- Create: `src/desktop_runtime.py`
- Test: `tests/test_desktop_runtime.py`

**Interfaces:**
- Produces:
  - `choose_port(preferred: int = 7000) -> int`
  - `local_origin(port: int) -> str`  → `"http://127.0.0.1:<port>"`
  - `augment_allowed_origins(origin: str, existing: str | None = None) -> str`
  - `make_uvicorn_server(app, host: str, port: int, log_level: str = "info")` → a `uvicorn.Server`
  - `wait_for_server_ready(url: str, timeout: float = 45.0, interval: float = 0.25, opener=None, sleep=time.sleep, now=time.monotonic) -> bool`

- [ ] **Step 1: Write failing tests**

Create `tests/test_desktop_runtime.py`:

```python
"""Unit tests for the pure desktop-launcher helpers (no GUI, no network)."""
import socket

import src.desktop_runtime as dr


def test_choose_port_returns_preferred_when_free():
    # Grab a definitely-free port, release it, then ask for it.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    free = s.getsockname()[1]
    s.close()
    assert dr.choose_port(free) == free


def test_choose_port_falls_back_when_preferred_busy():
    # Occupy a port, then ask for it — must get a different, usable port.
    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    busy_port = busy.getsockname()[1]
    try:
        chosen = dr.choose_port(busy_port)
        assert chosen != busy_port
        # The chosen port must itself be bindable.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", chosen))
        probe.close()
    finally:
        busy.close()


def test_local_origin_format():
    assert dr.local_origin(7000) == "http://127.0.0.1:7000"


def test_augment_allowed_origins_appends_when_missing():
    out = dr.augment_allowed_origins("http://127.0.0.1:7000",
                                     existing="http://localhost,http://127.0.0.1")
    parts = out.split(",")
    assert "http://127.0.0.1:7000" in parts
    assert "http://localhost" in parts  # existing preserved


def test_augment_allowed_origins_no_duplicate():
    out = dr.augment_allowed_origins("http://127.0.0.1:7000",
                                     existing="http://127.0.0.1:7000")
    assert out.split(",").count("http://127.0.0.1:7000") == 1


def test_make_uvicorn_server_config():
    async def dummy_app(scope, receive, send):  # minimal ASGI callable
        pass
    server = dr.make_uvicorn_server(dummy_app, "127.0.0.1", 7123)
    assert server.config.host == "127.0.0.1"
    assert server.config.port == 7123
    assert server.should_exit is False


def test_wait_for_server_ready_succeeds_after_retries():
    calls = {"n": 0}
    def opener(url):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("not up yet")
        return 200
    ok = dr.wait_for_server_ready("http://x/api/health", timeout=10.0,
                                  interval=0.0, opener=opener,
                                  sleep=lambda _s: None,
                                  now=lambda: 0.0)
    assert ok is True
    assert calls["n"] == 3


def test_wait_for_server_ready_times_out():
    times = iter([0.0, 0.1, 0.2, 999.0])
    def opener(url):
        raise ConnectionError("never up")
    ok = dr.wait_for_server_ready("http://x/api/health", timeout=1.0,
                                  interval=0.0, opener=opener,
                                  sleep=lambda _s: None,
                                  now=lambda: next(times))
    assert ok is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_desktop_runtime.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.desktop_runtime'`.

- [ ] **Step 3: Implement the module**

Create `src/desktop_runtime.py`:

```python
"""Pure, unit-testable helpers for the native desktop launcher.

No GUI and no top-level `webview` import so this module is safe to import in
any environment (including the headless Docker/server path). The launcher
(`launcher.py`) wires these together and owns the pywebview window.
"""
import os
import socket
import time
import urllib.request


def _port_bindable(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def choose_port(preferred: int = 7000) -> int:
    """Return `preferred` if bindable on loopback, else an OS-assigned free port."""
    return preferred if _port_bindable(preferred) else _free_port()


def local_origin(port: int) -> str:
    """The loopback origin/URL the window and CORS checks use."""
    return f"http://127.0.0.1:{port}"


def augment_allowed_origins(origin: str, existing: str | None = None) -> str:
    """Return an ALLOWED_ORIGINS value that includes `origin` exactly once.

    CORS origin matching is scheme+host+port, so the window's ported origin
    (http://127.0.0.1:<port>) must be added to the unported defaults.
    """
    if existing is None:
        existing = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1")
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if origin not in parts:
        parts.append(origin)
    return ",".join(parts)


def make_uvicorn_server(app, host: str, port: int, log_level: str = "info"):
    """Build a uvicorn.Server we can run on a thread and stop via should_exit."""
    import uvicorn
    config = uvicorn.Config(app=app, host=host, port=port, log_level=log_level)
    return uvicorn.Server(config)


def _default_opener(url: str) -> int:
    with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310 (loopback only)
        return resp.getcode()


def wait_for_server_ready(url: str, timeout: float = 45.0, interval: float = 0.25,
                          opener=None, sleep=time.sleep, now=time.monotonic) -> bool:
    """Poll `url` until it answers with a non-5xx status or the deadline passes.

    `opener(url) -> int` and `sleep`/`now` are injectable for tests.
    """
    opener = opener or _default_opener
    deadline = now() + timeout
    while now() < deadline:
        try:
            code = opener(url)
            if 200 <= code < 500:
                return True
        except Exception:
            pass
        sleep(interval)
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_desktop_runtime.py -v --import-mode=importlib`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/desktop_runtime.py tests/test_desktop_runtime.py
git commit -m "feat(desktop): port/origin/readiness launcher helpers"
```

---

### Task 2: Desktop runtime — frozen bundle & WebView2 detection helpers

Add the frozen-only helpers: locate the bundled embedding-model cache inside the PyInstaller bundle, and detect the Edge WebView2 runtime. Same module, distinct concern.

**Files:**
- Modify: `src/desktop_runtime.py`
- Test: `tests/test_desktop_runtime.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `bundled_fastembed_cache() -> str | None` — path to `<_MEIPASS>/fastembed_cache` when frozen and present, else `None`.
  - `webview2_runtime_available(read_pv=None) -> bool` — `read_pv() -> str | None` is injectable.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_desktop_runtime.py`:

```python
def test_bundled_fastembed_cache_none_when_not_frozen(monkeypatch):
    monkeypatch.delattr(dr.sys, "frozen", raising=False)
    assert dr.bundled_fastembed_cache() is None


def test_bundled_fastembed_cache_returns_path_when_frozen(monkeypatch, tmp_path):
    cache = tmp_path / "fastembed_cache"
    cache.mkdir()
    monkeypatch.setattr(dr.sys, "frozen", True, raising=False)
    monkeypatch.setattr(dr.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert dr.bundled_fastembed_cache() == str(cache)


def test_bundled_fastembed_cache_none_when_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(dr.sys, "frozen", True, raising=False)
    monkeypatch.setattr(dr.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert dr.bundled_fastembed_cache() is None  # no fastembed_cache subdir


def test_webview2_available_true_for_real_version():
    assert dr.webview2_runtime_available(read_pv=lambda: "121.0.2277.0") is True


def test_webview2_available_false_when_absent():
    assert dr.webview2_runtime_available(read_pv=lambda: None) is False


def test_webview2_available_false_for_zero_version():
    assert dr.webview2_runtime_available(read_pv=lambda: "0.0.0.0") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_desktop_runtime.py -k "bundled or webview2" -v --import-mode=importlib`
Expected: FAIL — `AttributeError: module 'src.desktop_runtime' has no attribute 'bundled_fastembed_cache'`.

- [ ] **Step 3: Implement the helpers**

Add `import sys` to the imports at the top of `src/desktop_runtime.py` (so it reads `import os`, `import socket`, `import sys`, `import time`, `import urllib.request`), then append:

```python
def bundled_fastembed_cache() -> str | None:
    """Path to the embedding-model cache bundled into the frozen app, if present.

    Returns None in normal (non-frozen) runs so the app uses its default cache.
    """
    if not getattr(sys, "frozen", False):
        return None
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return None
    path = os.path.join(base, "fastembed_cache")
    return path if os.path.isdir(path) else None


# Evergreen WebView2 Runtime registration key (per Microsoft docs).
_WEBVIEW2_KEY = (r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients"
                 r"\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}")


def _read_webview2_pv() -> str | None:
    """Read the installed WebView2 runtime version ('pv') from the registry."""
    try:
        import winreg
    except ImportError:
        return None  # non-Windows
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, _WEBVIEW2_KEY) as key:
                pv, _ = winreg.QueryValueEx(key, "pv")
                if pv:
                    return pv
        except OSError:
            continue
    return None


def webview2_runtime_available(read_pv=None) -> bool:
    """True when a real Edge WebView2 runtime version is registered."""
    read_pv = read_pv or _read_webview2_pv
    pv = read_pv()
    return bool(pv) and pv != "0.0.0.0"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_desktop_runtime.py -v --import-mode=importlib`
Expected: PASS (14 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/desktop_runtime.py tests/test_desktop_runtime.py
git commit -m "feat(desktop): frozen model-cache + WebView2 runtime detection"
```

---

### Task 3: WebView2 launcher rewrite

Replace `launcher.py`'s splash/tray/browser logic with the server-thread + window-main flow using the Task 1–2 helpers. Add `pywebview` as a desktop-only dependency.

**Files:**
- Modify: `launcher.py` (full rewrite of logic; keep the `NullWriter` stdio guard)
- Create: `requirements-desktop.txt`
- Test: `tests/test_desktop_runtime.py` (one import-safety test for `launcher`)

**Interfaces:**
- Consumes: all of Task 1–2 (`choose_port`, `local_origin`, `augment_allowed_origins`, `make_uvicorn_server`, `wait_for_server_ready`, `bundled_fastembed_cache`, `webview2_runtime_available`).
- Produces: `launcher.main()` (callable entrypoint). No new symbols other tasks consume.

- [ ] **Step 1: Write the failing import-safety test**

Append to `tests/test_desktop_runtime.py`:

```python
def test_launcher_imports_without_starting_gui():
    # Importing the launcher module must NOT start a server or window
    # (all side effects live under main()/__main__). It must expose main().
    import importlib
    launcher = importlib.import_module("launcher")
    assert hasattr(launcher, "main")
    assert callable(launcher.main)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_desktop_runtime.py::test_launcher_imports_without_starting_gui -v --import-mode=importlib`
Expected: FAIL — current `launcher.py` has no `main` attribute (logic is under `if __name__ == "__main__"`).

- [ ] **Step 3: Rewrite `launcher.py`**

Replace the entire contents of `launcher.py` with:

```python
# launcher.py
"""Native desktop entrypoint for Assist.

Runs the FastAPI backend (uvicorn) on a daemon thread, then opens a native
WebView2 window (pywebview) on the main thread pointing at the local server.
Closing the window stops the server and exits. In dev (`python launcher.py`)
it behaves the same; in a frozen build it also loads the bundled embedding
model offline.
"""
import os
import sys
import threading


# Windowed (no-console) builds have no real stdout/stderr; guard library
# code that calls .write()/.isatty() so it can't crash the process.
class NullWriter:
    def write(self, text):
        pass

    def flush(self):
        pass

    def isatty(self):
        return False


if sys.stdout is None:
    sys.stdout = NullWriter()
if sys.stderr is None:
    sys.stderr = NullWriter()


def _show_error(message: str) -> None:
    """Show a blocking native error box (Windows), else print."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, "Assist", 0x10)
    except Exception:
        print(message)


def main() -> None:
    from src.desktop_runtime import (
        choose_port, local_origin, augment_allowed_origins,
        make_uvicorn_server, wait_for_server_ready,
        bundled_fastembed_cache, webview2_runtime_available,
    )

    host = os.getenv("APP_BIND", "127.0.0.1")
    port = choose_port(int(os.getenv("APP_PORT", "7000")))
    origin = local_origin(port)

    # These env vars are read at import time by app.py / src.constants, so set
    # them BEFORE importing the app.
    os.environ["ALLOWED_ORIGINS"] = augment_allowed_origins(origin)
    cache = bundled_fastembed_cache()
    if cache and not os.getenv("FASTEMBED_CACHE_PATH"):
        os.environ["FASTEMBED_CACHE_PATH"] = cache

    from app import app

    server = make_uvicorn_server(app, host, port)
    threading.Thread(target=server.run, daemon=True).start()

    if not wait_for_server_ready(origin + "/api/health", timeout=45.0):
        _show_error("Assist could not start its background service in time.")
        server.should_exit = True
        return

    if not webview2_runtime_available():
        _show_error(
            "Assist needs the Microsoft Edge WebView2 Runtime.\n\n"
            "Install it from:\n"
            "https://developer.microsoft.com/microsoft-edge/webview2/"
        )
        server.should_exit = True
        return

    import webview
    webview.create_window("Assist", origin)
    webview.start()  # blocks until the window is closed

    server.should_exit = True


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `requirements-desktop.txt`**

```
# Desktop-only dependencies for the native Windows build (Phase 2).
# NOT part of the core requirements: the Docker/server path runs headless
# without these. The build script installs this file in addition to
# requirements.txt.
pywebview
```

- [ ] **Step 5: Run the import-safety test**

Run: `python -m pytest tests/test_desktop_runtime.py -v --import-mode=importlib`
Expected: PASS (15 tests total). The launcher import must not require `webview` (it is imported lazily inside `main`).

- [ ] **Step 6: Commit**

```bash
git add launcher.py requirements-desktop.txt tests/test_desktop_runtime.py
git commit -m "feat(desktop): WebView2 window launcher (server-thread + window-main)"
```

---

### Task 4: PyInstaller spec, model vendoring, and build script

Make the frozen build bundle the Phase-1 heavy deps and the embedding model, and emit `Assist`. Convert the build script to use a committed spec.

**Files:**
- Create: `scripts/fetch_embedding_model.py`
- Rename+modify: `Odysseus.spec` → `Assist.spec`
- Modify: `build-windows-portable.ps1`
- Test: `tests/test_desktop_packaging_assets.py`

**Interfaces:**
- Consumes: `launcher.py` (spec entry script), `bundled_fastembed_cache()`'s expected path (`fastembed_cache` bundle dir).
- Produces: `dist\Assist\Assist.exe` (verified manually) and a committed `Assist.spec` other build steps parse.

- [ ] **Step 1: Write failing asset-guard tests**

Create `tests/test_desktop_packaging_assets.py`:

```python
"""Guards that the packaging assets stay wired for the Phase 2 bundle.

These parse the committed build assets as text — they catch regressions
(missing heavy-dep collection, wrong app name, model cache not bundled)
without running a full PyInstaller/Inno build.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_spec_file_named_assist_exists():
    assert (ROOT / "Assist.spec").is_file()
    assert not (ROOT / "Odysseus.spec").exists()


def test_spec_builds_assist_named_app():
    spec = _read("Assist.spec")
    assert "name='Assist'" in spec or 'name="Assist"' in spec
    assert "launcher.py" in spec


def test_spec_collects_heavy_deps():
    spec = _read("Assist.spec")
    for pkg in ("chromadb", "onnxruntime", "fastembed"):
        assert pkg in spec, f"{pkg} not collected in Assist.spec"
    # pywebview backend is easy for PyInstaller to miss.
    assert "webview" in spec


def test_spec_bundles_embedding_model_cache():
    spec = _read("Assist.spec")
    assert "fastembed_cache" in spec


def test_build_script_uses_committed_spec_and_assist_name():
    ps = _read("build-windows-portable.ps1")
    assert "Assist.spec" in ps
    assert "fetch_embedding_model.py" in ps
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_desktop_packaging_assets.py -v --import-mode=importlib`
Expected: FAIL — `Assist.spec` does not exist yet (only `Odysseus.spec`).

- [ ] **Step 3: Create `scripts/fetch_embedding_model.py`**

```python
"""Vendor the default embedding model into build_assets/fastembed_cache.

Run at build time (before PyInstaller). Populates the cache using fastembed's
own layout by triggering a real embed, so the frozen app can load it offline
when FASTEMBED_CACHE_PATH points at the bundled copy.
"""
import os
import sys

ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build_assets", "fastembed_cache")
)
MODEL = os.getenv("FASTEMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def main() -> int:
    os.makedirs(ASSET_DIR, exist_ok=True)
    os.environ["FASTEMBED_CACHE_PATH"] = ASSET_DIR
    from fastembed import TextEmbedding

    print(f"Fetching embedding model {MODEL} into {ASSET_DIR} ...")
    emb = TextEmbedding(model_name=MODEL, cache_dir=ASSET_DIR)
    # Force the model files to download+materialize in the cache.
    list(emb.embed(["warmup"]))
    print("Embedding model cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Rename and rewrite the spec**

```bash
git mv Odysseus.spec Assist.spec
```

Replace the entire contents of `Assist.spec` with:

```python
# -*- mode: python ; coding: utf-8 -*-
# Committed PyInstaller spec for the Assist native Windows build (Phase 2).
from PyInstaller.utils.hooks import collect_all

# Heavy, dynamically-imported packages PyInstaller's static analysis misses.
# collect_all pulls their submodules, data files, and native binaries.
_collected_datas = []
_collected_binaries = []
_collected_hidden = []
for _pkg in ("chromadb", "onnxruntime", "fastembed", "tokenizers"):
    _d, _b, _h = collect_all(_pkg)
    _collected_datas += _d
    _collected_binaries += _b
    _collected_hidden += _h

datas = [
    ('static', 'static'),
    ('scripts', 'scripts'),
    ('mcp_servers', 'mcp_servers'),
    ('services/hwfit/data', 'services/hwfit/data'),
    ('config', 'config'),
    ('.env.example', '.env.example'),
    # Offline embedding model (populated by scripts/fetch_embedding_model.py).
    ('build_assets/fastembed_cache', 'fastembed_cache'),
] + _collected_datas

hiddenimports = [
    'webview', 'webview.platforms.edgechromium',
] + _collected_hidden

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=_collected_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Assist',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX can corrupt onnxruntime / native DLLs; keep off for safety.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['static\\icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Assist',
)
```

- [ ] **Step 5: Update `build-windows-portable.ps1`**

Replace the "Installing build dependencies" and "Building portable exe bundle" sections (the block from `Write-Step "Installing build dependencies"` through the end of the PyInstaller invocation) with:

```powershell
Write-Step "Installing build dependencies"
& $pyExe -m pip install --upgrade pip --quiet
& $pyExe -m pip install -r requirements.txt -r requirements-desktop.txt pyinstaller
if ($LASTEXITCODE -ne 0) { Fail "Dependency install failed." }

Write-Step "Vendoring offline embedding model"
& $pyExe scripts/fetch_embedding_model.py
if ($LASTEXITCODE -ne 0) { Fail "Embedding model fetch failed." }

Write-Step "Building portable exe bundle"
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
& $pyExe -m PyInstaller --noconfirm --clean Assist.spec
if ($LASTEXITCODE -ne 0) { Fail "PyInstaller build failed." }
```

Then update the closing success messages to reference `dist\Assist` instead of `dist\Odysseus`:

```powershell
Write-Host ""
Write-Host "Build complete." -ForegroundColor Green
Write-Host "Portable app folder: $PSScriptRoot\dist\Assist" -ForegroundColor Green
Write-Host "Distribute the whole folder (or zip it) so static assets and scripts stay with the exe." -ForegroundColor Green
```

- [ ] **Step 6: Run the asset-guard tests**

Run: `python -m pytest tests/test_desktop_packaging_assets.py -v --import-mode=importlib`
Expected: PASS (5 tests).

- [ ] **Step 7: Commit**

```bash
git add Assist.spec scripts/fetch_embedding_model.py build-windows-portable.ps1 tests/test_desktop_packaging_assets.py
git commit -m "build(desktop): committed Assist.spec, bundle heavy deps + embedding model"
```

> **Manual build verification (not a unit test):** on a Windows machine with build deps, run `powershell -ExecutionPolicy Bypass -File .\build-windows-portable.ps1` and confirm `dist\Assist\Assist.exe` launches a WebView2 window. The clean-machine smoke test in the appendix is the real gate.

---

### Task 5: Inno Setup installer

Wrap `dist\Assist` in `Assist-Setup.exe` with shortcuts and an uninstaller, keeping user data out of `{app}`.

**Files:**
- Create: `installer/Assist.iss`
- Create: `build-installer.ps1`
- Test: `tests/test_desktop_packaging_assets.py` (append installer guards)

**Interfaces:**
- Consumes: `dist\Assist\Assist.exe` from Task 4; `APP_VERSION` from `src/constants.py`.
- Produces: `installer/Output/Assist-Setup.exe` (verified manually).

- [ ] **Step 1: Append failing installer-guard tests**

Append to `tests/test_desktop_packaging_assets.py`:

```python
def test_installer_script_exists_and_names_assist():
    iss = _read("installer/Assist.iss")
    assert 'MyAppName "Assist"' in iss
    assert "Assist-Setup" in iss  # OutputBaseFilename
    assert r"dist\Assist\*" in iss  # bundles the built app folder


def test_installer_does_not_store_user_data_under_app():
    # User data must live in ~/.odysseus/data, never under {app}, so uninstall
    # preserves it. Guard against accidentally adding a data dir to [Files].
    iss = _read("installer/Assist.iss")
    assert "{userappdata}" not in iss
    assert ".odysseus" not in iss


def test_build_installer_script_wires_iscc_and_version():
    ps = _read("build-installer.ps1")
    assert "ISCC" in ps
    assert "Assist.iss" in ps
    assert "APP_VERSION" in ps  # version pulled from the single source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_desktop_packaging_assets.py -k installer -v --import-mode=importlib`
Expected: FAIL — `installer/Assist.iss` does not exist.

- [ ] **Step 3: Create `installer/Assist.iss`**

```iss
; Inno Setup script for the Assist native Windows installer (Phase 2).
; Version is passed in by build-installer.ps1 via /DMyAppVersion=...; the
; fallback keeps a manual `ISCC installer\Assist.iss` working.
#ifndef MyAppVersion
  #define MyAppVersion "1.0.1"
#endif
#define MyAppName "Assist"
#define MyAppExe "Assist.exe"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\Assist
DefaultGroupName=Assist
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExe}
OutputBaseFilename=Assist-Setup
OutputDir=Output
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern

[Files]
Source: "dist\Assist\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Assist"; Filename: "{app}\{#MyAppExe}"
Name: "{autodesktop}\Assist"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "Launch Assist"; Flags: nowait postinstall skipifsilent
```

Note: `Source` is relative to the `.iss` file's dir, so the compiler is invoked with the repo root as working dir (see build-installer.ps1). User data lives in `~/.odysseus/data`, never under `{app}`; there is deliberately no `[UninstallDelete]` for user data.

- [ ] **Step 4: Create `build-installer.ps1`**

```powershell
#Requires -Version 5.1
<#
  Build the Assist Windows installer. Runs the portable build first, then
  compiles installer\Assist.iss into installer\Output\Assist-Setup.exe.
  Requires Inno Setup (ISCC.exe) on PATH or at its default location.
#>
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Step($msg) { Write-Host ""; Write-Host ("==> " + $msg) -ForegroundColor Cyan }
function Fail($msg) { Write-Host ""; Write-Host ("ERROR: " + $msg) -ForegroundColor Red; exit 1 }

Write-Step "Building portable app folder"
& powershell -ExecutionPolicy Bypass -File .\build-windows-portable.ps1
if ($LASTEXITCODE -ne 0) { Fail "Portable build failed." }

Write-Step "Resolving version from src/constants.py (APP_VERSION)"
$verLine = Select-String -Path "src\constants.py" -Pattern 'APP_VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $verLine) { Fail "Could not read APP_VERSION from src/constants.py." }
$version = $verLine.Matches[0].Groups[1].Value
Write-Host ("Version: " + $version)

Write-Step "Locating Inno Setup (ISCC.exe)"
$iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    foreach ($c in @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
                     "$env:ProgramFiles\Inno Setup 6\ISCC.exe")) {
        if (Test-Path $c) { $iscc = $c; break }
    }
}
if (-not $iscc) { Fail "Inno Setup (ISCC.exe) not found. Install Inno Setup 6." }

Write-Step "Compiling installer"
& $iscc "/DMyAppVersion=$version" "installer\Assist.iss"
if ($LASTEXITCODE -ne 0) { Fail "Inno Setup compile failed." }

Write-Host ""
Write-Host "Installer built: $PSScriptRoot\installer\Output\Assist-Setup.exe" -ForegroundColor Green
```

- [ ] **Step 5: Run the installer-guard tests**

Run: `python -m pytest tests/test_desktop_packaging_assets.py -v --import-mode=importlib`
Expected: PASS (8 tests total).

- [ ] **Step 6: Commit**

```bash
git add installer/Assist.iss build-installer.ps1 tests/test_desktop_packaging_assets.py
git commit -m "build(desktop): Inno Setup installer (Assist-Setup.exe)"
```

---

## Appendix: Manual clean-machine smoke test (the real acceptance gate)

Run on a clean Windows 11 VM with **no Python and no Docker installed**:

1. Copy `Assist-Setup.exe` to the VM; run it. Confirm: installs to Program Files, creates Start Menu + desktop shortcuts, appears in "Add/Remove Programs" as "Assist".
2. **Disable the VM's network.** Launch Assist from the desktop shortcut. Confirm: a native window titled **Assist** opens (own taskbar entry, no browser chrome) within a reasonable time.
3. In the app: create a note/document and exercise **RAG + semantic memory** — confirm they work offline (bundled embedding model; no network).
4. **Re-enable network.** Run a web search — confirm keyless DuckDuckGo results return.
5. Close the window — confirm the process exits (no lingering `Assist.exe` in Task Manager).
6. Relaunch — confirm prior data persisted (data in `%USERPROFILE%\.odysseus\data`).
7. Uninstall via Add/Remove Programs — confirm the app is removed but `%USERPROFILE%\.odysseus\data` remains.
8. If the window fails to open: verify the Edge WebView2 Runtime dialog appears when the runtime is absent (test on a VM without it), and that the readiness-timeout error box appears if the server can't start.

## Self-Review

**Spec coverage:**
- Spec Component 1 (WebView2 launcher) → Task 3 (+ helpers in Tasks 1–2). ✓
- Spec Component 2 (port + origin) → Task 1 (`choose_port`, `local_origin`, `augment_allowed_origins`) + Task 3 wiring. ✓
- Spec Component 3 (bundled model + frozen cache path) → Task 2 (`bundled_fastembed_cache`) + Task 4 (`fetch_embedding_model.py`, spec `datas`) + Task 3 (env set). ✓
- Spec Component 4 (PyInstaller spec/build) → Task 4. ✓
- Spec Component 5 (WebView2 runtime check) → Task 2 (`webview2_runtime_available`) + Task 3 wiring. ✓
- Spec Component 6 (Inno Setup installer) → Task 5. ✓
- Spec Testing (unit for pure pieces + manual smoke) → Tasks 1–2 unit tests, Tasks 4–5 asset guards, Appendix smoke test. ✓
- Naming decision (Assist.exe now) → Task 4 spec/build, Task 5 installer, Task 3 window title. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full content. Manual build steps are explicitly labeled as manual (they cannot be unit-tested) and backed by the smoke-test appendix and text-guard tests. ✓

**Type consistency:** Helper names/signatures are identical between the Interfaces blocks, the implementations, the launcher's import list (Task 3), and the tests. `bundled_fastembed_cache()` bundle dir name `fastembed_cache` matches the spec `datas` target and the launcher env wiring. `Assist.spec` / `Assist-Setup.exe` / window title `Assist` are consistent across Tasks 3–5. ✓

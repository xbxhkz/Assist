# Assist Phase 2 — Native Desktop Packaging (Design)

**Date:** 2026-07-01
**Status:** Approved for planning
**Depends on:** Phase 1 (de-Docker the runtime) — merged to `dev`.
**Parent project:** [[assist-native-windows-project]] — see `2026-07-01-assist-native-windows-phase1-design.md` for the overall 4-phase framing and product decisions.

---

## Goal

Turn the container-free app (post Phase 1) into a **native Windows desktop application**: a WebView2 window launched from `Assist.exe`, delivered by an `Assist-Setup.exe` (Inno Setup) installer, with every dependency — including the local embedding model — bundled so it runs offline on a clean Windows machine with no Python, no Docker, and no manual setup.

Success criteria:
- `Assist-Setup.exe` installs to Program Files with Start Menu + desktop shortcuts and an uninstaller.
- Launching `Assist.exe` shows a native **WebView2 window** titled "Assist" (own taskbar entry, no browser chrome).
- On a clean Windows machine with no Python/Docker and **no network**, chat (API models aside), RAG, and semantic memory work immediately — the embedding model is bundled.
- Web search works (keyless DuckDuckGo default from Phase 1).
- User data persists across upgrades/uninstall in `~/.odysseus/data`.

## Existing scaffolding this phase evolves

The repo already ships a portable-build harness that this phase modifies rather than replaces:
- `launcher.py` — tkinter splash + pystray tray + `webbrowser.open` + uvicorn (browser+tray approach).
- `Odysseus.spec` / `build-windows-portable.ps1` — PyInstaller onedir (`--noconsole`, icon, data files). `hiddenimports` is empty and predates Phase 1's heavy deps.
- `src/runtime_paths.py` — already frozen-aware: `get_app_root()` → `sys._MEIPASS`; `get_default_data_dir()` → `~/.odysseus/data`. Covered by `tests/test_runtime_paths.py`.

## Confirmed decisions (Phase 2)

1. **Front door:** native **pywebview WebView2 window**; drop the browser-open + system tray.
2. **Deliverable:** **Inno Setup installer** (`Assist-Setup.exe`), not a bare folder or onefile.
3. **Embedding model:** **bundled** in the installer (offline-first RAG/memory).
4. **Launcher architecture:** **server-thread + window-main** (uvicorn in a background thread; pywebview owns the main thread).
5. **Naming:** emit `Assist.exe` / `dist\Assist\` / `Assist-Setup.exe` and window title "Assist" now. Internal Python module names and `ODYSSEUS_*` env vars stay (full rename is Phase 4).

## Architecture

```
Assist.exe (frozen, windowed)
  └─ launcher.py (main thread)
       ├─ start uvicorn.Server(app) on a daemon thread   ── binds 127.0.0.1:<port>
       ├─ poll http://127.0.0.1:<port>/api/health until ready (bounded timeout)
       ├─ webview.create_window("Assist", url); webview.start()   ── main thread, blocks
       └─ on window close → server.should_exit = True → process exit
```

The window loop blocking the main thread is required by pywebview; uvicorn therefore runs off-main. Shutdown is cooperative via `uvicorn.Server.should_exit`.

## Components

### 1. WebView2 launcher (`launcher.py`)
Replace the splash/tray/browser logic with:
- Build a `uvicorn.Config(app, host, port, log_level="info")` and a `uvicorn.Server(config)`; run `server.run()` on a daemon thread.
- A **readiness poll** helper: GET `/api/health` with a short per-try timeout, retrying up to a bounded deadline; returns when the server answers (replaces `time.sleep(3.5)`).
- `webview.create_window("Assist", url)` then `webview.start()` on the main thread.
- On `webview.start()` returning (window closed), set `server.should_exit = True` and exit.
- Keep the `sys.frozen` guard so `python launcher.py` still runs the same window in dev.
- Keep the `NullWriter` stdout/stderr guard (windowed builds have no console).

### 2. Port + origin resolution
A small helper module (pure, unit-testable):
- `choose_port(preferred=7000)` — return `preferred` if bindable, else an OS-assigned free port (bind to port 0).
- `local_origin(port)` → `http://127.0.0.1:<port>`.
- Inject that origin into the process env (`ALLOWED_ORIGINS`) **before** `app` is imported, so auth/CORS accept the window's requests on whatever port was chosen.

### 3. Bundled embedding model + frozen cache path
- Vendor the fastembed default model (`sentence-transformers/all-MiniLM-L6-v2`, ONNX) into a build-time asset dir and add it to PyInstaller `datas`.
- When frozen, set `FASTEMBED_CACHE_PATH` (already read by the app) to the bundled model dir so fastembed loads locally and never reaches the network. A pure resolver (`bundled_fastembed_cache()` → path under `sys._MEIPASS` when frozen, else `None`) is unit-testable.

### 4. PyInstaller spec/build updates
Update `Odysseus.spec` and `build-windows-portable.ps1`:
- `name='Assist'` (exe + COLLECT folder).
- `hiddenimports`: `chromadb` (+ its telemetry/onnx submodules commonly missed), `onnxruntime`, `fastembed`, `webview`/`pywebview` and its platform backend (`webview.platforms.edgechromium`), and any submodules PyInstaller's static analysis misses. Prefer PyInstaller **hooks** where an official one exists; otherwise explicit `hiddenimports`.
- `datas`: keep `static, scripts, mcp_servers, services/hwfit/data, config, .env.example`; **add** the bundled embedding model dir. (UI is static files — no templates dir. Confirmed `app.py` mounts `StaticFiles(directory=STATIC_DIR)`.)
- Build deps add `pywebview` (and drop the now-unused `pystray`/`Pillow` tray requirement unless still needed for the icon).
- Reconsider `upx=True` if it destabilizes onnxruntime/native DLLs (fall back to `upx=False` for native-heavy binaries).

### 5. WebView2 runtime check
Before creating the window, detect the Edge WebView2 runtime. If missing (rare on Win11), show a clear dialog with the Microsoft download link rather than crashing. A named function so it's swappable/testable.

### 6. Inno Setup installer
Add `installer/Assist.iss`:
- Source: `dist\Assist\*` → `{app}`.
- Program Files install dir, Start Menu group, desktop shortcut to `Assist.exe`, uninstaller.
- App/exe display name "Assist"; version from a single source (see Risks).
- **Do not** place user data under `{app}` — it stays in `~/.odysseus/data`, so uninstall leaves user data intact.
- Extend `build-windows-portable.ps1` (or a sibling `build-installer.ps1`) to invoke Inno Setup's `ISCC.exe` after the PyInstaller build.

## Data flow

1. `Assist.exe` starts → `launcher.py` chooses a port, sets `ALLOWED_ORIGINS`/`FASTEMBED_CACHE_PATH` env, imports `app`.
2. uvicorn serves the FastAPI app on `127.0.0.1:<port>`; ChromaDB runs embedded under `~/.odysseus/data/chroma` (Phase 1); fastembed loads the bundled ONNX model.
3. Launcher polls `/api/health`, then opens the WebView2 window at the local URL.
4. User interacts entirely within the window; closing it stops the server and exits.

## Error handling

- **Server fails to become ready within the deadline:** show an error dialog (WebView2 or a minimal native message) instead of a blank window; log to `~/.odysseus/data/logs`.
- **Port unavailable:** handled by `choose_port` fallback.
- **WebView2 runtime missing:** explicit dialog + download link (Component 5).
- **Windowed stdio:** `NullWriter` guard retained so library `print`/`isatty` calls don't crash.

## Testing

- **Unit (pytest, following `tests/test_runtime_paths.py`):** `choose_port` (preferred-free vs fallback), `local_origin`, origin-injection helper, readiness-poll helper (against a stub), and `bundled_fastembed_cache()` frozen/non-frozen resolution. These are the pure, logic-bearing pieces.
- **Manual build smoke test (checklist in the plan):** build `Assist-Setup.exe` → install on a clean Windows VM with no Python/Docker and network disabled → launch → verify the window titles "Assist", chat UI loads, RAG/memory work offline (bundled model), DuckDuckGo search works when network is re-enabled → uninstall leaves `~/.odysseus/data` intact.
- GUI/bundle wiring (pywebview, PyInstaller, Inno Setup) is validated by the smoke test, not unit tests — they aren't meaningfully unit-testable.

## Out of scope for Phase 2

- Bundled llama.cpp + native GGUF model download catalog (Phase 3).
- Full "Assist" rename of internal identifiers / `ODYSSEUS_*` env vars / integration folders / docs (Phase 4).
- Code signing / auto-update (future; noted as a risk, not built now).

## Risks & mitigations

- **PyInstaller misses dynamically-imported submodules (chromadb/onnxruntime/fastembed).** The single most likely failure. Mitigation: explicit `hiddenimports` + official hooks; the smoke test on a clean machine is the real gate (a dev machine hides missing bundles because deps are globally installed).
- **UPX corrupts native DLLs.** Mitigation: disable UPX for native binaries if the smoke test shows load failures.
- **Version drift** between `APP_VERSION` (constants.py = 1.0.1), the spec, and the installer. Mitigation: read the version from a single source (`APP_VERSION`) in the build script and pass it to Inno Setup.
- **WebView2 runtime absent** on older Windows. Mitigation: runtime check + download link (target is Win11, where it ships).
- **Embedding model licensing/size.** all-MiniLM-L6-v2 is Apache-2.0; ~80MB is acceptable for the offline-first promise. Note the license in `ACKNOWLEDGMENTS.md` during Phase 4 docs.

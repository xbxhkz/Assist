# Assist Phase 3a — Native Serve Runtime (Design)

**Date:** 2026-07-01
**Status:** Approved for planning
**Depends on:** Phase 1 (de-Docker) + Phase 2 (desktop packaging), both merged to `dev`.
**Parent project:** [[assist-native-windows-project]]. Phase 3 (bundled local models) is decomposed into 3a (serve runtime — this doc), 3b (download catalog), 3c (UI enrichment).

---

## Goal

Assist can run a local GGUF model natively — no Docker, no tmux, no build toolchain. It launches a bundled `llama-server.exe` against a `.gguf` file on disk, health-checks it, registers it as a local OpenAI-compatible endpoint so chat/agents can use it, and manages one-model-at-a-time start/stop with a minimal UI.

Success criteria:
- Dropping a `.gguf` into the models dir and clicking **Serve** launches `llama-server`, and within a bounded time a new local endpoint appears and is usable from the chat model picker.
- **Stop** terminates the process and removes/disables that endpoint.
- Starting a second model automatically stops the first (one-at-a-time).
- On a machine with a GPU-capable `llama-server` on `PATH`, that binary is preferred; otherwise the bundled CPU binary is used.
- Closing Assist stops any running local model (no orphaned processes).

## Confirmed decisions

1. **New native subsystem**, separate from the existing tmux/bash Cookbook (which is left untouched for Docker/server deployments and is not part of the frozen Windows app's local-model path).
2. **Bundled binary:** CPU build of `llama-server.exe`, but **prefer a GPU-capable `llama-server` already on `PATH`** when present.
3. **Concurrency:** one local model at a time — starting a model stops the previous one.
4. **Scope:** 3a serves a GGUF already on disk. Downloading (3b) and richer UI (3c) are separate sub-projects. 3a includes a **minimal working UI**.

## Architecture

New self-contained package; mirrors the Phase-2 server-subprocess lifecycle pattern.

```
src/localmodels/
  runtime.py   ── pure helpers: resolve_llama_binary(), build_serve_argv(), local_endpoint_url()
  manager.py   ── LocalModelManager (stateful singleton): start(model_path) / stop() / status() / list_models()
routes/localmodels_routes.py ── POST /api/localmodels/serve, POST /api/localmodels/stop,
                                 GET /api/localmodels/status, GET /api/localmodels/models
static/js/localModels.js     ── minimal modal UI (list .gguf, Serve/Stop, live status)
static/index.html            ── #localmodels-modal + tool-list/rail entry
scripts/fetch_llama_server.py ── build-time: vendor CPU llama-server.exe into build_assets/llama/
Assist.spec                   ── bundle build_assets/llama → llama/
src/constants.py              ── MODELS_DIR = <DATA_DIR>/models
```

## Components

### 1. Binary resolver — `runtime.py` (pure)
`resolve_llama_binary(path_lookup=shutil.which, frozen_base=None) -> str`: return a GPU-capable `llama-server`/`llama-server.exe` from `PATH` if found; else the bundled CPU binary at `<frozen_base>/llama/llama-server.exe` when frozen (`frozen_base` defaults to `sys._MEIPASS`); else a dev fallback (`build_assets/llama/llama-server.exe` under the repo). Raise a clear error if none resolves. `path_lookup`/`frozen_base` injectable → unit-testable without a real binary.

### 2. Command + URL builders — `runtime.py` (pure)
- `build_serve_argv(binary, model_path, port, ctx_size=4096, host="127.0.0.1") -> list[str]`: the llama-server argv (`--model`, `--host`, `--port`, `--ctx-size`, and OpenAI-compat flags). Loopback host only.
- `local_endpoint_url(port) -> str` → `http://127.0.0.1:<port>/v1`.
Both pure/tested.

### 3. LocalModelManager — `manager.py` (stateful singleton)
Injectable dependencies (`spawn`, `endpoint_store`, `port_chooser`, `readiness`) so behavior is testable without a real process or DB.
- `start(model_path) -> status`: (1) if a model is running, `stop()` it first; (2) `port = port_chooser()` (reuse `desktop_runtime.choose_port`); (3) `spawn(build_serve_argv(...))`; (4) `readiness(local_endpoint_url(port) + "/models")` (reuse `desktop_runtime.wait_for_server_ready`); on failure, kill + raise; (5) upsert a `ModelEndpoint(endpoint_kind="local", base_url=local_endpoint_url(port), name=<model filename>, is_enabled=True)` via `endpoint_store`; record `{model_path, pid, port, endpoint_id}`.
- `stop() -> status`: terminate the process (graceful then kill), disable/remove the endpoint, clear state.
- `status() -> dict`: `{running: bool, model, port, endpoint_id}`.
- `list_models() -> list`: `.gguf` files under `MODELS_DIR` (name + size).
- Registered to `stop()` on FastAPI shutdown and on launcher window close.

Follows the existing endpoint-upsert pattern in `routes/cookbook_routes.py:1162` (endpoint_kind, base_url, auto model discovery via `/v1/models`).

### 4. Models dir + constants
`MODELS_DIR = <DATA_DIR>/models` (created on demand). Frozen data dir is `~/.odysseus/data` (Phase 1/2), so models persist across app upgrades.

### 5. Bundling
`scripts/fetch_llama_server.py` (build-time, network): download a pinned prebuilt llama.cpp **Windows CPU** release into `build_assets/llama/` (`llama-server.exe` + required DLLs). `Assist.spec` adds `('build_assets/llama', 'llama')` to `datas`; `build-windows-portable.ps1` runs the fetch before PyInstaller (same pattern as the Phase-2 embedding model). `build_assets/` is already gitignored.

### 6. Routes — `routes/localmodels_routes.py`
- `GET /api/localmodels/models` → `list_models()`.
- `POST /api/localmodels/serve {model_path}` → `start()` (validates `model_path` is a `.gguf` under `MODELS_DIR` — reject traversal/outside paths).
- `POST /api/localmodels/stop` → `stop()`.
- `GET /api/localmodels/status` → `status()`.
Admin-guarded consistent with existing cookbook/model routes.

### 7. Minimal UI — `static/js/localModels.js` + `index.html`
A `#localmodels-modal` (following the `#cookbook-modal` structure) opened from a tool-list/rail entry. The module lists `.gguf` files (`GET /models`), shows **Serve**/**Stop** buttons and live **status** (running model + port) by polling `GET /status`. Reuses existing `style.css` classes (`modal`, `list-item`, buttons) — no new CSS system, no framework. Deliberately plain; 3c enriches it.

## Data flow

1. User drops `model.gguf` into `~/.odysseus/data/models` (3b will automate this).
2. Opens Local Models → sees the file → clicks **Serve**.
3. Manager picks a free loopback port, launches the resolved `llama-server` on it, polls `/v1/models` until ready.
4. Manager upserts a `local` `ModelEndpoint` at `http://127.0.0.1:<port>/v1`; the chat model picker now lists the model (auto-discovered).
5. **Stop** (or app close) kills the process and disables the endpoint.

## Error handling
- **Binary unresolvable:** clear error surfaced to the UI ("no llama-server found").
- **Model file missing / outside MODELS_DIR:** 400, no launch.
- **Server never becomes ready:** kill the process, surface a timeout error, don't register a dead endpoint.
- **Process dies while running:** `status()` reflects not-running; endpoint probe (existing) marks it offline.
- **App shutdown:** manager `stop()` runs so no orphaned `llama-server.exe`.

## Testing
- **Unit (pytest, `--import-mode=importlib`):** `resolve_llama_binary` (PATH-GPU vs bundled-CPU vs dev vs none), `build_serve_argv` (flags/host/port), `local_endpoint_url`, and `LocalModelManager` state transitions (start→running, one-at-a-time replacement, stop, readiness-failure cleanup) using an injected fake `spawn`/`endpoint_store`/`readiness`. Path-traversal rejection in the serve route.
- **JS:** a small node-backed test for `localModels.js` render/state if it fits the existing `*_js.py` harness; otherwise covered by the manual smoke test.
- **Manual smoke test (real gate, needs Windows + a small GGUF):** place a tiny GGUF in the models dir → Serve → confirm the endpoint appears and a chat completes against it → Stop → confirm the process and endpoint are gone → close app → confirm no orphan process. The actual `llama-server.exe` launch cannot run in the dev/CI environment here.

## Out of scope for 3a
- Download catalog + native downloader (3b).
- Enriched "Local Models" UI — hardware-fit recommendations, progress, per-model management (3c).
- GPU-bundled llama.cpp builds; multiple concurrent models; changes to the existing tmux Cookbook.

## Risks & mitigations
- **Prebuilt llama.cpp binary + DLL bundling on Windows.** The fetch script must capture all required runtime DLLs; the manual smoke test on a clean machine is the gate (a dev machine may hide missing DLLs). Pin the llama.cpp release for reproducibility.
- **Untestable core in this environment.** Mitigated by isolating all logic into pure/injectable units (like Phases 1–2); the subprocess launch is the only manual-only part.
- **Orphaned processes.** Mitigated by shutdown-hook `stop()` + one-at-a-time replacement.
- **Endpoint collisions with the existing Cookbook.** Use a distinct `name`/marker for native-local endpoints so the two subsystems don't fight over the same row.

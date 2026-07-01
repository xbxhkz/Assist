# Assist — Native Windows Application (Phase 1 Design)

**Date:** 2026-07-01
**Status:** Approved for planning
**Scope of this document:** Overall project framing + detailed design for **Phase 1 (De-Docker the runtime)**. Phases 2–4 are summarized for context and will each get their own spec.

---

## Background

Odysseus is a self-hosted AI workspace currently shipped as a **4-service Docker Compose stack**:

| Service     | Role                         | Container source        |
|-------------|------------------------------|-------------------------|
| `odysseus`  | Main FastAPI/uvicorn web app | Built from `Dockerfile` |
| `chromadb`  | Vector database (RAG/memory) | `chromadb/chroma` image |
| `searxng`   | Web search engine            | `searxng/searxng` image |
| `ntfy`      | Push notifications           | `binwiederhier/ntfy`    |

The goal is to ship it as a **single native Windows application named "Assist"** — a double-clickable `.exe` with all dependencies bundled and **no Docker requirement** on the end user's machine.

## Confirmed product decisions

1. **No Docker at all.** The native app must run with zero containers. Auxiliary services are replaced with native/embedded/in-process equivalents.
2. **Local models:** ship a native model catalog with download links; the app downloads GGUF files itself and serves them via a **bundled `llama.cpp` server binary** (OpenAI-compatible API). This is a native rework of the Docker-based "Cookbook."
3. **Desktop shell:** a **native WebView2 window** (via `pywebview`), not a browser tab or console.
4. **Packaging:** **pywebview + PyInstaller (onedir)** produce `Assist.exe`. Rejected alternatives: Tauri (adds Rust toolchain + IPC boundary) and Electron (bundles a redundant Chromium).
5. **Rename depth:** **user-facing branding only** for v1. Internal identifiers (`ODYSSEUS_*` env vars, module names, admin defaults) stay to avoid regressions.

## Phased delivery

Each phase is an independent sub-project with its own spec → plan → implementation cycle. They are sequential; later phases depend on earlier ones.

- **Phase 1 — De-Docker the runtime** *(this document).* App boots and runs on Windows with zero containers: embedded ChromaDB, keyless default search, notifications degraded to an optional integration.
- **Phase 2 — Native desktop packaging.** pywebview WebView2 shell + uvicorn-in-thread + PyInstaller onedir build + Windows installer, producing `Assist.exe`.
- **Phase 3 — Bundled local models.** Ship `llama-server.exe`, a native GGUF download catalog, and launch/serve integration (native Cookbook rework).
- **Phase 4 — "Assist" rebrand.** Window title, wordmark/icon, installer name, docs headings. User-facing only.

---

## Phase 1 — De-Docker the runtime

### Goal / success criteria

Odysseus boots and runs correctly on Windows as a plain uvicorn process with **zero containers**. Success:

- App starts on a clean Windows machine with no Docker installed.
- Chat, agents, documents, email, notes, tasks, and calendar work.
- RAG and semantic memory work against an **embedded** vector store.
- Web search returns results **without any API key**.
- Nothing in the default startup path hard-requires chromadb, searxng, or ntfy containers.

This phase is backend/runtime work only. Packaging, the WebView2 window, llama.cpp, and the rename are explicitly deferred to later phases.

### Component 1 — Embedded vector store

**Current:** [`src/chroma_client.py`](../../../src/chroma_client.py) builds a `chromadb.HttpClient(host, port)` pointing at a standalone ChromaDB service, after a TCP `_port_open` probe. Dependency is `chromadb-client` (HTTP-only) in [`requirements.txt`](../../../requirements.txt).

**Change:**
- When no external ChromaDB is configured (no `CHROMADB_HOST` set), create an **embedded** `chromadb.PersistentClient(path=<DATA_DIR>/chroma)` — no server, no socket.
- Keep the `HttpClient` path as an opt-in: if `CHROMADB_HOST` *is* explicitly set, behave exactly as today (backward compatible for existing Docker users). The `_port_open` probe applies only to the HTTP path.
- Swap the dependency in `requirements.txt` from `chromadb-client` → full `chromadb`.

**Why isolated:** every caller goes through `get_chroma_client()`. [`src/rag_singleton.py`](../../../src/rag_singleton.py) and [`src/embedding_lanes.py`](../../../src/embedding_lanes.py) consume the returned client and are unchanged. Existing graceful-degradation behavior (see `test_app_initializer_memory_vector_degraded.py`) is preserved.

**Interface contract:** `get_chroma_client()` returns a live, heartbeat-verified Chroma client (embedded or HTTP) or raises `RuntimeError` with a clear hint. Consumers are unaffected by which backend is used.

### Component 2 — Keyless web search default

**Current:** `search_provider` defaults to `"searxng"` ([`src/settings.py:59`](../../../src/settings.py#L59)), which requires the searxng container at `SEARXNG_INSTANCE`. The provider registry in [`services/search/providers.py`](../../../services/search/providers.py) already implements `duckduckgo` (no key, no URL — HTML scrape via BeautifulSoup), plus brave/google_pse/tavily/serper/searxng.

**Change:**
- Default `search_provider` → `"duckduckgo"` (keyless, no external service).
- SearXNG remains fully selectable for users who point `search_url` at their own instance via settings — no capability is removed, only the default changes.
- Update [`src/service_health.py`](../../../src/service_health.py) so an unreachable SearXNG no longer marks the stack unhealthy when it is not the active provider.

**Interface contract:** the search subsystem's public API (`services/search`) is unchanged; only the default configuration value and health reporting change.

### Component 3 — Notifications

**Current:** `ntfy` is an **optional agent integration** ([`src/integrations.py:95`](../../../src/integrations.py#L95)) pointing at an ntfy server, plus a shipped container.

**Change:**
- Stop shipping the ntfy container (a packaging concern realized in later phases; noted here for completeness).
- Keep `ntfy` as an opt-in integration users can point at their own server.
- Verify nothing in the default path (service health, task scheduler) hard-requires ntfy; treat its absence as normal, not degraded.

No new notification backend is introduced in Phase 1. (Native Windows toast notifications, if desired, are a future enhancement, not part of this scope.)

### Component 4 — Config & entry defaults

- `DATABASE_URL` already defaults to SQLite (`sqlite:///./data/app.db`) — no change needed.
- Embedded Chroma path resolves under the app's `DATA_DIR`; embeddings run locally via `fastembed` (already a core dependency).
- The Docker `entrypoint.sh` / `PUID` / `PGID` / `gosu` ownership logic is Linux-container-specific and is **not** part of the native startup path. Native launch is uvicorn started directly (the in-thread launch mechanism itself is Phase 2).
- Confirm `DATA_DIR` / `LOGS_DIR` resolve to writable per-user locations on Windows.

### Testing

Extend the existing test suite (respecting the taxonomy in [`tests/README.md`](../../../tests/README.md)):

- **Embedded client test:** `get_chroma_client()` with no `CHROMADB_HOST` returns a working `PersistentClient` writing under the data dir; the HTTP path still works when `CHROMADB_HOST` is set.
- **Default provider test:** the default `search_provider` is keyless (`duckduckgo`) and resolves without an API key or URL.
- **Service health test:** with no containers running and DuckDuckGo active, `service_health` reports healthy.
- Regression guard: run the fast lane (`pytest -m "not slow"`) green.

### Out of scope for Phase 1

- PyInstaller / packaging / installer (Phase 2)
- pywebview WebView2 window and uvicorn-in-thread launcher (Phase 2)
- llama.cpp bundling, GGUF download catalog, native serving (Phase 3)
- "Assist" rename / branding / icons (Phase 4)

### Risks & mitigations

- **Embedded `chromadb` pulls heavier deps (onnxruntime, etc.) than `chromadb-client`.** Acceptable — it is required for serverless mode and needed for packaging anyway. Verified at build time in Phase 2.
- **DuckDuckGo HTML scraping can rate-limit / change markup.** Mitigation: it is only the *default*; API providers and SearXNG remain one setting away. Existing provider guard tests cover error handling.
- **Windows path/permission differences.** Mitigation: resolve data/log dirs to per-user writable locations; covered by the config test.

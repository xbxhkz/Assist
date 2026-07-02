# Assist Phase 3b — Dynamic HF Catalog + Native GGUF Downloader (Design)

**Date:** 2026-07-02
**Status:** Approved for planning
**Depends on:** Phase 3a (native serve runtime), merged to `dev`. Extends the `src/localmodels/` package.
**Parent project:** [[assist-native-windows-project]]. Phase 3 = 3a (serve — done), 3b (download — this doc), 3c (enriched UI + hardware-aware ranking).

---

## Goal

From the Local Models modal, the user searches Hugging Face live for GGUF models, picks a specific quant file, and streams it into `MODELS_DIR` with a progress bar — where it appears in the 3a model list, ready to Serve. No new dependencies (`httpx` is already core), one download at a time, gated models and rate limits handled via the existing `HF_TOKEN` setting.

Success criteria:
- Typing a search term returns matching GGUF model repos from Hugging Face.
- Selecting a repo lists its `.gguf` files with human-readable sizes.
- Clicking Download streams the file into `MODELS_DIR`, shows live progress, and can be cancelled.
- On completion the file appears in the 3a Local Models list and can be Served.
- A cancelled or failed download leaves no partial `.gguf` in the list.

## Confirmed decisions

1. **Dynamic catalog:** query the Hugging Face API live (not a shipped static list).
2. **Post-download:** the file simply lands in the 3a list; the user clicks **Serve** when ready (download and serve stay separate deliberate steps).
3. **One download at a time** (consistent with one-model-at-a-time serve).
4. **No new dependency:** stream with the existing `httpx`.
5. **Package split:** `catalog.py` (search/listing) separate from `downloader.py` (transfer).
6. **Hardware-aware ranking is deferred to 3c — and is a REQUIRED 3c deliverable** (layer `services/hwfit` scoring/filtering onto these search results). 3b's search is plain most-downloaded / text search.

## Architecture (extends the 3a `src/localmodels/` package)

```
src/localmodels/
  catalog.py     ── search_gguf_models(), list_repo_gguf_files()  [network via injectable get_json]
  downloader.py  ── DownloadManager: start()/status()/cancel()  [streaming, atomic, one-at-a-time]
routes/localmodels_routes.py (extend) ── /catalog/search, /catalog/files, /download, /download/status, /download/cancel
static/js/localModels.js (extend)     ── search box + results + per-file Download + progress bar
```

## Components

### 1. Catalog service — `src/localmodels/catalog.py`
Network isolated behind an injectable `get_json(url, headers) -> Any` (default: an `httpx.get(...).json()`), so parsing/URL/token logic is unit-testable without network.
- `search_gguf_models(query, sort="downloads", limit=30, get_json=None) -> list[dict]`
  - Hits `https://huggingface.co/api/models?filter=gguf&sort=<sort>&direction=-1&limit=<limit>&search=<query>`.
  - Returns `[{"repo": <id>, "downloads": int, "likes": int}]` (empty `query` allowed → top GGUF repos).
- `list_repo_gguf_files(repo, get_json=None) -> list[dict]`
  - Hits `https://huggingface.co/api/models/<repo>/tree/main?recursive=1`; keeps entries with `type == "file"` and path ending `.gguf`.
  - Returns `[{"filename": <basename>, "size": int, "url": "https://huggingface.co/<repo>/resolve/main/<path>"}]`.
- `_hf_headers()` attaches `Authorization: Bearer <token>` when `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` (existing settings/env) is set; else no auth.

### 2. Downloader — `src/localmodels/downloader.py`
A stateful `DownloadManager` (mirrors the 3a manager's injectable-deps style). Injectable: `http_stream(url, headers)` (yields chunks + exposes total), `dest_dir` (default `MODELS_DIR`), `opener`.
- `start(url, filename) -> dict`: reject (409-style error) if a download is already running; validate `filename` is a safe `.gguf` **basename** (no separators/traversal); stream into `<dest_dir>/<filename>.part` tracking `bytes`/`total` (from `Content-Length`); on success atomically rename `.part` → `<filename>`. Runs the transfer on a background thread.
- `status() -> {"downloading": bool, "filename": str|None, "bytes": int, "total": int|None, "pct": float|None, "error": str|None}`.
- `cancel() -> dict`: set a cancel flag so the stream loop aborts, then delete the `.part`.
- Only a fully-renamed `.gguf` ever appears to `list_gguf_models`, so partials are never Servable.
- A module-level `get_download_manager()` singleton (like 3a's `get_manager()`).

### 3. Routes — extend `routes/localmodels_routes.py` (admin-guarded)
- `GET /api/localmodels/catalog/search?q=&sort=` → `search_gguf_models(...)`.
- `GET /api/localmodels/catalog/files?repo=` → `list_repo_gguf_files(...)`.
- `POST /api/localmodels/download {url, filename}` → `get_download_manager().start(...)`. Validate `url` host is `huggingface.co` (no arbitrary-URL fetch) and `filename` is a safe `.gguf` basename.
- `GET /api/localmodels/download/status` → `status()`.
- `POST /api/localmodels/download/cancel` → `cancel()`.

### 4. UI — extend `static/js/localModels.js` + the modal
Add to the Local Models modal: a search input + Search button; a results list (repo — downloads); clicking a result expands its `.gguf` files (filename — size) each with a **Download** button; a progress bar that polls `/download/status` while a download runs, with a Cancel button; on completion, call the existing `refresh()` so the file shows in the Serve list. Files already present in `MODELS_DIR` are marked "downloaded" and their Download button disabled. Reuses existing CSS classes — no new framework.

## Data flow

1. User types a query → `GET /catalog/search` → repo results.
2. User clicks a repo → `GET /catalog/files?repo=…` → `.gguf` files with sizes + resolve URLs.
3. User clicks Download on a file → `POST /download {url, filename}` starts a background stream into `MODELS_DIR/<file>.part`.
4. UI polls `GET /download/status` → progress bar; Cancel → `POST /download/cancel`.
5. On completion `.part` → `.gguf`; UI `refresh()` lists it; user clicks **Serve** (3a).

## Error handling
- **HF API / network error** (search or files): return an empty list + a surfaced error message; never crash the route.
- **Download start while one is running:** rejected with a clear "a download is already in progress" error.
- **Unsafe filename / non-huggingface URL:** 400, no fetch.
- **Transfer failure / cancel / disk-full:** `.part` deleted, `error` set in status, no partial `.gguf`.
- **Duplicate:** existing `<filename>.gguf` → UI marks downloaded, Download disabled (no clobber).

## Testing
- **Unit (pytest, `--import-mode=importlib`):**
  - `catalog.py`: `search_gguf_models` + `list_repo_gguf_files` with an injected fake `get_json` — assert the request URLs, `.gguf` filtering, size/URL construction, and token-header inclusion/omission.
  - `downloader.py`: `DownloadManager` start → progress accrues → complete → atomic rename to `.gguf`; cancel deletes the `.part`; one-at-a-time rejection; unsafe-filename rejection — all with a fake chunk-yielding stream, no network.
  - routes: catalog + download endpoints via `TestClient` + a fake manager/catalog; the non-huggingface-URL and unsafe-filename rejections.
- **Text-guard:** `localModels.js` calls the new endpoints; modal has the search/results/progress elements.
- **Manual smoke (real gate, needs network):** search a real term, download a small real GGUF, confirm it lands in the list and Serves; cancel mid-download and confirm no partial file. Real HF traffic isn't exercised in CI.

## Out of scope for 3b (explicitly 3c)
- **Hardware-aware ranking/filtering of results — REQUIRED in 3c** (apply `services/hwfit` `rank_models`/`analyze_model` to annotate/sort search hits by fit for the user's machine).
- Per-model delete + disk-usage management in the UI.
- Download resume across app restarts (a fresh download restarts the transfer).

## Risks & mitigations
- **HF API shape/rate limits.** Isolate all HF calls behind `get_json`/`http_stream` so a shape change is a one-file fix; attach `HF_TOKEN` for higher limits; short-circuit errors to empty results. The manual smoke test on real HF is the acceptance gate.
- **Untestable transfer in CI.** Mitigated by injecting a fake streaming source so progress/cancel/atomic-rename are unit-tested; only real network is manual.
- **Partial files polluting the Serve list.** Mitigated by `.part`→`.gguf` atomic rename — `list_gguf_models` only sees completed files.
- **Arbitrary-URL fetch via the download route.** Mitigated by validating the URL host is `huggingface.co` and the filename is a safe `.gguf` basename.

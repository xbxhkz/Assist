# Assist Phase 3c — Enriched, Hardware-Aware Local Models UI (Design)

**Date:** 2026-07-02
**Status:** Approved for planning
**Depends on:** Phase 3a (serve runtime) + 3b (HF catalog + downloader), both merged to `dev`. Extends `src/localmodels/`.
**Parent project:** [[assist-native-windows-project]]. Completes Phase 3 (3a serve, 3b download, 3c enriched UI).

---

## Goal

Make the Local Models modal hardware-aware and self-service: detect the machine, badge every downloadable GGUF by whether it fits, recommend models suited to the machine, and let the user delete downloaded models and see disk usage. hwfit is used **read-only** (no changes to `services/hwfit/` or the Cookbook).

Success criteria:
- A hardware header shows detected RAM and GPU/VRAM.
- Each GGUF file in search results shows a fit badge — **Fits on GPU** (green) / **Fits in RAM** (yellow) / **Too big** (red) — and files are sorted best-fit-first.
- A "Recommended for your machine" panel lists hwfit-ranked models; clicking one pre-fills the search box and runs the search.
- Each downloaded model has a **Delete** button (stops it first if serving); a header shows model count + total disk usage.

## Confirmed decisions

1. **Both** hardware-aware features: per-file **fit badges** AND a **recommendations panel**.
2. **Recommendation → download bridge:** clicking a recommendation **pre-fills the search box and runs the 3b search** (one download path; no auto-quant-picking).
3. **Hybrid fit estimate:** combine **size-based** (GGUF file bytes + KV overhead — always available, the quantized file is the weights) **and** hwfit's param-based `estimate_memory_gb` (when a param token like `7B` is parseable from the filename). Take the **conservative max** of the two so the app never over-promises "it fits."
4. **Include per-model management:** delete + disk usage.
5. hwfit used **read-only**; Cookbook untouched.

## Architecture (extends `src/localmodels/` + routes + the modal)

```
src/localmodels/
  hardware.py   ── get_hardware() [cached hwfit.detect_system wrapper]
                   fit_for_file(file, hardware, ctx) [hybrid size + hwfit estimate]
                   _verdict(needed_gb, hardware) [pure]
                   recommend_models(limit) [hwfit.rank_models wrapper]
  manager.py (extend) ── delete_model(filename) [stops it first if serving]
routes/localmodels_routes.py (extend) ── GET /hardware, GET /recommendations, POST /delete;
                                          /catalog/files annotated with fit; /models adds disk_bytes
static/js/localModels.js (extend)     ── hardware header, fit badges + sort, recommendations panel,
                                          per-model Delete, fmt dedup, Enter-to-search
```

## Components

### 1. Hardware detection — `hardware.py`
`get_hardware(detect=None) -> dict`: call `hwfit.detect_system()` (injectable `detect` for tests), normalize to `{"ram_gb": float, "has_gpu": bool, "gpu_name": str|None, "vram_gb": float}` (reading `available_ram_gb`, `has_gpu`, `gpu_name`, `gpu_vram_gb`). Cache the result process-wide (detection is slow); a `refresh=True` bypasses the cache. Route `GET /api/localmodels/hardware` returns it.

### 2. Hybrid fit — `hardware.py` (pure logic, hwfit read-only)
- `_infer_params_b(name) -> float | None`: regex a param token (`\b(\d+(?:\.\d+)?)\s*[bB]\b`) from the filename (e.g. `Qwen2.5-7B-…` → 7.0); `None` if absent.
- `fit_for_file(file, hardware, ctx=4096) -> dict`: `size_gb = size/1e9`; `size_needed = size_gb + _kv_overhead(ctx)`. If params inferable: `param_needed = hwfit.estimate_memory_gb({"parameter_count": f"{params}B"}, hwfit.infer_quantization_from_name(filename), ctx)`, else `None`. `needed_gb = max(size_needed, param_needed or 0)`. Returns `{"verdict", "needed_gb", "size_gb", "param_estimate_gb", "quant"}`.
- `_verdict(needed_gb, hardware) -> "gpu"|"ram"|"too_big"` (pure): `gpu` if `has_gpu and needed_gb <= vram_gb`; else `ram` if `needed_gb <= ram_gb`; else `too_big`.
The `/catalog/files` route annotates each file with `fit_for_file(...)`; the UI badges + sorts by verdict (gpu > ram > too_big) then size.

### 3. Recommendations — `hardware.py` + route + UI
`recommend_models(limit=8, rank=None) -> list`: call `hwfit.rank_models(get_hardware_system(), limit=limit)` (injectable `rank`), return `[{"name", "score"}]`. `get_hardware_system()` returns the raw hwfit system dict (from the same cached `detect_system`) that `rank_models` expects. Route `GET /api/localmodels/recommendations`. UI: a "Recommended for your machine" panel; clicking a row sets the search input to the model name and calls `doSearch()`.

### 4. Per-model management
- `manager.delete_model(filename)`: validate safe `.gguf` basename inside `MODELS_DIR`; if it's the currently-serving model (`status().model == filename`), `stop()` first; then remove the file. Returns updated status.
- Route `POST /api/localmodels/delete {filename}` (reuse the path-safety validator).
- `/models` response gains `disk_bytes` (sum of `.gguf` sizes). UI header: "N models · X GB"; each downloaded model gets a **Delete** button (confirm first).

### 5. Polish
- Collapse duplicate `fmtSize` (3a) / `fmtBytes` (3b) into a single `fmtBytes`.
- Enter key in the search input triggers `doSearch`.

## Data flow

1. Modal opens → `GET /hardware` → header; `GET /recommendations` → panel; `GET /models` → downloaded list + disk usage.
2. User searches (or clicks a recommendation, which pre-fills + searches) → `GET /catalog/files` returns files each annotated with `fit` → UI badges + sorts.
3. User downloads (3b) → file lands in the list. User can **Serve** (3a) or **Delete** (3c).

## Error handling
- **Hardware detection fails/slow:** return a safe default (`{ram_gb: 0, has_gpu: false, vram_gb: 0}`); fit then reports `too_big`/`ram` conservatively and the header shows "unknown". Never block the modal.
- **rank_models fails:** recommendations panel shows empty/"unavailable"; search still works.
- **Delete of a missing/unsafe/serving file:** unsafe/outside → 400; serving model is stopped first; missing file → idempotent success.
- **fit on a file with no size / unparseable name:** size-only path; unknown params → size-based verdict.

## Testing
- **Unit:** `_verdict` thresholds (gpu/ram/too_big with exact numbers); `_infer_params_b` (`7B`, `1.5B`, none); `fit_for_file` takes the conservative max of size vs param estimate (with an injected/known hwfit estimate); `get_hardware` normalization + caching with an injected `detect`; `recommend_models` with an injected `rank`; `manager.delete_model` (stops-if-serving via fake, removes file, rejects unsafe/outside names).
- **Routes:** `/hardware`, `/recommendations` (injected rank), `/delete` (fake manager), `/catalog/files` fit annotation, `/models` disk_bytes — via TestClient with fakes.
- **Text-guard UI:** hardware header, badge classes, recommendations panel, delete button, Enter-to-search wiring present; `/hardware`/`/recommendations`/`/delete` called.
- **Manual smoke (real gate):** on a real machine, header shows correct RAM/GPU; badges match reality (a too-large model reads red); recommendations are sensible; clicking one searches; delete removes a file and updates disk usage.

## Out of scope
- Download resume across restarts.
- Any change to `services/hwfit/` or the Cookbook.
- Multi-GPU sizing beyond total VRAM; per-layer offload modeling.

## Risks & mitigations
- **hwfit `detect_system` cost / platform variance.** Cache it; inject for tests; safe-default on failure so the modal never blocks.
- **Param inference from filenames is imperfect.** That's exactly why fit is a **hybrid taking the conservative max** — size-based is always correct as a floor; the param estimate only ever makes the verdict *more* conservative, never less.
- **`estimate_memory_gb` API drift** (it's hwfit-internal). Call it behind `fit_for_file` so a signature change is a one-function fix; unit-test `_verdict` independently of hwfit's numbers.
- **rank_models returns catalog families not GGUFs.** Bridged by pre-filling the 3b search (decision 2) rather than pretending a family is directly downloadable.

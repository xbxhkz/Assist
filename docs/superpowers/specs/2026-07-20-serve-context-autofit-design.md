# Serve-time Context Auto-Fit (Automatic Hardware Optimizer v1) — Design

**Goal:** Replace the blind fixed **16384** serve context with one auto-fitted to the
chosen GGUF's *trained* context and the detected VRAM, so a small card isn't crippled by
an oversized KV cache (keeping more layers on GPU → faster), a capable box isn't needlessly
capped, and a model trained for 4096 doesn't waste memory pretending it has 16384.
Deliberately does **not** touch GPU-layer fitting.

**Scope:** The v1 slice of the roadmap's "Automatic Hardware Optimizer". One cohesive
feature: extend the GGUF metadata reader, add a pure context recommender, and wire it into
the native Local Models serve path. Later/optional (explicit non-goals below): flash-attn,
GPU-layer/batch/tensor-split tuning, live benchmarking, the Cookbook serve path.

---

## Background — why this is the right slice

- **The native Local Models serve path hardcodes context.** `LocalModelManager.start()`
  (`src/localmodels/manager.py`) calls `build_serve_argv(binary, model_path, port,
  device=…, mmproj=…)` WITHOUT a `ctx_size`, so every model is served at the default
  **16384** (`src/localmodels/runtime.py:59` `build_serve_argv(..., ctx_size=16384, …)`),
  regardless of the model's trained limit or the machine's VRAM.
- **GPU layers are already auto-fit — and must NOT be overridden.** `build_serve_argv`
  deliberately sets no `-ngl` (`runtime.py:84-89`): the bundled Vulkan llama.cpp build
  auto-fits GPU layers to free VRAM, and passing `-ngl` *disables* that fitter — observed
  live as a hard OOM on the 6GB card. So auto-computing GPU layers would be a regression.
- **A larger context never OOMs here.** Because llama.cpp auto-offloads layers to fit the
  requested KV cache, an oversized context just pushes more layers to CPU (slower), it does
  not crash. So context-fitting is a **safe speed optimization**, not crash-avoidance — the
  worst case of a bad pick is "a little slower", which keeps the risk profile low.
- **The fit engine already models memory-vs-context**, but for model *ranking*
  (`services/hwfit/fit.py` `estimate_memory_gb(model, quant, ctx)`, `_try_quant_at` halves
  ctx to fit) — it feeds the Cookbook ranking UI, not the serve command. This feature applies
  the same idea at serve time for the *chosen local GGUF*.
- **Hardware facts are available**: `services/hwfit/hardware.detect_system()` reports
  `has_gpu`, `gpu_vram_gb`, `available_ram_gb`, etc.
- **GGUF numeric metadata is available but unread**: `src/gguf_meta.py` walks the GGUF KV
  block but currently returns only `general.architecture`. The block also holds
  `<arch>.context_length`, `<arch>.block_count`, `<arch>.attention.head_count`,
  `<arch>.attention.head_count_kv`, `<arch>.embedding_length` — everything needed for a
  KV-cache estimate.

## Architecture

A pure recommender over (model metadata, hardware facts) → recommended context, plus a
metadata-reader extension, wired into the one serve path that currently hardcodes 16384.

- **`src/gguf_meta.py`** (extend) — `read_gguf_metadata(path) -> dict`. Reuses the existing
  KV-block walk to capture the numeric fields alongside the architecture. Never raises;
  returns a partial dict (only the keys it found).
- **`src/localmodels/serve_tuning.py`** (new) — pure functions with no I/O:
  `estimate_kv_bytes_per_token(meta)` and `recommend_context(meta, hardware, *, requested=None)`.
- **`src/localmodels/manager.py`** (wire) — `start()` computes the context from the model
  file's metadata + detected hardware and passes it to the existing
  `build_serve_argv(ctx_size=…)` parameter, instead of relying on the 16384 default.

Read-only of the model file; no new dependencies; no serving-behavior change beyond the
context value (and GPU-layer auto-fit is untouched).

## The GGUF metadata reader (`read_gguf_metadata`)

Extends the existing `read_gguf_architecture` walk. Returns a dict with any of:
`{"architecture": str, "context_length": int, "block_count": int, "head_count": int,
"head_count_kv": int, "embedding_length": int}`. The keys are `<arch>.context_length` etc.,
so the walk first needs the architecture (written early as `general.architecture`) to know
the prefix; it captures scalar uint32/uint64 values by key. Bounded by the existing
`_MAX_KEYS`. Never raises (OSError/short-read/non-GGUF → `{}`), matching the module's
contract. The existing `read_gguf_architecture` stays as-is (callers unaffected).

## The recommender (`recommend_context`)

`estimate_kv_bytes_per_token(meta) -> int | None`:
KV bytes/token ≈ `2 (K and V) × block_count × head_count_kv × head_dim × 2 bytes` (f16 KV),
where `head_dim = embedding_length / head_count`. Returns `None` if the required fields are
missing (caller falls back to the heuristic).

`recommend_context(meta, hardware, *, requested=None) -> int`:
1. **Explicit override wins** — if `requested` is a positive int, return it clamped to
   `[FLOOR, HARD_CEILING]` (a caller/user choice always beats the recommender).
2. **Trained-context ceiling** — `trained = meta.get("context_length")` (fallback
   `DEFAULT_TRAINED = 8192` when unknown). The result never exceeds `trained`.
3. **VRAM budget** — with a GPU and a usable KV-bytes/token estimate: the largest "nice"
   context (from the ladder `[2048, 4096, 8192, 16384, 32768]`) whose KV cache is within
   `KV_VRAM_FRACTION` (target ~0.5) of `gpu_vram_gb`, so most VRAM stays available for
   weights/layers. When the KV estimate is unavailable, use a VRAM-tier fallback
   (e.g. `<6GB → 4096`, `<12GB → 8192`, `<24GB → 16384`, else `32768`).
4. **CPU-only** (no GPU) — bound to `CPU_CEILING = 8192` (huge context on CPU is very slow),
   and not above what `available_ram_gb` comfortably holds via the same KV estimate.
5. **Clamp + round** — final value clamped to `[FLOOR = 2048, min(trained, HARD_CEILING = 32768)]`
   and snapped down to a ladder value. Always returns a valid positive int; never raises.

Named constants (`FLOOR`, `HARD_CEILING`, `CPU_CEILING`, `DEFAULT_TRAINED`,
`KV_VRAM_FRACTION`, the context ladder, the VRAM-tier fallback table) live at module top so
the policy is tunable in one place.

## Wiring into the serve path

`LocalModelManager.start(model_path, device)` gains an injectable metadata reader and
hardware detector (default: the real `read_gguf_metadata` and `detect_system`), computes
`ctx = recommend_context(meta, hardware)`, and calls
`build_serve_argv(binary, model_path, port, ctx_size=ctx, device=device, mmproj=mmproj)`.
No `-ngl` is added (auto-fit preserved). The chosen context is recorded in the manager's
status/log so it's visible (transparency), e.g. `served ctx=8192 (auto)`.

An explicit caller-supplied context (should one ever be threaded through) is passed as
`requested=` and wins. v1 does not add a user-facing override UI — that's a later nicety.

## Error handling

The path never raises into serve startup: unreadable metadata → `{}` → VRAM-tier fallback;
`detect_system()` failure (caught) → conservative CPU-safe default (`8192`); a nonsensical
metadata value → the clamp keeps the result in range. If recommendation somehow fails, the
serve still proceeds at a safe default rather than aborting.

## Testing

All headless — no GPU, no served model, no benchmarking.

- **`recommend_context` (pure, injected dicts):** small-VRAM GPU → small ctx; large-VRAM →
  up to trained ctx; in **auto mode** (no `requested`) the result never exceeds the model's
  trained `context_length`; CPU-only → bounded by `CPU_CEILING`; missing KV fields →
  VRAM-tier fallback; a `requested=` override is honored (clamped only to `[FLOOR,
  HARD_CEILING]`, so it may intentionally exceed the trained ceiling); every auto result is
  within `[FLOOR, min(trained, HARD_CEILING)]` and is a ladder value.
- **`estimate_kv_bytes_per_token`:** a known small config → expected bytes/token; missing
  fields → `None`.
- **`read_gguf_metadata`:** build a tiny valid GGUF header in-test (bytes: `GGUF` magic,
  version, a handful of KV entries incl. `general.architecture` + `<arch>.context_length` +
  `<arch>.block_count`) and assert the dict; a truncated/non-GGUF blob → `{}` (never raises).
- **Wiring (`manager.start`):** inject a fake metadata reader + fake `detect_system` and a
  spy `build_serve_argv`/spawn; assert the argv carries the recommended `--ctx-size` and
  still no `-ngl`.
- **Manual check (owed by the user):** serve a real model and confirm the logged context and
  observed speed are sensible on the actual 6GB card — the automated tests prove the math and
  the wiring, not real-hardware tok/s.

## Non-goals (this sub-project)

- `-ngl` / GPU-layer override (auto-fit handles it; overriding regressed serving).
- **flash-attn** (llama.cpp Vulkan FA has been unreliable; a later KV-shrinking toggle).
- Batch size and multi-GPU tensor-split (multi-GPU serving is a separate pending sub-project;
  the target machine is single-GPU).
- Live tok/s benchmarking (needs a GPU + served model; not headless-testable — a later gate).
- The Cookbook serve path (`routes/cookbook_routes.py` model_serve) — a separate, advanced
  path with its own context handling.
- Quantization recommendation (already handled by the GGUF picker + `services/hwfit` fit
  engine at download/selection time).
- A user-facing override UI (v1 auto-applies + logs; manual override is honored if threaded).

# Native Local Image Generation (FLUX via bundled sd-server) — Design

**Date:** 2026-07-06
**Status:** Approved

## Goal

Let Assist **run text-to-image diffusion models locally** (FLUX.1 now, FLUX 2 when
the runtime supports it) on **CPU or GPU**, self-contained — the image twin of the
existing native local-LLM serving. A served model auto-registers as an image
endpoint so the app's current gallery/chat image generation uses it unchanged.

## Background — what already exists

- Image generation is OpenAI-compatible: the app POSTs to an image endpoint's
  `/images/generations` (`routes/gallery/gallery_routes.py`; `ModelEndpoint.model_type
  == "image"`). Any endpoint tagged `model_type="image"` is picked up automatically.
- The legacy **Cookbook** could serve FLUX.1 via `diffusers`+torch, but that path is
  tmux/Linux-oriented, needs torch installed, and has no FLUX 2 — not viable in the
  native frozen Windows app.
- The native LLM serving (`src/localmodels/`: runtime + manager + store + routes +
  native `.gguf` picker + endpoint registration) is the pattern to mirror.

## Runtime choice

**stable-diffusion.cpp `sd-server`** (bundled, like `llama.cpp`):
- Pure C/C++, GGUF weights, backends CPU / **Vulkan** / CUDA.
- **Exposes OpenAI `/v1/images/generations`** → plugs directly into the existing
  gallery/image flow with no server-side changes.
- GGUF quant (Q8→Q4) runs FLUX on ~8–10 GB GPUs; on the target 6 GB card it runs with
  CPU-offloaded text encoders/VAE (slower but works). No torch.

## Architecture — new `src/imagemodels/` subsystem

Mirrors `src/localmodels/`, one served image model at a time.

### 1. `runtime.py` (pure helpers)
- `resolve_sd_binary(device)` — prefer a user `sd-server` on PATH; else the bundled
  binary: `<_MEIPASS>/sd/sd-server.exe` (GPU=Vulkan build) or the CPU build; dev
  fallback under `build_assets/sd/`. Raises if none.
- `build_serve_argv(binary, files, port, device, host="127.0.0.1")` where `files` =
  `{diffusion_model, t5xxl, clip_l, vae}`:
  `sd-server --diffusion-model <flux.gguf> --t5xxl <t5> --clip_l <clip> --vae <vae>
  --host 127.0.0.1 --port <p>` plus **low-VRAM defaults on GPU** (keep encoders/VAE on
  CPU, flash-attention on) and `-t <threads>` on CPU. Exact flag names verified against
  the pinned sd-server in the plan.
- `local_image_endpoint_url(port)` → `http://127.0.0.1:{port}/v1`.
- `list_gguf_image_models(dir)` — list candidate diffusion `.gguf` files.

### 2. `manager.py` (`ImageModelManager`)
Same shape/lessons as `LocalModelManager`: injectable spawn/probe/port/register; one
subprocess; readiness probe of `/v1/models` (or `/health`); **size-scaled load timeout**
(image loads are slow) and **robust kill** (terminate → force process-tree kill);
`CREATE_NO_WINDOW`; stdout/stderr to `<data>/logs/sd-server.log`; fail-fast with the log
tail (surfaces "unsupported architecture" for an unsupported FLUX 2). On ready →
register the image endpoint; persist last-served for optional auto-serve. `start(files,
device)`, `stop()`, `status()`.

### 3. `store.py`
`register_image_endpoint(name, base_url)` → upsert `ModelEndpoint(model_type="image",
endpoint_kind="local", id "img-local-…")`; synchronous `/v1/models` probe to set
`cached_models` (so the model shows immediately, reusing the LLM fix); `unregister_…`.

### 4. Shared FLUX encoders + VAE
FLUX needs 4 files; the diffusion GGUF is the user's, the **T5-XXL + CLIP-L + VAE are
shared** across FLUX models. Resolution order for each aux file:
1. explicitly provided path; 2. same directory as the diffusion GGUF; 3. the shared
`<data>/image-models/encoders/` dir. A helper `download_flux_encoders()` fetches the
shared T5-XXL (Q8 GGUF), CLIP-L, and `ae.safetensors` from pinned public HF repos into
that dir, once. If any required file is unresolved, serve returns a clear 400 naming
what's missing and offering the download.

### 5. Routes — `routes/imagemodels_routes.py` (admin-guarded)
- `POST /api/imagemodels/serve` `{diffusion_model, device, t5xxl?, clip_l?, vae?}` →
  resolve the 4 files → `manager.start(...)` off the event loop (`asyncio.to_thread`);
  503 with the log tail on failure.
- `POST /stop`, `GET /status`, `GET /models` (local diffusion GGUFs + `external`/linked,
  reusing the linked-model pattern), `POST /add-external` (native picker path),
  `POST /encoders/download` + `GET /encoders/status`.
- Native picker: extend the launcher `js_api` with `pick_image_model()` (a `.gguf`/
  `.safetensors` open dialog), reusing the `pick_gguf` mechanism.

### 6. UI — "Image Models" card in the Local Models modal
Pick a FLUX GGUF (Browse / linked list), a **CPU / GPU** toggle, and Serve. Shows the
running image model + a Stop; a one-click "Download FLUX encoders (~6 GB, once)" when
they're missing; a 6 GB fit hint. Once served, gallery/chat image generation works with
no further action.

### 7. Packaging
`scripts/fetch_sd_server.py` (like `fetch_llama_server.py`) downloads a **pinned
stable-diffusion.cpp release**: the CPU (avx2) build and the Vulkan build for Windows
x64, extracting `sd-server.exe` + required `*.dll` into `build_assets/sd/{cpu,vulkan}/`;
`Assist.spec` bundles them under `sd/`. Exact release tag + asset names pinned in the
plan after verifying availability. (CUDA build not bundled — Vulkan gives GPU accel via
the driver; a user CUDA `sd-server` on PATH is preferred when present.)

## CPU vs GPU + low-VRAM

Device is the user's choice per serve. GPU path uses the Vulkan binary with
encoders/VAE on CPU + flash-attention so a 6 GB card can host the diffusion model. CPU
path uses the CPU binary with a sensible thread count. Fit hint warns when a model likely
won't fit.

## FLUX 2 caveat

sd.cpp's supported architectures today are SD / FLUX / Wan / Qwen-Image / Z-Image — FLUX
2 is not listed. FLUX.1 ships working; FLUX 2 serves **iff** the pinned sd.cpp supports
its architecture. If it can't load the user's FLUX 2 GGUF, the manager surfaces the
error (same as the LLM "unknown architecture" path) and FLUX.1 is unaffected; FLUX 2
lights up when we bump sd.cpp to a version that supports it.

## Non-goals

- No torch/diffusers bundling. No CUDA-toolkit bundling. No image *editing*/inpainting
  UI (text-to-image first). No new gallery/generation UI — reuse the existing image flow.
  No auto-download catalog of diffusion models in v1 (the user supplies GGUFs); only the
  shared FLUX encoders/VAE are downloadable.

## Testing

Injected-fake unit tests (no real binary/GPU), mirroring the 89 localmodels tests:
- `runtime`: `build_serve_argv` includes the 4 files + device flags; `resolve_sd_binary`
  precedence.
- `manager`: serve → ready → register; readiness-timeout → kill + raise; fail-fast on
  early process exit; stop unregisters; size-scaled timeout.
- `store`: `register_image_endpoint` upserts `model_type="image"` + probes `cached_models`.
- aux-file resolution: provided > sibling > shared dir; missing → clear error.
- Manual (frozen app): serve the user's FLUX.1 GGUF on CPU and on GPU, generate an image
  via the gallery; attempt FLUX 2 and record whether sd.cpp loads it.

## Files

- Create: `src/imagemodels/{__init__,runtime,manager,store,encoders}.py`,
  `routes/imagemodels_routes.py`, `scripts/fetch_sd_server.py`,
  `tests/test_imagemodels_*.py`.
- Modify: `app.py` (include router + shutdown stop hook), `launcher.py`
  (`pick_image_model`), `Assist.spec` (bundle `sd/`), the Local Models modal
  (`static/index.html` + a small `static/js/imageModels.js`), `src/constants.py`
  (IMAGE_MODELS_DIR under the data dir).

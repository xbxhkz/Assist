# ControlNet for Local Image Generation — Design

**Goal:** Let local image generation be guided by a **control image** — serve a
diffusion model with a ControlNet loaded, then generate with a per-request
control image + strength (starting with **canny** edge control, free via bundled
`cv2`, plus bring-your-own control maps for any ControlNet type).

**Scope (v1):** ControlNet for **SDXL / SD1.5 / FLUX.1** base models (where
ControlNet models exist and are reasonably sized). **Canny** preprocessing
in-app (cv2) + **bring-your-own control image** for other types (depth/pose/…).
ControlNet-model management (download/list/delete) mirrors the LoRA manager.

**Explicitly out of scope / honest limits:**
- **FLUX.2 klein has no ControlNet yet** (too new) — control works for the
  architectures above, and klein works automatically the day a FLUX.2 ControlNet
  ships (the plumbing is architecture-agnostic), but v1 cannot demo control on klein.
- No bundled **depth/pose preprocessor models** (MiDaS/DWpose) — those control
  types work only via a user-supplied preprocessed control image in v1.
- Multi-ControlNet (several controlnets at once) is out.

---

## Feasibility (spike outcome)

The gate that parked this before is cleared: the bundled `sd-server` binary
(build_assets/sd) contains the **underscored JSON field names** `control_image`
(11×), `control_strength` (10×), `control_net` — the HTTP-API request-field form
(CLI flags use hyphens), appearing in the same pattern as `init_image` (18×),
which the gallery already sends per-request for img2img. So the running server
accepts a **per-request `control_image` + `control_strength`**, with the
ControlNet **model** loaded at serve via `--control-net`. `--help` confirms
`--control-net`, `--control-image`, `--control-strength`, and
`--backend controlnet=cpu` (run the control-net on CPU to save VRAM).

**Residual risk (handled by an implementation gate, not a blocker):** whether a
real ControlNet fits + produces good results on the 6 GB RTX 4050 end-to-end.
Per the image-gen rule ("live-verify a full generation, not just serve
readiness"), the **first implementation task is a live GO/NO-GO**: serve a real
ControlNet + generate with a control image. If a FLUX.1 ControlNet OOMs,
**SDXL/SD1.5 is the guaranteed fallback** (small controlnets, well-supported).

## Background — what exists (and is reused)

- **Serving:** `src/imagemodels/runtime.py` `build_serve_argv(binary, files,
  port, device, …)` builds the `sd-server` argv; it already adds
  `--lora-model-dir loras_dir()` (runtime.py:83) and the fill-and-spill recipe
  (`--offload-to-cpu`, `--max-vram`, `--stream-layers`). The manager
  (`src/imagemodels/manager.py`) owns the served sd-server.
- **LoRA manager** (`src/imagemodels/loras.py`): `loras_dir()`
  (`IMAGE_MODELS_DIR/loras`), `_safe_stem_file`, `list_loras()`,
  `delete_lora(name)`, `download_to_loras(url, filename, …)`, plus
  `src/imagemodels/civitai.py` search. This is the exact template for the
  ControlNet-model registry.
- **Per-request image generation:** the app posts to sd-server's OpenAI-style
  `/images/generations` (`src/ai_interaction.py:1018`) and the gallery's img2img
  posts a per-request base64 `init_images`/`init_image` to `/sdapi/v1/img2img`
  (`routes/gallery/gallery_routes.py`). ControlNet reuses this per-request-image
  pattern, adding `control_image` + `control_strength`.
- **`cv2` (OpenCV) is bundled** — canny preprocessing needs no new dependency.

## Architecture

```
Image models card: pick a ControlNet model (like picking a LoRA)
  └── serve the base model WITH --control-net <cnet> [+ --backend controlnet=cpu]
        (one ControlNet per serve; switching type = re-serve)
        ▼
Control-generate UI: upload a source image → canny-preview (cv2) OR use-as-is
  → strength + prompt → POST control-generate
        ▼
route builds the sd-server request: { prompt, size, control_image (b64), control_strength }
        ▼
sd-server applies the ControlNet → image (saved to Gallery like any generation)
```

### Unit 1 — `src/imagemodels/controlnet.py` (registry + canny)

Mirrors `loras.py`:
- `controlnets_dir() -> str` (`IMAGE_MODELS_DIR/controlnets`),
  `list_controlnets() -> list`, `delete_controlnet(name) -> bool`,
  `download_to_controlnets(url, filename, *, headers=None, http_stream=None) -> dict`
  (atomic `.part`→replace, `_safe_stem_file` rejects `/ \ ..`). Civitai/HF/URL
  download reuse the LoRA/civitai plumbing.
- `preprocess_canny(image_bytes, low=100, high=200) -> bytes` — cv2 decode →
  grayscale → `cv2.Canny` → 3-channel → PNG bytes. Pure, injectable-free, unit-tested.

### Unit 2 — serve with `--control-net`

`build_serve_argv` gains a `control_net: str | None = None` param; when set, it
appends `["--control-net", control_net]` and (to protect 6 GB VRAM)
`["--backend", "controlnet=cpu"]`. The manager tracks the **active ControlNet**
(a setting/state, like the served model) and passes it on serve. No ControlNet
selected → argv unchanged (normal serving).

### Unit 3 — control-generate route

A route (new `POST /api/imagemodels/control-generate`, or the img2img route
extended) that accepts: source image, `control_type` (`canny` | `raw`),
`control_strength`, `prompt`, `size`. It preprocesses (canny via cv2 when
`control_type==canny`; `raw` passes the image through) and posts to the served
sd-server with `control_image` (base64) + `control_strength`, then saves the
result to the Gallery. Requires the model to have been served with a ControlNet
(else a clear "serve a model with a ControlNet first" error).

### Unit 4 — UI

- **ControlNet section** in the Image models card (inside Local Models),
  mirroring the LoRA section: search/download (Civitai/HF/URL), list, delete,
  and **select the active ControlNet** (which gets loaded on the next serve).
- **Control-generate panel** (in the Image area / gallery): upload a source
  image, choose **Canny** (shows a cv2 edge preview) or **Use as-is**, set
  strength (0–1, default ~0.9), enter a prompt → Generate. CSP-safe.

### Unit 5 — packaging

No new bundled binaries/deps (cv2 already ships; sd-server already supports
`--control-net`). ControlNet models are **user-downloaded** (not bundled), like
LoRAs and image models.

## Data flow / error handling

- No ControlNet selected → generation is unchanged.
- ControlNet ↔ base-architecture mismatch (e.g. an SDXL controlnet on a FLUX
  model) → sd-server fails to load; surface a clear error and don't leave a
  broken serve (mirror the existing serve-failure handling).
- OOM on 6 GB → the existing fill-and-spill ladder + `--backend controlnet=cpu`;
  if it still fails, the error tells the user to try a smaller base/controlnet.
- Canny preprocessing failure → clear error; never crash the route.

## Testing

- **Units:** `preprocess_canny` (a known input image → valid PNG, deterministic
  edges); `build_serve_argv` adds `--control-net`/`--backend controlnet=cpu` only
  when a ControlNet is set and is otherwise unchanged; the ControlNet registry
  (`controlnets_dir`/`list`/`delete`/`_safe_stem_file` path-safety) mirroring the
  LoRA tests; the control-generate route's request-body assembly with a mocked
  sd-server (asserts `control_image`/`control_strength` present).
- **Live-verify GATE (Task 1):** serve a real ControlNet (SDXL canny first — the
  safe path; then attempt a FLUX.1 ControlNet) + generate with a control image on
  the 6 GB card; confirm the control actually guides the output and it fits VRAM.
  GO/NO-GO for the FLUX path; SDXL/SD1.5 is the fallback.

## Non-goals (v1)

- Native FLUX.2-klein control (no ControlNet exists yet).
- Bundled depth/pose/scribble preprocessor models (bring-your-own for now).
- Multi-ControlNet, ControlNet for cloud image models, or img2img+control combos.

# Image-Gen Auto-Routing — Design

**Goal:** When the user is chatting with a standard text/agent model and asks to generate
an image, the app auto-serves the configured default image model (a local sd-server GGUF)
if nothing is already serving it, then generates — instead of failing with *"No endpoint
found."* Mirrors the shipped `ensure_vision_served` auto-serve pattern.

**Scope:** One cohesive fix over the shipped image-gen stack: make `generate_image`
reachable when a default image model is configured, and auto-serve that local default on
demand. Not a new image UI, not multi-model concurrency, not a change to sd-server or the
external-API path.

---

## Background — the confirmed gap

- **Image gen already exists**: `generate_image` is a built-in MCP tool
  (`mcp_servers/image_gen_server.py`, dispatched at `tool_execution.py:339`); the real
  logic is `do_generate_image(content, session_id, owner)` (`src/ai_interaction.py:919`).
  Local image models are served by a stateful `ImageModelManager`
  (`src/imagemodels/manager.py` `start(files, device, steps)` / `status()` / `list_models()`),
  which registers an OpenAI-compatible endpoint on serve.
- **Gap 1 — resolve-only, never auto-serve.** `do_generate_image` picks a model
  (tool arg → the `image_model` admin setting → auto-detect `gpt-image-*`/`dall-e-3` →
  any already-registered image endpoint), then calls `_resolve_model(model_spec)` which
  requires an **already-served/registered** endpoint. If the configured default
  `image_model` is a **local sd-server GGUF that isn't currently running**, `_resolve_model`
  raises and the tool returns *"No endpoint found with image model 'X'."* There is an
  `ensure_vision_served` (`src/localmodels/manager.py:363`) that auto-serves the vision model
  on demand — but **no `ensure_image_served` equivalent**.
- **Gap 2 — the tool can be hidden.** `generate_image` is dropped from the toolset unless
  `image_gen_enabled` is on (`src/agent_loop.py:1980`), so the agent may never attempt it.
- **The `files` dict is built inline in the route.** Serving a local image model needs a
  `files` dict (diffusion_model/checkpoint + the right encoders/VAE). That arch-detection +
  encoder-resolution logic (`resolve_flux_files` / `resolve_flux2_files` /
  `resolve_zimage_files` / `resolve_chroma_files` from `src/imagemodels/encoders.py`, plus
  the all-in-one `{"checkpoint": …}` case) currently lives **inline** in
  `routes/imagemodels_routes.py::serve()`. The auto-serve path needs the same logic, so it
  gets extracted into a shared helper.

## Architecture

Three small, isolated pieces plus one gate tweak, all mirroring the shipped vision precedent:

- **`resolve_image_files(model_path) -> dict`** — extracted from
  `routes/imagemodels_routes.py::serve()` into a shared module
  (`src/imagemodels/serve_resolve.py`). Detects the GGUF architecture and calls the right
  encoder resolver (flux / flux2 / zimage / chroma) or returns `{"checkpoint": path}` for an
  all-in-one SD/SDXL checkpoint. Raises `MissingEncoderError` (existing) with its actionable
  message when a required encoder/VAE is absent. The route is refactored to call it (DRY;
  behavior-preserving).
- **`ensure_image_served(owner, *, settings=None, manager=None, resolver=None, lister=None) -> dict`**
  — new, in `src/imagemodels/manager.py`. Mirrors `ensure_vision_served`, returning a small
  dict `{"model": <served id> | None, "error": <str> | None}`:
  1. **If the manager is already serving a local image model** (`manager.status()["running"]`)
     → return that running model's id (use it; **no mid-session swap**).
  2. Else read the default `image_model` (from `load_settings()`). If it names a **local**
     model (matches an entry from `ImageModelManager.list_models()` by filename/basename):
     `files = resolve_image_files(path)` → `manager.start(files, device="gpu")` (the existing
     GPU→CPU fallback ladder, which registers the endpoint) → return the served id.
  3. If the default is external (`gpt-image`/`dall-e`), unset, or unmatched → return
     `{"model": None}` so the existing `do_generate_image` path handles it unchanged.
  Never raises — a serve failure is caught and returned as `{"error": …}`. (v1 tradeoff:
  when a *different* local image model is already running, it's reused rather than swapped
  for the configured default; stop it manually to switch. This matches `ensure_vision_served`
  and avoids disrupting an in-progress session.)
- **`do_generate_image` wiring** (`src/ai_interaction.py`) — before `_resolve_model`, call
  `ensure_image_served(owner)`. If it returns an `error`, surface that. If it returns a
  `model`, use it as the `model_spec` (the local endpoint that's now running). Then
  `_resolve_model` finds the freshly-registered local endpoint and generation proceeds
  exactly as today. An external/unset default (`model: None`) leaves the existing resolution
  (setting → auto-detect → registered endpoint) untouched.
- **Gate tweak** (`src/agent_loop.py:1980`) — `generate_image` availability passes when
  `image_gen_enabled` is on **OR** a non-empty default `image_model` is configured, so a
  user who has set a default image model can ask for images without also flipping the
  separate enable toggle. Minimal relaxation; the external-API and disabled-with-no-default
  behavior is unchanged.

## Data flow

Text-model chat → user asks for an image → agent sees `generate_image` (gate passes) →
`do_generate_image` → the default `image_model` is a local GGUF and nothing is serving it →
`ensure_image_served` → `resolve_image_files` → `manager.start` serves sd-server + registers
the endpoint → `_resolve_model` finds it → `POST /images/generations` → image returned.
External defaults (gpt-image/dall-e) and an already-served local model keep their current
path unchanged.

## Error handling

`ensure_image_served` never raises into `do_generate_image`: missing encoders →
`MissingEncoderError`'s actionable message surfaced as an error; the manager's serve ladder
exhausted (GPU OOM → CPU also fails) → the manager's own tail-of-log error surfaced; a
default that doesn't match any local model → `None` (fall through to the existing external/
auto-detect path); no default and none auto-detected → the existing *"Configure one in
Admin → Image Generation"* error. A local serve can take time — the manager already scales
its readiness timeout by model size.

## Testing

Headless (no GPU, no real sd-server, no real generation):

- **`resolve_image_files`:** inject a fake architecture detector + fake encoder resolvers →
  assert the correct `files` dict per architecture (flux / flux2 / zimage / chroma /
  all-in-one checkpoint); a missing encoder → raises `MissingEncoderError`. The route still
  produces the same `files` it did before (behavior-preserving refactor).
- **`ensure_image_served`:** inject fake settings (`image_model` = a local name), a fake
  `list_models`, a spy manager (records `start(files, …)` + reports `status`), and a fake
  `resolve_image_files` → assert: serves the default and returns its id when nothing is
  running; returns the already-running model's id WITHOUT calling `start` when the manager is
  already serving (no swap); returns `{"model": None}` when the default is external/unset/
  unmatched; returns `{"error": …}` (never raises) when `start` raises.
- **`do_generate_image` wiring:** inject `ensure_image_served` + `_resolve_model` + a fake
  httpx client → a local default triggers auto-serve then proceeds to generation; an
  external default does not call the manager and follows the unchanged path; a serve failure
  surfaces as an error string, not a raise.
- **Gate:** `generate_image` is NOT in the disabled set when a default `image_model` is
  configured even with `image_gen_enabled` off; still disabled when neither is set.
- **Manual (owed by the user):** the real end-to-end path — chat with a text model, ask for
  an image, confirm the local default model auto-serves and returns an image on the 6GB GPU.
  The automated tests prove the serve-decision + routing plumbing, not real image output.

## Non-goals (this sub-project)

- Any change to sd-server / stable-diffusion.cpp itself, or the GPU→CPU serve ladder.
- Concurrency across multiple image models (the manager serves one at a time; auto-serve
  reuses or replaces the single slot, exactly as manual serving does today).
- A new image-generation UI or gallery change.
- Changing the external-API (gpt-image/dall-e) generation path.
- Auto-downloading a missing image model or its encoders (the actionable error already tells
  the user what to place where).

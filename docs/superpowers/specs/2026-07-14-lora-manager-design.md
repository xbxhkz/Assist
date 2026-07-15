# LoRA Manager Design

**Goal:** Download, manage, and apply LoRAs to local image generation. The bundled
sd-server already supports LoRA via `--lora-model-dir` + the `<lora:name:weight>`
prompt tag, so this feature is **management + serve wiring + UI**, with no change to
the image-generation request itself.

**Scope:** LoRA only. ControlNet, identity/reference (PhotoMaker/Kontext), and face
restoration are separate later sub-projects. First of the "image power-user tools"
initiative.

---

## Background — what exists

- sd-server (`build_assets/sd/vulkan/sd-server.exe`) supports `--lora-model-dir <dir>`
  and resolves a `<lora:filename:weight>` tag in the prompt against that dir. It does
  NOT currently receive that flag: `src/imagemodels/runtime.py:build_serve_argv`
  passes no lora/control/photomaker args.
- Image generation flows through `src/ai_interaction.py:do_generate_image` → POST to
  the served sd-server's `/images/generations` with `{prompt, size, quality}`. A
  `<lora:…>` tag lives inside `prompt`, so **no request change is needed** — only the
  serve flag + a way to get LoRA files on disk + a way to insert the tag.
- Image models live under `IMAGE_MODELS_DIR` (`~/.assist/data/image-models`, with an
  `encoders/` subdir today). The image-model serve/list routes
  (`routes/imagemodels_routes.py`) are admin-gated (`require_admin`).
- The app already has a Hugging Face file download mechanism (used by the model
  download catalog) to reuse for the HF source.

## Architecture

Five focused units:

```
UI (loras.js in Image models card)
  ├── search/download panel (Civitai | HF | URL | local import)
  └── installed list (insert <lora:name:w>, delete, show trigger words)
        │
        ▼
routes/loras_routes.py  (admin-gated)
  GET  /api/loras                       → registry.list_loras()
  GET  /api/loras/civitai/search?q=     → civitai.search()
  POST /api/loras/download {source,...} → civitai/hf/url downloader → loras/
  DELETE /api/loras/{name}              → registry.delete_lora()
        │
        ▼
src/imagemodels/loras.py (registry: loras dir, list, delete, resolve name)
src/imagemodels/civitai.py (search + download-url resolution + token)
        │
serve: build_serve_argv appends --lora-model-dir <loras_dir>
apply: prompt "<lora:name:weight>" flows unchanged through do_generate_image
```

### 1. Registry — `src/imagemodels/loras.py`

- `loras_dir()` → `os.path.join(IMAGE_MODELS_DIR, "loras")` (created on demand).
- `list_loras() -> list[dict]`: one entry per `*.safetensors` in the dir:
  `{"name": <stem>, "filename": <file>, "size": <bytes>}`. `name` (filename without
  extension) is exactly what goes in the `<lora:name:weight>` tag.
- `delete_lora(name) -> bool`: remove `<name>.safetensors` if present (path-safe: reject
  names containing `/`, `\`, or `..`).
- `save_stream(name, iterator)` helper for the downloaders to write atomically
  (`.part` → rename) so a half-download never appears in the list.

### 2. Serve wiring — `src/imagemodels/runtime.py:build_serve_argv`

Append `--lora-model-dir <loras_dir>` for every image serve (all devices/branches),
after the existing `--vae-tiling`. Harmless when no `<lora:>` tag is used. `loras_dir()`
is created if absent so the flag always points at a real directory.

### 3. Civitai client — `src/imagemodels/civitai.py`

- `search(query, *, limit=20, token=None) -> list[dict]`: `GET
  https://civitai.com/api/v1/models?types=LORA&query=<q>&limit=<n>`; return per-model
  `{"id", "name", "base_model", "trigger_words", "version_id", "download_url",
  "file_name", "size_kb"}` taken from the first/primary `modelVersions[0]` + its primary
  file. Tolerant of missing fields.
- `download_url_with_token(url, token)`: append the Civitai token (`?token=<t>`) when a
  token is configured — some files require it.
- Uses `httpx` (already a dep). Network/HTTP errors raise a clear exception the route maps
  to a 502/400.
- Optional setting `civitai_api_token` (default `""`) in `src/settings.py`.

### 4. Sources (the downloader in the route)

All three network sources resolve to a URL and share ONE streaming download
(`httpx` stream → `registry.save_stream`, atomic `.part`→rename). `POST
/api/loras/download` body `{source, ...}`:
- `source="civitai"`: `{download_url, file_name, token?}` → stream to `loras/<file_name>`.
- `source="hf"`: `{repo, filename}` → build the HF resolve URL
  (`https://huggingface.co/<repo>/resolve/main/<filename>`, the same pattern
  `src/localmodels/catalog.py` already uses) → stream to `loras/<basename>`.
- `source="url"`: `{url, name}` → stream any direct URL to `loras/<name>.safetensors`.
- `source="local"`: handled UI-side as a multipart file upload → saved to `loras/` via
  `registry.save_stream`. Filenames are basename-sanitized (no path traversal).

### 5. Apply

- **UI:** clicking an installed LoRA inserts `<lora:<name>:0.8>` into the image prompt
  box (default weight `0.8`, adjustable). Trigger words are shown next to each LoRA so
  the user knows what to add.
- **Agent:** the existing `generate_image` tool takes a prompt, so the model can include
  `<lora:name:weight>`. To let it discover installed LoRAs, the `generate_image` tool
  description notes that `list_loras` is available and `app_api` `GET /api/loras`
  returns the installed set. (No new dedicated tool — keep the surface small.)

### 6. UI — `static/js/loras.js`

A LoRA section inside the **Image models** card (Local Models): a tabbed
search/download panel (Civitai search box + results with Download buttons; HF repo/file;
direct URL; local file input) and an installed-LoRA list (name, trigger words, size,
Insert, Delete). CSP-safe (createElement + addEventListener, dynamic text via
`textContent`), mirroring `imagemodels`/`operator.js` style.

## Error handling

- Civitai/HF/URL download failure → route returns a clear error; UI shows it; no partial
  file remains (atomic `.part` rename).
- Gated Civitai file without a token → surfaced as "this file needs a Civitai API token
  (Settings)".
- `delete_lora` path traversal → rejected.
- Serving with an empty/absent loras dir → the dir is created; `--lora-model-dir` is
  still valid; generation without a `<lora:>` tag is unaffected (regression guard).

## Testing

- **registry** (`tests/test_loras_registry.py`): list only `.safetensors`; `name` = stem;
  delete removes the file; delete rejects `../`/slash names; `save_stream` atomic (no
  `.part` in the list mid-write).
- **serve argv** (`tests/test_imagemodels_runtime.py`): `build_serve_argv(...)` contains
  `--lora-model-dir` followed by the loras dir, on both cpu and gpu, and the existing
  argv assertions still pass (regression).
- **civitai** (`tests/test_civitai.py`): `search()` parses a mocked API JSON into the flat
  dict (trigger words, download_url, file_name); `download_url_with_token` appends the
  token; a malformed/missing-fields item degrades without raising.
- **routes** (`tests/test_loras_routes.py`): list/delete/search/download with fakes;
  admin-gated (unauth → rejected); download rejects a traversal `name`.
- **Live-verify (frozen app):** search Civitai for a small SDXL LoRA, download it, serve
  an SDXL model (juggernaut/RealVisXL), generate `a portrait <lora:<name>:0.8>` and
  confirm the LoRA visibly affects the image; generate WITHOUT a tag and confirm no
  regression; delete the LoRA.

## Non-goals

- ControlNet, PhotoMaker/identity, face restoration (separate sub-projects).
- LoRA training.
- Auto-selecting LoRAs by prompt intent (the user/agent chooses).
- Per-LoRA metadata beyond name/size/trigger-words.
- Civitai browse-by-category/infinite-scroll (a simple search box is enough for v1).

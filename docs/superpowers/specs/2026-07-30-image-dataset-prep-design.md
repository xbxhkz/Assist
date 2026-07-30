# Image AI Studio — Dataset Prep (Sub-project 1) — Design

**Goal:** An admin **Image Dataset** panel — sibling to the text-side Dataset Builder — that prepares an
image dataset for LoRA/diffusion fine-tuning: bring in images (upload a folder, or pull from the existing
Gallery), auto-caption each with the shipped vision model, let the admin review/edit captions, attach a
per-dataset trigger word, validate the set, and save it in the standard kohya-ss/diffusers on-disk convention
(`<image>.<ext>` + same-basename `<image>.txt` caption file) — usable with an external LoRA trainer today, and
feeding this app's own image-LoRA trainer once that ships (a later sub-project).

This is **sub-project 1 of a new "Image AI Studio" initiative**, mirroring how the text side grew: dataset
tools first, a training engine/GUI/serve-and-export later. Decomposition (this spec covers only #1):
1. **Dataset prep** (this spec) — caption generation, labeling/validation, export.
2. **Training engine** — headless image-LoRA training (SDXL/FLUX); needs its own feasibility spike (a
   heavier stack than text LoRA training).
3. **Training GUI** — admin panel to configure/run/monitor training.
4. **Serve + export** — likely lighter than the text side's equivalent: `sd-server` already resolves
   `<lora:name:weight>` prompt tags against `src/imagemodels/loras.py`'s `loras_dir()`, so a trained
   `.safetensors` LoRA may only need registering there, not a new conversion/serving step.

## Scope decisions (baked in)

- **Both image sources**: upload a fresh batch of images, *or* select existing images from the Gallery, into
  a working set. (Mirrors the "upload OR library" pattern from grounded generation.)
- **Auto-caption + manual edit**: the vision model drafts a caption per image; the admin reviews/edits before
  saving — the same "generate → stage → review → accept" shape as the text side's synthetic generation.
- **Natural-language captions only** (v1): reuses the already-shipped `analyze_image_with_vl_result` (Qwen-VL)
  — no new vision/tagging dependency. Comma-separated/booru-tag-style captioning (a genuinely different
  tool — a WD14-style multi-label tagger) is explicitly deferred.
- **Trigger word**: a single per-dataset token, auto-prepended to every caption — the standard LoRA/dreambooth
  activation-phrase convention.
- **Output = kohya-ss/diffusers convention**: `<image>` + `<same-basename>.txt` caption pairs in a folder —
  the de facto standard most external LoRA trainers already expect, so this tool is useful standalone even
  before this app has its own trainer.

## Architecture

**Backend (main app, Python 3.14):**

- **`src/image_dataset_tools/store.py`** — `ImageDatasetStore`: `save(name, entries, trigger_word) ->
  {"ok","path"}|{"error"}` (entries = `[{"image_bytes"|"image_path","caption"}]`; writes
  `<DATA_DIR>/training/image_datasets/<name>/NNNN.<ext>` + `NNNN.txt` pairs + a `meta.json` recording the
  trigger word and image count); `list()`, `load(name)`, `delete(name)`. Path-safe name sanitization,
  never-raises, atomic writes — mirrors `src/dataset_tools/store.py` exactly.
- **`src/image_dataset_tools/caption.py`** — `caption_image(path, *, vl_call=None) -> (caption: str|None,
  error: str|None)`. `vl_call` defaults to `analyze_image_with_vl_result` (already used by
  `diagnose_equipment`, webcam detection, and chat vision) with a fixed LoRA-caption-tuned prompt. Injectable
  for tests (no real vision call in the test suite). Never raises.
- **`src/image_dataset_tools/validate.py`** — `validate_image_set(entries) -> report`: per-entry checks —
  unreadable/corrupt image data, missing/empty caption, exact-duplicate images (content hash), resolution
  below a floor (e.g. 256px either dimension) — report shape `{total, valid, invalid, errors:[{index,
  message}], stats:{duplicates, avg_resolution, missing_captions}}`. Never raises (mirrors
  `src/dataset_tools/validate.py`'s contract).
- **Source adapters**: *Upload* — a multipart route accepting a batch of image files, saved to a temp working
  set (mirrors the doc-grounding upload route's temp-file + cleanup-in-`finally` pattern). *Gallery* — given a
  list of existing gallery image ids/paths, copy them into the working set (read-only against the gallery;
  never mutates gallery data).
- **`routes/image_dataset_routes.py`** (new, admin-gated router, mirrors `routes/dataset_routes.py`):
  - `POST /api/image-datasets/upload` (multipart) — intake a batch of images into a working set, returns
    per-image ids + thumbnails-ready paths.
  - `POST /api/image-datasets/from-gallery` (JSON `{ids: [...]}`) — intake selected gallery images.
  - `POST /api/image-datasets/caption` (JSON `{working_set_id}` or the image list) — auto-caption every image
    in the working set via `caption_image`; returns per-image `{id, caption, error?}`.
  - `POST /api/image-datasets/validate` — run `validate_image_set` over the current working set + edited
    captions; returns the report.
  - `POST /api/image-datasets` (save) — `{name, trigger_word, entries}` → `ImageDatasetStore.save`.
  - `GET /api/image-datasets`, `GET/DELETE /api/image-datasets/{name}` — list/load/delete saved datasets.
  - All routes never return 500 — failures ride a report/`error` field, same convention as the text dataset
    routes.

**Frontend:**
- A new **Image Dataset** modal (rail icon + sidebar Tools entry, both hidden by default and revealed only
  when `/api/auth/status` reports `is_admin` — the established both-surfaces pattern) with:
  - A source picker: **Upload** (multipart file input, reusing the Gallery's drag/drop-and-bulk-upload JS
    pattern) or **From Gallery** (a lightweight picker over the existing gallery grid).
  - A thumbnail grid of the working set; each thumbnail has an editable caption textarea beneath it and a
    **remove** button (drops that image from the working set — the only "exclude" mechanism in v1, per the
    Non-goals below).
  - **Caption all** button — calls `/caption`, fills in the per-image caption fields (still editable after).
  - A **Trigger word** input (applies to the whole dataset).
  - **Validate** button — renders the report (bad/duplicate/missing-caption images flagged inline on their
    thumbnails).
  - **Save as** (name) → saved-datasets list (load/delete), mirroring the text Dataset Builder's saved list.
  - Every caption/filename rendered via `innerHTML` is `esc()`'d first (the established XSS discipline).

## Data flow

Upload images *or* pick from Gallery → working set (thumbnails) → **Caption all** (Qwen-VL per image,
editable) → set trigger word → **Validate** (corrupt/missing-caption/duplicate/resolution report) → **Save**
→ `<DATA_DIR>/training/image_datasets/<name>/` (image + `.txt` caption pairs + `meta.json`).

## Error handling

- `caption_image` and `validate_image_set` never raise — a vision-model failure on one image is reported
  per-image (`{"caption": None, "error": "..."}`), it does not abort the batch.
- `ImageDatasetStore` never raises — path-safe names, atomic saves (serialize/copy before touching the
  destination directory, mirroring the text store's temp-then-replace discipline).
- Routes never return 500; the upload route wraps file I/O and cleans up temp files in `finally`.
- Admin-gated throughout (new router `dependencies=[Depends(require_admin)]`); the Gallery-pull adapter reads
  gallery data but never writes to it.

## Testing

- `caption_image` with a fake `vl_call` (success, error-returning, raising) — never-raises confirmed.
- `validate_image_set` — corrupt bytes, missing caption, duplicate images (same hash), tiny resolution, and a
  clean set — report shape and never-raises confirmed with hostile input (non-list entries, non-dict items).
- `ImageDatasetStore` — save/list/load/delete round-trip under `tmp_path`; path-traversal rejection; atomic
  save (a mid-save failure must not corrupt a previously-saved dataset, mirroring the text store's fix).
- `routes/image_dataset_routes` — TestClient with injected `vl_call`/store: admin-gated; upload; caption;
  validate; save; list/load/delete; never-500 on hostile bodies.
- Frontend: `node --check` + text-guard tests (modal ids, both-surfaces admin reveal, `esc()` usage) — no
  automated UI test for the visual panel itself.
- Manual GUI verification owed: upload a real folder of images (or pull from Gallery), caption, set a trigger
  word, validate, save, and confirm the on-disk `.txt` sidecar files are exactly what an external LoRA
  trainer (e.g. kohya-ss) expects.

## Non-goals (this sub-project)

- Image-LoRA **training** itself (sub-project 2+) — this only prepares the dataset.
- WD14/booru-tag-style captioning (a different tool/dependency — deferred).
- Multi-concept datasets (more than one trigger word per dataset).
- Per-image quality/include-exclude flags beyond what Validate reports (no manual "exclude this image"
  toggle in v1 — remove unwanted images from the working set before saving instead).
- Any change to the existing Gallery's own data/behavior — the Gallery adapter is read-only.

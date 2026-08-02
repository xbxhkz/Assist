# Image AI Studio — Training GUI (Sub-project 3) — Design

**Goal:** A frontend-only admin panel over the already-shipped, headless SDXL image-LoRA training engine
(`src/image_training/*`, `routes/image_training_routes.py`) — pick a saved Dataset Prep dataset, set
hyperparameters, start/stop a run, and watch live progress, without needing raw API calls. Sub-project 3 of
the Image AI Studio initiative (sub-project 1, dataset prep, and sub-project 2, the training engine, are both
shipped). Mirrors how the text-LoRA training arc split into engine → GUI → serve.

## Scope decisions (baked in)

- **New, separate modal** — not a tab inside the existing Image Dataset Prep modal. Mirrors the text-LoRA
  precedent exactly (`training-modal` is separate from the Dataset Builder modal): dataset-prep and training
  stay two distinct, focused tools.
- **No backend changes.** The 5 routes shipped with sub-project 2 (`GET /api/image-training/env`,
  `POST /api/image-training/env/setup`, `POST /api/image-training/runs`, `GET /api/image-training/runs/current`,
  `POST /api/image-training/runs/stop`) are already complete and sufficient for this GUI. This sub-project adds
  **zero** Python files.
- **No intermediate-checkpointing work.** A full run at the default 1000 steps takes several hours, and the
  engine currently only saves the LoRA at the very end (crash/close/stop loses the run). This is a known,
  already-tracked limitation from sub-project 2 — this sub-project stays GUI-only and surfaces the limitation
  as a plain, visible note in the panel, rather than expanding scope to add checkpointing.
- **No adapters/management section.** Unlike the text-LoRA GUI (which needed its own Adapters list because no
  other GGUF-adapter browser existed), a trained image LoRA lands directly in `loras_dir()` and is already
  visible in the existing LoRA browser (`static/js/loras.js` / `GET /api/loras`). This GUI does not duplicate
  that — it points the user at the existing browser once a run finishes.
- **Progress via polling, not push.** Image-LoRA training steps take ~20s each (vs. text-LoRA's much faster
  steps), so the same 1.5s-interval polling the text-LoRA GUI already uses is massive overkill in resolution
  terms and needs no new streaming infrastructure (SSE/WebSocket). Reuse the exact `setInterval` pattern.
- **`base_model` is not a form field.** `ImageTrainingConfig.validate()` already rejects anything outside a
  single-entry allowlist (`SUPPORTED_BASE_MODELS = {"stabilityai/stable-diffusion-xl-base-1.0"}`) — showing a
  picker with one valid choice adds a place for typos, not value. The value is sent implicitly (server-side
  default) by simply omitting the field from the request body.

## Architecture

One new HTML modal + one new ES module pair, following `training.js`/`trainingCore.js`'s exact shape:

- **`static/js/imageTrainingCore.js`** — pure helpers, no DOM, no fetch (mirrors `trainingCore.js`):
  - `formToConfig(v) -> object`: maps raw form-field strings to the `POST /api/image-training/runs` request
    body shape (`dataset_name`, `output_name`, `rank`, `lora_alpha`, `learning_rate`, `steps`, `resolution`),
    parsing numeric fields with the same proven defaults as fallbacks (`rank=4`, `lora_alpha=4`,
    `learning_rate=1e-4`, `steps=1000`, `resolution=1024`) the backend dataclass already defaults to.
  - `renderStatusLine(s) -> string`: maps a `/api/image-training/runs/current` status dict
    (`status`/`last_step`/`loss`/`vram_gb`/`peak_vram_gb`/`error`/`lora_path`) to the same
    `"status: X · step N · loss L · vram G GB"` join format `training.js`'s `renderStatus` already produces,
    plus a `lora_path` line when `status === "done"`.
- **`static/js/imageTraining.js`** — DOM controller, ES module (mirrors `training.js`):
  - Module-scope `$(id)`, `esc(s)`, `api(path, opts)`, `isAdmin()` helpers (identical implementations to
    `training.js`'s — small enough that duplicating rather than sharing matches the existing project
    convention of each panel module carrying its own copies of these four helpers).
  - `openImageTraining()` / `closeImageTraining()`: show/hide `#image-training-modal`, calling
    `refreshEnv()`, `refreshDatasetList()`, and `resumeIfRunning()` on open (mirrors `training.js`'s
    `openTraining`'s exact resume-on-reopen fix from the text-LoRA GUI's own whole-branch review).
  - `refreshEnv()`: `GET /api/image-training/env`, updates `#imgtrain-env-status`, enables/disables the run
    card and Start button based on `status === "ready"`.
  - `setupEnv()`: `POST /api/image-training/env/setup`, mirrors `training.js`'s `setupEnv` exactly
    (disable button during, show `Setting up…`/`Ready.`/`Failed: …`).
  - `refreshDatasetList()`: `GET /api/image-datasets`, populates a `<datalist>` of saved dataset names (the
    existing Dataset Prep list endpoint — already shipped, returns `{name, path, images, size}` per dataset).
  - `startRun()`: `formToConfig()` the form, `POST /api/image-training/runs`, show `Started.` or the surfaced
    error, then `startPolling()`.
  - `stopRun()`: `POST /api/image-training/runs/stop`.
  - `startPolling()`/`stopPolling()`/`pollStatus()`/`resumeIfRunning()`: identical shape to `training.js`'s —
    1.5s `setInterval`, stop polling on `done`/`error`/`stopped`, resume polling on reopen if `status ===
    "running"`.
  - `renderStatus(s)`: calls `imageTrainingCore.renderStatusLine(s)` and writes it into
    `#imgtrain-progress`; when `status === "done"`, also shows a fixed line pointing at the LoRA browser
    ("Done — find it in the LoRA manager (Image models card)").
  - `init()`: admin-gate reveal of `#rail-imagetraining` + `#tool-imagetraining-btn` (mirrors every other
    admin-only panel's `isAdmin()` gate), wire up all buttons, `Modals.register('image-training-modal', {...})`.

## HTML additions (`static/index.html`)

- `#image-training-modal` — new modal, placed after `#imagedataset-modal` (before `#dataset-modal`), 3 cards:
  1. **Environment card** (`#imgtrain-env-card`): status text (`#imgtrain-env-status`) + "Set up training
     environment" button (`#imgtrain-env-setup`, same one-time-download framing as the text-LoRA card, but
     noting this reuses the SAME venv the text trainer already provisioned — so for a user who already ran
     text-LoRA training once, this step is just adding `diffusers`, not a from-scratch multi-GB download).
  2. **Run card** (`#imgtrain-run-card`): dataset picker (`#imgtrain-dataset`, `<input list=...>` +
     `<datalist id="imgtrain-dataset-suggestions">`), output name (`#imgtrain-output-name`), a
     `<details><summary>Advanced</summary>` block with `#imgtrain-rank`/`#imgtrain-alpha`/`#imgtrain-lr`/
     `#imgtrain-steps`/`#imgtrain-resolution` number inputs pre-filled with the proven defaults, Start/Stop
     buttons (`#imgtrain-start`/`#imgtrain-stop`), and a fixed small-print note: "Results save only when the
     run finishes — stopping or closing the app mid-run does not keep partial progress."
  3. **Progress** (`#imgtrain-progress`): plain text status line, updated by polling.
- `#rail-imagetraining` — new icon-rail button (after `#rail-imagedataset`), `style="display:none"` until
  `isAdmin()` reveals it (matches every other admin-only rail button).
- `#tool-imagetraining-btn` — new sidebar Tools entry (after `#tool-imagedataset-btn`), same reveal pattern.
- Script tag for `imageTraining.js` (after `imageDataset.js`'s tag).
- Help section entry: "Image LoRA Training (Image AI Studio)" (after the existing "Image Dataset" entry),
  with a distinguishing phrase e.g. "train an SDXL LoRA from a prepared image dataset" for the existing
  Help-content uniqueness test convention this project already follows.

## Data flow

Admin opens the panel → `refreshEnv()` + `refreshDatasetList()` + `resumeIfRunning()` fire → admin picks a
saved dataset (populated from the already-shipped Dataset Prep tool) and an output name, optionally tweaks
Advanced hyperparameters → Start → `POST /api/image-training/runs` → polling renders live step/loss/VRAM →
on `done`, the panel points at the existing LoRA browser card where the new `.safetensors` is already listed
(no polling or refresh needed there — it's a plain directory listing, picked up next time that card loads).

## Error handling

Every `api()` call already throws on a non-OK response (matching `training.js`'s existing `api()` helper);
every caller catches and writes the message into the relevant status element — never an unhandled rejection,
never a thrown error reaching the DOM as a blank panel. `resumeIfRunning()`/`refreshEnv()`/
`refreshDatasetList()` swallow fetch failures silently (matching `training.js`'s identical background-refresh
calls), since a transient failure there shouldn't block opening the panel.

## Testing

- **`tests/test_image_training_core_js.py`** — Node-subprocess tests over `imageTrainingCore.js`, mirroring
  `tests/test_training_core_js.py`'s exact pattern (`subprocess.run(["node", "--input-type=module", "-e",
  script], encoding="utf-8")`): `formToConfig` produces the right keys/types/defaults from raw form-string
  input (including the same "numeric field is blank/garbage falls back to the proven default" cases
  `trainingCore.js`'s tests already cover for its own fields), `renderStatusLine` produces the right string
  for `running`/`done`/`error` status shapes.
- **`tests/test_image_training_ui.py`** — mirrors `tests/test_image_dataset_ui.py`'s exact 4-test shape (the
  sibling test file for the Dataset Prep panel this session already shipped): (1) an HTML-element-presence
  test asserting every new id (`image-training-modal`, `rail-imagetraining`, `tool-imagetraining-btn`, the
  `imageTraining.js` script tag, and each form-field id) is present in `static/index.html`; (2) a
  JS-wiring-presence test asserting `imageTraining.js`'s source contains the rail/sidebar ids, `isAdmin`,
  `Modals.register`, and each of the 5 `/api/image-training/...` endpoint strings it calls; (3) a syntax-only
  gate for `imageTraining.js` (write its source to a temp `.mjs` file, run `node --check` against it,
  assert exit 0 — the same technique used for every ES-module panel this session, since these modules are
  never imported by pytest); (4) a Help-manual-section test asserting the new Help entry's distinguishing
  phrase is present in `static/index.html`. This is the DOM/poll wiring's only automated coverage — nothing
  drives a real browser in this suite, matching every prior frontend sub-project this session.
- **Manual GUI verification is owed by the user** — same as every other frontend sub-project shipped this
  session. This spec does not claim the panel works end-to-end against a real GPU training run; that
  confirmation is explicitly deferred to you, matching how sub-project 2's real-training-quality check was
  also left to you.

## Non-goals (this sub-project)

- Intermediate checkpointing (tracked separately, not part of this GUI work).
- Any adapters/management UI (the existing LoRA browser already covers this).
- Serving the trained LoRA from the app UI (already possible today via the existing image-generation model
  picker's `<lora:name:weight>` tag support — no new work needed here).
- A dataset picker that lets you browse INTO a dataset's contents from this panel (the existing Dataset Prep
  modal already owns that; this panel only needs to pick a dataset BY NAME).

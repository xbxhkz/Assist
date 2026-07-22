# Local LoRA/QLoRA Training — Design

**Goal:** Fine-tune a small base model with **QLoRA** on the user's own GPU, entirely
locally, and produce a LoRA adapter — configured, launched, and monitored from an in-app
**Training** GUI. This is sub-project 1 of the "AI Studio" roadmap item. Later sub-projects
add merge-to-GGUF + serving the tuned model, and full-fp16 LoRA.

**Feasibility: PROVEN GO.** A spike ran a real 4-bit QLoRA fine-tune on this machine
(Windows, RTX 4050 6 GB) in a separate Python 3.11 CUDA venv: `torch 2.5.1+cu121`
(`cuda=True`), `bitsandbytes 0.49.2`, `peft 0.19.1`, `transformers 5.14.1`; 8 steps,
loss computed, a real `adapter_model.safetensors` saved, **peak VRAM 1.16 GB of 6.44** for
a 0.5B model. So 1–3B trains comfortably and ~7B QLoRA is realistic on 6 GB. Unlike the
EXL2 gate (NO-GO on Python-3.14 wheels), this is buildable — the training just runs in a
side Python, not the app's.

---

## The core constraint

The main app is **Python 3.14**, which has **no CUDA torch / bitsandbytes / peft wheels**.
So **all training code runs in a separate Python 3.11 CUDA venv**, and the main app only
*orchestrates* it as a subprocess — exactly how it drives `llama-server` / `sd-server`,
except the "binary" is `venv/python train.py`. The main app never imports the training
stack, so a broken training env can never destabilize it.

## Environment delivery: on-demand

The base installer stays lean (~0.8 GB). The training env (~3–4 GB: torch cu121 +
bitsandbytes + peft + transformers + accelerate + datasets + trl) is set up **on first
use**, using a vendored **`uv`** binary (a single ~30 MB self-contained exe, bundled in the
base installer). `uv` fetches Python 3.11 (python-build-standalone), creates the venv under
the data dir, and `uv pip install`s the stack (the exact sequence the spike proved). Needs
internet once; then trains offline.

## Architecture / components

**Backend (main app, Python 3.14):**

- **`scripts/fetch_uv.py`** + `Assist.spec` — vendor the `uv` binary into
  `build_assets/uv/`, bundled like `llama`/`sd`. A `resolve_uv_binary()` finds it at runtime
  (frozen `_MEIPASS/uv/…` or dev `build_assets/uv/…`), mirroring `resolve_llama_binary`.
- **`src/training/env.py`** — `TrainingEnv`: `status()` (not_installed / installing / ready /
  error), `ensure_ready(progress_cb)` runs the `uv` steps idempotently (skip if the venv
  python + a sentinel marker exist), captures install output, resolves the venv python path
  (`<DATA_DIR>/training/venv/Scripts/python.exe`). The `uv` runner is **injectable** for tests
  (command construction + ready-detection are what's tested; the real install is spike-proven).
- **`src/training/config.py`** — `TrainingConfig` schema + `validate()` (base model id
  non-empty; dataset path exists; LoRA r/alpha/dropout/steps/epochs/batch/lr sane) +
  `estimate_vram_gb(model_params_b)` and a `fits(free_vram_gb)` check (warn/block a too-big
  model). Pure, unit-tested.
- **`src/training/dataset.py`** — `load_jsonl(path)` + `validate_rows`: each line is either
  `{"text": …}` (raw SFT) or `{"instruction","response"}` / `{"prompt","completion"}`
  (formatted into a simple template); clear errors on malformed rows; returns a normalized
  list the sidecar consumes. Pure, unit-tested.
- **`src/training/manager.py`** — `TrainingManager`: given a `TrainingConfig`, `ensure_ready`
  the env, VRAM-gate the model, launch the sidecar, parse its streamed progress, track
  status/last-run, `stop()`. Mirrors `LocalModelManager` (injectable spawn/now/probe;
  never-raises into the route). One run at a time (a training run saturates the GPU).
- **`routes/training_routes.py`** — admin-gated `/api/training/*`: `GET env` (status),
  `POST env/setup` (kick off on-demand install, stream progress), `POST runs` (start a run
  from a config), `GET runs/current` (status + progress), `POST runs/stop`, `GET adapters`
  (list produced adapters). TestClient + admin-gate tested.

**Sidecar (runs INSIDE the Python 3.11 venv, never imported by the main app):**

- **`training_sidecar/train.py`** — reads a JSON config (base model, dataset file, output
  dir, LoRA params, hyperparams) → loads the base model in 4-bit (nf4, double-quant) →
  `prepare_model_for_kbit_training` → attaches a LoRA (peft) → trains (transformers `Trainer`
  or `trl` SFT) → emits progress as **JSON lines to stdout** (`{"event":"step","step":N,
  "loss":…,"vram_gb":…}`, plus `start`/`done`/`error`) → saves the adapter
  (`adapter_model.safetensors` + `adapter_config.json` + the run config). This is the spike
  script, productionized. It's covered by the spike + a real run, not by the Py3.14 unit
  suite (different interpreter). **Bundling:** `training_sidecar/` ships as a data dir in
  `Assist.spec` (like `scripts/`, `mcp_servers/`); the manager resolves `train.py` at
  frozen `_MEIPASS/training_sidecar/train.py` (dev: repo path) and runs it with the venv
  python — so a source edit to `train.py` requires a rebuild to reach the bundle.

**Frontend (the GUI):**

- **`static/` Training modal** (opened from the sidebar icon-rail, admin-only, like Workflows)
  + **`static/js/training.js`** (plain-script ESM style, like `imageModels.js`):
  - **Environment card** — status badge + a **"Set up training environment"** button that
    calls `POST env/setup` and shows install progress; the run form is disabled until *ready*.
  - **New-run form** — base model (HF repo id + a few suggested small models), a **dataset**
    picker/upload (.jsonl), a collapsible **Advanced** (LoRA r/alpha/dropout, steps-or-epochs,
    batch, learning rate) with sensible defaults, and a live **VRAM-fit hint** from the
    config estimate.
  - **Run monitor** — Start/Stop + a live progress area (step / loss / VRAM) polling
    `GET runs/current`.
  - **Adapters list** — produced adapters with their run config + folder.
  - A pure JS-core module (`static/js/trainingCore.js`) does the form→config mapping + the
    VRAM-fit hint (node-subprocess unit test); the DOM module gets a `node --check` syntax
    gate; the visual panel itself is **manual GUI verification owed by the user**.

## Data flow

User opens Training → (first time) clicks *Set up environment* → `TrainingEnv.ensure_ready`
runs `uv` (fetch Py3.11 → venv → pip install) → *ready*. User fills the run form → `POST
runs` with the config → `TrainingManager` validates + VRAM-gates → launches
`venv/python training_sidecar/train.py --config <path>` → the sidecar streams JSON-line
progress → the manager relays it to `GET runs/current` → the GUI shows step/loss/VRAM → on
finish, a LoRA adapter lands in `<DATA_DIR>/training/adapters/<run-id>/` and appears in the
adapters list.

## Scope decisions (baked in, adjustable)

- **QLoRA (4-bit) only** in v1 (the realistic 6 GB mode; full-fp16 LoRA deferred).
- **Base model = a HuggingFace repo id** (downloaded by the sidecar via `transformers`).
- **Dataset = a local JSONL file** in the two supported shapes above.
- **Model size gated by free VRAM** (comfortable ≤3B; a warning as it approaches ~7B).
- **One run at a time** (a run saturates the single GPU).

## Error handling

Env setup failures (no internet, `uv`/pip error, low disk) → a clear `error` status +
message, never a crash. A too-big model → a pre-flight VRAM warning (the user can still
proceed). A sidecar that dies (OOM / CUDA error) → captured from its stdout/log tail and
surfaced as the run's error (like the serve managers). All manager/route methods degrade to
`{"error": …}` and never raise into the app. The main app never imports the training stack.

## Testing

Headless, in the main app's Python-3.14 suite:
- `TrainingConfig.validate` + `estimate_vram_gb`/`fits` — pure unit tests (valid/invalid
  configs; a 0.5B fits, a 13B warns on 6 GB).
- `TrainingEnv` — `uv` command construction + ready-detection (marker/venv-python present)
  with an **injected runner**; never-raises on a failing runner.
- `dataset.load_jsonl` — both row shapes normalized; malformed rows → clear errors.
- `TrainingManager` — orchestration + progress-line parsing + VRAM gating with an **injected
  spawn**; a dying sidecar → error surfaced, no raise.
- `routes/training_routes` — TestClient: admin-gated; start/status/stop/adapters shapes.
- `trainingCore.js` — form→config mapping + VRAM hint (node subprocess, utf-8); DOM module
  `node --check` syntax gate.
- **The actual training run is proven by the spike** and owed as one real end-to-end run on
  the user's GPU (train a tiny adapter from the GUI). `train.py` isn't importable by the
  Py3.14 suite — it's covered by the spike + that manual run.
- **Frozen boot-check (owed after a rebuild):** the vendored `uv` binary resolves in the
  bundle and the training routes register.

## Non-goals (this sub-project)

- Merge-LoRA → GGUF conversion and serving the fine-tuned model locally (sub-project 3).
- Full-fp16 (non-quantized) LoRA training (a later option).
- Multi-GPU training; evaluation/validation splits, checkpoint resume, early stopping.
- Dataset creation/labeling tools, synthetic data (separate "AI Studio" pieces).
- Cloud training (the user chose local; HF-Jobs remains a future alternative).

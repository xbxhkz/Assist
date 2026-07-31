# Image AI Studio — LoRA Training Engine (Sub-project 2) — Design

**Goal:** A headless engine that fine-tunes an image-LoRA (SDXL or FLUX) from a dataset produced by the
shipped Dataset Prep tool (`<DATA_DIR>/training/image_datasets/<name>/`, image + `.txt` caption pairs), on
the user's 6GB RTX 4050. Sub-project 2 of the Image AI Studio initiative (sub-project 1, dataset prep, is
shipped). Mirrors how text-LoRA training was built: engine first (no GUI), gated by a feasibility spike,
serve/GUI/export as later sub-projects.

## The core idea

`sd-server` (the bundled `stable-diffusion.cpp`) is an **inference-only** C++ engine — it has no training
capability at all. Image-LoRA training needs a real Python training stack (torch + diffusers + accelerate),
the same category of dependency the text side already solved with an isolated Py3.11 CUDA sidecar venv
(`<DATA_DIR>/training/venv`, provisioned on-demand via vendored `uv`; the main Py3.14 app never imports any
of it). That sidecar's existing stack (`transformers`, `peft`, `bitsandbytes`, `accelerate`, per
`src/training/env.py`'s `STACK`) is **mostly reusable** for image training — diffusers' own LoRA training
scripts use `peft` internally — so this sub-project extends the existing sidecar rather than standing up a
second multi-gigabyte CUDA venv.

The open, genuinely uncertain question is VRAM: SDXL LoRA training is commonly quoted at 8–12GB, FLUX at
16GB+; the user's card is 6GB. Community reports place aggressive SDXL configs (rank-4/8 LoRA, batch size 1,
gradient checkpointing, 8-bit optimizer, capped resolution) around 6–8GB — plausible, not guaranteed. This
is a bigger feasibility risk than text training's (which proved out cleanly on a 0.5B model), so **this
sub-project is scoped as the feasibility spike plus a minimal engine wrapping only what the spike proves
works** — exactly mirroring text training's Task-1-is-a-spike structure.

## Scope decisions (baked in)

- **Both model families, feasibility-gated**: design for SDXL and FLUX, but the spike decides which (if
  either) are actually supported on this hardware. A family that doesn't fit is reported as unsupported, not
  shipped half-working.
- **Per-family toolchain routing, not a user-facing backend toggle**: the user picks a model family (SDXL or
  FLUX); internally, the engine uses whichever toolchain the spike proved works for *that* family. The spike
  tries **diffusers' official LoRA scripts first** (official provenance, reuses the sidecar's existing `peft`
  dependency, no vendoring) for each family; if a family doesn't fit under diffusers, the spike tries
  **kohya-ss's sd-scripts** (more memory-optimized, third-party, would need vendoring) for that family only.
  The shipped engine only wraps the toolchain(s) actually proven necessary — if diffusers covers both
  families, kohya-ss is never vendored at all.
- **Engine only, no GUI** (a later sub-project), matching the established pattern from text-LoRA training.
- **Extend the existing sidecar venv** (add `diffusers`, and `kohya-ss`'s scripts only if the spike proves
  they're needed) rather than a second training venv.

## Architecture

**Feasibility spike (Task 1, gates everything else):**
- Run live on this machine (not a mocked unit test — a real timed training-step probe, exactly like text
  training's spike): for SDXL, attempt a handful of real training steps via diffusers' official SDXL LoRA
  script with the most aggressive memory settings (rank 4-8, batch 1, gradient checkpointing, 8-bit AdamW,
  512-768px resolution, mixed precision). If it OOMs or fails to run, retry with kohya-ss's sd-scripts (same
  aggressive settings, its more mature low-VRAM path). Repeat the same diffusers→kohya-ss sequence for FLUX.
- Output: a **per-family verdict** — `{"sdxl": "diffusers"|"kohya"|"unsupported", "flux": "diffusers"|"kohya"|"unsupported"}`
  plus the actual peak VRAM/step time observed, so later hyperparameter defaults are grounded in measurement,
  not guesswork.

**Engine (built only for the family/toolchain combos the spike proves work):**
- `src/image_training/env.py` — extends the existing `TrainingEnv`'s `STACK` (mirrors
  `src/training/env.py`) with `diffusers` (and the vendored kohya-ss scripts, only if proven needed for some
  family). Same on-demand `uv`-provisioned venv, same never-raises `ensure_ready` contract.
- `src/image_training/manager.py` — orchestrates a training run: given `{model_family, dataset_path,
  base_model, rank, learning_rate, steps}`, resolves the venv, spawns the proven per-family training script
  (`image_training_sidecar/train_sdxl_lora.py` / `train_flux_lora.py`, each a thin wrapper choosing
  diffusers' or kohya-ss's actual training entry point per the spike's verdict) as a subprocess, and reports
  progress. Mirrors `src/training/manager.py`'s injectable-spawn, never-raises pattern.
- **Progress channel — the exact lesson from text training's own bug, applied from day one**: the spawned
  script's stdout must NOT let the trainer's default tqdm progress bar corrupt the JSON progress channel with
  `\r` redraws (text training's whole-branch review caught this as a cross-seam bug that per-task tests
  missed). Every training script this sub-project writes disables the library's default progress bar
  (diffusers' `disable_tqdm`-equivalent or kohya-ss's own flag) and emits its own line-delimited JSON progress
  instead, matching text training's proven channel design.
- `image_training_sidecar/` — the spawned scripts (never imported by the main app), mirroring
  `training_sidecar/`.

**Output:** a trained LoRA `.safetensors` file. No conversion step is anticipated — `.safetensors` is already
what `src/imagemodels/loras.py`'s `loras_dir()` registry and `sd-server`'s `<lora:name:weight>` prompt-tag
resolution expect, unlike text training's GGUF conversion requirement. Confirming this is a no-op is part of
the spike's job, not assumed here.

## Data flow

Admin (a later GUI, or a direct API call for now) selects a model family + a saved image dataset (from
Dataset Prep) + basic hyperparams → `image_training.manager` ensures the sidecar venv is ready (with
`diffusers`/kohya-ss as the spike determined) → spawns the family-appropriate training script against the
dataset folder → streams line-delimited JSON progress (step, loss, VRAM) → on completion, a `.safetensors`
LoRA is written to a location `imagemodels/loras.py`'s registry can pick up.

## Error handling

- The manager never raises — `ensure_ready`/`start` return status dicts, mirroring `TrainingEnv`/`manager.py`.
- A CUDA OOM during training is caught and surfaces as a clear, actionable message ("ran out of VRAM — try a
  lower LoRA rank, smaller batch size, or lower resolution"), not a raw CUDA stack trace.
- A model family with no working toolchain per the spike's verdict is refused up front with a clear
  "not supported on this hardware" message — never attempted and left to fail mid-run.
- The sidecar subprocess protocol (JSON-over-stdout, UTF-8 on Windows) follows the exact conventions text
  training already proved out (locale-safe encoding, stderr merged in, tqdm disabled).

## Testing

- Feasibility spike: run live against real hardware; not a mocked test. Its verdict is recorded (as this
  sub-project's own "memory" of what's proven) and gates which family/toolchain combinations the rest of the
  plan builds.
- `image_training.env`/`manager`: headless orchestration tests with injectable `spawn`/`run`, mirroring
  `tests/test_training_env.py`/`tests/test_training_manager.py`'s patterns exactly — never-raises on a
  missing venv, a spawn failure, a malformed progress line, etc.
- No GUI test needed (none shipped this sub-project).

## Non-goals (this sub-project)

- Training GUI (later sub-project, mirrors text training's arc).
- Serving the trained LoRA (likely much lighter than text training's serve sub-project, since `sd-server`
  already resolves `<lora:name:weight>` tags against the existing registry — but confirming/wiring that is
  deferred to its own sub-project, not assumed complete here).
- Any export/conversion format beyond whatever `.safetensors` the proven toolchain natively produces.
- Multi-LoRA / multi-concept training, LoRA merging, or any dreambooth-style full-model fine-tuning (LoRA
  only, matching the text side's scope).
- A model family the spike proves unsupported on this hardware is out of scope for this sub-project (may be
  revisited later if hardware changes or a lower-VRAM toolchain emerges).

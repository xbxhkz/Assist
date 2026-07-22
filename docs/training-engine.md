# Local LoRA/QLoRA Training Engine

Fine-tune a small base model with QLoRA on your own GPU, entirely locally.
This is the **headless engine** — an admin-gated HTTP API. The Training GUI is
a follow-up sub-project that will consume these same routes.

## How it works

The main app (Python 3.14) cannot run CUDA torch, so training runs in a
separate Python 3.11 CUDA venv, set up on first use via a vendored `uv` binary.
The app orchestrates a sidecar (`training_sidecar/train.py`) as a subprocess
and never imports the training stack itself. One run at a time.

## API (all admin-only, `/api/training`)

- `GET /env` — `{"status": "not_installed" | "ready" | "error"}`.
- `POST /env/setup` — build the training venv (one-time, downloads ~3–4 GB:
  torch cu121 + transformers + peft + bitsandbytes + accelerate + datasets + trl).
- `POST /runs` — start a run. Body:
  `{"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "dataset_path": "C:/data.jsonl",
    "steps": 100}` (or `"epochs"`; optional `lora_r`, `lora_alpha`,
  `lora_dropout`, `batch_size`, `learning_rate`, `max_seq_length`).
- `GET /runs/current` — `{"status", "last_step", "loss", "vram_gb",
  "peak_vram_gb", "error", "output_dir"}`.
- `POST /runs/stop` — kill the current run.
- `GET /adapters` — list produced adapters (`run_id`, `complete`, `base_model`, `path`).

## Dataset format (JSONL, one object per line)

Any of: `{"text": "..."}`, `{"instruction": "...", "response": "..."}`
(optional `"input"`), or `{"prompt": "...", "completion": "..."}`.

## Output

A LoRA adapter under `<DATA_DIR>/training/adapters/<run-id>/`
(`adapter_model.safetensors`, `adapter_config.json`, `run_config.json`).

## Owed (manual, needs the GPU)

- Before packaging: run `python scripts/fetch_uv.py` to vendor `uv` into
  `build_assets/uv/`, then rebuild the frozen exe.
- One real end-to-end run: `POST /env/setup`, then `POST /runs` with a small
  model (e.g. Qwen2.5-0.5B), watch `GET /runs/current`, confirm an adapter lands.
- Frozen boot-check: `resolve_uv_binary()` + `resolve_sidecar_script()` resolve
  in the bundle and the training routes register.

## Not yet (later sub-projects)

The admin Training GUI; merge-LoRA → GGUF + serving the tuned model;
full-fp16 (non-quantized) LoRA; multi-GPU; eval/checkpoint-resume.

# Export a Fine-Tuned Adapter as a Standalone GGUF — Design

**Goal:** From a locally-trained LoRA adapter, produce **one self-contained, quantized GGUF** the user
can use *outside* Assist (LM Studio, Ollama, sharing) — merge the adapter into the base weights, convert
the full model to GGUF, and quantize. Admin-gated, driven from the existing Adapters list. This is
sub-project 4 of the training initiative (engine=1, GUI=2, serve-adapter=3).

---

## Feasibility: PROVEN GO (spike ran the full chain on this machine)

A spike merged the real trained adapter into Qwen2.5-0.5B, converted the merged model to GGUF, quantized
it, and served the standalone GGUF — which generated coherent text with **no `--lora`** (model id = the
GGUF path itself). Every step + tool is confirmed:

- **Merge:** `PeftModel.from_pretrained(base, adapter).merge_and_unload()` → `save_pretrained(<merged_hf>)`
  (in the Python-3.11 CUDA venv; ~21 s for 0.5B).
- **Convert full model:** llama.cpp's `convert_hf_to_gguf.py` (a single thin file at commit `c0bc859`
  that reuses the already-vendored `conversion/` package + `gguf/`) → F16 GGUF. **Needs `sentencepiece`**
  — full-model vocab conversion imports it, whereas the LoRA converter never did (it only touches adapter
  tensors). This was the spike's key find.
- **Quantize:** `llama-quantize.exe <f16.gguf> <out.gguf> Q4_K_M` — from the same llama.cpp release
  (`b9867`) as the bundled `llama-server`; it's a thin launcher that uses the ggml/quantize DLLs already
  present in `build_assets/llama/vulkan/`. Fast (~2.3 s for 0.5B).
- **Serve:** the resulting GGUF serves via the existing `llama-server` path with no adapter flag.

**Sizes (0.5B):** merged HF 988 MB → F16 GGUF 994 MB → **Q4_K_M 398 MB**. The F16 intermediate ≈ 2× the
model, so temporaries are deleted after quantizing. Scales ~linearly with model size (a 7B F16 ≈ 14 GB) —
disk-heavy but manageable for the small models this 6 GB-GPU workflow targets.

## The core constraint

The Python-3.14 app cannot run the merge/convert stack (torch/peft/transformers/sentencepiece/gguf), so
**merge + convert run in the same Python-3.11 CUDA sidecar venv the training engine already builds**
(reused; add only `sentencepiece`). **Quantize needs no Python** — it's the bundled `llama-quantize.exe`,
run by the app exactly as it runs `llama-server`. The app orchestrates both as subprocesses and never
imports the stack.

## Architecture / components

**Sidecar (runs INSIDE the venv, never imported by the app):**
- **`training_sidecar/merge.py`** — reads an adapter dir + base model id → `merge_and_unload` → saves the
  merged HF model to a temp dir → spawns the vendored `convert_hf_to_gguf.py` → an **F16 GGUF** → deletes
  the temp HF model → emits one JSON line (`{"event":"done","f16_gguf":…}` / `{"event":"error",…}`), same
  stdout protocol as `train.py`/`convert.py`.

**Backend (main app, Python 3.14):**
- **Vendoring** — `training_sidecar/convert_hf_to_gguf.py` (from `c0bc859`, reuses vendored
  `conversion/` + `gguf/`); `llama-quantize.exe` into `build_assets/llama/{cpu,vulkan}/` via an update to
  `scripts/fetch_llama_server.py` (extract it alongside `llama-server.exe`); `"sentencepiece"` added to
  `TrainingEnv.STACK`. A `resolve_quantize_binary(device)` finds `llama-quantize.exe` next to the resolved
  llama binary (mirrors `resolve_llama_binary`).
- **`src/training/export_manager.py`** — `AdapterExporter`: given `run_id` + `quant`, run the merge
  sidecar → F16 GGUF, then (for a non-F16 quant) run `llama-quantize.exe` → the final GGUF in the exports
  dir, delete the F16 intermediate, and report status. Mirrors `AdapterConverter` (injectable spawn,
  blocking, **never-raises**; one export at a time). For `quant == "F16"` the F16 GGUF *is* the output
  (no quantize step).
- **`routes/training_routes.py`** (extend, admin-gated) — `POST /api/training/adapters/{run_id}/export`
  (body `{quant}`, validated against `{"Q4_K_M","Q5_K_M","Q8_0","F16"}` → 400 otherwise) runs the export
  off the event loop and returns the output path; `GET /api/training/adapters/{run_id}` also reports any
  already-exported files.

**Frontend (the existing Training/Adapters panel — additive):**
- Per-adapter **Export GGUF** control with a **quant dropdown** (`Q4_K_M` default, `Q5_K_M`, `Q8_0`,
  `F16`). On success it shows the output path and an **Open folder** button (reveal in the file manager,
  via the existing desktop file-open path). A pure `exportCore.js` helper (quant list + button state)
  gets a node-subprocess unit test; the DOM wiring gets a `node --check` gate; the panel is manual-GUI
  owed.

## Data flow

Adapter (`<DATA_DIR>/training/adapters/<run-id>/`) → **Export(quant)** → sidecar `merge.py`
(merge → merged HF temp → `convert_hf_to_gguf` → F16 GGUF temp, deletes the HF temp) → app runs
`llama-quantize.exe` (F16 → `<DATA_DIR>/training/exports/<base>-<run-id>-<quant>.gguf`) → deletes the F16
temp → surfaces the export path (+ Open folder). `F16` skips quantize and moves the F16 GGUF to exports.

## Error handling

- Missing/invalid `quant` → 400. A merge/convert failure (arch unsupported, OOM on a too-big model) →
  captured from the sidecar's `error` JSON; a quantize failure → captured from `llama-quantize`'s output
  tail; both surface as an error status, never a crash. Temporaries (merged HF, F16) are cleaned on both
  success and failure so a failed export doesn't strand multi-GB files. All managers/routes degrade to
  `{"error": …}` and never raise; the app never imports the merge stack.
- Low disk / large model → the error surfaces from the failing tool; a pre-flight free-disk estimate is a
  possible enhancement, not v1.

## Testing

Headless, in the Python-3.14 suite:
- Quant validation (accept the four; reject others) — pure.
- `AdapterExporter` orchestration with an **injected spawn**: merge→quantize sequence, the F16-skips-
  quantize path, temp cleanup, and never-raises on a failing/raising step.
- `resolve_quantize_binary` — frozen/dev resolution (mirrors `resolve_llama_binary`).
- `routes` — the export endpoint: admin-gated, quant 400s, manager error → 400.
- `exportCore.js` — quant list + button state via a node subprocess; DOM `node --check`.
- **The real merge+convert+quantize is proven by the spike**, plus one owed end-to-end manual run
  (train → export → confirm the GGUF loads in an external tool). `merge.py` (like `train.py`/`convert.py`)
  isn't importable by the Py3.14 suite — an `ast.parse` gate + the spike/manual run cover it.

## Scope decisions (baked in)

- **Quant selector** with `Q4_K_M` default + `Q5_K_M`, `Q8_0`, `F16`. No per-tensor/imatrix options.
- **Output = a file** in `<DATA_DIR>/training/exports/`, revealed via Open folder. Not a chosen-destination
  picker (v1); the user copies/moves it from there.
- **Reuse the training sidecar venv** (adds only `sentencepiece`); reuse the bundled llama binaries (adds
  only `llama-quantize.exe`).

## Non-goals (this sub-project)

- Auto-registering the exported GGUF as an in-app Local Model / serving it (serving a tuned model already
  works via the serve-adapter `--lora` feature; dropping the exported file into the models dir makes it
  appear in Local Models on its own, but auto-registration is out of scope here).
- imatrix / per-tensor quantization, GGUF split/sharding, and non-LoRA (full-finetune) export.
- A destination-folder picker or cloud upload.

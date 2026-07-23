# Serve a Fine-Tuned Adapter — Design

**Goal:** Take a locally-trained LoRA adapter (produced by the training engine) and make it
chattable in-app: convert the peft adapter to a **GGUF LoRA**, pair it with a matching **base
GGUF**, and serve it through the existing llama.cpp path with `--lora` — so the tuned model shows
up in the model picker and behaves like any other local model. This is sub-project 3 of the
training initiative (engine = sub-project 1, GUI = sub-project 2). Merging the adapter into a
**standalone** GGUF is a deliberate **later** sub-project, not this one.

---

## Feasibility: GATED (prove before building)

Unlike the training engine (spike-proven GO before design), this feature depends on two things
that have **not** been run on this machine yet, so **Task 1 of the plan is a feasibility gate**
(the user's explicit choice), and no manager/UI is built until it passes:

1. **Conversion** — llama.cpp's `convert_lora_to_gguf.py` must convert a real peft adapter
   (`adapter_model.safetensors` + `adapter_config.json`) into a GGUF LoRA, running inside the
   Python-3.11 CUDA sidecar venv. That script imports its sibling `convert_hf_to_gguf.py` and the
   `gguf` pip package, so the gate confirms the exact vendored file set + `gguf` version that works.
2. **Serving** — the bundled `llama-server` must load a base GGUF **plus** the converted adapter
   via `--lora` and produce output that **visibly differs** from the base alone (proving the
   adapter is actually applied, not silently ignored).

**What we already know (de-risks, but doesn't replace, the gate):** the bundled
`llama-server-impl.dll` carries extensive LoRA-adapter support (73 `adapter` / `adapter_lora_info`
/ `lora_invocation` / activated-LoRA string references), so serve-time LoRA is a real capability of
this exact build. The unknowns are the conversion tooling and the precise flag/behavior — which the
gate settles. If the gate is NO-GO (e.g. the bundled `llama-server` rejects the converted adapter,
or `convert_lora_to_gguf.py` can't handle the arch), we stop and reassess (fall back to the deferred
merge-to-standalone path, or a newer llama.cpp) rather than build on sand — the same discipline that
produced the EXL2 and ControlNet NO-GO calls.

The gate can run with a **throwaway, untrained** adapter (a few peft init steps — conversion and
serving don't care whether the weights are trained) plus a **small base GGUF** (e.g. a
Qwen2.5-0.5B GGUF from the existing catalog), so it does not block on the user's owed real training run.

---

## The core constraint

The main app is **Python 3.14** and cannot run the conversion stack (torch/transformers + the
llama.cpp `gguf` tooling). So **conversion runs in the same Python-3.11 CUDA sidecar venv the
training engine already builds** (see the training-engine design). The app only *orchestrates* the
convert subprocess — exactly as it orchestrates the training sidecar. The **serving** side needs no
Python: it's the existing bundled `llama-server` with one extra flag.

---

## Architecture / components

**Backend (main app, Python 3.14):**

- **`training_sidecar/` conversion scripts + `env.py` stack** — vendor llama.cpp's
  `convert_lora_to_gguf.py` and its `convert_hf_to_gguf.py` dependency into `training_sidecar/`
  (bundled as data, like `train.py`), and add `gguf` to `TrainingEnv.STACK` so the venv can run
  them. A small **`training_sidecar/convert.py`** wrapper reads an adapter dir, invokes the vendored
  `convert_lora_to_gguf.py`, writes `<adapter_dir>/adapter.gguf`, and emits a JSON `done`/`error`
  line (same stdout-protocol discipline as `train.py`). Conversion reads the base model's HF config
  (cached from training, else a small one-time download) to determine the architecture.
- **`src/training/convert_manager.py`** — `AdapterConverter`: given a `run_id`, ensure the sidecar
  env is ready, spawn `venv-python convert.py --adapter <dir>`, capture the result, and report
  status (`not_converted` / `converting` / `converted` / `error`). Mirrors `TrainingManager`
  (injectable spawn, never-raises, one conversion at a time). Reuses `TrainingEnv` + the
  UTF-8/`resolve_sidecar_script` machinery from sub-project 1.
- **`src/training/base_resolve.py`** — `resolve_base_gguf(base_model, models) -> {matched, candidates}`:
  pure logic that name-normalizes the adapter's recorded `base_model` (HF id, e.g.
  `Qwen/Qwen2.5-0.5B-Instruct`) and matches it against the local GGUF list in `MODELS_DIR` (e.g.
  `qwen2.5-0.5b-instruct-q4_k_m.gguf`). Returns the best local match (if any) plus all candidates,
  so the UI can auto-fill, prompt a pick, or offer a catalog download. Unit-tested, no I/O beyond
  the passed-in list.
- **`src/localmodels/runtime.py build_serve_argv`** — add an optional `lora: str = None` parameter
  that appends `--lora <path>` (behavior-preserving when `None`). The "no `-ngl` on GPU" auto-fit
  contract is untouched.
- **`src/localmodels/manager.py`** — a serve path that accepts an optional `lora` + a display
  `alias`, threads them through `build_serve_argv`, and registers the endpoint under the **tuned**
  name (so the picker shows e.g. `qwen2.5-0.5b-instruct · run-20260722-… (LoRA)`, not the bare base
  filename). Existing ctx-fit (driven by the **base** GGUF metadata) and the log-tail failure
  surfacing are reused unchanged.
- **`routes/training_routes.py`** (extend, admin-gated) — `POST /api/training/adapters/{run_id}/convert`
  (kick conversion off the event loop), `GET /api/training/adapters/{run_id}` (converted? / base-match
  / served?), `POST /api/training/adapters/{run_id}/serve` (body: chosen base GGUF → resolve → serve
  base + `--lora`). `list_adapters` gains `converted`/`adapter_gguf` fields.

**Frontend (the existing Training/Adapters panel — frontend-only additions):**

- Per-adapter actions in the Adapters list (from sub-project 2): **Convert to GGUF** (shown until
  `converted`), then **Serve** — which shows the auto-matched base for confirmation (or a
  pick/download when unmatched via the existing GGUF catalog), then a **served → open in chat** state.
  Reuses `training.js` patterns; a pure `serveCore.js`-style helper (base-name matching / button-state
  mapping) gets a node-subprocess unit test; the DOM wiring gets a `node --check` gate; the visual
  panel is manual-GUI-verification owed.

## Data flow

Trained adapter (`<DATA_DIR>/training/adapters/<run-id>/`: `adapter_model.safetensors` +
`run_config.json{base_model}`) → **Convert** (sidecar `convert.py` → `adapter.gguf` in the same dir)
→ **Base resolution** (`resolve_base_gguf` auto-matches a local GGUF; else user picks/downloads) →
**Serve** (`LocalModelManager` starts `llama-server --model base.gguf --lora adapter.gguf --alias
<tuned-name>`) → endpoint registered → the tuned model appears in the picker and is chattable. On
finish the Adapters list shows it as *served*.

## Error handling

- Conversion failure (arch unsupported by `convert_lora_to_gguf.py`, or a `gguf`/script version
  mismatch) → a clear `error` status with the sidecar's message tail; never a crash.
- No local base GGUF match → the Serve UI prompts a pick or a catalog download (not an error).
- Base/adapter architecture mismatch → `llama-server` exits or refuses; surfaced from its captured
  log tail (the existing serve-failure pattern), with a hint to check the base matches the adapter's
  `base_model`.
- All managers/routes degrade to `{"error": …}` / an error status and never raise. The main app
  never imports the conversion stack.

## Testing

Headless, in the Python-3.14 suite:
- `resolve_base_gguf` — name-normalized matching (exact, quant-suffix, `-instruct`/`-chat` variants;
  no-match), pure unit tests.
- `build_serve_argv` — `--lora` appended when given, absent when `None`; alias/ctx unaffected.
- `AdapterConverter` — orchestration + status parsing with an **injected** spawn; a failing/raising
  sidecar → error surfaced, never raises.
- `routes/training_routes` — the three new endpoints: admin-gated, convert/serve shapes, manager
  error → 400.
- Frontend — the pure serve-helper via a node subprocess (utf-8); DOM module `node --check`.
- **The real convert + `--lora` serve is proven by the Task-1 feasibility gate**, plus one owed
  end-to-end manual run on the 6 GB GPU (train → convert → serve → chat). `convert.py` (like
  `train.py`) is not importable by the Py3.14 suite — it's covered by the gate + that manual run.

## Scope decisions (baked in)

- **Serve-time single LoRA only** (`--lora <adapter.gguf>` at scale 1.0). No multi-adapter stacking,
  no `--lora-scaled` scale UI in v1.
- **Base = a local GGUF** the adapter was trained on; auto-matched by name, else user picks/downloads
  via the existing catalog. The user confirms the pairing before serving.
- **Conversion runs in the training sidecar venv** (reuses sub-project 1's env; adds only the `gguf`
  pip package + the vendored llama.cpp scripts).

## Non-goals (this sub-project)

- **Merging the adapter into a standalone GGUF** (full-model convert + quantize) — the deferred
  next sub-project the user explicitly wants later.
- Full-finetune (non-LoRA) serving; multi-adapter stacking; per-adapter scale/`--lora-scaled` UI.
- Auto-downloading a base GGUF with no confirmation (rejected — mismatch risk).
- Any change to the training engine or the training GUI's existing behavior beyond the additive
  per-adapter Convert/Serve controls.

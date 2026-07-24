# Serving a Fine-Tuned Adapter

After training a LoRA adapter, serve it locally as a chat model:

1. **Convert** — in the Training panel's Adapters list, click **Convert to GGUF**.
   This runs `convert_lora_to_gguf.py` in the training sidecar venv and writes
   `adapter.gguf` next to the adapter.
2. **Serve** — click **Serve**. The app auto-matches a local base GGUF by the
   adapter's recorded base model (or prompts you to pick/download one). It then
   runs `llama-server --model <base>.gguf --lora adapter.gguf`, aliased to the
   tuned name, and registers it as a local endpoint.
3. **Chat** — pick the tuned model (shown as `<base> · <run-id> (LoRA)`) in the
   model selector and chat.

## API (admin, /api/training)
- `GET /adapters/{run_id}` → status incl. `converted`, `adapter_gguf`, `base_match`.
- `POST /adapters/{run_id}/convert` → convert to a GGUF LoRA.
- `POST /adapters/{run_id}/serve` (body `{base_gguf}`) → serve base + `--lora`.
  `base_gguf` may be a bare GGUF filename (resolved against the models dir) or a full path.

## Feasibility (proven)
See `docs/serve-adapter-feasibility.md` — a real Qwen2.5-0.5B adapter converted to a
GGUF LoRA and the bundled `llama-server` applied it via `--lora` (base vs base+lora output
differs at temperature 0). Confirmed: convert via llama.cpp `convert_lora_to_gguf.py`
(commit `c0bc859`) + the `conversion/` package + the vendored matching `gguf/` package (NOT PyPI gguf — it mismatches); serve flag `--lora`.

## Owed (manual, needs the GPU)
- One real end-to-end run: train → Convert → Serve → chat, verifying the tuned
  output differs from the base. After a rebuild, a frozen boot-check that
  `resolve_convert_script()` resolves and the new routes register.

## Not yet
Merging the adapter into a standalone GGUF (a later sub-project); multi-adapter
stacking; per-adapter scale (`--lora-scaled`).

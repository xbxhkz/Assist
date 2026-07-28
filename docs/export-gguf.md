# Exporting a Fine-Tuned Adapter as a Standalone GGUF

After training a LoRA adapter, export a self-contained GGUF to use outside Assist
(LM Studio, Ollama, sharing):

1. In the Training panel's Adapters list, pick a **quant** (Q4_K_M default, Q5_K_M,
   Q8_0, or F16) and click **Export GGUF**.
2. Assist merges the adapter into the base (in the training sidecar venv), converts
   the full model to GGUF, and quantizes it. The result lands in
   `<DATA_DIR>/training/exports/<base>-<run-id>-<quant>.gguf`.
3. Click **Open folder** to reveal it, then copy/share the file.

## API (admin, /api/training)
- `POST /adapters/{run_id}/export` (body `{quant}`) → merge + convert + quantize.
- `POST /exports/reveal` → open the exports folder.
- `GET /adapters/{run_id}` → also lists this adapter's `exports`.

## Notes
- The F16 intermediate (~2× the model) is deleted after quantizing; only the final
  GGUF is kept. Large models need proportional free disk.
- Proven end-to-end (spike): merge → convert_hf_to_gguf → llama-quantize → the
  standalone GGUF serves with no `--lora`. Owed: one real GUI export on the 6 GB GPU.

## Not yet
Auto-registering the export as an in-app Local Model (drop it into the models dir to
serve it); imatrix / per-tensor quant; GGUF splitting.

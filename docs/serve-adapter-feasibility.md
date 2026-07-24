# Serve-a-Fine-Tuned-Adapter — Feasibility Gate Result

**Verdict: GO.** On this machine (Windows, RTX 4050 6 GB), a real peft LoRA adapter converts to a
GGUF LoRA and the bundled `llama-server` serves the base GGUF + the adapter via `--lora`, with output
that visibly changes when the adapter is applied.

## What was run

- **Adapter:** the real Qwen2.5-0.5B-Instruct LoRA from the training spike
  (`C:\tmp\train_spike\saved_adapter\`: `adapter_config.json` + `adapter_model.safetensors`,
  `base_model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"`).
- **Conversion env:** the training spike's Python 3.11 CUDA venv (`torch 2.5.1+cu121`) + `gguf 0.19.0`.
- **Convert tooling:** llama.cpp at commit **`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`** (2026-07-23):
  `convert_lora_to_gguf.py` + the `conversion/` package.
- **Base GGUF:** `qwen2.5-0.5b-instruct-q4_k_m.gguf` (official `Qwen/Qwen2.5-0.5B-Instruct-GGUF`).
- **Server:** the bundled `build_assets/llama/vulkan/llama-server.exe`.

## Confirmed invocation + flag (use these in the build)

- **Convert:** `python convert_lora_to_gguf.py --outfile <adapter_dir>/adapter.gguf <adapter_dir>`
  - No `--base` required — it reads the base model's HF config (cached from training; else a small
    one-time download of the config).
  - Produced `adapter.gguf` (192 tensors, 4.3 MB — `lora_a`/`lora_b` for q/k/v/output projections).
- **Serve:** `llama-server --model <base>.gguf --lora <adapter.gguf>` (single adapter). Confirmed in
  `llama-server --help`: `--lora FNAME  path to LoRA adapter`.

## Behavioral proof (temperature 0, seed 0 — greedy/deterministic)

Prompt: `Question: greet number 7\nAnswer:`
- **base:** ` The number 7 is a prime number, which means it has exactly two distinct positive divisors: 1 and itself`
- **base + --lora:** ` 7 is a prime number, so it has exactly two positive divisors: 1 and 7. Therefore,`
- **differ: true.** The only difference between the two runs is `--lora adapter.gguf`, so the change
  is caused by the adapter (a server that ignored `--lora` would produce identical greedy output).
  (The spike adapter is only 8 training steps, so the change is mechanical rather than a strong
  behavior shift — a real training run shows a stronger effect. Loading + applying is what the gate
  proves.)

## Implication for the plan (Task 4 correction)

`convert_lora_to_gguf.py` imports **`from conversion import …`** — llama.cpp refactored the model
classes into a **`conversion/` package** (82 `.py` files, ~1.2 MB). So the sidecar must vendor:
- `training_sidecar/convert_lora_to_gguf.py`
- `training_sidecar/conversion/` (the whole package, from commit `c0bc859`)
and **vendor the matching `gguf` package** (`training_sidecar/gguf/`, from the same `c0bc859` checkout's
`gguf-py/gguf/`). **CORRECTION (found by the first real train→convert run):** do NOT `pip install
gguf==0.19.0` — the PyPI 0.19.0 *release* lacks newer enums the `conversion/` package references
(`AttributeError: MODEL_ARCH.DFLASH`) yet reports the same version, silently breaking conversion. Vendoring
`gguf/` beside the convert scripts makes `import gguf` resolve to the matching code via `sys.path[0]`; the
gate originally passed only because it installed `gguf-py` FROM the checkout. Historical note below said to
llama.cpp checkout's `gguf-py`, which is 0.19.0 at this commit). **`convert_hf_to_gguf.py` is NOT
needed** for LoRA→GGUF (it's only for the deferred merge-to-standalone-GGUF sub-project). The
`conversion/` package must sit beside `convert_lora_to_gguf.py` so `from conversion import …` resolves
when the script runs (`python <dir>/convert_lora_to_gguf.py` puts `<dir>` on `sys.path[0]`).

## Gotchas
- `pip install gguf` alone is not enough — the version must match the vendored scripts (0.19.0 here).
- The bundled `llama-server` build does not log adapter loading verbosely (no `lora` line in its log);
  rely on the behavioral difference, not a log grep, to confirm application.

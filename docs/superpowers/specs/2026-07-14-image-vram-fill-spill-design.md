# Image-Model VRAM Fill-and-Spill Design

**Goal:** When serving a local image (diffusion) model on GPU, keep as much of
the model resident in VRAM as fits and stream the overflow from system RAM —
instead of today's blanket offload of *all* weights to RAM. Serving must never
hard-fail when a model is too big: it falls back automatically.

**Scope:** Image models only (sd-server / stable-diffusion.cpp). LLMs
(llama-server) already auto-fit GPU layers and spill to CPU — untouched. No new
settings or UI; the behavior is fully automatic.

---

## Background — current behavior

`src/imagemodels/runtime.py:build_serve_argv` appends `--offload-to-cpu` for
every GPU serve. That forces **all** diffusion/encoder weights into system RAM
and streams them to the GPU per use (sd-server logs `VRAM 0.00MB`). Consequences:

- A model that would fit in VRAM is kept in RAM anyway, running slower than it
  could.
- A model whose *compute* exceeds free VRAM still OOMs (the klein-9b crash),
  because `--offload-to-cpu` only relocates weights, not the compute buffer.

The bundled sd-server (stable-diffusion.cpp, commit `bb84971`) exposes a better
mechanism:

- `--max-vram <GiB>` — "maximum VRAM budget for graph-cut segmented execution."
  The graph is *segmented* to fit the budget, so it bounds compute buffers too,
  not just weights. `0` disables graph splitting; a negative value auto-detects
  free VRAM.
- `--stream-layers` — "residency + prefetch streaming on top of `--max-vram`"
  (no effect without `--max-vram`).

## Architecture

Three small, independently testable pieces plus a fallback in the existing serve
state machine:

```
route (device=gpu) ──▶ manager.start(files, device)
                          │  detect budget: vram_probe() → free_gb − MARGIN
                          │  build an ordered attempt ladder, spawn+await_ready
                          │  each; first ready wins, rest are tried on failure:
                          │
                          │   Tier 1  gpu  --max-vram N --stream-layers   (fill+spill)
                          │   Tier 2  gpu  --offload-to-cpu               (all-RAM)
                          │   Tier 3  cpu  CPU binary, CPU argv           (no GPU)
                          ▼
                       sd-server (Vulkan binary for gpu tiers, CPU binary for Tier 3)
```

### 1. VRAM detection — `services/hwfit/hardware.py`

Add `free_vram_gb()`:

- Runs `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`
  (reusing the module's existing `_run` + absolute-path fallbacks that
  `_detect_nvidia()` already uses).
- Returns the **first** GPU's free memory in GB as a float, or `None` when there
  is no NVIDIA GPU / nvidia-smi is unavailable / the value is non-numeric
  (unified-memory parts report `[N/A]`).
- Never raises (mirrors `_detect_nvidia`'s tolerance).

Rationale for **free** (not total): on a 6 GB card the chat + vision models are
often already resident, so live free VRAM is the honest budget and keeps us from
fighting them.

### 2. Serve argv — `src/imagemodels/runtime.py:build_serve_argv`

New keyword param `max_vram_gb=None`. Only the GPU device block changes:

```python
if device == "gpu":
    if max_vram_gb:                      # fill VRAM, stream the overflow
        argv += ["--max-vram", _fmt_gb(max_vram_gb), "--stream-layers"]
    else:                                # detection failed → proven all-RAM path
        argv += ["--offload-to-cpu"]
    if use_fa:                           # unchanged: SDXL checkpoints omit FA
        argv += ["--diffusion-fa"]
elif threads:
    argv += ["-t", str(threads)]
```

- `_fmt_gb` renders the budget as sd.cpp expects (e.g. `"5"` / `"4.5"`).
- `--vae-tiling` and every non-GPU branch are unchanged.
- `use_fa` is still `False` for SD/SDXL checkpoints (flash attention crashes them
  on the Vulkan build) and `True` otherwise.
- CPU device (`device != "gpu"`) is unaffected — `max_vram_gb` is ignored there.

### 3. Budget + fallback ladder — `src/imagemodels/manager.py:start`

- New injectable dependency `vram_probe=None` on `ImageModelManager.__init__`,
  defaulting to `hardware.free_vram_gb`. Injectable so tests supply a fake.
- Add module constant `VRAM_MARGIN_GB = 1.0` (headroom for compute buffers).
- `start(files, device, steps)` builds an **ordered list of attempts**, each a
  `(attempt_device, max_vram_gb)` pair, and tries them in order — the first that
  becomes ready wins; every failure terminates its dead proc, captures the
  sd-server log tail, and advances to the next attempt:

  ```python
  if device == "gpu":
      free = self._vram_probe()
      budget = max(1.0, free - VRAM_MARGIN_GB) if free else None
      attempts = ([("gpu", budget)] if budget else []) \
               + [("gpu", None), ("cpu", None)]
  else:
      attempts = [("cpu", None)]          # explicit CPU serve: no GPU tiers
  ```

  So a GPU request degrades **Tier 1** (`gpu`, `--max-vram N --stream-layers`) →
  **Tier 2** (`gpu`, `--offload-to-cpu`) → **Tier 3** (`cpu`, CPU binary + CPU
  argv). If VRAM detection returns `None`, Tier 1 is skipped (start at all-RAM).

- Each attempt: `binary = self._resolve_binary(attempt_device)`;
  `port = self._port_chooser()`;
  `argv = build_serve_argv(binary, files, port, device=attempt_device,
  threads=threads, steps=steps, max_vram_gb=max_vram_gb)`; spawn; `await_ready`.
  `resolve_sd_binary(device="cpu")` returns the bundled CPU sd-server
  (`_internal/sd/cpu/sd-server.exe`), so Tier 3 is a plain CPU serve — identical
  to the UI's "CPU" device.
- On the first ready attempt: register + record state, with `state["device"]` set
  to the **actual** `attempt_device` used (so a fallback to CPU is reflected in
  status). If **all** attempts fail → raise the existing `RuntimeError` with the
  last sd-server tail.
- The model-path resolution fix (`files.get("diffusion_model") or
  files.get("checkpoint")`) and single-active-server locking are retained.

### Data flow / error handling

- Only GPU requests get a budget + fallback ladder; an explicit CPU request is the
  single-attempt `[("cpu", None)]` — identical to today.
- Tier 1 is included only when a real budget was detected (`budget is not None`);
  a detection-failure GPU request starts at Tier 2 (`--offload-to-cpu`).
- Each attempt picks a fresh port and re-resolves its binary, so GPU→CPU fallback
  cleanly switches from the Vulkan build to the CPU build.
- `_spawn` already rotates `sd-server.log` → `.prev` per launch; failure tails are
  captured in memory before advancing, so the raised error still reports why the
  last attempt failed.

## Testing

**Unit — `build_serve_argv` (`tests/test_imagemodels_runtime.py`):**
- `max_vram_gb=5, device="gpu"` → contains `--max-vram 5 --stream-layers`, and
  **not** `--offload-to-cpu`.
- `max_vram_gb=None, device="gpu"` → contains `--offload-to-cpu`, not `--max-vram`
  (regression guard for today's behavior).
- SDXL checkpoint + `max_vram_gb=5` → `--max-vram` present, `--diffusion-fa` absent.
- `device="cpu"` ignores `max_vram_gb` (no `--max-vram`, no `--offload-to-cpu`).

**Unit — `free_vram_gb` (new `tests/test_hwfit_free_vram.py`, matching the
granular `test_hwfit_*` convention; monkeypatch `hardware._run`):**
- Mock `_run` → `"5432"` ⇒ returns ≈ 5.30 (5432 MiB / 1024).
- Mock multi-GPU output → returns the first row.
- Mock `_run` → `None` / `"[N/A]"` / `""` ⇒ returns `None`.

**Unit — `manager.start` fallback (`tests/test_imagemodels_manager.py`).** The
fake `resolve_binary` returns a device-tagged path (e.g. `f"/bin/sd-{device}"`)
so tests can assert which binary each attempt used:
- `vram_probe` → 6.0, first spawn ready ⇒ single spawn; argv has `--max-vram`
  (`≈5` after the 1 GB margin) + `--stream-layers`; registered once; state
  `device == "gpu"`.
- `vram_probe` → 6.0, Tier 1 never ready, Tier 2 ready ⇒ two spawns; first argv
  has `--max-vram`, second has `--offload-to-cpu`, both the `sd-gpu` binary;
  running, state `device == "gpu"`.
- `vram_probe` → 6.0, Tiers 1 & 2 never ready, Tier 3 ready ⇒ three spawns; third
  uses the `sd-cpu` binary and its argv has neither `--max-vram` nor
  `--offload-to-cpu`; running, state `device == "cpu"`.
- `vram_probe` → `None` ⇒ first attempt is `--offload-to-cpu` (Tier 1 skipped);
  on failure, falls through to the CPU attempt.
- Explicit `device="cpu"` ⇒ single CPU attempt; `vram_probe` not consulted.
- **All** attempts fail ⇒ `RuntimeError`, nothing registered (existing behavior).

**Live-verify (6 GB RTX 4050, manual, at packaging):**
- Small model (klein-4b): serves Tier 1, `sd-server.log` shows non-zero resident
  VRAM; generates.
- Large model (klein-9b, historically OOM): serves via Tier 1 streaming *or*
  cleanly falls back to Tier 2, and if the GPU still can't, lands on the Tier 3
  CPU serve — no unhandled 500.
- Confirm `--max-vram`/`--stream-layers` are accepted by the bundled binary (guard
  against the older `--auto-fit` VAE-OOM regression).
- Confirm the Tier 3 CPU binary (`sd/cpu/sd-server.exe`) is present in the frozen
  bundle and serves when forced.

## Non-goals

- LLM serving (already spills GPU→CPU).
- Any settings/UI or per-serve manual budget.
- Changing quantization, resolution, or step defaults.

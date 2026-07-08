# Native Local Image Generation (sd-server) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Serve GGUF diffusion models (FLUX.1 now) locally via a bundled stable-diffusion.cpp `sd-server`, on CPU or GPU, auto-registering an OpenAI-compatible `model_type="image"` endpoint so the existing gallery/chat image flow uses it.

**Architecture:** New `src/imagemodels/` subsystem mirroring `src/localmodels/` (runtime + manager + store), plus encoder resolution, routes, launcher picker, packaging, and a small UI card. Reuse the LLM lessons: sync `/v1/models` probe on register, size-scaled readiness timeout, robust process-tree kill, `CREATE_NO_WINDOW`, off-event-loop serve.

**Tech Stack:** FastAPI, httpx/urllib, subprocess; vanilla JS/CSS; PyInstaller; stable-diffusion.cpp `sd-server`. Tests: pytest with injected fakes (no real binary/GPU).

## Global Constraints

- `sd-server` FLUX flags: `--diffusion-model <gguf> --t5xxl <f> --clip_l <f> --vae <f> --host 127.0.0.1 --port <p>`; CPU adds `-t <threads>`; GPU (Vulkan build) keeps encoders/VAE on CPU for low VRAM. Verify exact flag spellings against the pinned binary before the packaging task.
- Image endpoint base_url = `http://127.0.0.1:{port}/v1` and `ModelEndpoint(model_type="image", endpoint_kind="local")` so `routes/gallery/gallery_routes.py` (`_visible_image_endpoint_query` filters `model_type=="image"`; POSTs `{base_url}/images/generations`) picks it up.
- One image model served at a time. Data dirs under `<DATA_DIR>`: `image-models/` (models), `image-models/encoders/` (shared T5/CLIP/VAE), `logs/sd-server.log`.
- FLUX needs 4 files; only the diffusion GGUF is user-supplied per serve. T5/CLIP/VAE are shared and resolved: explicit path > sibling of the GGUF > shared encoders dir.

---

### Task 1: `src/imagemodels/runtime.py` (pure helpers)

**Files:** Create `src/imagemodels/__init__.py` (empty), `src/imagemodels/runtime.py`; Test `tests/test_imagemodels_runtime.py`

**Interfaces:** Produces `local_image_endpoint_url(port)->str`, `build_serve_argv(binary, files: dict, port, device="cpu", host="127.0.0.1", threads=0)->list`, `resolve_sd_binary(device="cpu", path_lookup=shutil.which, frozen_base=None, dev_base=None)->str`, `list_gguf_image_models(dir)->list`.

- [x] **Step 1 — failing tests** `tests/test_imagemodels_runtime.py`:
```python
import os
import src.imagemodels.runtime as rt

def test_endpoint_url():
    assert rt.local_image_endpoint_url(8200) == "http://127.0.0.1:8200/v1"

def test_build_argv_flux_four_files_cpu():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/clip.safetensors", "vae": "/m/ae.safetensors"}
    argv = rt.build_serve_argv("/x/sd-server", files, 8200, device="cpu", threads=8)
    for f in ("--diffusion-model", "/m/flux.gguf", "--t5xxl", "/m/t5.gguf",
              "--clip_l", "/m/clip.safetensors", "--vae", "/m/ae.safetensors",
              "--host", "127.0.0.1", "--port", "8200", "-t", "8"):
        assert f in argv

def test_build_argv_gpu_offloads_encoders():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/clip.safetensors", "vae": "/m/ae.safetensors"}
    argv = rt.build_serve_argv("/x/sd-server", files, 8200, device="gpu")
    assert "--clip-on-cpu" in argv and "--vae-on-cpu" in argv

def test_resolve_prefers_path_binary():
    got = rt.resolve_sd_binary(device="cpu",
        path_lookup=lambda n: "/usr/bin/sd-server" if n == "sd-server" else None)
    assert got == "/usr/bin/sd-server"

def test_resolve_uses_bundled_gpu(tmp_path):
    b = tmp_path / "sd" / "vulkan"; b.mkdir(parents=True)
    name = "sd-server.exe" if os.name == "nt" else "sd-server"
    (b / name).write_text("x")
    got = rt.resolve_sd_binary(device="gpu", path_lookup=lambda n: None, frozen_base=str(tmp_path))
    assert got == str(b / name)

def test_list_filters_gguf(tmp_path):
    (tmp_path / "flux.gguf").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("n")
    got = rt.list_gguf_image_models(str(tmp_path))
    assert [m["name"] for m in got] == ["flux.gguf"]
```
- [x] **Step 2 — run, expect FAIL** (module missing): `python -m pytest tests/test_imagemodels_runtime.py --import-mode=importlib -q`
- [x] **Step 3 — implement** `src/imagemodels/runtime.py`:
```python
"""Pure helpers for native image-model serving (sd-server). No process/DB here."""
import os
import shutil
import sys


def local_image_endpoint_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


def build_serve_argv(binary, files, port, device="cpu", host="127.0.0.1", threads=0):
    """sd-server argv for a FLUX GGUF (diffusion + t5xxl + clip_l + vae)."""
    argv = [
        binary,
        "--diffusion-model", files["diffusion_model"],
        "--t5xxl", files["t5xxl"],
        "--clip_l", files["clip_l"],
        "--vae", files["vae"],
        "--host", host,
        "--port", str(port),
    ]
    if device == "gpu":
        # Keep the big encoders + VAE on CPU so a small (6GB) GPU can host the
        # diffusion model; flash-attention trims VRAM further.
        argv += ["--clip-on-cpu", "--vae-on-cpu", "--diffusion-fa"]
    elif threads:
        argv += ["-t", str(threads)]
    return argv


def _bundled_name():
    return "sd-server.exe" if os.name == "nt" else "sd-server"


def resolve_sd_binary(device="cpu", path_lookup=shutil.which, frozen_base=None, dev_base=None):
    found = path_lookup("sd-server") or path_lookup("sd-server.exe")
    if found:
        return found
    sub = "vulkan" if device == "gpu" else "cpu"
    name = _bundled_name()
    base = frozen_base
    if base is None and getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
    if base:
        cand = os.path.join(base, "sd", sub, name)
        if os.path.isfile(cand):
            return cand
    if dev_base is None:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dev_base = os.path.join(repo, "build_assets", "sd", sub)
    cand = os.path.join(dev_base, name)
    if os.path.isfile(cand):
        return cand
    raise RuntimeError("sd-server not found: no server on PATH and no bundled binary.")


def list_gguf_image_models(models_dir):
    out = []
    if os.path.isdir(models_dir):
        for fn in sorted(os.listdir(models_dir)):
            if fn.lower().endswith(".gguf"):
                p = os.path.join(models_dir, fn)
                out.append({"name": fn, "path": p, "size": os.path.getsize(p)})
    return out
```
- [x] **Step 4 — run, expect PASS.** **Step 5 — commit.**

---

### Task 2: `src/imagemodels/encoders.py` (aux-file resolution)

**Files:** Create `src/imagemodels/encoders.py`; Test `tests/test_imagemodels_encoders.py`

**Interfaces:** Produces `resolve_flux_files(diffusion_model, t5xxl=None, clip_l=None, vae=None) -> dict` (raises `MissingEncoderError(list_of_missing)` if any unresolved), `encoders_dir()->str`, `ENCODER_FILENAMES` (canonical names to look for).

- [x] **Step 1 — failing tests**:
```python
import os, pytest
import src.imagemodels.encoders as enc

def test_resolves_from_sibling(tmp_path, monkeypatch):
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(tmp_path / "img"))
    d = tmp_path / "flux"; d.mkdir()
    for f in ["flux.gguf", "t5xxl.gguf", "clip_l.safetensors", "ae.safetensors"]:
        (d / f).write_bytes(b"x")
    got = enc.resolve_flux_files(str(d / "flux.gguf"))
    assert got["t5xxl"].endswith("t5xxl.gguf") and got["vae"].endswith("ae.safetensors")

def test_explicit_paths_win(tmp_path, monkeypatch):
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(tmp_path / "img"))
    d = tmp_path / "flux"; d.mkdir()
    (d / "flux.gguf").write_bytes(b"x")
    t5 = d / "my_t5.gguf"; t5.write_bytes(b"x")
    clip = d / "c.safetensors"; clip.write_bytes(b"x")
    vae = d / "v.safetensors"; vae.write_bytes(b"x")
    got = enc.resolve_flux_files(str(d / "flux.gguf"), t5xxl=str(t5), clip_l=str(clip), vae=str(vae))
    assert got["t5xxl"] == str(t5)

def test_missing_raises_named(tmp_path, monkeypatch):
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(tmp_path / "img"))
    d = tmp_path / "flux"; d.mkdir()
    (d / "flux.gguf").write_bytes(b"x")
    with pytest.raises(enc.MissingEncoderError) as ei:
        enc.resolve_flux_files(str(d / "flux.gguf"))
    assert "t5xxl" in ei.value.missing
```
- [x] **Step 2 — run, expect FAIL.**
- [x] **Step 3 — implement** `src/imagemodels/encoders.py`: `IMAGE_MODELS_DIR` from `src.constants`; `encoders_dir()` = `<IMAGE_MODELS_DIR>/encoders`; `ENCODER_FILENAMES = {"t5xxl": ["t5xxl.gguf","t5xxl_fp16.safetensors","t5xxl_q8_0.gguf"], "clip_l": ["clip_l.safetensors"], "vae": ["ae.safetensors","vae.safetensors"]}`. `resolve_flux_files` for each of t5xxl/clip_l/vae: use explicit arg if given+exists; else first match by known filename in the GGUF's dir; else in `encoders_dir()`; collect missing; raise `MissingEncoderError(missing)` if any. Return `{"diffusion_model":..., "t5xxl":..., "clip_l":..., "vae":...}` of realpaths.
- [x] **Step 4 — run, expect PASS.** **Step 5 — commit.**

---

### Task 3: `src/imagemodels/store.py` (endpoint registration)

**Files:** Create `src/imagemodels/store.py`; Test `tests/test_imagemodels_store.py`

**Interfaces:** Produces `register_image_endpoint(name, base_url, session_factory=None, probe=None)->str`, `unregister_image_endpoint(id, session_factory=None)`.

- [x] Mirror `src/localmodels/store.py` exactly, but: `model_type="image"`, id prefix `"img-local-"`, and the sync probe hits the sd-server `/v1/models` (reuse `routes.model_routes._probe_endpoint`). Tests mirror `tests/test_localmodels_store.py` (in-memory SQLite): create/update sets `model_type=="image"`; probe populates `cached_models`; unregister deletes. **Commit.**

---

### Task 4: `src/imagemodels/manager.py` (`ImageModelManager`)

**Files:** Create `src/imagemodels/manager.py`; Test `tests/test_imagemodels_manager.py`

**Interfaces:** Produces `ImageModelManager` with `start(files: dict, device="cpu")->dict`, `stop()`, `status()`, `list_models()`; `get_manager()`.

- [x] Copy `src/localmodels/manager.py` structure and adapt: spawn `build_serve_argv(binary, files, port, device, threads)`; readiness probes `local_image_endpoint_url(port)+"/models"`; `_default_force_kill` + escalation; `CREATE_NO_WINDOW`; size-scaled timeout keyed on the diffusion GGUF size; log to `sd-server.log`; on ready → `register_image_endpoint(name=basename(diffusion_model), base_url=url)`; state stores `device`. Tests mirror `tests/test_imagemodels_manager.py` with the injected-fakes helper (FakeProc + autouse `IMAGE_MODELS_DIR` isolation fixture — persist writes stay off real data). **Commit.**

---

### Task 5: Routes + app wiring

**Files:** Create `routes/imagemodels_routes.py`; Modify `app.py` (include router + shutdown stop); `launcher.py` (`pick_image_model`).

- [x] `setup_imagemodels_routes()` admin-guarded (`Depends(require_admin)`): `POST /api/imagemodels/serve {diffusion_model, device, t5xxl?, clip_l?, vae?}` → `resolve_flux_files(...)` (400 `MissingEncoderError` names missing files) → `await asyncio.to_thread(get_manager().start, files, device)` (503 RuntimeError); `POST /stop`; `GET /status`; `GET /models` (list_gguf_image_models + linked); `POST /add-external {path}` (reuse the linked-model pattern for a picked file). `app.py`: `app.include_router(setup_imagemodels_routes())` + stop hook in the lifespan shutdown next to `get_manager().stop()`. `launcher.py`: add `pick_image_model()` to `_JsApi` (open dialog `*.gguf;*.safetensors`). **Verify** `python -c "import routes.imagemodels_routes"`. **Commit.**

---

### Task 6: Packaging — `scripts/fetch_sd_server.py` + `Assist.spec`

**Files:** Create `scripts/fetch_sd_server.py`; Modify `Assist.spec`.

- [x] `fetch_sd_server.py` (mirror `scripts/fetch_llama_server.py`): download a **pinned** `leejet/stable-diffusion.cpp` Windows release — the AVX2 (CPU) and Vulkan zips — extract `sd-server.exe` + `*.dll` into `build_assets/sd/cpu/` and `build_assets/sd/vulkan/`. Pin the tag + asset names after checking the releases page; error clearly if `sd-server.exe` is absent (older releases shipped only `sd`). `Assist.spec`: add `build_assets/sd` → `sd/` in `datas` (like the `llama` bundle). **Run the fetch** (network) and confirm both binaries land. **Commit** (build_assets gitignored; commit the script + spec).

---

### Task 7: UI — Image Models card

**Files:** Modify `static/index.html` (Local Models modal) + create `static/js/imageModels.js`.

- [x] Add an "Image Models" card: a Browse button (native `pick_image_model` when available) + local-model list, a **CPU / GPU** radio, a **Serve/Stop**, a running-status line, and a "Download FLUX encoders" affordance when a serve fails with missing encoders. `imageModels.js`: mirror `localModels.js` — `serve` posts `{diffusion_model, device}`, shows "Starting… (image models load slowly)", refresh status; on 400 missing-encoders, surface which files to add. **Commit.**

---

### Task 8: Rebuild + verify

- [x] `python -m PyInstaller --noconfirm --clean Assist.spec` (bundles `sd/`), then ISCC. Boot; `GET /api/imagemodels/models` responds.
- [ ] **User manual test:** serve their FLUX.1 GGUF on CPU then GPU, generate an image via the gallery; try FLUX 2 and record whether sd.cpp loads it (expected: FLUX.1 works; FLUX 2 = "unsupported architecture" unless the pinned sd.cpp supports it).

**Verified 2026-07-07:** 20/20 imagemodels tests pass. Packaged build (installer 466 MB, sd/cpu + sd/vulkan bundled in `_internal/sd/`) booted against an isolated data dir; `GET /api/imagemodels/models` → 200 `{"models":[]}`, `GET /api/imagemodels/status` → 200 not-running. Manual FLUX serve/generate test pending user.

**FLUX.2 finding (2026-07-07):** the pinned sd.cpp (`master-765-bb84971`) *does* support FLUX.2 — but via a different file set: `--llm <text encoder>` replaces `--t5xxl`/`--clip_l`, plus the flux2 VAE. FLUX.2 GGUFs carry the same `flux` architecture tag as FLUX.1 and no `general.name`, so they can't be told apart by header; the serve route branches on a filename heuristic (`looks_like_flux2`).

**GPU VAE-OOM fix (2026-07-07 evening):** first live GPU serve (klein Q4, RTX 4050 6GB) sampled fine but 500'd at 1024×1024: VAE decode wanted an 8.5GB Vulkan compute buffer, and `--auto-fit`'s tiling retry re-allocated the same size (broken for the flux2 VAE at bb84971). 512×512 verified working end-to-end via the OpenAI endpoint. Fix: drop `--auto-fit` (it *ignores* `--backend` per docs/backend.md) in favor of explicit `--backend diffusion=vulkan0,te=cpu,vae=cpu` + `--vae-tiling` on all devices (flag syntax parse-checked against the bundled binary).

**FLUX.2 support implemented (2026-07-07, same day):** per sd.cpp's own `docs/flux2.md` at the pinned commit, klein-4B/9B use a **Qwen3-4B/8B** `--llm` encoder (only FLUX.2-dev needs the 24B Mistral) with `--cfg-scale 1.0 --steps 4` (step-distilled). Added `resolve_flux2_files` (llm + flux2-specific VAE names — deliberately not matching FLUX.1's `ae.safetensors`), the `llm`-keyed argv layout in `build_serve_argv` (cfg 1.0 baked in for both families; steps 4 only for klein), and the serve-route branch. Encoders downloaded ungated: `Qwen3-4B-Q4_K_M.gguf` (unsloth) and `flux2_ae.safetensors` (ai-toolkit/flux2_vae, Apache-2.0).

**Manual-test fallout fixed (2026-07-07):** (a) `image_gen_enabled` ships default-False — flipped in the user's settings.json (chat/gallery gate, not a code bug); (b) GGUFs were routed to cards by directory, not architecture — image GGUFs downloaded into `data/models/` showed in the LLM card (llama-server: `unknown architecture: 'flux'`) and not in the Image Models card. Fixed with `src/gguf_meta.py` header sniffing: LLM list excludes diffusion/encoder archs, image card merges diffusion-arch GGUFs from the download dir, and the card renders a clickable model list.

## Self-Review

- **Spec coverage:** runtime (T1), encoders (T2), store (T3), manager (T4), routes+picker+wiring (T5), packaging (T6), UI (T7), rebuild/verify (T8). All spec components covered.
- **Placeholders:** T1–T5 carry full code/interfaces; the sd-server exact flag spellings + release asset names are explicitly "verify against the pinned binary" (T6/Global Constraints) because they depend on the external release — flagged, not hidden.
- **Consistency:** `build_serve_argv(binary, files, port, device, host, threads)`, `resolve_flux_files`, `register_image_endpoint`, `local_image_endpoint_url`, `ImageModelManager.start(files, device)` consistent across tasks.

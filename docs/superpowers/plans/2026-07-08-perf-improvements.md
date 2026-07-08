# Perf & UX Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Assist faster to use (GPU LLM serving, fast image decode, size/steps controls, RAM guard) and faster to develop (incremental rebuilds, static hot-copy).

**Architecture:** Extend the proven cpu/vulkan dual-binary pattern from `sd/` to `llama/`; thread a `device` param through the localmodels runtime→manager→routes→UI stack (mirroring imagemodels); add opt-in knobs (image_size setting, serve-time steps, TAESD fast decode) at the layers that already own them; guard RAM contention in the frontend where both subsystem statuses are visible.

**Tech Stack:** llama.cpp b9867 (CPU+Vulkan zips), stable-diffusion.cpp bb84971, FastAPI, pytest (`--import-mode=importlib`), vanilla JS, PowerShell build scripts, PyInstaller + Inno Setup.

## Global Constraints

- All pytest runs use `--import-mode=importlib` (ultralytics shadows `tests/` otherwise).
- llama.cpp release pin: `b9867`; assets `llama-b9867-bin-win-cpu-x64.zip` and `llama-b9867-bin-win-vulkan-x64.zip`.
- GPU = Vulkan builds only (`vulkan0` backend name; RTX 4050 6GB target). PATH-installed binaries still win resolution.
- `build_assets/` is gitignored — commit scripts/spec, not binaries.
- One LLM + one image model max, each singleton manager.
- Frontend text guards live in `tests/test_*_ui.py`; brand text must say Assist (tests/test_brand_strings.py).
- Windows install dir: `C:\Program Files\Assist`, static bundled at `_internal\static`.

---

### Task 1: Dev fast-loop — `-Fast` builds + static hot-copy

**Files:**
- Modify: `build-windows-portable.ps1` (param block + fetch/clean gating)
- Modify: `build-installer.ps1` (param passthrough)
- Create: `scripts/dev-sync-static.ps1`

**Interfaces:**
- Produces: `.\build-installer.ps1 -Fast` (incremental PyInstaller, skip pip/fetches when assets exist); `scripts\dev-sync-static.ps1` (robocopy `static\` into the installed app, self-elevating).

- [ ] **Step 1: Add `-Fast` to build-windows-portable.ps1.** At the top (after the `<# ... #>` comment block, before `$ErrorActionPreference`):

```powershell
param([switch]$Fast)
```

Gate the dependency/fetch steps — replace the unconditional bodies with:

```powershell
if ($Fast -and (Test-Path "build_assets\llama") -and (Test-Path "build_assets\sd")) {
    Write-Step "Fast build: skipping pip install + asset fetches (assets present)"
} else {
    Write-Step "Installing build dependencies"
    & $pyExe -m pip install --upgrade pip --quiet
    & $pyExe -m pip install -r requirements.txt -r requirements-desktop.txt pyinstaller
    if ($LASTEXITCODE -ne 0) { Fail "Dependency install failed." }

    Write-Step "Vendoring offline embedding model"
    & $pyExe scripts/fetch_embedding_model.py
    if ($LASTEXITCODE -ne 0) { Fail "Embedding model fetch failed." }

    Write-Step "Vendoring llama-server (CPU + Vulkan)"
    & $pyExe scripts/fetch_llama_server.py
    if ($LASTEXITCODE -ne 0) { Fail "llama-server fetch failed." }
}
```

And gate the clean:

```powershell
Write-Step "Building portable exe bundle"
if (-not $Fast) { Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue }
$piArgs = @('--noconfirm'); if (-not $Fast) { $piArgs += '--clean' }
& $pyExe -m PyInstaller @piArgs Assist.spec
if ($LASTEXITCODE -ne 0) { Fail "PyInstaller build failed." }
```

- [ ] **Step 2: Pass `-Fast` through build-installer.ps1.** Add `param([switch]$Fast)` at top (after `#Requires`), and change the portable-build call to:

```powershell
if ($Fast) { & .\build-windows-portable.ps1 -Fast } else { & .\build-windows-portable.ps1 }
```

- [ ] **Step 3: Create `scripts/dev-sync-static.ps1`:**

```powershell
#Requires -Version 5.1
<#
  Copy the repo's static\ tree into the installed app so UI-only changes are
  testable without a rebuild. Restart Assist afterwards to load them.
  Self-elevates (Program Files needs admin).
#>
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$dest = "C:\Program Files\Assist\_internal\static"
if (-not (Test-Path $dest)) { Write-Host "ERROR: $dest not found (is Assist installed?)" -ForegroundColor Red; exit 1 }
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$admin = (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Start-Process -FilePath "pwsh" -Verb RunAs -Wait -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',"$PSScriptRoot\dev-sync-static.ps1")
    exit $LASTEXITCODE
}
robocopy "$repo\static" $dest /E /NJH /NJS /NDL /NP | Out-Null
if ($LASTEXITCODE -ge 8) { Write-Host "robocopy failed ($LASTEXITCODE)" -ForegroundColor Red; exit 1 }
Write-Host "static synced to $dest — restart Assist to load changes." -ForegroundColor Green
exit 0
```

- [ ] **Step 4: Verify.** Run `powershell -ExecutionPolicy Bypass -File .\scripts\dev-sync-static.ps1` (accept elevation) — expect "static synced". Run `.\build-windows-portable.ps1 -Fast` — expect the skip message and a PyInstaller run without `--clean`.
- [ ] **Step 5: Commit** `build-windows-portable.ps1 build-installer.ps1 scripts/dev-sync-static.ps1` as `build: -Fast incremental builds + dev-sync-static hot-copy`.

---

### Task 2: Vendor Vulkan llama-server (fetch script + layout)

**Files:**
- Modify: `scripts/fetch_llama_server.py`

**Interfaces:**
- Produces: `build_assets/llama/cpu/llama-server.exe` and `build_assets/llama/vulkan/llama-server.exe` (+DLLs). The flat legacy layout (`build_assets/llama/llama-server.exe`) is superseded. `Assist.spec` needs no change — `('build_assets/llama', 'llama')` copies subdirs recursively.

- [ ] **Step 1: Rewrite the fetch script for both builds:**

```python
"""Vendor prebuilt llama-server (CPU + Vulkan) into build_assets/llama/.

Downloads pinned llama.cpp Windows release zips and extracts llama-server.exe
(plus DLLs) into build_assets/llama/cpu/ and build_assets/llama/vulkan/.
Pinned for reproducibility; bump LLAMA_TAG deliberately after verifying.
"""
import io
import os
import sys
import urllib.request
import zipfile

LLAMA_TAG = os.getenv("LLAMA_RELEASE_TAG", "b9867")
_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"
ZIPS = {
    "cpu": f"{_BASE}/llama-{LLAMA_TAG}-bin-win-cpu-x64.zip",
    "vulkan": f"{_BASE}/llama-{LLAMA_TAG}-bin-win-vulkan-x64.zip",
}
ASSET_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build_assets", "llama")
)


def _fetch(kind: str, url: str) -> bool:
    dest = os.path.join(ASSET_ROOT, kind)
    exe = os.path.join(dest, "llama-server.exe")
    if os.path.isfile(exe):
        print(f"{kind}: already vendored, skipping")
        return True
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {kind} llama-server from {url} ...")
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            base = os.path.basename(member)
            if not base:
                continue
            if base == "llama-server.exe" or base.lower().endswith(".dll"):
                with zf.open(member) as src, open(os.path.join(dest, base), "wb") as dst:
                    dst.write(src.read())
    if not os.path.isfile(exe):
        print(f"ERROR: llama-server.exe not found in {kind} zip", file=sys.stderr)
        return False
    print(f"{kind} llama-server vendored into {dest}")
    return True


def main() -> int:
    ok = all(_fetch(k, u) for k, u in ZIPS.items())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it** (network): `python scripts/fetch_llama_server.py`. Expect both `build_assets/llama/cpu/llama-server.exe` and `build_assets/llama/vulkan/llama-server.exe`. Delete stale flat files: `Remove-Item build_assets/llama/*.exe, build_assets/llama/*.dll -ErrorAction SilentlyContinue` (root level only).
- [ ] **Step 3: Smoke-test** `& build_assets\llama\vulkan\llama-server.exe --version` — expect a version banner mentioning vulkan or listing the RTX 4050.
- [ ] **Step 4: Commit** the script as `build(llm): vendor CPU + Vulkan llama-server (cpu/, vulkan/ layout)`.

---

### Task 3: Runtime — device-aware resolve + `-ngl` argv

**Files:**
- Modify: `src/localmodels/runtime.py` (`build_serve_argv`, `resolve_llama_binary`)
- Test: `tests/test_localmodels_runtime.py`

**Interfaces:**
- Consumes: `build_assets/llama/{cpu,vulkan}/` layout from Task 2.
- Produces: `build_serve_argv(binary, model_path, port, ctx_size=4096, host="127.0.0.1", device="cpu") -> list` (adds `-ngl 999 --flash-attn on` when `device=="gpu"`); `resolve_llama_binary(device="cpu", path_lookup=..., frozen_base=None, dev_base=None) -> str` (searches `llama/<cpu|vulkan>/`, falling back to the legacy flat `llama/` dir).

- [ ] **Step 1: Failing tests** (append to `tests/test_localmodels_runtime.py`):

```python
def test_build_serve_argv_gpu_offloads_layers():
    argv = rt.build_serve_argv("/x/llama-server", "/m/model.gguf", 8123, device="gpu")
    assert argv[argv.index("-ngl") + 1] == "999"
    assert "--flash-attn" in argv


def test_build_serve_argv_cpu_has_no_gpu_flags():
    argv = rt.build_serve_argv("/x/llama-server", "/m/model.gguf", 8123)
    assert "-ngl" not in argv


def test_resolve_binary_gpu_uses_vulkan_subdir(tmp_path):
    b = tmp_path / "llama" / "vulkan"; b.mkdir(parents=True)
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    (b / name).write_text("stub")
    got = rt.resolve_llama_binary(device="gpu", path_lookup=lambda n: None,
                                  frozen_base=str(tmp_path))
    assert got == str(b / name)


def test_resolve_binary_falls_back_to_legacy_flat_dir(tmp_path):
    b = tmp_path / "llama"; b.mkdir()
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    (b / name).write_text("stub")
    got = rt.resolve_llama_binary(device="cpu", path_lookup=lambda n: None,
                                  frozen_base=str(tmp_path))
    assert got == str(b / name)
```

- [ ] **Step 2: Run, expect FAIL:** `python -m pytest tests/test_localmodels_runtime.py --import-mode=importlib -q` (TypeError: unexpected keyword `device`).
- [ ] **Step 3: Implement.** In `build_serve_argv`, add `device="cpu"` keyword; after the base list append:

```python
    if device == "gpu":
        # Offload every layer to the GPU (llama caps at the model's layer
        # count); flash attention is a VRAM + speed win on RTX cards.
        argv += ["-ngl", "999", "--flash-attn", "on"]
```

In `resolve_llama_binary`, add `device="cpu"` keyword and search `llama/<sub>/<name>` then legacy `llama/<name>` under both `frozen_base` and `dev_base`:

```python
    sub = "vulkan" if device == "gpu" else "cpu"
    ...
    if base:
        for rel in (os.path.join("llama", sub, name), os.path.join("llama", name)):
            cand = os.path.join(base, rel)
            if os.path.isfile(cand):
                return cand
    if dev_base is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dev_base = os.path.join(repo_root, "build_assets", "llama")
    for cand in (os.path.join(dev_base, sub, name), os.path.join(dev_base, name)):
        if os.path.isfile(cand):
            return cand
```

- [ ] **Step 4: Run, expect PASS** (whole file). **Step 5: Commit** as `feat(llm): device-aware binary resolve + -ngl GPU argv`.

---

### Task 4: Manager — device passthrough + persistence

**Files:**
- Modify: `src/localmodels/manager.py` (`start`, `_persist_last_model`, `autoserve_last_model`, `status`)
- Test: `tests/test_localmodels_manager.py`

**Interfaces:**
- Consumes: Task 3 signatures.
- Produces: `LocalModelManager.start(model_path, device="cpu")`; `status()` includes `"device"`; `last_model.json` gains `"device"`; `autoserve_last_model` re-serves with the saved device.

- [ ] **Step 1: Failing tests** (append; mirror the file's existing fake-injection style — `resolve_binary` fakes must accept the new kwarg):

```python
def test_start_gpu_resolves_gpu_binary_and_reports_device():
    seen = []
    mgr, spawned, *_ = make_manager(resolve_binary=lambda device="cpu": seen.append(device) or "/b")
    st = mgr.start("/m/model.gguf", device="gpu")
    assert seen == ["gpu"]
    assert st["device"] == "gpu"
    assert "-ngl" in spawned[0][0]


def test_autoserve_restores_device(tmp_path):
    f = tmp_path / "last_model.json"
    m = tmp_path / "m.gguf"; m.write_bytes(b"x")
    f.write_text(json.dumps({"model_path": str(m), "device": "gpu"}))
    calls = {}
    class Mgr:
        def status(self): return {"running": False}
        def start(self, path, device="cpu"): calls["args"] = (path, device); return {"running": True}
    autoserve_last_model(manager=Mgr(), model_file=str(f))
    assert calls["args"] == (str(m), "gpu")
```

(Adjust `make_manager` to accept/forward a `resolve_binary` kwarg if it doesn't already; check its definition at the top of the test file first.)

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** `start(self, model_path: str, device: str = "cpu")`: call `self._resolve_binary(device=device)`; build argv with `device=device`; store `"device": device` in `self._state`; pass device to `_persist_last_model(model_path, device)` which writes `{"model_path": ..., "device": ...}`. `status()` returns `"device": self._state.get("device")` when running, `None` otherwise (keep existing keys). `autoserve_last_model`: read `device = (data or {}).get("device") or "cpu"` and call `mgr.start(path, device=device)`. Note: `resolve_binary` is stored at construction — call it as `self._resolve_binary(device=device)`; the default `resolve_llama_binary` accepts it after Task 3.
- [ ] **Step 4: Run the whole manager test file, expect PASS** (existing tests keep passing — default device "cpu" preserves behavior; update any test whose fake `resolve_binary` takes no kwargs).
- [ ] **Step 5: Commit** as `feat(llm): device passthrough + persisted autoserve device`.

---

### Task 5: Routes + LLM card UI — CPU/GPU toggle

**Files:**
- Modify: `routes/localmodels_routes.py` (serve)
- Modify: `static/index.html` (radio near `localmodels-status`)
- Modify: `static/js/localModels.js` (device() helper + serve payload + status line)
- Test: `tests/test_localmodels_routes.py`, `tests/test_localmodels_ui.py`

**Interfaces:**
- Consumes: `start(path, device)` from Task 4.
- Produces: `POST /api/localmodels/serve {model_path, device}`; radios `input[name="localmodels-device"]` (cpu default).

- [ ] **Step 1: Failing route test** (append to `tests/test_localmodels_routes.py`; extend `FakeManager.start` to `def start(self, model_path, device="cpu")` recording both):

```python
def test_serve_passes_device(client):
    c, fake, tmp = client
    f = tmp / "m.gguf"; f.write_bytes(b"x")
    r = c.post("/api/localmodels/serve", json={"model_path": str(f), "device": "gpu"})
    assert r.status_code == 200
    assert fake.device == "gpu"
```

- [ ] **Step 2: Failing UI test** (append to `tests/test_localmodels_ui.py`):

```python
def test_localmodels_ui_has_device_toggle():
    html = _read("static/index.html")
    assert 'name="localmodels-device"' in html
    js = _read("static/js/localModels.js")
    assert "localmodels-device" in js
```

- [ ] **Step 3: Run both, expect FAIL.**
- [ ] **Step 4: Implement route.** In `serve`:

```python
        device = (payload.get("device") or "cpu").strip().lower()
        if device not in ("cpu", "gpu"):
            device = "cpu"
        try:
            return await asyncio.to_thread(get_manager().start, safe, device)
```

- [ ] **Step 5: Implement UI.** `static/index.html` — directly under the `id="localmodels-status"` div add:

```html
      <div style="display:flex;gap:12px;align-items:center;margin-bottom:6px;font-size:12px;">
        <label><input type="radio" name="localmodels-device" value="cpu" checked> CPU</label>
        <label><input type="radio" name="localmodels-device" value="gpu"> GPU</label>
      </div>
```

`static/js/localModels.js` — add near the top of the module:

```javascript
  function serveDevice() {
    const el = document.querySelector('input[name="localmodels-device"]:checked');
    return el ? el.value : 'cpu';
  }
```

In the row Serve handler change the POST body to `JSON.stringify({ model_path: m.path, device: serveDevice() })`, and in `refresh()` extend the running status line to `` `Running: ${status.model} on ${status.device === 'gpu' ? 'GPU' : 'CPU'} (port ${status.port})` ``.

- [ ] **Step 6: Run all localmodels tests, expect PASS.** **Step 7: Commit** as `feat(llm): CPU/GPU toggle in Local Models card`.

---

### Task 6: Image size setting (backend + Settings UI)

**Files:**
- Modify: `src/settings.py` (DEFAULT_SETTINGS + `_PER_USER_KEYS`)
- Modify: `src/ai_interaction.py:905` (size default)
- Modify: `static/js/settings.js` (size select in the image section)
- Modify: `static/index.html` (the select element — find the image settings card holding `image-model-select`/`image-quality-select`; add `image-size-select` beside quality)
- Test: `tests/test_image_size_setting.py` (new)

**Interfaces:**
- Produces: setting key `"image_size"` (default `"1024x1024"`, per-user overridable); `do_generate_image` uses it when the content has no explicit line-3 size.

- [ ] **Step 1: Failing tests** (`tests/test_image_size_setting.py`):

```python
"""image_size setting: declared, per-user, and used as the generation default."""
import src.settings as settings


def test_image_size_has_default():
    assert settings.DEFAULT_SETTINGS.get("image_size") == "1024x1024"


def test_image_size_is_per_user():
    assert "image_size" in settings._PER_USER_KEYS


def test_settings_ui_offers_size_select():
    import pathlib
    root = pathlib.Path(settings.__file__).resolve().parents[1]
    html = (root / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="image-size-select"' in html
    js = (root / "static" / "js" / "settings.js").read_text(encoding="utf-8")
    assert "image_size" in js
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement backend.** `src/settings.py`: in DEFAULT_SETTINGS near `"image_gen_enabled": False` add `"image_size": "1024x1024",`; add `"image_size"` to `_PER_USER_KEYS` next to `"image_quality"`. In `src/ai_interaction.py` replace line 905:

```python
    from src.settings import get_user_setting
    _default_size = get_user_setting("image_size", owner or "", "1024x1024") or "1024x1024"
    size = lines[2].strip() if len(lines) > 2 and lines[2].strip() else _default_size
```

- [ ] **Step 4: Implement UI.** In `static/index.html`, inside the image-generation settings card next to the quality select, add:

```html
              <select id="image-size-select" class="admin-select">
                <option value="512x512">512 × 512 (fast)</option>
                <option value="768x768">768 × 768</option>
                <option value="1024x1024" selected>1024 × 1024</option>
              </select>
```

In `static/js/settings.js`, alongside `qualSel` add `const sizeSel = document.getElementById('image-size-select');`, load `if (settings.image_size && sizeSel) sizeSel.value = settings.image_size;`, include `image_size: sizeSel ? sizeSel.value : '1024x1024'` in the saveSettings POST body, and register `if (sizeSel) sizeSel.addEventListener('change', saveSettings);`.

- [ ] **Step 5: Run, expect PASS.** **Step 6: Commit** as `feat(imagegen): image_size setting (per-user) drives default generation size`.

---

### Task 7: Serve-time steps override for image models

**Files:**
- Modify: `src/imagemodels/runtime.py` (`build_serve_argv` steps param)
- Modify: `routes/imagemodels_routes.py` (accept `steps`)
- Modify: `src/imagemodels/manager.py` (`start(files, device, steps=None)` passthrough)
- Modify: `static/index.html` + `static/js/imageModels.js` (steps input)
- Test: `tests/test_imagemodels_runtime.py`, `tests/test_imagemodels_routes.py`, `tests/test_imagemodels_ui.py`

**Interfaces:**
- Produces: `build_serve_argv(..., steps=None)` — explicit steps wins over family defaults; `POST /api/imagemodels/serve {..., steps?}` clamped to 1–50; input `id="imagemodels-steps"`.

- [ ] **Step 1: Failing tests.** Runtime (append):

```python
def test_build_argv_explicit_steps_override():
    files = {"diffusion_model": "/m/flux2-klein.gguf", "llm": "/m/q.gguf",
             "vae": "/m/v.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, steps=8)
    assert argv[argv.index("--steps") + 1] == "8"


def test_build_argv_flux1_explicit_steps():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200, steps=12)
    assert argv[argv.index("--steps") + 1] == "12"
```

Routes (append; `FakeManager.start` becomes `def start(self, files, device="cpu", steps=None)` recording steps):

```python
def test_serve_clamps_and_passes_steps(client):
    c, fake, tmp = client
    f = tmp / "flux-2-klein-4b-Q8_0.gguf"; f.write_bytes(b"x")
    (tmp / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"x")
    (tmp / "flux2_ae.safetensors").write_bytes(b"x")
    r = c.post("/api/imagemodels/serve",
               json={"diffusion_model": str(f), "device": "cpu", "steps": 400})
    assert r.status_code == 200
    assert fake.steps == 50
```

UI (append to `test_imagemodels_ui.py` element list): `'id="imagemodels-steps"'`.

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Runtime — add `steps=None` param; build the family block so an explicit value replaces defaults:

```python
    if "llm" in files:
        argv += ["--llm", files["llm"], "--vae", files["vae"], "--cfg-scale", "1.0"]
        eff = steps or (4 if "klein" in os.path.basename(files["diffusion_model"]).lower() else None)
    else:
        argv += ["--t5xxl", files["t5xxl"], "--clip_l", files["clip_l"],
                 "--vae", files["vae"], "--cfg-scale", "1.0"]
        eff = steps
    if eff:
        argv += ["--steps", str(int(eff))]
```

Manager — `start(self, files: dict, device: str = "cpu", steps=None)`, pass `steps=steps` into `build_serve_argv`. Routes — parse and clamp before calling start:

```python
        steps = payload.get("steps")
        try:
            steps = max(1, min(50, int(steps))) if steps else None
        except (TypeError, ValueError):
            steps = None
        ...
        return await asyncio.to_thread(get_manager().start, files, device, steps)
```

UI — in `static/index.html` next to the imagemodels device radios add `<label>Steps <input id="imagemodels-steps" type="number" min="1" max="50" placeholder="auto" style="width:56px;"></label>`; in `imageModels.js` `serveModel`, read `const st = parseInt($('imagemodels-steps')?.value, 10); const body = { diffusion_model: path, device: device() }; if (st) body.steps = st;` and post `body`.

- [ ] **Step 4: Run all imagemodels tests, expect PASS.** **Step 5: Commit** as `feat(imagemodels): serve-time steps override (1-50, auto default)`.

---

### Task 8: TAESD fast decode (experimental, FLUX.1)

**Files:**
- Modify: `src/imagemodels/encoders.py` (optional `taesd` role)
- Modify: `src/imagemodels/runtime.py` (argv `--taesd`)
- Modify: `routes/imagemodels_routes.py` (`fast_decode` flag)
- Modify: `static/index.html` + `static/js/imageModels.js` (checkbox)
- Test: `tests/test_imagemodels_encoders.py`, `tests/test_imagemodels_runtime.py`, `tests/test_imagemodels_routes.py`, `tests/test_imagemodels_ui.py`

**Interfaces:**
- Produces: `find_taesd(diffusion_model) -> str|None` in encoders (searches `taef1.safetensors` next to the gguf, then the encoders dir; FLUX.1 only — returns None for flux2 names); `build_serve_argv` adds `--taesd <path>` when `files["taesd"]` present; serve payload `fast_decode: true` includes it when found; checkbox `id="imagemodels-fast-decode"`.

- [ ] **Step 1: Failing tests.** Encoders:

```python
def test_find_taesd_in_encoders_dir(tmp_path, monkeypatch):
    img = tmp_path / "img"; (img / "encoders").mkdir(parents=True)
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(img))
    (img / "encoders" / "taef1.safetensors").write_bytes(b"x")
    d = tmp_path / "m"; d.mkdir(); (d / "flux1-dev.gguf").write_bytes(b"x")
    assert enc.find_taesd(str(d / "flux1-dev.gguf")).endswith("taef1.safetensors")


def test_find_taesd_none_for_flux2(tmp_path, monkeypatch):
    img = tmp_path / "img"; (img / "encoders").mkdir(parents=True)
    monkeypatch.setattr(enc, "IMAGE_MODELS_DIR", str(img))
    (img / "encoders" / "taef1.safetensors").write_bytes(b"x")
    d = tmp_path / "m"; d.mkdir(); (d / "flux-2-klein.gguf").write_bytes(b"x")
    assert enc.find_taesd(str(d / "flux-2-klein.gguf")) is None
```

Runtime:

```python
def test_build_argv_taesd_flag():
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors",
             "taesd": "/m/taef1.safetensors"}
    argv = rt.build_serve_argv("/x/sd", files, 8200)
    assert argv[argv.index("--taesd") + 1] == "/m/taef1.safetensors"
```

Routes (FakeManager records files): serve flux1 sibling set + `taef1.safetensors` sibling + `{"fast_decode": True}` → `fake.files.get("taesd")` endswith taef1; without the flag → no `"taesd"` key. UI guard: `'id="imagemodels-fast-decode"'` in index.html, `fast_decode` in imageModels.js.

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Encoders:

```python
TAESD_FILENAMES = ["taef1.safetensors", "taesd_flux.safetensors"]


def find_taesd(diffusion_model):
    """Optional Tiny AutoEncoder for fast (lower-quality) decode. FLUX.1 only —
    no public TAE exists for FLUX.2, and sd.cpp would misdecode with taef1."""
    from src.imagemodels.runtime import looks_like_flux2
    if looks_like_flux2(diffusion_model):
        return None
    diff = os.path.realpath(diffusion_model or "")
    for d in (os.path.dirname(diff), encoders_dir()):
        for fn in TAESD_FILENAMES:
            cand = os.path.join(d, fn)
            if os.path.isfile(cand):
                return os.path.realpath(cand)
    return None
```

Runtime — after the family block: `if files.get("taesd"): argv += ["--taesd", files["taesd"]]`. Routes — after resolving `files`:

```python
        if payload.get("fast_decode"):
            from src.imagemodels.encoders import find_taesd
            tae = find_taesd(real)
            if tae:
                files["taesd"] = tae
```

UI — checkbox `<label><input type="checkbox" id="imagemodels-fast-decode"> Fast decode</label>` beside the steps input; `serveModel` adds `if ($('imagemodels-fast-decode')?.checked) body.fast_decode = true;`.

- [ ] **Step 4: Run all imagemodels tests, expect PASS.**
- [ ] **Step 5: Download taef1** into the shared encoders dir (network):

```powershell
curl.exe -sL --fail -o "$HOME\.assist\data\image-models\encoders\taef1.safetensors" https://huggingface.co/madebyollin/taef1/resolve/main/diffusion_pytorch_model.safetensors
```

Expect ~4.8MB. **Live verification is required before recommending it:** after the next reinstall, serve flux1-dev with Fast decode ON and generate once — if sd.cpp bb84971 rejects the file or output is garbage, record it in this plan and leave the checkbox off by default (it already defaults unchecked; the feature is additive-safe).

- [ ] **Step 6: Commit** as `feat(imagemodels): experimental TAESD fast decode for FLUX.1`.

---

### Task 9: RAM contention guard (both cards)

**Files:**
- Modify: `static/js/localModels.js` (check image status before LLM serve)
- Modify: `static/js/imageModels.js` (check LLM status before image serve)
- Test: `tests/test_localmodels_ui.py`, `tests/test_imagemodels_ui.py`

**Interfaces:**
- Consumes: `GET /api/localmodels/status`, `GET /api/imagemodels/status`, `POST .../stop` (all existing).
- Produces: a confirm-and-stop flow; no backend changes.

- [ ] **Step 1: Failing UI guards.** `test_localmodels_ui.py`: assert `"/api/imagemodels/status"` in localModels.js. `test_imagemodels_ui.py`: assert `"/api/localmodels/status"` in imageModels.js.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** In `imageModels.js` `serveModel`, before POSTing:

```javascript
    try {
      const llm = await api('/api/localmodels/status');
      if (llm.running) {
        const ok = confirm(`${llm.model} (LLM) is running. RAM is tight — stop it before loading the image model?`);
        if (ok) await api('/api/localmodels/stop', { method: 'POST' });
      }
    } catch (e) {}
```

Mirror in `localModels.js`'s row Serve handler (before its POST):

```javascript
        try {
          const img = await api('/api/imagemodels/status');
          if (img.running) {
            const ok = confirm(`${img.model} (image model) is running. RAM is tight — stop it before loading the LLM?`);
            if (ok) await api('/api/imagemodels/stop', { method: 'POST' });
          }
        } catch (e) {}
```

(Proceed with the serve either way — the user may have plenty of RAM; the guard is advisory.)

- [ ] **Step 4: Run both UI test files, expect PASS.** **Step 5: Commit** as `feat(models): RAM contention guard between LLM and image serves`.

---

### Task 10: Rebuild + verify

- [ ] **Step 1:** `python -m pytest tests/test_localmodels_runtime.py tests/test_localmodels_manager.py tests/test_localmodels_routes.py tests/test_localmodels_ui.py tests/test_imagemodels_runtime.py tests/test_imagemodels_encoders.py tests/test_imagemodels_routes.py tests/test_imagemodels_ui.py tests/test_imagemodels_manager.py tests/test_image_size_setting.py tests/test_gguf_meta.py tests/test_brand_strings.py --import-mode=importlib -q` — expect all green.
- [ ] **Step 2:** `.\build-installer.ps1 -Fast` (first -Fast exercise; assets exist so fetches skip, PyInstaller runs incremental). Expect a successful ISCC compile.
- [ ] **Step 3:** Boot-verify the packaged app against an isolated `ODYSSEUS_DATA_DIR` + `ODYSSEUS_INTERNAL_TOKEN`: `GET /api/localmodels/status` returns a `device` key; `GET /api/imagemodels/models` still 200.
- [ ] **Step 4:** Commit installer artifact (`git add -f installer/Output/Assist-Setup.exe`).
- [ ] **Step 5 — user manual test:** serve the Qwen LLM on GPU (expect visibly faster tokens; check `data/logs/llama-server.log` for `offloaded ... layers to GPU`); serve klein Q4 on GPU and generate at 1024; try flux1-dev with Fast decode ON and record the TAESD verdict in this plan.

## Self-Review

- **Spec coverage:** dev fast-loop (T1), GPU LLM (T2–T5), image size (T6), steps control (T7), TAESD (T8), RAM guard (T9), rebuild/verify (T10). The original ask's "quality→steps" is implemented as an explicit serve-time steps control (T7) — quality strings don't map cleanly onto a server-side flag, a number does.
- **Placeholders:** none — every step carries code or an exact command. TAESD's FLUX.1-compat uncertainty is explicitly a verification step with a safe default, not a TODO.
- **Type consistency:** `device="cpu"` threading matches imagemodels conventions; `start(files, device, steps)` order matches route call `(files, device, steps)`; `find_taesd` consumes `looks_like_flux2` (exists since 629108e).

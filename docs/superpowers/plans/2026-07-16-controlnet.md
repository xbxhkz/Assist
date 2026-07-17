# ControlNet for Local Image Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guide local image generation with a control image — serve a diffusion model with a ControlNet (`--control-net`) and generate with a per-request `control_image` + `control_strength`, starting with canny (cv2, free) + bring-your-own control maps, with a LoRA-manager-style ControlNet model registry and UI.

**Architecture:** Mirrors the existing image-gen serve+generate flow. A new `controlnet.py` registry mirrors `loras.py`; `build_serve_argv` gains a `control_net` arg; the `/serve` route reads the active-ControlNet setting and threads it through; a control-generate route reuses the img2img (`style_transfer`) pattern, sending `control_image`/`control_strength`; the UI mirrors the LoRA section. No new bundled deps (cv2 already ships; sd-server already supports `--control-net`).

**Tech Stack:** Python (FastAPI, sd-server subprocess, OpenCV `cv2`), vanilla JS frontend.

## Global Constraints

- **Feasibility gate is Task 1 (live GO/NO-GO)** — verify a real ControlNet serves + generates + fits the 6 GB RTX 4050 before building the app plumbing. Per the image-gen rule, verify a full GENERATION, not just serve readiness. If a FLUX.1 ControlNet OOMs, **SDXL/SD1.5 is the guaranteed fallback**; do not block v1 on FLUX.
- ControlNet **model** is loaded at serve via `--control-net <path>` (one per serve); the control **image** + strength are per-request (`control_image` base64, `control_strength`). Add `--backend controlnet=cpu` on GPU serves to keep the control-net off the 6 GB card.
- `controlnet.py` **mirrors `loras.py` exactly** (same `_safe_stem_file` path-safety rejecting `/ \ ..`, atomic `.part`→replace download); ControlNet models live in `IMAGE_MODELS_DIR/controlnets`.
- v1: **canny** (cv2) + **raw/bring-your-own** control image only; no bundled depth/pose detectors; SDXL/SD1.5/FLUX.1 (FLUX.2 klein has no ControlNet yet).
- No new bundled binaries or Python deps (cv2 already bundled). ControlNet models are user-downloaded, not bundled.
- Every pytest uses `--import-mode=importlib`. Stage specific files (never `git add -A`). Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: ControlNet serve+generate feasibility gate (live GO/NO-GO)

**Files:** none committed (a spike; findings recorded in the task report).

- [ ] **Step 1: Get a small SDXL ControlNet + confirm an SDXL base is present**

Download a small **SDXL canny** ControlNet (e.g. `diffusers/controlnet-canny-sdxl-1.0-small` or a GGUF-quantized SDXL controlnet) into a temp dir. Confirm an SDXL base checkpoint is already downloaded (the user has Juggernaut-XL / RealVisXL); if none, download a small SDXL checkpoint. Note exact paths.

- [ ] **Step 2: Serve sd-server directly with the ControlNet (bypass the app)**

Run the bundled binary directly (Vulkan build) to prove the API + fit:
```
build_assets/sd/vulkan/sd-server.exe -m <sdxl_checkpoint> --control-net <sdxl_canny_cnet> --backend controlnet=cpu --offload-to-cpu --vae-tiling --listen-ip 127.0.0.1 --listen-port 1239
```
Wait for the server to report ready (watch stdout). Note VRAM via `nvidia-smi`.

- [ ] **Step 3: Generate with a per-request control image**

Preprocess a test photo to canny with cv2 (`cv2.Canny`), base64-encode it, and POST:
```
POST http://127.0.0.1:1239/images/generations
{"prompt": "a photo of a house", "control_image": "<b64 canny png>", "control_strength": 0.9, "response_format": "b64_json"}
```
Expected: HTTP 200 with a `b64_json` image whose structure follows the canny edges (save it and eyeball it), and generation stays within 6 GB (no OOM in the sd-server log). Also confirm the server did NOT reject `control_image` as an unknown field.

- [ ] **Step 4: Record the verdict**

Write to the task report: exact commands, VRAM used, whether the control guided the output, and the GO/NO-GO. **GO** (SDXL works) → proceed to Task 2. If SDXL also fails to accept `control_image` per request → **BLOCKED**, escalate (the whole approach needs rethinking). If SDXL works but a later FLUX.1 controlnet OOMs → v1 ships SDXL/SD1.5; note FLUX deferred. (Do not commit anything; this task gates the rest.)

---

### Task 2: `controlnet.py` registry + canny preprocessing

**Files:**
- Create: `src/imagemodels/controlnet.py`
- Modify: `src/imagemodels/civitai.py` (add a `types` param to `search`, default `"LORA"`)
- Test: `tests/test_controlnet_registry.py`

**Interfaces:**
- Produces: `controlnets_dir() -> str`, `_safe_stem_file(name) -> str`, `list_controlnets() -> list[{name,filename,size}]`, `delete_controlnet(name) -> bool`, `download_to_controlnets(url, filename, *, headers=None, http_stream=None) -> dict`, `preprocess_canny(image_bytes, low=100, high=200) -> bytes` (PNG). `civitai.search(query, *, limit=20, token=None, get=None, types="LORA")`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_controlnet_registry.py`:

```python
import io
import numpy as np
import pytest
import src.imagemodels.controlnet as cn


def test_safe_stem_rejects_traversal():
    for bad in ["a/b", "a\\b", "..", "../x"]:
        with pytest.raises(ValueError):
            cn._safe_stem_file(bad)
    assert cn._safe_stem_file("canny-sdxl").endswith(".safetensors")
    assert cn._safe_stem_file("m.safetensors") == "m.safetensors"


def test_download_and_list_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(cn, "controlnets_dir", lambda: str(tmp_path))
    from contextlib import contextmanager
    @contextmanager
    def fake_stream(url, headers):
        yield 4, [b"AB", b"CD"]
    res = cn.download_to_controlnets("http://x/model", "cnet", http_stream=fake_stream)
    assert res["filename"] == "cnet.safetensors" and res["size"] == 4
    names = [c["name"] for c in cn.list_controlnets()]
    assert "cnet" in names
    assert cn.delete_controlnet("cnet") is True
    assert cn.list_controlnets() == []


def test_preprocess_canny_returns_png():
    import cv2
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.rectangle(img, (8, 8), (24, 24), (255, 255, 255), 2)
    ok, buf = cv2.imencode(".png", img)
    out = cn.preprocess_canny(buf.tobytes())
    assert out[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature
    assert cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR) is not None


def test_civitai_search_accepts_types(monkeypatch):
    import src.imagemodels.civitai as civ
    captured = {}
    def fake_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"items": []}
        return R()
    civ.search("edge", types="Controlnet", get=fake_get)
    assert captured["params"]["types"] == "Controlnet"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_controlnet_registry.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.imagemodels.controlnet'`).

- [ ] **Step 3: Create `controlnet.py` (mirror `loras.py`)**

Create `src/imagemodels/controlnet.py`:

```python
"""ControlNet model registry (mirrors loras.py) + canny preprocessing.
sd-server loads a ControlNet via --control-net <path from controlnets_dir()>;
the control image + strength are sent per-request."""
import os
from contextlib import contextmanager

from src.constants import IMAGE_MODELS_DIR


def controlnets_dir() -> str:
    d = os.path.join(IMAGE_MODELS_DIR, "controlnets")
    os.makedirs(d, exist_ok=True)
    return d


def _safe_stem_file(name: str) -> str:
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError("unsafe controlnet name")
    base = os.path.basename(name)
    if not base.lower().endswith(".safetensors"):
        base += ".safetensors"
    return base


def list_controlnets() -> list:
    d = controlnets_dir()
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith(".safetensors"):
            p = os.path.join(d, fn)
            out.append({"name": os.path.splitext(fn)[0], "filename": fn,
                        "size": os.path.getsize(p)})
    return out


def delete_controlnet(name: str) -> bool:
    p = os.path.join(controlnets_dir(), _safe_stem_file(name))
    if os.path.isfile(p):
        os.remove(p)
        return True
    return False


@contextmanager
def _default_http_stream(url, headers):
    import httpx
    with httpx.stream("GET", url, headers=headers or {}, timeout=None,
                      follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        yield total, r.iter_bytes()


def download_to_controlnets(url, filename, *, headers=None, http_stream=None) -> dict:
    http_stream = http_stream or _default_http_stream
    fn = _safe_stem_file(filename)
    dest = os.path.join(controlnets_dir(), fn)
    part = dest + ".part"
    try:
        with http_stream(url, headers) as (_total, chunks):
            with open(part, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass
    return {"name": os.path.splitext(fn)[0], "filename": fn,
            "size": os.path.getsize(dest)}


def preprocess_canny(image_bytes, low=100, high=200) -> bytes:
    """Decode image bytes -> canny edge map -> PNG bytes (3-channel)."""
    import cv2
    import numpy as np
    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("could not decode control image")
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, int(low), int(high))
    edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", edges3)
    if not ok:
        raise ValueError("failed to PNG-encode canny edges")
    return bytes(buf.tobytes())
```

- [ ] **Step 4: Add the `types` param to `civitai.search`**

In `src/imagemodels/civitai.py`, change `def search(query, *, limit=20, token=None, get=None)` to accept `types="LORA"`, and change the params line `"types": "LORA"` to `"types": types`. (Leave all other behavior unchanged so LoRA search still passes `"LORA"` by default.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_controlnet_registry.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add src/imagemodels/controlnet.py src/imagemodels/civitai.py tests/test_controlnet_registry.py
git commit -m "feat(controlnet): model registry (mirror loras) + canny preprocess + civitai types

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Serve with `--control-net` (build_serve_argv + manager threading)

**Files:**
- Modify: `src/imagemodels/runtime.py` (`build_serve_argv` gains `control_net=None`)
- Modify: `src/imagemodels/manager.py` (`start` gains `control_net=None`, passed through)
- Test: `tests/test_controlnet_serve_argv.py`

**Interfaces:**
- Consumes: `build_serve_argv(binary, files, port, device, host, threads, steps, max_vram_gb, control_net=None)`.
- Produces: when `control_net` is a path, argv contains `--control-net <path>` and (GPU) `--backend controlnet=cpu`; otherwise unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_controlnet_serve_argv.py`:

```python
import src.imagemodels.runtime as rt


def _argv(**kw):
    files = {"checkpoint": "/m/sdxl.gguf"}
    return rt.build_serve_argv("sd-server", files, 1234, device="gpu", **kw)


def test_no_controlnet_omits_flag():
    argv = _argv()
    assert "--control-net" not in argv


def test_controlnet_adds_flag_and_cpu_backend_on_gpu():
    argv = _argv(control_net="/cn/canny.safetensors")
    i = argv.index("--control-net")
    assert argv[i + 1] == "/cn/canny.safetensors"
    j = argv.index("--backend")
    assert argv[j + 1] == "controlnet=cpu"


def test_controlnet_cpu_device_no_cpu_backend_needed():
    argv = rt.build_serve_argv("sd-server", {"checkpoint": "/m/sdxl.gguf"}, 1234,
                               device="cpu", control_net="/cn/canny.safetensors")
    assert "--control-net" in argv  # model still loaded on cpu serve
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_controlnet_serve_argv.py --import-mode=importlib -q`
Expected: FAIL (`build_serve_argv() got an unexpected keyword argument 'control_net'`).

- [ ] **Step 3: Add `control_net` to `build_serve_argv`**

In `src/imagemodels/runtime.py`, change the signature to add `control_net=None` (last param):

```python
def build_serve_argv(binary, files, port, device="cpu", host="127.0.0.1",
                     threads=0, steps=None, max_vram_gb=None, control_net=None):
```

Immediately after the `--lora-model-dir` block (the `argv += ["--lora-model-dir", loras_dir(), "--vae-tiling", ...]` at ~line 83-84), add:

```python
    if control_net:
        argv += ["--control-net", control_net]
        if device == "gpu":
            # Keep the ControlNet off the 6GB card (it only guides sampling).
            argv += ["--backend", "controlnet=cpu"]
```

- [ ] **Step 4: Thread `control_net` through `manager.start`**

In `src/imagemodels/manager.py`, change `def start(self, files: dict, device: str = "cpu", steps=None) -> dict:` to add `control_net=None`, and pass it into the `build_serve_argv(...)` call (the one at ~line 166-168) by adding `control_net=control_net` to its kwargs.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_controlnet_serve_argv.py --import-mode=importlib -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/imagemodels/runtime.py src/imagemodels/manager.py tests/test_controlnet_serve_argv.py
git commit -m "feat(controlnet): serve with --control-net (+ controlnet=cpu backend on GPU)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: ControlNet routes (registry CRUD + active-select + control-generate)

**Files:**
- Create: `routes/controlnet_routes.py` (mirror `routes/loras_routes.py`)
- Modify: `app.py` (register `setup_controlnet_routes()` where `setup_loras_routes()` is registered)
- Modify: `routes/imagemodels_routes.py` (`/serve` reads the active-ControlNet setting, resolves its path, passes `control_net` to `manager.start`)
- Modify: `src/settings.py` (add `active_controlnet: ""` default to `DEFAULT_SETTINGS`)
- Test: `tests/test_controlnet_routes.py`

**Interfaces:**
- Consumes: `controlnet.list_controlnets/delete_controlnet/download_to_controlnets/preprocess_canny/controlnets_dir`, `civitai.search(types="Controlnet")`, the served image endpoint lookup used by `style_transfer`.
- Produces routes under prefix `/api/controlnets` (admin-gated like loras): `GET ""` (list), `GET /civitai/search?q=`, `POST /download` ({source: civitai|hf|url}), `DELETE /{name}`, `POST /select` ({name} → sets `active_controlnet`), `POST /control-generate` (multipart: image, prompt, control_type=canny|raw, control_strength, size).

- [ ] **Step 1: Write the failing test**

Create `tests/test_controlnet_routes.py`:

```python
import src.settings as settings


def test_active_controlnet_default():
    assert settings.DEFAULT_SETTINGS.get("active_controlnet") == ""


def test_control_generate_body_builds_control_fields(monkeypatch):
    # Unit-test the request-body assembly helper (extract it in the route module).
    import routes.controlnet_routes as cr
    body = cr._build_control_body(prompt="a house", control_b64="ZWRnZQ==",
                                  control_strength=0.8, size="1024x1024")
    assert body["prompt"] == "a house"
    assert body["control_image"] == "ZWRnZQ=="
    assert body["control_strength"] == 0.8
    assert body["response_format"] == "b64_json"
```

(The route file must expose a pure `_build_control_body(prompt, control_b64, control_strength, size) -> dict` so this is unit-testable without a live server.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_controlnet_routes.py --import-mode=importlib -q`
Expected: FAIL (`active_controlnet` missing / `routes.controlnet_routes` missing).

- [ ] **Step 3: Add the setting**

In `src/settings.py`, add to `DEFAULT_SETTINGS` (beside the other image settings): `"active_controlnet": "",`.

- [ ] **Step 4: Create `routes/controlnet_routes.py`**

Mirror `routes/loras_routes.py` for list/civitai-search/download/delete (using `controlnet.*` and `civitai.search(q, token=token, types="Controlnet")`), admin-gated the same way. Add a pure helper and two extra routes:

```python
def _build_control_body(prompt, control_b64, control_strength, size=None):
    body = {"prompt": prompt, "control_image": control_b64,
            "control_strength": float(control_strength), "response_format": "b64_json"}
    if size:
        body["size"] = size
    return body
```

- `POST /select` `{name}` → `set_setting("active_controlnet", name)` (empty string clears).
- `POST /control-generate` (multipart form, mirroring `style_transfer` at `routes/gallery/gallery_routes.py:430-471`): read `image`, `prompt`, `control_type` (`canny`|`raw`), `control_strength` (default 0.9), `size`. If `control_type == "canny"`, `control_b64 = base64(controlnet.preprocess_canny(image_bytes))`; else `control_b64 = base64(image_bytes)`. Resolve the served image endpoint (same `_first_visible_image_endpoint` helper `style_transfer` uses), `base_url` + `/v1`, POST `_build_control_body(...)` to `{base_url}/images/generations`, return `{image: b64}` on 200 (and, matching the app's convention, the caller/gallery saves it). Clear error if no served endpoint or the model wasn't served with a ControlNet.

- [ ] **Step 5: Thread the active ControlNet into `/serve`**

In `routes/imagemodels_routes.py` `/serve`, before calling `manager.start(...)`, read the active ControlNet and resolve its path:

```python
        from src.settings import get_setting
        from src.imagemodels import controlnet as _cn
        _active = get_setting("active_controlnet", "")
        _cn_path = None
        if _active:
            try:
                _cn_path = os.path.join(_cn.controlnets_dir(), _cn._safe_stem_file(_active))
                if not os.path.isfile(_cn_path):
                    _cn_path = None
            except ValueError:
                _cn_path = None
```

and pass `control_net=_cn_path` to the `manager.start(...)` call.

- [ ] **Step 6: Register the router**

In `app.py`, where `setup_loras_routes()` is imported and included, add `setup_controlnet_routes()` the same way (prefix `/api/controlnets`).

- [ ] **Step 7: Run tests + import smoke**

Run: `python -m pytest tests/test_controlnet_routes.py --import-mode=importlib -q` (Expected: PASS, 2 passed) and `python -c "import routes.controlnet_routes, app"` (no error).

- [ ] **Step 8: Commit**

```bash
git add routes/controlnet_routes.py routes/imagemodels_routes.py app.py src/settings.py tests/test_controlnet_routes.py
git commit -m "feat(controlnet): routes (registry CRUD + select + control-generate) + serve threading

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: ControlNet UI (Image models card section + control-generate panel)

**Files:**
- Create: `static/js/controlnet.js` (mirror `static/js/loras.js`)
- Modify: `static/index.html` (a "ControlNet" section in the Image models card, beside the LoRA section; a control-generate panel; the `<script>` tag)
- Test: none automated (DOM) — live-verified in Task 6.

- [ ] **Step 1: Add the ControlNet management section**

In `static/index.html`, beside the existing LoRA section in the Image models card, add a "ControlNet" section mirroring the LoRA UI: a search box (Civitai), a list of downloaded ControlNets with delete, a download-by-URL/HF field, and a **"Use for next serve"** selector that calls `POST /api/controlnets/select`. Add `<script src="/static/js/controlnet.js"></script>` beside `loras.js`.

- [ ] **Step 2: Create `controlnet.js`**

Create `static/js/controlnet.js` mirroring `static/js/loras.js` (list/search/download/delete against `/api/controlnets*`), plus a `select(name)` calling `POST /api/controlnets/select`. CSP-safe (`createElement`/`textContent`/`addEventListener`, no `innerHTML`-with-data).

- [ ] **Step 3: Add the control-generate panel**

In `static/index.html` (Image area / gallery), add a panel: upload a source image, choose **Canny** (show a cv2 preview by posting the image to control-generate in a preview mode, or just label it) or **Use as-is**, a strength slider (0–1, default 0.9), a prompt field, and a **Generate** button that POSTs multipart to `/api/controlnets/control-generate` and shows the returned image (saveable to the Gallery like other generations). CSP-safe.

- [ ] **Step 4: Self-review**

Grep `controlnet.js` for `innerHTML`/inline `on*=` (none); confirm it hits `/api/controlnets*`; confirm the select posts to `/select`; confirm the generate panel posts multipart with `image`/`prompt`/`control_type`/`control_strength`. Fix anything off.

- [ ] **Step 5: Commit**

```bash
git add static/js/controlnet.js static/index.html
git commit -m "feat(controlnet): Image-models ControlNet section + control-generate panel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Package + live-verify

**Files:**
- Modify: `installer/Output/Assist-Setup.exe` (rebuilt; force-added)

- [ ] **Step 1: Full affected-suite run**

Run: `python -m pytest tests/test_controlnet_registry.py tests/test_controlnet_serve_argv.py tests/test_controlnet_routes.py --import-mode=importlib -q`
Expected: PASS (9 passed).

- [ ] **Step 2: Build the installer**

Run: `powershell -ExecutionPolicy Bypass -File ./build-installer.ps1 -Fast`
Expected: ends with `Installer built: ...\installer\Output\Assist-Setup.exe`.

- [ ] **Step 3: Frozen import check**

Run: `./dist/Assist/Assist.exe --run-py -c "from src.imagemodels.controlnet import preprocess_canny, list_controlnets; from routes.controlnet_routes import _build_control_body; print('OK', _build_control_body('p','b',0.9)['control_strength'])"`
Expected: `OK 0.9` (frozen exe imports the new modules).

- [ ] **Step 4: Live-verify in the running app (manual)**

Reinstall, then as admin: Image models card → ControlNet section → download an SDXL canny ControlNet → **Use for next serve** → serve an SDXL base model → open the control-generate panel → upload a source photo → **Canny** → strength 0.9 → prompt → **Generate** → confirm the output follows the edges and saves to the Gallery. Also confirm: no ControlNet selected → normal generation still works; a mismatched controlnet (SDXL cnet on a FLUX base) surfaces a clear error, not a silent hang.

- [ ] **Step 5: Commit the installer**

```bash
git add -f installer/Output/Assist-Setup.exe
git commit -m "build: Assist-Setup.exe with ControlNet (canny + registry + control-generate)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Task 1 gates everything** — if the live spike shows sd-server won't accept a per-request `control_image` even for SDXL, STOP and escalate.
- ControlNet models are user-downloaded (not bundled); only cv2 (already bundled) is needed for canny.
- The registry/routes/UI are deliberate mirrors of the LoRA manager (`loras.py`/`loras_routes.py`/`loras.js`) — follow those files closely.
- Task 3 keeps `build_serve_argv` behavior unchanged when `control_net is None`.

# LoRA Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download (Civitai search + HF + direct URL + local import), manage, and apply LoRAs to local image generation, using sd-server's native `--lora-model-dir` + `<lora:name:weight>` prompt tag.

**Architecture:** A `loras/` folder under the image-models dir with a registry + atomic streaming downloader (`src/imagemodels/loras.py`); a Civitai search/download client (`src/imagemodels/civitai.py`); `build_serve_argv` points sd-server at the folder; admin-gated routes (`routes/loras_routes.py`) drive a LoRA section in the Image models card. Applying a LoRA is just a `<lora:name:weight>` tag in the image prompt — no change to the generation request.

**Tech Stack:** Python 3, httpx (existing dep), FastAPI, pytest (`--import-mode=importlib`), vanilla CSP-safe JS. No new dependencies.

## Global Constraints

- LoRA only (ControlNet / identity / face-restoration are separate sub-projects).
- LoRAs live in `os.path.join(IMAGE_MODELS_DIR, "loras")`; `name` = filename without extension (that is exactly the `<lora:name:weight>` reference). File type: `.safetensors`.
- `build_serve_argv` appends `--lora-model-dir <loras_dir>` on ALL devices; existing argv tests must still pass (regression).
- All network sources (civitai/hf/url) stream through `loras.download_to_loras` (atomic `.part`→rename); local import streams the uploaded bytes through the same helper.
- Filenames are `os.path.basename`-sanitized; reject any `name` containing `/`, `\`, or `..`.
- New setting `civitai_api_token` (default `""`) in `src/settings.py`.
- Routes are admin-gated: `APIRouter(prefix="/api/loras", dependencies=[Depends(require_admin)])`, matching `routes/imagemodels_routes.py:24-25`.
- Applying LoRA does NOT change the image-generation request — the tag lives in the prompt.
- CSP-safe UI (createElement + addEventListener; dynamic text via `textContent`).
- Run pytest with `--import-mode=importlib`. Stage only the files each task names — never `git add -A`.

---

### Task 1: LoRA storage (registry + atomic downloader) + serve wiring

**Files:**
- Create: `src/imagemodels/loras.py`
- Modify: `src/imagemodels/runtime.py` (the `--vae-tiling` line in `build_serve_argv`, ~line 82)
- Test: `tests/test_loras_registry.py`, and add one case to `tests/test_imagemodels_runtime.py`

**Interfaces:**
- Produces: `loras_dir() -> str`; `list_loras() -> list[dict]` (`{name, filename, size}`); `delete_lora(name) -> bool` (raises `ValueError` on unsafe name); `download_to_loras(url, filename, *, headers=None, http_stream=None) -> dict` (`{name, filename, size}`); module global `IMAGE_MODELS_DIR` (monkeypatchable).

- [ ] **Step 1: Write the failing registry tests**

Create `tests/test_loras_registry.py`:

```python
import os
from contextlib import contextmanager
import pytest
import src.imagemodels.loras as loras


def test_list_only_safetensors_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(loras, "IMAGE_MODELS_DIR", str(tmp_path))
    d = loras.loras_dir()
    open(os.path.join(d, "styleA.safetensors"), "wb").write(b"x")
    open(os.path.join(d, "note.txt"), "w").write("n")
    assert [x["name"] for x in loras.list_loras()] == ["styleA"]
    assert loras.list_loras()[0]["filename"] == "styleA.safetensors"
    assert loras.delete_lora("styleA") is True
    assert loras.list_loras() == []
    assert loras.delete_lora("styleA") is False   # already gone


def test_delete_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(loras, "IMAGE_MODELS_DIR", str(tmp_path))
    for bad in ("../secrets", "a/b", "..\\x"):
        with pytest.raises(ValueError):
            loras.delete_lora(bad)


def test_download_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(loras, "IMAGE_MODELS_DIR", str(tmp_path))
    @contextmanager
    def fake_stream(url, headers):
        yield 3, iter([b"ab", b"c"])
    res = loras.download_to_loras("http://x/f", "myLora", http_stream=fake_stream)
    assert res == {"name": "myLora", "filename": "myLora.safetensors", "size": 3}
    files = os.listdir(loras.loras_dir())
    assert files == ["myLora.safetensors"]         # no leftover .part
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_loras_registry.py --import-mode=importlib -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.imagemodels.loras'`).

- [ ] **Step 3: Implement the registry**

Create `src/imagemodels/loras.py`:

```python
"""LoRA registry + atomic streaming download. sd-server resolves a
<lora:name:weight> prompt tag against the --lora-model-dir this exposes."""
import os
from contextlib import contextmanager

from src.constants import IMAGE_MODELS_DIR


def loras_dir() -> str:
    d = os.path.join(IMAGE_MODELS_DIR, "loras")
    os.makedirs(d, exist_ok=True)
    return d


def _safe_stem_file(name: str) -> str:
    """Return a safe `<stem>.safetensors` basename, or raise ValueError."""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError("unsafe lora name")
    base = os.path.basename(name)
    if not base.lower().endswith(".safetensors"):
        base += ".safetensors"
    return base


def list_loras() -> list:
    d = loras_dir()
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith(".safetensors"):
            p = os.path.join(d, fn)
            out.append({"name": os.path.splitext(fn)[0], "filename": fn,
                        "size": os.path.getsize(p)})
    return out


def delete_lora(name: str) -> bool:
    p = os.path.join(loras_dir(), _safe_stem_file(name))
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


def download_to_loras(url, filename, *, headers=None, http_stream=None) -> dict:
    """Stream `url` into loras/<safe filename> atomically (.part -> rename).
    `http_stream` is an injectable context manager yielding (total, chunks)."""
    http_stream = http_stream or _default_http_stream
    fn = _safe_stem_file(filename)
    dest = os.path.join(loras_dir(), fn)
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
```

- [ ] **Step 4: Run to verify registry tests pass**

Run: `python -m pytest tests/test_loras_registry.py --import-mode=importlib -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Wire `--lora-model-dir` into the serve argv**

In `src/imagemodels/runtime.py`, replace the `--vae-tiling` line (currently
`argv += ["--vae-tiling", "--listen-ip", host, "--listen-port", str(port)]`) with:

```python
    from src.imagemodels.loras import loras_dir  # lazy: avoids import-order cycles
    argv += ["--lora-model-dir", loras_dir(),
             "--vae-tiling", "--listen-ip", host, "--listen-port", str(port)]
```

- [ ] **Step 6: Add the serve-argv regression + new test**

Append to `tests/test_imagemodels_runtime.py`:

```python
def test_build_argv_sets_lora_model_dir(monkeypatch, tmp_path):
    import src.imagemodels.loras as loras
    monkeypatch.setattr(loras, "IMAGE_MODELS_DIR", str(tmp_path))
    files = {"diffusion_model": "/m/flux.gguf", "t5xxl": "/m/t5.gguf",
             "clip_l": "/m/c.safetensors", "vae": "/m/v.safetensors"}
    for dev in ("cpu", "gpu"):
        argv = rt.build_serve_argv("/x/sd", files, 8200, device=dev)
        assert argv[argv.index("--lora-model-dir") + 1] == loras.loras_dir()
        assert "--vae-tiling" in argv     # existing flag still present
```

- [ ] **Step 7: Run the full runtime suite (regression)**

Run: `python -m pytest tests/test_imagemodels_runtime.py tests/test_loras_registry.py --import-mode=importlib -q`
Expected: PASS (all green — the existing `build_serve_argv` tests still pass because `--lora-model-dir` is additive).

- [ ] **Step 8: Commit**

```bash
git add src/imagemodels/loras.py src/imagemodels/runtime.py tests/test_loras_registry.py tests/test_imagemodels_runtime.py
git commit -m "feat(loras): registry + atomic downloader + --lora-model-dir serve wiring"
```

---

### Task 2: Civitai client + setting

**Files:**
- Create: `src/imagemodels/civitai.py`
- Modify: `src/settings.py` (add `civitai_api_token` to the defaults dict)
- Test: `tests/test_civitai.py`

**Interfaces:**
- Produces: `search(query, *, limit=20, token=None, get=None) -> list[dict]` (each: `id, name, base_model, trigger_words, version_id, download_url, file_name, size_kb`); `download_url_with_token(url, token) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_civitai.py`:

```python
import src.imagemodels.civitai as civitai


def test_search_flattens_primary_version_and_file():
    sample = {"items": [{"id": 1, "name": "Anime Style", "modelVersions": [
        {"id": 11, "baseModel": "SDXL 1.0", "trainedWords": ["anmstyle"],
         "files": [{"name": "anime.safetensors", "downloadUrl": "http://c/dl/11",
                    "sizeKB": 1234, "primary": True}]}]}]}
    res = civitai.search("anime", get=lambda u, p, h: sample)
    assert res[0]["name"] == "Anime Style"
    assert res[0]["base_model"] == "SDXL 1.0"
    assert res[0]["trigger_words"] == ["anmstyle"]
    assert res[0]["download_url"] == "http://c/dl/11"
    assert res[0]["file_name"] == "anime.safetensors"
    assert res[0]["size_kb"] == 1234


def test_search_tolerates_missing_fields():
    res = civitai.search("x", get=lambda u, p, h: {"items": [{"id": 2, "name": "Bare"}]})
    assert res[0]["download_url"] == "" and res[0]["trigger_words"] == []
    assert civitai.search("x", get=lambda u, p, h: {}) == []


def test_search_passes_lora_type_and_query():
    seen = {}
    def get(u, p, h):
        seen["params"] = p
        return {"items": []}
    civitai.search("dog", limit=5, get=get)
    assert seen["params"]["types"] == "LORA"
    assert seen["params"]["query"] == "dog" and seen["params"]["limit"] == 5


def test_download_url_with_token():
    assert civitai.download_url_with_token("http://c/dl", "T") == "http://c/dl?token=T"
    assert civitai.download_url_with_token("http://c/dl?x=1", "T") == "http://c/dl?x=1&token=T"
    assert civitai.download_url_with_token("http://c/dl", "") == "http://c/dl"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_civitai.py --import-mode=importlib -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.imagemodels.civitai'`).

- [ ] **Step 3: Implement**

Create `src/imagemodels/civitai.py`:

```python
"""Civitai LoRA search + download-URL/token helpers. `get` is injectable so
search is unit-testable without network."""
_API = "https://civitai.com/api/v1/models"


def download_url_with_token(url: str, token: str) -> str:
    if not token:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}token={token}"


def _flatten(item: dict) -> dict:
    versions = item.get("modelVersions") or []
    v = versions[0] if versions else {}
    files = v.get("files") or []
    f = next((x for x in files if x.get("primary")), files[0] if files else {})
    return {
        "id": item.get("id"),
        "name": item.get("name") or "",
        "base_model": v.get("baseModel") or "",
        "trigger_words": v.get("trainedWords") or [],
        "version_id": v.get("id"),
        "download_url": f.get("downloadUrl") or v.get("downloadUrl") or "",
        "file_name": f.get("name") or "",
        "size_kb": f.get("sizeKB") or 0,
    }


def _default_get(url, params, headers):
    import httpx
    r = httpx.get(url, params=params, headers=headers, timeout=30,
                  follow_redirects=True)
    r.raise_for_status()
    return r.json()


def search(query, *, limit=20, token=None, get=None) -> list:
    get = get or _default_get
    params = {"types": "LORA", "query": query or "", "limit": max(1, min(int(limit), 100))}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = get(_API, params, headers) or {}
    return [_flatten(it) for it in (data.get("items") or [])]
```

- [ ] **Step 4: Add the setting**

In `src/settings.py`, add to the defaults dict near the other image settings
(after `"image_model": "",` — the `image_*` group):

```python
    "civitai_api_token": "",
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_civitai.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add src/imagemodels/civitai.py src/settings.py tests/test_civitai.py
git commit -m "feat(loras): Civitai LoRA search client + civitai_api_token setting"
```

---

### Task 3: Routes

**Files:**
- Create: `routes/loras_routes.py`
- Modify: `app.py` (include the router next to the imagemodels include at `app.py:766-767`)
- Test: `tests/test_loras_routes.py`

**Interfaces:**
- Consumes: `loras.list_loras/delete_lora/download_to_loras` (Task 1); `civitai.search/download_url_with_token` (Task 2); `get_setting` (`src/settings.py`).
- Produces: `setup_loras_routes() -> APIRouter` with `GET /api/loras`, `GET /api/loras/civitai/search`, `POST /api/loras/download`, `POST /api/loras/upload`, `DELETE /api/loras/{name}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_loras_routes.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
from core.middleware import require_admin
import routes.loras_routes as lr


def make_client(monkeypatch):
    monkeypatch.setattr(lr.loras, "list_loras", lambda: [{"name": "a", "filename": "a.safetensors", "size": 9}])
    monkeypatch.setattr(lr.loras, "delete_lora", lambda n: n == "a")
    monkeypatch.setattr(lr.loras, "download_to_loras",
                        lambda url, fn, **k: {"name": "x", "filename": "x.safetensors", "size": 5})
    monkeypatch.setattr(lr.civitai, "search", lambda q, token=None: [{"name": "L", "download_url": "u"}])
    monkeypatch.setattr(lr, "get_setting", lambda k, d=None: "")
    app = FastAPI()
    app.include_router(lr.setup_loras_routes())
    app.dependency_overrides[require_admin] = lambda: None
    return TestClient(app)


def test_list(monkeypatch):
    r = make_client(monkeypatch).get("/api/loras")
    assert r.status_code == 200 and r.json()["loras"][0]["name"] == "a"


def test_civitai_search(monkeypatch):
    r = make_client(monkeypatch).get("/api/loras/civitai/search", params={"q": "anime"})
    assert r.status_code == 200 and r.json()["results"][0]["name"] == "L"


def test_download_civitai(monkeypatch):
    r = make_client(monkeypatch).post("/api/loras/download",
        json={"source": "civitai", "download_url": "http://c/dl", "file_name": "x.safetensors"})
    assert r.status_code == 200 and r.json()["lora"]["name"] == "x"


def test_download_bad_source(monkeypatch):
    r = make_client(monkeypatch).post("/api/loras/download", json={"source": "nope"})
    assert r.status_code == 400


def test_delete_ok_and_404(monkeypatch):
    c = make_client(monkeypatch)
    assert c.delete("/api/loras/a").status_code == 200
    assert c.delete("/api/loras/missing").status_code == 404


def test_delete_invalid_name(monkeypatch):
    # delete_lora raises ValueError for unsafe names → route maps to 400.
    def boom(n):
        raise ValueError("unsafe")
    monkeypatch.setattr(lr.loras, "delete_lora", boom)
    assert make_client(monkeypatch).delete("/api/loras/badname").status_code == 400
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_loras_routes.py --import-mode=importlib -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'routes.loras_routes'`).

- [ ] **Step 3: Implement the routes**

Create `routes/loras_routes.py`:

```python
"""Admin-gated LoRA management: list, Civitai search, download (civitai/hf/url),
upload, delete. Downloads run off the event loop (asyncio.to_thread)."""
import asyncio
from contextlib import contextmanager

from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile

from core.middleware import require_admin
from src.settings import get_setting
from src.imagemodels import loras, civitai


def setup_loras_routes() -> APIRouter:
    router = APIRouter(prefix="/api/loras", dependencies=[Depends(require_admin)])

    @router.get("")
    async def list_loras():
        return {"loras": loras.list_loras()}

    @router.get("/civitai/search")
    async def civitai_search(q: str = ""):
        token = get_setting("civitai_api_token", "") or None
        try:
            results = await asyncio.to_thread(civitai.search, q, token=token)
        except Exception as e:
            raise HTTPException(502, f"Civitai search failed: {e}")
        return {"results": results}

    @router.post("/download")
    async def download(body: dict = Body(...)):
        source = (body.get("source") or "").strip()
        token = get_setting("civitai_api_token", "") or None
        try:
            if source == "civitai":
                url = civitai.download_url_with_token(body.get("download_url", ""), token)
                fn = body.get("file_name") or "lora.safetensors"
                res = await asyncio.to_thread(loras.download_to_loras, url, fn)
            elif source == "hf":
                repo = (body.get("repo") or "").strip("/")
                filename = body.get("filename") or ""
                url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
                res = await asyncio.to_thread(loras.download_to_loras, url, filename)
            elif source == "url":
                res = await asyncio.to_thread(
                    loras.download_to_loras, body.get("url", ""), body.get("name") or "lora")
            else:
                raise HTTPException(400, "source must be civitai|hf|url")
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(502, f"LoRA download failed: {e}")
        return {"ok": True, "lora": res}

    @router.post("/upload")
    async def upload(file: UploadFile = File(...)):
        data = await file.read()

        @contextmanager
        def _mem(url, headers):
            yield len(data), iter([data])

        try:
            res = await asyncio.to_thread(
                loras.download_to_loras, "", file.filename or "lora", http_stream=_mem)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "lora": res}

    @router.delete("/{name}")
    async def delete(name: str):
        try:
            ok = loras.delete_lora(name)
        except ValueError:
            raise HTTPException(400, "invalid name")
        if not ok:
            raise HTTPException(404, "not found")
        return {"ok": True}

    return router
```

- [ ] **Step 4: Wire into app.py**

In `app.py`, next to the imagemodels include (lines 766-767):

```python
from routes.loras_routes import setup_loras_routes
app.include_router(setup_loras_routes())
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_loras_routes.py --import-mode=importlib -q`
Expected: PASS (6 passed).

- [ ] **Step 6: Commit**

```bash
git add routes/loras_routes.py app.py tests/test_loras_routes.py
git commit -m "feat(loras): admin-gated LoRA routes (list/search/download/upload/delete)"
```

---

### Task 4: UI — LoRA section in the Image models card

**Files:**
- Create: `static/js/loras.js`
- Modify: `static/index.html` (add a LoRA section inside the Image models card at ~line 1443-1456; add the `<script>` tag near the other image scripts)
- Test: manual/live-verify (Task 5); no unit-test framework for these DOM files.

**Interfaces:**
- Consumes: `GET /api/loras`, `GET /api/loras/civitai/search?q=`, `POST /api/loras/download`, `POST /api/loras/upload`, `DELETE /api/loras/{name}` (Task 3).

- [ ] **Step 1: Add the section markup**

In `static/index.html`, inside the Image models card (after the serve controls block ending ~line 1456's `<button id="imagemodels-serve-btn">`), add:

```html
      <div id="loras-section" style="margin-top:10px;border-top:1px solid var(--border,#333);padding-top:8px;">
        <div style="font-size:13px;font-weight:600;margin-bottom:4px;">LoRAs</div>
        <div style="display:flex;gap:4px;margin-bottom:6px;">
          <input id="lora-civitai-q" type="text" placeholder="Search Civitai LoRAs…" style="flex:1;min-width:0;">
          <button id="lora-civitai-search-btn" type="button">Search</button>
        </div>
        <div id="lora-search-results" style="max-height:180px;overflow:auto;margin-bottom:6px;"></div>
        <details style="margin-bottom:6px;">
          <summary style="cursor:pointer;font-size:12px;opacity:0.8;">Add from HF / URL / file</summary>
          <div style="display:flex;flex-direction:column;gap:4px;margin-top:4px;">
            <div style="display:flex;gap:4px;"><input id="lora-hf-repo" placeholder="hf repo (owner/name)" style="flex:1;min-width:0;"><input id="lora-hf-file" placeholder="file.safetensors" style="flex:1;min-width:0;"><button id="lora-hf-btn" type="button">Get</button></div>
            <div style="display:flex;gap:4px;"><input id="lora-url" placeholder="direct .safetensors URL" style="flex:1;min-width:0;"><input id="lora-url-name" placeholder="name" style="width:90px;"><button id="lora-url-btn" type="button">Get</button></div>
            <div style="display:flex;gap:4px;align-items:center;"><input id="lora-file" type="file" accept=".safetensors" style="flex:1;min-width:0;"><button id="lora-file-btn" type="button">Import</button></div>
          </div>
        </details>
        <div id="lora-installed"></div>
        <div id="lora-msg" style="font-size:12px;opacity:0.8;margin-top:4px;"></div>
      </div>
```

Add near the other image-model script tag in `static/index.html`:

```html
<script src="/static/js/loras.js"></script>
```

- [ ] **Step 2: Implement loras.js**

Create `static/js/loras.js` — a CSP-safe IIFE (createElement + addEventListener, dynamic text via `textContent`):

```javascript
// LoRA manager UI (Image models card). Search Civitai, download from HF/URL/file,
// list/delete installed LoRAs, copy the <lora:name:weight> tag to paste into a prompt.
(function () {
  function $(id) { return document.getElementById(id); }
  function msg(t, err) { const m = $('lora-msg'); if (m) { m.textContent = t || ''; m.style.color = err ? 'var(--red,#ff5555)' : ''; } }

  async function api(path, opts) {
    const res = await fetch(path, Object.assign({ credentials: 'same-origin' }, opts || {}));
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error((data && data.detail) || String(res.status));
    return data;
  }

  function btn(label, fn) { const b = document.createElement('button'); b.type = 'button'; b.textContent = label; b.addEventListener('click', fn); return b; }

  async function refreshInstalled() {
    const host = $('lora-installed'); if (!host) return;
    let loras = [];
    try { loras = (await api('/api/loras')).loras || []; } catch (e) { return; }
    host.innerHTML = '';
    if (!loras.length) { const e = document.createElement('div'); e.style.cssText = 'font-size:12px;opacity:0.6;'; e.textContent = 'No LoRAs installed.'; host.appendChild(e); return; }
    loras.forEach((l) => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;gap:6px;align-items:center;font-size:12px;padding:2px 0;';
      const nm = document.createElement('span'); nm.style.cssText = 'flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'; nm.textContent = l.name;
      const tag = '<lora:' + l.name + ':0.8>';
      row.appendChild(nm);
      row.appendChild(btn('Copy tag', () => { navigator.clipboard && navigator.clipboard.writeText(tag); msg('Copied ' + tag); }));
      row.appendChild(btn('Delete', async () => { try { await api('/api/loras/' + encodeURIComponent(l.name), { method: 'DELETE' }); msg('Deleted ' + l.name); refreshInstalled(); } catch (e) { msg('Delete failed: ' + e.message, true); } }));
      host.appendChild(row);
    });
  }

  async function download(body) {
    msg('Downloading…');
    try { const d = await api('/api/loras/download', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); msg('Installed ' + d.lora.name); refreshInstalled(); }
    catch (e) { msg('Download failed: ' + e.message, true); }
  }

  async function civitaiSearch() {
    const q = ($('lora-civitai-q') || {}).value || '';
    const host = $('lora-search-results'); if (!host) return;
    msg('Searching Civitai…'); host.innerHTML = '';
    let results = [];
    try { results = (await api('/api/loras/civitai/search?q=' + encodeURIComponent(q))).results || []; msg(''); }
    catch (e) { msg('Search failed: ' + e.message, true); return; }
    results.forEach((r) => {
      const row = document.createElement('div'); row.style.cssText = 'display:flex;gap:6px;align-items:center;font-size:12px;padding:3px 0;border-bottom:1px solid var(--border,#333);';
      const info = document.createElement('div'); info.style.cssText = 'flex:1;min-width:0;';
      const nm = document.createElement('div'); nm.style.cssText = 'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'; nm.textContent = r.name + (r.base_model ? '  ·  ' + r.base_model : '');
      info.appendChild(nm);
      if (r.trigger_words && r.trigger_words.length) { const tw = document.createElement('div'); tw.style.cssText = 'opacity:0.6;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'; tw.textContent = 'triggers: ' + r.trigger_words.join(', '); info.appendChild(tw); }
      row.appendChild(info);
      const dl = btn('Download', () => download({ source: 'civitai', download_url: r.download_url, file_name: r.file_name }));
      if (!r.download_url) dl.disabled = true;
      row.appendChild(dl);
      host.appendChild(row);
    });
    if (!results.length) { const e = document.createElement('div'); e.style.cssText = 'font-size:12px;opacity:0.6;'; e.textContent = 'No results.'; host.appendChild(e); }
  }

  async function uploadFile() {
    const inp = $('lora-file'); if (!inp || !inp.files || !inp.files[0]) { msg('Choose a .safetensors file first', true); return; }
    const fd = new FormData(); fd.append('file', inp.files[0]);
    msg('Importing…');
    try { const d = await api('/api/loras/upload', { method: 'POST', body: fd }); msg('Imported ' + d.lora.name); refreshInstalled(); }
    catch (e) { msg('Import failed: ' + e.message, true); }
  }

  document.addEventListener('DOMContentLoaded', () => {
    $('lora-civitai-search-btn') && $('lora-civitai-search-btn').addEventListener('click', civitaiSearch);
    $('lora-hf-btn') && $('lora-hf-btn').addEventListener('click', () => download({ source: 'hf', repo: ($('lora-hf-repo') || {}).value || '', filename: ($('lora-hf-file') || {}).value || '' }));
    $('lora-url-btn') && $('lora-url-btn').addEventListener('click', () => download({ source: 'url', url: ($('lora-url') || {}).value || '', name: ($('lora-url-name') || {}).value || 'lora' }));
    $('lora-file-btn') && $('lora-file-btn').addEventListener('click', uploadFile);
    refreshInstalled();
  });
})();
```

- [ ] **Step 3: Commit**

```bash
git add static/js/loras.js static/index.html
git commit -m "feat(loras): LoRA manager UI in the Image models card (CSP-safe)"
```

---

### Task 5: Package + live-verify

**Files:**
- Modify: `installer/Output/Assist-Setup.exe` (rebuilt; force-added like prior build commits)

**Interfaces:**
- Consumes: Tasks 1-4.

- [ ] **Step 1: Full affected-suite run**

Run: `python -m pytest tests/test_loras_registry.py tests/test_civitai.py tests/test_loras_routes.py tests/test_imagemodels_runtime.py tests/test_imagemodels_manager.py --import-mode=importlib -q`
Expected: PASS (all green).

- [ ] **Step 2: Build the installer**

Run: `powershell -ExecutionPolicy Bypass -File .\build-installer.ps1 -Fast`
Expected: ends with `Installer built: ...\installer\Output\Assist-Setup.exe`.

- [ ] **Step 3: Frozen import + serve-argv check**

Run: `./dist/Assist/Assist.exe --run-py -c "from src.imagemodels.loras import loras_dir, list_loras; from src.imagemodels.civitai import search; from src.imagemodels.runtime import build_serve_argv; a=build_serve_argv('sd',{'diffusion_model':'/m/f','t5xxl':'/m/t','clip_l':'/m/c','vae':'/m/v'},8200,device='gpu'); print('LORA_DIR_IN_ARGV', '--lora-model-dir' in a); print('DIR', loras_dir())"`
Expected: `LORA_DIR_IN_ARGV True` and a valid loras dir path.

- [ ] **Step 4: Live-verify in the running app (manual)**

Reinstall; then in the app's Image models card:
- Search Civitai for a small **SDXL** LoRA (e.g. a style LoRA) → results show name + trigger words → **Download** → it appears under installed.
- Serve an SDXL model (juggernaut-xl or RealVisXL) on GPU.
- In chat, generate `a portrait of a woman <lora:<name>:0.9>` (+ any trigger word) → confirm the LoRA visibly changes the output.
- Generate the same prompt WITHOUT the `<lora:…>` tag → confirm normal output (no regression).
- Delete the LoRA from the card → it disappears from the list.
- (Optional) Import a local `.safetensors` and paste a direct URL to confirm those sources.

- [ ] **Step 5: Commit the installer**

```bash
git add -f installer/Output/Assist-Setup.exe
git commit -m "build: Assist-Setup.exe with LoRA manager"
```

---

## Notes for the executor

- Every pytest run uses `--import-mode=importlib`.
- Tasks 1-3 are pure Python + injected fakes/TestClient (no network, no real sd-server). Task 4 is UI (verified live in Task 5).
- The safety-relevant property: `delete_lora`/`_safe_stem_file` must reject `/`, `\`, `..` — `tests/test_loras_registry.py::test_delete_rejects_traversal` is the guard.
- Do NOT change the image-generation request or `do_generate_image` — LoRA is applied purely via the `<lora:name:weight>` tag in the prompt plus the serve flag.

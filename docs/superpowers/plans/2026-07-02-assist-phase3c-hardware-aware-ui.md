# Assist Phase 3c — Hardware-Aware Local Models UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Local Models modal hardware-aware — detect the machine, badge each downloadable GGUF by fit, recommend models for the machine, and let the user delete downloaded models + see disk usage.

**Architecture:** Add `src/localmodels/hardware.py` (cached `hwfit.detect_system` wrapper + a hybrid size/param fit computation + `hwfit.rank_models` wrapper), a `delete_model` method on the existing manager, new/annotated routes, and modal UI. hwfit is used **read-only**.

**Tech Stack:** Python 3.14, FastAPI, `services/hwfit` (read-only), vanilla JS, pytest.

## Global Constraints

- **hwfit + Cookbook read-only:** do NOT modify `services/hwfit/`, `routes/cookbook_*`, or `src/cookbook_*`. Import hwfit functions; never edit them.
- **Hybrid fit, conservative max:** fit combines size-based (`file_size + KV overhead`) with hwfit `estimate_memory_gb` (when a `NB` param token is parseable from the filename); `needed_gb = max(size_needed, param_needed or 0)`. The param estimate may only make a verdict *more* conservative.
- **Safe-default on detection failure:** `get_hardware` returns `{ram_gb:0, has_gpu:false, gpu_name:None, vram_gb:0}` if detection raises; the modal never blocks.
- **Delete safety:** only a safe `.gguf` basename inside `MODELS_DIR`; if it's the currently-serving model, stop it first.
- **No internal module / `ODYSSEUS_*` renames.**
- **Test env:** pytest with `--import-mode=importlib`; new tests carry no `slow` marker.

## File Structure

- `src/localmodels/hardware.py` (new) — detection, fit, recommendations. Task 1.
- `src/localmodels/manager.py` (extend) — `delete_model`. Task 2.
- `routes/localmodels_routes.py` (extend) — `/delete`, `/models` disk_bytes (Task 2); `/hardware`, `/recommendations`, fit-annotated `/catalog/files` (Task 3).
- `static/js/localModels.js` + `static/index.html` (extend) — UI. Task 4.
- Tests: `tests/test_localmodels_hardware.py`, `tests/test_localmodels_delete.py`, `tests/test_localmodels_hardware_routes.py`, `tests/test_localmodels_hardware_ui.py`.

---

### Task 1: hardware.py — detection, hybrid fit, recommendations

**Files:**
- Create: `src/localmodels/hardware.py`
- Test: `tests/test_localmodels_hardware.py`

**Interfaces:**
- Consumes: `services.hwfit.hardware.detect_system`, `services.hwfit.fit.rank_models`, `services.hwfit.models.estimate_memory_gb` / `infer_quantization_from_name` (all read-only, injectable in fns).
- Produces:
  - `get_hardware_system(detect=None, refresh=False) -> dict` (raw hwfit system, cached)
  - `get_hardware(detect=None, refresh=False) -> {"ram_gb","has_gpu","gpu_name","vram_gb"}`
  - `_infer_params_b(name) -> float | None`
  - `_verdict(needed_gb, hardware) -> "gpu"|"ram"|"too_big"`
  - `fit_for_file(file, hardware, ctx=4096, estimate=None) -> {"verdict","needed_gb","size_gb","param_estimate_gb"}`
  - `recommend_models(limit=8, rank=None, detect=None) -> list[{"name","score"}]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_localmodels_hardware.py`:

```python
"""Unit tests for hardware detection + hybrid model-fit (hwfit injected)."""
import pytest

import src.localmodels.hardware as hw


@pytest.fixture(autouse=True)
def _reset_cache():
    hw._hw_cache = None
    yield
    hw._hw_cache = None


def _fake_detect():
    return {"available_ram_gb": 16.0, "has_gpu": True,
            "gpu_name": "RTX 3060", "gpu_vram_gb": 8.0}


def test_get_hardware_normalizes():
    got = hw.get_hardware(detect=_fake_detect, refresh=True)
    assert got == {"ram_gb": 16.0, "has_gpu": True,
                   "gpu_name": "RTX 3060", "vram_gb": 8.0}


def test_get_hardware_caches_detection():
    calls = {"n": 0}
    def det():
        calls["n"] += 1
        return _fake_detect()
    hw.get_hardware(detect=det, refresh=True)
    hw.get_hardware(detect=det)   # cached — no second detect call
    assert calls["n"] == 1


def test_get_hardware_safe_default_on_error():
    def boom():
        raise RuntimeError("no hw")
    got = hw.get_hardware(detect=boom, refresh=True)
    assert got == {"ram_gb": 0.0, "has_gpu": False, "gpu_name": None, "vram_gb": 0.0}


def test_infer_params_b():
    assert hw._infer_params_b("Qwen2.5-7B-Instruct-Q4_K_M.gguf") == 7.0
    assert hw._infer_params_b("tiny-1.5b-chat.gguf") == 1.5
    assert hw._infer_params_b("model-Q4.gguf") is None


def test_verdict_thresholds():
    gpu_hw = {"has_gpu": True, "vram_gb": 8.0, "ram_gb": 16.0}
    assert hw._verdict(6.0, gpu_hw) == "gpu"
    assert hw._verdict(12.0, gpu_hw) == "ram"      # exceeds vram, fits ram
    assert hw._verdict(20.0, gpu_hw) == "too_big"
    cpu_hw = {"has_gpu": False, "vram_gb": 0.0, "ram_gb": 16.0}
    assert hw._verdict(6.0, cpu_hw) == "ram"       # no gpu → ram path


def test_fit_takes_conservative_max_param_over_size():
    # size ~5GB; injected param estimate 9GB → needed=9 → too big for 8GB vram, fits 16 ram.
    hardware = {"has_gpu": True, "vram_gb": 8.0, "ram_gb": 16.0}
    f = {"filename": "big-7B-Q8.gguf", "size": 5_000_000_000}
    out = hw.fit_for_file(f, hardware, estimate=lambda m, q, c: 9.0)
    assert out["param_estimate_gb"] == 9.0
    assert out["needed_gb"] == 9.0
    assert out["verdict"] == "ram"


def test_fit_uses_size_when_param_larger_is_absent():
    # No param token → size-only; 5GB + kv(4096)=0.5 → 5.5 → fits 8GB gpu.
    hardware = {"has_gpu": True, "vram_gb": 8.0, "ram_gb": 16.0}
    f = {"filename": "model-Q4.gguf", "size": 5_000_000_000}
    out = hw.fit_for_file(f, hardware)
    assert out["param_estimate_gb"] is None
    assert out["needed_gb"] == 5.5
    assert out["verdict"] == "gpu"


def test_recommend_models_injected_rank():
    def fake_rank(system, limit=8):
        return [{"name": "Qwen2.5-7B", "score": 0.9}, {"name": "Phi-3.5", "score": 0.8},
                {"score": 0.1}]  # no name → skipped
    out = hw.recommend_models(limit=8, rank=fake_rank, detect=_fake_detect)
    assert out == [{"name": "Qwen2.5-7B", "score": 0.9}, {"name": "Phi-3.5", "score": 0.8}]


def test_recommend_models_empty_on_error():
    def boom(system, limit=8):
        raise RuntimeError("rank down")
    assert hw.recommend_models(rank=boom, detect=_fake_detect) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_localmodels_hardware.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.localmodels.hardware'`.

- [ ] **Step 3: Implement the module**

Create `src/localmodels/hardware.py`:

```python
"""Hardware detection + model-fit analysis for native local models (Phase 3c).

Uses services/hwfit READ-ONLY. Detection is cached (it is slow); fit is a
conservative hybrid of a size-based estimate and hwfit's param-based
estimate_memory_gb, taking the max so the app never over-promises a fit.
"""
import re
import threading

_PARAM_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![\w])")

_hw_cache = None
_hw_lock = threading.Lock()


def _default_detect():
    from services.hwfit.hardware import detect_system
    return detect_system()


def get_hardware_system(detect=None, refresh=False) -> dict:
    """Raw hwfit system dict, cached process-wide. Safe-defaults to {} on error."""
    global _hw_cache
    with _hw_lock:
        if _hw_cache is None or refresh:
            det = detect or _default_detect
            try:
                _hw_cache = det() or {}
            except Exception:
                _hw_cache = {}
        return _hw_cache


def get_hardware(detect=None, refresh=False) -> dict:
    """Normalized hardware summary for the UI."""
    s = get_hardware_system(detect=detect, refresh=refresh) or {}
    return {
        "ram_gb": float(s.get("available_ram_gb") or 0.0),
        "has_gpu": bool(s.get("has_gpu")),
        "gpu_name": s.get("gpu_name"),
        "vram_gb": float(s.get("gpu_vram_gb") or 0.0),
    }


def _infer_params_b(name):
    m = _PARAM_RE.search(name or "")
    return float(m.group(1)) if m else None


def _kv_overhead(ctx):
    # Rough runtime/KV headroom on top of the weights, scaled by context length.
    return 0.5 * (ctx / 4096.0)


def _verdict(needed_gb, hardware):
    if hardware.get("has_gpu") and needed_gb <= (hardware.get("vram_gb") or 0):
        return "gpu"
    if needed_gb <= (hardware.get("ram_gb") or 0):
        return "ram"
    return "too_big"


def fit_for_file(file, hardware, ctx=4096, estimate=None):
    """Hybrid fit verdict for a downloadable GGUF file (size + hwfit estimate)."""
    size_gb = float(file.get("size") or 0) / 1e9
    size_needed = size_gb + _kv_overhead(ctx)
    param_needed = None
    params = _infer_params_b(file.get("filename") or "")
    if params:
        try:
            from services.hwfit.models import (
                infer_quantization_from_name, estimate_memory_gb,
            )
            est = estimate or estimate_memory_gb
            quant = infer_quantization_from_name(file.get("filename") or "")
            param_needed = float(est({"parameter_count": f"{params}B"}, quant, ctx))
        except Exception:
            param_needed = None
    needed_gb = max(size_needed, param_needed or 0.0)
    return {
        "verdict": _verdict(needed_gb, hardware),
        "needed_gb": round(needed_gb, 2),
        "size_gb": round(size_gb, 2),
        "param_estimate_gb": round(param_needed, 2) if param_needed is not None else None,
    }


def recommend_models(limit=8, rank=None, detect=None):
    """hwfit-ranked model families for the detected machine."""
    system = get_hardware_system(detect=detect)
    try:
        from services.hwfit.fit import rank_models
        r = rank or rank_models
        out = r(system, limit=limit) or []
    except Exception:
        return []
    return [{"name": m.get("name"), "score": m.get("score")}
            for m in out if m.get("name")]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_localmodels_hardware.py -v --import-mode=importlib`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/localmodels/hardware.py tests/test_localmodels_hardware.py
git commit -m "feat(localmodels): hardware detection + hybrid model-fit + recommendations"
```

---

### Task 2: Delete model + disk usage

**Files:**
- Modify: `src/localmodels/manager.py` (add `delete_model`)
- Modify: `routes/localmodels_routes.py` (add `/delete`; `/models` returns `disk_bytes`)
- Test: `tests/test_localmodels_delete.py`

**Interfaces:**
- Consumes: `LocalModelManager` (Task 3a), `src.constants.MODELS_DIR`.
- Produces: `manager.delete_model(filename) -> dict` (status shape); `POST /api/localmodels/delete {filename}`; `/models` → `{"models": [...], "disk_bytes": int}`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_localmodels_delete.py`:

```python
"""delete_model + disk-usage tests."""
import pytest

import src.localmodels.manager as mgr_mod
from src.localmodels.manager import LocalModelManager


class FakeProc:
    def __init__(self, pid=1):
        self.pid = pid
    def terminate(self): pass
    def wait(self, timeout=None): pass
    def kill(self): pass


def _serving_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(mgr_mod, "MODELS_DIR", str(tmp_path))
    unreg = []
    mgr = LocalModelManager(
        spawn=lambda fn: None,           # we set state manually below
        port_chooser=lambda: 8123,
        readiness=lambda url: True,
        register_endpoint=lambda name, base_url: "local-0",
        unregister_endpoint=lambda eid: unreg.append(eid),
        resolve_binary=lambda: "/bin/llama-server",
    )
    return mgr, unreg


def test_delete_rejects_unsafe(tmp_path, monkeypatch):
    mgr, _ = _serving_manager(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        mgr.delete_model("../evil.gguf")
    with pytest.raises(ValueError):
        mgr.delete_model("a/b.gguf")
    with pytest.raises(ValueError):
        mgr.delete_model("note.txt")


def test_delete_removes_file(tmp_path, monkeypatch):
    mgr, _ = _serving_manager(tmp_path, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"xxxx")
    st = mgr.delete_model("m.gguf")
    assert not (tmp_path / "m.gguf").exists()
    assert st["running"] is False


def test_delete_stops_serving_model_first(tmp_path, monkeypatch):
    mgr, unreg = _serving_manager(tmp_path, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"xxxx")
    # Simulate m.gguf currently serving.
    mgr._proc = FakeProc()
    mgr._state = {"model_path": str(tmp_path / "m.gguf"), "port": 8123,
                  "endpoint_id": "local-0", "pid": 1}
    mgr.delete_model("m.gguf")
    assert unreg == ["local-0"]                 # endpoint torn down
    assert not (tmp_path / "m.gguf").exists()   # file removed
    assert mgr.status()["running"] is False


def test_delete_missing_file_is_ok(tmp_path, monkeypatch):
    mgr, _ = _serving_manager(tmp_path, monkeypatch)
    st = mgr.delete_model("nope.gguf")          # no error
    assert st["running"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_localmodels_delete.py -v --import-mode=importlib`
Expected: FAIL — `AttributeError: 'LocalModelManager' object has no attribute 'delete_model'`.

- [ ] **Step 3: Add `delete_model` to the manager**

Append this method inside the `LocalModelManager` class in `src/localmodels/manager.py` (after `list_models`):

```python
    def delete_model(self, filename: str) -> dict:
        """Delete a downloaded .gguf from MODELS_DIR; stop it first if serving."""
        base = os.path.basename(filename or "")
        if not base or base != filename or not base.lower().endswith(".gguf"):
            raise ValueError("invalid filename (must be a plain .gguf name)")
        with self._lock:
            if self._state and os.path.basename(
                    self._state.get("model_path") or "") == base:
                self._stop_locked()
            real_dir = os.path.realpath(MODELS_DIR)
            real = os.path.realpath(os.path.join(MODELS_DIR, base))
            try:
                inside = os.path.commonpath([real, real_dir]) == real_dir
            except ValueError:
                inside = False
            if inside and os.path.isfile(real):
                try:
                    os.remove(real)
                except Exception:
                    pass
        return self.status()
```

(`os` and `MODELS_DIR` are already imported at the top of `manager.py`.)

- [ ] **Step 4: Add the `/delete` route and `disk_bytes`**

In `routes/localmodels_routes.py`, replace the existing `/models` route body (currently `return {"models": get_manager().list_models()}`) with:

```python
    @router.get("/models")
    async def list_models():
        models = get_manager().list_models()
        return {"models": models,
                "disk_bytes": sum(int(m.get("size") or 0) for m in models)}
```

Then add a new route inside `setup_localmodels_routes()` (before `return router`):

```python
    @router.post("/delete")
    async def delete_model(payload: dict = Body(...)):
        filename = (payload.get("filename") or "").strip()
        try:
            return get_manager().delete_model(filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_localmodels_delete.py -v --import-mode=importlib`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/localmodels/manager.py routes/localmodels_routes.py tests/test_localmodels_delete.py
git commit -m "feat(localmodels): delete model (stop-if-serving) + disk usage"
```

---

### Task 3: Hardware, recommendations, and fit-annotated files routes

**Files:**
- Modify: `routes/localmodels_routes.py`
- Test: `tests/test_localmodels_hardware_routes.py`

**Interfaces:**
- Consumes: `hardware.get_hardware`, `hardware.recommend_models`, `hardware.fit_for_file`; `catalog.list_repo_gguf_files`.
- Produces: `GET /api/localmodels/hardware`, `GET /api/localmodels/recommendations`; `/catalog/files` files each gain a `fit` key.

- [ ] **Step 1: Write failing tests**

Create `tests/test_localmodels_hardware_routes.py`:

```python
"""Hardware/recommendations/fit-annotation route tests (fakes, no real hw)."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.localmodels_routes as lmr
from core.middleware import require_admin


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(lmr, "get_hardware",
                        lambda: {"ram_gb": 16.0, "has_gpu": True,
                                 "gpu_name": "RTX 3060", "vram_gb": 8.0})
    monkeypatch.setattr(lmr, "recommend_models",
                        lambda: [{"name": "Qwen2.5-7B", "score": 0.9}])
    monkeypatch.setattr(lmr, "list_repo_gguf_files",
                        lambda repo: [{"filename": "m-7B-Q4.gguf", "size": 4_000_000_000,
                                       "url": "https://huggingface.co/a/b/resolve/main/m-7B-Q4.gguf"}])
    monkeypatch.setattr(lmr, "fit_for_file",
                        lambda f, hw: {"verdict": "gpu", "needed_gb": 4.5,
                                       "size_gb": 4.0, "param_estimate_gb": 4.2})
    app = FastAPI()
    app.include_router(lmr.setup_localmodels_routes())
    app.dependency_overrides[require_admin] = lambda: None
    return TestClient(app)


def test_hardware_route(client):
    r = client.get("/api/localmodels/hardware")
    assert r.status_code == 200
    assert r.json()["gpu_name"] == "RTX 3060"


def test_recommendations_route(client):
    r = client.get("/api/localmodels/recommendations")
    assert r.status_code == 200
    assert r.json()["recommendations"][0]["name"] == "Qwen2.5-7B"


def test_catalog_files_annotated_with_fit(client):
    r = client.get("/api/localmodels/catalog/files", params={"repo": "a/b"})
    assert r.status_code == 200
    f = r.json()["files"][0]
    assert f["fit"]["verdict"] == "gpu"
    assert f["filename"] == "m-7B-Q4.gguf"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_localmodels_hardware_routes.py -v --import-mode=importlib`
Expected: FAIL — `/hardware` and `/recommendations` 404, and `catalog/files` has no `fit` key / `get_hardware` not importable in the module.

- [ ] **Step 3: Add the imports**

In `routes/localmodels_routes.py`, add at module level (with the other `from src.localmodels...` imports):

```python
from src.localmodels.hardware import get_hardware, recommend_models, fit_for_file
```

- [ ] **Step 4: Annotate `/catalog/files` and add the two GET routes**

Replace the existing `/catalog/files` route body (currently `return {"files": list_repo_gguf_files(repo)}`) with:

```python
    @router.get("/catalog/files")
    async def catalog_files(repo: str):
        files = list_repo_gguf_files(repo)
        hw = get_hardware()
        for f in files:
            f["fit"] = fit_for_file(f, hw)
        return {"files": files}
```

Add two routes inside `setup_localmodels_routes()` (before `return router`):

```python
    @router.get("/hardware")
    async def hardware():
        return get_hardware()

    @router.get("/recommendations")
    async def recommendations():
        return {"recommendations": recommend_models()}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_localmodels_hardware_routes.py -v --import-mode=importlib`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add routes/localmodels_routes.py tests/test_localmodels_hardware_routes.py
git commit -m "feat(localmodels): hardware + recommendations routes, fit-annotated files"
```

---

### Task 4: Hardware-aware UI

Extend the Local Models modal: hardware header, fit badges + sort, recommendations panel, per-model Delete + disk usage, and polish.

**Files:**
- Modify: `static/js/localModels.js`, `static/index.html`
- Test: `tests/test_localmodels_hardware_ui.py`

**Interfaces:**
- Consumes: `/api/localmodels/hardware`, `/recommendations`, `/delete`; `/models` `disk_bytes`; `/catalog/files` `fit`.
- Produces: `#localmodels-hardware`, `#localmodels-recommendations` in the modal.

- [ ] **Step 1: Write failing UI-wiring guard tests**

Create `tests/test_localmodels_hardware_ui.py`:

```python
"""Text guards for the Phase 3c hardware-aware UI."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_index_has_hardware_and_recommendations_elements():
    html = _read("static/index.html")
    assert 'id="localmodels-hardware"' in html
    assert 'id="localmodels-recommendations"' in html


def test_js_calls_hardware_recommendations_delete_and_uses_fit():
    js = _read("static/js/localModels.js")
    for ep in ("/api/localmodels/hardware", "/api/localmodels/recommendations",
               "/api/localmodels/delete"):
        assert ep in js, f"{ep} not called in localModels.js"
    assert ".fit" in js or "fit.verdict" in js  # badges consume the fit annotation
    assert "keydown" in js or "keypress" in js   # Enter-to-search wiring
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_localmodels_hardware_ui.py -v --import-mode=importlib`
Expected: FAIL — elements/endpoints not present yet.

- [ ] **Step 3: Add markup to the modal**

In `static/index.html`, inside `#localmodels-modal`'s `modal-content`, add a hardware header right after the opening header block (before `#localmodels-status` from 3a):

```html
      <div id="localmodels-hardware" style="font-size:12px;opacity:0.75;margin-bottom:8px;"></div>
```

And add a recommendations container right after the `#localmodels-search` row (the search input added in 3b):

```html
      <div id="localmodels-recommendations" style="margin-bottom:8px;"></div>
```

- [ ] **Step 4: Extend `localModels.js`**

Make these edits inside the IIFE in `static/js/localModels.js`:

(a) Collapse the duplicate size formatters — delete the `fmtSize` function (from 3a) and keep only `fmtBytes` (from 3b); update the one 3a call site `fmtSize(m.size)` to `fmtBytes(m.size)`.

(b) Add a hardware loader + verdict styling, placed before `window.LocalModels`:

```javascript
  let hardware = null;

  async function loadHardware() {
    const el = $('localmodels-hardware');
    if (!el) return;
    try { hardware = await api('/api/localmodels/hardware'); } catch (e) { hardware = null; }
    if (!hardware) { el.textContent = 'Hardware: unknown'; return; }
    const gpu = hardware.has_gpu
      ? `${hardware.gpu_name || 'GPU'} (${hardware.vram_gb} GB VRAM)` : 'no GPU';
    el.textContent = `Your machine: ${hardware.ram_gb} GB RAM · ${gpu}`;
  }

  function fitBadge(fit) {
    if (!fit) return '';
    const map = { gpu: ['Fits on GPU', '#3fb950'], ram: ['Fits in RAM', '#d29922'],
                  too_big: ['Too big', '#f85149'] };
    const [label, color] = map[fit.verdict] || ['', '#888'];
    return `<span style="color:${color};font-size:11px;margin-left:6px;">${label}</span>`;
  }

  const _verdictRank = { gpu: 0, ram: 1, too_big: 2 };

  async function loadRecommendations() {
    const el = $('localmodels-recommendations');
    if (!el) return;
    let data = { recommendations: [] };
    try { data = await api('/api/localmodels/recommendations'); } catch (e) {}
    if (!data.recommendations || !data.recommendations.length) { el.innerHTML = ''; return; }
    el.innerHTML = '<div style="font-size:11px;opacity:0.6;margin-bottom:4px;">Recommended for your machine</div>';
    data.recommendations.forEach((r) => {
      const chip = document.createElement('button');
      chip.textContent = r.name;
      chip.style.cssText = 'margin:2px 4px 2px 0;font-size:11px;';
      chip.onclick = () => {
        const inp = $('localmodels-search');
        if (inp) { inp.value = r.name; }
        doSearch();
      };
      el.appendChild(chip);
    });
  }
```

(c) In `listFiles` (from 3b), sort by fit then render the badge. Replace the `data.files.forEach(...)` loop so it (1) sorts and (2) appends `fitBadge`:

```javascript
    data.files.sort((a, b) =>
      (_verdictRank[(a.fit || {}).verdict] ?? 3) - (_verdictRank[(b.fit || {}).verdict] ?? 3)
      || (a.size - b.size));
    data.files.forEach((f) => {
      const row = document.createElement('div');
      row.className = 'list-item';
      row.style.paddingLeft = '18px';
      const label = document.createElement('span');
      label.className = 'grow';
      label.innerHTML = `${f.filename} — ${fmtBytes(f.size)}${fitBadge(f.fit)}`;
      const btn = document.createElement('button');
      const have = downloadedNames.has(f.filename);
      btn.textContent = have ? 'Downloaded' : 'Download';
      btn.disabled = have;
      btn.onclick = async () => {
        btn.disabled = true;
        try {
          await api('/api/localmodels/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: f.url, filename: f.filename }),
          });
          pollDownload();
        } catch (e) { alert('Download error: ' + e.message); btn.disabled = false; }
      };
      row.appendChild(label);
      row.appendChild(btn);
      afterRow.insertAdjacentElement('afterend', row);
    });
```

(d) In `refresh()` (from 3a), add a disk-usage header and a Delete button per model. After the models are fetched, set a header line and, inside the per-model row build, add a Delete button. Add this near the top of the models render (after `const data = await api('/api/localmodels/models')` succeeds and `downloadedNames` is set):

```javascript
    const statusEl = $('localmodels-status');
    if (statusEl && data.disk_bytes != null) {
      const base = status.running ? `Running: ${status.model} (port ${status.port})` : 'No model running';
      statusEl.textContent = `${base}  ·  ${data.models.length} models · ${fmtBytes(data.disk_bytes)}`;
    }
```

And in the per-model row (the existing loop that builds each model's Serve/Stop row), add a Delete button after the Serve/Stop button:

```javascript
      const del = document.createElement('button');
      del.textContent = 'Delete';
      del.style.marginLeft = '4px';
      del.onclick = async () => {
        if (!confirm('Delete ' + m.name + '?')) return;
        try { await api('/api/localmodels/delete', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: m.name }) }); }
        catch (e) { alert('Delete error: ' + e.message); }
        await refresh();
      };
      row.appendChild(del);
```

(e) In the `open()` function (3a), also load hardware + recommendations:

```javascript
  function open() {
    const modal = $('localmodels-modal');
    if (modal) { modal.classList.remove('hidden'); refresh(); loadHardware(); loadRecommendations(); }
  }
```

(f) Enter-to-search: in the `DOMContentLoaded` handler, after wiring the search button, add:

```javascript
    const searchInput = $('localmodels-search');
    if (searchInput) searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(); });
```

(g) Extend the export: `window.LocalModels = { open, close, refresh, doSearch, pollDownload, loadHardware, loadRecommendations };`

- [ ] **Step 5: Run the UI-wiring guard tests**

Run: `python -m pytest tests/test_localmodels_hardware_ui.py -v --import-mode=importlib`
Expected: PASS (2 tests).

> **Manual (not unit-tested):** open Local Models — confirm the hardware header, fit badges on search results (sorted best-fit first), the recommendations panel (click pre-fills + searches), per-model Delete, disk-usage line, and Enter-to-search. Covered by the 3c smoke test.

- [ ] **Step 6: Commit**

```bash
git add static/js/localModels.js static/index.html tests/test_localmodels_hardware_ui.py
git commit -m "feat(localmodels): hardware-aware UI (badges, recommendations, delete, polish)"
```

---

## Appendix: Manual smoke test (real acceptance gate)

With the app running:
1. Open **Local Models** → the hardware header shows real RAM + GPU/VRAM.
2. The **Recommended for your machine** panel lists models; click one → search runs for it.
3. Search results / files show **fit badges** (green/yellow/red), sorted best-fit first; a clearly-too-large model reads red.
4. Download a small model → it lands in the list; the header shows "N models · X GB".
5. Click **Delete** on a downloaded model → it's removed (and stopped first if it was serving); disk usage updates.
6. Press **Enter** in the search box → search runs.

## Self-Review

**Spec coverage:**
- Hardware detection (spec C1) → Task 1 `get_hardware` + Task 3 `/hardware`. ✓
- Hybrid fit (C2, decision 3) → Task 1 `fit_for_file` (size + hwfit `estimate_memory_gb`, conservative max) + Task 3 `/catalog/files` annotation + Task 4 badges/sort. ✓
- Recommendations (C3) → Task 1 `recommend_models` + Task 3 `/recommendations` + Task 4 panel (pre-fills search). ✓
- Per-model management (C4) → Task 2 `delete_model` + `/delete` + `disk_bytes` + Task 4 Delete button/header. ✓
- Polish (C5) → Task 4 (fmt dedup, Enter-to-search). ✓
- hwfit read-only / safe-default / delete-safety constraints → honored in Tasks 1-2 and their tests. ✓
- Testing (unit + routes + text-guard UI + manual smoke) → each task + appendix. ✓

**Placeholder scan:** No TBD/TODO; every code step is complete. Manual-only real-hardware/GUI steps are labeled and backed by unit fakes + the smoke appendix. ✓

**Type consistency:** `get_hardware()` dict `{ram_gb, has_gpu, gpu_name, vram_gb}` identical across Task 1, the Task 3 route + its fake, and Task 4 (`hardware.ram_gb`/`vram_gb`/`gpu_name`). `fit_for_file(...)` return `{verdict, needed_gb, size_gb, param_estimate_gb}` matches between Task 1, the Task 3 annotation, and Task 4's `fitBadge(fit.verdict)`. `recommend_models()` items `{name, score}` match across Task 1, Task 3, Task 4. `delete_model(filename)` / `/delete {filename}` / `status()` shape consistent between Task 2 and Task 4. `/models` `disk_bytes` int matches Task 2 route and Task 4 header. Endpoint paths identical across Tasks 2-4 and all tests. ✓

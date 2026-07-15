# Live Hardware Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A collapsible "Hardware monitor" sidebar panel showing ~60s sparkline history of CPU %, RAM %, and per-GPU VRAM % + GPU-util %, updating ~1s and polling only while open.

**Architecture:** A `usage()` function in hwfit (psutil for CPU/RAM + nvidia-smi for per-GPU VRAM/util, reusing the module's `_run`/`NVIDIA_PATH_CANDIDATES` pattern) behind a read-only `GET /api/hwfit/usage`; a CSP-safe sidebar `<details>` panel that polls it into per-metric ring buffers and draws canvas sparklines.

**Tech Stack:** Python 3, psutil (existing dep), nvidia-smi, FastAPI, pytest (`--import-mode=importlib`), vanilla CSP-safe JS + canvas. No new dependencies.

## Global Constraints

- `usage()` is in `services/hwfit/hardware.py`, **never raises**, returns
  `{cpu_percent, ram_used_gb, ram_total_gb, ram_percent, gpus: [{index, name, vram_used_gb, vram_total_gb, vram_percent, util_percent}]}`; `gpus == []` when there's no nvidia-smi / NVIDIA GPU.
- GPU query reuses the existing `_run` + `NVIDIA_PATH_CANDIDATES` fallback (same as `free_vram_gb`); nvidia-smi memory is MiB → `/1024` for GB; `[N/A]`/non-numeric rows are skipped; `total==0` must not divide-by-zero.
- `GET /api/hwfit/usage` is added to `setup_hwfit_routes()` (prefix `/api/hwfit`), read-only, ungated (same as `/system`), using the lazy-import handler style of the sibling endpoints.
- UI is CSP-safe: `createElement` + `addEventListener` + canvas 2d only; dynamic text via `textContent`; no inline handlers, no `innerHTML` with data. Poll **only while the panel is expanded**; stop on collapse.
- Run pytest with `--import-mode=importlib`. Stage only the files each task names — never `git add -A`.

---

### Task 1: `usage()` + `/api/hwfit/usage`

**Files:**
- Modify: `services/hwfit/hardware.py` (add a guarded `import psutil` + `usage()` + `_gpu_usage()`, near `free_vram_gb` ~line 200)
- Modify: `routes/hwfit_routes.py` (add `@router.get("/usage")` in `setup_hwfit_routes`)
- Test: `tests/test_hwfit_usage.py`, `tests/test_hwfit_usage_route.py`

**Interfaces:**
- Consumes: `_run(cmd)` + `NVIDIA_PATH_CANDIDATES` (existing in hardware.py).
- Produces: `usage() -> dict` (shape above); `GET /api/hwfit/usage → usage()`.

- [ ] **Step 1: Write the failing `usage()` tests**

Create `tests/test_hwfit_usage.py`:

```python
import services.hwfit.hardware as hw


class _FakeVM:
    used = 12e9
    total = 64e9
    percent = 23.9


class _FakePsutil:
    @staticmethod
    def cpu_percent():
        return 42.0

    @staticmethod
    def virtual_memory():
        return _FakeVM()


def test_usage_parses_cpu_ram_and_gpu(monkeypatch):
    monkeypatch.setattr(hw, "psutil", _FakePsutil)
    monkeypatch.setattr(hw, "_run", lambda cmd: "0, RTX 4050, 515, 6141, 6\n")
    u = hw.usage()
    assert u["cpu_percent"] == 42.0
    assert u["ram_total_gb"] == 64.0 and u["ram_percent"] == 23.9
    g = u["gpus"][0]
    assert g["index"] == 0 and g["name"] == "RTX 4050"
    assert g["vram_total_gb"] == 6.0 and abs(g["vram_percent"] - 8.4) < 0.3
    assert g["util_percent"] == 6.0


def test_usage_multi_gpu_in_order(monkeypatch):
    monkeypatch.setattr(hw, "psutil", None)
    monkeypatch.setattr(hw, "_run", lambda cmd: "0, A, 100, 1000, 10\n1, B, 200, 2000, 20\n")
    g = hw.usage()["gpus"]
    assert [x["index"] for x in g] == [0, 1] and g[1]["name"] == "B"


def test_usage_no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(hw, "psutil", None)
    monkeypatch.setattr(hw, "_run", lambda cmd: None)
    u = hw.usage()
    assert u["gpus"] == [] and u["cpu_percent"] == 0.0


def test_usage_tolerates_na_and_zero_total(monkeypatch):
    monkeypatch.setattr(hw, "psutil", None)
    monkeypatch.setattr(hw, "_run", lambda cmd: "0, Uni, [N/A], [N/A], [N/A]\n1, G, 0, 0, 0\n")
    g = hw.usage()["gpus"]
    assert [x["index"] for x in g] == [1] and g[0]["vram_percent"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_hwfit_usage.py --import-mode=importlib -q`
Expected: FAIL (`AttributeError: module 'services.hwfit.hardware' has no attribute 'usage'`).

- [ ] **Step 3: Implement `usage()`**

In `services/hwfit/hardware.py`, add a guarded psutil import near the top imports:

```python
try:
    import psutil
except Exception:  # pragma: no cover - psutil is a normal dep, guard is defensive
    psutil = None
```

And add these functions (place them right after `free_vram_gb`, ~line 200):

```python
def _gpu_usage() -> list:
    """Per-NVIDIA-GPU live VRAM + util via nvidia-smi. [] if unavailable."""
    q = ["--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"]
    out = _run(["nvidia-smi", *q])
    if not out:
        for _p in NVIDIA_PATH_CANDIDATES:
            out = _run([_p, *q])
            if out:
                break
    if not out:
        return []
    gpus = []
    for line in out.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            used = float(parts[2])
            total = float(parts[3])
            util = float(parts[4])
        except ValueError:
            continue  # [N/A] unified-memory rows, non-numeric fields
        gpus.append({
            "index": idx, "name": parts[1],
            "vram_used_gb": round(used / 1024.0, 2),
            "vram_total_gb": round(total / 1024.0, 2),
            "vram_percent": round(used / total * 100.0, 1) if total else 0.0,
            "util_percent": util,
        })
    return gpus


def usage() -> dict:
    """Live resource usage: CPU%, RAM, and per-NVIDIA-GPU VRAM+util. Never raises;
    degrades to zeros / empty gpus when a source is unavailable."""
    cpu = ram_used = ram_total = ram_pct = 0.0
    if psutil is not None:
        try:
            cpu = float(psutil.cpu_percent())
            vm = psutil.virtual_memory()
            ram_used = round(vm.used / 1e9, 2)
            ram_total = round(vm.total / 1e9, 2)
            ram_pct = float(vm.percent)
        except Exception:
            pass
    return {"cpu_percent": cpu, "ram_used_gb": ram_used, "ram_total_gb": ram_total,
            "ram_percent": ram_pct, "gpus": _gpu_usage()}
```

- [ ] **Step 4: Run to verify `usage()` passes**

Run: `python -m pytest tests/test_hwfit_usage.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Write the failing route test**

Create `tests/test_hwfit_usage_route.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.hwfit_routes as hr
import services.hwfit.hardware as hw


def test_usage_route(monkeypatch):
    monkeypatch.setattr(hw, "usage", lambda: {
        "cpu_percent": 5.0, "ram_used_gb": 1.0, "ram_total_gb": 8.0,
        "ram_percent": 12.5, "gpus": []})
    app = FastAPI()
    app.include_router(hr.setup_hwfit_routes())
    r = TestClient(app).get("/api/hwfit/usage")
    assert r.status_code == 200
    j = r.json()
    assert j["cpu_percent"] == 5.0 and j["ram_total_gb"] == 8.0 and j["gpus"] == []
```

- [ ] **Step 6: Run to verify it fails**

Run: `python -m pytest tests/test_hwfit_usage_route.py --import-mode=importlib -q`
Expected: FAIL (404 — route not defined yet).

- [ ] **Step 7: Add the route**

In `routes/hwfit_routes.py`, inside `setup_hwfit_routes()` (next to the other `@router.get` handlers, e.g. after `get_system`), add:

```python
    @router.get("/usage")
    def get_usage():
        from services.hwfit.hardware import usage
        return usage()
```

- [ ] **Step 8: Run both suites**

Run: `python -m pytest tests/test_hwfit_usage.py tests/test_hwfit_usage_route.py --import-mode=importlib -q`
Expected: PASS (5 passed).

- [ ] **Step 9: Commit**

```bash
git add services/hwfit/hardware.py routes/hwfit_routes.py tests/test_hwfit_usage.py tests/test_hwfit_usage_route.py
git commit -m "feat(hwfit): live usage() (cpu/ram/gpu) + GET /api/hwfit/usage"
```

---

### Task 2: Sidebar panel + sparklines UI

**Files:**
- Create: `static/js/hardwareMonitor.js`
- Modify: `static/index.html` (add the collapsible panel markup in the sidebar; add the `<script>` tag near the other sidebar-widget scripts)
- Test: manual/live-verify (Task 3); no unit-test framework for these DOM files.

**Interfaces:**
- Consumes: `GET /api/hwfit/usage` (Task 1).

- [ ] **Step 1: Add the panel markup**

In `static/index.html`, add a collapsible section in the sidebar near the other
sidebar widgets (e.g. beside the "Input control"/"Shell commands" consent rows
~line 963-971, or wherever sidebar panels live). Use a `<details>` so open/close
is native:

```html
        <details id="hwmon" class="list-item" style="display:block;padding:6px 8px;">
          <summary style="cursor:pointer;font-size:13px;font-weight:600;list-style:none;">Hardware monitor</summary>
          <div id="hwmon-body" style="margin-top:6px;"></div>
        </details>
```

Add the script tag next to the other sidebar-widget scripts (search `static/index.html`
for `inputControl.js` or `shellExec.js`):

```html
<script src="/static/js/hardwareMonitor.js"></script>
```

- [ ] **Step 2: Implement hardwareMonitor.js**

Create `static/js/hardwareMonitor.js` — a CSP-safe IIFE that polls only while the
`<details>` is open and draws canvas sparklines:

```javascript
// Live hardware monitor: polls /api/hwfit/usage while the sidebar panel is open,
// keeps a ~60-sample ring buffer per metric, draws canvas sparklines. CSP-safe.
(function () {
  const POLL_MS = 1000;
  const WINDOW = 60;                 // samples kept per series
  let timer = null;
  const series = {};                 // key -> [values]
  const canvases = {};               // key -> {canvas, label}

  function $(id) { return document.getElementById(id); }

  function push(key, val) {
    const s = series[key] || (series[key] = []);
    s.push(val);
    if (s.length > WINDOW) s.shift();
  }

  function ensureRow(key, title) {
    if (canvases[key]) return canvases[key];
    const body = $('hwmon-body');
    const row = document.createElement('div');
    row.style.cssText = 'margin-bottom:6px;';
    const label = document.createElement('div');
    label.style.cssText = 'font-size:11px;opacity:0.8;margin-bottom:2px;';
    label.textContent = title;
    const cv = document.createElement('canvas');
    cv.width = 180; cv.height = 26;
    cv.style.cssText = 'width:100%;height:26px;display:block;background:rgba(127,127,127,0.12);border-radius:3px;';
    row.appendChild(label); row.appendChild(cv);
    body.appendChild(row);
    return (canvases[key] = { canvas: cv, label: label });
  }

  function draw(key) {
    const ref = canvases[key]; if (!ref) return;
    const s = series[key] || [];
    const ctx = ref.canvas.getContext('2d');
    const w = ref.canvas.width, h = ref.canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (s.length < 2) return;
    ctx.strokeStyle = '#50fa7b';       // literal — canvas 2d can't resolve CSS var()
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < s.length; i++) {
      const x = (i / (WINDOW - 1)) * w;
      const y = h - Math.max(0, Math.min(100, s[i])) / 100 * (h - 2) - 1;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function render(u) {
    push('cpu', u.cpu_percent || 0);
    ensureRow('cpu', 'CPU ' + Math.round(u.cpu_percent || 0) + '%').label.textContent =
      'CPU ' + Math.round(u.cpu_percent || 0) + '%';
    draw('cpu');

    push('ram', u.ram_percent || 0);
    ensureRow('ram', '').label.textContent =
      'RAM ' + (u.ram_used_gb || 0) + '/' + (u.ram_total_gb || 0) + ' GB (' + Math.round(u.ram_percent || 0) + '%)';
    draw('ram');

    (u.gpus || []).forEach((g) => {
      const vk = 'gpu' + g.index + 'vram';
      push(vk, g.vram_percent || 0);
      ensureRow(vk, '').label.textContent =
        'GPU' + g.index + ' VRAM ' + g.vram_used_gb + '/' + g.vram_total_gb + ' GB (' + Math.round(g.vram_percent) + '%)';
      draw(vk);
      const uk = 'gpu' + g.index + 'util';
      push(uk, g.util_percent || 0);
      ensureRow(uk, '').label.textContent = 'GPU' + g.index + ' util ' + Math.round(g.util_percent) + '%';
      draw(uk);
    });

    if (!(u.gpus || []).length && !canvases.nogpu) {
      const body = $('hwmon-body');
      const n = document.createElement('div');
      n.id = 'hwmon-nogpu'; n.style.cssText = 'font-size:11px;opacity:0.6;';
      n.textContent = 'No NVIDIA GPU detected';
      body.appendChild(n);
      canvases.nogpu = true;
    }
  }

  async function poll() {
    try {
      const res = await fetch('/api/hwfit/usage', { credentials: 'same-origin' });
      if (res.ok) render(await res.json());
    } catch (e) { /* skip this tick */ }
  }

  function start() { if (!timer) { poll(); timer = setInterval(poll, POLL_MS); } }
  function stop() { if (timer) { clearInterval(timer); timer = null; } }

  document.addEventListener('DOMContentLoaded', () => {
    const d = $('hwmon');
    if (!d) return;
    d.addEventListener('toggle', () => (d.open ? start() : stop()));
    if (d.open) start();
  });
})();
```

- [ ] **Step 3: Commit**

```bash
git add static/js/hardwareMonitor.js static/index.html
git commit -m "feat(hwfit): live hardware monitor sidebar panel (sparklines, CSP-safe)"
```

Note: the exact sidebar insertion point + `<script>` placement are repo-specific —
the implementer should place the `<details>` beside the existing sidebar consent/panel
rows and the script next to `inputControl.js`/`shellExec.js`, adapting the surrounding
markup to match (ids `hwmon` / `hwmon-body` must stay as the JS expects).

---

### Task 3: Package + live-verify

**Files:**
- Modify: `installer/Output/Assist-Setup.exe` (rebuilt; force-added like prior build commits)

**Interfaces:**
- Consumes: Tasks 1-2.

- [ ] **Step 1: Full affected-suite run**

Run: `python -m pytest tests/test_hwfit_usage.py tests/test_hwfit_usage_route.py --import-mode=importlib -q`
Expected: PASS (5 passed).

- [ ] **Step 2: Build the installer**

Run: `powershell -ExecutionPolicy Bypass -File .\build-installer.ps1 -Fast`
Expected: ends with `Installer built: ...\installer\Output\Assist-Setup.exe`.

- [ ] **Step 3: Frozen `usage()` check (psutil bundled + real values)**

Run: `./dist/Assist/Assist.exe --run-py -c "from services.hwfit.hardware import usage; u=usage(); print('CPU', u['cpu_percent'], 'RAM%', u['ram_percent'], 'GPUS', len(u['gpus']), (u['gpus'][0]['vram_percent'] if u['gpus'] else 'n/a'))"`
Expected: real numbers (CPU ≥ 0, RAM% > 0), and (on this 6GB RTX 4050) one GPU with a live `vram_percent`. Confirms psutil is bundled in the frozen build.

- [ ] **Step 4: Live-verify in the running app (manual)**

Reinstall; then:
- Open the sidebar "Hardware monitor" panel → CPU / RAM / GPU VRAM / GPU util sparklines start updating ~1s.
- Serve an image model and generate → watch the GPU VRAM % sparkline climb during load/generation, then settle.
- Collapse the panel → confirm the `/api/hwfit/usage` polling stops (DevTools → Network).

- [ ] **Step 5: Commit the installer**

```bash
git add -f installer/Output/Assist-Setup.exe
git commit -m "build: Assist-Setup.exe with live hardware monitor"
```

---

## Notes for the executor

- Every pytest run uses `--import-mode=importlib`.
- Task 1 is pure Python + monkeypatched psutil/`_run` (no real nvidia-smi in tests). Task 2 is UI (verified live in Task 3).
- `usage()` must never raise — `tests/test_hwfit_usage.py::test_usage_no_nvidia_smi` and `::test_usage_tolerates_na_and_zero_total` are the guards.
- Do not change the existing hwfit detection/`free_vram_gb` behavior — only add `usage()`/`_gpu_usage()` and the one route.

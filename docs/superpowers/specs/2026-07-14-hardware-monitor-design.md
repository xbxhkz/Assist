# Live Hardware Monitor Design

**Goal:** A collapsible "Hardware monitor" sidebar panel showing ~60-second
sparkline history of CPU %, RAM %, and per-GPU VRAM % + GPU-util %, updating every
~1s. Polls only while open.

**Scope:** Read-only live monitor. No history persistence, no alerts/thresholds.
NVIDIA GPUs only for VRAM/util (AMD/Intel live-util out of scope for v1; CPU/RAM
still shown on all machines). One of the "smart hardware detection" gaps from the
feature audit; self-contained.

---

## Background — what exists

- `services/hwfit/hardware.py` already detects hardware statically (CPU, RAM, GPUs)
  and has an nvidia-smi runner `_run(cmd)` + `NVIDIA_PATH_CANDIDATES` absolute-path
  fallback, plus `free_vram_gb()` (added earlier) that follows exactly the
  `_run` + fallback + tolerate-`[N/A]` pattern this feature reuses.
- `routes/hwfit_routes.py:setup_hwfit_routes()` → `APIRouter(prefix="/api/hwfit")`,
  read-only and ungated (hardware detection is not admin-gated). Live usage is the
  same read-only nature, so `GET /api/hwfit/usage` belongs here.
- `psutil` (7.2.2) is installed and importable (`cpu_percent`, `virtual_memory`).
- Sidebar consent/panel widgets (`static/js/inputControl.js`, `screenAccess.js`,
  `shellExec.js`) + the operator panel establish the CSP-safe sidebar-section
  pattern to mirror.

## Architecture

Three units:

```
sidebar "Hardware monitor" panel (hardwareMonitor.js)
  └── while EXPANDED: poll GET /api/hwfit/usage every ~1000ms
        │  ring buffer (~60 samples/metric) → <canvas> sparkline + current value
        ▼
routes/hwfit_routes.py  GET /api/hwfit/usage → usage()
        ▼
services/hwfit/hardware.py  usage()  (psutil CPU/RAM + nvidia-smi per-GPU)
```

### 1. `usage()` — `services/hwfit/hardware.py`

`usage() -> dict`, **never raises**:

```
{
  "cpu_percent": float,          # psutil.cpu_percent() — % since last call
  "ram_used_gb": float,
  "ram_total_gb": float,
  "ram_percent": float,          # psutil.virtual_memory()
  "gpus": [                      # [] when no nvidia-smi / no NVIDIA GPU
    {"index": int, "name": str,
     "vram_used_gb": float, "vram_total_gb": float, "vram_percent": float,
     "util_percent": float}
  ]
}
```

- CPU: `psutil.cpu_percent()` (no interval → non-blocking, measures since the
  previous call; the ~1s poll cadence makes each sample a ~1s window).
- RAM: `psutil.virtual_memory()` → `used`/`total` (bytes → GB), `percent`.
- GPUs: `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu
  --format=csv,noheader,nounits`, run via the existing `_run` with the same
  `NVIDIA_PATH_CANDIDATES` absolute-path fallback `free_vram_gb()` uses. Parse each
  CSV line; a row with a non-numeric field (`[N/A]` unified-memory parts) is
  skipped, not fatal. `vram_percent = used/total*100` (guard total==0).
- Any psutil/parse exception is swallowed and that metric is omitted / defaulted
  (CPU/RAM default to `0.0`; gpus default to `[]`).

### 2. `GET /api/hwfit/usage` — `routes/hwfit_routes.py`

Add to `setup_hwfit_routes()`:

```python
@router.get("/usage")
def get_usage():
    return usage()
```

Read-only, ungated (consistent with `/api/hwfit/system`).

### 3. UI — `static/js/hardwareMonitor.js` + panel in `static/index.html`

- A collapsible "Hardware monitor" sidebar section (a `<details>` or a
  toggle-header + body, styled like the existing sidebar panels).
- On expand: start a ~1000ms poll of `/api/hwfit/usage`; on collapse: stop the
  interval (no idle polling).
- State: a ring buffer of the last ~60 samples per series (cpu%, ram%, and per-GPU
  vram% + util%). Each tick pushes the new values and trims to 60.
- Render: per series, a small `<canvas>` sparkline (line over the 60-sample window,
  0-100 y-scale) + a current-value text label (e.g. "CPU 42%", "RAM 12.1/64 GB",
  "GPU0 VRAM 5.2/6.0 GB", "GPU0 88%"). No GPU → a "No NVIDIA GPU detected" line
  (CPU/RAM still shown).
- CSP-safe: `createElement` + `addEventListener` + canvas 2d only; all dynamic text
  via `textContent`; no inline handlers, no `innerHTML` with data.

## Error handling

- `usage()` degrades: missing nvidia-smi/GPU → `gpus: []`; `[N/A]`/non-numeric rows
  skipped; psutil failure → CPU/RAM default to `0.0` rather than 500.
- A failed poll (network/500) skips that tick and keeps the last graph; the panel
  doesn't error out.
- Collapsed panel does zero work.

## Testing

- **`usage()`** (`tests/test_hwfit_usage.py`): monkeypatch `hardware.psutil` (or the
  cpu/mem calls) + `hardware._run`:
  - full nvidia-smi CSV (`0, RTX 4050, 515, 6141, 6`) → one gpu dict with
    `vram_used_gb≈0.50`, `vram_total_gb≈6.0`, `vram_percent≈8.4`, `util_percent==6`.
  - multi-line CSV → multiple gpus in index order.
  - `_run` → `None` (no nvidia-smi) → `gpus == []`, cpu/ram still populated.
  - a `[N/A]` row is skipped, not fatal; `total==0` doesn't divide-by-zero.
- **route** (`tests/test_hwfit_usage_route.py`): TestClient `GET /api/hwfit/usage`
  with monkeypatched `usage()` → 200 + the expected keys.
- **Live-verify (frozen app):** open the Hardware monitor panel → CPU/RAM/VRAM/util
  sparklines update ~1s; start an image generation and watch VRAM % climb; collapse
  the panel and confirm the network poll stops (DevTools network tab).

## Non-goals

- History persistence / logging (in-memory ring buffer only).
- Alerts, thresholds, or notifications.
- AMD / Intel / Apple GPU live VRAM+util (v1 shows CPU/RAM everywhere, NVIDIA GPU
  metrics where nvidia-smi is present).
- Per-process breakdown.
- Remote/SSH host usage (local machine only).

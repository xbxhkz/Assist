# Image AI Studio — Training GUI (Sub-project 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A frontend-only admin panel over the already-shipped, headless SDXL image-LoRA training engine — pick a saved Dataset Prep dataset, set hyperparameters, start/stop a run, and watch live progress.

**Architecture:** One new modal (`#image-training-modal`) + a pure core module (`imageTrainingCore.js`, form↔config mapping and status-line rendering, no DOM/fetch) + a DOM controller module (`imageTraining.js`, polling + admin-gated buttons), mirroring the existing text-LoRA Training panel's (`training.js`/`trainingCore.js`) exact shape.

**Tech Stack:** Vanilla ES modules, no build step, calling the 5 already-shipped `/api/image-training/*` routes. Zero backend changes.

## Global Constraints

- **No backend changes.** All 5 routes (`GET /api/image-training/env`, `POST /api/image-training/env/setup`, `POST /api/image-training/runs`, `GET /api/image-training/runs/current`, `POST /api/image-training/runs/stop`) already exist and are already tested — this plan only adds frontend files.
- **New, separate modal** — not a tab inside the existing `#imagedataset-modal`.
- **No adapters/management section.** A trained LoRA lands in `loras_dir()` and is already browsable via the existing `static/js/loras.js` / `GET /api/loras` — this panel does not duplicate that, it just points at it once a run finishes.
- **`base_model` is not a form field** — `ImageTrainingConfig` already defaults it server-side to the one supported value; `formToConfig` must never send a `base_model` key.
- **Progress via 1.5s polling** (`setInterval`), matching `training.js`'s exact pattern — no SSE/WebSocket.
- **Proven hyperparameter defaults (use these exact values as the Advanced section's pre-filled values and `formToConfig`'s fallbacks):** `rank=4`, `lora_alpha=4`, `learning_rate=0.0001` (1e-4), `steps=1000`, `resolution=1024`.
- **The panel must state, visibly, that results only save when a run finishes** — no intermediate checkpointing exists in the engine.
- **Admin-gated**: the rail button and sidebar entry stay `style="display:none"` until `isAdmin()` (calling `GET /api/auth/status`, checking `is_admin`) reveals them — matching every other admin-only panel in this app.
- **JS test files are Node-subprocess/`node --check` based** (no browser automation in this suite) — mirrors `tests/test_training_core_js.py` and `tests/test_image_dataset_ui.py` exactly.
- Stage specific files when committing; never `git add -A`. Do not stage `installer/Output/Assist-Setup.exe`. Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Pure core module (`imageTrainingCore.js`)

**Files:**
- Create: `static/js/imageTrainingCore.js`
- Test: `tests/test_image_training_core_js.py`

**Interfaces:**
- Produces: `formToConfig(v: object) -> object` — maps raw form-field strings to the `POST /api/image-training/runs` request body (`dataset_name`, `output_name`, `rank`, `lora_alpha`, `learning_rate`, `steps`, `resolution` — never a `base_model` key). `renderStatusLine(s: object) -> string` — maps a `/api/image-training/runs/current` status dict to a single display line. Both consumed by Task 2's `imageTraining.js`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_image_training_core_js.py
import json
import subprocess
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "static" / "js" / "imageTrainingCore.js"


def _node(expr):
    script = f"import * as m from {json.dumps(CORE.as_uri())};\n{expr}"
    p = subprocess.run(["node", "--input-type=module", "-e", script],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_form_to_config_uses_given_values():
    out = _node("console.log(JSON.stringify(m.formToConfig("
                "{dataset_name:'ds1', output_name:'my-lora', rank:'8', lora_alpha:'8', "
                "learning_rate:'0.0002', steps:'500', resolution:'768'})));")
    cfg = json.loads(out)
    assert cfg == {"dataset_name": "ds1", "output_name": "my-lora", "rank": 8,
                   "lora_alpha": 8, "learning_rate": 0.0002, "steps": 500, "resolution": 768}


def test_form_to_config_blank_numeric_fields_fall_back_to_proven_defaults():
    out = _node("console.log(JSON.stringify(m.formToConfig("
                "{dataset_name:'ds1', output_name:'my-lora', rank:'', lora_alpha:'', "
                "learning_rate:'', steps:'', resolution:''})));")
    cfg = json.loads(out)
    assert cfg["rank"] == 4 and cfg["lora_alpha"] == 4
    assert cfg["learning_rate"] == 0.0001
    assert cfg["steps"] == 1000 and cfg["resolution"] == 1024


def test_form_to_config_trims_names_and_omits_base_model():
    out = _node("console.log(JSON.stringify(m.formToConfig("
                "{dataset_name:'  ds1  ', output_name:'  my-lora  '})));")
    cfg = json.loads(out)
    assert cfg["dataset_name"] == "ds1" and cfg["output_name"] == "my-lora"
    assert "base_model" not in cfg


def test_render_status_line_running():
    out = _node("console.log(m.renderStatusLine("
                "{status:'running', last_step:5, loss:0.42, vram_gb:6.0}));")
    assert out == "status: running \u00b7 step 5 \u00b7 loss 0.42 \u00b7 vram 6 GB"


def test_render_status_line_done_shows_lora_path():
    out = _node("console.log(m.renderStatusLine("
                "{status:'done', peak_vram_gb:6.04, lora_path:'C:/loras/my-lora.safetensors'}));")
    assert out == "status: done \u00b7 saved: C:/loras/my-lora.safetensors"


def test_render_status_line_error():
    out = _node("console.log(m.renderStatusLine({status:'error', error:'ran out of VRAM'}));")
    assert out == "status: error \u00b7 (ran out of VRAM)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_image_training_core_js.py -v --import-mode=importlib`
Expected: FAIL (`imageTrainingCore.js` doesn't exist yet — Node raises a module-not-found error, `p.returncode != 0`, assertion fails on `p.stderr`)

- [ ] **Step 3: Write the implementation**

```javascript
// static/js/imageTrainingCore.js
// Pure helpers for the Image Training panel -- no DOM, no fetch. Mirrors
// static/js/trainingCore.js's shape for the SDXL image-LoRA training engine
// (src/image_training/config.py's ImageTrainingConfig fields). base_model is
// intentionally never produced here -- the server already defaults it to the
// one supported value (SUPPORTED_BASE_MODELS is a single-entry allowlist).
export function formToConfig(v) {
  return {
    dataset_name: (v.dataset_name || '').trim(),
    output_name: (v.output_name || '').trim(),
    rank: parseInt(v.rank, 10) || 4,
    lora_alpha: parseInt(v.lora_alpha, 10) || 4,
    learning_rate: parseFloat(v.learning_rate) || 1e-4,
    steps: parseInt(v.steps, 10) || 1000,
    resolution: parseInt(v.resolution, 10) || 1024,
  };
}

export function renderStatusLine(s) {
  const bits = ['status: ' + s.status];
  if (s.last_step != null) bits.push('step ' + s.last_step);
  if (s.loss != null) bits.push('loss ' + s.loss);
  if (s.vram_gb != null) bits.push('vram ' + s.vram_gb + ' GB');
  if (s.status === 'done' && s.lora_path != null) bits.push('saved: ' + s.lora_path);
  if (s.error) bits.push('(' + s.error + ')');
  return bits.join(' \u00b7 ');
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_image_training_core_js.py -v --import-mode=importlib`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add static/js/imageTrainingCore.js tests/test_image_training_core_js.py
git commit -m "feat(image-training-gui): add pure form/status core module"
```

---

### Task 2: Modal, DOM controller, sidebar wiring

**Files:**
- Modify: `static/index.html` (5 insertion points, see Step 3)
- Create: `static/js/imageTraining.js`
- Test: `tests/test_image_training_ui.py`

**Interfaces:**
- Consumes: `formToConfig`/`renderStatusLine` from Task 1's `static/js/imageTrainingCore.js`; `Modals.register(modalId, opts)` from the existing `static/js/modalManager.js` (already used by every other panel — same call shape as `training.js`'s `Modals.register('training-modal', {...})`); the 5 already-shipped `/api/image-training/*` routes; the already-shipped `GET /api/image-datasets` (returns `{"datasets": [{"name","path","images","size"}, ...]}`); the already-shipped `GET /api/auth/status` (returns `{"is_admin": bool, ...}`).
- Produces: nothing consumed by a later task — this is the final task of this sub-project.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_image_training_ui.py
import pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_index_has_image_training_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="image-training-modal"', 'id="rail-imagetraining"', 'id="tool-imagetraining-btn"',
               '/static/js/imageTraining.js',
               'id="imgtrain-env-status"', 'id="imgtrain-env-setup"',
               'id="imgtrain-dataset"', 'id="imgtrain-dataset-suggestions"',
               'id="imgtrain-output-name"', 'id="imgtrain-rank"', 'id="imgtrain-alpha"',
               'id="imgtrain-lr"', 'id="imgtrain-steps"', 'id="imgtrain-resolution"',
               'id="imgtrain-start"', 'id="imgtrain-stop"', 'id="imgtrain-progress"'):
        assert el in html, f"{el} missing from index.html"


def test_image_training_js_wires_admin_and_routes():
    src = (ROOT / "static" / "js" / "imageTraining.js").read_text(encoding="utf-8")
    for s in ('rail-imagetraining', 'tool-imagetraining-btn', 'isAdmin', 'Modals.register',
              '/api/image-training/env', '/api/image-training/env/setup',
              '/api/image-training/runs', '/api/image-training/runs/current',
              '/api/image-training/runs/stop', '/api/image-datasets'):
        assert s in src, f"{s} missing from imageTraining.js"


def test_image_training_js_syntax(tmp_path):
    src = (ROOT / "static" / "js" / "imageTraining.js").read_text(encoding="utf-8")
    mjs = tmp_path / "imageTraining.mjs"; mjs.write_text(src, encoding="utf-8")
    p = subprocess.run(["node", "--check", str(mjs)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr


def test_help_manual_has_image_training_section():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "train an SDXL LoRA from a prepared image dataset" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_image_training_ui.py -v --import-mode=importlib`
Expected: FAIL (`imageTraining.js` doesn't exist; the new HTML ids aren't in `index.html` yet)

- [ ] **Step 3: Write the HTML additions**

In `static/index.html`, insert a new modal immediately after the `imagedataset-modal`'s closing `</div>` (the line right before the `<!-- AI Studio: Dataset builder/validator — admin -->` comment that precedes `dataset-modal`):

```html
  <!-- Image AI Studio: LoRA Training GUI — admin -->
  <div id="image-training-modal" class="modal hidden">
    <div class="modal-content admin-modal-content" role="dialog" aria-label="Image LoRA training">
      <div class="modal-header">
        <h4>Image LoRA Training</h4>
        <button class="close-btn" id="imgtrain-close" aria-label="Close">&#x2716;</button>
      </div>
      <div class="admin-card" id="imgtrain-env-card">
        <div>Environment: <b id="imgtrain-env-status">checking…</b></div>
        <button id="imgtrain-env-setup" class="btn">Set up training environment (reuses the text-training venv; adds diffusers)</button>
        <div id="imgtrain-env-progress" style="font-size:12px;opacity:0.75;white-space:pre-wrap;"></div>
      </div>
      <div class="admin-card" id="imgtrain-run-card">
        <label>Dataset<br><input id="imgtrain-dataset" list="imgtrain-dataset-suggestions" placeholder="my-image-dataset" style="width:100%"><datalist id="imgtrain-dataset-suggestions"></datalist></label>
        <label>Output name<br><input id="imgtrain-output-name" placeholder="my-lora" style="width:100%"></label>
        <details><summary>Advanced</summary>
          <label>LoRA rank <input id="imgtrain-rank" type="number" value="4" style="width:70px"></label>
          <label>LoRA alpha <input id="imgtrain-alpha" type="number" value="4" style="width:70px"></label>
          <label>Learning rate <input id="imgtrain-lr" type="number" value="0.0001" step="0.0001" style="width:90px"></label>
          <label>Steps <input id="imgtrain-steps" type="number" value="1000" style="width:90px"></label>
          <label>Resolution <input id="imgtrain-resolution" type="number" value="1024" style="width:90px"></label>
        </details>
        <div style="font-size:12px;opacity:0.75;margin-top:6px">Results save only when the run finishes — stopping or closing the app mid-run does not keep partial progress.</div>
        <div style="margin-top:8px">
          <button id="imgtrain-start" class="btn">Start training</button>
          <button id="imgtrain-stop" class="btn">Stop</button>
        </div>
        <div id="imgtrain-progress" style="font-size:13px;margin-top:8px"></div>
      </div>
    </div>
  </div>

```

Insert a new rail button immediately after the existing `rail-imagedataset` button:

```html
    <button class="icon-rail-btn" id="rail-imagetraining" title="Image Training" style="display:none"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v3m0 12v3M3 12h3m12 0h3"/><circle cx="12" cy="12" r="4"/></svg></button>
```

Insert a new sidebar entry immediately after the existing `tool-imagedataset-btn` block:

```html
        <div class="list-item" id="tool-imagetraining-btn" style="display:none">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.5;"><path d="M12 3v3m0 12v3M3 12h3m12 0h3"/><circle cx="12" cy="12" r="4"/></svg>
          <span class="grow">Image Training</span>
        </div>
```

Insert a new script tag immediately after the existing `imageDataset.js` script tag:

```html
<script type="module" src="/static/js/imageTraining.js"></script>
```

Insert a new Help `<details>` entry immediately after the existing "Image Dataset (Image AI Studio)" `</details>` block:

```html
          <details>
            <summary style="cursor:pointer;font-weight:600;padding:6px 0;">Image LoRA Training (Image AI Studio)</summary>
            <p>Open <b>Image Training</b> from the sidebar (admin) to train an SDXL LoRA from a prepared image dataset. Pick a dataset saved from the <b>Image Dataset</b> panel, give it an <b>output name</b>, and click <b>Start training</b> — the first run also needs a one-time <b>environment setup</b> (adds the <code>diffusers</code> library to the training venv). Watch live step/loss/VRAM in the progress line. Results only save when the run finishes; once it's done, find your new LoRA in the <b>LoRA manager</b> (Image models card) to use it in image generation.</p>
          </details>
```

- [ ] **Step 4: Write the DOM controller**

```javascript
// static/js/imageTraining.js
// Image LoRA Training panel. ES module — DOM controller over the admin-gated
// /api/image-training engine. Admin-only: the rail button stays hidden unless
// /api/auth/status reports is_admin. Mirrors training.js (Modals, $, api, isAdmin).
import * as Modals from './modalManager.js';
import { formToConfig, renderStatusLine } from './imageTrainingCore.js';

function $(id) { return document.getElementById(id); }
let pollTimer = null;

async function api(path, opts) {
  const res = await fetch(path, Object.assign({ credentials: 'same-origin' }, opts || {}));
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error((d && d.errors ? d.errors.join('; ') : d) || String(res.status));
  }
  return data;
}

async function isAdmin() {
  try {
    const d = await (await fetch('/api/auth/status', { credentials: 'same-origin' })).json();
    return !!d.is_admin;
  } catch (e) { return false; }
}

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

function openImageTraining() {
  $('image-training-modal').classList.remove('hidden');
  refreshEnv(); refreshDatasetList(); resumeIfRunning();
}
function closeImageTraining() { $('image-training-modal').classList.add('hidden'); stopPolling(); }

async function refreshEnv() {
  try {
    const j = await api('/api/image-training/env');
    $('imgtrain-env-status').textContent = j.status || 'unknown';
    const ready = j.status === 'ready';
    $('imgtrain-run-card').style.opacity = ready ? '1' : '0.5';
    $('imgtrain-env-setup').style.display = ready ? 'none' : '';
    const start = $('imgtrain-start'); if (start) start.disabled = !ready;
  } catch (e) {
    $('imgtrain-env-status').textContent = 'error';
    const start = $('imgtrain-start'); if (start) start.disabled = true;
  }
}

async function setupEnv() {
  const p = $('imgtrain-env-progress');
  if (p) p.textContent = 'Setting up (adds diffusers to the training venv)…';
  $('imgtrain-env-setup').disabled = true;
  try {
    const j = await api('/api/image-training/env/setup', { method: 'POST' });
    if (p) p.textContent = j.ready ? 'Ready.' : ('Failed: ' + (j.error || 'unknown'));
  } catch (e) { if (p) p.textContent = 'Failed: ' + e.message; }
  $('imgtrain-env-setup').disabled = false; refreshEnv();
}

function collectConfig() {
  return formToConfig({
    dataset_name: $('imgtrain-dataset').value, output_name: $('imgtrain-output-name').value,
    rank: $('imgtrain-rank').value, lora_alpha: $('imgtrain-alpha').value,
    learning_rate: $('imgtrain-lr').value, steps: $('imgtrain-steps').value,
    resolution: $('imgtrain-resolution').value,
  });
}

async function startRun() {
  const prog = $('imgtrain-progress'); if (prog) prog.textContent = 'Starting…';
  try {
    await api('/api/image-training/runs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectConfig()),
    });
    if (prog) prog.textContent = 'Started.';
    startPolling();
  } catch (e) { if (prog) prog.textContent = 'Error: ' + e.message; }
}

async function stopRun() {
  const prog = $('imgtrain-progress'); if (prog) prog.textContent = 'Stopping…';
  try {
    await api('/api/image-training/runs/stop', { method: 'POST' });
    if (prog) prog.textContent = 'Stopped.';
  } catch (e) { if (prog) prog.textContent = 'Stop failed: ' + e.message; }
}

function startPolling() { stopPolling(); pollTimer = setInterval(pollStatus, 1500); pollStatus(); }
function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

function renderStatus(s) {
  const prog = $('imgtrain-progress'); if (!prog) return;
  let line = renderStatusLine(s);
  if (s.status === 'done') line += ' — find it in the LoRA manager (Image models card)';
  prog.textContent = line;
}

async function pollStatus() {
  try {
    const s = await api('/api/image-training/runs/current');
    renderStatus(s);
    if (s.status === 'done' || s.status === 'error' || s.status === 'stopped') stopPolling();
  } catch (e) {}
}

// On (re)open, show the current run's status and resume polling if it's live.
// startPolling() calls stopPolling() first, so this is idempotent with any timer.
async function resumeIfRunning() {
  try {
    const s = await api('/api/image-training/runs/current');
    if (s.status && s.status !== 'idle') renderStatus(s);
    if (s.status === 'running') startPolling();
  } catch (e) {}
}

async function refreshDatasetList() {
  try {
    const j = await api('/api/image-datasets');
    const dl = $('imgtrain-dataset-suggestions');
    if (dl) dl.innerHTML = (j.datasets || []).map(function (d) {
      return '<option value="' + esc(d.name) + '"></option>';
    }).join('');
  } catch (e) {}
}

function init() {
  // Admin-only: reveal BOTH the icon-rail button and the sidebar Tools entry.
  isAdmin().then(function (ok) {
    if (!ok) return;
    ['rail-imagetraining', 'tool-imagetraining-btn'].forEach(function (id) {
      const b = $(id); if (b) b.style.display = '';
    });
  });
  const rail = $('rail-imagetraining'); if (rail) rail.addEventListener('click', openImageTraining);
  const side = $('tool-imagetraining-btn'); if (side) side.addEventListener('click', openImageTraining);
  const x = $('imgtrain-close'); if (x) x.addEventListener('click', closeImageTraining);
  const setup = $('imgtrain-env-setup'); if (setup) setup.addEventListener('click', setupEnv);
  const start = $('imgtrain-start'); if (start) start.addEventListener('click', startRun);
  const stop = $('imgtrain-stop'); if (stop) stop.addEventListener('click', stopRun);
  Modals.register('image-training-modal', {
    railBtnId: 'rail-imagetraining', sidebarBtnId: 'tool-imagetraining-btn', closeFn: closeImageTraining,
  });
}

document.addEventListener('DOMContentLoaded', init);
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_image_training_ui.py -v --import-mode=importlib`
Expected: PASS (4 tests)

Then run Task 1's tests again to confirm nothing regressed:

Run: `python -m pytest tests/test_image_training_core_js.py tests/test_image_training_ui.py -v --import-mode=importlib`
Expected: PASS (10 tests)

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/imageTraining.js tests/test_image_training_ui.py
git commit -m "feat(image-training-gui): add training modal, DOM controller, sidebar wiring"
```

---

## Self-Review Notes

**Spec coverage:** The spec's "Architecture" section (`imageTrainingCore.js` + `imageTraining.js`) maps to Task 1 and Task 2's Step 4 respectively. "HTML additions" maps to Task 2's Step 3, with every named id present. "Data flow" is realized by `openImageTraining`'s `refreshEnv`/`refreshDatasetList`/`resumeIfRunning` sequence and `startRun`'s `startPolling` call. "Error handling" (never an unhandled rejection, background refreshes swallow failures) matches every `catch` block in `imageTraining.js`. "Testing" maps to Task 1's and Task 2's test files, mirroring `test_training_core_js.py` and `test_image_dataset_ui.py` exactly as specified. "Non-goals" (no checkpointing, no adapters UI, no serving UI, no in-panel dataset browsing) are honored — no task touches any of them.

**Placeholder scan:** No TBD/TODO; every step has literal, runnable code; no "similar to Task N" references — full code given in both tasks.

**Type consistency:** `formToConfig`'s output keys (`dataset_name`, `output_name`, `rank`, `lora_alpha`, `learning_rate`, `steps`, `resolution`) match exactly what `collectConfig()` passes through to it in Task 2, and match the field names `ImageTrainingConfig`/`routes/image_training_routes.py` (already shipped) expect. `renderStatusLine`'s expected input fields (`status`, `last_step`, `loss`, `vram_gb`, `lora_path`, `error`) match the shape `ImageTrainingManager.status()` (already shipped) actually returns. HTML ids referenced in `imageTraining.js` (`imgtrain-env-status`, `imgtrain-run-card`, `imgtrain-env-setup`, `imgtrain-start`, `imgtrain-dataset`, `imgtrain-output-name`, `imgtrain-rank`, `imgtrain-alpha`, `imgtrain-lr`, `imgtrain-steps`, `imgtrain-resolution`, `imgtrain-progress`, `imgtrain-dataset-suggestions`, `imgtrain-close`, `rail-imagetraining`, `tool-imagetraining-btn`) all appear in Task 2's Step 3 HTML block.

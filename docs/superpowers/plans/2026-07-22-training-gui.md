# Training GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An admin-only in-app **Training** panel (modal opened from the sidebar icon-rail) that drives the already-shipped `/api/training/*` engine: set up the training environment, configure and launch a QLoRA run, watch live progress, and list produced adapters.

**Architecture:** Pure frontend, no backend change. A pure ES-module core (`trainingCore.js`) does form→config mapping + the VRAM-fit hint (unit-tested via a node subprocess); a DOM controller (`training.js`) renders the modal, gates the rail button on `is_admin`, and polls the run status — mirroring the existing `static/js/workflows.js` admin-modal pattern exactly.

**Tech Stack:** Vanilla ES modules (`<script type="module">`), the shared `modalManager.js`, the shipped `/api/training/*` + `/api/hwfit/usage` + `/api/auth/status` routes. Frontend-only; Python is touched only for an unrelated Help-manual section.

## Global Constraints

- **Frontend-only. NO backend change.** The `/api/training/*` routes shipped in the training-engine sub-project (commits `cc05d1a..3632a92`); this GUI consumes them unchanged. Do not modify `routes/`, `src/training/`, `training_sidecar/`, or `app.py`.
- **Admin-only.** The rail button `#rail-training` ships `style="display:none"` and is revealed only when `/api/auth/status` returns `is_admin` (the API itself is admin-gated server-side). Mirror `workflows.js`'s `isAdmin()` gate.
- **Mirror the `workflows.js` pattern exactly:** `import * as Modals from './modalManager.js'`; `$ = (id) => document.getElementById(id)`; an `api(path, opts)` fetch helper (`credentials:'same-origin'`, JSON, throw on `!res.ok` reading `data.detail`); `open/close` toggling `.classList` `hidden`; `init()` on `DOMContentLoaded` that reveals the rail button for admins, wires buttons, and calls `Modals.register('training-modal', {railBtnId:'rail-training', closeFn})`.
- **JS-core tests run via a node subprocess** and read output with `encoding="utf-8"` (Windows cp1252 default corrupts non-ASCII). The DOM module (`training.js`) gets a `node --check` syntax gate on a temp `.mjs` copy. **There is NO automated UI test — the visual panel is manual-GUI verification owed by the user** (Task 3 ships the runbook).
- **Escape server-provided strings** before `innerHTML` (adapter `run_id`/`base_model`).
- pytest `--import-mode=importlib`. Commit directly to `dev`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Stage only the files each task names — never `git add -A`; never stage `installer/Output/Assist-Setup.exe`, `assistappicon.png`, `assistlogo.png`, `build_assets/uv/uv.exe`.
- Node.js must be on PATH (used by the JS tests). ~220 unrelated pre-existing test failures exist elsewhere — run only the test files each task names.

### The shipped route contract (consumed by this GUI — do not change, match exactly)

- `GET /api/training/env` → `{"status": "not_installed" | "ready" | "error", "error"?}`
- `POST /api/training/env/setup` → `{"ready": bool, "error": str|null}` (long — multi-GB install; the route runs it off the event loop)
- `POST /api/training/runs` body `{base_model, dataset_path, steps|epochs, lora_r?, lora_alpha?, lora_dropout?, batch_size?, learning_rate?, max_seq_length?}` → `{"started": true, "run_id", "warning": str|null}` OR HTTP 400 `{"detail": "<msg>"}`
- `GET /api/training/runs/current` → `{"status": "idle"|"running"|"done"|"error"|"stopped", "last_step", "loss", "vram_gb", "peak_vram_gb", "error", "output_dir", "run_id"?}`
- `POST /api/training/runs/stop` → `{"stopped": bool}`
- `GET /api/training/adapters` → `{"adapters": [{"run_id", "complete": bool, "base_model": str|null, "path"}]}`
- `GET /api/hwfit/usage` → `{... "gpus": [{"index","name","vram_used_gb","vram_total_gb","vram_percent","util_percent"}]}` (free VRAM = `vram_total_gb - vram_used_gb`)
- `GET /api/auth/status` → `{... "is_admin": bool}`

---

### Task 1: `trainingCore.js` — pure form→config + VRAM hint

**Files:**
- Create: `static/js/trainingCore.js`
- Test: `tests/test_training_core_js.py`

**Interfaces:**
- Produces (ESM exports): `parseParamsB(modelId) -> number|null`; `estimateVramGb(paramsB) -> number`; `vramHint(paramsB, freeGib) -> {level, label}` (`level` ∈ `"fits"|"tight"|"too_big"|"unknown"`); `formToConfig(values) -> object` (the `/api/training/runs` body; steps XOR epochs). Mirrors `src/training/config.py` (`_FIXED_OVERHEAD_GB=1.0`, `_PER_B_GB=0.8`, the same fit thresholds).

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_core_js.py`:

```python
import json
import subprocess
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORE = ROOT / "static" / "js" / "trainingCore.js"


def _node(expr):
    script = f"import * as m from {json.dumps(CORE.as_uri())};\n{expr}"
    p = subprocess.run(["node", "--input-type=module", "-e", script],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_parse_params_b():
    out = _node("console.log(JSON.stringify(["
                "m.parseParamsB('Qwen/Qwen2.5-0.5B-Instruct'),"
                "m.parseParamsB('meta-llama/Meta-Llama-3-8B'),"
                "m.parseParamsB('openai-community/gpt2')]));")
    assert json.loads(out) == [0.5, 8, None]


def test_vram_hint_levels():
    out = _node("console.log(JSON.stringify(["
                "m.vramHint(0.5, 6.44).level, m.vramHint(13, 6.44).level, "
                "m.vramHint(0.5, null).level]));")
    assert json.loads(out) == ["fits", "too_big", "unknown"]


def test_form_to_config_steps_xor_epochs():
    out = _node("console.log(JSON.stringify(m.formToConfig("
                "{base_model:'x', dataset_path:'d.jsonl', mode:'steps', steps:'50', "
                "epochs:'3', lora_r:'8', batch_size:'1', learning_rate:'0.0002'})));")
    cfg = json.loads(out)
    assert cfg["steps"] == 50 and cfg.get("epochs") is None
    assert cfg["base_model"] == "x" and cfg["lora_r"] == 8


def test_form_to_config_epochs_mode():
    out = _node("console.log(JSON.stringify(m.formToConfig("
                "{base_model:'x', dataset_path:'d.jsonl', mode:'epochs', steps:'50', epochs:'3'})));")
    cfg = json.loads(out)
    assert cfg["epochs"] == 3 and cfg.get("steps") is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_training_core_js.py --import-mode=importlib -q`
Expected: FAIL (module file missing → node error / assertion).

- [ ] **Step 3: Implement**

Create `static/js/trainingCore.js`:

```javascript
// Pure helpers for the Training panel — no DOM, no fetch. Mirrors
// src/training/config.py (estimate_vram_gb / fit_level / parse_params_b) so the
// client-side hint matches the server's soft VRAM gate.
export function parseParamsB(modelId) {
  if (typeof modelId !== 'string') return null;
  const m = modelId.replace(/_/g, '-').match(/(\d+(?:\.\d+)?)\s*[bB]\b/);
  return m ? parseFloat(m[1]) : null;
}

const FIXED = 1.0, PER_B = 0.8;
export function estimateVramGb(paramsB) {
  const pb = Number(paramsB);
  if (!isFinite(pb)) return FIXED;
  return Math.round((FIXED + PER_B * Math.max(pb, 0)) * 100) / 100;
}

export function vramHint(paramsB, freeGib) {
  if (freeGib == null || paramsB == null) return { level: 'unknown', label: 'VRAM: unknown' };
  const est = estimateVramGb(paramsB);
  let level = 'too_big';
  if (est <= 0.8 * freeGib) level = 'fits';
  else if (est <= freeGib) level = 'tight';
  const note = level === 'fits' ? 'fits' : level === 'tight' ? 'tight — may OOM' : 'likely too big';
  return { level, label: '~' + est + ' GB of ' + freeGib + ' GB — ' + note };
}

export function formToConfig(v) {
  const cfg = {
    base_model: (v.base_model || '').trim(),
    dataset_path: (v.dataset_path || '').trim(),
    lora_r: parseInt(v.lora_r, 10) || 8,
    lora_alpha: parseInt(v.lora_alpha, 10) || 16,
    lora_dropout: (v.lora_dropout != null && v.lora_dropout !== '') ? parseFloat(v.lora_dropout) : 0.05,
    batch_size: parseInt(v.batch_size, 10) || 1,
    learning_rate: parseFloat(v.learning_rate) || 2e-4,
    max_seq_length: parseInt(v.max_seq_length, 10) || 512,
  };
  if (v.mode === 'epochs') { cfg.epochs = parseFloat(v.epochs) || 1; cfg.steps = null; }
  else { cfg.steps = parseInt(v.steps, 10) || 100; cfg.epochs = null; }
  return cfg;
}
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_training_core_js.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add static/js/trainingCore.js tests/test_training_core_js.py
git commit -m "feat(training-gui): pure JS core (form->config, VRAM hint)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Training modal + rail button + `training.js` controller

**Files:**
- Modify: `static/index.html` (add the Training modal, the `#rail-training` icon-rail button, a `modulepreload` for `trainingCore.js`, and the `training.js` module script)
- Create: `static/js/training.js`
- Test: `tests/test_training_ui.py`

**Interfaces:**
- Consumes: `static/js/trainingCore.js` (Task 1) exports `formToConfig`, `vramHint`, `parseParamsB`; the shipped `/api/training/*`, `/api/hwfit/usage`, `/api/auth/status` routes; `modalManager.js` `Modals.register(id, {railBtnId, closeFn})`.
- Produces: a `#training-modal`, a `#rail-training` opener (admin-revealed), and `static/js/training.js` wiring env-setup, the run form, live progress polling, and the adapters list.

- [ ] **Step 1: Write the failing test** (text-guard + `node --check` syntax gate)

Create `tests/test_training_ui.py`:

```python
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_index_has_training_modal_and_scripts():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="training-modal"', 'id="rail-training"',
               '/static/js/training.js', '/static/js/trainingCore.js'):
        assert el in html, f"{el} missing from index.html"


def test_training_js_references_shipped_routes():
    src = (ROOT / "static" / "js" / "training.js").read_text(encoding="utf-8")
    for route in ('/api/training/env', '/api/training/env/setup', '/api/training/runs',
                  '/api/training/runs/current', '/api/training/runs/stop',
                  '/api/training/adapters', '/api/auth/status'):
        assert route in src, f"{route} not referenced in training.js"


def test_training_js_syntax_ok(tmp_path):
    src = (ROOT / "static" / "js" / "training.js").read_text(encoding="utf-8")
    mjs = tmp_path / "training.mjs"
    mjs.write_text(src, encoding="utf-8")
    p = subprocess.run(["node", "--check", str(mjs)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_training_ui.py --import-mode=importlib -q`
Expected: FAIL (elements + `training.js` missing).

- [ ] **Step 3: Create `static/js/training.js`**

Create `static/js/training.js`:

```javascript
// Training panel (LoRA/QLoRA). ES module — DOM controller over the admin-gated
// /api/training engine. Admin-only: the rail button stays hidden unless
// /api/auth/status reports is_admin. Mirrors workflows.js (Modals, $, api, isAdmin).
import * as Modals from './modalManager.js';
import { formToConfig, vramHint, parseParamsB } from './trainingCore.js';

function $(id) { return document.getElementById(id); }
let pollTimer = null;
let freeGib = null;

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

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

function openTraining() { $('training-modal').classList.remove('hidden'); refreshEnv(); refreshAdapters(); loadFreeVram(); }
function closeTraining() { $('training-modal').classList.add('hidden'); stopPolling(); }

async function loadFreeVram() {
  try {
    const j = await api('/api/hwfit/usage');
    const g = j.gpus && j.gpus[0];
    if (g && g.vram_total_gb != null && g.vram_used_gb != null) {
      freeGib = Math.round((g.vram_total_gb - g.vram_used_gb) * 100) / 100;
    } else { freeGib = null; }
  } catch (e) { freeGib = null; }
  updateHint();
}

function updateHint() {
  const model = $('training-model');
  const h = vramHint(parseParamsB(model ? model.value : ''), freeGib);
  const el = $('training-vram-hint'); if (el) el.textContent = h.label;
}

async function refreshEnv() {
  try {
    const j = await api('/api/training/env');
    $('training-env-status').textContent = j.status || 'unknown';
    const ready = j.status === 'ready';
    $('training-run-card').style.opacity = ready ? '1' : '0.5';
    $('training-env-setup').style.display = ready ? 'none' : '';
  } catch (e) { $('training-env-status').textContent = 'error'; }
}

async function setupEnv() {
  const p = $('training-env-progress');
  if (p) p.textContent = 'Setting up (downloads several GB, one time)…';
  $('training-env-setup').disabled = true;
  try {
    const j = await api('/api/training/env/setup', { method: 'POST' });
    if (p) p.textContent = j.ready ? 'Ready.' : ('Failed: ' + (j.error || 'unknown'));
  } catch (e) { if (p) p.textContent = 'Failed: ' + e.message; }
  $('training-env-setup').disabled = false; refreshEnv();
}

function collectConfig() {
  return formToConfig({
    base_model: $('training-model').value, dataset_path: $('training-dataset').value,
    mode: $('training-mode').value, steps: $('training-steps').value, epochs: $('training-epochs').value,
    lora_r: $('training-r').value, lora_alpha: $('training-alpha').value, lora_dropout: $('training-dropout').value,
    batch_size: $('training-batch').value, learning_rate: $('training-lr').value,
  });
}

async function startRun() {
  const prog = $('training-progress'); if (prog) prog.textContent = 'Starting…';
  try {
    const j = await api('/api/training/runs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectConfig()),
    });
    if (prog) prog.textContent = j.warning ? ('Started — ' + j.warning) : 'Started.';
    startPolling();
  } catch (e) { if (prog) prog.textContent = 'Error: ' + e.message; }
}

async function stopRun() { try { await api('/api/training/runs/stop', { method: 'POST' }); } catch (e) {} }

function startPolling() { stopPolling(); pollTimer = setInterval(pollStatus, 1500); pollStatus(); }
function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

async function pollStatus() {
  try {
    const s = await api('/api/training/runs/current');
    const bits = ['status: ' + s.status];
    if (s.last_step != null) bits.push('step ' + s.last_step);
    if (s.loss != null) bits.push('loss ' + s.loss);
    if (s.vram_gb != null) bits.push('vram ' + s.vram_gb + ' GB');
    if (s.error) bits.push('(' + s.error + ')');
    const prog = $('training-progress'); if (prog) prog.textContent = bits.join(' · ');
    if (s.status === 'done' || s.status === 'error' || s.status === 'stopped') { stopPolling(); refreshAdapters(); }
  } catch (e) {}
}

async function refreshAdapters() {
  try {
    const j = await api('/api/training/adapters');
    const host = $('training-adapters');
    if (host) {
      host.innerHTML = (j.adapters || []).map(function (a) {
        return '<div>' + (a.complete ? '✅' : '⏳') + ' ' + esc(a.run_id) + ' — ' + esc(a.base_model || '?') + '</div>';
      }).join('') || 'None yet.';
    }
  } catch (e) {}
}

function init() {
  isAdmin().then(function (ok) { const b = $('rail-training'); if (b && ok) b.style.display = ''; });
  const rail = $('rail-training'); if (rail) rail.addEventListener('click', openTraining);
  const x = $('training-close'); if (x) x.addEventListener('click', closeTraining);
  const setup = $('training-env-setup'); if (setup) setup.addEventListener('click', setupEnv);
  const start = $('training-start'); if (start) start.addEventListener('click', startRun);
  const stop = $('training-stop'); if (stop) stop.addEventListener('click', stopRun);
  const model = $('training-model'); if (model) model.addEventListener('input', updateHint);
  Modals.register('training-modal', { railBtnId: 'rail-training', closeFn: closeTraining });
}

document.addEventListener('DOMContentLoaded', init);
```

- [ ] **Step 4: Add the modal to `static/index.html`**

Find the line `<div id="workflows-modal" class="modal hidden">` and insert this block IMMEDIATELY BEFORE it (modal order is irrelevant; this is a stable anchor):

```html
  <!-- Local Training (LoRA/QLoRA) — admin -->
  <div id="training-modal" class="modal hidden">
    <div class="modal-content admin-modal-content" role="dialog" aria-label="Training">
      <div class="modal-header">
        <h4>Training (LoRA / QLoRA)</h4>
        <button class="close-btn" id="training-close" aria-label="Close training">&#x2716;</button>
      </div>
      <div class="admin-card" id="training-env-card">
        <div>Environment: <b id="training-env-status">checking…</b></div>
        <button id="training-env-setup" class="btn">Set up training environment (one-time ~3–4 GB)</button>
        <div id="training-env-progress" style="font-size:12px;opacity:0.75;white-space:pre-wrap;"></div>
      </div>
      <div class="admin-card" id="training-run-card">
        <label>Base model (HuggingFace id)<br><input id="training-model" list="training-model-suggestions" placeholder="Qwen/Qwen2.5-0.5B-Instruct" style="width:100%"></label>
        <datalist id="training-model-suggestions">
          <option value="Qwen/Qwen2.5-0.5B-Instruct"></option>
          <option value="Qwen/Qwen2.5-1.5B-Instruct"></option>
          <option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0"></option>
          <option value="meta-llama/Llama-3.2-1B-Instruct"></option>
        </datalist>
        <div id="training-vram-hint" style="font-size:12px;opacity:0.8;margin:4px 0;"></div>
        <label>Dataset (.jsonl file path)<br><input id="training-dataset" placeholder="C:\path\to\data.jsonl" style="width:100%"></label>
        <details><summary>Advanced</summary>
          <label>Mode <select id="training-mode"><option value="steps">steps</option><option value="epochs">epochs</option></select></label>
          <label>Steps <input id="training-steps" type="number" value="100" style="width:90px"></label>
          <label>Epochs <input id="training-epochs" type="number" value="3" step="0.5" style="width:90px"></label>
          <label>LoRA r <input id="training-r" type="number" value="8" style="width:70px"></label>
          <label>LoRA alpha <input id="training-alpha" type="number" value="16" style="width:70px"></label>
          <label>Dropout <input id="training-dropout" type="number" value="0.05" step="0.01" style="width:70px"></label>
          <label>Batch <input id="training-batch" type="number" value="1" style="width:70px"></label>
          <label>LR <input id="training-lr" type="number" value="0.0002" step="0.0001" style="width:90px"></label>
        </details>
        <div style="margin-top:8px">
          <button id="training-start" class="btn">Start training</button>
          <button id="training-stop" class="btn">Stop</button>
        </div>
        <div id="training-progress" style="font-size:13px;margin-top:8px"></div>
      </div>
      <div class="admin-card">
        <div style="font-weight:600">Adapters</div>
        <div id="training-adapters"></div>
      </div>
    </div>
  </div>
```

- [ ] **Step 5: Add the icon-rail button + the script/preload to `static/index.html`**

Find the `id="rail-workflows"` button line (a single-line `<button …>…</button>`) and insert this button IMMEDIATELY AFTER it:

```html
    <button class="icon-rail-btn" id="rail-training" title="Training" style="display:none"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v3m0 12v3M3 12h3m12 0h3"/><circle cx="12" cy="12" r="4"/></svg></button>
```

Find the line `<link rel="modulepreload" href="/static/js/markdown.js">` and insert immediately after it:

```html
  <link rel="modulepreload" href="/static/js/trainingCore.js">
```

Find the line `<script type="module" src="/static/js/workflows.js"></script>` and insert immediately after it:

```html
<script type="module" src="/static/js/training.js"></script>
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_training_ui.py tests/test_training_core_js.py --import-mode=importlib -q`
Expected: PASS (3 + 4 = 7 passed). The `node --check` gate confirms `training.js` parses as an ES module.

- [ ] **Step 7: Commit**

```bash
git add static/index.html static/js/training.js tests/test_training_ui.py
git commit -m "feat(training-gui): admin Training modal + controller (env, run, progress, adapters)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Finalize — Help-manual section + manual-verification runbook

**Files:**
- Modify: `static/index.html` (Help-manual "Training" section)
- Create: `docs/training-gui-manual-verification.md`
- Test: `tests/test_training_ui.py` (extend), and confirm `tests/test_help_ui.py` still passes

**Interfaces:**
- Consumes: the panel from Task 2.

- [ ] **Step 1: Add a Help-manual entry test**

Append to `tests/test_training_ui.py`:

```python
def test_help_manual_has_training_section():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    # a phrase unique to the Help-manual paragraph (NOT present in Task 2's modal),
    # so this test genuinely gates the Help section rather than the modal.
    assert "fine-tune a small model on your own GPU" in html
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_training_ui.py::test_help_manual_has_training_section --import-mode=importlib -q`
Expected: FAIL (the Help paragraph phrase isn't in `index.html` yet — Task 2's modal uses different copy; the Help `<details>` block added in Step 3 introduces this exact phrase).

- [ ] **Step 3: Add the Help-manual section**

In `static/index.html`, find the Help-manual `Workflows (visual automation)` `<details>` block (search for `Workflows (visual automation)`) and insert this sibling `<details>` immediately after that block closes:

```html
          <details>
            <summary style="cursor:pointer;font-weight:600;padding:6px 0;">Training (LoRA / QLoRA)</summary>
            <p>Open <b>Training</b> from the sidebar (admin only) to fine-tune a small model on your own GPU. The first time, click <b>Set up training environment</b> — Assist downloads a one-time (~3–4 GB) CUDA training toolkit. Then pick a small <b>base model</b> (a HuggingFace id, e.g. Qwen2.5-0.5B/1.5B), point it at a <b>.jsonl dataset</b> (each line <code>{"text": …}</code> or <code>{"instruction": …, "response": …}</code>), and click <b>Start training</b>. It uses <b>QLoRA</b> (4-bit) so small models fit ~6 GB; the VRAM hint shows whether your model fits. Progress (step / loss / VRAM) updates live, and the finished <b>LoRA adapter</b> appears in the Adapters list. (Serving the fine-tuned model is coming in a later update.)</p>
          </details>
```

- [ ] **Step 4: Create the manual-verification runbook**

Create `docs/training-gui-manual-verification.md`:

```markdown
# Training GUI — Manual Verification Runbook

The Training panel has no automated UI test (the run form drives real GPU work).
After a frozen rebuild, an admin verifies it by hand.

## Prerequisites
- `python scripts/fetch_uv.py` has vendored `uv` into `build_assets/uv/`, and the
  app was rebuilt (clean) so `training_sidecar/` + `uv` are bundled.
- Logged in as an admin.

## Steps
1. **Rail button (admin gate):** the Training icon appears in the sidebar icon-rail.
   Log in as a non-admin (or hit `/api/auth/status` returning `is_admin:false`) →
   the button is hidden.
2. **Open the panel:** click the Training icon → the modal opens with an
   Environment card, a run form, and an Adapters list.
3. **Env setup:** if status is `not_installed`, click **Set up training
   environment**. It runs the one-time (~3–4 GB) install; on success the status
   flips to `ready` and the setup button hides. (First run only; needs internet.)
4. **VRAM hint:** type `Qwen/Qwen2.5-0.5B-Instruct` in the base-model field → the
   hint reads roughly `~1.4 GB of <free> GB — fits`. Type a `13B` model → `likely
   too big`.
5. **Start a run:** set a `.jsonl` dataset path (a few `{"text": …}` lines),
   keep Steps small (e.g. 20), click **Start training**. The progress line updates
   live (`status: running · step N · loss … · vram … GB`).
6. **Completion:** on finish, status shows `done` and the run appears in the
   **Adapters** list with ✅ and the base model. Confirm the adapter files exist
   under `<DATA_DIR>/training/adapters/<run-id>/`.
7. **Stop:** start another run and click **Stop** → status shows `stopped` (not
   `error`).

## Owed regressions to watch
- Live `step` events must reach the progress line (guards the `disable_tqdm`
  fix — a tqdm bar on the JSON channel would blank the telemetry).
- A user **Stop** must read `stopped`, never `error`.
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_training_ui.py tests/test_help_ui.py --import-mode=importlib -q`
Expected: PASS (Task 2's 3 + the new Help test = 4 in test_training_ui.py, and test_help_ui.py unchanged/green — the new `<details>` doesn't remove any required Help string).

- [ ] **Step 6: Commit**

```bash
git add static/index.html docs/training-gui-manual-verification.md tests/test_training_ui.py
git commit -m "docs(training-gui): Help-manual Training section + manual-verification runbook

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Frontend-only.** No file under `routes/`, `src/`, `training_sidecar/`, or `app.py` changes. If a task seems to need a backend change, STOP — the routes already exist; re-check the route contract in Global Constraints.
- **Node is required** for the JS tests (`node --input-type=module` and `node --check`). If `node` isn't on PATH, that's an environment problem to surface, not a reason to weaken the tests.
- **Manual GUI verification is owed by the user** (Task 3 ships the runbook) — the automated tests prove the core mapping (Task 1) and that the module parses + references the right routes and elements (Task 2), not that the panel looks or behaves right in a browser. This lands after the training engine's own owed manual GPU run.
- **Scope:** the admin Training panel only. NO dataset upload endpoint (path input is intentional — a local desktop app), NO adapter serving / merge-to-GGUF (later sub-project), NO editing a run mid-flight.
```

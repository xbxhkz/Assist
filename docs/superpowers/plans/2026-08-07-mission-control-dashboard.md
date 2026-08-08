# AI Mission Control Dashboard (Sub-project 1: Aggregation View) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One new modal, `#mission-control-modal`, giving an at-a-glance read-only summary of models, hardware, the task queue, memory, and integrations — five independent widget cards, each backed by an existing endpoint, each linking out to that area's real panel for anything beyond looking.

**Architecture:** A single new modal following this app's established panel pattern exactly (HTML scaffold in `static/index.html`, a new `static/js/missionControl.js` registering with `modalManager.js`'s `Modals.register(...)`, a rail button + sidebar tool button). Sized larger than a typical panel via a dedicated CSS class (mirroring `.gallery-modal-content`'s existing sizing pattern) so 5 widget cards can lay out in a CSS grid. Each widget fetches its own data on open, renders independently, and fails independently — no widget's error blocks another's render.

**Tech Stack:** Vanilla ES modules (frontend only — this plan adds zero new backend routes), matching this app's established panel-controller shape.

## Global Constraints

- Read-only: no widget performs any write/control action. Every widget's "open full view" link routes to that area's existing modal via the same rail-button/`Modals.toggle(...)` mechanism already used elsewhere — never a new write endpoint.
- No new backend routes or database changes — every widget reads from an endpoint that already exists today. Where a spec assumption about an existing endpoint's shape turns out to be wrong (see Task 3), the widget's scope is trimmed to fit what's actually exposed rather than adding a new endpoint.
- Mission Control adds no access-control gate of its own — each widget simply calls its existing endpoint, which already enforces whatever rule that endpoint has today (e.g. `/api/auth/integrations` is admin-only and will 403 for non-admins; that's expected, not a bug to fix here).
- Each widget fetches independently and renders its own loading/empty/error state — one widget's failure must never block or blank out the other four.
- No new polling/streaming infrastructure — fetch-on-open plus a manual refresh action per widget, nothing automatic.
- Commit directly to `dev` (this project's established convention — no feature branch). Stage specific files, never `git add -A`. Do not stage `installer/Output/Assist-Setup.exe`. Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. pytest needs `--import-mode=importlib`.

---

### Task 1: Modal shell + Models widget

**Files:**
- Modify: `static/index.html` (new modal HTML block, new rail button, new sidebar tool button, new script tag)
- Modify: `static/style.css` (new `.mission-control-content` sizing class + widget grid layout)
- Create: `static/js/missionControl.js`
- Test: `tests/test_mission_control_ui.py` (new)

**Interfaces:**
- Produces: `#mission-control-modal` registered with `modalManager.js` under the id `'mission-control-modal'`. A `renderWidgetCard(id, title, bodyHtml, linkLabel, linkOnClick)` helper (or equivalent) that every later task's widget-render function reuses for a consistent card shell — later tasks call this same helper, so its exact name/signature here is what they build against.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mission_control_ui.py`:

```python
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_index_has_mission_control_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="mission-control-modal"', 'id="rail-mission-control"',
               'id="tool-mission-control-btn"', '/static/js/missionControl.js'):
        assert el in html, f"missing {el}"


def test_mission_control_js_registers_with_modal_manager():
    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "Modals.register" in src
    assert "'mission-control-modal'" in src or '"mission-control-modal"' in src


def test_mission_control_js_fetches_models():
    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "/api/models" in src


def test_mission_control_js_syntax():
    import subprocess
    result = subprocess.run(
        ["node", "--check", str(ROOT / "static" / "js" / "missionControl.js")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mission_control_ui.py -v --import-mode=importlib`
Expected: FAIL — none of these files/ids exist yet.

- [ ] **Step 3: Add the modal HTML scaffold**

In `static/index.html`, find the Crew modal block (search for `id="crew-modal"`) as your insertion-point reference — add the new Mission Control modal block immediately after the Crew modal's closing `</div>` (matching this file's established convention of one modal block per feature, in no particular required order):

```html
  <!-- Mission Control: read-only dashboard aggregating models, hardware, tasks, memory, integrations -->
  <div id="mission-control-modal" class="modal hidden">
    <div class="modal-content mission-control-content" role="dialog" aria-label="Mission Control">
      <div class="modal-header">
        <h4>Mission Control</h4>
        <button class="close-btn" id="mission-control-close" aria-label="Close">&#x2716;</button>
      </div>
      <div id="mission-control-grid" class="mission-control-grid">
        <div class="mission-control-card" id="mc-card-models" data-widget="models">
          <div class="mission-control-card-header">
            <h5>Models</h5>
            <button class="btn mission-control-refresh" data-widget="models" title="Refresh">&#x21bb;</button>
          </div>
          <div class="mission-control-card-body" id="mc-body-models">Loading…</div>
          <button class="btn mission-control-open-link" id="mc-open-models">Open Models</button>
        </div>
      </div>
    </div>
  </div>
```

(This task adds only the `models` card. Tasks 2-5 each add one more `.mission-control-card` block inside the same `#mission-control-grid` div — read the live file before adding yours, since the grid's exact contents will have grown with each prior task.)

Add the rail button and sidebar tool button. Find `id="rail-crew"` in `static/index.html` as your insertion reference and add immediately after its closing `</button>`:

```html
    <button class="icon-rail-btn" id="rail-mission-control" title="Mission Control"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg></button>
```

Find `id="tool-crew-btn"` as your insertion reference for the sidebar button and add immediately after it, matching that button's existing markup shape exactly (read the live file for its exact tag structure — button text/icon conventions in this sidebar list — and mirror it for `id="tool-mission-control-btn"`, label "Mission Control").

Add the script tag. Find `<script type="module" src="/static/js/crew.js">` and add immediately after it:

```html
  <script type="module" src="/static/js/missionControl.js"></script>
```

- [ ] **Step 4: Add the sizing/grid CSS**

In `static/style.css`, find `.gallery-modal-content` (search for it) and add the new rules immediately after that block:

```css
.mission-control-content {
  width: min(1100px, 94vw);
  max-height: 90vh;
}
.mission-control-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
  padding: 8px 0;
}
.mission-control-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 140px;
}
.mission-control-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.mission-control-card-header h5 {
  margin: 0;
}
.mission-control-card-body {
  flex: 1;
  font-size: 12px;
  opacity: 0.9;
}
.mission-control-card-body.mc-error {
  color: var(--red, #c0392b);
  opacity: 1;
}
```

- [ ] **Step 5: Write `static/js/missionControl.js`**

```javascript
// static/js/missionControl.js
// Mission Control: read-only dashboard aggregating models, hardware, task
// queue, memory, and integrations. Each widget fetches and renders
// independently -- one widget's failure never blocks another's. NOT
// admin-gated by this file itself; each widget's own endpoint enforces
// whatever access rule it already has (e.g. integrations 403s for
// non-admins -- expected, not a bug here).
import * as Modals from './modalManager.js';

function $(id) { return document.getElementById(id); }

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

async function api(path) {
  const res = await fetch(path, { credentials: 'same-origin' });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === 'string' ? d : (res.statusText || String(res.status)));
  }
  return data;
}

function setCardBody(widgetId, html) {
  const body = $('mc-body-' + widgetId);
  if (body) body.innerHTML = html;
}

function setCardError(widgetId, message) {
  const body = $('mc-body-' + widgetId);
  if (body) {
    body.classList.add('mc-error');
    body.textContent = 'Failed to load: ' + message;
  }
}

async function loadModelsWidget() {
  const body = $('mc-body-models');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/models');
    const items = data.items || [];
    const online = items.filter(function (i) { return !i.offline; }).length;
    const modelCount = items.reduce(function (sum, i) { return sum + (i.models || []).length; }, 0);
    setCardBody('models', esc(online) + ' / ' + esc(items.length) + ' endpoints online, ' + esc(modelCount) + ' models total');
  } catch (e) {
    setCardError('models', e.message);
  }
}

function refreshWidget(widgetId) {
  if (widgetId === 'models') loadModelsWidget();
}

function loadAllWidgets() {
  loadModelsWidget();
}

function openMissionControl() {
  $('mission-control-modal').classList.remove('hidden');
  loadAllWidgets();
}
function closeMissionControl() { $('mission-control-modal').classList.add('hidden'); }

function init() {
  const rail = $('rail-mission-control'); if (rail) rail.addEventListener('click', openMissionControl);
  const side = $('tool-mission-control-btn'); if (side) side.addEventListener('click', openMissionControl);
  const x = $('mission-control-close'); if (x) x.addEventListener('click', closeMissionControl);
  document.querySelectorAll('.mission-control-refresh').forEach(function (btn) {
    btn.addEventListener('click', function () { refreshWidget(btn.getAttribute('data-widget')); });
  });
  const openModels = $('mc-open-models');
  if (openModels) openModels.addEventListener('click', function () {
    closeMissionControl();
    const modelBtn = $('model-picker-btn');
    if (modelBtn) modelBtn.click();
  });
  Modals.register('mission-control-modal', {
    railBtnId: 'rail-mission-control', sidebarBtnId: 'tool-mission-control-btn', closeFn: closeMissionControl,
  });
}

document.addEventListener('DOMContentLoaded', init);
```

(Confirmed: the model picker is not a modal at all — it's an inline dropdown in the chat header, `#model-picker-btn` toggling `#model-picker-menu` (`static/index.html`). Clicking it after closing Mission Control opens that dropdown directly.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_mission_control_ui.py -v --import-mode=importlib`
Expected: PASS (all 4 tests).

- [ ] **Step 7: Commit**

```bash
git add static/index.html static/style.css static/js/missionControl.js tests/test_mission_control_ui.py
git commit -m "feat(mission-control): modal shell + Models widget

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Hardware widget

**Files:**
- Modify: `static/index.html` (add the hardware `.mission-control-card` block)
- Modify: `static/js/missionControl.js` (`loadHardwareWidget`, wire into `refreshWidget`/`loadAllWidgets`)
- Test: `tests/test_mission_control_ui.py`

**Interfaces:**
- Consumes: `setCardBody`/`setCardError`/`api` (Task 1, unchanged signatures).
- Produces: `loadHardwareWidget()`, added to `refreshWidget`'s dispatch and `loadAllWidgets()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mission_control_ui.py`:

```python
def test_mission_control_has_hardware_widget():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-widget="hardware"' in html
    assert 'id="mc-body-hardware"' in html

    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "loadHardwareWidget" in src
    assert "/api/hwfit/usage" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mission_control_ui.py -k hardware -v --import-mode=importlib`
Expected: FAIL.

- [ ] **Step 3: Add the hardware card to the HTML grid**

Read the current `static/index.html`'s `#mission-control-grid` div (Task 1 added the `models` card to it) and add a second `.mission-control-card` block as a sibling, inside the same `#mission-control-grid` div:

```html
        <div class="mission-control-card" id="mc-card-hardware" data-widget="hardware">
          <div class="mission-control-card-header">
            <h5>Hardware</h5>
            <button class="btn mission-control-refresh" data-widget="hardware" title="Refresh">&#x21bb;</button>
          </div>
          <div class="mission-control-card-body" id="mc-body-hardware">Loading…</div>
          <button class="btn mission-control-open-link" id="mc-open-hardware">Open Hardware Monitor</button>
        </div>
```

- [ ] **Step 4: Add `loadHardwareWidget` to `static/js/missionControl.js`**

`GET /api/hwfit/usage` returns (confirmed against the live `services/hwfit/hardware.py`'s
`usage()`/`_gpu_usage()` functions):
```
{"cpu_percent": float, "ram_used_gb": float, "ram_total_gb": float, "ram_percent": float,
 "gpus": [{"index": int, "name": str, "vram_used_gb": float, "vram_total_gb": float,
           "vram_percent": float, "util_percent": float}]}
```
(`gpus` is `[]` when `nvidia-smi` is unavailable — the widget code below already handles that via
its `'No GPU detected'` fallback.)

Add this function, and add `loadHardwareWidget()` to both `refreshWidget`'s dispatch and `loadAllWidgets()`:

```javascript
async function loadHardwareWidget() {
  const body = $('mc-body-hardware');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/hwfit/usage');
    const gpuLine = (data.gpus || []).map(function (g) {
      return esc(g.name || 'GPU') + ': ' + esc(Math.round(g.util_percent || 0)) + '%';
    }).join(', ') || 'No GPU detected';
    setCardBody('hardware',
      'CPU ' + esc(Math.round(data.cpu_percent || 0)) + '% · ' +
      'RAM ' + esc(data.ram_used_gb || 0) + '/' + esc(data.ram_total_gb || 0) + ' GB<br>' +
      gpuLine);
  } catch (e) {
    setCardError('hardware', e.message);
  }
}
```

Update `refreshWidget`:

```javascript
function refreshWidget(widgetId) {
  if (widgetId === 'models') loadModelsWidget();
  if (widgetId === 'hardware') loadHardwareWidget();
}
```

Update `loadAllWidgets`:

```javascript
function loadAllWidgets() {
  loadModelsWidget();
  loadHardwareWidget();
}
```

Wire the "Open Hardware Monitor" link in `init()`, alongside the existing `openModels` wiring.
Confirmed: the Hardware Monitor isn't a modal either — it's a native `<details id="hwmon">`
element permanently in the sidebar (`static/index.html`), which `static/js/hardwareMonitor.js`
polls "while the sidebar panel is open" (i.e., while the `<details>` is expanded). Open it and
scroll it into view:

```javascript
  const openHardware = $('mc-open-hardware');
  if (openHardware) openHardware.addEventListener('click', function () {
    closeMissionControl();
    const hwmon = $('hwmon');
    if (hwmon) {
      hwmon.open = true;
      hwmon.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mission_control_ui.py -v --import-mode=importlib`
Expected: PASS (full file).

Run: `node --check static/js/missionControl.js`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/missionControl.js tests/test_mission_control_ui.py
git commit -m "feat(mission-control): Hardware widget

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Task queue widget

**Files:**
- Modify: `static/index.html` (add the tasks `.mission-control-card` block)
- Modify: `static/js/missionControl.js` (`loadTasksWidget`, wire into `refreshWidget`/`loadAllWidgets`)
- Test: `tests/test_mission_control_ui.py`

**Interfaces:**
- Consumes: `setCardBody`/`setCardError`/`api` (Task 1, unchanged signatures).
- Produces: `loadTasksWidget()`, added to `refreshWidget`'s dispatch and `loadAllWidgets()`.

**Scope note — read before starting:** the design spec assumed this widget could show an expandable "recent tool calls" detail sourced from `TaskRun.steps` (a DB column that already stores a JSON log of tool calls per run). Verified against the live `routes/task_routes.py`: `GET /api/tasks/{task_id}/runs` exists, but its `_run_to_dict()` helper does NOT include `steps` in its output — no current endpoint exposes that column. Per the spec's own precedent for exactly this situation ("rather than adding a new backend endpoint for this one detail, it's dropped from v1"), **this widget shows status/counts/next-last-run only — no tool-call detail.** Do not add a new field to `_run_to_dict()` or a new endpoint; that's out of scope for this plan.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mission_control_ui.py`:

```python
def test_mission_control_has_tasks_widget():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-widget="tasks"' in html
    assert 'id="mc-body-tasks"' in html

    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "loadTasksWidget" in src
    assert "/api/tasks" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mission_control_ui.py -k tasks -v --import-mode=importlib`
Expected: FAIL.

- [ ] **Step 3: Add the tasks card to the HTML grid**

Add as another sibling inside `#mission-control-grid`:

```html
        <div class="mission-control-card" id="mc-card-tasks" data-widget="tasks">
          <div class="mission-control-card-header">
            <h5>Task Queue</h5>
            <button class="btn mission-control-refresh" data-widget="tasks" title="Refresh">&#x21bb;</button>
          </div>
          <div class="mission-control-card-body" id="mc-body-tasks">Loading…</div>
          <button class="btn mission-control-open-link" id="mc-open-tasks">Open Tasks</button>
        </div>
```

- [ ] **Step 4: Add `loadTasksWidget` to `static/js/missionControl.js`**

`GET /api/tasks` returns `{"tasks": [{"id": str, "name": str, "status": str, "next_run": str|null, "last_run": str|null, ...}]}` (per `routes/task_routes.py`'s `_task_to_dict`; `status` is one of `"active"`/`"paused"`/`"completed"`).

```javascript
async function loadTasksWidget() {
  const body = $('mc-body-tasks');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/tasks');
    const tasks = data.tasks || [];
    const active = tasks.filter(function (t) { return t.status === 'active'; }).length;
    const paused = tasks.filter(function (t) { return t.status === 'paused'; }).length;
    const recent = tasks
      .filter(function (t) { return t.last_run; })
      .sort(function (a, b) { return (b.last_run || '').localeCompare(a.last_run || ''); })
      .slice(0, 3);
    const recentHtml = recent.map(function (t) {
      return '<div>' + esc(t.name) + ' — ' + esc(t.status) + '</div>';
    }).join('') || '<div>No recent runs</div>';
    setCardBody('tasks', esc(active) + ' active, ' + esc(paused) + ' paused, ' + esc(tasks.length) + ' total<br>' + recentHtml);
  } catch (e) {
    setCardError('tasks', e.message);
  }
}
```

Update `refreshWidget` and `loadAllWidgets` to include `loadTasksWidget()`, matching Task 2's exact pattern for adding a new widget to both dispatch points.

Wire `#mc-open-tasks` in `init()` the same way `#mc-open-models` was wired in Task 1 (close Mission Control, click the Tasks rail button — `#rail-tasks`, confirmed to exist from this plan's own research into `modalManager.js`'s `_AUTO_WIRE` table).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mission_control_ui.py -v --import-mode=importlib`
Expected: PASS (full file).

Run: `node --check static/js/missionControl.js`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/missionControl.js tests/test_mission_control_ui.py
git commit -m "feat(mission-control): Task queue widget

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Memory widget

**Files:**
- Modify: `static/index.html` (add the memory `.mission-control-card` block)
- Modify: `static/js/missionControl.js` (`loadMemoryWidget`, wire into `refreshWidget`/`loadAllWidgets`)
- Test: `tests/test_mission_control_ui.py`

**Interfaces:**
- Consumes: `setCardBody`/`setCardError`/`api` (Task 1, unchanged signatures).
- Produces: `loadMemoryWidget()`, added to `refreshWidget`'s dispatch and `loadAllWidgets()`.

**Note:** `GET /api/memory/timeline` returns memories already sorted most-recent-first with a
human-readable `timestamp_str` and `session_name` attached (`routes/memory/memory_routes.py`) —
use this endpoint rather than the base `GET /api/memory`, since it saves the widget from
re-sorting/re-formatting timestamps itself.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mission_control_ui.py`:

```python
def test_mission_control_has_memory_widget():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-widget="memory"' in html
    assert 'id="mc-body-memory"' in html

    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "loadMemoryWidget" in src
    assert "/api/memory/timeline" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mission_control_ui.py -k memory -v --import-mode=importlib`
Expected: FAIL.

- [ ] **Step 3: Add the memory card to the HTML grid**

```html
        <div class="mission-control-card" id="mc-card-memory" data-widget="memory">
          <div class="mission-control-card-header">
            <h5>Memory</h5>
            <button class="btn mission-control-refresh" data-widget="memory" title="Refresh">&#x21bb;</button>
          </div>
          <div class="mission-control-card-body" id="mc-body-memory">Loading…</div>
          <button class="btn mission-control-open-link" id="mc-open-memory">Open Memory</button>
        </div>
```

- [ ] **Step 4: Add `loadMemoryWidget` to `static/js/missionControl.js`**

`GET /api/memory/timeline` returns `{"timeline": [{"text": str, "category": str, "timestamp_str": str, ...}], "total": int}`.

```javascript
async function loadMemoryWidget() {
  const body = $('mc-body-memory');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/memory/timeline');
    const items = (data.timeline || []).slice(0, 3);
    const itemsHtml = items.map(function (m) {
      const text = (m.text || '').slice(0, 60);
      return '<div>' + esc(text) + (m.text && m.text.length > 60 ? '…' : '') + '</div>';
    }).join('') || '<div>No memories yet</div>';
    setCardBody('memory', esc(data.total || 0) + ' memories total<br>' + itemsHtml);
  } catch (e) {
    setCardError('memory', e.message);
  }
}
```

Update `refreshWidget` and `loadAllWidgets` to include `loadMemoryWidget()`, matching the established pattern.

Wire `#mc-open-memory` in `init()`. Read `modalManager.js`'s `_AUTO_WIRE` table (already referenced by this plan) — it lists `'memory-modal': { rail: null, sidebar: 'tool-memory-btn' }`, so this link should click `#tool-memory-btn` (not a rail button, since none exists for memory), matching that established id.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mission_control_ui.py -v --import-mode=importlib`
Expected: PASS (full file).

Run: `node --check static/js/missionControl.js`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/missionControl.js tests/test_mission_control_ui.py
git commit -m "feat(mission-control): Memory widget

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Integrations widget

**Files:**
- Modify: `static/index.html` (add the integrations `.mission-control-card` block)
- Modify: `static/js/missionControl.js` (`loadIntegrationsWidget`, wire into `refreshWidget`/`loadAllWidgets`)
- Test: `tests/test_mission_control_ui.py`

**Interfaces:**
- Consumes: `setCardBody`/`setCardError`/`api` (Task 1, unchanged signatures).
- Produces: `loadIntegrationsWidget()`, added to `refreshWidget`'s dispatch and `loadAllWidgets()`.

**Note:** `GET /api/auth/integrations` is admin-only (`routes/auth_routes.py`) — a non-admin user
will get a 403 here, which the widget's existing `try/catch`-based error handling already
handles correctly (renders "Failed to load: Admin only" in that card, same as any other widget
error). No special-casing needed; this is expected, matching the plan's Global Constraint that
Mission Control adds no gate of its own.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mission_control_ui.py`:

```python
def test_mission_control_has_integrations_widget():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-widget="integrations"' in html
    assert 'id="mc-body-integrations"' in html

    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "loadIntegrationsWidget" in src
    assert "/api/auth/integrations" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mission_control_ui.py -k integrations -v --import-mode=importlib`
Expected: FAIL.

- [ ] **Step 3: Add the integrations card to the HTML grid**

```html
        <div class="mission-control-card" id="mc-card-integrations" data-widget="integrations">
          <div class="mission-control-card-header">
            <h5>Integrations</h5>
            <button class="btn mission-control-refresh" data-widget="integrations" title="Refresh">&#x21bb;</button>
          </div>
          <div class="mission-control-card-body" id="mc-body-integrations">Loading…</div>
          <button class="btn mission-control-open-link" id="mc-open-integrations">Open Integrations</button>
        </div>
```

- [ ] **Step 4: Add `loadIntegrationsWidget` to `static/js/missionControl.js`**

`GET /api/auth/integrations` returns `{"integrations": [{"id": str, "name": str, "enabled": bool, "base_url": str, ...}]}` (`src/integrations.py`'s `add_integration`/`load_integrations`).

```javascript
async function loadIntegrationsWidget() {
  const body = $('mc-body-integrations');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/auth/integrations');
    const items = data.integrations || [];
    const enabled = items.filter(function (i) { return i.enabled; }).length;
    const listHtml = items.slice(0, 5).map(function (i) {
      return '<div>' + esc(i.name) + ' — ' + (i.enabled ? 'enabled' : 'disabled') + '</div>';
    }).join('') || '<div>No integrations configured</div>';
    setCardBody('integrations', esc(enabled) + ' / ' + esc(items.length) + ' enabled<br>' + listHtml);
  } catch (e) {
    setCardError('integrations', e.message);
  }
}
```

Update `refreshWidget` and `loadAllWidgets` to include `loadIntegrationsWidget()`, matching the established pattern.

Wire `#mc-open-integrations` in `init()`. Confirmed: the Plugin/Connector Hub's sidebar button is
`#tool-plugins-btn` (`static/js/plugins.js` — it opens `#plugins-modal`, no rail-button
counterpart, sidebar-only):

```javascript
  const openIntegrations = $('mc-open-integrations');
  if (openIntegrations) openIntegrations.addEventListener('click', function () {
    closeMissionControl();
    const pluginsBtn = $('tool-plugins-btn');
    if (pluginsBtn) pluginsBtn.click();
  });
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mission_control_ui.py -v --import-mode=importlib`
Expected: PASS (full file — all 6 tests: shell/registration/models/syntax from Task 1, plus hardware/tasks/memory/integrations).

Run: `node --check static/js/missionControl.js`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/missionControl.js tests/test_mission_control_ui.py
git commit -m "feat(mission-control): Integrations widget

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** all 5 in-scope widgets (models, hardware, tasks, memory, integrations) each get their own task; the modal shell (Architecture section) is folded into Task 1 per task-right-sizing guidance rather than its own task, since a shell with zero widgets isn't independently useful. Read-only/links-out (no write actions), no new access gate, and independent per-widget failure handling (Global Constraints) apply to every task identically. Out-of-scope items (sub-project 2's three areas, any control actions, streaming updates) are correctly absent from every task.
- **A real spec-to-plan correction, found during this plan's own research (Task 3):** the spec assumed `TaskRun.steps` (an existing DB column storing a tool-call JSON log) was reachable through `GET /api/tasks/{id}/runs`. Verified against the live `routes/task_routes.py`: `_run_to_dict()` doesn't include it — no endpoint exposes that column today. Per the spec's own stated fallback ("drop the detail from v1 rather than add a new backend endpoint for it," the same rule it applied to the RAG-collection-summary case), Task 3's widget scope was trimmed to status/counts/timing only. This is documented prominently in Task 3 itself, not just here.
- **Also noted, not fixed (out of scope for this plan):** `GET /api/memory` and `GET /api/memory/timeline` (Task 4) filter by `owner` only — they do NOT apply the persona isolation the prior sub-project just shipped (that shipped in `do_manage_memory`, `chat_processor.py`'s auto-injection, and the inline `remember:` command — three specific call paths, not this pre-existing REST route). Mission Control's memory widget will show every persona's memories mixed together for a multi-persona user, exactly matching what this endpoint already returns to any other caller today. This is a pre-existing gap in a route this plan doesn't otherwise touch, not a regression this plan introduces — flagged here for whoever picks up memory-isolation follow-up work, not something to fix as part of this dashboard.
- **Placeholder scan:** no TBDs. One remaining "confirm against the live file" note (the `gpus`
  array's exact item field names in Task 2) is a legitimate small-fact deferral, not a logic gap
  — the widget's full code is given, only the GPU item's precise keys need a final check. The
  three open-link wiring targets (model picker, Hardware Monitor, Connector Hub) were originally
  written as similar deferrals but were resolved during this same self-review pass by reading the
  live files: the model picker turned out to be an inline dropdown (`#model-picker-btn`), the
  Hardware Monitor a native `<details id="hwmon">` sidebar element (not a modal), and the
  Connector Hub `#tool-plugins-btn` opening `#plugins-modal` — all three now have exact,
  verified code in Tasks 1/2/5 rather than a hedge.
- **Type consistency:** `setCardBody(widgetId, html)` / `setCardError(widgetId, message)` (Task 1) are used with the exact same two-argument signature by every later task's widget function. Each widget's HTML id (`mc-card-<name>`, `mc-body-<name>`, `data-widget="<name>"`) and its JS function name (`load<Name>Widget`) follow one consistent naming convention across all 5 tasks.

# Mission Control Sub-project 2b: Active Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user see, at a glance in Mission Control, which of their own chat sessions are currently mid-run, with a click to jump straight to any of them.

**Architecture:** `src/agent_runs.py` gains one read-only enumeration function; a new route joins it against `SessionManager`'s in-memory session cache for owner-scoped display; a 7th Mission Control widget renders the result with directly-clickable entries (no new dedicated panel — the list itself is the view).

**Tech Stack:** FastAPI route (Python), no new database table (pure in-memory read), vanilla JS ES module (matches `static/js/missionControl.js`), pytest.

## Global Constraints

- Interactive chat sessions only — scheduled/background task executions are out of scope (already covered by the Tasks widget).
- Owner-scoped only — no admin-wide mode, matching every Mission Control widget shipped so far.
- Read-only — no Stop/kill action from the dashboard. Clicking an active agent's session name jumps to that session in chat (where the existing Stop button lives), via `window.sessionModule.selectSession(sessionId)` — the same mechanism sub-project 2a's Tool Call History panel already uses.
- `src/agent_runs.py`'s existing write path (`start()`, `stop()`, `_drain()`, eviction) must not change — this plan adds exactly one new read-only function to that file and nothing else.
- A session id present in the run registry but no longer in `SessionManager.sessions` (evicted/deleted) is skipped, not an error.
- `esc()` applied to every piece of session-derived text (`session_id`, `session_name`) before it reaches `innerHTML` — same XSS-hygiene bar every widget meets.
- Owner resolution uses `effective_user(request)`, not `get_current_user(request)` — matching the fix already applied to sub-project 2a's route (`effective_user`'s docstring names "sessions, chat history" as exactly this kind of route).

---

### Task 1: Active-agents enumeration — `list_active()` + join + route

**Files:**
- Modify: `src/agent_runs.py` (add `list_active()`)
- Create: `routes/agent_runs_routes.py` (join function + route)
- Modify: `app.py` (mount the new router)
- Test: `tests/test_agent_runs.py` (new)
- Test: `tests/test_agent_runs_routes.py` (new)

**Interfaces:**
- Produces: `agent_runs.list_active() -> List[str]` — session ids currently `status == "running"`.
- Produces: `list_active_agents(session_manager, owner: Optional[str]) -> List[Dict]` in `routes/agent_runs_routes.py` — each dict `{"session_id": str, "session_name": str}`.
- Produces: `setup_agent_runs_routes(session_manager) -> APIRouter`, mounted at `GET /api/agent-runs/active`. Response: `{"active": [...]}`.
- Consumes: `src.agent_runs._RUNS` (existing, read-only), `SessionManager.sessions` (existing in-memory `Dict[str, Session]`, `core/session_manager.py:74`), `Session.owner`/`Session.name` (existing dataclass fields, `core/models.py:66-74`), `src.auth_helpers.effective_user`.

- [ ] **Step 1: Write the failing tests for `list_active()`**

Create `tests/test_agent_runs.py`:

```python
"""list_active() enumerates running session ids -- a pure read added
alongside src.agent_runs' existing per-session run tracking, without
touching its write path (start/stop/eviction). See
docs/superpowers/specs/2026-08-11-mission-control-active-agents-design.md.
"""
from src import agent_runs


def _put_run(session_id, status):
    run = agent_runs._Run()
    run.status = status
    agent_runs._RUNS[session_id] = run


def test_list_active_returns_only_running_sessions(monkeypatch):
    monkeypatch.setattr(agent_runs, "_RUNS", {})
    _put_run("running-1", "running")
    _put_run("done-1", "done")
    _put_run("error-1", "error")
    _put_run("stopped-1", "stopped")
    _put_run("running-2", "running")

    assert sorted(agent_runs.list_active()) == ["running-1", "running-2"]


def test_list_active_empty_when_nothing_running(monkeypatch):
    monkeypatch.setattr(agent_runs, "_RUNS", {})
    _put_run("done-1", "done")

    assert agent_runs.list_active() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_runs.py -v --import-mode=importlib`
Expected: FAIL — `AttributeError: module 'src.agent_runs' has no attribute 'list_active'`.

- [ ] **Step 3: Add `list_active()` to `src/agent_runs.py`**

Find this existing function:

```python
def get_status(session_id: str) -> Optional[str]:
    r = _RUNS.get(session_id)
    return r.status if r else None
```

Immediately after it, insert:

```python
def list_active() -> List[str]:
    """Session ids currently mid-run ("running" status).

    Read-only -- does not touch the run registry's write path
    (start/stop/eviction). Used by Mission Control's Active Agents widget;
    see docs/superpowers/specs/2026-08-11-mission-control-active-agents-design.md.
    """
    return [sid for sid, run in _RUNS.items() if run.status == "running"]
```

Find this existing import line:

```python
from typing import AsyncGenerator, Dict, Optional
```

Replace it with:

```python
from typing import AsyncGenerator, Dict, List, Optional
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_runs.py -v --import-mode=importlib`
Expected: PASS (both tests).

- [ ] **Step 5: Write the failing tests for the join function and route**

Create `tests/test_agent_runs_routes.py`:

```python
"""list_active_agents() joins agent_runs.list_active() against
SessionManager's in-memory session cache for owner-scoped display.
GET /api/agent-runs/active exposes it. See
docs/superpowers/specs/2026-08-11-mission-control-active-agents-design.md.
"""
import asyncio
from types import SimpleNamespace

import routes.agent_runs_routes as arr
from src import agent_runs


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _request(user):
    return SimpleNamespace(state=SimpleNamespace(current_user=user))


class _FakeSession:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name


class _FakeSessionManager:
    def __init__(self, sessions):
        self.sessions = sessions


def test_list_active_agents_filters_by_owner(monkeypatch):
    monkeypatch.setattr(agent_runs, "list_active", lambda: ["s1", "s2", "s3"])
    sm = _FakeSessionManager({
        "s1": _FakeSession(owner="alice", name="Alice A"),
        "s2": _FakeSession(owner="bob", name="Bob B"),
        "s3": _FakeSession(owner="alice", name="Alice C"),
    })

    out = arr.list_active_agents(sm, owner="alice")

    assert out == [
        {"session_id": "s1", "session_name": "Alice A"},
        {"session_id": "s3", "session_name": "Alice C"},
    ]


def test_list_active_agents_skips_missing_session(monkeypatch):
    monkeypatch.setattr(agent_runs, "list_active", lambda: ["gone", "s1"])
    sm = _FakeSessionManager({"s1": _FakeSession(owner="alice", name="Alice A")})

    out = arr.list_active_agents(sm, owner="alice")

    assert out == [{"session_id": "s1", "session_name": "Alice A"}]


def test_list_active_agents_none_owner_matches_none_owner_sessions(monkeypatch):
    monkeypatch.setattr(agent_runs, "list_active", lambda: ["s1", "s2"])
    sm = _FakeSessionManager({
        "s1": _FakeSession(owner=None, name="Shared"),
        "s2": _FakeSession(owner="alice", name="Alice A"),
    })

    out = arr.list_active_agents(sm, owner=None)

    assert out == [{"session_id": "s1", "session_name": "Shared"}]


def test_route_scopes_to_effective_user(monkeypatch):
    captured = {}

    def fake_list_active_agents(session_manager, owner):
        captured["owner"] = owner
        return [{"session_id": "s1", "session_name": "Alice A"}]

    monkeypatch.setattr(arr, "list_active_agents", fake_list_active_agents)
    monkeypatch.setattr(arr, "effective_user", lambda request: "alice")
    router = arr.setup_agent_runs_routes(session_manager=object())
    handler = _route(router, "/api/agent-runs/active", "GET")

    out = asyncio.run(handler(request=_request("alice")))

    assert captured["owner"] == "alice"
    assert out == {"active": [{"session_id": "s1", "session_name": "Alice A"}]}
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_agent_runs_routes.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'routes.agent_runs_routes'`.

- [ ] **Step 7: Create `routes/agent_runs_routes.py`**

```python
"""Active-agents enumeration (Mission Control sub-project 2b).

Reads two already-existing pieces of in-memory state -- src.agent_runs'
per-session run status and SessionManager's in-memory session cache -- to
answer "which of the caller's own chat sessions are running right now."
Adds no new write path. See
docs/superpowers/specs/2026-08-11-mission-control-active-agents-design.md.
"""
from typing import Dict, List, Optional

from fastapi import APIRouter, Request

from src import agent_runs
from src.auth_helpers import effective_user


def list_active_agents(session_manager, owner: Optional[str]) -> List[Dict]:
    active = []
    for session_id in agent_runs.list_active():
        session = session_manager.sessions.get(session_id)
        if session is None:
            continue
        if session.owner != owner:
            continue
        active.append({"session_id": session_id, "session_name": session.name})
    return active


def setup_agent_runs_routes(session_manager) -> APIRouter:
    router = APIRouter(prefix="/api/agent-runs", tags=["agent-runs"])

    @router.get("/active")
    async def get_active_agents(request: Request):
        user = effective_user(request)
        return {"active": list_active_agents(session_manager, user)}

    return router
```

- [ ] **Step 8: Wire the router into `app.py`**

Find this existing block:

```python
# Cleanup
from routes.cleanup_routes import setup_cleanup_routes
app.include_router(setup_cleanup_routes(session_manager))
```

Immediately after it, insert:

```python

# Active agents (Mission Control sub-project 2b)
from routes.agent_runs_routes import setup_agent_runs_routes
app.include_router(setup_agent_runs_routes(session_manager))
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/test_agent_runs.py tests/test_agent_runs_routes.py -v --import-mode=importlib`
Expected: PASS (all 6 tests: 2 from Step 1, 4 from Step 5).

Run: `python -c "import app"` (from the repo root) to confirm `app.py` still imports cleanly with the new router mounted.
Expected: no output, exit code 0.

- [ ] **Step 10: Commit**

```bash
git add src/agent_runs.py routes/agent_runs_routes.py app.py tests/test_agent_runs.py tests/test_agent_runs_routes.py
git commit -m "feat(agent-runs): active-agents enumeration + GET /api/agent-runs/active

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Mission Control "Active Agents" widget (7th widget)

**Files:**
- Modify: `static/index.html` (add the 7th `.mission-control-card` — no `mission-control-open-link` button, unlike every other widget: entries are directly clickable, there's no separate view to open)
- Modify: `static/js/missionControl.js` (`loadActiveAgentsWidget`, wire into `refreshWidget`/`loadAllWidgets`/`init`)
- Modify: `tests/test_mission_control_ui.py` (new tests + extend `test_mission_control_loadAllWidgets_wires_all_loaders`'s `expected_calls` list — this widget has no `mc-open-*` button, so `test_mission_control_link_targets_exist_in_html` and `test_mission_control_open_handlers_wire_to_correct_targets` do NOT need a new entry)

**Interfaces:**
- Consumes: `GET /api/agent-runs/active` (Task 1), `$`/`esc`/`api`/`setCardBody`/`setCardError` (existing, `static/js/missionControl.js`), `window.sessionModule.selectSession(sessionId)` (existing, `static/js/sessions.js:1783` — same mechanism sub-project 2a's `toolCallLog.js` already uses).
- Produces: `loadActiveAgentsWidget()`, added to `refreshWidget`'s dispatch and `loadAllWidgets()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mission_control_ui.py`:

```python
def test_mission_control_has_active_agents_widget():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-widget="active-agents"' in html
    assert 'id="mc-body-active-agents"' in html

    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "loadActiveAgentsWidget" in src
    assert "/api/agent-runs/active" in src


def test_mission_control_active_agents_uses_select_session_for_jump():
    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "sessionModule" in src
    assert "selectSession" in src
```

Then, in the same file, find `test_mission_control_loadAllWidgets_wires_all_loaders` and add a 7th entry to its `expected_calls` list:

```python
    expected_calls = [
        'loadModelsWidget()',
        'loadHardwareWidget()',
        'loadTasksWidget()',
        'loadMemoryWidget()',
        'loadIntegrationsWidget()',
        'loadToolCallsWidget()',
        'loadActiveAgentsWidget()',
    ]
```

Do NOT modify `test_mission_control_link_targets_exist_in_html` or `test_mission_control_open_handlers_wire_to_correct_targets` — this widget has no `mc-open-*` open-link button (entries are directly clickable instead), so those two tests' existing 6-entry maps are already complete and correct as-is.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mission_control_ui.py -v --import-mode=importlib`
Expected: FAIL — the 2 new tests fail (widget doesn't exist), and `test_mission_control_loadAllWidgets_wires_all_loaders` fails (`loadActiveAgentsWidget()` not yet in `loadAllWidgets()`'s body).

- [ ] **Step 3: Add the 7th card to `static/index.html`**

Find this block (the Tool Calls card, the last card inside `#mission-control-grid`, immediately before the grid's closing `</div>`):

```html
        <div class="mission-control-card" id="mc-card-tool-calls" data-widget="tool-calls">
          <div class="mission-control-card-header">
            <h5>Tool Calls</h5>
            <button class="btn mission-control-refresh" data-widget="tool-calls" title="Refresh">&#x21bb;</button>
          </div>
          <div class="mission-control-card-body" id="mc-body-tool-calls">Loading…</div>
          <button class="btn mission-control-open-link" id="mc-open-tool-calls">Open Tool Call History</button>
        </div>
```

Immediately after it, insert:

```html
        <div class="mission-control-card" id="mc-card-active-agents" data-widget="active-agents">
          <div class="mission-control-card-header">
            <h5>Active Agents</h5>
            <button class="btn mission-control-refresh" data-widget="active-agents" title="Refresh">&#x21bb;</button>
          </div>
          <div class="mission-control-card-body" id="mc-body-active-agents">Loading…</div>
        </div>
```

- [ ] **Step 4: Add `loadActiveAgentsWidget` to `static/js/missionControl.js`**

Find `loadToolCallsWidget`'s closing `}` followed by `function refreshWidget(widgetId) {`. Immediately after `loadToolCallsWidget`'s closing brace and before `refreshWidget`, insert:

```javascript
async function loadActiveAgentsWidget() {
  const body = $('mc-body-active-agents');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/agent-runs/active');
    const items = data.active || [];
    const listHtml = items.map(function (a) {
      return '<div><a href="#" class="mc-active-agent-session" data-session-id="' + esc(a.session_id) + '">' + esc(a.session_name || 'Unknown') + '</a></div>';
    }).join('') || '<div>No active agents right now</div>';
    setCardBody('active-agents', esc(items.length) + ' active<br>' + listHtml);
  } catch (e) {
    setCardError('active-agents', e.message);
  }
}
```

Update `refreshWidget` to:

```javascript
function refreshWidget(widgetId) {
  if (widgetId === 'models') loadModelsWidget();
  if (widgetId === 'hardware') loadHardwareWidget();
  if (widgetId === 'tasks') loadTasksWidget();
  if (widgetId === 'memory') loadMemoryWidget();
  if (widgetId === 'integrations') loadIntegrationsWidget();
  if (widgetId === 'tool-calls') loadToolCallsWidget();
  if (widgetId === 'active-agents') loadActiveAgentsWidget();
}
```

Update `loadAllWidgets` to:

```javascript
function loadAllWidgets() {
  loadModelsWidget();
  loadHardwareWidget();
  loadTasksWidget();
  loadMemoryWidget();
  loadIntegrationsWidget();
  loadToolCallsWidget();
  loadActiveAgentsWidget();
}
```

Find the `openToolCalls` handler block in `init()`:

```javascript
  const openToolCalls = $('mc-open-tool-calls');
  if (openToolCalls) openToolCalls.addEventListener('click', function () {
    closeMissionControl();
    const toolCallsBtn = $('tool-tool-calls-btn');
    if (toolCallsBtn) toolCallsBtn.click();
  });
```

Immediately after it, insert (this widget has no `mc-open-*` button — instead, a click-delegation listener directly on the card body handles jumping to a session, mirroring `toolCallLog.js`'s own delegation pattern for its session-jump links):

```javascript
  const activeAgentsBody = $('mc-body-active-agents');
  if (activeAgentsBody) activeAgentsBody.addEventListener('click', function (ev) {
    const link = ev.target.closest('.mc-active-agent-session');
    if (!link) return;
    ev.preventDefault();
    const sid = link.getAttribute('data-session-id');
    if (sid) {
      closeMissionControl();
      if (window.sessionModule && window.sessionModule.selectSession) {
        window.sessionModule.selectSession(sid);
      }
    }
  });
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mission_control_ui.py -v --import-mode=importlib`
Expected: PASS (all 13 tests: the original 11 from sub-project 2a plus the 2 new ones).

Run: `node --check static/js/missionControl.js`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/missionControl.js tests/test_mission_control_ui.py
git commit -m "feat(mission-control): Active Agents widget (7th widget)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

# Workflow Triggers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a saved workflow automatically — on a schedule, from a chat/session event, or via an external webhook — by making a workflow a new `ScheduledTask` type executed through the shipped `run_workflow` engine.

**Architecture:** A workflow trigger is a `ScheduledTask` with `task_type="workflow"` (workflow id in the existing `action` column, fixed-inputs JSON in `prompt` — zero schema migration). The scheduler gets one new execution branch that resolves inputs (a pure convention) and calls `run_workflow`. An optional `context` (webhook body / event payload) is threaded through the scheduler's fire path so a webhook body or a chat message reaches the workflow's input nodes. Trigger CRUD reuses `/api/tasks`; a Triggers panel is added to the editor.

**Tech Stack:** Python/FastAPI (scheduler, event_bus, routes), vanilla-JS ES module (editor panel). Reuses sub-project 1's engine (`src/workflows/`) and sub-project 2's editor unchanged.

## Global Constraints

- A workflow trigger = `ScheduledTask(task_type="workflow")`: `action` = workflow id, `prompt` = fixed-inputs JSON (object) or empty. **No new DB columns / no migration.**
- Input precedence (lowest→highest): input-node `default` (applied by the engine) < trigger fixed inputs < firing context. Context = a dict; use `context["inputs"]` if it's a dict, else the context dict itself; overlay only keys matching actual input-node names. This one rule covers webhook body, webhook `{"inputs":…}`, and the event `{"message": …}` case.
- **Admin-only**: creating a `workflow` trigger requires admin (a workflow runs the LLM + arbitrary agent tools). Runs execute with the owner's privileges; each tool's own gate still applies.
- The scheduler's `context` param is **optional and backward-compatible** — every existing caller passes nothing and behaves exactly as before.
- No test runs a real model, endpoint, or tool: `run_workflow` (or its `model_call`/`tool_dispatch`) is mocked at the seam.
- Every pytest run uses `--import-mode=importlib`. JS syntax is gated via `node --check` on a temp `.mjs` (a bare `.js` is parsed as CommonJS and rejects `import`). Stage specific files (never `git add -A`). Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Commit directly to `dev`.

---

### Task 1: Input resolution convention (pure)

**Files:**
- Create: `src/workflows/triggers.py`
- Test: `tests/test_workflow_triggers_resolve.py`

**Interfaces:**
- Produces: `resolve_trigger_inputs(workflow, fixed_inputs=None, context=None) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_triggers_resolve.py`:

```python
from src.workflows.triggers import resolve_trigger_inputs


def _wf(*names):
    return {"nodes": [{"id": n, "type": "input", "config": {"name": n}} for n in names]}


def test_schedule_uses_fixed_inputs_only_and_ignores_unknown():
    wf = _wf("topic")
    assert resolve_trigger_inputs(wf, {"topic": "AI", "nope": "x"}, None) == {"topic": "AI"}


def test_no_inputs_returns_empty():
    assert resolve_trigger_inputs(_wf(), {}, None) == {}
    assert resolve_trigger_inputs(_wf("a"), None, None) == {}


def test_webhook_top_level_body_maps_by_name():
    wf = _wf("topic", "lang")
    assert resolve_trigger_inputs(wf, {}, {"topic": "cats", "extra": 1}) == {"topic": "cats"}


def test_webhook_inputs_wrapper_takes_precedence_over_siblings():
    wf = _wf("topic")
    # when an "inputs" object is present, the top-level siblings are NOT scanned
    ctx = {"inputs": {"topic": "wrapped"}, "topic": "toplevel"}
    assert resolve_trigger_inputs(wf, {}, ctx) == {"topic": "wrapped"}


def test_event_message_injected_into_message_input_when_present():
    assert resolve_trigger_inputs(_wf("message"), {}, {"message": "hello"}) == {"message": "hello"}
    # no 'message' input -> event contributes nothing
    assert resolve_trigger_inputs(_wf("topic"), {"topic": "t"}, {"message": "hello"}) == {"topic": "t"}


def test_context_overrides_fixed():
    wf = _wf("topic")
    assert resolve_trigger_inputs(wf, {"topic": "fixed"}, {"topic": "ctx"}) == {"topic": "ctx"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_triggers_resolve.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.workflows.triggers'`).

- [ ] **Step 3: Write the implementation**

Create `src/workflows/triggers.py`:

```python
"""Resolve a triggered workflow run's inputs from the trigger's fixed inputs and
the firing context (webhook body / event payload). Pure — no scheduler, engine, or
DB. Precedence: input-node default (applied by the engine) < fixed inputs < context.

Context rule (one rule for every source): use context["inputs"] if it is a dict,
otherwise the context dict itself; overlay only keys that name an actual input node.
This covers a webhook body, a webhook {"inputs": {...}} envelope, and the event
{"message": "..."} payload alike."""


def _input_names(workflow):
    names = set()
    for n in (workflow.get("nodes") or []):
        if n.get("type") == "input":
            nm = (n.get("config") or {}).get("name")
            if nm:
                names.add(nm)
    return names


def resolve_trigger_inputs(workflow, fixed_inputs=None, context=None):
    """Return {name: value} for the workflow's input nodes. Unknown names are
    ignored; any input name omitted here is defaulted by the engine's run_input."""
    names = _input_names(workflow)
    out = {}
    for k, v in (fixed_inputs or {}).items():
        if k in names:
            out[k] = v
    ctx = context or {}
    inner = ctx.get("inputs") if isinstance(ctx, dict) else None
    candidate = inner if isinstance(inner, dict) else ctx
    if isinstance(candidate, dict):
        for k, v in candidate.items():
            if k in names:
                out[k] = v
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workflow_triggers_resolve.py --import-mode=importlib -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/workflows/triggers.py tests/test_workflow_triggers_resolve.py
git commit -m "feat(workflows): resolve_trigger_inputs — convention for triggered runs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Scheduler — workflow execution branch + context threading

**Files:**
- Modify: `src/task_scheduler.py` (add `context` param to `run_task_now`/`_execute_task`/`_execute_task_locked`; add the `workflow` dispatch branch; add `_execute_workflow_task`)
- Test: `tests/test_scheduler_workflow_task.py`

**Interfaces:**
- Consumes (Task 1): `resolve_trigger_inputs`. Also `src.workflows.store.get_workflow`, `src.workflows.engine.run_workflow` (shipped).
- Produces: `TaskScheduler._execute_workflow_task(task, *, context=None) -> (result: str, success: bool)`; `run_task_now(task_id, *, force=False, context=None)`; `_execute_task(task_id, *, bypass_model_slot=False, release_executing=True, context=None)`; `_execute_task_locked(..., context=None)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scheduler_workflow_task.py`:

```python
import asyncio
import types

import pytest

import src.workflows.store as store
import src.workflows.engine as engine
from src.task_scheduler import TaskScheduler


def _run(coro):
    return asyncio.run(coro)


def _bare_scheduler():
    # Bypass __init__ — _execute_workflow_task uses no instance state.
    return TaskScheduler.__new__(TaskScheduler)


def _task(action="flow1", prompt='{"topic": "AI"}', owner="admin"):
    return types.SimpleNamespace(action=action, prompt=prompt, owner=owner, name="t")


def test_missing_workflow_returns_error_not_raise(monkeypatch):
    monkeypatch.setattr(store, "get_workflow", lambda wid: None)
    s = _bare_scheduler()
    result, success = _run(s._execute_workflow_task(_task(), context=None))
    assert success is False and "not found" in result


def test_runs_workflow_and_summarizes_success(monkeypatch):
    wf = {"nodes": [{"id": "i", "type": "input", "config": {"name": "topic"}}]}
    monkeypatch.setattr(store, "get_workflow", lambda wid: wf)
    seen = {}

    async def fake_run(w, inputs, ctx, **kw):
        seen["inputs"] = inputs
        seen["ctx"] = ctx
        return {"outputs": {"answer": "hi"}, "log": [{"node": "o", "status": "ok"}]}
    monkeypatch.setattr(engine, "run_workflow", fake_run)

    s = _bare_scheduler()
    result, success = _run(s._execute_workflow_task(_task(prompt='{"topic":"cats"}'),
                                                     context={"topic": "dogs"}))
    assert success is True
    assert "answer=hi" in result
    assert seen["inputs"] == {"topic": "dogs"}     # context overrides fixed
    assert seen["ctx"] == {"owner": "admin"}


def test_node_error_marks_run_failed(monkeypatch):
    wf = {"nodes": []}
    monkeypatch.setattr(store, "get_workflow", lambda wid: wf)

    async def fake_run(w, inputs, ctx, **kw):
        return {"outputs": {}, "log": [{"node": "l", "status": "error", "error": "boom"}]}
    monkeypatch.setattr(engine, "run_workflow", fake_run)

    s = _bare_scheduler()
    result, success = _run(s._execute_workflow_task(_task(prompt=None), context=None))
    assert success is False and "1 error" in result


def test_run_task_now_threads_context(monkeypatch):
    s = TaskScheduler.__new__(TaskScheduler)
    s._executing = set()
    s._executing_lock = asyncio.Lock()
    captured = {}

    async def fake_execute(task_id, **kw):
        captured["task_id"] = task_id
        captured["context"] = kw.get("context")
    s._execute_task = fake_execute

    async def go():
        await s.run_task_now("t1", context={"a": 1})
        await asyncio.sleep(0)      # let the created task run
    _run(go())
    assert captured["task_id"] == "t1" and captured["context"] == {"a": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scheduler_workflow_task.py --import-mode=importlib -q`
Expected: FAIL (`AttributeError: … has no attribute '_execute_workflow_task'`; and `run_task_now` doesn't accept `context`).

- [ ] **Step 3: Write the implementation**

In `src/task_scheduler.py`:

(a) Add `import json` if not already imported at the top (it is — confirm the line `import json` exists near the other imports; if not, add it).

(b) Change `run_task_now` (~line 2215) to thread `context`:

```python
    async def run_task_now(self, task_id: str, *, force: bool = False, context=None):
        """Manually or event/webhook trigger a task execution."""
        if force:
            asyncio.create_task(self._execute_task(
                task_id, bypass_model_slot=True, release_executing=False, context=context))
            return True
        async with self._executing_lock:
            if task_id in self._executing:
                return False
            self._executing.add(task_id)
        asyncio.create_task(self._execute_task(task_id, context=context))
        return True
```

(c) Change `_execute_task` (~line 725) signature and thread `context` to BOTH `_execute_task_locked` calls (~lines 753 and 762):

```python
    async def _execute_task(self, task_id: str, *, bypass_model_slot: bool = False,
                            release_executing: bool = True, context=None):
```
At each `await self._execute_task_locked(...)` call inside `_execute_task`, add `context=context` to the keyword arguments.

(d) Change `_execute_task_locked` (~line 804) signature to accept `context`:

```python
    async def _execute_task_locked(
        self,
        task_id: str,
        run_id: str,
        *,
        release_executing: bool = True,
        gate_foreground: bool = True,
        context=None,
    ):
```

(e) Add the `workflow` branch in the `task_type` dispatch (the `if task_type == "action": … elif task_type == "research": … else:` block, ~line 910). Insert a new `elif` before the final `else`:

```python
                elif task_type == "workflow":
                    result, success = await self._execute_workflow_task(task, context=context)
                    run.status = "success" if success else "error"
                    run.result = result
                    if not success:
                        run.error = result
```

(f) Add the executor method (place it near `_execute_action`):

```python
    async def _execute_workflow_task(self, task, *, context=None):
        """Run a task_type='workflow' trigger: load the workflow, resolve its
        inputs (fixed + firing context), run it, and summarize. Returns
        (result_summary, success). A failing node is captured in the summary and
        marks the run failed; a missing workflow returns an error, never raises."""
        from src.workflows.store import get_workflow
        from src.workflows.engine import run_workflow
        from src.workflows.triggers import resolve_trigger_inputs

        wid = task.action
        wf = get_workflow(wid) if wid else None
        if not wf:
            return f"workflow '{wid}' not found", False
        try:
            fixed = json.loads(task.prompt) if task.prompt else {}
            if not isinstance(fixed, dict):
                fixed = {}
        except (ValueError, TypeError):
            fixed = {}
        inputs = resolve_trigger_inputs(wf, fixed, context)
        result = await run_workflow(wf, inputs, {"owner": task.owner})
        outputs = result.get("outputs") or {}
        log = result.get("log") or []
        errors = sum(1 for e in log if e.get("status") == "error")
        oks = sum(1 for e in log if e.get("status") == "ok")
        out_str = "; ".join(f"{k}={v}" for k, v in outputs.items()) or "(no outputs)"
        summary = f"{out_str} · {oks} ok, {errors} error"
        return summary, errors == 0
```

Note: the method imports `get_workflow`/`run_workflow` from their modules at call time; the test monkeypatches `src.workflows.store.get_workflow` and `src.workflows.engine.run_workflow`, so the `from … import …` inside the method must resolve those names off the (patched) module — which it does, because `from src.workflows.store import get_workflow` re-reads the module attribute each call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scheduler_workflow_task.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/task_scheduler.py tests/test_scheduler_workflow_task.py
git commit -m "feat(scheduler): workflow task branch + optional context threading

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Event payload — carry the chat message to the workflow

**Files:**
- Modify: `src/event_bus.py` (`fire_event`/`_handle_event` gain an optional `payload`, threaded to `run_task_now(context=…)`)
- Modify: `routes/chat_helpers.py` (`fire_message_event` fires the message text as payload)
- Test: `tests/test_event_bus_payload.py`

**Interfaces:**
- Consumes (Task 2): `run_task_now(task_id, *, context=None)`.
- Produces: `fire_event(event_name, owner=None, payload=None)`; `_handle_event(event_name, owner=None, payload=None)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_event_bus_payload.py`:

```python
import asyncio
import types

import src.event_bus as eb


def _run(coro):
    return asyncio.run(coro)


def test_fire_event_forwards_payload_to_handler(monkeypatch):
    seen = {}

    async def fake_handle(name, owner=None, payload=None):
        seen.update(name=name, owner=owner, payload=payload)
    monkeypatch.setattr(eb, "_handle_event", fake_handle)

    async def go():
        eb.fire_event("message_sent", "admin", payload={"message": "hi"})
        await asyncio.sleep(0)
    _run(go())
    assert seen == {"name": "message_sent", "owner": "admin", "payload": {"message": "hi"}}


def test_handle_event_passes_payload_as_context_at_threshold(monkeypatch):
    # One active event task at threshold 1; assert run_task_now gets context=payload.
    task = types.SimpleNamespace(id="t1", name="t", trigger_count=1, trigger_counter=0)

    class _Q:
        def filter(self, *a): return self
        def all(self): return [task]

    class _DB:
        def query(self, *a): return _Q()
        def commit(self): pass
        def close(self): pass
    monkeypatch.setattr(eb, "SessionLocal", lambda: _DB(), raising=False)
    # _handle_event imports SessionLocal locally from core.database; patch there too.
    import core.database as cdb
    monkeypatch.setattr(cdb, "SessionLocal", lambda: _DB())
    monkeypatch.setattr(eb, "_resolve_event_owner", lambda o: o)

    captured = {}

    class _Sched:
        async def run_task_now(self, task_id, *, context=None):
            captured.update(task_id=task_id, context=context)
    monkeypatch.setattr(eb, "_task_scheduler", _Sched())

    _run(eb._handle_event("message_sent", "admin", payload={"message": "hi"}))
    assert captured == {"task_id": "t1", "context": {"message": "hi"}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_event_bus_payload.py --import-mode=importlib -q`
Expected: FAIL (`fire_event`/`_handle_event` don't accept `payload`).

- [ ] **Step 3: Write the implementation**

In `src/event_bus.py`:

(a) `fire_event` — add `payload` and forward it:

```python
def fire_event(event_name: str, owner: Optional[str] = None, payload: Optional[dict] = None):
    """Fire an event — increments counters and triggers tasks that hit threshold.
    `payload` (optional) is passed to a triggered task as its run context (e.g.
    {"message": "..."} for message_sent). Safe from sync and async contexts."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_handle_event(event_name, owner, payload))
    except RuntimeError:
        asyncio.run(_handle_event(event_name, owner, payload))
```

(b) `_handle_event` — add `payload` and pass it as `context` on the fire:

```python
async def _handle_event(event_name: str, owner: Optional[str] = None, payload: Optional[dict] = None):
```
and change the fire line from `await _task_scheduler.run_task_now(task.id)` to:
```python
                    await _task_scheduler.run_task_now(task.id, context=payload)
```

In `routes/chat_helpers.py`, `fire_message_event` — change `fire_event("message_sent", user)` to carry the message:

```python
    fire_event("message_sent", user, payload={"message": message})
```
(The `message` parameter is already in scope in `fire_message_event`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_event_bus_payload.py --import-mode=importlib -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/event_bus.py routes/chat_helpers.py tests/test_event_bus_payload.py
git commit -m "feat(events): carry an optional payload to triggered tasks (message_sent)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Routes — create validation (admin) + webhook body → context

**Files:**
- Modify: `routes/task_routes.py` (`create_task` workflow validation + auto-name; `webhook_trigger` reads the body and threads it as context)
- Test: `tests/test_workflow_trigger_routes.py`

**Interfaces:**
- Consumes (Task 2): `run_task_now(task_id, *, context=None)`.
- Produces: module-level `validate_workflow_task_create(action, prompt, is_admin) -> None` (raises `HTTPException`), called by `create_task` for `task_type="workflow"`; `webhook_trigger(task_id, token, request)` passes the POST body as run context.

**Why an extracted validator:** `create_task`'s `_owner`/`_is_admin` are **closures nested inside `setup_task_routes`**, and the tasks route's auth is awkward to fake in a bare `TestClient`. Extracting the workflow-specific validation to a **module-level pure function** makes it directly unit-testable (booleans in, `HTTPException` out) without auth plumbing, and `create_task` simply calls it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_trigger_routes.py`:

```python
import pytest
from fastapi import HTTPException

import routes.task_routes as tr


def test_missing_action_is_400():
    with pytest.raises(HTTPException) as ei:
        tr.validate_workflow_task_create(None, None, True)
    assert ei.value.status_code == 400


def test_non_admin_is_403():
    with pytest.raises(HTTPException) as ei:
        tr.validate_workflow_task_create("flow1", None, False)
    assert ei.value.status_code == 403


def test_non_object_prompt_is_400():
    for bad in ("[1,2]", "not json", '"a string"', "5"):
        with pytest.raises(HTTPException) as ei:
            tr.validate_workflow_task_create("flow1", bad, True)
        assert ei.value.status_code == 400


def test_valid_workflow_task_passes():
    # admin, valid id, JSON-object prompt (and empty prompt) → no raise
    assert tr.validate_workflow_task_create("flow1", '{"topic": "AI"}', True) is None
    assert tr.validate_workflow_task_create("flow1", None, True) is None
    assert tr.validate_workflow_task_create("flow1", "", True) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_trigger_routes.py --import-mode=importlib -q`
Expected: FAIL (`module 'routes.task_routes' has no attribute 'validate_workflow_task_create'`).

- [ ] **Step 3: Write the implementation**

In `routes/task_routes.py`:

(a) Confirm `import json` is present at the top (add it if missing). Confirm `HTTPException` and `Request` are imported from `fastapi` (they are used already; add `Request` if missing).

(b) Add the module-level validator (near the top, after the imports / alongside other module-level helpers):

```python
def validate_workflow_task_create(action, prompt, is_admin: bool) -> None:
    """Validate a task_type='workflow' create request. Raises HTTPException on:
    missing workflow id (400), non-admin (403), or a `prompt` that is not a JSON
    object (400). A workflow runs the LLM + arbitrary tools, so it is admin-only."""
    if not action:
        raise HTTPException(400, "Workflow id (action) is required for workflow tasks")
    if not is_admin:
        raise HTTPException(403, "Workflow triggers require admin privileges")
    if prompt:
        try:
            parsed = json.loads(prompt)
        except (ValueError, TypeError):
            raise HTTPException(400, "Workflow fixed inputs must be a JSON object")
        if not isinstance(parsed, dict):
            raise HTTPException(400, "Workflow fixed inputs must be a JSON object")
```

(c) In `create_task`, call it right after the existing action check (after the line `raise HTTPException(400, "Action name is required for action tasks")`, before `_require_admin_for_task_action`):

```python
        if req.task_type == "workflow":
            validate_workflow_task_create(req.action, req.prompt, _is_admin(user))
```

(d) In the auto-name block (the `if not name:` chain), add a workflow case:

```python
            elif req.task_type == "workflow":
                name = f"Workflow: {req.action}"
```
(Place this `elif` alongside the existing `if req.task_type == "action":` / `elif req.prompt:` branches, before the final `else: name = "Untitled Task"`.)

(d) Change `webhook_trigger` (~line 1045) to read the POST body and thread it as context:

```python
    @router.post("/{task_id}/webhook/{token}")
    async def webhook_trigger(task_id: str, token: str, request: Request):
        """Unauthenticated endpoint — the token IS the auth."""
        db = SessionLocal()
        try:
            task = db.query(ScheduledTask).filter(
                ScheduledTask.id == task_id,
                ScheduledTask.webhook_token == token,
                ScheduledTask.status == "active",
            ).first()
            if not task:
                raise HTTPException(404, "Not found")
            if (
                is_admin_only_task_action(task.task_type, task.action)
                and not owner_has_admin_task_privileges(task.owner)
            ):
                task.status = "paused"
                task.next_run = None
                db.commit()
                raise HTTPException(403, f"Action '{task.action}' requires admin privileges")
        finally:
            db.close()
        try:
            body = await request.json()
        except Exception:
            body = None
        started = await task_scheduler.run_task_now(task_id, context=body)
        if not started:
            raise HTTPException(409, "Task is already running")
        return {"ok": True, "message": "Task triggered via webhook"}
```
(`Request` is already imported in this module — confirm; if not, add it to the FastAPI import.)

- [ ] **Step 4: Run tests + import smoke**

Run: `python -m pytest tests/test_workflow_trigger_routes.py --import-mode=importlib -q` (Expected: PASS, 4 passed) and `python -c "import app"` (no error).

- [ ] **Step 5: Commit**

```bash
git add routes/task_routes.py tests/test_workflow_trigger_routes.py
git commit -m "feat(tasks): workflow-trigger create validation (admin) + webhook body context

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Editor — Triggers panel

**Files:**
- Modify: `static/index.html` (a `wf-triggers` panel container inside the workflows modal)
- Modify: `static/js/workflows.js` (the Triggers panel: list/add/delete via `/api/tasks`)
- Test: `tests/test_workflow_editor_shell.py` (extend the string-assert + the existing `node --check` guards this task)

**Interfaces:**
- Consumes: module globals `graph`, `currentId`, `$`, `api`, `msg`, `G` (from sub-project 2); `save` (Task-5 of sub-project 2). The `/api/tasks` API (`GET`, `POST`, `DELETE /{id}`).
- Produces: `renderTriggers()`, wired to open with the modal.

- [ ] **Step 1: Add the panel container to `index.html`**

In `static/index.html`, inside the `#wf-inspector` aside's sibling area — add a triggers container at the end of the toolbar column, right before the closing of the canvas column `</div>` that precedes `#wf-inspector`. Concretely, add this block immediately after the `#wf-results` div:

```html
          <div id="wf-triggers" style="border-top:1px solid var(--border);padding:6px;max-height:180px;overflow:auto;display:none;"></div>
```

Also add a toggle button to the toolbar — in `#wf-toolbar`, after the `wf-run` button:

```html
            <button id="wf-triggers-btn">Triggers</button>
```

- [ ] **Step 2: Add the Triggers panel code to `workflows.js`**

In `static/js/workflows.js`, add these functions (near `renderResults`) and wire the toggle in `init()`.

```javascript
// ── triggers panel: schedule/event/webhook a saved workflow via /api/tasks ──
const _EVENTS = ['message_sent', 'session_created'];

async function renderTriggers() {
  const host = $('wf-triggers');
  if (!host) return;
  host.style.display = '';
  host.innerHTML = '';
  if (!currentId) {
    host.textContent = 'Save the workflow first to add triggers.';
    host.style.opacity = '0.6';
    return;
  }
  host.style.opacity = '';
  let all = [];
  try { all = (await api('/api/tasks')).tasks || []; }   // list_tasks -> {"tasks": [...]}
  catch (e) { host.textContent = 'Could not load triggers: ' + e.message; return; }
  const mine = all.filter((t) => t.task_type === 'workflow' && t.action === currentId);
  const list = document.createElement('div');
  mine.forEach((t) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;align-items:center;font-size:12px;padding:2px 0;';
    const label = document.createElement('span');
    label.style.cssText = 'flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
    const when = t.trigger_type === 'schedule' ? (t.schedule || 'schedule')
      : t.trigger_type === 'event' ? ('on ' + (t.trigger_event || 'event'))
      : 'webhook';
    label.textContent = `${t.name || t.id} · ${when}`;
    row.appendChild(label);
    if (t.trigger_type === 'webhook' && t.webhook_token) {
      row.appendChild(_btn('Copy URL', () => {
        const url = `${location.origin}/api/tasks/${t.id}/webhook/${t.webhook_token}`;
        if (navigator.clipboard) navigator.clipboard.writeText(url);
        msg('Webhook URL copied');
      }));
    }
    row.appendChild(_btn('Delete', async () => {
      try { await api('/api/tasks/' + encodeURIComponent(t.id), { method: 'DELETE' }); renderTriggers(); }
      catch (e) { msg('Delete failed: ' + e.message, true); }
    }));
    list.appendChild(row);
  });
  if (!mine.length) {
    const e = document.createElement('div');
    e.style.cssText = 'font-size:12px;opacity:0.6;'; e.textContent = 'No triggers yet.';
    list.appendChild(e);
  }
  host.appendChild(list);
  host.appendChild(_triggerForm());
}

function _btn(label, fn) {
  const b = document.createElement('button'); b.type = 'button'; b.textContent = label;
  b.style.cssText = 'font-size:11px;'; b.addEventListener('click', fn); return b;
}

function _triggerForm() {
  const form = document.createElement('div');
  form.style.cssText = 'margin-top:8px;border-top:1px dashed var(--border);padding-top:6px;font-size:12px;';
  const typeSel = document.createElement('select');
  ['schedule', 'event', 'webhook'].forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v; typeSel.appendChild(o); });
  const cfg = document.createElement('span');
  function renderCfg() {
    cfg.innerHTML = '';
    if (typeSel.value === 'schedule') {
      const s = document.createElement('select'); s.id = 'wf-tr-sched';
      ['once', 'daily', 'weekly'].forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v; s.appendChild(o); });
      const time = document.createElement('input'); time.id = 'wf-tr-time'; time.value = '09:00'; time.style.width = '60px';
      cfg.appendChild(s); cfg.appendChild(time);
    } else if (typeSel.value === 'event') {
      const s = document.createElement('select'); s.id = 'wf-tr-event';
      _EVENTS.forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v; s.appendChild(o); });
      const n = document.createElement('input'); n.id = 'wf-tr-count'; n.type = 'number'; n.value = '1'; n.style.width = '48px';
      cfg.appendChild(s); cfg.appendChild(n);
    }
  }
  typeSel.addEventListener('change', renderCfg); renderCfg();
  // fixed-inputs mini-editor: one row per input node
  const inputsWrap = document.createElement('div');
  inputsWrap.style.cssText = 'margin:4px 0;';
  const inputNodes = graph.nodes.filter((n) => n.type === 'input');
  inputNodes.forEach((n) => {
    const nm = (n.config || {}).name || n.id;
    const row = document.createElement('div'); row.style.cssText = 'display:flex;gap:4px;align-items:center;';
    const lab = document.createElement('span'); lab.textContent = nm; lab.style.cssText = 'width:80px;opacity:0.7;';
    const val = document.createElement('input'); val.dataset.input = nm; val.value = (n.config || {}).default || ''; val.style.flex = '1';
    row.appendChild(lab); row.appendChild(val); inputsWrap.appendChild(row);
  });
  form.appendChild(typeSel); form.appendChild(cfg);
  form.appendChild(inputsWrap);
  form.appendChild(_btn('Add trigger', () => _addTrigger(typeSel.value, inputsWrap)));
  return form;
}

async function _addTrigger(triggerType, inputsWrap) {
  const fixed = {};
  inputsWrap.querySelectorAll('input[data-input]').forEach((el) => {
    if (el.value !== '') fixed[el.dataset.input] = el.value;
  });
  const body = {
    task_type: 'workflow', action: currentId, trigger_type: triggerType,
    prompt: JSON.stringify(fixed),
  };
  if (triggerType === 'schedule') {
    body.schedule = ($('wf-tr-sched') || {}).value || 'daily';
    body.scheduled_time = ($('wf-tr-time') || {}).value || '09:00';
  } else if (triggerType === 'event') {
    body.trigger_event = ($('wf-tr-event') || {}).value || 'message_sent';
    body.trigger_count = parseInt(($('wf-tr-count') || {}).value || '1', 10);
  }
  try {
    await api('/api/tasks', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    msg('Trigger added'); renderTriggers();
  } catch (e) { msg('Add trigger failed: ' + e.message, true); }
}
```

In `init()`, wire the toggle (after the Save/Run wiring):

```javascript
  const trBtn = $('wf-triggers-btn');
  if (trBtn) trBtn.addEventListener('click', () => {
    const host = $('wf-triggers');
    if (host && host.style.display === 'none') renderTriggers();
    else if (host) host.style.display = 'none';
  });
```

- [ ] **Step 3: Extend the shell test**

In `tests/test_workflow_editor_shell.py`, add to `test_index_html_wires_the_editor`:

```python
    assert 'id="wf-triggers"' in html
    assert 'id="wf-triggers-btn"' in html
```

- [ ] **Step 4: Verify syntax + manual check**

Run: `python -m pytest tests/test_workflow_editor_shell.py --import-mode=importlib -q` (Expected: PASS — the `node --check` guards the new JS compiles, and the HTML asserts pass).
Manual (as admin): open a saved workflow → click **Triggers** → the panel lists existing triggers and an add-form → pick `webhook`, Add → the trigger appears with a **Copy URL** button; POST that URL with a JSON body and confirm the workflow runs (a `TaskRun` appears in the Tasks modal) → add a `schedule` (daily) trigger and confirm it lists → Delete removes it. If the workflow is unsaved, the panel says to save first.

- [ ] **Step 5: Commit**

```bash
git add static/index.html static/js/workflows.js tests/test_workflow_editor_shell.py
git commit -m "feat(workflows): editor Triggers panel (schedule/event/webhook via /api/tasks)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Tasks 1-4 are TDD** (Python, real unit tests). **Task 5 is build + `node --check` + a manual checklist** (DOM panel; the pure logic it needs already exists). Do not skip Task 5's manual checklist — it's the behavioral gate.
- The scheduler's `context` param is optional everywhere; every pre-existing caller of `run_task_now`/`_execute_task`/`_execute_task_locked` keeps working unchanged (they pass no context → `None` → schedule/manual behavior).
- No new DB columns. `action` = workflow id, `prompt` = fixed-inputs JSON.
- No test may run a real model/tool: `run_workflow` is mocked in Task 2; Task 4 mocks the scheduler.
- Security is load-bearing: creating a `workflow` trigger is admin-only (Task 4). Do not weaken it.
- Scope: this is triggers only. Do NOT build the Skill export bridge or richer nodes (later sub-projects), and do NOT add a per-input mapping UI or synchronous webhook-output responses (v1 non-goals).

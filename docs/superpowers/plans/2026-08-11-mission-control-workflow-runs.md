# Mission Control Sub-project 2c: Live Workflow Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an admin see, at a glance in Mission Control, which workflows are currently
executing right now, however they were triggered (agent tool, direct API, or scheduled task).

**Architecture:** A new in-memory module, `src/workflow_runs.py`, mirrors sub-project 2b's
`src/agent_runs.py` pattern exactly — a live snapshot with no persisted history. Each of
`run_workflow()`'s 3 existing call sites gets wrapped with `start()`/`finish()` calls; `run_workflow()`
itself is never modified. A new admin-gated route exposes the registry; an 8th Mission Control
widget displays it.

**Tech Stack:** FastAPI route (Python), no new database table (pure in-memory read, matching
sub-project 2b), vanilla JS ES module (matches `static/js/missionControl.js`), pytest.

## Global Constraints

- All 3 trigger paths tracked: the admin-only agent tool (`src/agent_tools/workflow_tool.py`),
  the direct API route (`routes/workflow_routes.py`), and the scheduled-task executor
  (`src/task_scheduler.py`).
- Live snapshot only — no new database table, no persisted history. A run disappears from the
  registry the instant it finishes, success or failure alike.
- Status only, no per-node progress — `run_workflow()`'s internals (`src/workflows/engine.py`)
  must NOT be modified. Every task in this plan only wraps existing call sites.
- No new access-control gate — reuse the exact same admin gate the rest of the workflows
  subsystem already has: `dependencies=[Depends(require_admin)]` from `core.middleware`,
  matching `routes/workflow_routes.py:30`'s own router construction verbatim.
- `finish()` must always run, even if `run_workflow()` raises — every wrap uses a `try/finally`
  around the `run_workflow()` call specifically (not the whole surrounding function).
- `esc()` applied to every piece of workflow-derived text (`workflow_name`, `trigger`) before it
  reaches `innerHTML`, matching every widget's established XSS-hygiene bar.

---

### Task 1: `src/workflow_runs.py` — in-memory active-run registry

**Files:**
- Create: `src/workflow_runs.py`
- Test: `tests/test_workflow_runs.py`

**Interfaces:**
- Produces: `start(workflow_id: str, workflow_name: str, owner: Optional[str], trigger: str) -> str`
  (returns a generated `run_id`), `finish(run_id: str) -> None` (no-op if `run_id` unknown),
  `list_active() -> List[Dict]` (each dict: `{"run_id", "workflow_id", "workflow_name", "owner",
  "trigger", "started_at"}`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_workflow_runs.py`:

```python
"""workflow_runs tracks currently-executing workflow runs in memory -- a live
snapshot mirroring src/agent_runs.py's own established pattern (Mission
Control sub-project 2b). No persisted history: a run disappears from
list_active() the instant finish() is called, success or failure alike. See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md.
"""
from src import workflow_runs


def test_start_adds_a_running_entry(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})

    run_id = workflow_runs.start("wf-1", "My Workflow", "alice", "api")

    active = workflow_runs.list_active()
    matching = [r for r in active if r["run_id"] == run_id]
    assert len(matching) == 1
    entry = matching[0]
    assert entry["workflow_id"] == "wf-1"
    assert entry["workflow_name"] == "My Workflow"
    assert entry["owner"] == "alice"
    assert entry["trigger"] == "api"
    assert "started_at" in entry


def test_finish_removes_the_entry(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    run_id = workflow_runs.start("wf-1", "My Workflow", "alice", "api")

    workflow_runs.finish(run_id)

    assert workflow_runs.list_active() == []


def test_finish_on_unknown_run_id_does_not_raise(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})

    workflow_runs.finish("does-not-exist")  # must not raise


def test_concurrent_runs_of_the_same_workflow_get_distinct_entries(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})

    run_id_1 = workflow_runs.start("wf-1", "My Workflow", "alice", "api")
    run_id_2 = workflow_runs.start("wf-1", "My Workflow", "bob", "scheduled")

    assert run_id_1 != run_id_2
    ids = {r["run_id"] for r in workflow_runs.list_active()}
    assert run_id_1 in ids
    assert run_id_2 in ids
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_workflow_runs.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.workflow_runs'`.

- [ ] **Step 3: Create `src/workflow_runs.py`**

```python
"""In-memory tracker for currently-executing workflow runs (Mission Control
sub-project 2c). Mirrors src/agent_runs.py's own pattern: a live snapshot of
what's running right now, with no persisted history -- a run disappears the
instant it finishes, success or failure alike. See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

_RUNS: Dict[str, Dict] = {}


def start(workflow_id: str, workflow_name: str, owner: Optional[str], trigger: str) -> str:
    run_id = str(uuid.uuid4())
    _RUNS[run_id] = {
        "run_id": run_id,
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "owner": owner,
        "trigger": trigger,
        "started_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
    }
    return run_id


def finish(run_id: str) -> None:
    _RUNS.pop(run_id, None)


def list_active() -> List[Dict]:
    return list(_RUNS.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_workflow_runs.py -v --import-mode=importlib`
Expected: PASS (all 4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/workflow_runs.py tests/test_workflow_runs.py
git commit -m "feat(workflow-runs): in-memory active-run registry

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Wrap all 3 `run_workflow()` call sites

**Files:**
- Modify: `src/agent_tools/workflow_tool.py`
- Modify: `routes/workflow_routes.py`
- Modify: `src/task_scheduler.py`
- Test: `tests/test_workflow_tool_active_run.py` (new)
- Test: `tests/test_workflow_routes_active_run.py` (new)
- Test: `tests/test_task_scheduler_workflow_active_run.py` (new)

**Interfaces:**
- Consumes: `workflow_runs.start(workflow_id, workflow_name, owner, trigger)` / `.finish(run_id)`
  (Task 1).
- Produces: nothing new — this task only adds instrumentation around the 3 existing
  `run_workflow()` calls. `run_workflow()`'s own signature and behavior are unchanged.

**Read this before starting**: each of the 3 files below already has a `try`/`except` (or bare
call) surrounding its `run_workflow(...)` invocation. In every case, add ONLY a
`workflow_runs.start(...)` call immediately before the existing `try`, and a
`finally: workflow_runs.finish(run_id)` clause on the SAME existing `try` block — do not
restructure anything else in these functions.

- [ ] **Step 1: Write the 3 failing tests**

Create `tests/test_workflow_tool_active_run.py`:

```python
"""run_workflow_tool registers/deregisters an active run around run_workflow(),
even when run_workflow raises -- workflow_runs.finish() must always run via a
finally block. See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md."""
import asyncio

import pytest

import src.agent_tools.workflow_tool as workflow_tool_module
from src import workflow_runs


def test_run_workflow_tool_registers_and_deregisters_active_run(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    monkeypatch.setattr(
        workflow_tool_module.store, "get_workflow",
        lambda wid: {"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []},
    )

    captured = {}

    async def fake_run_workflow(wf, inputs, ctx):
        captured["active_during_run"] = list(workflow_runs.list_active())
        return {"outputs": {}, "log": []}

    monkeypatch.setattr(workflow_tool_module, "run_workflow", fake_run_workflow)

    asyncio.run(workflow_tool_module.run_workflow_tool('{"id": "wf-1"}', {"owner": "alice"}))

    assert len(captured["active_during_run"]) == 1
    entry = captured["active_during_run"][0]
    assert entry["workflow_id"] == "wf-1"
    assert entry["workflow_name"] == "My Workflow"
    assert entry["owner"] == "alice"
    assert entry["trigger"] == "agent_tool"
    assert workflow_runs.list_active() == []


def test_run_workflow_tool_deregisters_even_on_error(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    monkeypatch.setattr(
        workflow_tool_module.store, "get_workflow",
        lambda wid: {"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []},
    )

    async def fake_run_workflow(wf, inputs, ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(workflow_tool_module, "run_workflow", fake_run_workflow)

    with pytest.raises(RuntimeError):
        asyncio.run(workflow_tool_module.run_workflow_tool('{"id": "wf-1"}', {"owner": "alice"}))

    assert workflow_runs.list_active() == []
```

**Before finalizing this test file**: confirm `run_workflow` is imported at MODULE level in
`routes/workflow_routes.py` (e.g. `from src.workflows.engine import run_workflow` near the top
of the file, not inside the `run()` handler itself). The route body calls it as a bare name
(`await run_workflow(...)`), which strongly implies a module-level import, but this must be
confirmed by reading the file before relying on
`monkeypatch.setattr(workflow_routes_module, "run_workflow", fake_run_workflow)` — if it turns
out to be a function-local import instead (like `task_scheduler.py`'s pattern in Step 5 below),
that monkeypatch target is wrong and the fake would never actually intercept the call, silently
running the real `run_workflow` against a malformed fake `wf` dict instead. If it is
function-local, patch `src.workflows.engine.run_workflow` directly instead (the same technique
Task 2's task_scheduler test already uses, for the same reason).

Create `tests/test_workflow_routes_active_run.py`:

```python
"""POST /api/workflows/{wid}/run registers/deregisters an active run around
run_workflow(), even when run_workflow raises. See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md."""
import asyncio
from types import SimpleNamespace

import pytest

import routes.workflow_routes as workflow_routes_module
from src import workflow_runs


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _request(user):
    return SimpleNamespace(state=SimpleNamespace(current_user=user))


def test_run_route_registers_and_deregisters_active_run(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    monkeypatch.setattr(
        workflow_routes_module.store, "get_workflow",
        lambda wid: {"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []},
    )

    captured = {}

    async def fake_run_workflow(wf, inputs, ctx):
        captured["active_during_run"] = list(workflow_runs.list_active())
        return {"outputs": {}, "log": []}

    monkeypatch.setattr(workflow_routes_module, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(workflow_routes_module, "get_current_user", lambda request: "alice")

    router = workflow_routes_module.setup_workflow_routes()
    handler = _route(router, "/api/workflows/{wid}/run", "POST")

    asyncio.run(handler(wid="wf-1", request=_request("alice"), body={}))

    assert len(captured["active_during_run"]) == 1
    entry = captured["active_during_run"][0]
    assert entry["workflow_id"] == "wf-1"
    assert entry["workflow_name"] == "My Workflow"
    assert entry["owner"] == "alice"
    assert entry["trigger"] == "api"
    assert workflow_runs.list_active() == []


def test_run_route_deregisters_even_on_error(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    monkeypatch.setattr(
        workflow_routes_module.store, "get_workflow",
        lambda wid: {"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []},
    )

    async def fake_run_workflow(wf, inputs, ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(workflow_routes_module, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(workflow_routes_module, "get_current_user", lambda request: "alice")

    router = workflow_routes_module.setup_workflow_routes()
    handler = _route(router, "/api/workflows/{wid}/run", "POST")

    with pytest.raises(RuntimeError):
        asyncio.run(handler(wid="wf-1", request=_request("alice"), body={}))

    assert workflow_runs.list_active() == []
```

Create `tests/test_task_scheduler_workflow_active_run.py`:

```python
"""_execute_workflow_task registers/deregisters an active run around
run_workflow(), even when run_workflow raises. See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md.

_execute_workflow_task's body never references `self`, so it's called here
via the class directly with self=None -- a lightweight way to unit-test one
method without constructing a full TaskScheduler (which needs a live DB
session). Its 3 local imports (get_workflow, run_workflow,
resolve_trigger_inputs) are patched at their real source modules, since a
`from X import Y` done freshly inside the method picks up whatever X.Y is at
call time.
"""
import asyncio
from types import SimpleNamespace

import pytest

import src.task_scheduler as task_scheduler_module
import src.workflows.engine as engine_module
import src.workflows.store as store_module
from src import workflow_runs


def _fake_task(owner="alice", action="wf-1", prompt=None):
    return SimpleNamespace(owner=owner, action=action, prompt=prompt)


def test_execute_workflow_task_registers_and_deregisters_active_run(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    monkeypatch.setattr(
        store_module, "get_workflow",
        lambda wid: {"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []},
    )

    captured = {}

    async def fake_run_workflow(wf, inputs, ctx):
        captured["active_during_run"] = list(workflow_runs.list_active())
        return {"outputs": {}, "log": []}

    monkeypatch.setattr(engine_module, "run_workflow", fake_run_workflow)

    task = _fake_task()
    summary, success = asyncio.run(
        task_scheduler_module.TaskScheduler._execute_workflow_task(None, task, context=None)
    )

    assert len(captured["active_during_run"]) == 1
    entry = captured["active_during_run"][0]
    assert entry["workflow_id"] == "wf-1"
    assert entry["workflow_name"] == "My Workflow"
    assert entry["owner"] == "alice"
    assert entry["trigger"] == "scheduled"
    assert workflow_runs.list_active() == []
    assert success is True


def test_execute_workflow_task_deregisters_even_on_error(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    monkeypatch.setattr(
        store_module, "get_workflow",
        lambda wid: {"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []},
    )

    async def fake_run_workflow(wf, inputs, ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine_module, "run_workflow", fake_run_workflow)

    task = _fake_task()
    with pytest.raises(RuntimeError):
        asyncio.run(
            task_scheduler_module.TaskScheduler._execute_workflow_task(None, task, context=None)
        )

    assert workflow_runs.list_active() == []
```

**Before finalizing this third test file**: read `src/workflows/triggers.py`'s
`resolve_trigger_inputs(wf, fixed, context)` (called, unmocked, inside `_execute_workflow_task`
with `wf={"id": "wf-1", "name": "My Workflow", "nodes": [], "edges": []}`, `fixed={}`,
`context=None` given the fake task above). Confirm it returns a plain dict (or otherwise doesn't
raise) for this minimal input combination. If it needs different/additional fields on `wf` or a
non-`None` `context` to avoid raising, adjust `_fake_task`'s `prompt` or the `context=` argument
passed to `_execute_workflow_task` accordingly — the exact values don't matter for this test,
only that `resolve_trigger_inputs` returns without raising before `run_workflow` is ever reached.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_workflow_tool_active_run.py tests/test_workflow_routes_active_run.py tests/test_task_scheduler_workflow_active_run.py -v --import-mode=importlib`
Expected: FAIL — each test's `monkeypatch.setattr(..., "run_workflow", ...)` or
`workflow_runs.list_active()` assertions fail because no wrapping exists yet (e.g. `captured`
stays empty, or `AttributeError` if `workflow_runs` isn't imported into a module yet — the exact
failure varies per file, but none should pass).

- [ ] **Step 3: Wrap `src/agent_tools/workflow_tool.py`'s call site**

Add `from src import workflow_runs` to this file's imports (find the existing import section
near the top of the file — it already imports `run_workflow` from somewhere in
`src.workflows.engine` and `WorkflowError`; add the new import alongside those, matching this
file's existing import style).

Find this block:

```python
    child = dict(ctx)
    child["_in_workflow"] = True
    child["owner"] = ctx.get("owner")
    try:
        result = await run_workflow(wf, inputs, child)
    except WorkflowError as e:
        return {"error": "; ".join(e.errors)}
```

Replace it with:

```python
    child = dict(ctx)
    child["_in_workflow"] = True
    child["owner"] = ctx.get("owner")
    run_id = workflow_runs.start(wid, wf.get("name") or wid, ctx.get("owner"), "agent_tool")
    try:
        result = await run_workflow(wf, inputs, child)
    except WorkflowError as e:
        return {"error": "; ".join(e.errors)}
    finally:
        workflow_runs.finish(run_id)
```

- [ ] **Step 4: Wrap `routes/workflow_routes.py`'s call site**

Add `from src import workflow_runs` to this file's imports (find the existing import section
near the top of the file — it already imports `run_workflow`, `get_current_user`, and other
`src.`/`core.` symbols; add the new import alongside those, matching this file's existing import
style).

Find this block:

```python
        owner = get_current_user(request)
        try:
            return await run_workflow(wf, body.get("inputs") or {}, {"owner": owner})
        except WorkflowError as e:
            raise _bad_request(e.errors)
```

Replace it with:

```python
        owner = get_current_user(request)
        run_id = workflow_runs.start(wid, wf.get("name") or wid, owner, "api")
        try:
            return await run_workflow(wf, body.get("inputs") or {}, {"owner": owner})
        except WorkflowError as e:
            raise _bad_request(e.errors)
        finally:
            workflow_runs.finish(run_id)
```

- [ ] **Step 5: Wrap `src/task_scheduler.py`'s call site**

`_execute_workflow_task` already has 3 LOCAL imports inside its own body (not at module level):
`from src.workflows.store import get_workflow`, `from src.workflows.engine import run_workflow`,
`from src.workflows.triggers import resolve_trigger_inputs`. Add a 4th local import alongside
them, matching this function's existing local-import style:

Find this block:

```python
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
```

Replace it with:

```python
        from src.workflows.store import get_workflow
        from src.workflows.engine import run_workflow
        from src.workflows.triggers import resolve_trigger_inputs
        from src import workflow_runs

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
        run_id = workflow_runs.start(wid, wf.get("name") or wid, task.owner, "scheduled")
        try:
            result = await run_workflow(wf, inputs, {"owner": task.owner})
        finally:
            workflow_runs.finish(run_id)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_workflow_tool_active_run.py tests/test_workflow_routes_active_run.py tests/test_task_scheduler_workflow_active_run.py -v --import-mode=importlib`
Expected: PASS (all 6 tests).

Also run the pre-existing test suites for these 3 files to confirm no regressions:

Run: `pytest tests/test_workflow_routes.py tests/test_workflow_tool.py -v --import-mode=importlib`
(if either file doesn't exist, skip it — search `tests/` for the closest-matching existing test
file names for `workflow_routes`/`workflow_tool`/`task_scheduler` before running, since exact
filenames may differ from this guess)
Expected: PASS, no new failures.

- [ ] **Step 7: Commit**

```bash
git add src/agent_tools/workflow_tool.py routes/workflow_routes.py src/task_scheduler.py tests/test_workflow_tool_active_run.py tests/test_workflow_routes_active_run.py tests/test_task_scheduler_workflow_active_run.py
git commit -m "feat(workflow-runs): wrap all 3 run_workflow() call sites with active-run tracking

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `GET /api/workflow-runs/active` route

**Files:**
- Create: `routes/workflow_runs_routes.py`
- Modify: `app.py`
- Test: `tests/test_workflow_runs_routes.py`

**Interfaces:**
- Consumes: `workflow_runs.list_active()` (Task 1).
- Produces: `setup_workflow_runs_routes() -> APIRouter`, mounted at `GET /api/workflow-runs/active`.
  Response: `{"active": [...]}` (same shape `list_active()` returns).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_workflow_runs_routes.py`:

```python
"""GET /api/workflow-runs/active exposes workflow_runs.list_active(), gated by
the same admin-only dependency the rest of the workflows subsystem uses
(routes/workflow_routes.py's own router). See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md."""
import asyncio

from core.middleware import require_admin
import routes.workflow_runs_routes as workflow_runs_routes_module
from src import workflow_runs


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def test_route_returns_active_workflow_runs(monkeypatch):
    monkeypatch.setattr(workflow_runs, "_RUNS", {})
    workflow_runs.start("wf-1", "My Workflow", "alice", "api")

    router = workflow_runs_routes_module.setup_workflow_runs_routes()
    handler = _route(router, "/api/workflow-runs/active", "GET")

    out = asyncio.run(handler())

    assert len(out["active"]) == 1
    assert out["active"][0]["workflow_id"] == "wf-1"


def test_router_is_admin_gated():
    router = workflow_runs_routes_module.setup_workflow_runs_routes()
    dependency_callables = [d.dependency for d in router.dependencies]
    assert require_admin in dependency_callables
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_workflow_runs_routes.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError: No module named 'routes.workflow_runs_routes'`.

- [ ] **Step 3: Create `routes/workflow_runs_routes.py`**

```python
"""Live workflow-run enumeration (Mission Control sub-project 2c).

Reads src.workflow_runs' in-memory registry of currently-executing workflow
runs. Admin-gated the same way as the rest of the workflows subsystem
(routes/workflow_routes.py) -- no new access-control decision, just reuse of
the existing gate. See
docs/superpowers/specs/2026-08-11-mission-control-workflow-runs-design.md.
"""
from fastapi import APIRouter, Depends

from core.middleware import require_admin
from src import workflow_runs


def setup_workflow_runs_routes() -> APIRouter:
    router = APIRouter(prefix="/api/workflow-runs", tags=["workflow-runs"],
                        dependencies=[Depends(require_admin)])

    @router.get("/active")
    async def get_active_workflow_runs():
        return {"active": workflow_runs.list_active()}

    return router
```

- [ ] **Step 4: Wire the router into `app.py`**

Find this existing block:

```python
# Workflows (headless node-graph automation: input/template/llm/tool/output)
from routes.workflow_routes import setup_workflow_routes
app.include_router(setup_workflow_routes())
```

Immediately after it, insert:

```python

# Live workflow-run tracking (Mission Control sub-project 2c)
from routes.workflow_runs_routes import setup_workflow_runs_routes
app.include_router(setup_workflow_runs_routes())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_workflow_runs_routes.py -v --import-mode=importlib`
Expected: PASS (both tests).

Run: `python -c "import app"` (from the repo root) to confirm `app.py` still imports cleanly with
the new router mounted.
Expected: no output, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add routes/workflow_runs_routes.py app.py tests/test_workflow_runs_routes.py
git commit -m "feat(workflow-runs): GET /api/workflow-runs/active route

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Mission Control "Workflow Runs" widget (8th widget)

**Files:**
- Modify: `static/index.html` (add the 8th `.mission-control-card`, WITH an
  `mission-control-open-link` button — unlike sub-project 2b's Active Agents widget, this one
  links to the existing Workflows editor panel)
- Modify: `static/js/missionControl.js` (`loadWorkflowRunsWidget`, wire into
  `refreshWidget`/`loadAllWidgets`/`init`)
- Modify: `tests/test_mission_control_ui.py` (new test + extend all 3 existing seam-guard tests,
  since this widget DOES have an `mc-open-*` button)

**Interfaces:**
- Consumes: `GET /api/workflow-runs/active` (Task 3), `$`/`esc`/`api`/`setCardBody`/`setCardError`
  (existing, `static/js/missionControl.js`), `#tool-workflows-btn` (existing Workflows panel's
  sidebar button, `static/index.html`).
- Produces: `loadWorkflowRunsWidget()`, added to `refreshWidget`'s dispatch and `loadAllWidgets()`.

**Note on the open-link target**: `#tool-workflows-btn` (and its rail counterpart
`#rail-workflows`) carry `style="display:none"` by default and are only revealed for admins by
`static/js/workflows.js`'s own `init()` (an `isAdmin()` check). This mirrors exactly how
sub-project 1's `mc-open-integrations` already targets `#tool-plugins-btn`, another
conditionally-gated button — no special-casing needed here either.

- [ ] **Step 1: Write the failing test and extend the 3 existing seam-guard tests**

Append to `tests/test_mission_control_ui.py`:

```python
def test_mission_control_has_workflow_runs_widget():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-widget="workflow-runs"' in html
    assert 'id="mc-body-workflow-runs"' in html

    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "loadWorkflowRunsWidget" in src
    assert "/api/workflow-runs/active" in src
```

Then, in the same file, find `test_mission_control_loadAllWidgets_wires_all_loaders` and add the
8th entry to its `expected_calls` list:

```python
    expected_calls = [
        'loadModelsWidget()',
        'loadHardwareWidget()',
        'loadTasksWidget()',
        'loadMemoryWidget()',
        'loadIntegrationsWidget()',
        'loadToolCallsWidget()',
        'loadActiveAgentsWidget()',
        'loadWorkflowRunsWidget()',
    ]
```

Find `test_mission_control_link_targets_exist_in_html` and add the 7th mapping to its `targets`
dict (Active Agents has no `mc-open-*` entry here, so this was 6 before, becomes 7):

```python
    targets = {
        'model-picker-btn': 'models',
        'hwmon': 'hardware',
        'rail-tasks': 'tasks',
        'tool-memory-btn': 'memory',
        'tool-plugins-btn': 'integrations',
        'tool-tool-calls-btn': 'tool-calls',
        'tool-workflows-btn': 'workflow-runs',
    }
```

Find `test_mission_control_open_handlers_wire_to_correct_targets` and update BOTH its hardcoded
handler count AND its `targets` dict (was 6 `mc-open-*` handlers, becomes 7 — Active Agents
never had one, this task adds the 7th):

```python
    starts = list(re.finditer(r"const\s+open\w+\s*=\s*\$\('(mc-open-[\w-]+)'\);", src))
    assert len(starts) == 7, "expected 7 mc-open-* handlers, found %d" % len(starts)

    targets = {
        'mc-open-models': 'model-picker-btn',
        'mc-open-hardware': 'hwmon',
        'mc-open-tasks': 'rail-tasks',
        'mc-open-memory': 'tool-memory-btn',
        'mc-open-integrations': 'tool-plugins-btn',
        'mc-open-tool-calls': 'tool-tool-calls-btn',
        'mc-open-workflow-runs': 'tool-workflows-btn',
    }
```

(Read the current file first to confirm the exact surrounding code these 2 tests have — the
`expected_calls`/`targets` list/dict contents above are exact, but match them into the test
bodies' existing structure exactly as currently written.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mission_control_ui.py -v --import-mode=importlib`
Expected: FAIL — the new test fails (widget doesn't exist), and both extended tests fail
(`loadWorkflowRunsWidget()` not in `loadAllWidgets()`'s body; only 6 `mc-open-*` handlers exist,
not 7).

- [ ] **Step 3: Add the 8th card to `static/index.html`**

Find this block (the Active Agents card, the last card inside `#mission-control-grid`,
immediately before the grid's closing `</div>`):

```html
        <div class="mission-control-card" id="mc-card-active-agents" data-widget="active-agents">
          <div class="mission-control-card-header">
            <h5>Active Agents</h5>
            <button class="btn mission-control-refresh" data-widget="active-agents" title="Refresh">&#x21bb;</button>
          </div>
          <div class="mission-control-card-body" id="mc-body-active-agents">Loading…</div>
        </div>
```

Immediately after it, insert:

```html
        <div class="mission-control-card" id="mc-card-workflow-runs" data-widget="workflow-runs">
          <div class="mission-control-card-header">
            <h5>Workflow Runs</h5>
            <button class="btn mission-control-refresh" data-widget="workflow-runs" title="Refresh">&#x21bb;</button>
          </div>
          <div class="mission-control-card-body" id="mc-body-workflow-runs">Loading…</div>
          <button class="btn mission-control-open-link" id="mc-open-workflow-runs">Open Workflows</button>
        </div>
```

- [ ] **Step 4: Add `loadWorkflowRunsWidget` to `static/js/missionControl.js`**

Find `loadActiveAgentsWidget`'s closing `}` followed by `function refreshWidget(widgetId) {`.
Immediately after `loadActiveAgentsWidget`'s closing brace and before `refreshWidget`, insert:

```javascript
async function loadWorkflowRunsWidget() {
  const body = $('mc-body-workflow-runs');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/workflow-runs/active');
    const items = data.active || [];
    const listHtml = items.map(function (w) {
      return '<div>' + esc(w.workflow_name || w.workflow_id || '?') + ' (' + esc(w.trigger || '?') + ')</div>';
    }).join('') || '<div>No workflows running right now</div>';
    setCardBody('workflow-runs', esc(items.length) + ' running<br>' + listHtml);
  } catch (e) {
    setCardError('workflow-runs', e.message);
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
  if (widgetId === 'workflow-runs') loadWorkflowRunsWidget();
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
  loadWorkflowRunsWidget();
}
```

Find the click-delegation block for `mc-body-active-agents` in `init()` (the last handler-related
block before `Modals.register(...)`):

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

Immediately after it, insert:

```javascript
  const openWorkflowRuns = $('mc-open-workflow-runs');
  if (openWorkflowRuns) openWorkflowRuns.addEventListener('click', function () {
    closeMissionControl();
    const workflowsBtn = $('tool-workflows-btn');
    if (workflowsBtn) workflowsBtn.click();
  });
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mission_control_ui.py -v --import-mode=importlib`
Expected: PASS (all 15 tests: the original 14 from sub-project 2b plus the 1 new one).

Run: `node --check static/js/missionControl.js`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/missionControl.js tests/test_mission_control_ui.py
git commit -m "feat(mission-control): Workflow Runs widget (8th widget)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

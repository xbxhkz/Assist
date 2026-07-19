# Workflows as Agent Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single admin-only builtin `run_workflow` tool that lists saved workflows or runs one by id via the shipped engine, so the agent can invoke a workflow from chat.

**Architecture:** A thin `RunWorkflowTool` handler in `src/agent_tools/` (a `SomeTool().execute(content, ctx)` async handler) that reuses `store.list_workflows`/`get_workflow` + `run_workflow`. Task 1 builds and unit-tests the handler in isolation; Task 2 wires it at every builtin-tool registration surface with a parity test. No engine/editor change.

**Tech Stack:** Python. Reuses `src/workflows/` (engine + store) unchanged.

## Global Constraints

- One tool, `run_workflow`. Handler `RunWorkflowTool().execute(content: str, ctx: dict) -> dict`. `content` is a JSON string of the call args (`{"id"?, "inputs"?, "action"?}`); `ctx` carries `owner` (and other keys).
- **List mode** (no `id`, or `action=="list"`): return the saved workflows, each with its input-node names. **Run mode** (`id` given): load + `await run_workflow(wf, inputs, child_ctx)` + return outputs and a status tally.
- **Admin-only** — the tool goes in `NON_ADMIN_BLOCKED_TOOLS` and **NOT** in `PLAN_MODE_READONLY_TOOLS` (it executes). This is a security boundary: the engine's inner-tool dispatch is ungated, so only admins may run a workflow.
- **Recursion guard:** the handler refuses when `ctx.get("_in_workflow")` is set, and it passes a **copied** child ctx with `_in_workflow=True` to the engine (never mutating the caller's ctx). Blocks all workflow-invokes-workflow nesting.
- The handler NEVER raises into the agent loop — every failure returns `{"error": …}`.
- No test runs a real model/tool: `store` and `run_workflow` are mocked at the seam.
- Register at EVERY surface (a missing one silently half-registers): `TOOL_HANDLERS`, `TOOL_TAGS`, `FUNCTION_TOOL_SCHEMAS`, `NON_ADMIN_BLOCKED_TOOLS`, `BUILTIN_TOOL_DESCRIPTIONS`, `agent_loop.TOOL_SECTIONS` + `_DOMAIN_TOOL_MAP`, and a parity test.
- pytest `--import-mode=importlib`. Stage specific files (never `git add -A`). Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Commit directly to `dev`.

---

### Task 1: The `run_workflow` tool handler

**Files:**
- Create: `src/agent_tools/workflow_tool.py`
- Test: `tests/test_workflow_tool.py`

**Interfaces:**
- Produces: `class RunWorkflowTool` with `async def execute(self, content, ctx) -> dict`, and the module-level `async def run_workflow_tool(content, ctx) -> dict` it delegates to.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_tool.py`:

```python
import asyncio
import json

import src.agent_tools.workflow_tool as wt


def _run(coro):
    return asyncio.run(coro)


def _exec(content, ctx=None):
    return _run(wt.RunWorkflowTool().execute(content, ctx or {}))


def test_list_mode_returns_workflows_with_input_names(monkeypatch):
    monkeypatch.setattr(wt.store, "list_workflows", lambda: [{"id": "flow1", "name": "Flow One", "nodes": 2}])
    monkeypatch.setattr(wt.store, "get_workflow", lambda wid: {
        "id": "flow1", "name": "Flow One",
        "nodes": [{"id": "i", "type": "input", "config": {"name": "topic"}},
                  {"id": "o", "type": "output", "config": {"name": "answer"}}], "edges": []})
    out = _exec("")
    data = json.loads(out["output"])
    assert data["workflows"] == [{"id": "flow1", "name": "Flow One", "inputs": ["topic"]}]


def test_action_list_also_lists(monkeypatch):
    monkeypatch.setattr(wt.store, "list_workflows", lambda: [])
    out = _exec(json.dumps({"action": "list"}))
    assert json.loads(out["output"]) == {"workflows": []}


def test_run_mode_runs_and_summarizes(monkeypatch):
    wf = {"id": "flow1", "name": "F", "nodes": [], "edges": []}
    monkeypatch.setattr(wt.store, "get_workflow", lambda wid: wf if wid == "flow1" else None)
    seen = {}

    async def fake_run(w, inputs, ctx, **kw):
        seen["inputs"] = inputs
        seen["ctx"] = ctx
        return {"outputs": {"answer": "hi"}, "log": [{"node": "o", "status": "ok"}]}
    monkeypatch.setattr(wt, "run_workflow", fake_run)

    out = _exec(json.dumps({"id": "flow1", "inputs": {"topic": "AI"}}), {"owner": "admin"})
    assert "answer" in out["output"] and "1 ok" in out["output"]
    assert seen["inputs"] == {"topic": "AI"}
    assert seen["ctx"]["owner"] == "admin" and seen["ctx"]["_in_workflow"] is True


def test_missing_workflow_is_error_not_raise(monkeypatch):
    monkeypatch.setattr(wt.store, "get_workflow", lambda wid: None)
    out = _exec(json.dumps({"id": "nope"}))
    assert "not found" in out["error"]


def test_invalid_graph_surfaces_as_error(monkeypatch):
    monkeypatch.setattr(wt.store, "get_workflow", lambda wid: {"nodes": [], "edges": []})

    async def bad_run(w, inputs, ctx, **kw):
        raise wt.WorkflowError(["cycle detected"])
    monkeypatch.setattr(wt, "run_workflow", bad_run)
    out = _exec(json.dumps({"id": "flow1"}))
    assert "cycle detected" in out["error"]


def test_recursion_guard_refuses_when_in_workflow(monkeypatch):
    # even a run request refuses if called from inside a workflow
    called = {"run": False}

    async def fake_run(*a, **k):
        called["run"] = True
        return {"outputs": {}, "log": []}
    monkeypatch.setattr(wt, "run_workflow", fake_run)
    out = _exec(json.dumps({"id": "flow1"}), {"_in_workflow": True})
    assert "error" in out and called["run"] is False


def test_child_ctx_does_not_mutate_caller(monkeypatch):
    monkeypatch.setattr(wt.store, "get_workflow", lambda wid: {"nodes": [], "edges": []})

    async def fake_run(w, inputs, ctx, **kw):
        return {"outputs": {}, "log": []}
    monkeypatch.setattr(wt, "run_workflow", fake_run)
    caller_ctx = {"owner": "admin"}
    _exec(json.dumps({"id": "flow1"}), caller_ctx)
    assert "_in_workflow" not in caller_ctx     # caller's ctx untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_tool.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.agent_tools.workflow_tool'`).

- [ ] **Step 3: Write the implementation**

Create `src/agent_tools/workflow_tool.py`:

```python
"""The `run_workflow` builtin tool: list saved workflows or run one by id via the
shipped engine. Admin-only (registered in NON_ADMIN_BLOCKED_TOOLS) — the engine's
inner-tool dispatch is ungated, so only admins may run a workflow. A ctx
`_in_workflow` flag blocks workflow-invokes-workflow nesting."""
import json

from src.workflows import store
from src.workflows.engine import run_workflow
from src.workflows.model import WorkflowError


def _input_names(wf):
    return [nm for n in (wf.get("nodes") or [])
            if n.get("type") == "input"
            for nm in [(n.get("config") or {}).get("name")] if nm]


def _parse_args(content):
    try:
        args = json.loads(content) if content and content.strip() else {}
    except (ValueError, TypeError):
        return {}
    return args if isinstance(args, dict) else {}


async def run_workflow_tool(content, ctx):
    ctx = ctx or {}
    if ctx.get("_in_workflow"):
        return {"error": "run_workflow cannot be called from within a workflow (no nesting)"}
    args = _parse_args(content)
    wid = (args.get("id") or "").strip()

    if not wid or args.get("action") == "list":
        items = []
        for w in store.list_workflows():
            wf = store.get_workflow(w.get("id")) or {}
            items.append({"id": w.get("id"), "name": w.get("name"), "inputs": _input_names(wf)})
        return {"output": json.dumps({"workflows": items}, indent=2)}

    wf = store.get_workflow(wid)
    if not wf:
        return {"error": f"workflow '{wid}' not found"}
    inputs = args.get("inputs")
    inputs = inputs if isinstance(inputs, dict) else {}
    child = dict(ctx)
    child["_in_workflow"] = True
    child["owner"] = ctx.get("owner")
    try:
        result = await run_workflow(wf, inputs, child)
    except WorkflowError as e:
        return {"error": "; ".join(e.errors)}
    outputs = result.get("outputs") or {}
    log = result.get("log") or []
    oks = sum(1 for e in log if e.get("status") == "ok")
    errs = sum(1 for e in log if e.get("status") == "error")
    skips = sum(1 for e in log if e.get("status") == "skipped")
    return {"output": f"Outputs: {json.dumps(outputs)} · {oks} ok, {errs} error, {skips} skipped"}


class RunWorkflowTool:
    async def execute(self, content, ctx):
        return await run_workflow_tool(content, ctx)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workflow_tool.py --import-mode=importlib -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/agent_tools/workflow_tool.py tests/test_workflow_tool.py
git commit -m "feat(workflows): run_workflow agent tool handler (list/run + recursion guard)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Register the tool at every surface

**Files:**
- Modify: `src/agent_tools/__init__.py` (import + `TOOL_HANDLERS` + `TOOL_TAGS`)
- Modify: `src/tool_schemas.py` (function schema + any tool-name list)
- Modify: `src/tool_security.py` (`NON_ADMIN_BLOCKED_TOOLS`)
- Modify: `src/tool_index.py` (`TOOL_INDEX` description)
- Modify: `src/agent_loop.py` (`TOOL_SECTIONS` + `_DOMAIN_TOOL_MAP`)
- Test: `tests/test_workflow_tool_registration.py`

**Interfaces:**
- Consumes (Task 1): `RunWorkflowTool` from `src.agent_tools.workflow_tool`.
- Produces: `run_workflow` present at every registration surface; absent from `PLAN_MODE_READONLY_TOOLS`.

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_workflow_tool_registration.py`:

```python
def test_run_workflow_registered_everywhere():
    from src.agent_tools import TOOL_HANDLERS, TOOL_TAGS
    from src.agent_tools.workflow_tool import RunWorkflowTool
    assert "run_workflow" in TOOL_HANDLERS
    assert "run_workflow" in TOOL_TAGS
    # handler is the tool's execute
    assert TOOL_HANDLERS["run_workflow"].__self__.__class__ is RunWorkflowTool


def test_run_workflow_is_admin_only_and_not_plan_readonly():
    import src.tool_security as ts
    assert "run_workflow" in ts.NON_ADMIN_BLOCKED_TOOLS
    # executes side effects -> must NOT be plan-mode read-only
    assert "run_workflow" not in getattr(ts, "PLAN_MODE_READONLY_TOOLS", set())


def test_run_workflow_has_schema():
    from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
    names = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS if s.get("type") == "function"}
    assert "run_workflow" in names


def test_run_workflow_in_index_and_agent_loop():
    import src.tool_index as ti
    import src.agent_loop as al
    assert "run_workflow" in ti.BUILTIN_TOOL_DESCRIPTIONS
    assert "run_workflow" in al.TOOL_SECTIONS
    assert any("run_workflow" in tools for tools in al._DOMAIN_TOOL_MAP.values())
```

These module-level names are verified against the current files: `FUNCTION_TOOL_SCHEMAS`
(`src/tool_schemas.py:24`), `BUILTIN_TOOL_DESCRIPTIONS` (`src/tool_index.py:69`),
`NON_ADMIN_BLOCKED_TOOLS`/`PLAN_MODE_READONLY_TOOLS` (`src/tool_security.py:40`/`103`),
`TOOL_SECTIONS`/`_DOMAIN_TOOL_MAP` (`src/agent_loop.py:324`/`294`). If any has moved, fix
BOTH the registration and the assertion to the real name — never weaken an assertion to pass.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_tool_registration.py --import-mode=importlib -q`
Expected: FAIL (not registered anywhere yet).

- [ ] **Step 3: Register at each surface**

Read each file's current structure first, then add `run_workflow` following the sibling pattern:

1. **`src/agent_tools/__init__.py`:** add `from .workflow_tool import RunWorkflowTool` with the other `from .` imports (~lines 22-26); add `"run_workflow": RunWorkflowTool().execute,` to the `TOOL_HANDLERS` dict (before its closing `}` ~line 86); add `"run_workflow"` to the `TOOL_TAGS` set (~line 100+).

2. **`src/tool_schemas.py`:** add to the `FUNCTION_TOOL_SCHEMAS` list (`:24`; model the `list_models` entry) an object:
```python
    {
        "type": "function",
        "function": {
            "name": "run_workflow",
            "description": "Run a saved workflow (a stored multi-step LLM/tool/branch graph) by id and return its outputs, or list available workflows. Call with no id (or action='list') to list saved workflows and each one's input names; call with id + inputs to run one. Admin only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The workflow id to run. Omit (or action='list') to list available workflows."},
                    "action": {"type": "string", "description": "Set to 'list' to list workflows."},
                    "inputs": {"type": "object", "description": "Input name -> value for the workflow's input nodes."}
                },
                "required": []
            }
        }
    },
```
If the module also keeps a flat tool-name list/set, add `"run_workflow"` there too.

3. **`src/tool_security.py`:** add `"run_workflow"` to the `NON_ADMIN_BLOCKED_TOOLS` set (the set opened ~line 40). Do NOT add it to `PLAN_MODE_READONLY_TOOLS`.

4. **`src/tool_index.py`:** add to the `BUILTIN_TOOL_DESCRIPTIONS` dict (`:69`): `"run_workflow": "Run a saved workflow by id (or list saved workflows) via the workflow engine.",`.

5. **`src/agent_loop.py`:**
   - `TOOL_SECTIONS` (~line 324) — add an entry modeling the `list_models`/`manage_tasks` sections (~lines 525/529):
     ```python
     "run_workflow": "- ```run_workflow``` — Run a saved workflow by id, or list saved workflows. Args (JSON): {\"id\": \"<workflow id>\", \"inputs\": {\"name\": \"value\"}} or {\"action\": \"list\"}. Admin only.",
     ```
   - `_DOMAIN_TOOL_MAP` (~line 294) — add `"run_workflow"` to the automation-related domain that contains `manage_tasks` (the `notes_calendar_tasks` set at ~line 299, or the most appropriate existing domain); if there's a dedicated automation domain, prefer that.

- [ ] **Step 4: Run the parity test + import smoke**

Run: `python -m pytest tests/test_workflow_tool_registration.py --import-mode=importlib -q` (Expected: PASS, 4 passed) and `python -c "import app"` (no error — confirms all the registration modules still import cleanly).

- [ ] **Step 5: Run the Task-1 handler suite (no regression)**

Run: `python -m pytest tests/test_workflow_tool.py --import-mode=importlib -q`
Expected: PASS (7 passed — the handler is unchanged; registration is additive).

- [ ] **Step 6: Commit**

```bash
git add src/agent_tools/__init__.py src/tool_schemas.py src/tool_security.py src/tool_index.py src/agent_loop.py tests/test_workflow_tool_registration.py
git commit -m "feat(workflows): register run_workflow at every tool surface (admin-only)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Both tasks are TDD.** Task 1's handler is fully unit-tested with `store`/`run_workflow` mocked (no real model/tool). Task 2 proves registration at every surface via a parity test + `import app`.
- **Admin-only is a security boundary, not a style choice** — `run_workflow` MUST be in `NON_ADMIN_BLOCKED_TOOLS` and MUST NOT be in `PLAN_MODE_READONLY_TOOLS`. Do not weaken this.
- **Verify the real module-level names** in Task 2 (`TOOL_SCHEMAS`, `TOOL_INDEX`, `PLAN_MODE_READONLY_TOOLS`, `TOOL_SECTIONS`, `_DOMAIN_TOOL_MAP`) against the files before writing — if one differs, fix the registration AND the assertion to the real name; never weaken an assertion to pass.
- The "ask the agent to run a workflow from chat" end-to-end path needs a live model — it's a short **manual** check, owed by the human, not automated here.
- Scope: one tool. Do NOT build per-workflow dynamic tools, workflow nesting/composition, a non-admin path, or streaming (v1 non-goals).

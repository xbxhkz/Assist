# Workflow Engine Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless workflow engine — define a workflow as a JSON node-graph (input/template/llm/tool/output nodes + edges) and run it as a data-flow DAG, returning outputs + a per-node run log.

**Architecture:** A new `src/workflows/` package with four focused modules — `model.py` (shapes, ports, validate, topo_sort), `nodes.py` (per-type executors + default model/tool calls), `engine.py` (run_workflow), `store.py` (JSON persistence) — plus an admin-gated `routes/workflow_routes.py`. LLM nodes reuse `resolve_endpoint`; tool nodes reuse `TOOL_HANDLERS`. Both are injectable so tests never touch a real model or tool.

**Tech Stack:** Python, FastAPI, httpx. Sub-project 1 of the visual workflow builder (the editor comes later and will edit this JSON).

## Global Constraints

- Node types are exactly: **input, template, llm, tool, output**. Ports per type: input `()→value`; template `{slots}→text`; llm `{slots}→text`; tool `{slots}→result`; output `value→()`. Input ports of template/llm/tool are **derived** from the `{slot}` names in their text config.
- **Text-only wires** for v1. No branching, loops, or typed ports.
- `model_call` and `tool_dispatch` are **injectable**; tests MUST mock them — no real model/endpoint/tool executes in any test.
- Engine is **partial-on-failure**: a node exception is logged (`status:"error"`), its downstream dependents are `status:"skipped"`, the run returns `{outputs, log}` and never crashes. Only an **invalid graph** raises (`WorkflowError` → route 400).
- Workflows persist as JSON under `DATA_DIR/workflows/`; ids are path-safe (reject `/ \ ..`), mirroring the Skills/LoRA safety pattern.
- Routes are **admin-gated**: `APIRouter(prefix="/api/workflows", dependencies=[Depends(require_admin)])` (a workflow runs the LLM + arbitrary tools; each tool's own gates still apply).
- Every pytest run uses `--import-mode=importlib`. Stage specific files (never `git add -A`). Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Workflow model — ports, validate, topo_sort

**Files:**
- Create: `src/workflows/__init__.py` (empty), `src/workflows/model.py`
- Test: `tests/test_workflow_model.py`

**Interfaces:**
- Produces: `NODE_TYPES`, `class WorkflowError(Exception)` (with `.errors: list[str]`), `slots_of(text) -> list[str]`, `input_ports(node) -> list[str]`, `output_ports(node) -> list[str]`, `validate(wf) -> list[str]`, `topo_sort(wf) -> list[str]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_model.py`:

```python
import pytest
import src.workflows.model as m


def _wf(nodes, edges):
    return {"id": "w1", "name": "W", "nodes": nodes, "edges": edges}


def test_slots_of_extracts_unique_ordered():
    assert m.slots_of("Hi {name}, you are {age}. Bye {name}") == ["name", "age"]
    assert m.slots_of("no slots") == []


def test_ports_per_type():
    assert m.input_ports({"type": "input", "config": {"name": "q"}}) == []
    assert m.output_ports({"type": "input", "config": {}}) == ["value"]
    assert m.input_ports({"type": "template", "config": {"template": "{a}-{b}"}}) == ["a", "b"]
    assert m.output_ports({"type": "template", "config": {}}) == ["text"]
    assert m.input_ports({"type": "llm", "config": {"prompt": "sum {doc}"}}) == ["doc"]
    assert m.output_ports({"type": "llm", "config": {}}) == ["text"]
    assert m.input_ports({"type": "tool", "config": {"args": "{path}"}}) == ["path"]
    assert m.output_ports({"type": "tool", "config": {}}) == ["result"]
    assert m.input_ports({"type": "output", "config": {}}) == ["value"]
    assert m.output_ports({"type": "output", "config": {}}) == []


def _linear():
    return _wf(
        [{"id": "i", "type": "input", "config": {"name": "q"}},
         {"id": "t", "type": "template", "config": {"template": "Q: {q}"}},
         {"id": "o", "type": "output", "config": {"name": "answer"}}],
        [{"from_node": "i", "from_port": "value", "to_node": "t", "to_port": "q"},
         {"from_node": "t", "from_port": "text", "to_node": "o", "to_port": "value"}],
    )


def test_validate_accepts_valid_graph():
    assert m.validate(_linear()) == []


def test_validate_flags_unknown_type_and_dup_ids():
    wf = _wf([{"id": "a", "type": "bogus", "config": {}},
              {"id": "a", "type": "input", "config": {"name": "q"}}], [])
    errs = m.validate(wf)
    assert any("unknown node type" in e for e in errs)
    assert any("duplicate node id" in e for e in errs)


def test_validate_flags_dangling_edge_and_bad_port():
    wf = _linear()
    wf["edges"].append({"from_node": "nope", "from_port": "value", "to_node": "o", "to_port": "value"})
    assert any("unknown node" in e for e in m.validate(wf))
    wf2 = _linear()
    wf2["edges"][0]["to_port"] = "zzz"
    assert any("invalid input port" in e for e in m.validate(wf2))


def test_validate_flags_unwired_slot():
    wf = _linear()
    wf["edges"] = [e for e in wf["edges"] if e["to_node"] != "t"]  # drop the wire into {q}
    assert any("unwired input port" in e for e in m.validate(wf))


def test_validate_detects_cycle():
    wf = _wf(
        [{"id": "t1", "type": "template", "config": {"template": "{x}"}},
         {"id": "t2", "type": "template", "config": {"template": "{y}"}}],
        [{"from_node": "t1", "from_port": "text", "to_node": "t2", "to_port": "y"},
         {"from_node": "t2", "from_port": "text", "to_node": "t1", "to_port": "x"}],
    )
    assert any("cycle" in e for e in m.validate(wf))


def test_topo_sort_orders_and_raises_on_cycle():
    assert m.topo_sort(_linear()) == ["i", "t", "o"]
    wf = _wf(
        [{"id": "t1", "type": "template", "config": {"template": "{x}"}},
         {"id": "t2", "type": "template", "config": {"template": "{y}"}}],
        [{"from_node": "t1", "from_port": "text", "to_node": "t2", "to_port": "y"},
         {"from_node": "t2", "from_port": "text", "to_node": "t1", "to_port": "x"}],
    )
    with pytest.raises(m.WorkflowError):
        m.topo_sort(wf)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_model.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.workflows'`).

- [ ] **Step 3: Write the implementation**

Create `src/workflows/__init__.py` (empty file). Create `src/workflows/model.py`:

```python
"""Workflow graph model: node/edge shapes, port derivation, validation, topo sort.

A workflow is {"id","name","nodes":[{"id","type","config"}],"edges":[
{"from_node","from_port","to_node","to_port"}]}. Text-only wires; the input
ports of template/llm/tool nodes are DERIVED from the {slot} names in their
text config, so the editor never hand-declares them."""
import re
from collections import deque

NODE_TYPES = ("input", "template", "llm", "tool", "output")

_SLOT_RE = re.compile(r"\{(\w+)\}")

# type -> the config key whose {slots} become that node's input ports
_SLOT_SOURCE = {"template": "template", "llm": "prompt", "tool": "args"}
_OUTPUT_PORT = {"input": "value", "template": "text", "llm": "text", "tool": "result"}


class WorkflowError(Exception):
    """Invalid workflow graph. `.errors` holds the human-readable reasons."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def slots_of(text):
    """Ordered unique {slot} names in `text`."""
    out = []
    for name in _SLOT_RE.findall(text or ""):
        if name not in out:
            out.append(name)
    return out


def input_ports(node):
    t = node.get("type")
    if t == "output":
        return ["value"]
    key = _SLOT_SOURCE.get(t)
    if not key:
        return []          # input nodes (and unknown types) take no wires
    return slots_of((node.get("config") or {}).get(key, ""))


def output_ports(node):
    port = _OUTPUT_PORT.get(node.get("type"))
    return [port] if port else []


def validate(wf):
    """Return a list of human-readable errors ([] means valid)."""
    errors = []
    nodes = wf.get("nodes") or []
    edges = wf.get("edges") or []
    by_id = {}
    for n in nodes:
        nid = n.get("id")
        if nid in by_id:
            errors.append(f"duplicate node id: {nid}")
        by_id[nid] = n
        if n.get("type") not in NODE_TYPES:
            errors.append(f"unknown node type: {n.get('type')} (node {nid})")
    for e in edges:
        src, dst = e.get("from_node"), e.get("to_node")
        if src not in by_id:
            errors.append(f"edge references unknown node: {src}")
        if dst not in by_id:
            errors.append(f"edge references unknown node: {dst}")
        if src in by_id and e.get("from_port") not in output_ports(by_id[src]):
            errors.append(f"invalid output port '{e.get('from_port')}' on node {src}")
        if dst in by_id and e.get("to_port") not in input_ports(by_id[dst]):
            errors.append(f"invalid input port '{e.get('to_port')}' on node {dst}")
    # every declared input port must be wired
    wired = {(e.get("to_node"), e.get("to_port")) for e in edges}
    for n in nodes:
        for port in input_ports(n):
            if (n.get("id"), port) not in wired:
                errors.append(f"unwired input port '{port}' on node {n.get('id')}")
    if not errors:
        try:
            topo_sort(wf)
        except WorkflowError as ex:
            errors.extend(ex.errors)
    return errors


def topo_sort(wf):
    """Node ids in execution order (Kahn). Raises WorkflowError on a cycle."""
    nodes = wf.get("nodes") or []
    edges = wf.get("edges") or []
    ids = [n.get("id") for n in nodes]
    indeg = {i: 0 for i in ids}
    adj = {i: [] for i in ids}
    for e in edges:
        src, dst = e.get("from_node"), e.get("to_node")
        if src in indeg and dst in indeg:
            adj[src].append(dst)
            indeg[dst] += 1
    q = deque([i for i in ids if indeg[i] == 0])
    order = []
    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    if len(order) != len(ids):
        raise WorkflowError(["cycle detected in workflow graph"])
    return order
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workflow_model.py --import-mode=importlib -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/workflows/__init__.py src/workflows/model.py tests/test_workflow_model.py
git commit -m "feat(workflows): graph model — ports, validate, topo_sort

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Node executors + default model/tool calls

**Files:**
- Create: `src/workflows/nodes.py`
- Test: `tests/test_workflow_nodes.py`

**Interfaces:**
- Consumes (Task 1): nothing at runtime (executors are standalone).
- Produces: `_fill(template, inputs) -> str`, `async run_input(config, run_inputs) -> dict`, `async run_template(config, inputs) -> dict`, `async run_llm(config, inputs, *, model_call) -> dict`, `async run_tool(config, inputs, ctx, *, tool_dispatch) -> dict`, `async run_output(config, inputs) -> dict`, `async default_model_call(prompt, model=None, system=None, owner=None) -> str`, `async default_tool_dispatch(tool, args, ctx) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_nodes.py`:

```python
import asyncio
import pytest
import src.workflows.nodes as nd


def _run(coro):
    return asyncio.run(coro)


def test_fill_substitutes_and_blanks_missing():
    assert nd._fill("Hi {name} ({name})", {"name": "Ada"}) == "Hi Ada (Ada)"
    assert nd._fill("Hi {missing}", {}) == "Hi "
    assert nd._fill("plain", {}) == "plain"


def test_run_input_uses_run_value_then_default():
    assert _run(nd.run_input({"name": "q"}, {"q": "hello"})) == {"value": "hello"}
    assert _run(nd.run_input({"name": "q", "default": "d"}, {})) == {"value": "d"}
    assert _run(nd.run_input({"name": "q"}, {})) == {"value": ""}


def test_run_template_fills():
    assert _run(nd.run_template({"template": "Q: {q}"}, {"q": "why"})) == {"text": "Q: why"}


def test_run_llm_calls_injected_model():
    seen = {}
    async def fake_model(prompt, model=None, system=None):
        seen.update(prompt=prompt, model=model, system=system)
        return "ANSWER"
    out = _run(nd.run_llm({"prompt": "sum {doc}", "model": "m1", "system": "sys"},
                          {"doc": "text"}, model_call=fake_model))
    assert out == {"text": "ANSWER"}
    assert seen["prompt"] == "sum text" and seen["model"] == "m1" and seen["system"] == "sys"


def test_run_tool_calls_injected_dispatch():
    seen = {}
    async def fake_dispatch(tool, args, ctx):
        seen.update(tool=tool, args=args, ctx=ctx)
        return "RESULT"
    out = _run(nd.run_tool({"tool": "find_files", "args": "{path}"},
                           {"path": "/tmp"}, {"owner": "u"}, tool_dispatch=fake_dispatch))
    assert out == {"result": "RESULT"}
    assert seen["tool"] == "find_files" and seen["args"] == "/tmp" and seen["ctx"] == {"owner": "u"}


def test_run_output_passthrough():
    assert _run(nd.run_output({"name": "answer"}, {"value": "v"})) == {}


def test_default_tool_dispatch_unknown_tool_raises(monkeypatch):
    monkeypatch.setattr(nd, "_tool_handlers", lambda: {})
    with pytest.raises(RuntimeError):
        _run(nd.default_tool_dispatch("nope", "", {}))


def test_default_tool_dispatch_returns_output_and_raises_on_error(monkeypatch):
    async def ok(content, ctx):
        return {"output": "OK", "exit_code": 0}
    async def bad(content, ctx):
        return {"error": "boom", "exit_code": 1}
    monkeypatch.setattr(nd, "_tool_handlers", lambda: {"ok": ok, "bad": bad})
    assert _run(nd.default_tool_dispatch("ok", "args", {})) == "OK"
    with pytest.raises(RuntimeError):
        _run(nd.default_tool_dispatch("bad", "args", {}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_nodes.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.workflows.nodes'`).

- [ ] **Step 3: Write the implementation**

Create `src/workflows/nodes.py`:

```python
"""Per-type node executors + the default LLM/tool calls.

`model_call` and `tool_dispatch` are injected by the engine so tests never hit a
real endpoint or tool. Defaults reuse the app's existing infrastructure:
resolve_endpoint (chat/completions) and TOOL_HANDLERS."""
import re

_SLOT_RE = re.compile(r"\{(\w+)\}")


def _fill(template, inputs):
    """Replace each {slot} with str(inputs[slot]); missing slots become ''."""
    def sub(match):
        return str(inputs.get(match.group(1), ""))
    return _SLOT_RE.sub(sub, template or "")


async def run_input(config, run_inputs):
    name = config.get("name", "")
    if name in (run_inputs or {}):
        return {"value": str((run_inputs or {})[name])}
    return {"value": str(config.get("default", ""))}


async def run_template(config, inputs):
    return {"text": _fill(config.get("template", ""), inputs)}


async def run_llm(config, inputs, *, model_call):
    prompt = _fill(config.get("prompt", ""), inputs)
    text = await model_call(prompt, model=config.get("model"), system=config.get("system"))
    return {"text": str(text)}


async def run_tool(config, inputs, ctx, *, tool_dispatch):
    args = _fill(config.get("args", ""), inputs)
    result = await tool_dispatch(config.get("tool", ""), args, ctx)
    return {"result": str(result)}


async def run_output(config, inputs):
    # The engine records inputs["value"] under config["name"]; nothing to emit.
    return {}


# ── defaults (reuse existing app infrastructure) ──

def _tool_handlers():
    from src.agent_tools import TOOL_HANDLERS  # lazy: avoids import cycles
    return TOOL_HANDLERS


async def default_model_call(prompt, model=None, system=None, owner=None):
    import httpx
    from src.endpoint_resolver import resolve_endpoint
    url, resolved_model, headers = resolve_endpoint("default", owner=owner)
    if not url:
        raise RuntimeError("no default model endpoint configured")
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    body = {"model": model or resolved_model, "messages": messages, "stream": False}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(url, json=body, headers=headers or {})
        r.raise_for_status()
        data = r.json()
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")


async def default_tool_dispatch(tool, args, ctx):
    handler = _tool_handlers().get(tool)
    if not handler:
        raise RuntimeError(f"unknown tool: {tool}")
    res = await handler(args, ctx or {})
    if isinstance(res, dict):
        if res.get("error"):
            raise RuntimeError(str(res["error"]))
        return str(res.get("output", ""))
    return str(res)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workflow_nodes.py --import-mode=importlib -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/workflows/nodes.py tests/test_workflow_nodes.py
git commit -m "feat(workflows): node executors + default model/tool calls (injectable)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Execution engine (`run_workflow`)

**Files:**
- Create: `src/workflows/engine.py`
- Test: `tests/test_workflow_engine.py`

**Interfaces:**
- Consumes: `model.validate`, `model.topo_sort`, `model.WorkflowError`; `nodes.run_input/run_template/run_llm/run_tool/run_output`, `nodes.default_model_call`, `nodes.default_tool_dispatch`.
- Produces: `async run_workflow(wf, inputs=None, ctx=None, *, model_call=None, tool_dispatch=None) -> {"outputs": dict, "log": list}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_engine.py`:

```python
import asyncio
import pytest
import src.workflows.engine as eng
from src.workflows.model import WorkflowError


def _run(coro):
    return asyncio.run(coro)


def _wf():
    return {"id": "w", "name": "W", "nodes": [
        {"id": "i", "type": "input", "config": {"name": "q"}},
        {"id": "t", "type": "template", "config": {"template": "Q: {q}"}},
        {"id": "l", "type": "llm", "config": {"prompt": "{p}"}},
        {"id": "o", "type": "output", "config": {"name": "answer"}},
    ], "edges": [
        {"from_node": "i", "from_port": "value", "to_node": "t", "to_port": "q"},
        {"from_node": "t", "from_port": "text", "to_node": "l", "to_port": "p"},
        {"from_node": "l", "from_port": "text", "to_node": "o", "to_port": "value"},
    ]}


async def _fake_model(prompt, model=None, system=None):
    return f"ECHO[{prompt}]"


def test_runs_linear_workflow_and_logs():
    res = _run(eng.run_workflow(_wf(), {"q": "why"}, model_call=_fake_model))
    assert res["outputs"] == {"answer": "ECHO[Q: why]"}
    assert [e["status"] for e in res["log"]] == ["ok", "ok", "ok", "ok"]
    assert [e["node"] for e in res["log"]] == ["i", "t", "l", "o"]
    assert all("ms" in e for e in res["log"])


def test_invalid_graph_raises():
    bad = _wf()
    bad["edges"].append({"from_node": "zz", "from_port": "value", "to_node": "o", "to_port": "value"})
    with pytest.raises(WorkflowError):
        _run(eng.run_workflow(bad, {}, model_call=_fake_model))


def test_node_failure_skips_dependents_and_returns_partial():
    async def boom(prompt, model=None, system=None):
        raise RuntimeError("model down")
    res = _run(eng.run_workflow(_wf(), {"q": "why"}, model_call=boom))
    status = {e["node"]: e["status"] for e in res["log"]}
    assert status["i"] == "ok" and status["t"] == "ok"
    assert status["l"] == "error" and status["o"] == "skipped"
    assert "model down" in (next(e for e in res["log"] if e["node"] == "l")["error"] or "")
    assert res["outputs"] == {}          # partial: the output never resolved


def test_tool_node_uses_injected_dispatch():
    wf = {"id": "w", "name": "W", "nodes": [
        {"id": "i", "type": "input", "config": {"name": "p"}},
        {"id": "tl", "type": "tool", "config": {"tool": "find_files", "args": "{p}"}},
        {"id": "o", "type": "output", "config": {"name": "files"}},
    ], "edges": [
        {"from_node": "i", "from_port": "value", "to_node": "tl", "to_port": "p"},
        {"from_node": "tl", "from_port": "result", "to_node": "o", "to_port": "value"},
    ]}
    async def fake_dispatch(tool, args, ctx):
        return f"{tool}:{args}"
    res = _run(eng.run_workflow(wf, {"p": "/tmp"}, {"owner": "u"}, tool_dispatch=fake_dispatch))
    assert res["outputs"] == {"files": "find_files:/tmp"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_engine.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.workflows.engine'`).

- [ ] **Step 3: Write the implementation**

Create `src/workflows/engine.py`:

```python
"""Workflow execution: validate -> topo sort -> run each node, flowing text
values along edges. Partial-on-failure: a node error is logged, its dependents
are skipped, and the run still returns {outputs, log}."""
import logging
import time

from src.workflows import nodes as N
from src.workflows.model import WorkflowError, topo_sort, validate

logger = logging.getLogger(__name__)

_LOG_MAX = 500  # truncate node output in the run log


async def _run_node(node, node_inputs, run_inputs, ctx, model_call, tool_dispatch):
    t = node.get("type")
    cfg = node.get("config") or {}
    if t == "input":
        return await N.run_input(cfg, run_inputs)
    if t == "template":
        return await N.run_template(cfg, node_inputs)
    if t == "llm":
        return await N.run_llm(cfg, node_inputs, model_call=model_call)
    if t == "tool":
        return await N.run_tool(cfg, node_inputs, ctx, tool_dispatch=tool_dispatch)
    if t == "output":
        return await N.run_output(cfg, node_inputs)
    raise RuntimeError(f"unknown node type: {t}")


async def run_workflow(wf, inputs=None, ctx=None, *, model_call=None, tool_dispatch=None):
    """Run `wf` with `inputs` -> {"outputs": {name: value}, "log": [entry]}.

    Raises WorkflowError if the graph is invalid (the route maps that to 400)."""
    errs = validate(wf)
    if errs:
        raise WorkflowError(errs)
    model_call = model_call or N.default_model_call
    tool_dispatch = tool_dispatch or N.default_tool_dispatch
    run_inputs = inputs or {}
    by_id = {n["id"]: n for n in wf.get("nodes") or []}
    edges = wf.get("edges") or []

    produced = {}      # node id -> {port: value}
    failed = set()     # nodes that errored or were skipped
    outputs = {}
    log = []

    for nid in topo_sort(wf):
        node = by_id[nid]
        incoming = [e for e in edges if e.get("to_node") == nid]
        upstream_bad = any(e.get("from_node") in failed for e in incoming)
        if upstream_bad:
            failed.add(nid)
            log.append({"node": nid, "type": node.get("type"), "status": "skipped",
                        "output": "", "error": None, "ms": 0})
            continue
        node_inputs = {e["to_port"]: produced.get(e["from_node"], {}).get(e["from_port"], "")
                       for e in incoming}
        started = time.monotonic()
        try:
            out = await _run_node(node, node_inputs, run_inputs, ctx, model_call, tool_dispatch)
            produced[nid] = out
            if node.get("type") == "output":
                outputs[(node.get("config") or {}).get("name", nid)] = node_inputs.get("value", "")
            shown = str(next(iter(out.values()), "")) if out else str(node_inputs.get("value", ""))
            log.append({"node": nid, "type": node.get("type"), "status": "ok",
                        "output": shown[:_LOG_MAX], "error": None,
                        "ms": int((time.monotonic() - started) * 1000)})
        except Exception as e:
            failed.add(nid)
            logger.info("workflow node %s failed: %s", nid, e)
            log.append({"node": nid, "type": node.get("type"), "status": "error",
                        "output": "", "error": str(e),
                        "ms": int((time.monotonic() - started) * 1000)})
    return {"outputs": outputs, "log": log}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workflow_engine.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/workflows/engine.py tests/test_workflow_engine.py
git commit -m "feat(workflows): DAG execution engine (partial-on-failure + run log)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Workflow store (JSON persistence)

**Files:**
- Create: `src/workflows/store.py`
- Test: `tests/test_workflow_store.py`

**Interfaces:**
- Produces: `workflows_dir() -> str`, `_safe_id(wid) -> str`, `list_workflows() -> list[dict]`, `get_workflow(wid) -> dict|None`, `save_workflow(wf) -> dict`, `delete_workflow(wid) -> bool`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_store.py`:

```python
import pytest
import src.workflows.store as st


def test_safe_id_rejects_traversal():
    for bad in ["a/b", "a\\b", "..", "../x", ""]:
        with pytest.raises(ValueError):
            st._safe_id(bad)
    assert st._safe_id("my-flow") == "my-flow"


def test_save_get_list_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "workflows_dir", lambda: str(tmp_path))
    wf = {"id": "flow1", "name": "Flow One", "nodes": [], "edges": []}
    saved = st.save_workflow(wf)
    assert saved["id"] == "flow1"
    assert st.get_workflow("flow1")["name"] == "Flow One"
    assert [w["id"] for w in st.list_workflows()] == ["flow1"]
    assert st.delete_workflow("flow1") is True
    assert st.get_workflow("flow1") is None
    assert st.list_workflows() == []
    assert st.delete_workflow("flow1") is False


def test_save_slugifies_id_from_name_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "workflows_dir", lambda: str(tmp_path))
    saved = st.save_workflow({"name": "My Cool Flow!", "nodes": [], "edges": []})
    assert saved["id"] == "my-cool-flow"
    assert st.get_workflow("my-cool-flow") is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_store.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'src.workflows.store'`).

- [ ] **Step 3: Write the implementation**

Create `src/workflows/store.py`:

```python
"""Workflow persistence: one JSON file per workflow under DATA_DIR/workflows.
Mirrors the path-safety of the Skills/LoRA stores (ids may not traverse)."""
import json
import os
import re

from src.constants import DATA_DIR


def workflows_dir():
    d = os.path.join(DATA_DIR, "workflows")
    os.makedirs(d, exist_ok=True)
    return d


def _safe_id(wid):
    if not wid or "/" in wid or "\\" in wid or ".." in wid:
        raise ValueError("unsafe workflow id")
    return os.path.basename(wid)


def _slugify(name):
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or "workflow"


def _path(wid):
    return os.path.join(workflows_dir(), _safe_id(wid) + ".json")


def list_workflows():
    out = []
    for fn in sorted(os.listdir(workflows_dir())):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(workflows_dir(), fn), "r", encoding="utf-8") as f:
                wf = json.load(f)
            out.append({"id": wf.get("id", fn[:-5]), "name": wf.get("name", fn[:-5]),
                        "nodes": len(wf.get("nodes") or [])})
        except (OSError, json.JSONDecodeError):
            continue
    return out


def get_workflow(wid):
    try:
        with open(_path(wid), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_workflow(wf):
    wid = wf.get("id") or _slugify(wf.get("name"))
    wf = dict(wf, id=_safe_id(wid))
    with open(_path(wf["id"]), "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2)
    return wf


def delete_workflow(wid):
    p = _path(wid)
    if os.path.isfile(p):
        os.remove(p)
        return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workflow_store.py --import-mode=importlib -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/workflows/store.py tests/test_workflow_store.py
git commit -m "feat(workflows): JSON store under DATA_DIR/workflows (path-safe)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Admin-gated routes (`/api/workflows`) + registration

**Files:**
- Create: `routes/workflow_routes.py`
- Modify: `app.py` (register the router beside the other `app.include_router(setup_*_routes())` calls, ~line 618)
- Test: `tests/test_workflow_routes.py`

**Interfaces:**
- Consumes: `store.list_workflows/get_workflow/save_workflow/delete_workflow`; `engine.run_workflow`; `model.validate`, `model.WorkflowError`.
- Produces: `setup_workflow_routes() -> APIRouter` at prefix `/api/workflows`, admin-gated.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_routes.py`:

```python
import routes.workflow_routes as wr


def test_router_is_admin_gated_and_prefixed():
    router = wr.setup_workflow_routes()
    assert router.prefix == "/api/workflows"
    assert router.dependencies, "router must carry the require_admin dependency"
    paths = {r.path for r in router.routes}
    assert "/api/workflows" in paths
    assert "/api/workflows/{wid}" in paths
    assert "/api/workflows/{wid}/run" in paths
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflow_routes.py --import-mode=importlib -q`
Expected: FAIL (`No module named 'routes.workflow_routes'`).

- [ ] **Step 3: Write the implementation**

Create `routes/workflow_routes.py`:

```python
"""Admin-gated workflow management + execution.

A workflow runs the LLM and arbitrary agent tools (each tool's own admin/consent
gates still apply on top), so the whole router is behind require_admin."""
import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from core.middleware import require_admin, get_current_user
from src.workflows import store
from src.workflows.engine import run_workflow
from src.workflows.model import WorkflowError, validate

logger = logging.getLogger(__name__)


def setup_workflow_routes() -> APIRouter:
    router = APIRouter(prefix="/api/workflows", dependencies=[Depends(require_admin)])

    @router.get("")
    async def list_workflows():
        return {"workflows": store.list_workflows()}

    @router.post("")
    async def save_workflow(body: dict = Body(...)):
        errs = validate(body)
        if errs:
            raise HTTPException(400, {"errors": errs})
        try:
            return store.save_workflow(body)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @router.get("/{wid}")
    async def get_workflow(wid: str):
        try:
            wf = store.get_workflow(wid)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not wf:
            raise HTTPException(404, "workflow not found")
        return wf

    @router.delete("/{wid}")
    async def delete_workflow(wid: str):
        try:
            return {"deleted": store.delete_workflow(wid)}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @router.post("/{wid}/run")
    async def run(wid: str, request: Request, body: dict = Body(default={})):
        try:
            wf = store.get_workflow(wid)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not wf:
            raise HTTPException(404, "workflow not found")
        owner = get_current_user(request)
        try:
            return await run_workflow(wf, body.get("inputs") or {}, {"owner": owner})
        except WorkflowError as e:
            raise HTTPException(400, {"errors": e.errors})

    return router
```

- [ ] **Step 4: Register the router**

In `app.py`, beside the other `app.include_router(setup_*_routes())` calls (~line 618), add the import with its siblings and:

```python
app.include_router(setup_workflow_routes())
```

- [ ] **Step 5: Run tests + import smoke**

Run: `python -m pytest tests/test_workflow_routes.py --import-mode=importlib -q` (Expected: PASS, 1 passed) and `python -c "import app"` (no error).

- [ ] **Step 6: Full affected-suite run**

Run: `python -m pytest tests/test_workflow_model.py tests/test_workflow_nodes.py tests/test_workflow_engine.py tests/test_workflow_store.py tests/test_workflow_routes.py --import-mode=importlib -q`
Expected: PASS (24 passed).

- [ ] **Step 7: Commit**

```bash
git add routes/workflow_routes.py app.py tests/test_workflow_routes.py
git commit -m "feat(workflows): admin-gated /api/workflows CRUD + run

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- Every pytest run uses `--import-mode=importlib`.
- No test may hit a real model, endpoint, or tool: `run_llm`/`run_tool` take injected `model_call`/`tool_dispatch`, and `default_tool_dispatch` is tested via a monkeypatched `_tool_handlers`.
- The engine only raises for an **invalid graph**; a failing node is a *run outcome* (logged + dependents skipped + partial `outputs`, HTTP 200).
- This is sub-project 1 — no visual editor, branching, loops, typed ports, triggers, or Skill bridge. Don't add them.

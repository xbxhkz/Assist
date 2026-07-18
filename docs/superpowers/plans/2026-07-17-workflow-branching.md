# Workflow Branching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `branch` router node — one `value` input, one output port per case + `else`, decided by exact-label `match` or an `llm` classification — so a workflow takes different paths, with un-taken branches skipped.

**Architecture:** One new node type across the shipped engine + editor. `branch` derives output ports from a config `cases` list. The engine adds an `inactive_ports` set and one clause to its existing skip check — a branch succeeds while deactivating its un-taken ports, and the engine's existing skip-cascade does the rest. The editor's canvas renders the ports for free; only a cases-list inspector is net-new.

**Tech Stack:** Python (model/nodes/engine), vanilla-JS ES modules (workflowGraph.js core + workflows.js view). Reuses sub-project 1's engine and sub-project 2's editor.

## Global Constraints

- `branch` node: input port `value`; output ports = `config["cases"] + ["else"]`; config `{"mode":"match"|"llm","cases":[str,…],"prompt"?:str}`.
- **match** mode: route `value` to the case whose label equals it (case-insensitive, trimmed), else `else`. **llm** mode: the model classifies `value` into a case; off-list → `else`.
- Engine: a `branch` succeeds and deactivates its non-chosen output ports; a node fed by an inactive port is skipped (rides the existing skip-cascade). **No merge-back** in v1 (a node downstream of both a taken and an un-taken branch is skipped).
- `output_ports`/`input_ports` are the single source of truth for validate + editor + engine; the JS core mirrors the Python rules **exactly** (a cross-language parity test enforces it).
- `output_ports` is called inside `validate` on untrusted JSON — its branch arm must be **crash-safe** (coerce a bad `cases` to `[]`).
- No new backend deps. No real model/tool in any test (`model_call` is mocked). pytest `--import-mode=importlib`; JS core via `node --input-type=module` (subprocess, `encoding="utf-8"`, skip if no node); JS view syntax via `node --check` on a temp `.mjs`. Stage specific files (never `git add -A`). Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Commit directly to `dev`.

---

### Task 1: Model — branch type, ports, validation

**Files:**
- Modify: `src/workflows/model.py`
- Test: `tests/test_workflow_branch_model.py`

**Interfaces:**
- Produces: `NODE_TYPES` includes `"branch"`; `input_ports(branch) -> ["value"]`; `output_ports(branch) -> cases + ["else"]` (crash-safe); `validate` flags bad branch config.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_branch_model.py`:

```python
import src.workflows.model as m


def _branch(cases, mode="match", nid="b"):
    return {"id": nid, "type": "branch", "config": {"mode": mode, "cases": cases}}


def test_branch_in_node_types():
    assert "branch" in m.NODE_TYPES


def test_branch_ports():
    b = _branch(["yes", "no"])
    assert m.input_ports(b) == ["value"]
    assert m.output_ports(b) == ["yes", "no", "else"]


def test_output_ports_crash_safe_on_bad_cases():
    assert m.output_ports({"type": "branch", "config": {"cases": "nope"}}) == ["else"]
    assert m.output_ports({"type": "branch", "config": {}}) == ["else"]
    assert m.output_ports({"type": "branch", "config": {"cases": [1, "", "ok"]}}) == ["ok", "else"]


def _wf(branch_cfg, extra_nodes=None, edges=None):
    nodes = [{"id": "i", "type": "input", "config": {"name": "q"}},
             {"id": "b", "type": "branch", "config": branch_cfg}]
    nodes += extra_nodes or []
    base_edges = [{"from_node": "i", "from_port": "value", "to_node": "b", "to_port": "value"}]
    return {"id": "w", "name": "W", "nodes": nodes, "edges": base_edges + (edges or [])}


def test_valid_branch_workflow_passes():
    wf = _wf({"mode": "match", "cases": ["yes", "no"]},
             extra_nodes=[{"id": "o", "type": "output", "config": {"name": "r"}}],
             edges=[{"from_node": "b", "from_port": "yes", "to_node": "o", "to_port": "value"}])
    assert m.validate(wf) == []


def test_validate_flags_bad_branch_config():
    assert any("cases" in e for e in m.validate(_wf({"mode": "match", "cases": []})))
    assert any("cases" in e for e in m.validate(_wf({"mode": "match", "cases": "x"})))
    assert any("duplicate case" in e for e in m.validate(_wf({"mode": "match", "cases": ["a", "a"]})))
    assert any("reserved" in e for e in m.validate(_wf({"mode": "match", "cases": ["else"]})))
    assert any("mode" in e for e in m.validate(_wf({"mode": "bogus", "cases": ["a"]})))


def test_validate_flags_edge_from_unknown_case_port():
    wf = _wf({"mode": "match", "cases": ["yes"]},
             extra_nodes=[{"id": "o", "type": "output", "config": {"name": "r"}}],
             edges=[{"from_node": "b", "from_port": "maybe", "to_node": "o", "to_port": "value"}])
    assert any("invalid output port" in e for e in m.validate(wf))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_branch_model.py --import-mode=importlib -q`
Expected: FAIL (`"branch" not in NODE_TYPES`, `output_ports` returns `[]` for branch, etc.).

- [ ] **Step 3: Write the implementation**

In `src/workflows/model.py`:

(a) Add `"branch"` to `NODE_TYPES`:
```python
NODE_TYPES = ("input", "template", "llm", "tool", "output", "branch")
```

(b) Extend `input_ports` — branch takes a single `value` (like output):
```python
def input_ports(node):
    t = node.get("type")
    if t in ("output", "branch"):
        return ["value"]
    key = _SLOT_SOURCE.get(t)
    if not key:
        return []          # input nodes (and unknown types) take no wires
    return slots_of((node.get("config") or {}).get(key, ""))
```

(c) Extend `output_ports` — branch's ports come from `cases` (crash-safe):
```python
def output_ports(node):
    if node.get("type") == "branch":
        cases = (node.get("config") or {}).get("cases")
        cases = cases if isinstance(cases, list) else []
        return [c for c in cases if isinstance(c, str) and c.strip()] + ["else"]
    port = _OUTPUT_PORT.get(node.get("type"))
    return [port] if port else []
```

(d) In `validate`, inside the node loop, after the `unknown node type` check (right after the line `errors.append(f"unknown node type: {n.get('type')} (node {nid})")`'s `if`), add a branch-config check:
```python
        if n.get("type") == "branch":
            cfg = n.get("config") or {}
            cases = cfg.get("cases")
            if not isinstance(cases, list) or not cases:
                errors.append(f"branch node {nid} must have a non-empty 'cases' list")
            else:
                seen = set()
                for c in cases:
                    if not isinstance(c, str) or not c.strip():
                        errors.append(f"branch node {nid} has an empty/non-string case")
                    elif c.strip().lower() == "else":
                        errors.append(f"branch node {nid}: case 'else' is reserved (auto fallback)")
                    elif c in seen:
                        errors.append(f"branch node {nid} has duplicate case '{c}'")
                    seen.add(c)
            if cfg.get("mode", "match") not in ("match", "llm"):
                errors.append(f"branch node {nid} mode must be 'match' or 'llm'")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workflow_branch_model.py --import-mode=importlib -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/workflows/model.py tests/test_workflow_branch_model.py
git commit -m "feat(workflows): branch node type — ports from cases + validation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Executor — `run_branch`

**Files:**
- Modify: `src/workflows/nodes.py`
- Test: `tests/test_workflow_branch_node.py`

**Interfaces:**
- Produces: `async run_branch(config, inputs, *, model_call) -> {"active": str, "value": str}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_branch_node.py`:

```python
import asyncio
import src.workflows.nodes as nd


def _run(coro):
    return asyncio.run(coro)


async def _never(*a, **k):
    raise AssertionError("match mode must not call the model")


def test_match_routes_to_equal_case_case_insensitive():
    out = _run(nd.run_branch({"mode": "match", "cases": ["Yes", "No"]},
                             {"value": " yes "}, model_call=_never))
    assert out == {"active": "Yes", "value": " yes "}


def test_match_falls_to_else_on_no_match():
    out = _run(nd.run_branch({"mode": "match", "cases": ["yes", "no"]},
                             {"value": "maybe"}, model_call=_never))
    assert out["active"] == "else"


def test_llm_routes_to_returned_case():
    async def fake(prompt, model=None, system=None):
        return "no"
    out = _run(nd.run_branch({"mode": "llm", "cases": ["yes", "no"], "prompt": "reply?"},
                             {"value": "an angry email"}, model_call=fake))
    assert out["active"] == "no"


def test_llm_tolerates_chatty_answer_via_substring():
    async def fake(prompt, model=None, system=None):
        return "I think the answer is YES."
    out = _run(nd.run_branch({"mode": "llm", "cases": ["yes", "no"]},
                             {"value": "x"}, model_call=fake))
    assert out["active"] == "yes"


def test_llm_off_list_falls_to_else():
    async def fake(prompt, model=None, system=None):
        return "purple"
    out = _run(nd.run_branch({"mode": "llm", "cases": ["yes", "no"]},
                             {"value": "x"}, model_call=fake))
    assert out["active"] == "else"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_branch_node.py --import-mode=importlib -q`
Expected: FAIL (`run_branch` not defined).

- [ ] **Step 3: Write the implementation**

Append to `src/workflows/nodes.py`:

```python
def _match_case(resp, cases):
    """Map an llm response to a case: exact (trimmed, case-insensitive) first, then
    the first case whose label appears as a substring; else 'else'."""
    r = (resp or "").strip().lower()
    for c in cases:
        if c.strip().lower() == r:
            return c
    for c in cases:
        if c.strip().lower() in r:
            return c
    return "else"


async def run_branch(config, inputs, *, model_call):
    """Route `value` to one case output. match: equal-label (else 'else'). llm: the
    model classifies value into a case. Returns {"active": chosen, "value": value}."""
    value = str((inputs or {}).get("value", ""))
    cases = [c for c in (config.get("cases") or []) if isinstance(c, str) and c.strip()]
    if config.get("mode", "match") == "llm":
        guidance = config.get("prompt") or ""
        prompt = ((guidance + "\n\n") if guidance else "") + (
            f"Input:\n{value}\n\nChoose exactly one of these labels: "
            f"{', '.join(cases)}.\nAnswer with only the label.")
        resp = str(await model_call(prompt, model=config.get("model"), system=config.get("system")))
        chosen = _match_case(resp, cases)
    else:
        chosen = next((c for c in cases if c.strip().lower() == value.strip().lower()), "else")
    return {"active": chosen, "value": value}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workflow_branch_node.py --import-mode=importlib -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/workflows/nodes.py tests/test_workflow_branch_node.py
git commit -m "feat(workflows): run_branch executor (match + llm classification)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Engine — deactivate un-taken branch ports

**Files:**
- Modify: `src/workflows/engine.py`
- Test: `tests/test_workflow_branch_engine.py`

**Interfaces:**
- Consumes: `nodes.run_branch`; `model.output_ports`.
- Produces: `run_workflow` routes through a taken branch and skips the un-taken branch's downstream.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_branch_engine.py`:

```python
import asyncio
import src.workflows.engine as eng


def _run(coro):
    return asyncio.run(coro)


def _wf():
    # input -> branch(match) -> [yes -> out_yes] / [no -> out_no]
    return {"id": "w", "name": "W", "nodes": [
        {"id": "i", "type": "input", "config": {"name": "q"}},
        {"id": "b", "type": "branch", "config": {"mode": "match", "cases": ["yes", "no"]}},
        {"id": "ty", "type": "template", "config": {"template": "Y:{v}"}},
        {"id": "oy", "type": "output", "config": {"name": "yes_out"}},
        {"id": "on", "type": "output", "config": {"name": "no_out"}},
    ], "edges": [
        {"from_node": "i", "from_port": "value", "to_node": "b", "to_port": "value"},
        {"from_node": "b", "from_port": "yes", "to_node": "ty", "to_port": "v"},
        {"from_node": "ty", "from_port": "text", "to_node": "oy", "to_port": "value"},
        {"from_node": "b", "from_port": "no", "to_node": "on", "to_port": "value"},
    ]}


def test_taken_branch_runs_untaken_skips_and_cascades():
    res = _run(eng.run_workflow(_wf(), {"q": "yes"}))
    status = {e["node"]: e["status"] for e in res["log"]}
    assert status["b"] == "ok"
    assert status["ty"] == "ok" and status["oy"] == "ok"   # taken path (+ cascade through ty)
    assert status["on"] == "skipped"                        # un-taken path
    assert res["outputs"] == {"yes_out": "Y:yes"}           # only the taken output resolved


def test_other_branch_taken():
    res = _run(eng.run_workflow(_wf(), {"q": "no"}))
    status = {e["node"]: e["status"] for e in res["log"]}
    assert status["on"] == "ok"
    assert status["ty"] == "skipped" and status["oy"] == "skipped"   # cascade skip
    assert res["outputs"] == {"no_out": "no"}


def test_llm_branch_routes():
    async def fake_model(prompt, model=None, system=None):
        return "no"
    wf = _wf()
    wf["nodes"][1]["config"] = {"mode": "llm", "cases": ["yes", "no"]}
    res = _run(eng.run_workflow(wf, {"q": "whatever"}, model_call=fake_model))
    status = {e["node"]: e["status"] for e in res["log"]}
    assert status["on"] == "ok" and status["oy"] == "skipped"
    assert res["outputs"] == {"no_out": "whatever"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_branch_engine.py --import-mode=importlib -q`
Expected: FAIL (branch runs but its un-taken port isn't deactivated → `on`/`oy` run instead of skip; or `_run_node` raises "unknown node type: branch").

- [ ] **Step 3: Write the implementation**

In `src/workflows/engine.py`:

(a) Import `output_ports`:
```python
from src.workflows.model import WorkflowError, output_ports, topo_sort, validate
```

(b) Add the `branch` arm to `_run_node` (before the final `raise`):
```python
    if t == "branch":
        return await N.run_branch(cfg, node_inputs, model_call=model_call)
```

(c) In `run_workflow`, initialize `inactive_ports` beside `produced`/`failed`:
```python
    produced = {}      # node id -> {port: value}
    failed = set()     # nodes that errored or were skipped
    inactive_ports = set()  # (node_id, port) a branch chose NOT to take
    outputs = {}
    log = []
```

(d) Extend the skip check to also skip a node fed by an inactive branch port:
```python
        incoming = [e for e in edges if e.get("to_node") == nid]
        upstream_bad = (any(e.get("from_node") in failed for e in incoming)
                        or any((e.get("from_node"), e.get("from_port")) in inactive_ports
                               for e in incoming))
```

(e) In the success path, special-case `branch` (beside the `output` special-case). Replace the block:
```python
            out = await _run_node(node, node_inputs, run_inputs, ctx, model_call, tool_dispatch)
            produced[nid] = out
            if node.get("type") == "output":
                outputs[(node.get("config") or {}).get("name", nid)] = node_inputs.get("value", "")
            shown = str(next(iter(out.values()), "")) if out else str(node_inputs.get("value", ""))
```
with:
```python
            out = await _run_node(node, node_inputs, run_inputs, ctx, model_call, tool_dispatch)
            if node.get("type") == "branch":
                chosen = out.get("active")
                produced[nid] = {chosen: out.get("value", "")}
                for p in output_ports(node):
                    if p != chosen:
                        inactive_ports.add((nid, p))
                shown = str(chosen)
            else:
                produced[nid] = out
                if node.get("type") == "output":
                    outputs[(node.get("config") or {}).get("name", nid)] = node_inputs.get("value", "")
                shown = str(next(iter(out.values()), "")) if out else str(node_inputs.get("value", ""))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workflow_branch_engine.py --import-mode=importlib -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the existing engine suite (no regression)**

Run: `python -m pytest tests/test_workflow_engine.py --import-mode=importlib -q`
Expected: PASS (unchanged — the branch path is additive; non-branch nodes are unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/workflows/engine.py tests/test_workflow_branch_engine.py
git commit -m "feat(workflows): engine skips un-taken branch ports (inactive_ports)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: JS core — branch ports + prune outgoing edges

**Files:**
- Modify: `static/js/workflowGraph.js`
- Test: `tests/test_workflow_branch_graph_js.py`

**Interfaces:**
- Consumes (Python): `src.workflows.model.output_ports` / `input_ports` (for the cross-language parity test).
- Produces: `NODE_TYPES` includes `'branch'`; `outputPortsOf(branch)` = `cases + ['else']`; `inputPortsOf(branch)` = `['value']`; `setConfig` prunes outgoing edges whose `from_port` is no longer valid.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_branch_graph_js.py`:

```python
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.workflows.model import input_ports, output_ports

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node binary not on PATH")


def _node_eval(source):
    r = subprocess.run(["node", "--input-type=module", "-e", source],
                       cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(r.stdout)


def test_branch_in_node_types_and_ports_match_python():
    out = _node_eval(
        "import { NODE_TYPES, inputPortsOf, outputPortsOf } from './static/js/workflowGraph.js';"
        "const b = {type:'branch', config:{mode:'match', cases:['yes','no']}};"
        "console.log(JSON.stringify({has: NODE_TYPES.includes('branch'),"
        "inp: inputPortsOf(b), out: outputPortsOf(b),"
        "bad: outputPortsOf({type:'branch', config:{cases:'nope'}})}));"
    )
    b = {"type": "branch", "config": {"mode": "match", "cases": ["yes", "no"]}}
    assert out["has"] is True
    assert out["inp"] == input_ports(b) == ["value"]
    assert out["out"] == output_ports(b) == ["yes", "no", "else"]        # cross-language parity
    assert out["bad"] == output_ports({"type": "branch", "config": {"cases": "nope"}}) == ["else"]


def test_setconfig_prunes_outgoing_edge_when_case_removed():
    out = _node_eval(
        "import { createGraph, setConfig } from './static/js/workflowGraph.js';"
        "const g = createGraph({id:'w', name:'W', nodes:["
        "{id:'b', type:'branch', config:{mode:'match', cases:['yes','no']}},"
        "{id:'o', type:'output', config:{name:'r'}}], edges:["
        "{from_node:'b', from_port:'no', to_node:'o', to_port:'value'}]});"
        "setConfig(g, 'b', {mode:'match', cases:['yes']});"   // drop 'no' -> its outgoing edge invalid
        "console.log(JSON.stringify({edges: g.edges.length}));"
    )
    assert out == {"edges": 0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_branch_graph_js.py --import-mode=importlib -q`
Expected: FAIL (`NODE_TYPES` lacks `branch`; `outputPortsOf(branch)` returns `[]`; the outgoing edge isn't pruned).

- [ ] **Step 3: Write the implementation**

In `static/js/workflowGraph.js`:

(a) Add `'branch'` to `NODE_TYPES`:
```javascript
export const NODE_TYPES = ['input', 'template', 'llm', 'tool', 'output', 'branch'];
```

(b) Extend `inputPortsOf` — branch takes `value`:
```javascript
export function inputPortsOf(node) {
  const t = node && node.type;
  if (t === 'output' || t === 'branch') return ['value'];
  const key = _SLOT_SOURCE[t];
  if (!key) return [];                 // input (and unknown) take no wires
  return slotsOf((node.config || {})[key]);
}
```

(c) Extend `outputPortsOf` — branch's ports from `cases` (mirrors Python, crash-safe):
```javascript
export function outputPortsOf(node) {
  if (node && node.type === 'branch') {
    const cases = (node.config || {}).cases;
    const list = Array.isArray(cases) ? cases.filter((c) => typeof c === 'string' && c.trim()) : [];
    return list.concat(['else']);
  }
  const p = _OUTPUT_PORT[node && node.type];
  return p ? [p] : [];
}
```

(d) Extend `_pruneEdges` to also drop OUTGOING edges whose `from_port` is no longer a valid output port (branch is the first node whose output ports change). Find the `_pruneEdges` function and replace its body:
```javascript
function _pruneEdges(graph, id) {
  const node = nodeById(graph, id);
  if (!node) return;
  const ins = inputPortsOf(node);
  const outs = outputPortsOf(node);
  graph.edges = graph.edges.filter(
    (e) => (e.to_node !== id || ins.includes(e.to_port))
        && (e.from_node !== id || outs.includes(e.from_port)),
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workflow_branch_graph_js.py --import-mode=importlib -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the existing JS-core suites (no regression)**

Run: `python -m pytest tests/test_workflow_graph_core_js.py tests/test_workflow_graph_edges_js.py --import-mode=importlib -q`
Expected: PASS (the `_pruneEdges` change is additive — existing nodes have fixed output ports, so their outgoing edges are never pruned).

- [ ] **Step 6: Commit**

```bash
git add static/js/workflowGraph.js tests/test_workflow_branch_graph_js.py
git commit -m "feat(workflows): JS core branch ports + prune outgoing edges on config change

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Editor — branch inspector (mode + cases + prompt)

**Files:**
- Modify: `static/js/workflows.js`
- Test: none new (the existing `tests/test_workflow_editor_shell.py` `node --check` guards syntax; the panel is manual-verified)

**Interfaces:**
- Consumes: `graph`, `selected`, `$`, `G` (`setConfig`, `nodeById`), `render` (from sub-project 2).
- Produces: `renderInspector` renders a branch-specific inspector.

- [ ] **Step 1: Add the branch inspector branch to `renderInspector`**

In `static/js/workflows.js`, in `renderInspector`, after the `title` is appended and `const cfg = Object.assign({}, node.config);` is set, add a branch special-case that renders and returns before the generic `_FIELDS` loop. Insert immediately after the `const cfg = ...` line:

```javascript
  if (node.type === 'branch') { renderBranchInspector(host, node, cfg); return; }
```

Then add the `renderBranchInspector` function (near `renderInspector`):

```javascript
function renderBranchInspector(host, node, cfg) {
  const FCSS = 'width:100%;background:var(--panel);color:var(--text);border:1px solid var(--border);'
    + 'border-radius:4px;padding:4px 6px;font-size:12px;box-sizing:border-box;';
  function lbl(t) {
    const l = document.createElement('label');
    l.textContent = t; l.style.cssText = 'display:block;font-size:10px;opacity:0.7;margin-top:6px;';
    return l;
  }
  // mode dropdown (changing it shows/hides the prompt -> full render)
  host.appendChild(lbl('mode'));
  const modeSel = document.createElement('select');
  modeSel.style.cssText = FCSS;
  ['match', 'llm'].forEach((mo) => {
    const o = document.createElement('option'); o.value = mo; o.textContent = mo;
    if ((cfg.mode || 'match') === mo) o.selected = true;
    modeSel.appendChild(o);
  });
  modeSel.addEventListener('change', () => {
    cfg.mode = modeSel.value; G.setConfig(graph, node.id, cfg); render();
  });
  host.appendChild(modeSel);
  // cases list editor
  host.appendChild(lbl('cases'));
  const cases = Array.isArray(cfg.cases) ? cfg.cases.slice() : [];
  cases.forEach((c, i) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:4px;margin-top:2px;';
    const inp = document.createElement('input');
    inp.value = c; inp.style.cssText = FCSS + 'flex:1;';
    inp.addEventListener('input', () => { cases[i] = inp.value; });
    inp.addEventListener('change', () => {   // relabel port on blur -> prune orphaned wires + redraw
      cfg.cases = cases.filter((x) => x && x.trim()); G.setConfig(graph, node.id, cfg);
      render({ keepInspector: true });
    });
    const del = document.createElement('button');
    del.textContent = '×'; del.style.cssText = 'font-size:12px;';
    del.addEventListener('click', () => {
      cases.splice(i, 1); cfg.cases = cases; G.setConfig(graph, node.id, cfg); render();
    });
    row.appendChild(inp); row.appendChild(del); host.appendChild(row);
  });
  const addBtn = document.createElement('button');
  addBtn.textContent = '+ case'; addBtn.style.cssText = 'margin-top:4px;font-size:11px;';
  addBtn.addEventListener('click', () => {
    cases.push(`case${cases.length + 1}`); cfg.cases = cases;
    G.setConfig(graph, node.id, cfg); render();
  });
  host.appendChild(addBtn);
  // prompt (llm mode only)
  if ((cfg.mode || 'match') === 'llm') {
    host.appendChild(lbl('prompt'));
    const pf = document.createElement('textarea');
    pf.value = cfg.prompt || ''; pf.style.cssText = FCSS + 'min-height:56px;resize:vertical;';
    pf.addEventListener('input', () => {
      cfg.prompt = pf.value; G.setConfig(graph, node.id, cfg); render({ keepInspector: true });
    });
    host.appendChild(pf);
  }
}
```

- [ ] **Step 2: Verify syntax + no regression**

Run: `python -m pytest tests/test_workflow_editor_shell.py --import-mode=importlib -q`
Expected: PASS (the `node --check` via temp `.mjs` confirms the new JS compiles; the HTML asserts are unaffected).

- [ ] **Step 3: Manual verification (owed by human — cannot run in a headless env)**

As an admin, in the workflow editor: add a **branch** node (it appears in the "+ Node" palette). Select it → the inspector shows a **mode** dropdown, a **cases** list with **+ case** / **×**, and (in llm mode) a **prompt** box. Add cases `yes`/`no` → the node grows an output dot per case + `else`. Wire `input → branch.value`, and `branch.yes`/`branch.no` to two separate output paths. Save. Run with an input that matches `yes` → the results panel shows the `yes` path `ok` and the `no` path `skipped`, and only the `yes` output appears. Switch the branch to **llm** mode with a prompt and confirm the model's choice routes the run. Rename a case → its wire drops (pruned); delete a case → same.

- [ ] **Step 4: Commit**

```bash
git add static/js/workflows.js
git commit -m "feat(workflows): branch inspector — mode, cases-list editor, llm prompt

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Tasks 1-4 are TDD** (Python + Node-harness). **Task 5 is build + `node --check` + a manual checklist** (the cases-list inspector; the canvas/palette recognize `branch` for free via the port functions). Do not skip Task 5's manual checklist — it is the behavioral gate.
- **Cross-language parity is load-bearing:** the JS `outputPortsOf`/`inputPortsOf` for a branch must equal the Python `output_ports`/`input_ports` for the same config — Task 4's test asserts it directly against the Python functions.
- `output_ports` (Python) and `outputPortsOf` (JS) are called on untrusted config during validate/render — keep the branch arm crash-safe (coerce a bad `cases` to `[]`).
- The engine change is additive: non-branch nodes and existing workflows behave exactly as before. Do not alter `topo_sort`, the error path, or normal data-flow.
- Scope: branching only. Do NOT build merge/join nodes, loops, typed ports, or a match-mode operator DSL — all explicit non-goals.

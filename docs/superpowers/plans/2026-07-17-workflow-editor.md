# Visual Workflow Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser node-canvas editor — drag nodes, wire ports, edit config, save, and run — that reads/writes the exact JSON the shipped `/api/workflows` engine executes.

**Architecture:** Two ES-module files. `static/js/workflowGraph.js` is a **pure, DOM-free core** (the in-memory graph + all testable logic: port derivation, cycle-guarded wiring, engine-JSON round-trip, auto-layout) — unit-tested in Node like `static/js/calendar/utils.js`. `static/js/workflows.js` is the **DOM/SVG view + controller** (IIFE-style module): renders nodes as absolutely-positioned divs, wires as one SVG overlay, handles pointer drag/wire, and calls the CRUD+run API. A full-screen modal opens from a new admin-only icon-rail button. No backend changes.

**Tech Stack:** Vanilla ES modules (no framework, no build, no CDN), SVG, pointer events. Backend already shipped (`/api/workflows` CRUD + run, uniform `{"errors":[...]}` error bodies).

## Global Constraints

- **Node types are exactly:** input, template, llm, tool, output. Ports: input `()→value`; template `{slots}→text`; llm `{slots}→text`; tool `{slots}→result`; output `value→()`. The input ports of template/llm/tool are DERIVED from the `{slot}` names in their text config (`/\{(\w+)\}/g`, ordered-unique) — the JS rule must match Python's `model.slots_of` exactly.
- **Text-only wires.** One edge per input port; output→input direction only; no self-loops; no edge that would create a cycle. No branching, loops, or typed ports.
- **Frontend only — no backend changes.** Node `x`/`y` positions ride inside the workflow JSON (keys the engine ignores and the store round-trips).
- **Dependency-free / offline** — no CDN libraries (the app is local-first; CSP allows `'self'` scripts + inline `style=""`, which is all this uses).
- **ES modules**, loaded via `<script type="module">` (precedent: `static/js/calendar.js` importing `static/js/calendar/utils.js`).
- **Admin-only** entry point — the API is admin-gated; the rail button is hidden unless `GET /api/auth/status` returns `is_admin: true`.
- JS-core tests run under Node via `subprocess` from pytest, skipped when `node` is absent (precedent: `tests/test_calendar_utils_dates_js.py`). Stage specific files (never `git add -A`). Commit messages end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Commit directly to `dev` (no feature branch).

---

### Task 1: Graph core — nodes, ports, config, layout, serialize

**Files:**
- Create: `static/js/workflowGraph.js`
- Test: `tests/test_workflow_graph_core_js.py`

**Interfaces:**
- Produces (all named ESM exports): `NODE_TYPES`, `slotsOf(text)->string[]`, `inputPortsOf(node)->string[]`, `outputPortsOf(node)->string[]`, `createGraph(wf?)->graph`, `addNode(graph,type,x?,y?)->node`, `removeNode(graph,id)->void`, `setConfig(graph,id,config)->void`, `setNodeId(graph,oldId,newId)->void` (throws `Error` on empty/duplicate), `setNodePos(graph,id,x,y)->void`, `setName(graph,name)->void`, `nodeById(graph,id)->node|undefined`, `topoOrder(graph)->string[]` (throws `Error` on cycle), `autoLayout(graph)->void`, `runInputNames(graph)->{name,default}[]`, `toJSON(graph)->wf`. A `graph` is `{id,name,nodes:[{id,type,config,x,y}],edges:[{from_node,from_port,to_node,to_port}]}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_graph_core_js.py`:

```python
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node binary not on PATH")


def _node_eval(source: str):
    result = subprocess.run(
        ["node", "--input-type=module", "-e", source],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_slots_of_ordered_unique():
    out = _node_eval(
        "import { slotsOf } from './static/js/workflowGraph.js';"
        "console.log(JSON.stringify({"
        "a: slotsOf('Hi {name}, you are {age}. Bye {name}'),"
        "b: slotsOf('no slots'), c: slotsOf('')}));"
    )
    assert out == {"a": ["name", "age"], "b": [], "c": []}


def test_ports_per_type():
    out = _node_eval(
        "import { inputPortsOf, outputPortsOf } from './static/js/workflowGraph.js';"
        "const mk=(t,c)=>({type:t,config:c});"
        "console.log(JSON.stringify({"
        "iIn: inputPortsOf(mk('input',{name:'q'})), iOut: outputPortsOf(mk('input',{})),"
        "tIn: inputPortsOf(mk('template',{template:'{a}-{b}'})), tOut: outputPortsOf(mk('template',{})),"
        "lIn: inputPortsOf(mk('llm',{prompt:'sum {doc}'})), lOut: outputPortsOf(mk('llm',{})),"
        "toIn: inputPortsOf(mk('tool',{args:'{path}'})), toOut: outputPortsOf(mk('tool',{})),"
        "oIn: inputPortsOf(mk('output',{})), oOut: outputPortsOf(mk('output',{}))}));"
    )
    assert out == {"iIn": [], "iOut": ["value"], "tIn": ["a", "b"], "tOut": ["text"],
                   "lIn": ["doc"], "lOut": ["text"], "toIn": ["path"], "toOut": ["result"],
                   "oIn": ["value"], "oOut": []}


def test_add_node_unique_ids_and_roundtrip():
    out = _node_eval(
        "import { createGraph, addNode, setConfig, toJSON } from './static/js/workflowGraph.js';"
        "const g = createGraph({id:'w', name:'W'});"
        "const a = addNode(g,'template',10,20); const b = addNode(g,'template',30,40);"
        "setConfig(g, a.id, {template:'Q: {q}'});"
        "const wf = toJSON(g);"
        "console.log(JSON.stringify({ids:[a.id,b.id], distinct: a.id!==b.id,"
        "n0:{id:wf.nodes[0].id,type:wf.nodes[0].type,x:wf.nodes[0].x,y:wf.nodes[0].y,cfg:wf.nodes[0].config},"
        "meta:{id:wf.id,name:wf.name}}));"
    )
    assert out["distinct"] is True
    assert out["n0"]["type"] == "template" and out["n0"]["x"] == 10 and out["n0"]["y"] == 20
    assert out["n0"]["cfg"] == {"template": "Q: {q}"}
    assert out["meta"] == {"id": "w", "name": "W"}


def test_create_graph_preserves_positions_from_json():
    out = _node_eval(
        "import { createGraph, toJSON, nodeById } from './static/js/workflowGraph.js';"
        "const g = createGraph({id:'w',name:'W',"
        "nodes:[{id:'i',type:'input',config:{name:'q'},x:5,y:7}],edges:[]});"
        "console.log(JSON.stringify({pos:[nodeById(g,'i').x,nodeById(g,'i').y],"
        "wf: toJSON(g).nodes[0]}));"
    )
    assert out["pos"] == [5, 7]
    assert out["wf"] == {"id": "i", "type": "input", "config": {"name": "q"}, "x": 5, "y": 7}


def test_set_node_id_rejects_empty_and_duplicate():
    out = _node_eval(
        "import { createGraph, addNode, setNodeId } from './static/js/workflowGraph.js';"
        "const g = createGraph(); const a = addNode(g,'input'); const b = addNode(g,'input');"
        "let empty=false, dup=false, ok=false;"
        "try { setNodeId(g,a.id,''); } catch(e){ empty=true; }"
        "try { setNodeId(g,a.id,b.id); } catch(e){ dup=true; }"
        "try { setNodeId(g,a.id,'fresh'); ok = (a.id==='fresh'); } catch(e){}"
        "console.log(JSON.stringify({empty,dup,ok}));"
    )
    assert out == {"empty": True, "dup": True, "ok": True}


def test_topo_order_and_autolayout_and_run_inputs():
    out = _node_eval(
        "import { createGraph, topoOrder, autoLayout, runInputNames, nodeById } from './static/js/workflowGraph.js';"
        "const g = createGraph({id:'w',name:'W',nodes:["
        "{id:'i',type:'input',config:{name:'q',default:'d'}},"
        "{id:'t',type:'template',config:{template:'Q: {q}'}},"
        "{id:'o',type:'output',config:{name:'answer'}}],"
        "edges:[{from_node:'i',from_port:'value',to_node:'t',to_port:'q'},"
        "{from_node:'t',from_port:'text',to_node:'o',to_port:'value'}]});"
        "autoLayout(g);"
        "console.log(JSON.stringify({order: topoOrder(g),"
        "layers:[nodeById(g,'i').x, nodeById(g,'t').x, nodeById(g,'o').x],"
        "inputs: runInputNames(g)}));"
    )
    assert out["order"] == ["i", "t", "o"]
    assert out["layers"][0] < out["layers"][1] < out["layers"][2]  # left-to-right by depth
    assert out["inputs"] == [{"name": "q", "default": "d"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_graph_core_js.py --import-mode=importlib -q`
Expected: FAIL (Node throws `Cannot find module .../static/js/workflowGraph.js` → non-zero exit → `CalledProcessError`).

- [ ] **Step 3: Write the implementation**

Create `static/js/workflowGraph.js`:

```javascript
// Pure, DOM-free workflow graph core: node/port model, cycle-guarded wiring,
// auto-layout, and engine-JSON round-trip. Unit-tested in Node. The {slot}
// derivation MUST match Python's model.slots_of (re \{(\w+)\}, ordered-unique).
export const NODE_TYPES = ['input', 'template', 'llm', 'tool', 'output'];

const _SLOT_RE = /\{(\w+)\}/g;
const _SLOT_SOURCE = { template: 'template', llm: 'prompt', tool: 'args' };
const _OUTPUT_PORT = { input: 'value', template: 'text', llm: 'text', tool: 'result' };

export function slotsOf(text) {
  const out = [];
  const s = text == null ? '' : String(text);
  let m;
  _SLOT_RE.lastIndex = 0;
  while ((m = _SLOT_RE.exec(s)) !== null) {
    if (!out.includes(m[1])) out.push(m[1]);
  }
  return out;
}

export function inputPortsOf(node) {
  const t = node && node.type;
  if (t === 'output') return ['value'];
  const key = _SLOT_SOURCE[t];
  if (!key) return [];                 // input (and unknown) take no wires
  return slotsOf((node.config || {})[key]);
}

export function outputPortsOf(node) {
  const p = _OUTPUT_PORT[node && node.type];
  return p ? [p] : [];
}

export function createGraph(wf = {}) {
  const nodes = (wf.nodes || []).map((n) => ({
    id: n.id, type: n.type, config: Object.assign({}, n.config || {}),
    x: typeof n.x === 'number' ? n.x : 0, y: typeof n.y === 'number' ? n.y : 0,
  }));
  const edges = (wf.edges || []).map((e) => ({
    from_node: e.from_node, from_port: e.from_port, to_node: e.to_node, to_port: e.to_port,
  }));
  return { id: wf.id || '', name: wf.name || '', nodes, edges };
}

export function nodeById(graph, id) {
  return graph.nodes.find((n) => n.id === id);
}

export function addNode(graph, type, x = 40, y = 40) {
  let i = 1;
  while (nodeById(graph, `${type}${i}`)) i += 1;
  const node = { id: `${type}${i}`, type, config: {}, x, y };
  graph.nodes.push(node);
  return node;
}

// Drop any edge whose to_port is no longer a valid input port of `id`
// (e.g. after a template's {slots} changed). Shared by removeNode/setConfig.
function _pruneEdges(graph, id) {
  const node = nodeById(graph, id);
  if (!node) return;
  const ins = inputPortsOf(node);
  graph.edges = graph.edges.filter(
    (e) => e.to_node !== id || ins.includes(e.to_port),
  );
}

export function removeNode(graph, id) {
  graph.nodes = graph.nodes.filter((n) => n.id !== id);
  graph.edges = graph.edges.filter((e) => e.from_node !== id && e.to_node !== id);
}

export function setConfig(graph, id, config) {
  const node = nodeById(graph, id);
  if (!node) return;
  node.config = Object.assign({}, config);
  _pruneEdges(graph, id);
}

export function setNodeId(graph, oldId, newId) {
  const id = (newId || '').trim();
  if (!id) throw new Error('node id cannot be empty');
  if (id !== oldId && nodeById(graph, id)) throw new Error('duplicate node id');
  const node = nodeById(graph, oldId);
  if (!node) throw new Error('unknown node');
  node.id = id;
  graph.edges.forEach((e) => {
    if (e.from_node === oldId) e.from_node = id;
    if (e.to_node === oldId) e.to_node = id;
  });
}

export function setNodePos(graph, id, x, y) {
  const node = nodeById(graph, id);
  if (node) { node.x = x; node.y = y; }
}

export function setName(graph, name) { graph.name = name || ''; }

export function topoOrder(graph) {
  const ids = graph.nodes.map((n) => n.id);
  const indeg = {};
  const adj = {};
  ids.forEach((i) => { indeg[i] = 0; adj[i] = []; });
  graph.edges.forEach((e) => {
    if (e.from_node in indeg && e.to_node in indeg) {
      adj[e.from_node].push(e.to_node);
      indeg[e.to_node] += 1;
    }
  });
  const q = ids.filter((i) => indeg[i] === 0);
  const order = [];
  while (q.length) {
    const cur = q.shift();
    order.push(cur);
    adj[cur].forEach((nxt) => { indeg[nxt] -= 1; if (indeg[nxt] === 0) q.push(nxt); });
  }
  if (order.length !== ids.length) throw new Error('cycle detected');
  return order;
}

// Assign x/y by longest-path depth (column) and stacking index (row).
export function autoLayout(graph) {
  const order = topoOrder(graph);          // throws on cycle
  const preds = {};
  graph.nodes.forEach((n) => { preds[n.id] = []; });
  graph.edges.forEach((e) => { if (preds[e.to_node]) preds[e.to_node].push(e.from_node); });
  const layer = {};
  order.forEach((id) => {
    layer[id] = preds[id].length ? Math.max(...preds[id].map((p) => layer[p])) + 1 : 0;
  });
  const rowInLayer = {};
  graph.nodes.forEach((n) => {
    const L = layer[n.id] || 0;
    const row = rowInLayer[L] || 0;
    rowInLayer[L] = row + 1;
    n.x = 40 + L * 210;
    n.y = 40 + row * 120;
  });
}

export function runInputNames(graph) {
  return graph.nodes
    .filter((n) => n.type === 'input')
    .map((n) => ({ name: (n.config || {}).name || '', default: (n.config || {}).default || '' }));
}

export function toJSON(graph) {
  return {
    id: graph.id, name: graph.name,
    nodes: graph.nodes.map((n) => ({ id: n.id, type: n.type, config: n.config, x: n.x, y: n.y })),
    edges: graph.edges.map((e) => ({
      from_node: e.from_node, from_port: e.from_port, to_node: e.to_node, to_port: e.to_port,
    })),
  };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workflow_graph_core_js.py --import-mode=importlib -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add static/js/workflowGraph.js tests/test_workflow_graph_core_js.py
git commit -m "feat(workflows): graph core — nodes, ports, layout, JSON round-trip

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Graph core — cycle-guarded wiring

**Files:**
- Modify: `static/js/workflowGraph.js` (append edge functions)
- Test: `tests/test_workflow_graph_edges_js.py`

**Interfaces:**
- Consumes (Task 1): `createGraph`, `addNode`, `setConfig`, `inputPortsOf`, `outputPortsOf`, `nodeById`, `removeNode`.
- Produces: `canConnect(graph,fromNode,fromPort,toNode,toPort)->bool`, `addEdge(...)->bool` (replaces any existing edge into that input port; returns false and adds nothing if `!canConnect`), `removeEdge(graph,fromNode,fromPort,toNode,toPort)->void`, `unwiredPorts(graph)->{node,port}[]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_graph_edges_js.py`:

```python
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node binary not on PATH")

_SETUP = (
    "import { createGraph, addNode, setConfig, canConnect, addEdge, removeEdge,"
    " removeNode, unwiredPorts, nodeById } from './static/js/workflowGraph.js';"
    "const g = createGraph();"
    "const i = addNode(g,'input'); const t = addNode(g,'template');"
    "setConfig(g, t.id, {template:'Q: {q}'});"
)


def _node_eval(body: str):
    result = subprocess.run(
        ["node", "--input-type=module", "-e", _SETUP + body],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_add_valid_edge_and_reject_bad_ports_and_self_loop():
    out = _node_eval(
        "const good = addEdge(g, i.id, 'value', t.id, 'q');"
        "const badFromPort = addEdge(g, i.id, 'nope', t.id, 'q');"
        "const badToPort = addEdge(g, i.id, 'value', t.id, 'zzz');"
        "const self = addEdge(g, t.id, 'text', t.id, 'q');"
        "console.log(JSON.stringify({good, badFromPort, badToPort, self, edges: g.edges.length}));"
    )
    assert out == {"good": True, "badFromPort": False, "badToPort": False, "self": False, "edges": 1}


def test_one_edge_per_input_replaces():
    out = _node_eval(
        "const i2 = addNode(g,'input');"
        "addEdge(g, i.id, 'value', t.id, 'q');"
        "addEdge(g, i2.id, 'value', t.id, 'q');"   # re-wire same input port
        "console.log(JSON.stringify({count: g.edges.length, from: g.edges[0].from_node === i2.id}));"
    )
    assert out == {"count": 1, "from": True}


def test_cycle_is_rejected():
    out = _node_eval(
        "const t2 = addNode(g,'template'); setConfig(g, t2.id, {template:'{x}'});"
        "setConfig(g, t.id, {template:'{q}{y}'});"   # t now has ports q, y
        "addEdge(g, t.id, 'text', t2.id, 'x');"       # t -> t2
        "const back = canConnect(g, t2.id, 'text', t.id, 'y');"  # t2 -> t would cycle
        "const added = addEdge(g, t2.id, 'text', t.id, 'y');"
        "console.log(JSON.stringify({back, added, edges: g.edges.length}));"
    )
    assert out == {"back": False, "added": False, "edges": 1}


def test_setconfig_prunes_orphaned_edge_and_removenode_drops_edges():
    out = _node_eval(
        "addEdge(g, i.id, 'value', t.id, 'q');"
        "setConfig(g, t.id, {template:'no slots now'});"   # {q} gone -> edge orphaned
        "const afterPrune = g.edges.length;"
        "setConfig(g, t.id, {template:'{q}'}); addEdge(g, i.id, 'value', t.id, 'q');"
        "removeNode(g, i.id);"                              # dropping the source node
        "console.log(JSON.stringify({afterPrune, afterRemove: g.edges.length}));"
    )
    assert out == {"afterPrune": 0, "afterRemove": 0}


def test_unwired_ports_lists_missing_inputs():
    out = _node_eval(
        "const o = addNode(g,'output');"          # output has input port 'value', unwired
        "console.log(JSON.stringify({unwired: unwiredPorts(g)}));"
    )
    # t has 'q' unwired, o has 'value' unwired (input node has no input ports)
    pairs = {(u["node"], u["port"]) for u in out["unwired"]}
    assert pairs == {("template1", "q"), ("output1", "value")}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_workflow_graph_edges_js.py --import-mode=importlib -q`
Expected: FAIL (Node throws `addEdge is not a function` → non-zero exit → `CalledProcessError`).

- [ ] **Step 3: Write the implementation**

Append to `static/js/workflowGraph.js`:

```javascript
// ── edges: cycle-guarded wiring ──

// Can `to` already reach `from` by following edges? Used to reject an edge
// from->to that would close a cycle.
function _reaches(graph, start, target) {
  const seen = new Set();
  const stack = [start];
  while (stack.length) {
    const cur = stack.pop();
    if (cur === target) return true;
    if (seen.has(cur)) continue;
    seen.add(cur);
    graph.edges.forEach((e) => { if (e.from_node === cur) stack.push(e.to_node); });
  }
  return false;
}

export function canConnect(graph, fromNode, fromPort, toNode, toPort) {
  if (fromNode === toNode) return false;
  const from = nodeById(graph, fromNode);
  const to = nodeById(graph, toNode);
  if (!from || !to) return false;
  if (!outputPortsOf(from).includes(fromPort)) return false;
  if (!inputPortsOf(to).includes(toPort)) return false;
  if (_reaches(graph, toNode, fromNode)) return false;   // would create a cycle
  return true;
}

export function addEdge(graph, fromNode, fromPort, toNode, toPort) {
  if (!canConnect(graph, fromNode, fromPort, toNode, toPort)) return false;
  // one edge per input port: drop any existing wire into (toNode,toPort)
  graph.edges = graph.edges.filter((e) => !(e.to_node === toNode && e.to_port === toPort));
  graph.edges.push({ from_node: fromNode, from_port: fromPort, to_node: toNode, to_port: toPort });
  return true;
}

export function removeEdge(graph, fromNode, fromPort, toNode, toPort) {
  graph.edges = graph.edges.filter((e) => !(
    e.from_node === fromNode && e.from_port === fromPort
    && e.to_node === toNode && e.to_port === toPort));
}

export function unwiredPorts(graph) {
  const wired = new Set(graph.edges.map((e) => `${e.to_node} ${e.to_port}`));
  const out = [];
  graph.nodes.forEach((n) => {
    inputPortsOf(n).forEach((port) => {
      if (!wired.has(`${n.id} ${port}`)) out.push({ node: n.id, port });
    });
  });
  return out;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_workflow_graph_edges_js.py --import-mode=importlib -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add static/js/workflowGraph.js tests/test_workflow_graph_edges_js.py
git commit -m "feat(workflows): graph core — cycle-guarded wiring + unwired hints

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Modal shell + admin-gated rail button + view bootstrap

**Files:**
- Modify: `static/index.html` (rail button, modal markup, module script tag)
- Create: `static/js/workflows.js` (bootstrap only)
- Test: `tests/test_workflow_editor_shell.py`

**Interfaces:**
- Consumes: `static/js/modalManager.js` (`register`), `static/js/workflowGraph.js` (`createGraph`).
- Produces (module-level, used by Tasks 4-5): globals `graph`, `currentId`, `selected`; helpers `$(id)`, `api(path,opts)`, `msg(text,isErr)`, `openWorkflows()`, `closeWorkflows()`, `refreshList()` and `render()` (stubs here, replaced in Tasks 5 and 4 respectively).

- [ ] **Step 1: Add the rail button (admin-only, hidden by default)**

In `static/index.html`, immediately after the `rail-theme` button (the `<button ... id="rail-theme" ...>` line, ~706) and before the `<div style="flex:1"></div>` spacer, add:

```html
    <button class="icon-rail-btn" id="rail-workflows" title="Workflows" style="display:none"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><path d="M10 6.5h4a2 2 0 0 1 2 2v5.5"/></svg></button>
```

- [ ] **Step 2: Add the modal markup**

In `static/index.html`, next to the other tool modals (e.g. right after the `memory-modal` `</div>` that closes it, or anywhere among the sibling `<div id="...-modal" class="modal hidden">` blocks), add:

```html
  <!-- Visual Workflow Editor -->
  <div id="workflows-modal" class="modal hidden">
    <div class="modal-content" role="dialog" aria-label="Workflows" style="background:var(--bg);width:92vw;height:88vh;max-width:1400px;display:flex;flex-direction:column;">
      <div class="modal-header">
        <h4><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><path d="M10 6.5h4a2 2 0 0 1 2 2v5.5"/></svg>Workflows</h4>
        <button class="close-btn" id="wf-close" aria-label="Close workflows">✖</button>
      </div>
      <div class="modal-body" style="flex:1;display:flex;min-height:0;padding:0;">
        <aside id="wf-list" style="width:190px;border-right:1px solid var(--border);overflow:auto;padding:6px;"></aside>
        <div style="flex:1;display:flex;flex-direction:column;min-width:0;">
          <div id="wf-toolbar" style="display:flex;gap:8px;align-items:center;padding:6px;border-bottom:1px solid var(--border);flex-wrap:wrap;">
            <input id="wf-name" placeholder="Workflow name" style="flex:1;min-width:120px;background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:4px 6px;">
            <button id="wf-add">+ Node ▾</button>
            <button id="wf-save">Save</button>
            <button id="wf-run">Run ▶</button>
            <span id="wf-msg" style="font-size:12px;opacity:0.85;"></span>
          </div>
          <div id="wf-canvas" style="flex:1;position:relative;overflow:auto;background:var(--panel);min-height:0;">
            <svg id="wf-wires" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:visible;"></svg>
          </div>
          <div id="wf-results" style="height:180px;border-top:1px solid var(--border);overflow:auto;padding:6px;display:none;"></div>
        </div>
        <aside id="wf-inspector" style="width:250px;border-left:1px solid var(--border);overflow:auto;padding:8px;display:none;"></aside>
      </div>
    </div>
  </div>
```

- [ ] **Step 3: Add the module script tag**

In `static/index.html`, alongside the other `<script type="module" ...>` tags (~2859-2873), add:

```html
<script type="module" src="/static/js/workflows.js"></script>
```

- [ ] **Step 4: Write the bootstrap module**

Create `static/js/workflows.js`:

```javascript
// Visual workflow editor. ES module: imports the pure graph core and the modal
// manager. This file is the DOM/SVG view + controller. Admin-only (the API is
// admin-gated); the rail button stays hidden unless /api/auth/status says so.
import * as Modals from './modalManager.js';
import * as G from './workflowGraph.js';

let graph = G.createGraph();
let currentId = null;             // server id of the loaded workflow (null = unsaved)
let selected = null;              // {kind:'node'|'edge', ...} — set by the view (Task 4)

function $(id) { return document.getElementById(id); }

function msg(text, isErr) {
  const m = $('wf-msg');
  if (m) { m.textContent = text || ''; m.style.color = isErr ? 'var(--red,#ff5555)' : ''; }
}

async function api(path, opts) {
  const res = await fetch(path, Object.assign({ credentials: 'same-origin' }, opts || {}));
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data && data.detail;
    const errs = d && d.errors ? d.errors.join('; ') : (d || String(res.status));
    throw new Error(errs);
  }
  return data;
}

// Replaced in Task 4 (canvas render) and Task 5 (list). Stubs keep openWorkflows
// callable after Task 3.
function render() {}
async function refreshList() {}

function openWorkflows() {
  $('workflows-modal').classList.remove('hidden');
  refreshList();
  render();
}

function closeWorkflows() {
  $('workflows-modal').classList.add('hidden');
}

async function isAdmin() {
  try {
    const d = await (await fetch('/api/auth/status', { credentials: 'same-origin' })).json();
    return !!d.is_admin;
  } catch (e) { return false; }
}

function init() {
  isAdmin().then((ok) => { const b = $('rail-workflows'); if (b && ok) b.style.display = ''; });
  const rail = $('rail-workflows'); if (rail) rail.addEventListener('click', openWorkflows);
  const x = $('wf-close'); if (x) x.addEventListener('click', closeWorkflows);
  Modals.register('workflows-modal', {
    railBtnId: 'rail-workflows', sidebarBtnId: 'tool-workflows-btn', closeFn: closeWorkflows,
  });
}

document.addEventListener('DOMContentLoaded', init);
```

- [ ] **Step 5: Write the shell test**

Create `tests/test_workflow_editor_shell.py`:

```python
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_index_html_wires_the_editor():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="rail-workflows"' in html
    assert 'id="workflows-modal"' in html
    assert 'src="/static/js/workflows.js"' in html
    assert 'id="wf-canvas"' in html and 'id="wf-wires"' in html


@pytest.mark.skipif(not shutil.which("node"), reason="node binary not on PATH")
def test_workflows_js_is_syntactically_valid():
    # Copy to a .mjs temp file so node parses it as an ES module (a bare .js is
    # treated as CommonJS and would reject `import`). --check validates syntax
    # only — it does not resolve the relative imports, so no DOM is executed.
    src = (ROOT / "static" / "js" / "workflows.js").read_text(encoding="utf-8")
    fd, tmp = tempfile.mkstemp(suffix=".mjs")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(src)
        subprocess.run(["node", "--check", tmp], check=True, capture_output=True, text=True)
    finally:
        os.unlink(tmp)
```

- [ ] **Step 6: Run tests + manual check**

Run: `python -m pytest tests/test_workflow_editor_shell.py --import-mode=importlib -q` (Expected: PASS, 2 passed — or 1 passed + 1 skipped without node).
Manual: as an admin, load the app → a Workflows icon appears on the rail → clicking it opens an empty full-screen modal → ✖ closes it. As a non-admin, the icon stays hidden.

- [ ] **Step 7: Commit**

```bash
git add static/index.html static/js/workflows.js tests/test_workflow_editor_shell.py
git commit -m "feat(workflows): editor modal shell + admin-gated rail launcher

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Canvas — render nodes/wires, drag, wiring, inspector

**Files:**
- Modify: `static/js/workflows.js` (replace the `render()` stub; add canvas/interaction/inspector code; extend `init()` with the palette handler)
- Test: `tests/test_workflow_editor_shell.py` (the existing `node --check` test now also guards this task)

**Interfaces:**
- Consumes: `G.*` (Task 1-2), module globals `graph`/`selected`/`$`/`render` (Task 3).
- Produces: `render()` (full canvas render), `selectNode(id)`, `selectEdge(edge)`, `renderInspector()`. Used by Task 5's run highlighting via `render()`.

- [ ] **Step 1: Replace the `render()` stub and add the canvas/interaction code**

In `static/js/workflows.js`, replace the line `function render() {}` with the block below (everything through `renderInspector`):

```javascript
// ── canvas geometry ──
const HEADER = 26, PORT_GAP = 20, NODE_W = 150, DOT = 10;

function portOffsetY(idx) { return HEADER + PORT_GAP / 2 + idx * PORT_GAP; }

function portPos(node, dir, port) {
  const ports = dir === 'in' ? G.inputPortsOf(node) : G.outputPortsOf(node);
  const i = Math.max(0, ports.indexOf(port));
  return { x: dir === 'in' ? node.x : node.x + NODE_W, y: node.y + portOffsetY(i) };
}

function nodeHeight(node) {
  const rows = Math.max(G.inputPortsOf(node).length, G.outputPortsOf(node).length, 1);
  return HEADER + rows * PORT_GAP + 8;
}

let pending = null;   // active wire drag: {fromNode, fromPort}

function render() {
  const canvas = $('wf-canvas');
  if (!canvas) return;
  // wipe node divs (keep the <svg> overlay)
  Array.from(canvas.querySelectorAll('.wf-node')).forEach((el) => el.remove());
  graph.nodes.forEach((n) => canvas.appendChild(renderNode(n)));
  renderWires();
  renderInspector();
  const unwired = G.unwiredPorts(graph).length;
  const runBtn = $('wf-run');
  if (runBtn) runBtn.textContent = unwired ? `Run ▶ (${unwired} unwired)` : 'Run ▶';
}

function renderNode(node) {
  const div = document.createElement('div');
  div.className = 'wf-node';
  div.dataset.id = node.id;
  const sel = selected && selected.kind === 'node' && selected.id === node.id;
  div.style.cssText = `position:absolute;left:${node.x}px;top:${node.y}px;width:${NODE_W}px;`
    + `height:${nodeHeight(node)}px;background:var(--bg);border:1px solid ${sel ? 'var(--accent)' : 'var(--border)'};`
    + 'border-radius:6px;font-size:11px;user-select:none;box-shadow:0 1px 3px rgba(0,0,0,0.3);';

  const head = document.createElement('div');
  head.className = 'wf-node-head';
  head.style.cssText = `height:${HEADER}px;display:flex;align-items:center;gap:4px;padding:0 4px;`
    + 'border-bottom:1px solid var(--border);cursor:move;background:var(--panel);border-radius:6px 6px 0 0;';
  const type = document.createElement('span');
  type.textContent = node.type;
  type.style.cssText = 'opacity:0.6;text-transform:uppercase;font-size:9px;';
  const idIn = document.createElement('input');
  idIn.value = node.id;
  idIn.style.cssText = 'flex:1;min-width:0;background:transparent;border:none;color:var(--text);font-size:11px;';
  idIn.addEventListener('pointerdown', (e) => e.stopPropagation());  // don't start a drag
  idIn.addEventListener('change', () => {
    try { G.setNodeId(graph, node.id, idIn.value); render(); }
    catch (err) { idIn.value = node.id; msg(err.message, true); }
  });
  const del = document.createElement('button');
  del.textContent = '×';
  del.style.cssText = 'border:none;background:transparent;color:var(--text);cursor:pointer;font-size:14px;line-height:1;';
  del.addEventListener('pointerdown', (e) => e.stopPropagation());
  del.addEventListener('click', () => {
    G.removeNode(graph, node.id);
    if (selected && selected.kind === 'node' && selected.id === node.id) selected = null;
    render();
  });
  head.appendChild(type); head.appendChild(idIn); head.appendChild(del);
  head.addEventListener('pointerdown', (e) => startNodeDrag(e, node));
  head.addEventListener('click', () => selectNode(node.id));
  div.appendChild(head);

  G.inputPortsOf(node).forEach((port, i) => div.appendChild(portDot(node, 'in', port, i)));
  G.outputPortsOf(node).forEach((port, i) => div.appendChild(portDot(node, 'out', port, i)));
  return div;
}

function portDot(node, dir, port, i) {
  const wiredIn = dir === 'in'
    && !graph.edges.some((e) => e.to_node === node.id && e.to_port === port);
  const dot = document.createElement('div');
  dot.title = port;
  dot.dataset.node = node.id; dot.dataset.port = port; dot.dataset.dir = dir;
  const top = portOffsetY(i) - DOT / 2;
  const left = dir === 'in' ? -DOT / 2 : NODE_W - DOT / 2;
  dot.style.cssText = `position:absolute;top:${top}px;left:${left}px;width:${DOT}px;height:${DOT}px;`
    + `border-radius:50%;background:${wiredIn ? 'var(--red,#ff5555)' : 'var(--accent)'};`
    + 'border:1px solid var(--bg);cursor:crosshair;';
  const label = document.createElement('span');
  label.textContent = port;
  label.style.cssText = `position:absolute;top:${portOffsetY(i) - 7}px;font-size:9px;opacity:0.7;`
    + (dir === 'in' ? 'left:8px;' : `left:${NODE_W - 8}px;transform:translateX(-100%);`);
  dot.addEventListener('pointerdown', (e) => {
    e.stopPropagation();
    if (dir === 'out') startWire(e, node.id, port);
  });
  dot.addEventListener('pointerup', (e) => {
    if (dir === 'in' && pending) finishWire(node.id, port);
  });
  const wrap = document.createDocumentFragment();
  wrap.appendChild(dot); wrap.appendChild(label);
  const holder = document.createElement('div');
  holder.appendChild(wrap);
  return holder;
}

// ── node dragging ──
function startNodeDrag(e, node) {
  const canvas = $('wf-canvas');
  const rect = canvas.getBoundingClientRect();
  const offX = e.clientX - rect.left + canvas.scrollLeft - node.x;
  const offY = e.clientY - rect.top + canvas.scrollTop - node.y;
  function move(ev) {
    G.setNodePos(graph, node.id,
      Math.max(0, ev.clientX - rect.left + canvas.scrollLeft - offX),
      Math.max(0, ev.clientY - rect.top + canvas.scrollTop - offY));
    render();
  }
  function up() { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); }
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
}

// ── wiring ──
function canvasPoint(e) {
  const canvas = $('wf-canvas');
  const rect = canvas.getBoundingClientRect();
  return { x: e.clientX - rect.left + canvas.scrollLeft, y: e.clientY - rect.top + canvas.scrollTop };
}

function startWire(e, fromNode, fromPort) {
  pending = { fromNode, fromPort };
  function move(ev) { renderWires(canvasPoint(ev)); }
  function up() {
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', up);
    pending = null; renderWires();
  }
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
}

function finishWire(toNode, toPort) {
  if (!pending) return;
  const ok = G.addEdge(graph, pending.fromNode, pending.fromPort, toNode, toPort);
  if (!ok) msg('Cannot connect (type/direction/cycle)', true); else msg('');
  pending = null; render();
}

function bezier(x1, y1, x2, y2) {
  const dx = Math.max(30, Math.abs(x2 - x1) / 2);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

function renderWires(cursor) {
  const svg = $('wf-wires');
  if (!svg) return;
  svg.innerHTML = '';
  graph.edges.forEach((e) => {
    const from = G.nodeById(graph, e.from_node);
    const to = G.nodeById(graph, e.to_node);
    if (!from || !to) return;
    const a = portPos(from, 'out', e.from_port);
    const b = portPos(to, 'in', e.to_port);
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', bezier(a.x, a.y, b.x, b.y));
    path.setAttribute('fill', 'none');
    const sel = selected && selected.kind === 'edge' && selected.edge
      && selected.edge.to_node === e.to_node && selected.edge.to_port === e.to_port;
    path.setAttribute('stroke', sel ? 'var(--accent)' : 'var(--text)');
    path.setAttribute('stroke-width', sel ? '2.5' : '1.5');
    path.setAttribute('opacity', '0.8');
    path.style.pointerEvents = 'stroke';
    path.style.cursor = 'pointer';
    path.addEventListener('click', () => selectEdge(e));
    svg.appendChild(path);
  });
  if (pending && cursor) {
    const from = G.nodeById(graph, pending.fromNode);
    const a = portPos(from, 'out', pending.fromPort);
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', bezier(a.x, a.y, cursor.x, cursor.y));
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', 'var(--accent)');
    path.setAttribute('stroke-width', '2');
    path.setAttribute('stroke-dasharray', '4 3');
    svg.appendChild(path);
  }
}

// ── selection + inspector ──
function selectNode(id) { selected = { kind: 'node', id }; render(); }
function selectEdge(edge) { selected = { kind: 'edge', edge }; render(); }

const _FIELDS = {
  input: [['name', 'text'], ['default', 'text']],
  template: [['template', 'area']],
  llm: [['prompt', 'area'], ['model', 'text'], ['system', 'area']],
  tool: [['tool', 'text'], ['args', 'area']],
  output: [['name', 'text']],
};

function renderInspector() {
  const host = $('wf-inspector');
  if (!host) return;
  if (!selected || selected.kind !== 'node') { host.style.display = 'none'; host.innerHTML = ''; return; }
  const node = G.nodeById(graph, selected.id);
  if (!node) { host.style.display = 'none'; return; }
  host.style.display = '';
  host.innerHTML = '';
  const title = document.createElement('div');
  title.textContent = `${node.type} · ${node.id}`;
  title.style.cssText = 'font-weight:600;margin-bottom:8px;font-size:12px;';
  host.appendChild(title);
  const cfg = Object.assign({}, node.config);
  (_FIELDS[node.type] || []).forEach(([key, kind]) => {
    const label = document.createElement('label');
    label.textContent = key;
    label.style.cssText = 'display:block;font-size:10px;opacity:0.7;margin-top:6px;';
    const field = document.createElement(kind === 'area' ? 'textarea' : 'input');
    field.value = cfg[key] || '';
    field.style.cssText = 'width:100%;background:var(--panel);color:var(--text);border:1px solid var(--border);'
      + 'border-radius:4px;padding:4px 6px;font-size:12px;box-sizing:border-box;'
      + (kind === 'area' ? 'min-height:56px;resize:vertical;' : '');
    field.addEventListener('input', () => {
      cfg[key] = field.value;
      G.setConfig(graph, node.id, cfg);
      render();                 // re-derive ports live (e.g. new {slot})
    });
    host.appendChild(label); host.appendChild(field);
  });
}
```

- [ ] **Step 2: Add the node palette to `init()`**

In `static/js/workflows.js`, inside `init()` (after the `wf-close` wiring, before the `Modals.register` call), add:

```javascript
  const add = $('wf-add');
  if (add) add.addEventListener('click', () => {
    const type = window.prompt('Node type: input, template, llm, tool, or output', 'template');
    if (!type || !G.NODE_TYPES.includes(type)) return;
    const canvas = $('wf-canvas');
    const node = G.addNode(graph, type,
      40 + (canvas ? canvas.scrollLeft : 0), 40 + (canvas ? canvas.scrollTop : 0));
    selectNode(node.id);
  });
```

- [ ] **Step 3: Verify syntax + manual check**

Run: `python -m pytest tests/test_workflow_editor_shell.py --import-mode=importlib -q` (Expected: PASS — `node --check` guards the new code compiles).
Manual: open the modal → "+ Node ▾" adds a template → select it → type `Q: {q}` into the template field; a `q` input port appears live → add an input node → drag from the input's `value` output dot to the template's `q` input dot; a wire connects → drag nodes by their headers (wires follow) → click a wire, it highlights → × on a node removes it and its wires. Try to wire a cycle (template→template→back); the second wire is refused with a message.

- [ ] **Step 4: Commit**

```bash
git add static/js/workflows.js
git commit -m "feat(workflows): canvas — node render, drag, port wiring, inspector

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: List, save, run & results

**Files:**
- Modify: `static/js/workflows.js` (replace the `refreshList()` stub; add save/run/results; wire the Save/Run buttons in `init()`)
- Test: `tests/test_workflow_editor_shell.py` (the `node --check` test guards this task too)

**Interfaces:**
- Consumes: `api` (Task 3), `G.toJSON`/`G.createGraph`/`G.autoLayout`/`G.runInputNames` (Task 1), `render`/`selectNode` (Task 4), globals `graph`/`currentId`.
- Produces: `refreshList()`, `loadWorkflow(id)`, `newWorkflow()`, `save()`, `run()`, `renderResults(result)`.

- [ ] **Step 1: Replace the `refreshList()` stub and add list/load/save/run/results**

In `static/js/workflows.js`, replace the line `async function refreshList() {}` with:

```javascript
async function refreshList() {
  const host = $('wf-list');
  if (!host) return;
  host.innerHTML = '';
  const newBtn = document.createElement('button');
  newBtn.textContent = '+ New';
  newBtn.style.cssText = 'width:100%;margin-bottom:6px;';
  newBtn.addEventListener('click', newWorkflow);
  host.appendChild(newBtn);
  let list = [];
  try { list = (await api('/api/workflows')).workflows || []; }
  catch (e) { msg(e.message, true); return; }
  list.forEach((w) => {
    const row = document.createElement('div');
    row.textContent = w.name || w.id;
    row.title = `${w.id} · ${w.nodes} nodes`;
    row.style.cssText = 'padding:4px 6px;cursor:pointer;border-radius:4px;font-size:12px;overflow:hidden;'
      + 'text-overflow:ellipsis;white-space:nowrap;' + (w.id === currentId ? 'background:var(--panel);' : '');
    row.addEventListener('click', () => loadWorkflow(w.id));
    host.appendChild(row);
  });
  if (!list.length) {
    const e = document.createElement('div');
    e.style.cssText = 'font-size:12px;opacity:0.6;padding:4px;';
    e.textContent = 'No workflows yet.';
    host.appendChild(e);
  }
}

function newWorkflow() {
  graph = G.createGraph();
  currentId = null;
  selected = null;
  const nameEl = $('wf-name'); if (nameEl) nameEl.value = '';
  const results = $('wf-results'); if (results) { results.style.display = 'none'; results.innerHTML = ''; }
  msg('');
  render();
}

async function loadWorkflow(id) {
  let wf;
  try { wf = await api(`/api/workflows/${encodeURIComponent(id)}`); }
  catch (e) { msg(e.message, true); return; }
  graph = G.createGraph(wf);
  currentId = wf.id;
  if (graph.nodes.some((n) => !n.x && !n.y)) { try { G.autoLayout(graph); } catch (e) { /* cyclic legacy */ } }
  const nameEl = $('wf-name'); if (nameEl) nameEl.value = graph.name || '';
  selected = null;
  const results = $('wf-results'); if (results) { results.style.display = 'none'; results.innerHTML = ''; }
  msg('');
  render();
  refreshList();
}

async function save() {
  const nameEl = $('wf-name');
  if (nameEl) G.setName(graph, nameEl.value);
  const body = G.toJSON(graph);
  if (currentId) body.id = currentId;
  const saved = await api('/api/workflows', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  currentId = saved.id;
  graph.id = saved.id;
  await refreshList();
  return saved;
}

async function run() {
  try { await save(); }
  catch (e) { msg(e.message, true); return; }
  const inputs = {};
  for (const { name, default: def } of G.runInputNames(graph)) {
    if (!name) continue;
    const v = window.prompt(`Input "${name}":`, def || '');
    if (v === null) { msg('Run cancelled'); return; }
    inputs[name] = v;
  }
  let result;
  try {
    result = await api(`/api/workflows/${encodeURIComponent(currentId)}/run`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ inputs }),
    });
  } catch (e) { msg(e.message, true); return; }
  msg('Ran');
  renderResults(result);
}

const _PILL = { ok: '#3ba55d', error: '#ff5555', skipped: '#888' };

function renderResults(result) {
  const host = $('wf-results');
  if (!host) return;
  host.style.display = '';
  host.innerHTML = '';
  const outs = document.createElement('div');
  outs.style.cssText = 'font-size:12px;margin-bottom:8px;';
  const outEntries = Object.entries(result.outputs || {});
  outs.innerHTML = '<b>Outputs:</b> ' + (outEntries.length
    ? outEntries.map(([k, v]) => `${escapeHtml(k)} = ${escapeHtml(String(v))}`).join(' · ')
    : '<span style="opacity:0.6">(none)</span>');
  host.appendChild(outs);
  (result.log || []).forEach((entry) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:8px;align-items:center;font-size:11px;padding:2px 0;cursor:pointer;';
    const pill = document.createElement('span');
    pill.textContent = entry.status;
    pill.style.cssText = `background:${_PILL[entry.status] || '#888'};color:#fff;border-radius:8px;`
      + 'padding:0 6px;font-size:9px;text-transform:uppercase;';
    const label = document.createElement('span');
    label.style.cssText = 'flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
    label.textContent = `${entry.node} (${entry.type}) ${entry.error || entry.output || ''}`;
    const ms = document.createElement('span');
    ms.style.cssText = 'opacity:0.6;'; ms.textContent = `${entry.ms}ms`;
    row.appendChild(pill); row.appendChild(label); row.appendChild(ms);
    row.addEventListener('click', () => selectNode(entry.node));
    host.appendChild(row);
  });
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}
```

- [ ] **Step 2: Wire the Save/Run buttons in `init()`**

In `static/js/workflows.js`, inside `init()` (after the palette handler from Task 4), add:

```javascript
  const saveBtn = $('wf-save');
  if (saveBtn) saveBtn.addEventListener('click', () => {
    save().then((s) => msg(`Saved (${s.id})`)).catch((e) => msg(e.message, true));
  });
  const runBtn = $('wf-run');
  if (runBtn) runBtn.addEventListener('click', run);
```

- [ ] **Step 3: Verify syntax + manual end-to-end check**

Run: `python -m pytest tests/test_workflow_editor_shell.py --import-mode=importlib -q` (Expected: PASS).
Manual (as admin): build input→template→llm→output wired end to end → set a name → Save (it appears in the left list; reopening it restores node positions) → Run → answer the input prompt → the results panel shows the outputs and a per-node log with green/red/grey pills; clicking a log row selects that node on the canvas. Force a failure (an llm node pointing at no endpoint) → its row is red, downstream rows grey, and the run still returns (HTTP 200). Save an invalid graph state is prevented by the canvas; a server validation error surfaces in the status line.

- [ ] **Step 4: Commit**

```bash
git add static/js/workflows.js
git commit -m "feat(workflows): list, save, run + per-node results panel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Tasks 1-2 are strict TDD** (Node-harness unit tests, red→green). **Tasks 3-5 are build + `node --check` + a manual verification checklist** — the DOM/SVG/pointer behavior isn't worth mocking, and all the pure logic it needs was already extracted into the tested core. Do not skip the manual checklist; it is the real gate for those tasks.
- The `{slot}` rule in `slotsOf` MUST stay identical to Python's `model.slots_of` — the port-derivation tests are the cross-language contract check.
- No backend changes. Node `x`/`y` persist because the engine ignores unknown node keys and the store round-trips them.
- `node --check` validates module syntax without executing (so it never needs a DOM). If `node` isn't on PATH those checks skip — the string-assertion test on `index.html` still runs.
- Scope: this is the v1 editor. Do NOT add model/tool dropdowns, undo/redo, multi-select, copy/paste, or canvas zoom/pan — they are explicit non-goals.

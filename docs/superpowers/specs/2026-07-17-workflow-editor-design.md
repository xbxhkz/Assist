# Visual Workflow Editor — Design

**Goal:** A browser-based visual node-canvas editor for workflows — drag nodes on
a canvas, wire their ports, edit their config, save, and run — that reads and
writes the exact JSON the shipped headless engine already executes.

**Scope:** Sub-project 2 of the visual workflow builder. Builds entirely on the
existing admin-gated `/api/workflows` CRUD + run API (sub-project 1). **Frontend
only — no backend changes.** Non-goals (later sub-projects): scheduling / chat
triggers, the Skill export bridge, and richer node kinds (branching, loops, typed
ports). This editor targets the current text-only, five-type node model.

---

## Background — what this builds on

- **The engine + API already exist** (`src/workflows/`, `routes/workflow_routes.py`):
  a workflow is `{"id","name","nodes":[{"id","type","config"}],"edges":[{"from_node",
  "from_port","to_node","to_port"}]}`. Endpoints (all admin-gated): `GET /api/workflows`
  (list), `POST /api/workflows` (save → returns saved), `GET /api/workflows/{id}`,
  `DELETE /api/workflows/{id}`, `POST /api/workflows/{id}/run` (body `{"inputs":{...}}`)
  → `{"outputs","log"}`. Error bodies are uniformly `{"errors":[...]}`.
- **Node types + ports** (the editor mirrors these): input `()→value`; template
  `{slots}→text`; llm `{slots}→text`; tool `{slots}→result`; output `value→()`. The
  input ports of template/llm/tool are DERIVED from the `{slot}` names in their text
  config (`re` `\{(\w+)\}`, ordered-unique) — never hand-declared.
- **The engine ignores unknown node keys** — `validate` reads only `id`/`type`/
  `config`; the store round-trips the whole dict. So node **`x`/`y` positions live
  inside the workflow JSON** with zero engine/API changes.
- **Frontend conventions** (followed here): vanilla-JS IIFE modules in `static/js/`,
  DOM built programmatically, `fetch` with `credentials:'same-origin'` to JSON APIs,
  wired on `DOMContentLoaded`. Major features launch from the left **icon rail**
  (`icon-rail-btn`) into a full-screen modal (Cookbook, Memory, Gallery…). CSP:
  external JS files load as `'self'`; inline `style=""` is permitted; SVG/canvas and
  pointer events are unrestricted. The app is **local-first / offline** — no CDN
  libraries.

## Architecture

Two frontend modules, splitting pure logic from the DOM (the same split used for
voice-conversation: a testable core + a thin view):

- **`static/js/workflowGraph.js` — pure, DOM-free core.** Holds the in-memory
  workflow and all unit-testable logic:
  - `createGraph(wf?)` → a graph object seeded from an engine-JSON workflow (or empty).
  - `slotsOf(text)` — ordered-unique `{slot}` names; **must match** the Python
    `model.slots_of` rule exactly.
  - `inputPorts(node)` / `outputPorts(node)` — derived per the port table above.
  - `addNode(type)`, `removeNode(id)` (drops attached edges), `setConfig(id, config)`
    (re-derives ports; flags edges into now-removed ports), `setNodePos(id, x, y)`,
    `setNodeId(oldId, newId)` (rejects duplicate/empty ids), `setName(name)`.
  - `canConnect(fromNode, fromPort, toNode, toPort)` / `addEdge(...)` — enforces:
    output→input direction, one edge per input port (re-wire replaces), no self-loop,
    and **no edge that would create a cycle** (runs the same Kahn reachability check
    the engine uses). `removeEdge(...)`.
  - `unwiredPorts()` — input ports with no incoming edge (for hints).
  - `toJSON()` / round-trips `{id,name,nodes:[{id,type,config,x,y}],edges}`; unknown
    keys preserved.
- **`static/js/workflows.js` — DOM/SVG view + controller** (IIFE). Renders the modal,
  the saved-workflow list pane, the canvas, the config inspector, and the results
  panel; owns all pointer interaction and all API calls; delegates every state
  mutation to the core. Nodes are absolutely-positioned HTML divs on a relative
  container; **wires are a single SVG overlay** of `<path>` Béziers.

**Entry point:** a new `rail-workflows` `icon-rail-btn` (admin-only, matching the
admin-gated API), opening a full-screen `#workflows-modal`. Markup added to
`static/index.html`; the module loads like its siblings.

**Rendering choice — HTML node divs + one SVG wire-overlay** (over pure-`<canvas>`
or pure-SVG nodes): node bodies stay real DOM so text, inputs, buttons, CSS theming
and accessibility "just work"; only the wires, which need free-form geometry, are
SVG. Dependency-free and offline, consistent with the app and its CSP.

## Canvas, nodes, ports & wiring

- **Node card:** title bar (`type` label + editable `id` + × delete), input-port dots
  down the left edge, output-port dot on the right. Ports render from the core's
  derivation, so typing `{q}` into a template's text makes a `q` input port appear live.
- **Palette:** "+ node ▾" adds one of the five types at canvas center. Delete via the
  title-bar × (removes the node and its edges through the core).
- **Move:** pointer-down on a title bar → drag; `pointermove` updates `x/y` in the
  core and re-renders (wires follow); `pointerup` ends. Pointer events, not HTML5 DnD.
- **Wire:** pointer-down on an **output** port starts a pending wire (an SVG path
  tracking the cursor); `pointerup` on a compatible **input** port commits the edge via
  `addEdge` (which applies all the guards above); release elsewhere cancels. Clicking a
  wire selects it; Delete/Backspace or its × removes it.
- **Config inspector** (right pane, fields = the exact engine `config` shape):
  - input: `name`, `default`
  - template: `template` (textarea)
  - llm: `prompt` (textarea), `model` (optional; blank = the app's default endpoint),
    `system` (optional)
  - tool: `tool` (name), `args` (textarea)
  - output: `name`

  Editing a `{slots}` field re-derives that node's input ports immediately and flags any
  now-orphaned wire. **v1: `model` and `tool` are free-text** — dropdowns populated from
  available models/tools are a deferred enhancement (they would add a backend call; this
  sub-project stays zero-backend).

## Load, save & run

- **Load:** open → `GET /api/workflows` populates the list pane. Select →
  `GET /api/workflows/{id}` → core lays nodes at stored `x/y`, or auto-arranges
  left-to-right in topological order when a node lacks a position (e.g. a workflow
  authored before positions existed). "+ New" → empty canvas.
- **Save:** serialize the core → `POST /api/workflows`. On success the returned saved
  workflow (with resolved `id`) becomes current and the list refreshes. The name is a
  field atop the canvas; a blank id means the server slugifies from the name (handled by
  the store, including collision-uniquifying).
- **Run:** "Run ▶" scans for `input` nodes; if any, a small prompt collects one value per
  input name (pre-filled with each input's `default`), then `POST /api/workflows/{id}/run`
  with `{"inputs":{...}}`. Because run targets a *saved* workflow, Run **saves first if
  there are unsaved edits**, so a stale version can't run.
- **Results** (bottom panel) from `{"outputs","log"}`:
  - **outputs** (the `output` nodes' recorded values) shown prominently at the top;
  - **log** as one row per node in execution order: a status pill (`ok` green / `error`
    red / `skipped` grey), node id+type, truncated `output`, `error` if present, and `ms`.
    Hovering/selecting a row highlights that node on the canvas, so a failure traces
    visually to where it happened. This directly surfaces the engine's partial-on-failure
    behavior (failed red, dependents grey, resolved outputs still shown).

## Error handling & validation UX

Prevent-then-report: the core makes most invalid graphs undrawable, so a server 400 is a
backstop, not the primary path.

- **Live, non-blocking hints:** an input port with no incoming wire shows a subtle
  "unwired" marker; the Run button shows the unwired-port count. Save is always allowed
  (work-in-progress); the server validates on both save and run.
- **Server errors:** one shared fetch helper reads the unified `{"errors":[...]}` body and
  shows the messages in a status line — a failed save lists exactly which nodes/ports are
  wrong; a `400` on run (invalid graph) and a `404` (deleted underneath you) each get a
  clear message.
- **A node failing at runtime is not an error dialog** — it is the red log row (that path
  is HTTP 200 by the engine's contract).

## Testing

- **`workflowGraph.js` (pure core) is unit-tested** via the repo's `node --input-type=module`
  harness driven from pytest (paths passed as `Path.as_uri()`, per this project's
  Windows-ESM requirement): port derivation from `{slots}` matches the Python rules;
  add/remove node & edge; the one-edge-per-input and cycle-rejection guards; unwired-slot
  detection; `setConfig` re-derivation orphaning a wire; round-trip serialize→deserialize
  producing engine-equivalent JSON including `x/y`.
- The **port-derivation test doubles as a cross-language contract check** against Python's
  `slots_of`/`input_ports` — the place drift would surface.
- **The DOM/SVG view is not unit-tested** (drag geometry against a live pointer isn't worth
  mocking): covered by a manual verification checklist in the plan — create each node type,
  wire them, save, reload, run, read the log — plus the frozen-build smoke check that the
  new module and rail button load.

## Non-goals (this sub-project)

- Scheduling / chat triggers; the Skill export/import bridge (sub-project 3).
- Branching, loops, typed ports (the engine is text-only wires in v1).
- Model/tool dropdowns in the inspector (free-text in v1; would require a backend list call).
- Multi-select, copy/paste, undo/redo, canvas zoom/pan-minimap (polish; not v1).

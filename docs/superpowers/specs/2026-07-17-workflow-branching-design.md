# Workflow Branching — Design

**Goal:** Add conditional branching to the workflow builder: a `branch` node that
routes its input to one of several case outputs — deterministically or by an LLM
decision — so a workflow can take different paths, with the un-taken paths skipped.

**Scope:** Sub-project 3b of the visual workflow builder. Extends the shipped
engine (`src/workflows/`) and editor (`static/js/workflowGraph.js`,
`workflows.js`) with one new node type. Non-goals (later pieces): merge/join
nodes, loops, typed ports, the Skill export bridge.

---

## Background — what this builds on

- **The engine is a data-flow DAG** (`src/workflows/engine.py`): `run_workflow`
  topo-sorts, runs each node, flows values along edges, and is partial-on-failure —
  a node is skipped when `any(e.from_node in failed …)`, where `failed` accumulates
  BOTH errors and skips, so a skip cascades downstream automatically.
- **Ports are the single source of truth** (`src/workflows/model.py`):
  `input_ports(node)` / `output_ports(node)` drive validation, the editor canvas,
  and execution alike. Today: `input_ports` derives from `{slots}` in a node's text
  config (template/llm/tool) or is `["value"]` (output); `output_ports` is a fixed
  single name per type.
- **The editor** renders a node's dots by iterating those same port functions
  (`static/js/workflowGraph.js` mirrors the Python rules; `workflows.js` renders +
  inspects), so a node type defined in the ports functions renders and wires for free.

## The `branch` node

A sixth node type — a **router**. One `value` input; one output port per **case**
plus an always-present **`else`** fallback; it decides a case, forwards `value` to
that case's port, and leaves the other case ports un-taken.

- **Input port:** `value` (single, fixed — the thing routed and tested).
- **Output ports:** the config `cases` list **+ `["else"]`**. First node whose output
  ports come from config, not a fixed name.
- **Config:** `{"mode": "match" | "llm", "cases": [str, …], "prompt"?: str}`.
  - **match** (deterministic, no model): route `value` to the case whose label equals
    it (case-insensitive, trimmed); no match → `else`.
  - **llm** (agentic): ask the model to classify `value` into one of `cases` (the
    optional `prompt` guides it, e.g. "Does this email need a reply?"); the returned
    label picks the case; anything off-list → `else`.
- **Executor** `run_branch(config, inputs, *, model_call) -> {"active": <chosen case>,
  "value": <routed value>}` — names the chosen port and carries the value through.

Example: `input(email) → branch(llm "needs reply?", cases ["yes","no"]) →
[yes → llm(draft) → output] / [no → output("no action")]`. The `yes` path runs with
the email; the `no` path is skipped.

## Engine change — selective port deactivation

The whole runtime change rides the existing skip-propagation. Add one concept and
three small edits to `run_workflow`'s loop:

1. **`inactive_ports`** — a `set` of `(node_id, port)` a branch chose NOT to take.
2. **Branch special-case** (beside the `output` one): when a `branch` runs, its
   executor returns `{"active": chosen, "value": v}`; the engine sets
   `produced[nid] = {chosen: v}` (the taken port carries the value) and adds
   `(nid, p)` to `inactive_ports` for every OTHER port `p` in `output_ports(node)`.
3. **Extended skip check:** a node is skipped if
   `any(e.from_node in failed …)` **or** `any((e.from_node, e.from_port) in
   inactive_ports …)`. Because a skip lands the node in `failed`, everything
   downstream of an un-taken branch cascades with no extra code.
4. `_run_node` gains a `branch` arm → `run_branch(cfg, node_inputs, model_call=model_call)`.

No change to `topo_sort`, errors, or normal data-flow. `run_workflow` imports
`output_ports` from `model` (for step 2's "other ports" enumeration).

**Accepted v1 limitation — no merge-back.** Because a node fed by ANY inactive
incoming port is skipped, a node downstream of BOTH a taken and an un-taken branch is
always skipped. So branches run to their own terminal/`output` nodes; they don't
rejoin. A join-if-any node is a documented non-goal.

## Model & validation (`src/workflows/model.py`)

- `NODE_TYPES` gains `"branch"`.
- `input_ports(branch)` → `["value"]` (extend the `output → ["value"]` special-case).
- `output_ports(branch)` → `config["cases"] + ["else"]` (new: reads config).
- `validate` gains branch checks (same error-string style):
  - `cases` is a non-empty list of unique, non-empty strings; none equal `"else"`
    (reserved for the fallback);
  - `mode` is `"match"` or `"llm"`;
  - the `value` input port is wired (covered by the existing "every input port must be
    wired" rule, since `value` is now a derived input port);
  - edges from a branch use a `from_port` in `cases + ["else"]` (covered by the
    existing "valid output port" edge check once `output_ports` returns the cases).
- `topo_sort` unchanged — a branch is an ordinary DAG node; the no-cycle invariant holds.

## Node executor (`src/workflows/nodes.py`)

`run_branch(config, inputs, *, model_call)`:
- `value = str(inputs.get("value", ""))`; `cases = config.get("cases") or []`;
  `mode = config.get("mode", "match")`.
- **match:** `chosen = next((c for c in cases if c.strip().lower() ==
  value.strip().lower()), "else")`.
- **llm:** build a classification prompt (the value, the case list, the optional
  guidance `prompt`, "answer with only one label"); `resp = await model_call(prompt,
  …)`; map `resp` to a case deterministically — first a case whose label equals
  `resp` (trimmed, case-insensitive); else the first case (in `cases` order) whose
  label appears as a substring of `resp` (case-insensitive), which tolerates a chatty
  "The answer is yes."; else `"else"`.
- returns `{"active": chosen, "value": value}`.

`model_call` is injected (default reused from the engine), so tests never hit a model.

## Editor (`static/js/workflowGraph.js` + `workflows.js`)

- **`workflowGraph.js` (pure core):** add `"branch"` to `NODE_TYPES`; mirror Python —
  `inputPortsOf(branch) → ["value"]`, `outputPortsOf(branch) → cases + ["else"]`. This
  is the **cross-language parity** point — a Node test asserts the JS port output
  equals Python's `output_ports`/`input_ports` for the same branch config.
- **Palette** lists `NODE_TYPES`, so `branch` appears automatically.
- **Canvas** renders output dots by iterating `outputPortsOf`, so a branch shows one
  dot per case + `else` for free.
- **`workflows.js` inspector:** a branch-specific inspector — a `mode` dropdown
  (match/llm), a **cases list editor** (add/remove/edit case rows — the one net-new
  UI widget), and an optional `prompt` textarea shown in llm mode. Relabeling a case
  relabels its output port live and drops any now-orphaned wire (reusing the existing
  `setConfig` prune).

No new backend calls, no new files.

## Error handling

- Invalid branch config (empty/duplicate/`"else"` cases, bad `mode`, unwired `value`)
  → a `validate` error string → the route's existing 400 path.
- llm-mode model failure → the branch node raises like any node → captured in the log,
  its downstream skipped (existing partial-on-failure).
- An off-list / empty model response → routed to `else` (no crash).

## Testing

- **`model.py` (unit):** `output_ports(branch)` = `cases + ["else"]`;
  `input_ports(branch)` = `["value"]`; `validate` flags empty/duplicate/`"else"`
  cases, bad `mode`, and an unwired `value`.
- **`nodes.py` `run_branch` (unit):** match routes to the equal case and to `else`;
  llm with a **mocked `model_call`** routes to the returned case and to `else`
  off-list. No real model runs.
- **`engine.py` (unit):** a branch workflow where the taken path produces its output
  and the un-taken path's nodes are logged `skipped` and absent from `outputs` — for
  both a match branch and a mocked-llm branch; plus a cascade test (a chain behind the
  un-taken branch all skips).
- **`workflowGraph.js` (Node unit):** `outputPortsOf`/`inputPortsOf` for a branch
  EQUAL the Python `output_ports`/`input_ports` for the same config (cross-language
  contract); `NODE_TYPES` includes `branch`.
- **Editor view:** `node --check` + a manual checklist (add a branch, edit cases, wire
  per-case paths, run, watch one path light up and the other grey out).

## Non-goals (this sub-project)

- Merge/join nodes (run-if-any); loops (the no-cycle invariant stays); typed ports.
- A match-mode operator/expression DSL (`>`, `contains`, regex) — exact-label match only.
- llm-mode few-shot/temperature config beyond the optional `prompt`.
- The workflow→Skill export bridge (a separate remaining piece of sub-project 3).

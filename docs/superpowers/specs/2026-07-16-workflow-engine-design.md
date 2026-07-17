# Workflow Engine Foundation — Design

**Goal:** A headless workflow engine: define a workflow as a JSON node-graph and
**run it** — DAG execution with text values flowing along wires — returning the
outputs plus a per-node run log.

**Scope:** Sub-project 1 of the visual workflow builder. The runnable engine +
node model + persistence + a run API. **No visual editor** (sub-project 2), no
branching/loops/typed ports, no scheduling/chat triggers, no Skill export bridge
— those are later sub-projects. This is a self-contained, fully unit-testable
backend.

---

## Background — what this builds on (and relates to)

- **No existing workflow/node engine** — greenfield. Distinct from **Skills**
  (`services/memory/skills.py` — prose `SKILL.md` procedures *injected into the
  agent prompt*, not executed) and **scheduled Tasks** (`src/task_scheduler.py`).
  A workflow is a *structured, executed graph*; a Skill is *prose guidance*. They
  will bridge later (export a workflow → a Skill), not here.
- **LLM node reuses** the endpoint layer: `resolve_endpoint(...)`
  (`src/endpoint_resolver.py`) yields `(url, api_key)` for a `/chat/completions`
  POST — the same path the app uses everywhere.
- **Tool node reuses** `TOOL_HANDLERS` (`src/agent_tools/__init__.py` /
  dispatched in `src/tool_execution.py`): a name→handler map; a handler takes
  `(content, ctx)` and returns a result dict. The tool's *own* admin/consent
  gates still apply when a workflow calls it.
- **Persistence** mirrors the `DATA_DIR` file pattern (`src/constants.py`):
  workflows are JSON files under `DATA_DIR/workflows/`.

## Data model (`src/workflows/model.py`)

```
Workflow = {"id": str, "name": str, "nodes": [Node], "edges": [Edge]}
Node     = {"id": str, "type": "input"|"template"|"llm"|"tool"|"output", "config": {...}}
Edge     = {"from_node": str, "from_port": str, "to_node": str, "to_port": str}
```

Each node type has named **input ports** and **output ports**. For template/llm/
tool nodes the input ports are the `{slot}` names found in the node's text
config (derived, not hand-declared). Node configs + ports:

| type      | input ports              | output port | config |
|-----------|--------------------------|-------------|--------|
| input     | (none)                   | `value`     | `{"name": str, "default"?: str}` |
| template  | the `{slots}` in template| `text`      | `{"template": str}` |
| llm       | the `{slots}` in prompt  | `text`      | `{"prompt": str, "model"?: str, "system"?: str}` |
| tool      | the `{slots}` in args    | `result`    | `{"tool": str, "args": str}` |
| output    | `value`                  | (none)      | `{"name": str}` |

`validate(wf) -> list[str]` returns errors (empty = valid): unique node ids;
known types; every edge's `from_node`/`to_node` exist and `from_port`/`to_port`
are valid for those node types; the graph is a **DAG** (topological sort
succeeds); every derived input slot has an incoming edge.

## Node executors (`src/workflows/nodes.py`)

`_fill(template: str, inputs: dict) -> str` replaces each `{slot}` with
`str(inputs.get(slot, ""))` (slots = `re.findall(r"\{(\w+)\}", template)`). Per-type:

- `run_input(config, run_inputs) -> {"value": run_inputs.get(config["name"], config.get("default","")) }`
- `run_template(config, inputs) -> {"text": _fill(config["template"], inputs)}`
- `run_llm(config, inputs, *, model_call) -> {"text": await model_call(_fill(config["prompt"], inputs), config.get("model"), config.get("system"))}`
- `run_tool(config, inputs, ctx, *, tool_dispatch) -> {"result": await tool_dispatch(config["tool"], _fill(config["args"], inputs), ctx)}`
- `run_output(config, inputs) -> {}` (the engine records `inputs["value"]` under `config["name"]`)

`model_call` and `tool_dispatch` are **injectable** so tests never hit a real
model or tool. Defaults:
- default `model_call(prompt, model, system)` — `resolve_endpoint(...)` →
  `POST {url}` with `{"model": model or <app default>, "messages": [system?, {"role":"user","content":prompt}], "stream": False}` → `choices[0].message.content`.
- default `tool_dispatch(tool, args, ctx)` — `TOOL_HANDLERS[tool](args, ctx)` →
  the handler's `output`/text (unknown tool → clear error).

## Execution engine (`src/workflows/engine.py`)

`async run_workflow(wf, inputs, ctx, *, model_call=None, tool_dispatch=None) -> {"outputs": {name: value}, "log": [entry]}`:

1. `errs = validate(wf)`; if any → raise `WorkflowError(errs)` (route → 400).
2. Topologically sort nodes (Kahn's algorithm).
3. In order, for each node: gather `node_inputs` = `{edge.to_port: outputs[edge.from_node][edge.from_port]}` over incoming edges; run its executor; store `outputs[node.id]`. `input` nodes read from the run's `inputs` dict; `output` nodes record `inputs["value"]` under `config["name"]`.
4. **On a node exception:** log `status:"error"` with the message; mark the node failed; any node with an upstream failed/skipped node is **skipped** (`status:"skipped"`) — the run continues and returns a partial result, never crashes.
5. `log` entry per node: `{"node": id, "type": type, "status": "ok"|"error"|"skipped", "output": <truncated str>, "error": str|None, "ms": int}`.

## Persistence (`src/workflows/store.py`)

`WORKFLOWS_DIR = os.path.join(DATA_DIR, "workflows")`. `_safe_id(id)` rejects
`/ \ ..` (like the Skills/LoRA path-safety). `list_workflows()`,
`get_workflow(id)`, `save_workflow(wf)` (writes `<id>.json`, id slugified from
name if absent), `delete_workflow(id) -> bool`.

## Routes (`routes/workflow_routes.py`)

Prefix `/api/workflows`, **admin-gated** (a workflow runs the LLM + arbitrary
tools = agent-level power): `GET ""` (list), `POST ""` (create/save → returns the
saved workflow), `GET /{id}`, `DELETE /{id}`, and
`POST /{id}/run` (body `{"inputs": {name: value}}`) → `run_workflow(...)` →
`{"outputs", "log"}` (validation error → 400).

## Error handling

- Invalid graph (cycle / dangling edge / unknown type / unwired slot) → 400 with
  the specific reason(s) from `validate`.
- Per-node execution failure (LLM/tool error, unknown tool) → captured in the
  log, dependents skipped, run returns partial `{outputs, log}` (HTTP 200 — a
  node failing is a run result, not a request error).
- Missing run input for an `input` node → its `default` (or empty string).

## Testing

- **`model.validate`:** valid graph passes; cycle detected; dangling edge; unknown
  node type; unwired template slot flagged.
- **`nodes`:** `_fill` slot substitution (incl. missing slot → empty); each
  executor with mocked `model_call`/`tool_dispatch`; `run_output` recording.
- **`engine.run_workflow`:** a small `input → template → llm(mock) → output`
  graph returns the expected output + an all-`ok` log; a failing node → its
  dependents `skipped`, partial outputs, error logged; topo order respected.
- **route:** CRUD round-trip against `store` + `POST /{id}/run` with a mocked
  engine → `{outputs, log}`; invalid graph → 400.

All tests use injected/mocked `model_call` + `tool_dispatch` — no real model,
endpoint, or tool executes.

## Non-goals (this sub-project)

- The visual node-canvas editor (sub-project 2 — it will edit this JSON).
- Branching/conditionals, loops, typed ports (text-only wires for v1).
- Scheduling / chat triggers; the Skill export/import bridge.
- Multi-output-per-port fan-in merging beyond last-writer (v1: one edge per input port).

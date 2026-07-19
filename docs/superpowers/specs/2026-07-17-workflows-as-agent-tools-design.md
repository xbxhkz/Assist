# Workflows as Agent Tools — Design

**Goal:** Let the agent invoke a saved workflow from chat by adding a single
admin-only builtin tool, `run_workflow`, that lists saved workflows or runs one
by id and returns its outputs — reusing the shipped engine.

**Scope:** Sub-project 3c of the visual workflow builder (the useful reframe of
the "Skill export bridge"). A workflow becomes the fourth way to run: editor
"Run", triggers, `POST /api/workflows/{id}/run`, and now the agent. Non-goals:
per-workflow dynamic tools, workflow-invokes-workflow composition, a non-admin
path, loops/typed ports, a literal prose→SKILL.md export.

---

## Background — what this builds on

- **The engine is shipped** (`src/workflows/`): `store.list_workflows()` /
  `store.get_workflow(id)`; `async run_workflow(wf, inputs, ctx, *, model_call=None,
  tool_dispatch=None) -> {"outputs", "log"}` — partial-on-failure (a failing node is
  in the log, not raised), using default `model_call`/`tool_dispatch` when none
  injected. Branching (sub-project 3b) is part of the same engine.
- **Builtin tools** live in `src/agent_tools/` as `SomeTool().execute(content, ctx)`
  async handlers, registered in a `TOOL_HANDLERS` name→handler map. A tool is wired
  at several registration surfaces (handler map, tags, schema, security lists, index,
  agent-loop sections/domain map, a parity test) — missing any one silently
  half-registers the tool.
- **Security fact (load-bearing):** the engine's `default_tool_dispatch`
  (`src/workflows/nodes.py`) calls `TOOL_HANDLERS[tool](args, ctx)` **directly, with
  no `tool_security` check** — so a workflow's `tool` nodes run ungated by the
  agent-loop admin gate. `/api/workflows` and workflow triggers are already
  admin-only for this reason.

## The `run_workflow` tool

One new builtin tool. Handler `RunWorkflowTool().execute(content, ctx)`. It reads
two optional args from the call: `id` (workflow to run) and `inputs` (a name→value
object). Two modes:

- **List mode** (no `id`, or `action:"list"`): return the saved workflows via
  `store.list_workflows()`, each augmented with its **input-node names** (computed
  from the stored graph's `type=="input"` nodes' `config.name`), so the model learns
  what exists and what inputs each needs.
- **Run mode** (`id` given): `wf = store.get_workflow(id)`; missing → error. Else
  `await run_workflow(wf, inputs or {}, child_ctx)` with the engine's default
  `model_call`/`tool_dispatch`. Return the workflow's `outputs` + a one-line status
  tally (`N ok, M error, K skipped`).

The tool is a **thin adapter** over the shipped engine + store — no new execution
logic, no engine/editor change. It may live in its own module in the agent-tools
layer (following the sibling pattern) — purely additive.

## Registration

`run_workflow` is wired at every builtin-tool registration surface (the plan
enumerates and verifies each precisely; the list below is the intent):

- `src/agent_tools/__init__.py` — `TOOL_HANDLERS["run_workflow"] =
  RunWorkflowTool().execute`; add `"run_workflow"` to `TOOL_TAGS` (an unlisted tag →
  the native call is rejected as an unknown function before dispatch).
- `src/tool_schemas.py` — the OpenAI-style function schema (`id`, `inputs`, both
  optional, `required: []`) + any tool-name list.
- `src/tool_security.py` — add `"run_workflow"` to **`NON_ADMIN_BLOCKED_TOOLS`**
  (admin-only). Deliberately **NOT** in `PLAN_MODE_READONLY_TOOLS` — the tool
  executes side effects, so a plan-mode agent must not call it (the read-only list
  mode doesn't change this — the tool as a whole executes).
- `src/tool_index.py` — the tool-index entry.
- `agent_loop.py` — `TOOL_SECTIONS` (so **local models see it** in the prompt tool
  list) **and** `_DOMAIN_TOOL_MAP` (so it loads for the right domain).
- The registration-parity test — include `run_workflow` (or it silently skips it).

## Security & recursion

- **Admin-only** (`NON_ADMIN_BLOCKED_TOOLS`) is the boundary: since inner `tool`
  nodes run ungated, only an admin — who could invoke those inner tools directly —
  may run a workflow. No escalation; consistent with the API + triggers.
- **Recursion guard:** `run_workflow` is in `TOOL_HANDLERS`, so a workflow `tool`
  node could set `tool:"run_workflow"` and nest unbounded. When the tool runs a
  workflow it passes a **marked child context** — `child = dict(ctx or {});
  child["_in_workflow"] = True; child["owner"] = <owner>` — to `run_workflow`
  (never mutating the caller's ctx). The engine threads that ctx to `tool` nodes via
  `default_tool_dispatch`, so a nested `run_workflow` invocation sees
  `ctx.get("_in_workflow")` set and **refuses** with a clear error. This blocks all
  workflow-invokes-workflow nesting in v1 (self or other); it can't be bypassed from
  inside a workflow because the flag rides the engine-controlled ctx.

## Result shape & error handling

The tool returns a result dict (`{"output": <str>}` on success, `{"error": <str>}`
on failure — never raising into the agent loop):

- **List mode** → the workflows as `{id, name, inputs: [names]}`.
- **Run mode** → the `outputs` shown prominently + the status tally. A node failing
  at runtime is **not** a tool error (partial-on-failure): the tool returns whatever
  outputs resolved and notes the failed nodes.
- **Errors** (as `{"error": …}`): bad/missing `id` → "workflow '<id>' not found";
  invalid graph → `run_workflow` raises `WorkflowError`, caught and surfaced as the
  validation reasons; a nested call → the recursion-guard refusal.

## Testing

- **Tool handler (unit; `run_workflow` mocked at the seam — no real model/tool):**
  list mode returns saved workflows + input names; run mode loads/runs/returns
  outputs; a missing workflow → error (no raise); an invalid graph (`WorkflowError`)
  → error (no raise); the recursion guard — a `ctx` with `_in_workflow` set →
  refuses; the child ctx handed to the engine has `_in_workflow=True` + `owner`
  **without mutating** the caller's ctx.
- **Registration parity (unit):** assert `run_workflow` ∈ `TOOL_HANDLERS`,
  `TOOL_TAGS`, the schema names, `NON_ADMIN_BLOCKED_TOOLS`, `tool_index`, and
  `agent_loop`'s `TOOL_SECTIONS`/`_DOMAIN_TOOL_MAP`; and **∉** `PLAN_MODE_READONLY_TOOLS`.
- **No automated end-to-end agent test** (needs a live model): the "ask the agent to
  run a workflow from chat" path is a short manual check.

## Non-goals (this sub-project)

- Per-workflow dynamic tools (one tool per saved workflow — fights the static
  registration model).
- Workflow-invokes-workflow composition / nesting (blocked by the recursion guard).
- Any non-admin path to running a workflow.
- Streaming/async results (the tool runs synchronously and returns outputs).
- A literal workflow→prose `SKILL.md` export (superseded by this callable-tool reframe).
- Loops / typed ports (separate deferred pieces).

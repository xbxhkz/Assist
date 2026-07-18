# Workflow Triggers — Design

**Goal:** Run a saved workflow automatically — on a schedule, from a chat/session
event, or via an external webhook — by making a workflow a new kind of the
existing `ScheduledTask`, executed through the already-shipped `run_workflow`
engine.

**Scope:** Sub-project 3 of the visual workflow builder (the first of its three
independent pieces — triggers). Builds on sub-project 1 (the headless engine +
`/api/workflows` store) and sub-project 2 (the visual editor). It adds a trigger
*type* and its execution branch; it does NOT modify the engine or the editor's
graph logic. Non-goals (later pieces): the workflow→Skill export bridge, and
richer nodes (branching/loops/typed ports).

---

## Background — what this builds on

- **The scheduler is mature** (`src/task_scheduler.py`, `core.database.ScheduledTask`,
  `routes/task_routes.py`). `ScheduledTask` already has `task_type` (today
  `llm`/`action`/`research`), `trigger_type` (`schedule`/`event`/`webhook`),
  `schedule`/`scheduled_time`/`scheduled_day`/`scheduled_date`/`cron_expression`/
  `next_run`, `trigger_event`/`trigger_count`, `webhook_token`, `output_target`,
  and a `TaskRun` history. `TaskScheduler._execute_task_locked` dispatches on
  `task_type` (~`src/task_scheduler.py:910`). `routes/task_routes.py` `create_task`
  already mints a `webhook_token` when `trigger_type == "webhook"`.
- **The workflow engine is shipped** (sub-project 1): `src/workflows/store.py`
  `get_workflow(id)`, and `src/workflows/engine.py`
  `async run_workflow(wf, inputs, ctx, *, model_call=None, tool_dispatch=None)
  -> {"outputs": {...}, "log": [...]}` — partial-on-failure (a failing node is
  captured in the log, not raised), using default `model_call` (`resolve_endpoint`)
  and `tool_dispatch` (`TOOL_HANDLERS`) when none are injected.
- **The editor is shipped** (sub-project 2): `static/js/workflows.js` — the full-
  screen modal where a workflow is authored and saved.

## Data model — a workflow trigger IS a ScheduledTask

A workflow trigger is a `ScheduledTask` with **`task_type = "workflow"`**. It reuses
the mature scheduler wholesale (all `trigger_type`s, cron/time model, `next_run`,
event dispatch, `webhook_token`, `TaskRun` records, pause/resume, Tasks UI).

**Zero schema migration** — overload the existing per-type columns exactly as
`llm`/`action`/`research` already do:
- `action` → the **workflow id** to run (mirrors "action = builtin action name").
- `prompt` → a JSON string of the trigger's **fixed inputs** map
  (`{"name": "value", …}`), or empty/`{}`.

Example: `ScheduledTask(task_type="workflow", action="daily-digest",
prompt='{"topic":"AI"}', trigger_type="schedule", schedule="daily",
scheduled_time="08:00")`. No new table or column.

## Execution — the scheduler's `workflow` branch

`_execute_task_locked`'s `task_type` dispatch gets a new
`elif task_type == "workflow":` branch (sibling to `action`/`research`), so it
inherits the run-record lifecycle, notifications, and concurrency guard:

1. **Load** the workflow via `store.get_workflow(task.action)`. Missing → the run
   records `workflow '<id>' not found` and ends (no crash); the trigger stays active.
2. **Resolve inputs** (see below) → `{name: value}`.
3. **Run**: `await run_workflow(wf, inputs, {"owner": task.owner})`. Uses the
   engine's default `model_call`/`tool_dispatch`, so each `llm`/`tool` node behaves
   exactly as in a manual editor run; `ctx={"owner": task.owner}` carries the owner
   so each tool's own consent/admin gate applies.
4. **Record** the `TaskRun`: result summary = the `output` nodes' values + a per-node
   status tally (e.g. `answer="…" · 3 ok, 1 error`); the full `{outputs, log}` is
   kept for the run-detail view. Because the engine is partial-on-failure, a node
   failure produces a *recorded run*, not a scheduler exception.

**Concurrency:** a workflow run acquires the scheduler's model slot like an `llm`
task (workflows usually invoke the local model), so triggered runs don't stampede
the GPU.

## Input resolution — the convention (`resolve_trigger_inputs`)

A pure function `resolve_trigger_inputs(workflow, fixed_inputs, context) ->
{name: value}` builds the run inputs by a deterministic precedence chain, lowest to
highest:

1. **Node default** — each `input` node's `config.default`. (The engine already
   applies this as its final fallback, so the resolver need not duplicate it; it may
   simply omit a name it has no value for and let the engine default it.)
2. **Trigger fixed inputs** — the `prompt` JSON map, overlaid by name. This is the
   whole story for a **schedule** trigger.
3. **Context** — overlaid last, only for event/webhook:
   - **Webhook**: the POST body JSON. If it has an `inputs` object, those entries
     override by name; otherwise top-level body keys matching input names override.
   - **Event** (`message_sent`/`session_created`): the triggering **message text** is
     injected into an input named `message` **if the workflow has one**; otherwise the
     event contributes nothing.

Only names matching actual `input` nodes are used; unknown keys are ignored. The
resolver is independent of the scheduler and engine, so it is fully unit-testable.

## Trigger creation & management UI

A **"Triggers" panel** in the workflow editor modal lists the current workflow's
triggers and offers **+ Add trigger**, all via the existing `/api/tasks` API (no new
CRUD endpoints). The add form is driven by trigger type:
- **Schedule** → `once/daily/weekly/cron` + the time/day/cron field the Tasks system
  already uses.
- **Event** → a picker of supported events (`message_sent`, `session_created`) +
  optional `trigger_count` ("every N").
- **Webhook** → no config; on save the server mints a `webhook_token` and the panel
  shows the ready-to-POST trigger URL with a copy button.

Below the type, a **fixed-inputs mini-editor**: one row per `input` node (name shown,
value editable, pre-filled with the node's default). These rows serialize to the
`prompt` JSON.

Creating a trigger requires the workflow to be **saved first** (it references the
workflow by id); the panel prompts to Save if unsaved. Delete is `DELETE
/api/tasks/{id}`. Because triggers are `ScheduledTask`s, they also appear in the
existing **Tasks modal** with full run history — the editor panel is a focused
convenience view, not a parallel store. **v1** = add/list/delete + webhook-URL
display in the editor; rich edits (changing an existing schedule) use the Tasks modal.

## Results

Every triggered run is a `TaskRun` carrying the workflow's `outputs` + per-node `log`.
- **Schedule/event** runs record the `TaskRun` and fire the scheduler's existing
  notification; if `output_target` is a session, a compact summary posts there
  (existing mechanism).
- **Webhook** fires the run through the same scheduler path the existing task-webhook
  trigger uses and returns its standard ack (run id + status); the workflow's
  `{outputs, log}` lands on the `TaskRun`. (Synchronously returning outputs in the
  webhook response is a later enhancement; v1 matches existing task-webhook behavior.)

## Security

This is the riskiest surface: a workflow runs the LLM + arbitrary agent tools, and a
trigger runs it **without a human in the loop**, possibly from an external webhook.

- **Creating a workflow trigger is admin-only** — matching the admin-gated
  `/api/workflows` API. The trigger-creation path in `create_task` enforces admin for
  `task_type="workflow"`.
- The run executes with the **owner's privileges**; each tool's own consent/admin
  gate still applies on top (the engine dispatches through `TOOL_HANDLERS` with
  `ctx={"owner": ...}`).
- The **webhook token** is long and random (`secrets.token_urlsafe(32)`). A leaked
  token can fire agent-level power, so the panel treats the URL as a secret; deleting
  the trigger revokes it.

## Error handling

- Missing/deleted workflow → the run records a clear error; the trigger stays active.
- A node failing at runtime → a partial run recorded (engine HTTP-200 contract), never
  a scheduler crash.
- Invalid fixed-inputs JSON at create time (not a JSON object) → HTTP 400.
- A non-admin attempting to create a `workflow` trigger → 403.

## Testing

- **`resolve_trigger_inputs`** (pure): precedence default < fixed < context; webhook
  body and `inputs`-object merge; event message → `message` input (present/absent);
  unknown keys ignored; a schedule trigger uses only fixed inputs.
- **Scheduler `workflow` branch**: with a **mocked `run_workflow`**, a fired trigger
  loads the workflow, resolves inputs, and records a `TaskRun` with the outputs/log
  summary; a missing workflow records the error and doesn't raise.
- **`create_task` validation** for `task_type="workflow"`: requires a non-empty
  `action` (workflow id); `prompt`, if present, must parse as a JSON object; a
  non-admin owner → 403.
- No test runs a real model, endpoint, or tool (the engine's `model_call`/
  `tool_dispatch` are mocked, or `run_workflow` itself is mocked at the scheduler seam).

## Non-goals (this sub-project)

- The workflow→Skill export bridge; richer nodes (branching/loops/typed ports).
- Explicit per-input mapping UI (convention-based only in v1).
- Synchronous webhook responses that return the workflow outputs inline (v1 returns
  the scheduler's ack; the outputs are on the `TaskRun`).
- Rich in-editor editing of an existing trigger's schedule (use the Tasks modal).
- New DB columns / schema migration (existing columns are reused).

# "AI Mission Control" Dashboard — Sub-project 2c: Live Workflow Execution — Design Spec

## Context

Sub-project 1 shipped a 5-widget read-only aggregation dashboard. Its design deferred three areas
needing genuinely new infrastructure to an unscoped "sub-project 2" — chat-level tool-call logs
(shipped as sub-project 2a), active agents (shipped as sub-project 2b), and live workflow
execution (this spec, sub-project 2c) — **completing the full 3-part sub-project 2 scope**.

Live-code research (re-confirmed at the start of this sub-project's brainstorming, since some
time had passed since the original decomposition) found the workflow-engine facts from the
original roadmap still hold exactly:

- **`run_workflow()`** (`src/workflows/engine.py:33`) runs a workflow's full node graph to
  completion synchronously within one `async def` call — topo-sorts nodes, executes each via
  `_run_node()`, returns `{"outputs": ..., "log": [...]}` once every node has run. Nothing is
  persisted mid-run; no DB session opens, no module-level state is written anywhere in the
  function or its node executors.
- **No `WorkflowRun` DB model exists** — confirmed by an exhaustive list of every
  `class ...(Base)` in `core/database.py` (26 models, none workflow-related). Workflow
  *definitions* themselves aren't DB rows either — one JSON file per workflow under
  `DATA_DIR/workflows/<id>.json` (`src/workflows/store.py`).
- **Exactly 3 call sites**, all confirmed still collapsing the per-node `log` to a summary
  (or, for the direct API, returning it once in an HTTP response with nothing kept server-side
  after):
  1. The admin-only agent tool (`src/agent_tools/workflow_tool.py:55-63`) — collapses to a
     one-line `"Outputs: {...} · N ok, N error, N skipped"` string.
  2. The direct API route (`routes/workflow_routes.py:63-75`, `POST /api/workflows/{wid}/run`,
     admin-gated) — returns the full `{outputs, log}` dict in the HTTP response only.
  3. The scheduled-task executor (`src/task_scheduler.py:1272-1299`, `_execute_workflow_task`)
     — collapses to a summary string written into `TaskRun.result`/`.error`.
- **A relevant precedent exists but isn't wired to workflows**: `task_scheduler.py` already has
  a live-progress mechanism, `_set_run_progress(run_id, message)` (`task_scheduler.py:354-368`),
  which overwrites `TaskRun.result` mid-run for polling — used today by scheduled *action*-type
  tasks via a `progress_cb` parameter. `_execute_workflow_task` does not receive or use it.
  This spec does not adopt this mechanism (see Architecture below for why), but it's worth
  recording as prior art for anyone extending scheduled-task progress tracking generally.
- **`TaskRun.steps`** ("JSON log of agent tool calls") remains a schema column nothing writes to
  — reconfirmed, not touched by this spec either.
- Every node execution already produces one uniform `log` entry
  (`{"node", "type", "status", "output", "error", "ms"}`) regardless of node type (6 types:
  `input`, `template`, `llm`, `tool`, `output`, `branch`) — this spec does not need to read that
  shape, since per-node detail is explicitly out of scope (see Scope Decisions), but it's the
  reason a future per-node-progress feature would be a contained addition, not a redesign.

## Scope Decisions (resolved during brainstorming, not left open)

- **All 3 trigger paths are covered** — agent tool, direct API, and scheduled task all show up
  in the same live list. (Rejected: scoping to scheduled-tasks-only, which would have been the
  smallest lift by reusing `_set_run_progress`/`TaskRun.result`, but would have missed
  manually-triggered and agent-triggered runs — arguably the more valuable, more "live" case for
  a dashboard. Rejected: scoping to agent-tool+API-only, which would have missed the one trigger
  path that already has SOME infrastructure to lean on.)
- **Live snapshot only, no persisted history** — a run appears while in progress and disappears
  the instant it finishes (success or failure alike), mirroring sub-project 2b's Active Agents
  widget exactly. No new database table. (Rejected: a `WorkflowRun` table with persisted
  history, matching Tool Call Log's approach — bigger lift, and "live workflow execution" was
  the roadmap's own framing; a history browser is a different, separately-scoped feature if ever
  wanted.)
- **Status only, no per-node progress** — the widget shows that a workflow is running (which
  one, how it was triggered, when it started), not which node it's currently on.
  (Rejected: per-node progress, which would require adding a progress-callback parameter to
  `run_workflow()` itself and threading it through the node-execution loop — a real, if
  contained, change to code every dashboard sub-project so far has deliberately left untouched.
  This remains a natural future extension: the callback slot and the uniform per-node `log`
  shape described above are exactly what such a feature would consume, but building it now
  would be scope creep past what this spec's brainstorming settled on.)

## Goal

A user (admin — see Access Control) opens Mission Control and sees, at a glance, which
workflows are currently executing right now, however they were triggered, with no new database
table and no changes to `run_workflow()`'s execution logic.

## Architecture

**A new module, `src/workflow_runs.py`**, structurally mirroring `src/agent_runs.py`'s
established pattern from sub-project 2b exactly: an in-memory dict keyed by a generated run id
(a run id, not a workflow id, since two concurrent runs of the *same* workflow — e.g. a manual
trigger racing a scheduled one — are structurally possible and must not collide). Three
functions:

- `start(workflow_id: str, workflow_name: str, owner: Optional[str], trigger: str) -> str` —
  generates a run id, stores `{workflow_id, workflow_name, owner, trigger, started_at}`, returns
  the id. `trigger` is one of `"agent_tool"`, `"api"`, `"scheduled"` — a plain string, not an
  enum, matching this codebase's existing string-based status conventions (e.g. `agent_runs.py`'s
  own `"running" | "done" | "error" | "stopped"` status field).
- `finish(run_id: str) -> None` — removes the entry unconditionally (success or failure alike —
  "live" means currently running; once done, gone, no history kept).
- `list_active() -> List[Dict]` — returns every current entry as a list of dicts, each
  `{"run_id", "workflow_id", "workflow_name", "owner", "trigger", "started_at"}`.

**Each of the 3 existing `run_workflow()` call sites gets wrapped**, not modified internally:

```python
run_id = workflow_runs.start(workflow_id, workflow_name, owner, "agent_tool")  # or "api" / "scheduled"
try:
    result = await run_workflow(wf, inputs, ctx, ...)
finally:
    workflow_runs.finish(run_id)
```

`run_workflow()` itself is not touched — this is purely additive scaffolding around its existing
call sites, the same "add a read/wrapper, never touch the delicate core" discipline sub-project
2b applied to `agent_runs.py`. The exact values available for `workflow_id`/`workflow_name`/
`owner` at each of the 3 call sites (e.g. does the agent-tool context reliably expose an
`owner`?) are confirmed against live code at plan-writing time, not guessed here.

**A new route**, `GET /api/workflow-runs/active`, wraps `workflow_runs.list_active()`. No new
access-control gate of its own — see Access Control below for why none is needed.

**An 8th Mission Control widget**, "Workflow Runs" — unlike sub-project 2b's Active Agents
widget (which has no open-link button because no separate view exists), this widget *does* get
a standard `mc-open-*` open-link button, "Open Workflows," matching the original 6 widgets'
pattern — a real Workflows editor/list panel already exists from an earlier initiative, so
there's a natural place to link to. Entries themselves are non-interactive display only (no
per-entry click action, since there's no per-run detail view to jump to — this spec doesn't
build one).

## Access Control

**No new gate — this widget's data endpoint is expected to inherit the existing admin-only
gate the whole workflows subsystem already has** (`/api/workflows` CRUD+run is admin-gated per
sub-project 1 of the original workflow-builder initiative; the agent tool and direct-API run
paths are both already admin-only). This spec does not add a fresh access check to
`GET /api/workflow-runs/active` — it's expected to reuse whatever gate already protects
workflow visibility generally, the same "Mission Control adds no gate of its own" principle
every prior widget has followed. The exact existing gate mechanism (a route dependency, a
decorator, an inline check) is confirmed against live code at plan-writing time. A non-admin
caller sees the same per-widget "Failed to load: Admin only" error state the Integrations
widget already demonstrates (sub-project 1) — not a special case, the established pattern.

## Data Flow

Widget open/refresh → `GET /api/workflow-runs/active` → `workflow_runs.list_active()` → returned
as-is (no additional owner filtering beyond the admin gate itself, since only admins can trigger
workflows in the first place — an admin dashboard reasonably shows all admins' in-flight runs,
not just the viewer's own, matching how the workflows subsystem itself has no per-admin
ownership scoping today).

## Error Handling

Independent per-widget failure handling, matching every widget shipped so far. A `finish()` call
is wrapped in the caller's own `finally` block at each of the 3 call sites, so a run is always
untracked even if `run_workflow()` itself raises — no entry can "leak" and appear stuck as
perpetually running after its actual run has ended.

## Testing

**Backend:** unit tests for `workflow_runs.start()`/`finish()`/`list_active()` (a run appears
after `start()`, disappears after `finish()`, multiple concurrent runs of the same workflow id
get distinct entries). Route tests for `GET /api/workflow-runs/active` (matching the established
`_route()`-based direct-call test style). One test per call-site wrapper confirming `finish()`
still runs when `run_workflow()` raises (the `finally`-block guarantee).

**Frontend:** source-presence tests for the 8th widget matching the established pattern —
including extending the two seam-guard tests from sub-project 2a's final review
(`test_mission_control_loadAllWidgets_wires_all_loaders`,
`test_mission_control_link_targets_exist_in_html`/
`test_mission_control_open_handlers_wire_to_correct_targets`, since THIS widget, unlike Active
Agents, does have an `mc-open-*` button and belongs in all three seam-guard tests). `esc()`
applied to `workflow_name` before `innerHTML` (a user-controlled string).

Manual GUI verification (does a real workflow run actually show up while executing and
disappear when done, across all 3 trigger paths) is owed by the user, same as every prior
frontend sub-project this session — this is the hardest area to unit-test meaningfully, since it
depends on real concurrent execution timing across 3 different subsystems.

## Out of Scope

- Per-node live progress within a run (see Scope Decisions) — a natural future extension, not
  built here.
- Persisted run history (see Scope Decisions) — a different, separately-scoped feature if ever
  wanted.
- Any Stop/kill action on a running workflow from the dashboard — read-only, matching this
  entire dashboard's v1 principle.
- Any change to `run_workflow()`'s internal execution logic, node dispatch, or the shape of its
  `log` entries.
- Admin-wide vs. per-admin-owner scoping distinctions beyond what the workflows subsystem
  already has today.

**This completes the full 3-part sub-project 2 scope** (tool-call logs, active agents, live
workflow execution) that sub-project 1's original design deferred.

# "AI Mission Control" Dashboard — Sub-project 1: Aggregation View — Design Spec

## Context

From the user's personal feature roadmap (`New AI Additions.txt`): "AI Mission Control" — a
single dashboard where users can see all running models, monitor active agents, watch workflows
execute in real time, view hardware usage, manage queued tasks, browse memory and knowledge
bases, inspect tool calls and logs, and control connected devices and integrations.

This is a genuinely multi-subsystem idea — 8 distinct areas, decomposed at brainstorming time
into two sub-projects:

- **Sub-project 1 (this spec):** aggregate the 5 areas already backed by real, queryable data
  today — models, hardware usage, the task queue (including its existing tool-call log),
  memory/knowledge bases, and connected integrations.
- **Sub-project 2 (deferred, unscoped):** the 3 areas that need genuinely new backend
  infrastructure before any dashboard could show them — a live active-agent registry (today's
  interactive chat agent loop is ephemeral, nothing persists a "what's running right now" list),
  persisted step-by-step workflow-run state (today `run_workflow()` runs synchronously to
  completion within one call, nothing to poll or stream mid-run), and a tool-call log for
  ordinary interactive chat (today's `TaskRun.steps` JSON log only covers scheduled-task-
  triggered agent runs, not regular chat sessions). Sub-project 2 isn't designed here — it needs
  its own brainstorming pass once sub-project 1 ships, since "what backend do these three
  actually need" is itself an open design question, not a known quantity like sub-project 1's
  building blocks.

## Current State (as of this spec)

- **Panels are modals, universally.** Every existing panel in this app — including Tasks, the
  closest existing analog to a dashboard-style list view — is implemented as a modal registered
  in `static/js/modalManager.js`'s registry (`{modal-id: {rail, sidebar}}`), opened via a rail
  button and/or sidebar tool button, closed via a close button. There is no "full page" concept
  distinct from the modal system anywhere in this codebase.
- **The 5 in-scope data sources already exist and are independently reachable:**
  - `GET /api/models` (`routes/model_routes.py`) — owner-scoped list of registered endpoints and
    their models, with online/offline status per endpoint.
  - `GET /api/hwfit/usage` (`routes/hwfit_routes.py`) — live hardware usage (RAM/GPU/CPU), the
    same data the existing sidebar sparkline widget already renders. No admin gate at the route
    level.
  - `GET /api/tasks` (`routes/task_routes.py`, optional `status` filter) — `ScheduledTask` rows
    (status, `next_run`, `last_run`). Each task's executions are `TaskRun` rows
    (`core/database.py`), which already include a `steps` column — "JSON log of agent tool
    calls" — for that specific run.
  - `GET /api/memory` (`routes/memory/memory_routes.py`) — the memory system's list endpoint
    (just gained persona isolation in the prior sub-project). A `GET /api/memory/timeline` route
    also already exists in the same file, discovered during this spec's research — worth reusing
    if its shape fits this widget's summary view, confirmed at plan-writing time.
  - `GET /api/auth/integrations` (`routes/auth_routes.py`) — the Plugin/Connector Hub's existing
    connector list.
- **No live/streaming infrastructure exists for any of these** — every one of the 5 is a
  request/response snapshot today, not a subscription or a websocket. This spec does not change
  that; each widget below is a fetch-on-open snapshot, refreshed on demand.

## Goal

One new modal, `#mission-control-modal`, giving an at-a-glance read-only summary of models,
hardware, the task queue, memory/knowledge bases, and integrations — without duplicating any
existing panel's detail view or control actions. A user opens it, sees 5 widget cards, and can
click through to any area's real panel to actually do something.

## Architecture

A single new modal following this app's established panel pattern exactly (a rail button, a
sidebar tool button, a close button, registration in `modalManager.js`) — not a new "page"
concept, since none exists anywhere else in this app and inventing one here would be a bigger,
independent architectural change this sub-project doesn't need. Sized larger than a typical
panel (most of the viewport) so its 5 widget cards can lay out in a CSS grid rather than
stacking, since it aggregates more at once than any existing single-purpose modal does.

Each widget is an independent unit: its own fetch call, its own loading/error/empty state, its
own compact summary rendering, and a link/header that opens the corresponding real panel
(`modalManager.js`'s existing `Modals.toggle(...)`/rail-button mechanics) for anything beyond
looking. No widget's failure affects any other widget — this mirrors the "extend, don't touch"
spirit of how the last few sub-projects have integrated with pre-existing panels.

**Access control:** Mission Control adds no gate of its own. Each widget simply calls its area's
existing endpoint, which already enforces whatever access rule that area already has today
(owner-scoped, admin-only, or open) — a user who can't see something in its own panel won't see
it in Mission Control either, and nothing here changes who can see what.

## Data Flow

On modal open, all 5 widgets fetch in parallel (no sequencing dependency between them):

1. **Models** — `GET /api/models`, rendered as: endpoint count, model count, and an
   online/offline indicator per endpoint (reusing the existing `offline` field already in the
   response).
2. **Hardware** — `GET /api/hwfit/usage`, rendered as a compact version of what the sidebar
   sparkline already shows (RAM/GPU/CPU at a glance) — reusing that widget's existing rendering
   logic where practical rather than re-deriving it, confirmed at plan-writing time.
3. **Task queue** — `GET /api/tasks`, rendered as: counts by status (active/paused/completed),
   and the most-recently-run few tasks with an expandable "recent tool calls" detail sourced from
   that task's latest `TaskRun.steps`.
4. **Memory & knowledge bases** — `GET /api/memory`, rendered as: total memory count, and the
   most-recently-added few entries. RAG collection summary included if `GET /api/memory` (or a
   sibling endpoint discovered at plan-writing time) already exposes it; if not, this piece is
   dropped from v1 rather than adding a new backend endpoint for it (see Out of Scope).
5. **Integrations** — `GET /api/auth/integrations`, rendered as: connected count and a
   connected/disconnected indicator per integration.

Every widget's "open full view" affordance calls the same `Modals.toggle(...)` mechanism (or
equivalent rail-button click) the sidebar already uses to open that panel directly — Mission
Control itself never duplicates any of those panels' own rendering or write logic.

## Error Handling

No new fail-open/security logic — nothing in this design is a security boundary, since every
widget reads data the user could already see by opening that area's own panel. Per-widget
failure handling only:

- A widget's fetch failing (network error, the underlying endpoint 500ing, hardware detection
  erroring on an unusual config) renders that one card in an error state with a retry option —
  the other 4 widgets are unaffected.
- An empty result (no tasks, no memories, no integrations configured) renders that widget's
  empty state, not an error.

## Testing

No new backend to test — this design intentionally adds none (see Architecture). Frontend:
source-presence tests matching this app's established style for modal-based panels (confirmed at
plan-writing time by reading a recent example, e.g. the Crew panel's test file) — HTML scaffold
present, the modal registers correctly in `modalManager.js`, each widget's fetch call targets the
correct endpoint, and a `node --check` syntax gate. Manual GUI verification (does it actually
look right, do the "open full view" links actually navigate correctly) is owed by the user, same
as every other frontend-only sub-project this session.

## Out of Scope

- Sub-project 2 (active agents, live workflow execution, chat-level tool-call logs) — deferred,
  unscoped, needs its own brainstorming pass.
- Any control/write action inside Mission Control itself (canceling a task, disconnecting an
  integration, editing a memory) — v1 is read-only; every such action happens in that area's own
  existing panel, reached via Mission Control's links.
- Live/streaming updates for any widget — every widget is a fetch-on-open snapshot with a manual
  refresh, not a websocket or polling loop.
- A RAG-collection summary inside the memory widget, if no existing endpoint already exposes a
  cheap summary of it — rather than adding a new backend endpoint for this one detail, it's
  dropped from v1 and the memory widget covers plain memories only.
- Any change to the 5 existing endpoints this design reads from, or to the panels they already
  power (Tasks, Crew's memory tooling, Connector Hub, the sidebar hardware sparkline, the model
  picker) — this is purely additive.

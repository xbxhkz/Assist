# "AI Mission Control" Dashboard — Sub-project 2a: Chat Tool-Call Log — Design Spec

## Context

Sub-project 1 shipped a 5-widget read-only aggregation dashboard (Models, Hardware, Task Queue,
Memory, Integrations), each reading an existing endpoint with zero new backend. Its design spec
deferred three areas that needed genuinely new infrastructure — active agents, live workflow
execution, and chat-level tool-call logs — to an unscoped "sub-project 2," explicitly noting
"what backend do these three actually need" was itself an open question.

Live-code research at the start of this sub-project's brainstorming found the three deferred
areas are very differently sized, contrary to the original roadmap's assumption that none of
them had any backing infrastructure:

- **Chat-level tool-call logs (this spec)** — turns out the data already exists. Every agent
  turn's tool calls are captured as `tool_events` (`src/agent_loop.py:2991` initializes the list,
  `:4016` appends one entry per tool call) and persisted as part of the assistant `ChatMessage`'s
  `meta_data` JSON blob (`core/session_manager.py:249`). There is no dedicated, queryable table
  for it, but nothing new needs to be *written* — only *read differently*.
- **Active agents** — medium lift, deferred. Three separate in-memory mechanisms
  (`src/agent_runs.py`'s `_RUNS` dict, `routes/chat_routes.py`'s `_active_streams`,
  `src/interactive_gate.py`'s boolean gate) each track partial state but none can enumerate
  "what's running right now" — only point-lookups or a single yes/no.
- **Live workflow execution** — biggest lift, deferred. `run_workflow()`
  (`src/workflows/engine.py:33`) runs a workflow's full node graph to completion synchronously
  within one function call; nothing is persisted mid-run at any of its three call sites
  (agent tool, direct API, scheduled task). No `WorkflowRun` DB model exists at all.

Given the size mismatch, this spec covers only the smallest of the three — chat-level tool-call
logs — as its own mini-sub-project ("2a"). Active agents and live workflow execution remain
deferred, each needing its own brainstorming pass when picked up.

**A related, separate finding from the same research, worth recording but not fixing here:**
`TaskRun.steps` (`core/database.py:664`, "JSON log of agent tool calls") is dead code — no
code anywhere constructs a `TaskRun` with `steps=` or ever assigns `.steps` on one. It isn't
"populated for scheduled tasks only," as sub-project 1's plan assumed when it dropped a
tool-call-log column from the Tasks widget — it's populated nowhere at all today. This spec does
not touch scheduled-task tool-call logging; it only covers ordinary interactive chat, which
already has real (if unindexed) persisted data via `ChatMessage.meta_data`.

## Goal

A user can browse and filter their own past tool calls across all their chat sessions — what
tool ran, when, in which session, with what result — without adding any new database table or
changing how tool calls are recorded today. A new Mission Control widget summarizes recent
activity and links to a new dedicated panel for the full browsable/filterable view.

## Architecture

**Query layer — no schema change.** A new function, `list_tool_calls(db, owner, session_id=None,
tool_name=None, since=None, until=None, limit=50, offset=0)`, joins `ChatMessage` to `Session` to
scope by `owner` (matching the owner-scoping every other Mission Control widget's endpoint
already does), filters to assistant messages with non-null `meta_data`, and walks them
**newest-first in batches** rather than loading full history: fetch a batch ordered by
`ChatMessage.timestamp.desc()`, parse each message's `meta_data` JSON, flatten any `tool_events`
entries into normalized records, and stop once enough records exist to satisfy `limit` + `offset`
— extending to another batch if a page runs dry (sparse tool usage deep in old history). This
bounds the common case (recent tool calls are in recent messages) without indexing the JSON blob.

Each flattened record: `{session_id, session_name, message_id, timestamp, round, tool, command,
output, exit_code}` — mirroring the raw `tool_event` shape (`round`, `tool`, `command`, `output`,
`exit_code` from `src/agent_loop.py:3993-3998`) plus session context for display. `command` is
already a human-readable display string (not raw args) and `output` is already-formatted result
text — this is the same data the chat UI itself renders for a tool-call card, not a new
representation. There is no per-tool-call timestamp; entries within one turn share the
containing message's timestamp and are ordered by `round`.

**Route.** `GET /api/tool-calls` (owner-gated like any other chat/session read, no admin
requirement — matches the "owner-scoped, own calls only" visibility decision below), accepting
`session_id`, `tool_name`, `since`, `until`, `limit`, `offset` query params, returning
`{"tool_calls": [...], "has_more": bool}`.

**Panel.** A new "Tool Call History" modal, registered via `Modals.register()` like every other
panel in this app. Lists entries newest-first: session name, tool name, round, a truncated
command/output preview, exit code if present. A session filter dropdown and a tool-name filter;
"load more" pagination using `has_more`/`offset`. Clicking a session name jumps to that session
in chat, reusing the existing session-open mechanism the sidebar chat list already uses.

**Mission Control widget.** A 6th `.mission-control-card`, "Tool Calls," following the exact
widget contract sub-project 1 established (`id="mc-card-tool-calls"`, `data-widget="tool-calls"`,
body `id="mc-body-tool-calls"`, refresh button, `id="mc-open-tool-calls"` link) — count of recent
calls, 2-3 most-recent previews (tool name + truncated command), link opens the new panel. No
changes to the other 5 widgets or their wiring.

**Access control:** owner-scoped, no new gate beyond what every chat/session read already
enforces — a user only ever sees tool calls from their own sessions, exactly matching the
"owner-scoped, own calls only" decision (no admin-only or admin-sees-all mode in this v1).

## Data Flow

1. Panel or widget opens → `GET /api/tool-calls?limit=N` (widget: small N for a preview; panel:
   page-sized N with pagination).
2. `list_tool_calls` batch-scans the caller's own `ChatMessage` rows newest-first, joined through
   `Session.owner`, parses `meta_data`, flattens `tool_events`.
3. Response returns newest-first; the panel's session/tool-name filters narrow the query
   (`session_id` is a real, indexed column; `tool_name` is not indexed inside the JSON blob, so
   that filter applies after parsing, same as the base scan — acceptable at this app's
   local/personal scale, same trade-off sub-project 1's design accepted for its own endpoints).

## Error Handling

A message's `meta_data` that fails to parse as JSON (corrupt or legacy row) is skipped, not
fatal — the scan continues past it. A message with valid `meta_data` but no `tool_events` key is
skipped (an ordinary turn with no tool use) — this is the common case, not an error. Route
failures render the standard per-widget error state in Mission Control (matching every other
widget's independent-failure handling from sub-project 1) and a normal error banner in the
dedicated panel. No new access-control gate beyond the existing owner check every chat/session
query already applies — mirrors sub-project 1's "Mission Control adds no gate of its own"
principle, extended here to the new panel as well since the panel itself reads only owner-scoped
data.

## Testing

**Backend:** unit tests for `list_tool_calls` covering: pagination that spans more than one
internal batch, session-id filtering, tool-name filtering, corrupt-JSON rows being skipped
without raising, messages with no `tool_events` being skipped, and owner isolation (a second
user's tool calls never appear in the first user's results) — the same owner-scoping bar every
other Mission Control widget's endpoint already meets. Route tests for `GET /api/tool-calls`
covering the same filters plus the owner-gate itself (an unauthenticated/wrong-owner request
never sees another user's data).

**Frontend:** source-presence tests for the new widget matching the established
`test_mission_control_ui.py` style (card markup present, correctly wired into
`refreshWidget()`/`loadAllWidgets()`, `esc()` applied to tool/command/output text before
`innerHTML` — command and output are user/agent-generated text, the same XSS-hygiene bar the
Memory widget met in sub-project 1). A new equivalent test file for the panel (modal
registration, filter controls present, pagination wiring).

Manual GUI verification (does the panel look right, does filtering/pagination actually work, does
clicking a session name correctly jump to that session) is owed by the user, same as every other
frontend sub-project this session.

## Out of Scope

- Active agents and live workflow execution — remain deferred, each needs its own brainstorming
  pass.
- Any dedicated DB table or index for tool calls — this spec deliberately queries existing data;
  a future live-monitoring feature (explicitly deferred, see sub-project 1's spec) may revisit
  this trade-off, but nothing here should be built anticipating that.
- Admin-wide visibility across all users' tool calls — v1 is owner-scoped only, per the
  visibility decision above.
- Populating `TaskRun.steps` for scheduled-task tool-call logging — a separate, real gap noted
  during this spec's research, but out of scope for a spec about interactive chat.
- Any write/control action inside the new panel (re-running a tool call, deleting history) — v1
  is read-only browsing, matching sub-project 1's overall read-only principle for this dashboard
  initiative.

# "AI Mission Control" Dashboard — Sub-project 2b: Active Agents — Design Spec

## Context

Sub-project 1 shipped a 5-widget read-only aggregation dashboard. Its design deferred three
areas needing genuinely new infrastructure to an unscoped "sub-project 2" — active agents, live
workflow execution, and chat-level tool-call logs. Sub-project 2a (tool-call logs) shipped, and
found the three deferred areas were very differently sized once the live code was checked, not
uniformly "zero infrastructure" as the original roadmap assumed.

This spec covers the next of the three — **active agents** — as its own mini-sub-project ("2b").
Live-code research (carried over from sub-project 2a's decomposition pass, re-confirmed here)
found three separate in-memory mechanisms already track partial per-session execution state, but
none can enumerate "what's running right now" across sessions:

- **`src/agent_runs.py`** — a detached-run manager: `_RUNS: Dict[str, _Run]` keyed by
  `session_id`, each with a `status` field (`"running" | "done" | "error" | "stopped"`). Exposes
  `is_active(session_id)` and `get_status(session_id)` — point lookups only, for a session id
  you already know. No enumeration function exists. In-memory only, does not survive a server
  restart (by design — a run mid-flight when the process restarts is gone regardless).
  Confirmed via a fresh read of the file: `_Run` (`__slots__ = ("buffer", "subscribers",
  "status", "task", "evict_task")`) carries no owner or session-name field today.
- **`routes/chat_routes.py`**'s `_active_streams` dict — a separate, narrower "partial-save
  safety net" cache, semantically different from a run registry (used to recover a
  still-in-progress response's partial text, not to answer "is this running").
- **`src/interactive_gate.py`**'s `_ACTIVE_REQUESTS` counter — a process-global boolean gate
  ("is anything active," for scheduling decisions), not an enumerable registry.

`agent_runs.py`'s `_RUNS` is the natural consolidation point: it already has clean lifecycle
hooks (`start()`/`stop()`/completion) and a `status` field per session. It just lacks (1) an
enumeration function and (2) enough context (owner, session name) to display a useful list — both
addressed below without touching its existing, delicate write path (cancellation races, eviction
timers already correct and reviewed).

**Scope decisions** (resolved during brainstorming, not left open):
- **Interactive chat sessions only** — scheduled/background task executions are out of scope;
  they already have status/history in the Tasks widget (sub-project 1), and the original roadmap
  gap this sub-project closes was specifically about the interactive chat agent loop.
- **Owner-scoped only** — matches every Mission Control widget shipped so far, including
  sub-project 2a's Tool Call Log. No admin-wide "everyone's active agents" mode in this v1.
- **Read-only, no Stop action on the dashboard** — matches this dashboard's v1 principle
  throughout (read-only, link out for actions). Clicking an active agent's session name jumps to
  that session in chat, where the existing Stop button already lives.

## Goal

A user opens Mission Control and sees, at a glance, which of their own chat sessions are
currently mid-run (an agent loop actively working), with a way to jump straight to any of them.

## Architecture

**`src/agent_runs.py` gains one new read-only function**, `list_active() -> List[str]`, returning
the session ids currently `status == "running"` in `_RUNS`. This is the ONLY change to this file
— no new fields on `_Run`, no changes to `start()`/`stop()`/`_drain()`/eviction. Deliberately
mirrors sub-project 2a's own discipline (add a read, never touch the writer) — `agent_runs.py` is
delicate async code (cancellation races, grace-period eviction) that no task in this sub-project
should risk destabilizing for the sake of a dashboard widget.

**A new thin join function** (new small module, `src/active_agents.py`, since neither
`agent_runs.py` nor `routes/mission_control`-adjacent code is the right home for something that
reads from BOTH `agent_runs` and `SessionManager`) takes the running session ids from
`list_active()`, looks each one up in `SessionManager.sessions` (an in-memory
`Dict[str, Session]` — confirmed via `core/session_manager.py:74` — no DB query needed), filters
to sessions whose `.owner` matches the caller, and returns `[{"session_id": str, "session_name":
str}, ...]`. A session id present in `_RUNS` but no longer in `SessionManager.sessions` (evicted
or deleted between the run starting and this being read) is silently skipped, not an error.

**A new route**, `GET /api/agent-runs/active`, owner-gated like every other Mission Control
endpoint, wraps that join function and returns `{"active": [...]}`.

**A 7th Mission Control widget**, "Active Agents" — no new dedicated panel this time (unlike
sub-project 2a's Tool Call Log), since this is a short live list, not something needing deep
browsing, filtering, or history. Follows the same card contract as the other 6: a
`.mission-control-card`, a `load<Name>Widget()` function, wired into `refreshWidget()` and
`loadAllWidgets()`. Its entries are directly clickable — clicking a session name jumps to that
session in chat via `window.sessionModule.selectSession(sessionId)`, the exact mechanism
sub-project 2a's Tool Call History panel already established — no separate "open full view" link
needed since there's no separate view to open.

**Access control:** owner-scoped, no new gate — mirrors every other widget. A user only ever sees
their own currently-running sessions.

## Data Flow

Widget open or manual refresh → `GET /api/agent-runs/active` → `list_active()` returns running
session ids → the join function filters/resolves them against the caller's own
`SessionManager.sessions` entries → returns owner-scoped `{session_id, session_name}` pairs →
widget renders a short list (or "No active agents right now").

No polling or streaming — this is a fetch-on-open-plus-manual-refresh snapshot, identical to
every other widget in this dashboard. A user watching a live response stream already sees it
directly in the chat UI; Mission Control's job is a glanceable summary, not a live feed.

## Error Handling

A session id present in `agent_runs._RUNS` but absent from `SessionManager.sessions` is skipped,
not an error (covers the eviction/deletion race described above). Route failure renders the
standard per-widget error state, independent of the other 6 widgets. No new access-control gate
beyond the existing owner-scoping every widget already has.

## Testing

**Backend:** unit tests for `list_active()` (returns only `"running"` sessions, not
`"done"`/`"error"`/`"stopped"`) and the join function (owner filtering — another user's active
session never appears; a `_RUNS` entry with no matching `SessionManager` session is skipped, not
raised). Route tests for `GET /api/agent-runs/active` (owner-scoping, matching the established
`_route()`-based direct-call test style used by sub-project 2a's route tests).

**Frontend:** source-presence tests for the 7th widget matching sub-project 2a's own extended
seam-guard pattern (the widget must appear in `loadAllWidgets()`'s actual body, not just exist as
a named function — sub-project 2a's final review found and fixed exactly this class of gap for
its own widget, so this sub-project bakes the stronger test in from the start rather than
repeating the miss). `esc()` applied to `session_name` before `innerHTML` (user-controlled text,
same XSS-hygiene bar every widget meets).

Manual GUI verification (does the list actually update when a chat session starts/finishes, does
clicking a session jump correctly) is owed by the user, same as every prior frontend sub-project
this session — this is the one area of this feature that's hardest to unit-test meaningfully,
since it depends on real concurrent request timing.

## Out of Scope

- Scheduled/background task executions (already covered by the Tasks widget's status/counts).
- Admin-wide visibility across all users' active agents.
- A Stop/kill action from the dashboard itself.
- Any new field on `agent_runs.py`'s `_Run` class, or any change to its write path.
- Live/streaming updates to the widget — fetch-on-open-plus-manual-refresh only, matching every
  other Mission Control widget.
- The remaining deferred sub-project-2 area (live workflow execution) — still unscoped, needs its
  own brainstorming pass when picked up.

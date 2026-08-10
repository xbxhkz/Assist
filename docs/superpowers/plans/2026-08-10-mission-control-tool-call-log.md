# Mission Control Sub-project 2a: Chat Tool-Call Log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user browse and filter their own past tool calls across all their chat sessions — a new query function over existing data, a new read-only route, a new dedicated panel, and a 6th Mission Control widget.

**Architecture:** Tool-call data already exists — every agent turn's `tool_events` is persisted inside the assistant `ChatMessage.meta_data` JSON blob (`src/agent_loop.py:3993-4016`, `core/session_manager.py:249`). No new table. A new query function (`list_tool_calls`) batch-scans the caller's own messages newest-first and flattens `tool_events` into normalized records; a thin route wraps it; a new panel renders and paginates the results; Mission Control's existing 5-widget dashboard gains a 6th card linking to that panel.

**Tech Stack:** FastAPI route (Python), SQLAlchemy query over the existing `ChatMessage`/`Session` tables, vanilla JS ES module (matches `static/js/missionControl.js`), pytest.

## Global Constraints

- Owner-scoped, own calls only — no admin-only or admin-sees-all mode in this v1 (per the spec's Visibility decision).
- No new database table or migration — query `ChatMessage.meta_data` at read time (per the spec's Storage decision).
- No changes to how `tool_events` is written today (`src/agent_loop.py`) — this plan only reads existing data differently.
- Every piece of tool-call-derived text (`command`, `output`, `tool`, `session_name`) that reaches `innerHTML` must go through the `esc()` helper before interpolation — same XSS-hygiene bar the Memory widget met in sub-project 1 (`docs/superpowers/plans/2026-08-07-mission-control-dashboard.md` Task 4).
- Read-only — no write/control action anywhere in this feature (matches sub-project 1's overall read-only principle).
- A caller-supplied `session_id` filter must never leak another owner's session — the owner filter applies unconditionally, `session_id` only narrows within it (this is a security property, not just a filter — test it explicitly).
- New Mission Control widget follows the exact card contract sub-project 1 established: `.mission-control-card` with `id="mc-card-<name>"`, `data-widget="<name>"`, a refresh button, body `id="mc-body-<name>"`, `id="mc-open-<name>"` link; `load<Name>Widget()` using the existing `$`/`esc`/`api`/`setCardBody`/`setCardError` helpers already defined in `static/js/missionControl.js`; one line each in `refreshWidget()` and `loadAllWidgets()`.
- **Lesson from sub-project 1's final review, applied here from the start**: that review found `tests/test_mission_control_ui.py`'s per-widget tests only proved a loader function's *name* appears in the file, not that it's actually *called* from `loadAllWidgets()` — a real gap fixed after the fact (commit `1d796f79`) by adding `test_mission_control_loadAllWidgets_wires_all_loaders` (extracts `loadAllWidgets()`'s body via regex, asserts all loader calls appear inside it) and `test_mission_control_link_targets_exist_in_html`. Task 4 below extends BOTH of those existing regression-guard tests to cover the 6th widget, not just adds a 7th standalone presence test — this closes the same seam pre-emptively instead of leaving it for a future final review to catch again.

---

### Task 1: `list_tool_calls` query function

**Files:**
- Create: `routes/tool_calls_routes.py` (function only this task — the route handler is added in Task 2)
- Test: `tests/test_tool_calls_query.py`

**Interfaces:**
- Produces: `list_tool_calls(db, owner: Optional[str], session_id: Optional[str] = None, tool_name: Optional[str] = None, since: Optional[datetime] = None, until: Optional[datetime] = None, limit: int = 50, offset: int = 0) -> Tuple[List[Dict], bool]` — returns `(records, has_more)`. Each record: `{"session_id": str, "session_name": str, "message_id": str, "timestamp": Optional[str], "round": Any, "tool": Any, "command": Any, "output": Any, "exit_code": Any}`.
- Produces (module-level, used by tests via monkeypatch): `_BATCH_SIZE` (int constant, the internal DB-fetch batch size).
- Consumes: `core.database.ChatMessage`, `core.database.Session`, `core.database.SessionLocal` (existing).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tool_calls_query.py`:

```python
"""list_tool_calls flattens persisted tool_events out of chat history -- no
dedicated table, see
docs/superpowers/specs/2026-08-10-mission-control-tool-call-log-design.md.
"""
import json
import uuid
from datetime import datetime, timedelta

import routes.tool_calls_routes as tcr
from core.database import ChatMessage as DBChatMessage, Session as DBSession, SessionLocal


def _unique_owner():
    return "owner-" + uuid.uuid4().hex[:10]


def _make_session(owner):
    db = SessionLocal()
    try:
        sid = str(uuid.uuid4())
        db.add(DBSession(id=sid, name="s", endpoint_url="http://x", model="m", owner=owner))
        db.commit()
        return sid
    finally:
        db.close()


def _add_message(session_id, meta_data, ts, role="assistant"):
    db = SessionLocal()
    try:
        mid = str(uuid.uuid4())
        db.add(DBChatMessage(
            id=mid, session_id=session_id, role=role, content="...",
            meta_data=meta_data, timestamp=ts,
        ))
        db.commit()
        return mid
    finally:
        db.close()


def _add_tool_events(session_id, tool_events, ts):
    return _add_message(session_id, json.dumps({"tool_events": tool_events}), ts)


def test_flattens_tool_events_newest_first():
    owner = _unique_owner()
    sid = _make_session(owner)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    _add_tool_events(sid, [{"round": 1, "tool": "run_command", "command": "ls", "output": "a b c", "exit_code": 0}], t0)
    _add_tool_events(sid, [{"round": 1, "tool": "web_search", "command": "q", "output": "results", "exit_code": None}], t0 + timedelta(minutes=1))

    db = SessionLocal()
    try:
        records, has_more = tcr.list_tool_calls(db, owner=owner)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["web_search", "run_command"]
    assert has_more is False


def test_pagination_spans_multiple_internal_batches(monkeypatch):
    monkeypatch.setattr(tcr, "_BATCH_SIZE", 2)
    owner = _unique_owner()
    sid = _make_session(owner)
    base = datetime(2026, 1, 1, 12, 0, 0)
    for i in range(5):
        _add_tool_events(sid, [{"round": 1, "tool": "tool%d" % i, "command": "c", "output": "o", "exit_code": 0}],
                          base + timedelta(minutes=i))

    db = SessionLocal()
    try:
        records, has_more = tcr.list_tool_calls(db, owner=owner, limit=3)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["tool4", "tool3", "tool2"]
    assert has_more is True


def test_pagination_offset_gets_next_page(monkeypatch):
    monkeypatch.setattr(tcr, "_BATCH_SIZE", 2)
    owner = _unique_owner()
    sid = _make_session(owner)
    base = datetime(2026, 1, 1, 12, 0, 0)
    for i in range(5):
        _add_tool_events(sid, [{"round": 1, "tool": "tool%d" % i, "command": "c", "output": "o", "exit_code": 0}],
                          base + timedelta(minutes=i))

    db = SessionLocal()
    try:
        records, has_more = tcr.list_tool_calls(db, owner=owner, limit=3, offset=3)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["tool1", "tool0"]
    assert has_more is False


def test_session_id_filter():
    owner = _unique_owner()
    sid_a = _make_session(owner)
    sid_b = _make_session(owner)
    _add_tool_events(sid_a, [{"round": 1, "tool": "toolA", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 1))
    _add_tool_events(sid_b, [{"round": 1, "tool": "toolB", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 2))

    db = SessionLocal()
    try:
        records, _ = tcr.list_tool_calls(db, owner=owner, session_id=sid_a)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["toolA"]


def test_session_id_filter_cannot_leak_another_owners_session():
    owner_a = _unique_owner()
    owner_b = _unique_owner()
    sid_b = _make_session(owner_b)
    _add_tool_events(sid_b, [{"round": 1, "tool": "bob_tool", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 1))

    db = SessionLocal()
    try:
        records, _ = tcr.list_tool_calls(db, owner=owner_a, session_id=sid_b)
    finally:
        db.close()

    assert records == []


def test_tool_name_filter():
    owner = _unique_owner()
    sid = _make_session(owner)
    _add_tool_events(sid, [
        {"round": 1, "tool": "run_command", "command": "c1", "output": "o1", "exit_code": 0},
        {"round": 2, "tool": "web_search", "command": "c2", "output": "o2", "exit_code": None},
    ], datetime(2026, 1, 1))

    db = SessionLocal()
    try:
        records, _ = tcr.list_tool_calls(db, owner=owner, tool_name="web_search")
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["web_search"]


def test_corrupt_json_row_is_skipped_not_raised():
    owner = _unique_owner()
    sid = _make_session(owner)
    _add_message(sid, "{not valid json", datetime(2026, 1, 1))
    _add_tool_events(sid, [{"round": 1, "tool": "ok_tool", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 2))

    db = SessionLocal()
    try:
        records, _ = tcr.list_tool_calls(db, owner=owner)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["ok_tool"]


def test_message_without_tool_events_key_is_skipped():
    owner = _unique_owner()
    sid = _make_session(owner)
    _add_message(sid, json.dumps({"model": "some-model"}), datetime(2026, 1, 1))
    _add_tool_events(sid, [{"round": 1, "tool": "ok_tool", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 2))

    db = SessionLocal()
    try:
        records, _ = tcr.list_tool_calls(db, owner=owner)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["ok_tool"]


def test_owner_isolation():
    owner_a = _unique_owner()
    owner_b = _unique_owner()
    sid_a = _make_session(owner_a)
    sid_b = _make_session(owner_b)
    _add_tool_events(sid_a, [{"round": 1, "tool": "alice_tool", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 1))
    _add_tool_events(sid_b, [{"round": 1, "tool": "bob_tool", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 2))

    db = SessionLocal()
    try:
        records, _ = tcr.list_tool_calls(db, owner=owner_a)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["alice_tool"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tool_calls_query.py -v --import-mode=importlib`
Expected: FAIL — `ModuleNotFoundError` or `AttributeError: module 'routes.tool_calls_routes' has no attribute 'list_tool_calls'` (the module doesn't exist yet).

- [ ] **Step 3: Implement `list_tool_calls`**

Create `routes/tool_calls_routes.py`:

```python
"""Chat-level tool-call log (Mission Control sub-project 2a).

Tool-call data already exists -- every agent turn's tool_events is persisted
inside the assistant ChatMessage.meta_data JSON blob (src/agent_loop.py).
This module queries that existing data at read time; it does not add a new
table. See docs/superpowers/specs/2026-08-10-mission-control-tool-call-log-design.md.
"""
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from core.database import ChatMessage as DBChatMessage, Session as DBSession, SessionLocal

_BATCH_SIZE = 200


def list_tool_calls(
    db,
    owner: Optional[str],
    session_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[Dict], bool]:
    query = (
        db.query(DBChatMessage, DBSession.name)
        .join(DBSession, DBChatMessage.session_id == DBSession.id)
        .filter(
            DBChatMessage.role == "assistant",
            DBChatMessage.meta_data.isnot(None),
        )
    )
    if owner is None:
        query = query.filter(DBSession.owner.is_(None))
    else:
        query = query.filter(DBSession.owner == owner)
    if session_id:
        query = query.filter(DBChatMessage.session_id == session_id)
    if since is not None:
        query = query.filter(DBChatMessage.timestamp >= since)
    if until is not None:
        query = query.filter(DBChatMessage.timestamp <= until)
    query = query.order_by(DBChatMessage.timestamp.desc())

    records: List[Dict] = []
    has_more = False
    seen = 0
    batch_offset = 0

    while True:
        batch = query.offset(batch_offset).limit(_BATCH_SIZE).all()
        if not batch:
            break
        batch_offset += len(batch)

        for message, session_name in batch:
            try:
                meta = json.loads(message.meta_data)
            except (ValueError, TypeError):
                continue
            if not isinstance(meta, dict):
                continue
            events = meta.get("tool_events")
            if not isinstance(events, list):
                continue
            for event in events:
                if not isinstance(event, dict):
                    continue
                if tool_name and event.get("tool") != tool_name:
                    continue
                if seen < offset:
                    seen += 1
                    continue
                if len(records) >= limit:
                    has_more = True
                    break
                records.append({
                    "session_id": message.session_id,
                    "session_name": session_name,
                    "message_id": message.id,
                    "timestamp": (message.timestamp.isoformat() + "Z") if message.timestamp else None,
                    "round": event.get("round"),
                    "tool": event.get("tool"),
                    "command": event.get("command"),
                    "output": event.get("output"),
                    "exit_code": event.get("exit_code"),
                })
            if has_more:
                break
        if has_more or len(batch) < _BATCH_SIZE:
            break

    return records, has_more
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tool_calls_query.py -v --import-mode=importlib`
Expected: PASS (all 9 tests).

- [ ] **Step 5: Commit**

```bash
git add routes/tool_calls_routes.py tests/test_tool_calls_query.py
git commit -m "feat(tool-calls): list_tool_calls query over existing chat history

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `GET /api/tool-calls` route + app.py wiring

**Files:**
- Modify: `routes/tool_calls_routes.py` (add `setup_tool_calls_routes()`)
- Modify: `app.py` (mount the new router)
- Test: `tests/test_tool_calls_routes.py`

**Interfaces:**
- Consumes: `list_tool_calls(db, owner, session_id=None, tool_name=None, since=None, until=None, limit=50, offset=0)` (Task 1).
- Produces: `setup_tool_calls_routes() -> APIRouter`, mounted at `GET /api/tool-calls`. Response: `{"tool_calls": [...], "has_more": bool}`. Query params: `session_id`, `tool_name`, `since`, `until` (ISO 8601 strings; invalid format → 400), `limit` (int, clamped to `[1, 200]`), `offset` (int, clamped to `>= 0`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tool_calls_routes.py`:

```python
"""GET /api/tool-calls must scope to the caller and validate its date filters.

Mirrors the direct-endpoint-call test style used by
tests/test_memory_routes_session_owner.py -- no HTTP layer needed, the route
function is called directly with a fake request.
"""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import routes.tool_calls_routes as tcr


def _route(router, path, method):
    for r in router.routes:
        if r.path == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(path)


def _request(user):
    return SimpleNamespace(state=SimpleNamespace(current_user=user))


def test_owner_scopes_to_caller(monkeypatch):
    captured = {}

    def fake_list_tool_calls(db, owner, **kwargs):
        captured["owner"] = owner
        return [], False

    monkeypatch.setattr(tcr, "list_tool_calls", fake_list_tool_calls)
    router = tcr.setup_tool_calls_routes()
    handler = _route(router, "/api/tool-calls", "GET")

    out = asyncio.run(handler(request=_request("alice")))

    assert captured["owner"] == "alice"
    assert out == {"tool_calls": [], "has_more": False}


def test_invalid_since_returns_400(monkeypatch):
    monkeypatch.setattr(tcr, "list_tool_calls", lambda db, owner, **kw: ([], False))
    router = tcr.setup_tool_calls_routes()
    handler = _route(router, "/api/tool-calls", "GET")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(handler(request=_request("alice"), since="not-a-date"))

    assert exc.value.status_code == 400


def test_invalid_until_returns_400(monkeypatch):
    monkeypatch.setattr(tcr, "list_tool_calls", lambda db, owner, **kw: ([], False))
    router = tcr.setup_tool_calls_routes()
    handler = _route(router, "/api/tool-calls", "GET")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(handler(request=_request("alice"), until="not-a-date"))

    assert exc.value.status_code == 400


def test_limit_and_offset_are_clamped(monkeypatch):
    captured = {}

    def fake_list_tool_calls(db, owner, **kwargs):
        captured.update(kwargs)
        return [], False

    monkeypatch.setattr(tcr, "list_tool_calls", fake_list_tool_calls)
    router = tcr.setup_tool_calls_routes()
    handler = _route(router, "/api/tool-calls", "GET")

    asyncio.run(handler(request=_request("alice"), limit=99999, offset=-5))

    assert captured["limit"] == 200
    assert captured["offset"] == 0


def test_filters_are_forwarded(monkeypatch):
    captured = {}

    def fake_list_tool_calls(db, owner, **kwargs):
        captured.update(kwargs)
        return [], False

    monkeypatch.setattr(tcr, "list_tool_calls", fake_list_tool_calls)
    router = tcr.setup_tool_calls_routes()
    handler = _route(router, "/api/tool-calls", "GET")

    asyncio.run(handler(request=_request("alice"), session_id="s1", tool_name="web_search"))

    assert captured["session_id"] == "s1"
    assert captured["tool_name"] == "web_search"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tool_calls_routes.py -v --import-mode=importlib`
Expected: FAIL — `AttributeError: module 'routes.tool_calls_routes' has no attribute 'setup_tool_calls_routes'`.

- [ ] **Step 3: Add the route handler**

Append to `routes/tool_calls_routes.py` (after `list_tool_calls`, keep the existing imports and add these):

```python
from fastapi import APIRouter, HTTPException, Request

from src.auth_helpers import get_current_user


def setup_tool_calls_routes() -> APIRouter:
    router = APIRouter(prefix="/api/tool-calls", tags=["tool-calls"])

    @router.get("")
    async def get_tool_calls(
        request: Request,
        session_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        user = get_current_user(request)

        since_dt = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid 'since' -- expected ISO 8601")
        until_dt = None
        if until:
            try:
                until_dt = datetime.fromisoformat(until)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid 'until' -- expected ISO 8601")

        safe_limit = max(1, min(limit, 200))
        safe_offset = max(0, offset)

        db = SessionLocal()
        try:
            records, has_more = list_tool_calls(
                db, owner=user, session_id=session_id, tool_name=tool_name,
                since=since_dt, until=until_dt, limit=safe_limit, offset=safe_offset,
            )
        finally:
            db.close()
        return {"tool_calls": records, "has_more": has_more}

    return router
```

(`Optional`, `datetime`, and `SessionLocal` are already imported at the top of the file from Task 1 — do not duplicate those imports.)

- [ ] **Step 4: Wire the router into `app.py`**

Modify `app.py`. Find this existing block (it mounts the Hardware Optimizer's route file the same way — no prefix collision, no shared router file, exactly the pattern this new route follows):

```python
# Hardware model fitting (cookbook "What Fits?" tab)
from routes.hwfit_routes import setup_hwfit_routes
app.include_router(setup_hwfit_routes())
```

Immediately after it, insert:

```python

# Chat tool-call history (Mission Control sub-project 2a)
from routes.tool_calls_routes import setup_tool_calls_routes
app.include_router(setup_tool_calls_routes())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_tool_calls_routes.py -v --import-mode=importlib`
Expected: PASS (all 5 tests).

Run: `python -c "import app"` (from the repo root) to confirm `app.py` still imports cleanly with the new router mounted.
Expected: no output, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add routes/tool_calls_routes.py app.py tests/test_tool_calls_routes.py
git commit -m "feat(tool-calls): GET /api/tool-calls route

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Tool Call History panel

**Files:**
- Modify: `static/index.html` (modal markup, rail button, sidebar button, script tag)
- Modify: `static/style.css` (panel styling)
- Create: `static/js/toolCallLog.js`
- Test: `tests/test_tool_call_log_ui.py`

**Interfaces:**
- Consumes: `GET /api/tool-calls` (Task 2), `Modals.register(id, opts)` (`static/js/modalManager.js`, existing), `window.sessionModule.selectSession(sessionId)` (`static/js/sessions.js:1783`, existing).
- Produces: modal `#tool-call-log-modal`; `id="rail-tool-calls"` (icon rail button), `id="tool-tool-calls-btn"` (sidebar button) — these two ids are what Task 4's Mission Control widget links to.

**A resolved scope note (not a hedge — decided here, not left open):** the spec describes "a session filter dropdown." Building a real `<select>` would require a new session-enumeration mechanism this plan doesn't otherwise need. Instead: a plain text input for filtering by exact session id (matching the API's exact-match `session_id` param), plus a per-entry "filter" quick-link that fills that input from a call the user is already looking at — this delivers the same filtering capability without inventing a new endpoint. The spec's other explicit requirement — clicking a session name jumps to that session in chat — is unchanged and lives on a separate control from the filter quick-link, so the two behaviors don't collide.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tool_call_log_ui.py`:

```python
"""Source-presence tests for the Tool Call History panel (Mission Control
sub-project 2a) -- mirrors tests/test_mission_control_ui.py's established
style: HTML scaffold present, modal registers correctly, fetch call targets
the right endpoint, esc() applied to tool-call text, node --check syntax gate.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_index_has_tool_call_log_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in (
        'id="tool-call-log-modal"',
        'id="rail-tool-calls"',
        'id="tool-tool-calls-btn"',
        'id="tool-call-log-list"',
        'id="tool-call-log-tool-filter"',
        'id="tool-call-log-session-filter-input"',
        'id="tool-call-log-apply-filter"',
        'id="tool-call-log-clear-filter"',
        'id="tool-call-log-more"',
        '/static/js/toolCallLog.js',
    ):
        assert el in html, "missing %s" % el


def test_tool_call_log_js_registers_with_modal_manager():
    src = (ROOT / "static" / "js" / "toolCallLog.js").read_text(encoding="utf-8")
    assert "Modals.register" in src
    assert "'tool-call-log-modal'" in src or '"tool-call-log-modal"' in src


def test_tool_call_log_js_fetches_tool_calls():
    src = (ROOT / "static" / "js" / "toolCallLog.js").read_text(encoding="utf-8")
    assert "/api/tool-calls" in src


def test_tool_call_log_js_escapes_command_and_output():
    src = (ROOT / "static" / "js" / "toolCallLog.js").read_text(encoding="utf-8")
    assert "esc(c.command" in src
    assert "esc((c.output" in src


def test_tool_call_log_js_uses_select_session_for_jump():
    src = (ROOT / "static" / "js" / "toolCallLog.js").read_text(encoding="utf-8")
    assert "sessionModule" in src
    assert "selectSession" in src


def test_tool_call_log_js_syntax():
    result = subprocess.run(
        ["node", "--check", str(ROOT / "static" / "js" / "toolCallLog.js")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tool_call_log_ui.py -v --import-mode=importlib`
Expected: FAIL — `static/js/toolCallLog.js` does not exist, `'id="tool-call-log-modal"'` not in `index.html`.

- [ ] **Step 3: Add the modal, rail button, and sidebar button to `static/index.html`**

Find this block (the Mission Control modal, immediately preceding text `<div id="mission-control-modal"`):

```html
  <div id="mission-control-modal" class="modal hidden">
```

Immediately **before** it, insert:

```html
  <div id="tool-call-log-modal" class="modal hidden">
    <div class="modal-content tool-call-log-content" role="dialog" aria-label="Tool Call History">
      <div class="modal-header">
        <h4>Tool Call History</h4>
        <button class="close-btn" id="tool-call-log-close" aria-label="Close">&#x2716;</button>
      </div>
      <div class="tool-call-log-filters">
        <input type="text" id="tool-call-log-tool-filter" placeholder="Tool name...">
        <input type="text" id="tool-call-log-session-filter-input" placeholder="Session id...">
        <button class="btn" id="tool-call-log-apply-filter">Filter</button>
        <button class="btn" id="tool-call-log-clear-filter">Clear</button>
      </div>
      <div id="tool-call-log-list" class="tool-call-log-list">Loading…</div>
      <button class="btn" id="tool-call-log-more" style="display:none">Load more</button>
    </div>
  </div>

```

Find this line (the Mission Control rail button):

```html
    <button class="icon-rail-btn" id="rail-mission-control" title="Mission Control"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg></button>
```

Immediately after it, insert:

```html
    <button class="icon-rail-btn" id="rail-tool-calls" title="Tool Call History"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg></button>
```

Find this block (the Mission Control sidebar entry, ending right before the "Admin-only tools" comment):

```html
        <div class="list-item" id="tool-mission-control-btn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.5;"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
          <span class="grow">Mission Control</span>
        </div>

        <!-- Admin-only tools: hidden by default, revealed by their module's
             isAdmin() check (same gate as their icon-rail buttons). -->
```

Insert a new sidebar entry between the Mission Control `</div>` and the comment:

```html
        <div class="list-item" id="tool-mission-control-btn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.5;"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
          <span class="grow">Mission Control</span>
        </div>

        <div class="list-item" id="tool-tool-calls-btn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.5;"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
          <span class="grow">Tool Call History</span>
        </div>

        <!-- Admin-only tools: hidden by default, revealed by their module's
             isAdmin() check (same gate as their icon-rail buttons). -->
```

Find this line:

```html
<script type="module" src="/static/js/missionControl.js"></script>
```

Immediately before it, insert:

```html
<script type="module" src="/static/js/toolCallLog.js"></script>
```

- [ ] **Step 4: Add panel styling to `static/style.css`**

Find this block (Mission Control's own styling, near the end of the stylesheet):

```css
.mission-control-content {
  width: min(1100px, 94vw);
  max-height: 90vh;
}
```

Immediately before it, insert:

```css
.tool-call-log-content {
  width: min(900px, 92vw);
  max-height: 90vh;
  display: flex;
  flex-direction: column;
}
.tool-call-log-filters {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}
.tool-call-log-filters input {
  flex: 1;
  min-width: 140px;
}
.tool-call-log-list {
  overflow-y: auto;
  max-height: 60vh;
}
.tool-call-log-list.tool-call-log-error {
  color: var(--red, #c0392b);
}
.tool-call-log-entry {
  border-bottom: 1px solid var(--border);
  padding: 8px 0;
}
.tool-call-log-meta {
  font-size: 12px;
  opacity: 0.75;
  margin-bottom: 4px;
}
.tool-call-log-command {
  font-family: monospace;
  font-size: 12px;
}
.tool-call-log-output {
  font-family: monospace;
  font-size: 12px;
  opacity: 0.8;
  white-space: pre-wrap;
  word-break: break-word;
}

```

- [ ] **Step 5: Create `static/js/toolCallLog.js`**

```javascript
// static/js/toolCallLog.js
import * as Modals from './modalManager.js';

function $(id) { return document.getElementById(id); }

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

async function api(path) {
  const res = await fetch(path, { credentials: 'same-origin' });
  const data = await res.json().catch(function () { return {}; });
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === 'string' ? d : (res.statusText || String(res.status)));
  }
  return data;
}

let _offset = 0;
const _PAGE_SIZE = 50;
let _toolFilter = '';
let _sessionFilter = '';

function _renderEntries(entries, append) {
  const list = $('tool-call-log-list');
  if (!list) return;
  const html = entries.map(function (c) {
    const cmd = esc(c.command || '');
    const output = esc((c.output || '').slice(0, 200));
    const exitCode = (c.exit_code === null || c.exit_code === undefined) ? '' :
      ' <span class="tool-call-log-exit">exit ' + esc(c.exit_code) + '</span>';
    const sid = esc(c.session_id || '');
    return (
      '<div class="tool-call-log-entry">' +
      '<div class="tool-call-log-meta">' +
      '<a href="#" class="tool-call-log-session" data-session-id="' + sid + '">' + esc(c.session_name || 'Unknown') + '</a>' +
      ' (<a href="#" class="tool-call-log-session-filter" data-session-id="' + sid + '">filter</a>)' +
      ' &middot; ' + esc(c.tool || '?') + exitCode +
      ' &middot; <span class="tool-call-log-time">' + esc(c.timestamp || '') + '</span>' +
      '</div>' +
      '<div class="tool-call-log-command">' + cmd + '</div>' +
      '<div class="tool-call-log-output">' + output + '</div>' +
      '</div>'
    );
  }).join('');
  if (append) {
    list.insertAdjacentHTML('beforeend', html);
  } else {
    list.innerHTML = html || '<div>No tool calls yet</div>';
  }
}

async function loadToolCallLog(append) {
  const list = $('tool-call-log-list');
  if (list && !append) list.classList.remove('tool-call-log-error');
  const moreBtn = $('tool-call-log-more');
  try {
    const params = new URLSearchParams();
    params.set('limit', String(_PAGE_SIZE));
    params.set('offset', String(append ? _offset : 0));
    if (_sessionFilter) params.set('session_id', _sessionFilter);
    if (_toolFilter) params.set('tool_name', _toolFilter);
    const data = await api('/api/tool-calls?' + params.toString());
    const entries = data.tool_calls || [];
    _renderEntries(entries, append);
    _offset = (append ? _offset : 0) + entries.length;
    if (moreBtn) moreBtn.style.display = data.has_more ? '' : 'none';
  } catch (e) {
    if (list) {
      list.classList.add('tool-call-log-error');
      list.textContent = 'Failed to load: ' + e.message;
    }
  }
}

function openToolCallLog() {
  $('tool-call-log-modal').classList.remove('hidden');
  _offset = 0;
  loadToolCallLog(false);
}

function closeToolCallLog() {
  $('tool-call-log-modal').classList.add('hidden');
}

function init() {
  const rail = $('rail-tool-calls');
  if (rail) rail.addEventListener('click', openToolCallLog);
  const side = $('tool-tool-calls-btn');
  if (side) side.addEventListener('click', openToolCallLog);
  const x = $('tool-call-log-close');
  if (x) x.addEventListener('click', closeToolCallLog);

  const applyBtn = $('tool-call-log-apply-filter');
  if (applyBtn) applyBtn.addEventListener('click', function () {
    const toolInput = $('tool-call-log-tool-filter');
    const sessionInput = $('tool-call-log-session-filter-input');
    _toolFilter = toolInput ? toolInput.value.trim() : '';
    _sessionFilter = sessionInput ? sessionInput.value.trim() : '';
    _offset = 0;
    loadToolCallLog(false);
  });

  const clearBtn = $('tool-call-log-clear-filter');
  if (clearBtn) clearBtn.addEventListener('click', function () {
    _toolFilter = '';
    _sessionFilter = '';
    const toolInput = $('tool-call-log-tool-filter');
    const sessionInput = $('tool-call-log-session-filter-input');
    if (toolInput) toolInput.value = '';
    if (sessionInput) sessionInput.value = '';
    _offset = 0;
    loadToolCallLog(false);
  });

  const moreBtn = $('tool-call-log-more');
  if (moreBtn) moreBtn.addEventListener('click', function () {
    loadToolCallLog(true);
  });

  const list = $('tool-call-log-list');
  if (list) list.addEventListener('click', function (ev) {
    const jump = ev.target.closest('.tool-call-log-session');
    if (jump) {
      ev.preventDefault();
      const sid = jump.getAttribute('data-session-id');
      if (sid) {
        closeToolCallLog();
        if (window.sessionModule && window.sessionModule.selectSession) {
          window.sessionModule.selectSession(sid);
        }
      }
      return;
    }
    const filterLink = ev.target.closest('.tool-call-log-session-filter');
    if (filterLink) {
      ev.preventDefault();
      const sid = filterLink.getAttribute('data-session-id');
      if (sid) {
        _sessionFilter = sid;
        const sessionInput = $('tool-call-log-session-filter-input');
        if (sessionInput) sessionInput.value = sid;
        _offset = 0;
        loadToolCallLog(false);
      }
    }
  });

  Modals.register('tool-call-log-modal', {
    railBtnId: 'rail-tool-calls', sidebarBtnId: 'tool-tool-calls-btn', closeFn: closeToolCallLog,
  });
}

document.addEventListener('DOMContentLoaded', init);
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_tool_call_log_ui.py -v --import-mode=importlib`
Expected: PASS (all 6 tests).

Run: `node --check static/js/toolCallLog.js`
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add static/index.html static/style.css static/js/toolCallLog.js tests/test_tool_call_log_ui.py
git commit -m "feat(tool-calls): Tool Call History panel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Mission Control "Tool Calls" widget (6th widget)

**Files:**
- Modify: `static/index.html` (add the 6th `.mission-control-card`)
- Modify: `static/js/missionControl.js` (`loadToolCallsWidget`, wire into `refreshWidget`/`loadAllWidgets`/`init`)
- Modify: `tests/test_mission_control_ui.py` (new widget test + extend the two seam-guard tests from sub-project 1's final review)

**Interfaces:**
- Consumes: `GET /api/tool-calls` (Task 2), `$`/`esc`/`api`/`setCardBody`/`setCardError` (existing, `static/js/missionControl.js`), `#tool-tool-calls-btn` (Task 3, this widget's open-link target).
- Produces: `loadToolCallsWidget()`, added to `refreshWidget`'s dispatch and `loadAllWidgets()`.

**Note on the widget's summary number:** unlike Memory's widget (which had a true `total` from its endpoint), `GET /api/tool-calls` doesn't return a total count — only up to `limit` items plus `has_more`. The widget fetches `limit=3` and shows `"<count> recent"` with a `+` suffix when `has_more` is true (e.g. `"3+ recent"`), rather than inventing a fake total.

- [ ] **Step 1: Write the failing test and extend the two seam-guard tests**

Append to `tests/test_mission_control_ui.py`:

```python
def test_mission_control_has_tool_calls_widget():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'data-widget="tool-calls"' in html
    assert 'id="mc-body-tool-calls"' in html

    src = (ROOT / "static" / "js" / "missionControl.js").read_text(encoding="utf-8")
    assert "loadToolCallsWidget" in src
    assert "/api/tool-calls" in src
```

Then, in the same file, find `test_mission_control_loadAllWidgets_wires_all_loaders` and add the 6th call to its `expected_calls` list:

```python
    expected_calls = [
        'loadModelsWidget()',
        'loadHardwareWidget()',
        'loadTasksWidget()',
        'loadMemoryWidget()',
        'loadIntegrationsWidget()',
        'loadToolCallsWidget()',
    ]
```

Find `test_mission_control_link_targets_exist_in_html` and add the 6th mapping to its `targets` dict:

```python
    targets = {
        'model-picker-btn': 'models',
        'hwmon': 'hardware',
        'rail-tasks': 'tasks',
        'tool-memory-btn': 'memory',
        'tool-plugins-btn': 'integrations',
        'tool-tool-calls-btn': 'tool-calls',
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mission_control_ui.py -v --import-mode=importlib`
Expected: FAIL — `test_mission_control_has_tool_calls_widget` fails (widget doesn't exist), and the two extended tests fail (`loadToolCallsWidget()` not in `loadAllWidgets()`'s body; `tool-tool-calls-btn` not referenced in `missionControl.js` yet).

- [ ] **Step 3: Add the 6th card to `static/index.html`**

Find this block (the Integrations card, the last card inside `#mission-control-grid`, immediately before its closing `</div></div>` that ends the grid and the modal):

```html
        <div class="mission-control-card" id="mc-card-integrations" data-widget="integrations">
          <div class="mission-control-card-header">
            <h5>Integrations</h5>
            <button class="btn mission-control-refresh" data-widget="integrations" title="Refresh">&#x21bb;</button>
          </div>
          <div class="mission-control-card-body" id="mc-body-integrations">Loading…</div>
          <button class="btn mission-control-open-link" id="mc-open-integrations">Open Integrations</button>
        </div>
```

Immediately after it, insert:

```html
        <div class="mission-control-card" id="mc-card-tool-calls" data-widget="tool-calls">
          <div class="mission-control-card-header">
            <h5>Tool Calls</h5>
            <button class="btn mission-control-refresh" data-widget="tool-calls" title="Refresh">&#x21bb;</button>
          </div>
          <div class="mission-control-card-body" id="mc-body-tool-calls">Loading…</div>
          <button class="btn mission-control-open-link" id="mc-open-tool-calls">Open Tool Call History</button>
        </div>
```

- [ ] **Step 4: Add `loadToolCallsWidget` to `static/js/missionControl.js`**

Find `loadIntegrationsWidget`'s closing `}` followed by `function refreshWidget(widgetId) {`. Immediately after `loadIntegrationsWidget`'s closing brace and before `refreshWidget`, insert:

```javascript
async function loadToolCallsWidget() {
  const body = $('mc-body-tool-calls');
  if (body) body.classList.remove('mc-error');
  try {
    const data = await api('/api/tool-calls?limit=3');
    const items = data.tool_calls || [];
    const listHtml = items.map(function (c) {
      const cmd = (c.command || '').slice(0, 40);
      return '<div>' + esc(c.tool || '?') + ': ' + esc(cmd) + '</div>';
    }).join('') || '<div>No tool calls yet</div>';
    const suffix = data.has_more ? '+' : '';
    setCardBody('tool-calls', esc(items.length) + suffix + ' recent<br>' + listHtml);
  } catch (e) {
    setCardError('tool-calls', e.message);
  }
}
```

Update `refreshWidget` to:

```javascript
function refreshWidget(widgetId) {
  if (widgetId === 'models') loadModelsWidget();
  if (widgetId === 'hardware') loadHardwareWidget();
  if (widgetId === 'tasks') loadTasksWidget();
  if (widgetId === 'memory') loadMemoryWidget();
  if (widgetId === 'integrations') loadIntegrationsWidget();
  if (widgetId === 'tool-calls') loadToolCallsWidget();
}
```

Update `loadAllWidgets` to:

```javascript
function loadAllWidgets() {
  loadModelsWidget();
  loadHardwareWidget();
  loadTasksWidget();
  loadMemoryWidget();
  loadIntegrationsWidget();
  loadToolCallsWidget();
}
```

Find the `mc-open-integrations` handler block in `init()`:

```javascript
  const openIntegrations = $('mc-open-integrations');
  if (openIntegrations) openIntegrations.addEventListener('click', function () {
    closeMissionControl();
    const pluginsBtn = $('tool-plugins-btn');
    if (pluginsBtn) pluginsBtn.click();
  });
```

Immediately after it, insert:

```javascript
  const openToolCalls = $('mc-open-tool-calls');
  if (openToolCalls) openToolCalls.addEventListener('click', function () {
    closeMissionControl();
    const toolCallsBtn = $('tool-tool-calls-btn');
    if (toolCallsBtn) toolCallsBtn.click();
  });
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mission_control_ui.py -v --import-mode=importlib`
Expected: PASS (all 11 tests: the original 10 from sub-project 1 plus the new `test_mission_control_has_tool_calls_widget`).

Run: `node --check static/js/missionControl.js`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/missionControl.js tests/test_mission_control_ui.py
git commit -m "feat(mission-control): Tool Calls widget (6th widget)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

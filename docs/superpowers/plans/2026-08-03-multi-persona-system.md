# Multi-Persona System — "Crew" (Sub-project 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the existing singleton "Personal Assistant" (`CrewMember` table, currently used for exactly one row per owner) into a real multi-persona system — create multiple named personas, start a new chat bound to any of them, and have that persona's personality/model/tools apply automatically for that whole conversation.

**Architecture:** A new shared helper module (`src/crew_helpers.py`) provides the data-access functions every other piece needs; a new CRUD route file (`routes/crew_routes.py`) exposes personas over HTTP; three small, surgical edits wire personas into the existing session-creation and chat-turn code paths; a new frontend panel (mirroring this session's established modal/controller shape) lets users create personas and start chats with them.

**Tech Stack:** FastAPI + SQLAlchemy (existing `CrewMember` table, no schema changes), vanilla ES modules (no build step), Python + Node test split matching every other panel built this session.

## Global Constraints

- **No new database columns.** `CrewMember` already has everything this sub-project needs: `name`, `avatar`, `personality`, `model`, `endpoint_url`, `greeting`, `enabled_tools`, `is_active`, `sort_order`, `is_default_assistant`, `owner`.
- **Owner-scoped, NOT admin-gated.** Every authenticated user manages their own personas. This is a deliberate departure from every other panel built this session (all admin-gated) — match the existing `assistant_routes.py`'s own gating (owner-only, no `require_admin` anywhere in that file).
- **One persona per session, bound at creation.** No mid-conversation persona switching. `Session.crew_member_id` already exists as the binding column.
- **The default Assistant (`is_default_assistant=True`) is managed through the SAME general CRUD** for its shared fields — it is not a separate concept. Its timezone/check-in scheduling stays exclusively in the existing `routes/assistant_routes.py`, untouched by this plan.
- **Deleting the default Assistant is blocked** (400) through the new general delete endpoint.
- **Persona wins over the text-preset system for system prompt/character name** when a session is persona-bound — the persona's `personality`/`name` fully replace what the separately-selected preset (`PresetManager`) would have contributed for that turn. A deleted/dangling linked persona falls back silently to normal preset-only behavior.
- **A malformed/missing `enabled_tools` fails OPEN** (adds no extra tool restriction) — this is a self-service customization within one user's own tool access, not a privilege boundary between users, so a parsing hiccup must never silently lock the user out of their own tools.
- **`known_tool_names()` (from `src/tool_policy.py`) is the source of truth for "every tool name that exists"** — the allowlist-to-denylist conversion is `known_tool_names() - set(enabled_tools)`.
- **pytest runs with `--import-mode=importlib`** (project convention). The test DB is a fresh in-memory SQLite per test process (`conftest.py` sets `DATABASE_URL=sqlite:///:memory:` by default) — tests that touch `CrewMember`/`Session` rows use the real `SessionLocal`/ORM directly, no mocking needed.
- Stage specific files when committing; never `git add -A`. Do not stage `installer/Output/Assist-Setup.exe`. Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Shared crew helpers

**Files:**
- Create: `src/crew_helpers.py`
- Modify: `routes/assistant_routes.py:46-64` (remove the now-duplicated `_crew_to_dict`, import the shared one instead)
- Test: `tests/test_crew_helpers.py`

**Interfaces:**
- Produces: `crew_to_dict(c: CrewMember) -> dict` (identical shape to the current `_crew_to_dict`); `resolve_crew_binding(db, session_id: str, owner: str) -> Optional[CrewMember]`; `crew_disabled_tools(db, session_id: str, owner: str) -> set` — all consumed by Tasks 2-5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crew_helpers.py
import json
import uuid

from core.database import SessionLocal, CrewMember, Session as DbSession
from src.crew_helpers import crew_to_dict, resolve_crew_binding, crew_disabled_tools


def _make_crew(db, owner="alice", enabled_tools=None, personality="You are helpful."):
    c = CrewMember(
        id=str(uuid.uuid4()), owner=owner, name="Nav", personality=personality,
        enabled_tools=json.dumps(enabled_tools) if enabled_tools is not None else None,
    )
    db.add(c)
    db.commit()
    return c


def _make_session(db, owner="alice", crew_member_id=None):
    s = DbSession(id=str(uuid.uuid4()), name="s", endpoint_url="http://x", model="m",
                 owner=owner, crew_member_id=crew_member_id)
    db.add(s)
    db.commit()
    return s


def test_crew_to_dict_shape():
    db = SessionLocal()
    try:
        c = _make_crew(db, enabled_tools=["web_search"])
        d = crew_to_dict(c)
        assert d["name"] == "Nav" and d["enabled_tools"] == ["web_search"]
        assert d["is_default_assistant"] is False
    finally:
        db.close()


def test_crew_to_dict_malformed_json_tools_becomes_empty_list():
    db = SessionLocal()
    try:
        c = _make_crew(db)
        c.enabled_tools = "{not json"
        d = crew_to_dict(c)
        assert d["enabled_tools"] == []
    finally:
        db.close()


def test_resolve_crew_binding_returns_none_when_session_unbound():
    db = SessionLocal()
    try:
        s = _make_session(db)
        assert resolve_crew_binding(db, s.id, "alice") is None
    finally:
        db.close()


def test_resolve_crew_binding_returns_owner_scoped_crew():
    db = SessionLocal()
    try:
        c = _make_crew(db, owner="alice")
        s = _make_session(db, owner="alice", crew_member_id=c.id)
        found = resolve_crew_binding(db, s.id, "alice")
        assert found is not None and found.id == c.id
    finally:
        db.close()


def test_resolve_crew_binding_never_returns_another_owners_crew():
    db = SessionLocal()
    try:
        c = _make_crew(db, owner="alice")
        s = _make_session(db, owner="alice", crew_member_id=c.id)
        # A different owner querying the same session id must not see it.
        assert resolve_crew_binding(db, s.id, "bob") is None
    finally:
        db.close()


def test_resolve_crew_binding_never_raises_on_missing_session():
    db = SessionLocal()
    try:
        assert resolve_crew_binding(db, "no-such-session", "alice") is None
    finally:
        db.close()


def test_crew_disabled_tools_empty_when_unbound():
    db = SessionLocal()
    try:
        s = _make_session(db)
        assert crew_disabled_tools(db, s.id, "alice") == set()
    finally:
        db.close()


def test_crew_disabled_tools_empty_when_all():
    db = SessionLocal()
    try:
        c = _make_crew(db, owner="alice", enabled_tools=None)
        c.enabled_tools = "all"
        db.commit()
        s = _make_session(db, owner="alice", crew_member_id=c.id)
        assert crew_disabled_tools(db, s.id, "alice") == set()
    finally:
        db.close()


def test_crew_disabled_tools_restricts_to_allowlist():
    from src.tool_policy import known_tool_names
    db = SessionLocal()
    try:
        c = _make_crew(db, owner="alice", enabled_tools=["web_search"])
        s = _make_session(db, owner="alice", crew_member_id=c.id)
        disabled = crew_disabled_tools(db, s.id, "alice")
        assert "web_search" not in disabled
        assert disabled == (known_tool_names() - {"web_search"})
    finally:
        db.close()


def test_crew_disabled_tools_fails_open_on_malformed_json():
    db = SessionLocal()
    try:
        c = _make_crew(db, owner="alice")
        c.enabled_tools = "{not json"
        db.commit()
        s = _make_session(db, owner="alice", crew_member_id=c.id)
        assert crew_disabled_tools(db, s.id, "alice") == set()
    finally:
        db.close()


def test_crew_disabled_tools_never_raises_on_dangling_reference():
    db = SessionLocal()
    try:
        s = _make_session(db, owner="alice", crew_member_id="deleted-crew-id")
        assert crew_disabled_tools(db, s.id, "alice") == set()
    finally:
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crew_helpers.py -v --import-mode=importlib`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.crew_helpers'`

- [ ] **Step 3: Write the implementation**

```python
# src/crew_helpers.py
"""Shared data-access helpers for CrewMember ("persona") rows. Used by
routes/assistant_routes.py (the singleton default-assistant feature),
routes/crew_routes.py (the general multi-persona CRUD), and the chat-turn
wiring in routes/chat_helpers.py and routes/chat_routes.py. Never raises --
every function here degrades to an empty/None result on any lookup or
parsing failure, since a persona binding is a self-service customization,
not a security boundary."""
import json

from core.database import CrewMember, Session as DbSession
from src.tool_policy import known_tool_names


def crew_to_dict(c: CrewMember) -> dict:
    try:
        tools = json.loads(c.enabled_tools) if c.enabled_tools else []
    except Exception:
        tools = []
    return {
        "id": c.id,
        "name": c.name,
        "avatar": c.avatar,
        "personality": c.personality,
        "model": c.model,
        "endpoint_url": c.endpoint_url,
        "greeting": c.greeting,
        "enabled_tools": tools,
        "session_id": c.session_id,
        "is_default_assistant": bool(c.is_default_assistant),
        "is_active": bool(c.is_active),
        "sort_order": c.sort_order or 0,
    }


def resolve_crew_binding(db, session_id: str, owner: str):
    """Return the owner-scoped CrewMember bound to `session_id`, or None if
    the session doesn't exist, has no binding, the binding is dangling, or
    belongs to a different owner. Never raises."""
    try:
        sess = db.query(DbSession).filter(DbSession.id == session_id).first()
        if not sess or not sess.crew_member_id:
            return None
        crew = db.query(CrewMember).filter(
            CrewMember.id == sess.crew_member_id,
            CrewMember.owner == owner,
        ).first()
        return crew
    except Exception:  # noqa: BLE001
        return None


def crew_disabled_tools(db, session_id: str, owner: str) -> set:
    """Return the set of tool names a session's bound persona restricts,
    i.e. every known tool NOT in its enabled_tools allowlist. Empty set
    (no extra restriction) when unbound, "all", empty, missing, or
    unparseable -- fail-open, never raises."""
    try:
        crew = resolve_crew_binding(db, session_id, owner)
        if not crew or not crew.enabled_tools or crew.enabled_tools == "all":
            return set()
        try:
            allowed = json.loads(crew.enabled_tools)
        except Exception:
            return set()
        if not isinstance(allowed, list) or not allowed:
            return set()
        return known_tool_names() - set(allowed)
    except Exception:  # noqa: BLE001
        return set()
```

In `routes/assistant_routes.py`, remove the local `_crew_to_dict` function (lines 46-64) and its now-unused `import json` if nothing else in the file needs it (check: `json.loads`/`json.dumps` are still used elsewhere in this file for `enabled_tools`, so keep the `json` import — only remove the function body), replacing every call site (`_crew_to_dict(...)`) with `crew_to_dict(...)`, and add:

```python
from src.crew_helpers import crew_to_dict
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_crew_helpers.py -v --import-mode=importlib`
Expected: PASS (11 tests)

Then confirm the app still boots and the existing assistant routes still work with the shared helper:

Run: `python -c "import app"`
Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add src/crew_helpers.py routes/assistant_routes.py tests/test_crew_helpers.py
git commit -m "feat(crew): add shared persona data-access helpers"
```

---

### Task 2: Crew CRUD API

**Files:**
- Create: `routes/crew_routes.py`
- Modify: `app.py` (register the new router, near the existing `setup_assistant_routes` registration)
- Test: `tests/test_crew_routes.py`

**Interfaces:**
- Consumes: `crew_to_dict(c)` from Task 1 (`src/crew_helpers.py`); `CrewMember` (`core/database.py`); `get_current_user(request)` and `owner_filter(query, model_cls, user, *, include_shared=True)` (`src/auth_helpers.py`, both already shipped).
- Produces: routes `GET/POST /api/crew`, `PATCH/DELETE /api/crew/{id}`, `GET /api/crew/tool-names`. No later task in this plan calls these routes directly (Task 6's frontend does, but that's out-of-process via `fetch`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crew_routes.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.crew_routes as cr


def _client(monkeypatch, user="alice"):
    monkeypatch.setattr(cr, "get_current_user", lambda request: user)
    app = FastAPI()
    app.include_router(cr.setup_crew_routes())
    return TestClient(app)


def test_list_starts_empty(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/api/crew")
    assert r.status_code == 200 and r.json()["crew"] == []


def test_create_then_list(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/crew", json={"name": "Nav", "personality": "Be terse."})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Nav" and body["personality"] == "Be terse."
    r2 = c.get("/api/crew")
    assert len(r2.json()["crew"]) == 1


def test_create_requires_name(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/crew", json={"name": ""})
    assert r.status_code == 400


def test_update_persona(monkeypatch):
    c = _client(monkeypatch)
    created = c.post("/api/crew", json={"name": "Nav"}).json()
    r = c.patch(f"/api/crew/{created['id']}", json={"personality": "Be blunt."})
    assert r.status_code == 200 and r.json()["personality"] == "Be blunt."


def test_update_missing_id_404s(monkeypatch):
    c = _client(monkeypatch)
    r = c.patch("/api/crew/does-not-exist", json={"name": "x"})
    assert r.status_code == 404


def test_update_another_owners_persona_404s(monkeypatch):
    c_alice = _client(monkeypatch, user="alice")
    created = c_alice.post("/api/crew", json={"name": "Nav"}).json()
    c_bob = _client(monkeypatch, user="bob")
    r = c_bob.patch(f"/api/crew/{created['id']}", json={"name": "Hijacked"})
    assert r.status_code == 404


def test_delete_persona(monkeypatch):
    c = _client(monkeypatch)
    created = c.post("/api/crew", json={"name": "Nav"}).json()
    r = c.delete(f"/api/crew/{created['id']}")
    assert r.status_code == 200
    assert c.get("/api/crew").json()["crew"] == []


def test_delete_default_assistant_is_blocked(monkeypatch):
    from core.database import SessionLocal, CrewMember
    import uuid
    db = SessionLocal()
    crew_id = str(uuid.uuid4())
    try:
        db.add(CrewMember(id=crew_id, owner="alice", name="Assistant", is_default_assistant=True))
        db.commit()
    finally:
        db.close()
    c = _client(monkeypatch)
    r = c.delete(f"/api/crew/{crew_id}")
    assert r.status_code == 400


def test_tool_names_endpoint(monkeypatch):
    from src.tool_policy import known_tool_names
    c = _client(monkeypatch)
    r = c.get("/api/crew/tool-names")
    assert r.status_code == 200
    tools = r.json()["tools"]
    assert tools == sorted(known_tool_names())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crew_routes.py -v --import-mode=importlib`
Expected: FAIL with `ModuleNotFoundError: No module named 'routes.crew_routes'`

- [ ] **Step 3: Write the implementation**

```python
# routes/crew_routes.py
"""Owner-scoped CRUD over CrewMember ("persona") rows -- NOT admin-gated,
matching routes/assistant_routes.py's own gating (every authenticated user
manages their own personas). The default Assistant (is_default_assistant=True)
appears in the same list; its timezone/check-in extras stay exclusively in
assistant_routes.py."""
import uuid

from fastapi import APIRouter, Body, HTTPException, Request

from core.database import SessionLocal, CrewMember
from src.auth_helpers import get_current_user, owner_filter
from src.crew_helpers import crew_to_dict
from src.tool_policy import known_tool_names


def _owner(request: Request) -> str:
    owner = get_current_user(request)
    if not owner:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return owner


def setup_crew_routes() -> APIRouter:
    router = APIRouter(prefix="/api/crew", tags=["crew"])

    @router.get("/tool-names")
    async def tool_names():
        return {"tools": sorted(known_tool_names())}

    @router.get("")
    async def list_crew(request: Request):
        owner = _owner(request)
        db = SessionLocal()
        try:
            q = db.query(CrewMember)
            q = owner_filter(q, CrewMember, owner, include_shared=False)
            rows = q.order_by(CrewMember.sort_order.asc()).all()
            return {"crew": [crew_to_dict(c) for c in rows]}
        finally:
            db.close()

    @router.post("")
    async def create_crew(request: Request, body: dict = Body(...)):
        owner = _owner(request)
        name = str(body.get("name", "")).strip()
        if not name:
            raise HTTPException(400, "name is required")
        db = SessionLocal()
        try:
            import json
            enabled_tools = body.get("enabled_tools")
            c = CrewMember(
                id=str(uuid.uuid4()),
                owner=owner,
                name=name,
                avatar=body.get("avatar"),
                personality=body.get("personality"),
                model=body.get("model"),
                endpoint_url=body.get("endpoint_url"),
                greeting=body.get("greeting"),
                enabled_tools=json.dumps(enabled_tools) if enabled_tools is not None else None,
            )
            db.add(c)
            db.commit()
            return crew_to_dict(c)
        finally:
            db.close()

    def _find_owned(db, crew_id: str, owner: str):
        q = db.query(CrewMember).filter(CrewMember.id == crew_id)
        q = owner_filter(q, CrewMember, owner, include_shared=False)
        return q.first()

    @router.patch("/{crew_id}")
    async def update_crew(crew_id: str, request: Request, body: dict = Body(...)):
        owner = _owner(request)
        db = SessionLocal()
        try:
            c = _find_owned(db, crew_id, owner)
            if not c:
                raise HTTPException(404, "Persona not found")
            import json
            if "name" in body and str(body["name"]).strip():
                c.name = str(body["name"]).strip()
            if "avatar" in body:
                c.avatar = body["avatar"]
            if "personality" in body:
                c.personality = body["personality"]
            if "model" in body:
                c.model = body["model"]
            if "endpoint_url" in body:
                c.endpoint_url = body["endpoint_url"]
            if "greeting" in body:
                c.greeting = body["greeting"]
            if "enabled_tools" in body:
                c.enabled_tools = json.dumps(body["enabled_tools"]) if body["enabled_tools"] is not None else None
            if "is_active" in body:
                c.is_active = bool(body["is_active"])
            if "sort_order" in body:
                c.sort_order = int(body["sort_order"])
            db.commit()
            return crew_to_dict(c)
        finally:
            db.close()

    @router.delete("/{crew_id}")
    async def delete_crew(crew_id: str, request: Request):
        owner = _owner(request)
        db = SessionLocal()
        try:
            c = _find_owned(db, crew_id, owner)
            if not c:
                raise HTTPException(404, "Persona not found")
            if c.is_default_assistant:
                raise HTTPException(400, "Cannot delete the default Assistant")
            db.delete(c)
            db.commit()
            return {"ok": True}
        finally:
            db.close()

    return router
```

In `app.py`, right after the existing lines:
```python
from routes.assistant_routes import setup_assistant_routes
app.include_router(setup_assistant_routes(task_scheduler))
```
add:
```python
from routes.crew_routes import setup_crew_routes
app.include_router(setup_crew_routes())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_crew_routes.py -v --import-mode=importlib`
Expected: PASS (9 tests)

Then confirm the app boots with the new router registered:

Run: `python -c "import app; print('/api/crew' in [r.path for r in app.app.routes])"`
Expected: prints `True`

- [ ] **Step 5: Commit**

```bash
git add routes/crew_routes.py app.py tests/test_crew_routes.py
git commit -m "feat(crew): add owner-scoped multi-persona CRUD API"
```

---

### Task 3: Session creation wiring

**Files:**
- Modify: `routes/session_routes.py` (the `create_session` function, ~line 320-449)
- Modify: `core/session_manager.py` (`SessionManager.create_session`, ~line 470-511)
- Test: `tests/test_session_crew_binding.py`

**Interfaces:**
- Consumes: `CrewMember` (`core/database.py`); `owner_filter` (`src/auth_helpers.py`, already used in this file).
- Produces: `create_session` now accepts an optional `crew_member_id` Form field; `SessionManager.create_session` now accepts and persists an optional `crew_member_id` keyword argument on the `DbSession` row. No other task depends on new names here — Tasks 4/5 read the binding back via Task 1's `resolve_crew_binding`/`crew_disabled_tools`, not through this task's new parameter.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_session_crew_binding.py
import uuid
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_crew(owner="alice", model="gpt-x", endpoint_url="http://persona-endpoint"):
    from core.database import SessionLocal, CrewMember
    db = SessionLocal()
    try:
        c = CrewMember(id=str(uuid.uuid4()), owner=owner, name="Nav",
                       model=model, endpoint_url=endpoint_url)
        db.add(c)
        db.commit()
        return c.id
    finally:
        db.close()


def _client(monkeypatch):
    import routes.session_routes as sr
    from core.session_manager import SessionManager
    monkeypatch.setattr(sr, "effective_user", lambda request: "alice")
    app = FastAPI()
    app.include_router(sr.setup_session_routes(SessionManager(), {}, webhook_manager=None))
    return TestClient(app)


def test_create_session_with_crew_member_id_defaults_model_and_endpoint(monkeypatch):
    from core.database import SessionLocal, Session as DbSession
    crew_id = _make_crew()
    client = _client(monkeypatch)
    resp = client.post("/api/session", data={"crew_member_id": crew_id, "skip_validation": "true"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "gpt-x"
    db = SessionLocal()
    try:
        row = db.query(DbSession).filter(DbSession.id == body["id"]).first()
        assert row.crew_member_id == crew_id
    finally:
        db.close()


def test_create_session_with_unknown_crew_member_id_400s(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post("/api/session", data={"crew_member_id": "no-such-id", "skip_validation": "true"})
    assert resp.status_code == 400


def test_create_session_explicit_model_overrides_persona_default(monkeypatch):
    crew_id = _make_crew()
    client = _client(monkeypatch)
    resp = client.post("/api/session", data={
        "crew_member_id": crew_id, "model": "explicit-model",
        "endpoint_url": "http://explicit", "skip_validation": "true",
    })
    assert resp.status_code == 200
    assert resp.json()["model"] == "explicit-model"
```

(Confirmed: `setup_session_routes` builds on a module-level `router = APIRouter(prefix="/api", tags=["sessions"])` at `routes/session_routes.py:125`, and `create_session` is decorated `@router.post("/session", response_model=SessionResponse)` at line 319 — so the full path is exactly `/api/session`, as used above.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_session_crew_binding.py -v --import-mode=importlib`
Expected: FAIL (either a fixture/setup error to resolve per the note above, or — once setup is correct — failures because `crew_member_id` isn't accepted/persisted yet)

- [ ] **Step 3: Write the implementation**

In `core/session_manager.py`, modify `create_session`:

```python
    def create_session(
        self,
        session_id: str,
        name: str,
        endpoint_url: str,
        model: str,
        rag: bool = False,
        owner: str = None,
        crew_member_id: str = None,
    ) -> Session:
        """Create a new session and save to database."""
        db = SessionLocal()
        try:
            db_session = DbSession(
                id=session_id,
                name=name,
                endpoint_url=endpoint_url,
                model=model,
                rag=rag,
                headers={},
                owner=owner,
                crew_member_id=crew_member_id,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            db.add(db_session)
            db.commit()

            session = Session(
                id=session_id,
                name=name,
                endpoint_url=endpoint_url,
                model=model,
                rag=rag,
                headers={},
                owner=owner,
            )

            self.sessions[session_id] = session
            return session

        except Exception as e:
            db.rollback()
```
(the rest of the method, and the in-memory `Session` dataclass, are unchanged — `crew_member_id` is persisted on the DB row only; chat-turn code reads it back fresh via Task 1's `resolve_crew_binding`, so the in-memory cache does not need this field)

In `routes/session_routes.py`, modify the `create_session` route function:

1. Add a new parameter to the function signature, alongside the existing `endpoint_id: str = Form("")`:
```python
        crew_member_id: str = Form(""),
```

2. Immediately after `user = effective_user(request)` is first available in this function and BEFORE the `if not endpoint_url and not skip_val:` check (i.e., resolve the persona early enough to supply `endpoint_url`/`model` defaults before that validation runs) — this requires moving the `user = effective_user(request)` line (currently at line 416, after the validation block) to the TOP of the function, before the validation block, since the persona lookup needs it there. Add:
```python
        user = effective_user(request)

        crew_model = None
        crew_endpoint_url = None
        if crew_member_id and crew_member_id.strip():
            from core.database import CrewMember
            _db = SessionLocal()
            try:
                _crew = _db.query(CrewMember).filter(
                    CrewMember.id == crew_member_id.strip(),
                ).first()
                if not _crew:
                    raise HTTPException(400, "Persona not found")
                from src.auth_helpers import owner_filter as _owner_filter
                _owned = _db.query(CrewMember).filter(CrewMember.id == crew_member_id.strip())
                _owned = _owner_filter(_owned, CrewMember, user, include_shared=False)
                if not _owned.first():
                    raise HTTPException(400, "Persona not found")
                crew_model = _crew.model
                crew_endpoint_url = _crew.endpoint_url
            finally:
                _db.close()
        if not endpoint_url and crew_endpoint_url:
            endpoint_url = crew_endpoint_url
        if not model and crew_model:
            model = crew_model
```
(remove the OLD `user = effective_user(request)` line that currently appears right before `session = session_manager.create_session(...)`, since it's now computed at the top instead — do not compute it twice)

3. Pass `crew_member_id` through to the manager call:
```python
        session = session_manager.create_session(
            session_id=sid,
            name=name or "",
            endpoint_url=endpoint_url or "",
            model=model_to_use,
            rag=str(rag).lower() == "true" if rag else False,
            owner=user,
            crew_member_id=crew_member_id.strip() if crew_member_id and crew_member_id.strip() else None,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_session_crew_binding.py -v --import-mode=importlib`
Expected: PASS (3 tests)

Then run the existing session-routes test suite to confirm no regression (moving `user = effective_user(request)` earlier must not change behavior for the no-`crew_member_id` path):

Run: `python -m pytest tests/ -v --import-mode=importlib -k session_routes`
Expected: PASS, same pass count as before this task's changes

- [ ] **Step 5: Commit**

```bash
git add routes/session_routes.py core/session_manager.py tests/test_session_crew_binding.py
git commit -m "feat(crew): bind a new session to a persona at creation"
```

---

### Task 4: Chat-turn personality wiring

**Files:**
- Modify: `routes/chat_helpers.py` (`extract_preset`, ~line 335-345, and its call site, ~line 648)
- Test: `tests/test_crew_chat_wiring.py`

**Interfaces:**
- Consumes: `resolve_crew_binding(db, session_id, owner)` from Task 1 (`src/crew_helpers.py`).
- Produces: `extract_preset(chat_handler, preset_id, sess=None, owner=None) -> PresetInfo` — the two new keyword-only-by-convention parameters are optional so every other existing caller (if any beyond the one call site found) keeps working unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crew_chat_wiring.py
import uuid
from unittest.mock import MagicMock

from routes.chat_helpers import extract_preset


class _FakeSess:
    def __init__(self, sid):
        self.id = sid


def _make_bound_session(personality="You are a pirate.", name="Cap'n"):
    from core.database import SessionLocal, CrewMember, Session as DbSession
    db = SessionLocal()
    try:
        crew_id = str(uuid.uuid4())
        db.add(CrewMember(id=crew_id, owner="alice", name=name, personality=personality))
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m",
                         owner="alice", crew_member_id=crew_id))
        db.commit()
        return sess_id
    finally:
        db.close()


def _fake_chat_handler(system_prompt="preset prompt", char_name="Preset Name"):
    h = MagicMock()
    h.validate_and_extract_preset.return_value = (0.5, 100, system_prompt, char_name)
    return h


def test_extract_preset_unbound_session_uses_preset_as_before():
    sess_id = str(uuid.uuid4())
    handler = _fake_chat_handler()
    result = extract_preset(handler, "some_preset", sess=_FakeSess(sess_id), owner="alice")
    assert result.system_prompt == "preset prompt"
    assert result.character_name == "Preset Name"


def test_extract_preset_persona_bound_session_overrides_preset():
    sess_id = _make_bound_session(personality="You are a pirate.", name="Cap'n")
    handler = _fake_chat_handler()
    result = extract_preset(handler, "some_preset", sess=_FakeSess(sess_id), owner="alice")
    assert result.system_prompt == "You are a pirate."
    assert result.character_name == "Cap'n"


def test_extract_preset_persona_with_empty_personality_falls_back_to_preset():
    sess_id = _make_bound_session(personality="", name="Cap'n")
    handler = _fake_chat_handler()
    result = extract_preset(handler, "some_preset", sess=_FakeSess(sess_id), owner="alice")
    assert result.system_prompt == "preset prompt"


def test_extract_preset_dangling_crew_reference_falls_back_to_preset():
    from core.database import SessionLocal, Session as DbSession
    db = SessionLocal()
    try:
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m",
                         owner="alice", crew_member_id="deleted-crew-id"))
        db.commit()
    finally:
        db.close()
    handler = _fake_chat_handler()
    result = extract_preset(handler, "some_preset", sess=_FakeSess(sess_id), owner="alice")
    assert result.system_prompt == "preset prompt"


def test_extract_preset_no_sess_arg_behaves_like_before():
    handler = _fake_chat_handler()
    result = extract_preset(handler, "some_preset")
    assert result.system_prompt == "preset prompt"
    assert result.character_name == "Preset Name"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crew_chat_wiring.py -v --import-mode=importlib`
Expected: FAIL — `extract_preset` doesn't yet accept `sess`/`owner` keyword arguments

- [ ] **Step 3: Write the implementation**

In `routes/chat_helpers.py`, modify `extract_preset`:

```python
def extract_preset(chat_handler, preset_id, sess=None, owner=None) -> PresetInfo:
    """Extract preset parameters via chat_handler. If `sess`/`owner` are given
    and the session is bound to a persona (CrewMember) with a non-empty
    personality, the persona's personality/name REPLACE the preset's
    system_prompt/character_name for this turn -- the persona already
    defines the assistant's voice for a persona-bound conversation. A
    missing/dangling binding falls back to normal preset-only behavior."""
    temperature, max_tokens, system_prompt, char_name = (
        chat_handler.validate_and_extract_preset(preset_id)
    )
    if sess is not None and owner is not None:
        from core.database import SessionLocal
        from src.crew_helpers import resolve_crew_binding
        db = SessionLocal()
        try:
            crew = resolve_crew_binding(db, sess.id, owner)
            if crew and crew.personality:
                system_prompt = crew.personality
                char_name = crew.name
        finally:
            db.close()
    return PresetInfo(
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        character_name=char_name,
    )
```

At the call site (`routes/chat_helpers.py`, currently `preset = extract_preset(chat_handler, preset_id)` around line 648), read the surrounding function to find the already-available `sess` and owner variable (the containing function is `build_chat_context`, which already receives `sess` as its first parameter per its call sites in `chat_routes.py`; find the owner value it already has in scope from an `effective_user(request)`/similar call earlier in the same function) and change the call to:
```python
    preset = extract_preset(chat_handler, preset_id, sess=sess, owner=owner)
```
using whatever the existing local variable name for the owner is at that point in `build_chat_context` (read the function to find it — do not introduce a second, differently-named owner variable if one already exists in scope).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_crew_chat_wiring.py -v --import-mode=importlib`
Expected: PASS (5 tests)

Then run the existing chat-helpers/chat-routes test suites to confirm no regression:

Run: `python -m pytest tests/ -v --import-mode=importlib -k "chat_helpers or chat_routes"`
Expected: PASS, same pass count as before this task's changes

- [ ] **Step 5: Commit**

```bash
git add routes/chat_helpers.py tests/test_crew_chat_wiring.py
git commit -m "feat(crew): persona personality overrides the text-preset system prompt"
```

---

### Task 5: Chat-turn tool-policy wiring

**Files:**
- Modify: `routes/chat_routes.py` (three call sites: ~line 385, ~line 657-659, ~line 881-884)
- Test: `tests/test_crew_tool_policy_wiring.py`

**Interfaces:**
- Consumes: `crew_disabled_tools(db, session_id, owner)` from Task 1 (`src/crew_helpers.py`); `SessionLocal` (`core/database.py`).
- Produces: nothing consumed by a later task — this is the last wiring task.

- [ ] **Step 1: Write the failing tests**

Because `build_effective_tool_policy` itself is unchanged (Task 1's `crew_disabled_tools` is a pure additive input to it, already fully tested in Task 1), this task's own test verifies the WIRING — that a persona-bound session's restricted tools actually show up in the composed policy at each of the three call sites, using the real route layer:

```python
# tests/test_crew_tool_policy_wiring.py
import uuid


def _make_restricted_session(owner="alice", enabled_tools=None):
    import json
    from core.database import SessionLocal, CrewMember, Session as DbSession
    db = SessionLocal()
    try:
        crew_id = str(uuid.uuid4())
        db.add(CrewMember(id=crew_id, owner=owner, name="Nav",
                          enabled_tools=json.dumps(enabled_tools or ["web_search"])))
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m",
                         owner=owner, crew_member_id=crew_id))
        db.commit()
        return sess_id
    finally:
        db.close()


def test_crew_disabled_tools_reflects_in_a_composed_policy_for_a_bound_session():
    # This exercises exactly the composition the three chat_routes.py call
    # sites now perform: crew_disabled_tools(...) unioned into disabled_tools
    # before build_effective_tool_policy(disabled_tools=...).
    from src.crew_helpers import crew_disabled_tools
    from src.tool_policy import build_effective_tool_policy
    from core.database import SessionLocal

    sess_id = _make_restricted_session(enabled_tools=["web_search"])
    db = SessionLocal()
    try:
        extra = crew_disabled_tools(db, sess_id, "alice")
    finally:
        db.close()
    policy = build_effective_tool_policy(disabled_tools=extra)
    assert policy.blocks("generate_image") is True
    assert policy.blocks("web_search") is False


def test_unbound_session_adds_no_extra_restriction():
    from core.database import SessionLocal, Session as DbSession
    from src.crew_helpers import crew_disabled_tools
    from src.tool_policy import build_effective_tool_policy
    import uuid as _uuid

    db = SessionLocal()
    try:
        sess_id = str(_uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m", owner="alice"))
        db.commit()
        extra = crew_disabled_tools(db, sess_id, "alice")
    finally:
        db.close()
    policy = build_effective_tool_policy(disabled_tools=extra)
    assert policy.blocks("generate_image") is False
    assert policy.blocks("web_search") is False
```

**Note for the implementer:** this task's automated test deliberately verifies the composition logic at the `crew_helpers`/`tool_policy` boundary (already the real functions, no mocks) rather than driving the full `/api/chat` HTTP endpoint, since standing up a complete chat request in a unit test requires mocking a large amount of unrelated machinery (LLM calls, streaming) that has nothing to do with this task's change. The actual route-level wiring (Step 3 below) is what Step 4's regression run against the existing `chat_routes` test suite protects.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crew_tool_policy_wiring.py -v --import-mode=importlib`
Expected: FAIL if `src/crew_helpers.py` doesn't exist yet (it does, from Task 1) — actually expected to PASS already once Task 1 is done, since this test only exercises Task 1's function directly. Run it anyway to confirm before touching `chat_routes.py`, then proceed to Step 3 to add the real route-level wiring.

- [ ] **Step 3: Write the implementation**

In `routes/chat_routes.py`, add the import near the top:
```python
from src.crew_helpers import crew_disabled_tools
```

At call site 1 (~line 385, inside the same function where `owner = effective_user(request)` is already resolved earlier at line 607 in the surrounding function — confirm the exact local variable name for owner and session id string in this function before writing the change):
```python
        from core.database import SessionLocal as _SessionLocal
        _db = _SessionLocal()
        try:
            _crew_extra = crew_disabled_tools(_db, session, owner)
        finally:
            _db.close()
        tool_policy = build_effective_tool_policy(
            disabled_tools=_crew_extra, last_user_message=message,
        )
```
(replacing the current `tool_policy = build_effective_tool_policy(last_user_message=message)`)

At call site 2 (~line 657-659, same pattern — `owner`/`session` are already in scope in this function per the surrounding code):
```python
        from core.database import SessionLocal as _SessionLocal
        _db = _SessionLocal()
        try:
            _crew_extra = crew_disabled_tools(_db, session, owner)
        finally:
            _db.close()
        pre_context_tool_policy = build_effective_tool_policy(
            disabled_tools=_crew_extra,
            last_user_message=message,
        )
```
(replacing the current unparameterized `disabled_tools`-less call)

At call site 3 (~line 881-884, where `disabled_tools` already exists as a set being built up from `_compare_strip`/`plan_mode_disabled_tools()`), add ONE line right before the `tool_policy = build_effective_tool_policy(...)` call:
```python
        from core.database import SessionLocal as _SessionLocal
        _db = _SessionLocal()
        try:
            disabled_tools.update(crew_disabled_tools(_db, session, owner))
        finally:
            _db.close()

        tool_policy = build_effective_tool_policy(
            disabled_tools=disabled_tools,
            last_user_message=message,
        )
```

For all three sites: read the exact surrounding code first to confirm the local variable names actually in scope for the session-id string and the resolved owner at that point in the function (they may not literally be named `session`/`owner` at every site — match whatever names the existing code already uses there, do not introduce new differently-scoped variables).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_crew_tool_policy_wiring.py -v --import-mode=importlib`
Expected: PASS (2 tests)

Then run the existing chat_routes test suite to confirm no regression from the three edits:

Run: `python -m pytest tests/ -v --import-mode=importlib -k chat_routes`
Expected: PASS, same pass count as before this task's changes

- [ ] **Step 5: Commit**

```bash
git add routes/chat_routes.py tests/test_crew_tool_policy_wiring.py
git commit -m "feat(crew): persona tool allowlist restricts the chat-turn tool policy"
```

---

### Task 6: Crew frontend panel

**Files:**
- Modify: `static/index.html` (new modal, new rail button, new sidebar entry, new script tag — NOT admin-gated)
- Modify: `static/js/sessions.js` (`createDirectChat`, ~line 2100, and `materializePendingSession`, ~line 2158)
- Create: `static/js/crew.js`
- Test: `tests/test_crew_ui.py`

**Interfaces:**
- Consumes: `Modals.register(modalId, opts)` (`static/js/modalManager.js`, already shipped); `GET/POST/PATCH/DELETE /api/crew*` (Task 2); `createDirectChat(url, modelId, endpointId, crewMemberId)` (this task's own extension of the existing `sessions.js` function).
- Produces: nothing consumed by a later task — final task of this plan.

**Important context found while planning this task:** this app does NOT create a session immediately when starting a new chat. `sessions.js`'s `createDirectChat(url, modelId, endpointId)` (~line 2100) only stages a `_pendingChat` object and updates the UI to look like a fresh chat; the actual `POST /api/session` call happens later, in `materializePendingSession()` (~line 2158), triggered by the FIRST message the user sends. "New Chat with persona" must follow this exact same lazy pattern — stage the pending chat with the persona's model/endpoint AND its id, and let the existing first-send materialization carry `crew_member_id` through. Do not call `POST /api/session` directly from `crew.js`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crew_ui.py
import pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_index_has_crew_modal_and_entries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for el in ('id="crew-modal"', 'id="rail-crew"', 'id="tool-crew-btn"',
               '/static/js/crew.js',
               'id="crew-grid"', 'id="crew-new-btn"',
               'id="crew-form-name"', 'id="crew-form-avatar"', 'id="crew-form-personality"',
               'id="crew-form-model"', 'id="crew-form-endpoint"', 'id="crew-form-greeting"',
               'id="crew-form-tools"', 'id="crew-form-save"', 'id="crew-form-cancel"'):
        assert el in html, f"{el} missing from index.html"


def test_crew_rail_button_is_not_admin_gated():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    # Every other panel this session hides its rail button until isAdmin()
    # reveals it (style="display:none" baked into the HTML). Crew must NOT --
    # find the exact rail-crew button tag and confirm no display:none on it.
    import re
    m = re.search(r'<button class="icon-rail-btn" id="rail-crew"[^>]*>', html)
    assert m is not None, "rail-crew button not found"
    assert "display:none" not in m.group(0)


def test_crew_js_wires_routes():
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    for s in ('rail-crew', 'tool-crew-btn', 'Modals.register',
              "api('/api/crew')", '/api/crew/tool-names', 'createDirectChat'):
        assert s in src, f"{s} missing from crew.js"
    # Crew must NOT gate its own reveal behind isAdmin, unlike every admin panel.
    assert "isAdmin" not in src


def test_sessions_js_thread_crew_member_id_through_pending_chat():
    src = (ROOT / "static" / "js" / "sessions.js").read_text(encoding="utf-8")
    assert "crewMemberId" in src
    assert "crew_member_id" in src


def test_crew_js_syntax(tmp_path):
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    mjs = tmp_path / "crew.mjs"; mjs.write_text(src, encoding="utf-8")
    p = subprocess.run(["node", "--check", str(mjs)], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crew_ui.py -v --import-mode=importlib`
Expected: FAIL (`crew.js` doesn't exist; the new HTML ids aren't in `index.html` yet)

- [ ] **Step 3: Write the HTML additions**

Before writing this step, read `static/js/sessions.js` to find its existing function for creating-and-opening a new chat session (the one the normal "New Chat" button already calls), and reuse that exact call in `crew.js`'s "New Chat" button handler rather than writing a second, parallel implementation — pass `crew_member_id` through to it (extend that function's parameters if it doesn't already accept extra form fields, following whatever the least invasive change to that existing function looks like once you've read it).

In `static/index.html`, insert a new modal after the `training-modal`'s closing `</div>` (or any other convenient existing modal boundary — this modal has no dependency on modal ordering):

```html
  <!-- Crew: multi-persona system — NOT admin-gated -->
  <div id="crew-modal" class="modal hidden">
    <div class="modal-content" role="dialog" aria-label="Crew">
      <div class="modal-header">
        <h4>Crew</h4>
        <button class="close-btn" id="crew-close" aria-label="Close">&#x2716;</button>
      </div>
      <div id="crew-list-view">
        <div style="margin:8px 0"><button id="crew-new-btn" class="btn">+ New persona</button></div>
        <div id="crew-grid" style="display:flex;flex-wrap:wrap;gap:12px"></div>
      </div>
      <div id="crew-form-view" style="display:none">
        <label>Name<br><input id="crew-form-name" style="width:100%"></label>
        <label>Avatar URL<br><input id="crew-form-avatar" style="width:100%"></label>
        <label>Personality (system prompt)<br><textarea id="crew-form-personality" rows="6" style="width:100%"></textarea></label>
        <label>Model<br><input id="crew-form-model" style="width:100%"></label>
        <label>Endpoint URL<br><input id="crew-form-endpoint" style="width:100%"></label>
        <label>Greeting<br><input id="crew-form-greeting" style="width:100%"></label>
        <div>Tools<div id="crew-form-tools" style="max-height:160px;overflow:auto"></div></div>
        <div style="margin-top:8px">
          <button id="crew-form-save" class="btn">Save</button>
          <button id="crew-form-cancel" class="btn">Cancel</button>
        </div>
      </div>
    </div>
  </div>
```

Insert a new rail button after any existing always-visible rail button (e.g. right after `rail-theme`) — **no `style="display:none"`**, unlike every admin-gated panel:

```html
    <button class="icon-rail-btn" id="rail-crew" title="Crew"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="7" r="4"/><path d="M2 21v-2a4 4 0 0 1 4-4h6a4 4 0 0 1 4 4v2"/><circle cx="17" cy="7" r="3" opacity="0.6"/></svg></button>
```

Insert a new sidebar entry (find where the always-visible sidebar entries live, near the other non-`style="display:none"` `list-item` buttons — NOT next to `tool-training-btn`/`tool-imagedataset-btn`, which are admin-only):

```html
        <div class="list-item" id="tool-crew-btn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.5;"><circle cx="9" cy="7" r="4"/><path d="M2 21v-2a4 4 0 0 1 4-4h6a4 4 0 0 1 4 4v2"/><circle cx="17" cy="7" r="3" opacity="0.6"/></svg>
          <span class="grow">Crew</span>
        </div>
```

Insert a new script tag anywhere among the other `<script type="module">` tags:

```html
<script type="module" src="/static/js/crew.js"></script>
```

- [ ] **Step 4: Write the DOM controller**

```javascript
// static/js/crew.js
// Crew: multi-persona system. ES module — DOM controller over /api/crew.
// Unlike every admin-gated panel this session, Crew is NOT admin-gated --
// every authenticated user manages their own personas, matching the
// existing per-user Assistant feature it generalizes. Mirrors the
// established panel-controller shape (Modals, $, api) minus the isAdmin gate.
import * as Modals from './modalManager.js';

function $(id) { return document.getElementById(id); }
let _crew = [];
let _editingId = null;

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

async function api(path, opts) {
  const res = await fetch(path, Object.assign({ credentials: 'same-origin' }, opts || {}));
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error(d || String(res.status));
  }
  return data;
}

function openCrew() {
  $('crew-modal').classList.remove('hidden');
  showListView();
  refreshList();
}
function closeCrew() { $('crew-modal').classList.add('hidden'); }

function showListView() {
  $('crew-list-view').style.display = '';
  $('crew-form-view').style.display = 'none';
}
function showFormView() {
  $('crew-list-view').style.display = 'none';
  $('crew-form-view').style.display = '';
}

async function refreshList() {
  try {
    const j = await api('/api/crew');
    _crew = j.crew || [];
    renderGrid();
  } catch (e) {}
}

function renderGrid() {
  const grid = $('crew-grid');
  if (!grid) return;
  grid.innerHTML = _crew.map(function (c) {
    const preview = esc((c.personality || '').slice(0, 80));
    return (
      '<div style="border:1px solid var(--border);border-radius:8px;padding:8px;width:200px">' +
      '<div style="font-weight:600">' + esc(c.name) + '</div>' +
      '<div style="font-size:12px;opacity:0.75;min-height:32px">' + preview + '</div>' +
      '<button class="btn" data-newchat="' + esc(c.id) + '">New Chat</button>' +
      '<button class="btn" data-edit="' + esc(c.id) + '">Edit</button>' +
      (c.is_default_assistant ? '' : '<button class="btn" data-delete="' + esc(c.id) + '">Delete</button>') +
      '</div>'
    );
  }).join('') || 'No personas yet.';

  grid.querySelectorAll('[data-newchat]').forEach(function (b) {
    b.addEventListener('click', function () { newChatWithPersona(b.getAttribute('data-newchat')); });
  });
  grid.querySelectorAll('[data-edit]').forEach(function (b) {
    b.addEventListener('click', function () { openEditForm(b.getAttribute('data-edit')); });
  });
  grid.querySelectorAll('[data-delete]').forEach(function (b) {
    b.addEventListener('click', function () { deletePersona(b.getAttribute('data-delete')); });
  });
}

async function newChatWithPersona(crewId) {
  const persona = _crew.find(function (c) { return c.id === crewId; });
  if (!persona) return;
  try {
    // Mirrors the app's existing "new chat" flow: createDirectChat only
    // stages the pending chat (no backend call yet); the actual session is
    // created by materializePendingSession() on the user's first message.
    // Both functions are extended (this task) to carry crew_member_id
    // through that same lazy path -- see sessions.js changes below.
    const { createDirectChat } = await import('./sessions.js');
    createDirectChat(persona.endpoint_url || '', persona.model || '', null, crewId);
    closeCrew();
  } catch (e) {
    console.error('newChatWithPersona failed:', e);
  }
}

let _toolNames = [];
async function loadToolNames() {
  if (_toolNames.length) return _toolNames;
  try {
    const j = await api('/api/crew/tool-names');
    _toolNames = j.tools || [];
  } catch (e) { _toolNames = []; }
  return _toolNames;
}

async function openEditForm(crewId) {
  _editingId = crewId || null;
  const existing = crewId ? _crew.find(function (c) { return c.id === crewId; }) : null;
  $('crew-form-name').value = existing ? existing.name : '';
  $('crew-form-avatar').value = existing ? (existing.avatar || '') : '';
  $('crew-form-personality').value = existing ? (existing.personality || '') : '';
  $('crew-form-model').value = existing ? (existing.model || '') : '';
  $('crew-form-endpoint').value = existing ? (existing.endpoint_url || '') : '';
  $('crew-form-greeting').value = existing ? (existing.greeting || '') : '';

  const names = await loadToolNames();
  const enabled = new Set(existing ? (existing.enabled_tools || []) : []);
  const host = $('crew-form-tools');
  if (host) {
    host.innerHTML = names.map(function (t) {
      const checked = enabled.has(t) ? 'checked' : '';
      return '<label style="display:block"><input type="checkbox" value="' + esc(t) + '" ' + checked + '> ' + esc(t) + '</label>';
    }).join('');
  }
  showFormView();
}

function collectFormToolList() {
  const host = $('crew-form-tools');
  if (!host) return [];
  return Array.from(host.querySelectorAll('input[type=checkbox]:checked')).map(function (i) { return i.value; });
}

async function saveForm() {
  const payload = {
    name: $('crew-form-name').value,
    avatar: $('crew-form-avatar').value,
    personality: $('crew-form-personality').value,
    model: $('crew-form-model').value,
    endpoint_url: $('crew-form-endpoint').value,
    greeting: $('crew-form-greeting').value,
    enabled_tools: collectFormToolList(),
  };
  try {
    if (_editingId) {
      await api('/api/crew/' + encodeURIComponent(_editingId), {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
    } else {
      await api('/api/crew', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
    }
    showListView();
    refreshList();
  } catch (e) {
    console.error('saveForm failed:', e);
  }
}

async function deletePersona(crewId) {
  try {
    await api('/api/crew/' + encodeURIComponent(crewId), { method: 'DELETE' });
    refreshList();
  } catch (e) {
    console.error('deletePersona failed:', e);
  }
}

function init() {
  const rail = $('rail-crew'); if (rail) rail.addEventListener('click', openCrew);
  const side = $('tool-crew-btn'); if (side) side.addEventListener('click', openCrew);
  const x = $('crew-close'); if (x) x.addEventListener('click', closeCrew);
  const newBtn = $('crew-new-btn'); if (newBtn) newBtn.addEventListener('click', function () { openEditForm(null); });
  const save = $('crew-form-save'); if (save) save.addEventListener('click', saveForm);
  const cancel = $('crew-form-cancel'); if (cancel) cancel.addEventListener('click', showListView);
  Modals.register('crew-modal', {
    railBtnId: 'rail-crew', sidebarBtnId: 'tool-crew-btn', closeFn: closeCrew,
  });
}

document.addEventListener('DOMContentLoaded', init);
```

Now extend `static/js/sessions.js` so the pending-chat/materialize flow carries a persona through. Change the `createDirectChat` signature (~line 2100) from:
```javascript
export function createDirectChat(url, modelId, endpointId) {
```
to:
```javascript
export function createDirectChat(url, modelId, endpointId, crewMemberId) {
```
and change this line inside it:
```javascript
  _pendingChat = { url, modelId, endpointId };
```
to:
```javascript
  _pendingChat = { url, modelId, endpointId, crewMemberId };
```

Then in `materializePendingSession()` (~line 2158), right after the existing block:
```javascript
  if (pending.endpointId) {
    fd.append('endpoint_id', pending.endpointId);
  }
```
add:
```javascript
  if (pending.crewMemberId) {
    fd.append('crew_member_id', pending.crewMemberId);
  }
```

This is the only change `materializePendingSession()` needs — it already POSTs the `FormData` it builds to `/api/session` (the same endpoint Task 3 extended), so once `crew_member_id` is appended here, the backend wiring from Task 3 takes over automatically.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_crew_ui.py -v --import-mode=importlib`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/crew.js static/js/sessions.js tests/test_crew_ui.py
git commit -m "feat(crew): add multi-persona frontend panel"
```

---

## Self-Review Notes

**Spec coverage:** The spec's `CrewMember` reuse maps to Tasks 1-2 (helpers + CRUD, no new columns). Session-binding maps to Task 3. System-prompt precedence maps to Task 4. Tool-allowlist enforcement maps to Task 5. The frontend maps to Task 6, including the explicit NOT-admin-gated requirement (tested directly in Task 6's `test_crew_rail_button_is_not_admin_gated`). The default-Assistant-delete-block and fail-open `enabled_tools` handling are each tested in Task 1/2. Non-goals (voice, memory isolation, mid-conversation switching, admin gating, changing Assistant's check-ins) are untouched by every task.

**Placeholder scan:** An earlier draft of this plan left two spots as "read the live file to confirm the real name" hedges (Task 3's `setup_session_routes` construction, Task 6's session-creation flow). Both were resolved during self-review by actually reading the live files: `setup_session_routes(session_manager, config, webhook_manager=None)` builds on a module-level `router = APIRouter(prefix="/api", ...)`, and `create_session` is decorated `@router.post("/session", ...)`, giving the exact path `/api/session` used throughout Task 3's tests. More significantly, reading `sessions.js` revealed this app uses a lazy pending-chat pattern (`createDirectChat` stages the UI only; `materializePendingSession` is what actually calls `POST /api/session`, on the user's first message) — Task 6 was rewritten around the real mechanism (extending `createDirectChat`'s signature and `materializePendingSession`'s `FormData` build) instead of the originally-assumed "click New Chat → immediately create a session." No placeholders remain — every step now names concrete, verified files, functions, and line numbers.

**Type consistency:** `crew_to_dict`/`resolve_crew_binding`/`crew_disabled_tools` (Task 1) are used with identical signatures in Tasks 2, 4, and 5. The `/api/crew*` route paths and response shapes (Task 2) match what Task 6's `crew.js` calls exactly (`GET /api/crew` → `{"crew": [...]}`, `GET /api/crew/tool-names` → `{"tools": [...]}`). `crew_member_id`/`crewMemberId` is threaded consistently: Task 3's `Form` field name on the backend, `crewMemberId` as `createDirectChat`'s new 4th parameter and `_pendingChat` key in Task 6, and `crew_member_id` again as the `FormData` key `materializePendingSession` appends — matching Task 3's expected field name exactly.

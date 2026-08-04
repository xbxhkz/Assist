import uuid
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import core.database as cdb
import core.session_manager as sm
import routes.session_routes as sr

# Dedicated in-memory test DB (mirrors tests/test_crew_routes.py's pattern).
# The app's default core.database.engine uses SingletonThreadPool, which is
# per-thread: FastAPI's TestClient runs sync route handlers in a worker
# thread via anyio's threadpool, a DIFFERENT thread than the one that ran
# core.database's module-level init_db()/create_all() on import. That worker
# thread would get a brand-new, empty ":memory:" connection ("no such table:
# sessions"). StaticPool shares a single connection across all threads,
# avoiding that split-brain DB. SessionLocal must be patched in every module
# that captured the original binding via `from core.database import
# SessionLocal` at import time (routes.session_routes, core.session_manager);
# core.database.SessionLocal itself is patched too since this test's helpers
# re-import it fresh at call time.
_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
cdb.Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)


@pytest.fixture(autouse=True)
def _use_test_db(monkeypatch):
    """Redirect every module's SessionLocal to the dedicated test DB above.

    Must be autouse (not folded into _client()) because each test's first
    line is `crew_id = _make_crew()`, called BEFORE `_client(monkeypatch)` -
    patching only inside _client() would run too late and _make_crew() would
    write into the real default (SingletonThreadPool) DB instead of this
    StaticPool one. Autouse fixtures resolve before any test body code runs,
    so this is in effect for _make_crew()'s call too.
    """
    monkeypatch.setattr(cdb, "SessionLocal", _TS)
    monkeypatch.setattr(sm, "SessionLocal", _TS)
    monkeypatch.setattr(sr, "SessionLocal", _TS)


def _make_crew(owner="alice", model="gpt-x", endpoint_url=None, endpoint_id=None):
    from core.database import SessionLocal, CrewMember
    db = SessionLocal()
    try:
        c = CrewMember(id=str(uuid.uuid4()), owner=owner, name="Nav",
                       model=model, endpoint_url=endpoint_url, endpoint_id=endpoint_id)
        db.add(c)
        db.commit()
        return c.id
    finally:
        db.close()


def _make_endpoint(owner="alice", is_enabled=True, base_url="http://persona-endpoint/v1"):
    from core.database import SessionLocal, ModelEndpoint
    db = SessionLocal()
    try:
        ep = ModelEndpoint(id=str(uuid.uuid4()), name="Persona EP", base_url=base_url,
                            owner=owner, is_enabled=is_enabled)
        db.add(ep)
        db.commit()
        return ep.id
    finally:
        db.close()


class _AllowAllAdmin:
    """Minimal auth_manager stub so the pre-existing, crew-unrelated
    "raw endpoint_url requires a registered endpoint for non-admins" guard
    (_reject_raw_endpoint_url_for_non_admin in routes/session_routes.py)
    doesn't 403 test_create_session_explicit_model_overrides_persona_default,
    which posts a raw endpoint_url with no endpoint_id. That guard predates
    and is unrelated to this task's crew binding; treating the test user as
    admin exercises it the same way the real admin path would.
    """
    def is_admin(self, user):
        return True


class _DenyAllAdmin:
    """Auth_manager stub reporting every user as non-admin -- used by the
    Critical-3 regression tests below to prove a non-admin never picks up a
    persona's raw endpoint_url."""
    def is_admin(self, user):
        return False


def _client(monkeypatch, admin=True):
    monkeypatch.setattr(sr, "effective_user", lambda request: "alice")
    # routes/session_routes.py declares `router` as a module-level singleton
    # that setup_session_routes() re-decorates onto on every call, without
    # ever clearing it - harmless in production (the real app calls it
    # exactly once at startup) but not test-isolated: other test files in
    # this suite also call setup_session_routes() (some with a MagicMock
    # session_manager), and those stale closures accumulate on the same
    # router forever. Starlette matches the FIRST-registered handler for a
    # given path, so a live TestClient POST to "/api/session" run after one
    # of those other test files would silently hit a leftover MagicMock
    # closure instead of this test's own (confirmed: this exact
    # cross-file-order fragility already exists on clean dev HEAD between
    # test_archived_sessions_model_filter.py and
    # test_session_list_owner_scope.py, unrelated to this task).
    # Swap in a fresh, empty router just for this test rather than mutating
    # the shared one in place: setup_session_routes()'s `@router...`
    # decorators resolve `router` as a module-global by name at call time,
    # so pointing sr.router at a new APIRouter here redirects them, and
    # monkeypatch restores the original shared object (with whatever other
    # tests already put on it, untouched) as soon as this test ends. That
    # avoids polluting - or being polluted by - any other test file in
    # either direction.
    monkeypatch.setattr(sr, "router", APIRouter(prefix="/api", tags=["sessions"]))
    app = FastAPI()
    app.state.auth_manager = _AllowAllAdmin() if admin else _DenyAllAdmin()
    app.include_router(sr.setup_session_routes(sm.SessionManager(), {}, webhook_manager=None))
    return TestClient(app)


def test_create_session_with_crew_member_id_defaults_model_and_endpoint(monkeypatch):
    from core.database import SessionLocal, Session as DbSession
    eid = _make_endpoint(owner="alice", base_url="http://persona-endpoint/v1")
    crew_id = _make_crew(endpoint_id=eid)
    client = _client(monkeypatch)
    resp = client.post("/api/session", data={"crew_member_id": crew_id, "skip_validation": "true"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "gpt-x"
    db = SessionLocal()
    try:
        row = db.query(DbSession).filter(DbSession.id == body["id"]).first()
        assert row.crew_member_id == crew_id
        assert "persona-endpoint" in (row.endpoint_url or "")
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


def test_create_session_non_admin_persona_raw_endpoint_url_not_applied(monkeypatch):
    """Critical 3 regression: a persona with a raw endpoint_url and no
    endpoint_id, created by a non-admin, must NOT have that raw URL applied
    to a session created by a non-admin -- the app's own registered-endpoint
    guard exists specifically to prevent a client from dialing an arbitrary
    host, and the crew branch used to inject the raw URL *after* that guard
    had already run against the client's (empty) fields. With no endpoint_id
    and no other endpoint_url supplied, this must surface as the normal
    "endpoint_url is required" 400, not a silently-wrong-but-200 session."""
    # Deliberately NOT passing skip_validation=true here: with it set, the
    # "endpoint_url is required" check is bypassed entirely regardless of
    # what the crew branch resolves, which would mask the very regression
    # this test exists to catch.
    crew_id = _make_crew(endpoint_url="http://attacker.example/v1")
    client = _client(monkeypatch, admin=False)
    resp = client.post("/api/session", data={"crew_member_id": crew_id})
    assert resp.status_code == 400
    assert "attacker" not in resp.text


def test_create_session_admin_persona_raw_endpoint_url_still_applies(monkeypatch):
    """Don't regress the admin/Assistant path: an admin's own persona with a
    raw endpoint_url (e.g. the Assistant's endpoint_url set via the separate
    assistant_routes.py path) must keep working exactly as before."""
    from core.database import SessionLocal, Session as DbSession
    crew_id = _make_crew(endpoint_url="http://persona-endpoint/v1")
    client = _client(monkeypatch, admin=True)
    resp = client.post("/api/session", data={"crew_member_id": crew_id, "skip_validation": "true"})
    assert resp.status_code == 200
    body = resp.json()
    db = SessionLocal()
    try:
        row = db.query(DbSession).filter(DbSession.id == body["id"]).first()
        assert row.endpoint_url == "http://persona-endpoint/v1"
    finally:
        db.close()

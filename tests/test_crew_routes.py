from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.crew_routes as cr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from core.database import Base, CrewMember, ModelEndpoint
import uuid

# Set up test database at module level
_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)
cr.SessionLocal = _TS


import pytest


@pytest.fixture(autouse=True)
def clear_db():
    """Clear the crew_members and model_endpoints tables before each test."""
    db = _TS()
    try:
        db.query(CrewMember).delete()
        db.query(ModelEndpoint).delete()
        db.commit()
    finally:
        db.close()
    yield


def _make_endpoint(owner="alice", is_enabled=True):
    eid = str(uuid.uuid4())
    db = _TS()
    try:
        db.add(ModelEndpoint(
            id=eid, name="Local", base_url="http://localhost:8002/v1",
            owner=owner, is_enabled=is_enabled,
        ))
        db.commit()
    finally:
        db.close()
    return eid


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
    crew_id = str(uuid.uuid4())
    db = _TS()
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
    # Capture the expected tools before client creation triggers additional imports
    expected_tools = sorted(known_tool_names())
    c = _client(monkeypatch)
    r = c.get("/api/crew/tool-names")
    assert r.status_code == 200
    tools = r.json()["tools"]
    # The endpoint calls known_tool_names() fresh, which may load additional modules
    # So we compare against what the endpoint would have seen (call it again after app init)
    assert tools == sorted(known_tool_names())


def test_update_sort_order_rejects_non_numeric(monkeypatch):
    c = _client(monkeypatch)
    created = c.post("/api/crew", json={"name": "Nav"}).json()
    r = c.patch(f"/api/crew/{created['id']}", json={"sort_order": "abc"})
    assert r.status_code == 400


def test_update_avatar_coerces_non_string_instead_of_crashing(monkeypatch):
    c = _client(monkeypatch)
    created = c.post("/api/crew", json={"name": "Nav"}).json()
    r = c.patch(f"/api/crew/{created['id']}", json={"avatar": {"nested": "object"}})
    assert r.status_code == 200
    assert isinstance(r.json()["avatar"], str)


def test_create_coerces_non_string_fields_instead_of_crashing(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/crew", json={"name": "Nav", "personality": {"a": 1}, "avatar": 42})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["personality"], str)
    assert isinstance(body["avatar"], str)


def test_create_no_longer_accepts_raw_endpoint_url(monkeypatch):
    """The general Crew CRUD stops persisting a client-supplied raw endpoint_url
    (Critical 2/3 fix) -- the column stays None even if the client sends it."""
    c = _client(monkeypatch)
    r = c.post("/api/crew", json={"name": "Nav", "endpoint_url": "http://attacker.example/v1"})
    assert r.status_code == 200
    body = r.json()
    assert body["endpoint_url"] is None
    assert body.get("endpoint_id") is None


def test_create_with_valid_owner_scoped_endpoint_id_succeeds(monkeypatch):
    eid = _make_endpoint(owner="alice")
    c = _client(monkeypatch, user="alice")
    r = c.post("/api/crew", json={"name": "Nav", "endpoint_id": eid})
    assert r.status_code == 200
    assert r.json()["endpoint_id"] == eid


def test_create_with_shared_endpoint_id_succeeds(monkeypatch):
    """owner=None ModelEndpoint rows are shared/visible to every user."""
    eid = _make_endpoint(owner=None)
    c = _client(monkeypatch, user="alice")
    r = c.post("/api/crew", json={"name": "Nav", "endpoint_id": eid})
    assert r.status_code == 200
    assert r.json()["endpoint_id"] == eid


def test_create_with_nonexistent_endpoint_id_400s(monkeypatch):
    c = _client(monkeypatch, user="alice")
    r = c.post("/api/crew", json={"name": "Nav", "endpoint_id": "no-such-endpoint"})
    assert r.status_code == 400


def test_create_with_another_owners_endpoint_id_400s(monkeypatch):
    eid = _make_endpoint(owner="bob")
    c = _client(monkeypatch, user="alice")
    r = c.post("/api/crew", json={"name": "Nav", "endpoint_id": eid})
    assert r.status_code == 400


def test_create_with_disabled_endpoint_id_400s(monkeypatch):
    eid = _make_endpoint(owner="alice", is_enabled=False)
    c = _client(monkeypatch, user="alice")
    r = c.post("/api/crew", json={"name": "Nav", "endpoint_id": eid})
    assert r.status_code == 400


def test_update_with_valid_owner_scoped_endpoint_id_succeeds(monkeypatch):
    eid = _make_endpoint(owner="alice")
    c = _client(monkeypatch, user="alice")
    created = c.post("/api/crew", json={"name": "Nav"}).json()
    r = c.patch(f"/api/crew/{created['id']}", json={"endpoint_id": eid})
    assert r.status_code == 200
    assert r.json()["endpoint_id"] == eid


def test_update_with_another_owners_endpoint_id_400s(monkeypatch):
    eid = _make_endpoint(owner="bob")
    c = _client(monkeypatch, user="alice")
    created = c.post("/api/crew", json={"name": "Nav"}).json()
    r = c.patch(f"/api/crew/{created['id']}", json={"endpoint_id": eid})
    assert r.status_code == 400


def test_update_no_longer_accepts_raw_endpoint_url(monkeypatch):
    c = _client(monkeypatch, user="alice")
    created = c.post("/api/crew", json={"name": "Nav"}).json()
    r = c.patch(f"/api/crew/{created['id']}", json={"endpoint_url": "http://attacker.example/v1"})
    assert r.status_code == 200
    assert r.json()["endpoint_url"] is None


def test_create_with_non_list_enabled_tools_400s_and_does_not_commit(monkeypatch):
    """Finding 9 write-side validation: a non-list enabled_tools (e.g. the
    bare int 5, which json.dumps'd and reloaded is still not a list) must be
    rejected with 400 rather than persisted -- persisting it would poison
    every subsequent GET /api/crew for this owner (crew_to_dict crash),
    since db.commit() happens before crew_to_dict(c) is called in the
    create path."""
    c = _client(monkeypatch, user="alice")
    r = c.post("/api/crew", json={"name": "x", "enabled_tools": 5})
    assert r.status_code == 400
    # The row must not have been committed -- the list stays empty.
    assert c.get("/api/crew").json()["crew"] == []


def test_update_with_non_list_enabled_tools_400s_and_does_not_commit(monkeypatch):
    c = _client(monkeypatch, user="alice")
    created = c.post("/api/crew", json={"name": "Nav"}).json()
    r = c.patch(f"/api/crew/{created['id']}", json={"enabled_tools": 5})
    assert r.status_code == 400
    # The existing row must be unaffected -- enabled_tools stays whatever it was before.
    after = c.get("/api/crew").json()["crew"]
    assert len(after) == 1
    assert after[0]["enabled_tools"] == []


def test_create_with_list_of_unhashable_elements_enabled_tools_400s_and_does_not_commit(monkeypatch):
    """Fix-wave-3 Finding 1: a list VALUE whose elements aren't strings
    (e.g. [["web_search"]] or [{"a": 1}]) still passes a bare
    isinstance(enabled_tools, list) check -- it's the elements that break
    `any(t in _EMAIL_TOOLS for t in tools)` with an unhashable-type
    TypeError. Must 400 and never commit the row (a committed poisoned row
    would 500 every subsequent GET /api/crew for this owner, forever)."""
    c = _client(monkeypatch, user="alice")
    for bad_tools in ([["web_search"]], [{"a": 1}]):
        r = c.post("/api/crew", json={"name": "x", "enabled_tools": bad_tools})
        assert r.status_code == 400
        # The row must not have been committed.
        assert c.get("/api/crew").json()["crew"] == []
        # And the owner's list endpoint must still work -- no poisoned row.
        r2 = c.get("/api/crew")
        assert r2.status_code == 200


def test_update_with_list_of_unhashable_elements_enabled_tools_400s_and_does_not_commit(monkeypatch):
    c = _client(monkeypatch, user="alice")
    created = c.post("/api/crew", json={"name": "Nav"}).json()
    for bad_tools in ([["web_search"]], [{"a": 1}]):
        r = c.patch(f"/api/crew/{created['id']}", json={"enabled_tools": bad_tools})
        assert r.status_code == 400
        # The existing row must be unaffected -- enabled_tools stays empty.
        after = c.get("/api/crew").json()["crew"]
        assert len(after) == 1
        assert after[0]["enabled_tools"] == []
        # And the owner's list endpoint must still work -- no poisoned row.
        r2 = c.get("/api/crew")
        assert r2.status_code == 200


def test_update_endpoint_id_to_null_clears_it(monkeypatch):
    eid = _make_endpoint(owner="alice")
    c = _client(monkeypatch, user="alice")
    created = c.post("/api/crew", json={"name": "Nav", "endpoint_id": eid}).json()
    assert created["endpoint_id"] == eid
    r = c.patch(f"/api/crew/{created['id']}", json={"endpoint_id": None})
    assert r.status_code == 200
    assert r.json()["endpoint_id"] is None


def test_create_persona_with_tts_voice(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/crew", json={"name": "Nav", "tts_voice": "nova"})
    assert r.status_code == 200
    assert r.json()["tts_voice"] == "nova"


def test_create_persona_without_tts_voice_defaults_to_none(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/crew", json={"name": "Nav"})
    assert r.status_code == 200
    assert r.json()["tts_voice"] is None


def test_update_persona_tts_voice(monkeypatch):
    c = _client(monkeypatch)
    created = c.post("/api/crew", json={"name": "Nav"}).json()
    r = c.patch(f"/api/crew/{created['id']}", json={"tts_voice": "af_heart"})
    assert r.status_code == 200
    assert r.json()["tts_voice"] == "af_heart"


def test_update_persona_tts_voice_coerces_non_string_instead_of_crashing(monkeypatch):
    c = _client(monkeypatch)
    created = c.post("/api/crew", json={"name": "Nav"}).json()
    r = c.patch(f"/api/crew/{created['id']}", json={"tts_voice": 12345})
    assert r.status_code == 200
    assert isinstance(r.json()["tts_voice"], str)

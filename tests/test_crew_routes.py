from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.crew_routes as cr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from core.database import Base, CrewMember
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
    """Clear the crew_members table before each test."""
    db = _TS()
    try:
        db.query(CrewMember).delete()
        db.commit()
    finally:
        db.close()
    yield


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

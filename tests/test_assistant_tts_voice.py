import uuid
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes.assistant_routes as ar
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from core.database import Base, CrewMember
import pytest

_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)
ar.SessionLocal = _TS


@pytest.fixture(autouse=True)
def clear_db():
    db = _TS()
    try:
        db.query(CrewMember).delete()
        db.commit()
    finally:
        db.close()
    yield


class _StubTaskScheduler:
    """Never actually called: every test pre-seeds the default-assistant row,
    so _get_or_create's lazy-seed branch (the only caller of this) never
    runs. Exists only so setup_assistant_routes has something to hold."""
    async def ensure_assistant_defaults(self, owner):
        raise AssertionError("should not be called — test pre-seeds the Assistant row")


def _seed_assistant(owner="alice", tts_voice=None):
    db = _TS()
    try:
        crew_id = str(uuid.uuid4())
        db.add(CrewMember(id=crew_id, owner=owner, name="Assistant",
                          is_default_assistant=True, tts_voice=tts_voice))
        db.commit()
        return crew_id
    finally:
        db.close()


def _client(monkeypatch, user="alice"):
    monkeypatch.setattr(ar, "get_current_user", lambda request: user)
    app = FastAPI()
    app.include_router(ar.setup_assistant_routes(_StubTaskScheduler()))
    return TestClient(app)


def test_get_assistant_settings_returns_tts_voice(monkeypatch):
    _seed_assistant(tts_voice="nova")
    c = _client(monkeypatch)
    r = c.get("/api/assistant/settings")
    assert r.status_code == 200
    assert r.json()["crew"]["tts_voice"] == "nova"


def test_update_assistant_settings_accepts_tts_voice(monkeypatch):
    _seed_assistant(tts_voice=None)
    c = _client(monkeypatch)
    r = c.patch("/api/assistant/settings", json={"tts_voice": "af_heart"})
    assert r.status_code == 200
    assert r.json()["crew"]["tts_voice"] == "af_heart"


def test_update_assistant_settings_tts_voice_absent_leaves_it_unchanged(monkeypatch):
    _seed_assistant(tts_voice="nova")
    c = _client(monkeypatch)
    r = c.patch("/api/assistant/settings", json={"name": "Assistant"})
    assert r.status_code == 200
    assert r.json()["crew"]["tts_voice"] == "nova"


def test_update_assistant_settings_tts_voice_empty_string_clears_it(monkeypatch):
    _seed_assistant(tts_voice="nova")
    c = _client(monkeypatch)
    r = c.patch("/api/assistant/settings", json={"tts_voice": ""})
    assert r.status_code == 200
    assert r.json()["crew"]["tts_voice"] is None

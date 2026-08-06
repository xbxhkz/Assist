"""HTTP-level tests proving routes/tts_routes.py actually wires a session's
resolved persona voice into TTSService, not just that the resolution
helper and the service both work in isolation (both already covered by
tests/test_tts_voice_override.py). A fake TTSService records what
voice_override it was actually called with on each request."""
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import routes.tts_routes as tr
from core.database import Base, CrewMember, Session as DbSession

_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)


import pytest


@pytest.fixture(autouse=True)
def clear_db():
    db = _TS()
    try:
        db.query(CrewMember).delete()
        db.query(DbSession).delete()
        db.commit()
    finally:
        db.close()
    yield


class _RecordingTTSService:
    available = True

    def __init__(self):
        self.synthesize_calls = []
        self.get_stats_calls = []

    def synthesize(self, text, use_cache=True, voice_override=None):
        self.synthesize_calls.append(voice_override)
        return b"fake-audio-bytes"

    def get_stats(self, voice_override=None):
        self.get_stats_calls.append(voice_override)
        return {"available": True, "ready": True, "provider": "local",
                "voice": voice_override or "alloy"}

    def synthesize_to_base64(self, text):
        return "ZmFrZQ=="


def _make_bound_session(owner="alice", tts_voice="nova"):
    db = _TS()
    try:
        crew_id = str(uuid.uuid4())
        db.add(CrewMember(id=crew_id, owner=owner, name="Nav", tts_voice=tts_voice))
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m",
                         owner=owner, crew_member_id=crew_id))
        db.commit()
        return sess_id
    finally:
        db.close()


def _client(monkeypatch, svc):
    import core.database as coredb
    monkeypatch.setattr(coredb, "SessionLocal", _TS)
    monkeypatch.setattr(tr, "effective_user", lambda request: "alice")
    app = FastAPI()
    app.include_router(tr.setup_tts_routes(svc))
    return TestClient(app)


def test_synthesize_route_resolves_persona_voice_from_session_id(monkeypatch):
    sess_id = _make_bound_session(tts_voice="nova")
    svc = _RecordingTTSService()
    c = _client(monkeypatch, svc)
    r = c.post("/api/tts/synthesize", json={"text": "hi", "format": "audio", "session_id": sess_id})
    assert r.status_code == 200
    assert svc.synthesize_calls == ["nova"]


def test_synthesize_route_without_session_id_uses_no_override(monkeypatch):
    svc = _RecordingTTSService()
    c = _client(monkeypatch, svc)
    r = c.post("/api/tts/synthesize", json={"text": "hi", "format": "audio"})
    assert r.status_code == 200
    assert svc.synthesize_calls == [None]


def test_synthesize_route_dangling_session_id_uses_no_override(monkeypatch):
    svc = _RecordingTTSService()
    c = _client(monkeypatch, svc)
    r = c.post("/api/tts/synthesize", json={"text": "hi", "format": "audio", "session_id": "does-not-exist"})
    assert r.status_code == 200
    assert svc.synthesize_calls == [None]


def test_stats_route_resolves_persona_voice_from_session_id(monkeypatch):
    sess_id = _make_bound_session(tts_voice="nova")
    svc = _RecordingTTSService()
    c = _client(monkeypatch, svc)
    r = c.get(f"/api/tts/stats?session_id={sess_id}")
    assert r.status_code == 200
    assert r.json()["voice"] == "nova"
    assert svc.get_stats_calls == ["nova"]


def test_stats_route_without_session_id_uses_no_override(monkeypatch):
    svc = _RecordingTTSService()
    c = _client(monkeypatch, svc)
    r = c.get("/api/tts/stats")
    assert r.status_code == 200
    assert svc.get_stats_calls == [None]

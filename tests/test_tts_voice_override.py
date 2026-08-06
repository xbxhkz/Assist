"""Session-bound persona voice override for TTS synthesis. Mirrors the
fail-open pattern already established for personality (chat_helpers.py)
and tool policy (crew_helpers.py's crew_disabled_tools): no binding,
a dangling binding, or an empty tts_voice must all silently fall back
to the existing global voice, never raise or surface an error."""
import uuid


def _make_bound_session(owner="alice", tts_voice="nova"):
    from core.database import SessionLocal, CrewMember, Session as DbSession
    db = SessionLocal()
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


def test_synthesize_uses_voice_override_when_given(monkeypatch):
    from services.tts.tts_service import TTSService
    svc = TTSService(cache_dir="__test_tts_cache__")
    calls = []
    monkeypatch.setattr(svc, "_load_settings", lambda: {
        "tts_enabled": True, "tts_provider": "local", "tts_model": "tts-1",
        "tts_voice": "alloy", "tts_speed": "1",
    })
    monkeypatch.setattr(svc, "_get_kokoro", lambda: type("K", (), {
        "available": True,
        "synthesize_raw": staticmethod(lambda text, voice: calls.append(voice) or b"fake-audio"),
    })())
    svc.synthesize("hello", use_cache=False, voice_override="nova")
    assert calls == ["nova"]


def test_synthesize_falls_back_to_global_voice_when_no_override(monkeypatch):
    from services.tts.tts_service import TTSService
    svc = TTSService(cache_dir="__test_tts_cache__")
    calls = []
    monkeypatch.setattr(svc, "_load_settings", lambda: {
        "tts_enabled": True, "tts_provider": "local", "tts_model": "tts-1",
        "tts_voice": "alloy", "tts_speed": "1",
    })
    monkeypatch.setattr(svc, "_get_kokoro", lambda: type("K", (), {
        "available": True,
        "synthesize_raw": staticmethod(lambda text, voice: calls.append(voice) or b"fake-audio"),
    })())
    svc.synthesize("hello", use_cache=False, voice_override=None)
    assert calls == ["alloy"]


def test_get_stats_reflects_voice_override(monkeypatch):
    from services.tts.tts_service import TTSService
    svc = TTSService(cache_dir="__test_tts_cache__")
    monkeypatch.setattr(svc, "_load_settings", lambda: {
        "tts_enabled": True, "tts_provider": "browser", "tts_model": "tts-1",
        "tts_voice": "alloy", "tts_speed": "1",
    })
    stats = svc.get_stats(voice_override="nova")
    assert stats["voice"] == "nova"


def test_get_stats_without_override_uses_global_voice(monkeypatch):
    from services.tts.tts_service import TTSService
    svc = TTSService(cache_dir="__test_tts_cache__")
    monkeypatch.setattr(svc, "_load_settings", lambda: {
        "tts_enabled": True, "tts_provider": "browser", "tts_model": "tts-1",
        "tts_voice": "alloy", "tts_speed": "1",
    })
    stats = svc.get_stats(voice_override=None)
    assert stats["voice"] == "alloy"


def test_resolve_effective_voice_uses_persona_binding(monkeypatch):
    import routes.tts_routes as tr
    from core.database import SessionLocal
    sess_id = _make_bound_session(tts_voice="nova")
    db = SessionLocal()
    try:
        voice = tr._resolve_effective_voice(db, sess_id, "alice")
    finally:
        db.close()
    assert voice == "nova"


def test_resolve_effective_voice_none_for_unbound_session(monkeypatch):
    import routes.tts_routes as tr
    from core.database import SessionLocal, Session as DbSession
    db = SessionLocal()
    try:
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m", owner="alice"))
        db.commit()
        voice = tr._resolve_effective_voice(db, sess_id, "alice")
    finally:
        db.close()
    assert voice is None


def test_resolve_effective_voice_none_for_empty_persona_voice(monkeypatch):
    import routes.tts_routes as tr
    from core.database import SessionLocal
    sess_id = _make_bound_session(tts_voice=None)
    db = SessionLocal()
    try:
        voice = tr._resolve_effective_voice(db, sess_id, "alice")
    finally:
        db.close()
    assert voice is None


def test_resolve_effective_voice_none_on_dangling_session(monkeypatch):
    import routes.tts_routes as tr
    from core.database import SessionLocal
    db = SessionLocal()
    try:
        voice = tr._resolve_effective_voice(db, "does-not-exist", "alice")
    finally:
        db.close()
    assert voice is None

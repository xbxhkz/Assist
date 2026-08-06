# Multi-Persona System: Voice Per Persona Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A chat session bound to a Crew persona speaks its replies (manual play, auto-play, and hands-free Voice Conversation alike) in that persona's own TTS voice if it has one set, across all three TTS providers (local Kokoro, OpenAI-compatible API endpoint, browser Web Speech API), falling back to the current global voice everywhere else.

**Architecture:** Add a nullable `CrewMember.tts_voice` column. Voice resolution happens at the route layer (`routes/tts_routes.py`), not inside `TTSService` — a small helper resolves a session's bound persona via the already-shared `resolve_crew_binding`, then passes a plain `voice_override: str | None` down into `TTSService.synthesize()`/`get_stats()`. On the frontend, `tts-ai.js` becomes session-aware by importing `sessions.js`'s existing `getCurrentSessionId()` — no other frontend call site (the auto-play queue, streaming sentence chunking, the manual play button) needs to change, since they all funnel through the same two leaf methods (`synthesize()` for server-synthesized audio, `_playBrowser()`/`_findBrowserVoice()` for the browser provider) that become session-aware internally.

**Tech Stack:** FastAPI + SQLAlchemy (backend), vanilla ES modules (frontend), pytest with `--import-mode=importlib`.

## Global Constraints

- No new database columns beyond the one nullable `tts_voice` string on `CrewMember` — same idempotent `ALTER TABLE` migration pattern as `CrewMember.endpoint_id` (`core/database.py:1612`, registered in `init_db()` at `core/database.py:1877`).
- Voice resolution must be fail-open, matching every other persona-override feature in this codebase: no session/persona binding, a dangling binding, an empty `tts_voice`, or any lookup error must silently fall back to today's exact global-voice behavior — never an error surfaced to a TTS caller.
- `resolve_crew_binding(db, session_id, owner)` (`src/crew_helpers.py:46`) already re-scopes to the requesting owner internally — reuse it as-is, do not reimplement owner scoping.
- Any new free-text field written through `routes/crew_routes.py`'s `create_crew`/`update_crew` must use the same `_s()`-style "non-`None`, non-`str` → `str(x)`" coercion every other free-text field on that model already gets (`avatar`/`personality`/`model`/`greeting`) — never accept a non-string value uncoerced.
- No validation of a voice name against a fixed list or a registered resource — a voice is an opaque string handed to whichever provider is configured, exactly like the existing global `tts_voice` setting.
- Per-task tests use real DB rows via the ORM (no mocks) for backend fail-open behavior, matching every prior Crew task's test style.
- Commit directly to `dev` (this project's established convention — no feature branch). Stage specific files, never `git add -A`. Do not stage `installer/Output/Assist-Setup.exe`. Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. pytest needs `--import-mode=importlib`.

---

### Task 1: `CrewMember.tts_voice` column + shared dict

**Files:**
- Modify: `core/database.py:557-558` (add column), `core/database.py:1838` area (add migration function), `core/database.py:1877` (register migration)
- Modify: `src/crew_helpers.py:26-43` (`crew_to_dict`)
- Test: `tests/test_crew_helpers.py`

**Interfaces:**
- Produces: `CrewMember.tts_voice` (nullable `String` column). `crew_to_dict(c)` includes `"tts_voice": c.tts_voice` in its output dict — the shared shape every later task's frontend code reads.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_crew_helpers.py`:

```python
def test_crew_to_dict_includes_tts_voice():
    import uuid
    from core.database import SessionLocal, CrewMember
    db = SessionLocal()
    try:
        c = CrewMember(id=str(uuid.uuid4()), owner="alice", name="Nav", tts_voice="nova")
        db.add(c)
        db.commit()
        from src.crew_helpers import crew_to_dict
        d = crew_to_dict(c)
        assert d["tts_voice"] == "nova"
    finally:
        db.close()


def test_crew_to_dict_tts_voice_defaults_to_none():
    import uuid
    from core.database import SessionLocal, CrewMember
    db = SessionLocal()
    try:
        c = CrewMember(id=str(uuid.uuid4()), owner="alice", name="Nav")
        db.add(c)
        db.commit()
        from src.crew_helpers import crew_to_dict
        d = crew_to_dict(c)
        assert d["tts_voice"] is None
    finally:
        db.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_crew_helpers.py -k tts_voice -v --import-mode=importlib`
Expected: FAIL — `TypeError: 'tts_voice' is an invalid keyword argument for CrewMember` (column doesn't exist yet).

- [ ] **Step 3: Add the column**

In `core/database.py`, the `CrewMember` class currently reads (around line 546-566):

```python
class CrewMember(TimestampMixin, Base):
    """A custom AI persona ('crew member') with its own personality, model, tools, and memory scope."""
    __tablename__ = "crew_members"

    id            = Column(String, primary_key=True, index=True)
    owner         = Column(String, nullable=True, index=True)
    name          = Column(String, nullable=False)
    avatar        = Column(String, nullable=True)
    user_name     = Column(String, nullable=True)          # what they call the user
    personality   = Column(Text, nullable=True)             # system prompt
    model         = Column(String, nullable=True)
    endpoint_url  = Column(String, nullable=True)
    endpoint_id   = Column(String, nullable=True)          # registered ModelEndpoint.id (non-admin personas bind here, not to a raw URL)
    greeting      = Column(Text, nullable=True)
    enabled_tools = Column(Text, nullable=True)             # JSON array or "all"
    session_id    = Column(String, ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True)
    is_active     = Column(Boolean, default=True)
    sort_order    = Column(Integer, default=0)
    is_default_assistant = Column(Boolean, default=False)   # singleton per-owner "personal assistant"
    timezone      = Column(String, nullable=True)           # IANA tz name (e.g. "America/New_York") for scheduled check-ins
```

Add one line right after `greeting`:

```python
    greeting      = Column(Text, nullable=True)
    tts_voice     = Column(String, nullable=True)           # provider-specific TTS voice name override; None = use the global default
    enabled_tools = Column(Text, nullable=True)             # JSON array or "all"
```

- [ ] **Step 4: Add the migration function**

Find `_migrate_add_crew_endpoint_id_column` (`core/database.py:1612`) — copy its exact shape for the new column:

```python
def _migrate_add_crew_tts_voice_column():
    """Add tts_voice column to crew_members table if it doesn't exist."""
    import sqlite3
    db_path = DATABASE_URL.replace("sqlite:///", "")
    if not os.path.exists(db_path):
        return
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(crew_members)")
        columns = [row[1] for row in cursor.fetchall()]
        if "tts_voice" not in columns:
            conn.execute("ALTER TABLE crew_members ADD COLUMN tts_voice TEXT")
            conn.commit()
            logging.getLogger(__name__).info("Migrated: added 'tts_voice' column to crew_members")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Migration check failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
```

Place it directly after `_migrate_add_crew_endpoint_id_column`'s definition.

- [ ] **Step 5: Register the migration**

In `init_db()` (`core/database.py:1877`), change:

```python
    _migrate_add_crew_endpoint_id_column()
```

to:

```python
    _migrate_add_crew_endpoint_id_column()
    _migrate_add_crew_tts_voice_column()
```

- [ ] **Step 6: Update `crew_to_dict`**

In `src/crew_helpers.py`, the return dict currently reads (lines 26-43):

```python
    return {
        "id": c.id,
        "name": c.name,
        "avatar": c.avatar,
        "personality": c.personality,
        "model": c.model,
        "endpoint_url": c.endpoint_url,
        "endpoint_id": c.endpoint_id,
        "greeting": c.greeting,
        "enabled_tools": tools,
        ...
```

Add `tts_voice` right after `greeting`:

```python
        "greeting": c.greeting,
        "tts_voice": c.tts_voice,
        "enabled_tools": tools,
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_crew_helpers.py -v --import-mode=importlib`
Expected: PASS (all tests in the file, not just the two new ones — confirms no existing `crew_to_dict` test asserted an exact/frozen dict shape that would break from the new key).

- [ ] **Step 8: Commit**

```bash
git add core/database.py src/crew_helpers.py tests/test_crew_helpers.py
git commit -m "feat(crew): add CrewMember.tts_voice column

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `routes/crew_routes.py` accepts `tts_voice`

**Files:**
- Modify: `routes/crew_routes.py:42-87` (`create_crew`), `routes/crew_routes.py:94-141` (`update_crew`)
- Test: `tests/test_crew_routes.py`

**Interfaces:**
- Consumes: `crew_to_dict` (Task 1) already returns `tts_voice`.
- Produces: `POST /api/crew` and `PATCH /api/crew/{id}` both accept an optional `tts_voice` string field in the request body, coerced non-string→`str(x)` like every sibling free-text field.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_crew_routes.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_crew_routes.py -k tts_voice -v --import-mode=importlib`
Expected: FAIL — `tts_voice` is silently dropped (not in the constructor / never assigned), so the first two tests fail on `assert r.json()["tts_voice"] == "nova"` (KeyError or None-vs-"nova" mismatch) and the update tests fail because the field is never touched by `update_crew`.

- [ ] **Step 3: Wire `create_crew`**

`routes/crew_routes.py`'s `create_crew` currently builds the `CrewMember` like this (lines 68-82):

```python
            def _s(key):
                v = body.get(key)
                return v if v is None or isinstance(v, str) else str(v)

            c = CrewMember(
                id=str(uuid.uuid4()),
                owner=owner,
                name=name,
                avatar=_s("avatar"),
                personality=_s("personality"),
                model=_s("model"),
                endpoint_id=endpoint_id,
                greeting=_s("greeting"),
                enabled_tools=json.dumps(enabled_tools) if enabled_tools is not None else None,
            )
```

Add `tts_voice=_s("tts_voice")` right after `greeting`:

```python
                greeting=_s("greeting"),
                tts_voice=_s("tts_voice"),
                enabled_tools=json.dumps(enabled_tools) if enabled_tools is not None else None,
```

- [ ] **Step 4: Wire `update_crew`**

`routes/crew_routes.py`'s `update_crew` currently has (lines 124-125):

```python
            if "greeting" in body:
                c.greeting = body["greeting"] if body["greeting"] is None or isinstance(body["greeting"], str) else str(body["greeting"])
```

Add the same pattern for `tts_voice` right after:

```python
            if "greeting" in body:
                c.greeting = body["greeting"] if body["greeting"] is None or isinstance(body["greeting"], str) else str(body["greeting"])
            if "tts_voice" in body:
                c.tts_voice = body["tts_voice"] if body["tts_voice"] is None or isinstance(body["tts_voice"], str) else str(body["tts_voice"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_crew_routes.py -v --import-mode=importlib`
Expected: PASS (full file — confirms no regression in the other CRUD tests).

- [ ] **Step 6: Commit**

```bash
git add routes/crew_routes.py tests/test_crew_routes.py
git commit -m "feat(crew): accept tts_voice on persona create/update

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: TTS voice resolution — backend

**Files:**
- Modify: `services/tts/tts_service.py` (`synthesize`, `_cache_key`, `get_stats`)
- Modify: `routes/tts_routes.py` (both routes + a new resolution helper)
- Test: `tests/test_tts_voice_override.py` (new file), `tests/test_tts_routes.py` if it exists (check first — if no such file exists, create route-level tests inline in the new test file instead)

**Interfaces:**
- Consumes: `resolve_crew_binding(db, session_id, owner)` (`src/crew_helpers.py:46`, unchanged), `effective_user(request)` (`src.auth_helpers`, unchanged).
- Produces: `TTSService.synthesize(text, use_cache=True, voice_override: str | None = None)`, `TTSService.get_stats(voice_override: str | None = None) -> dict`. `POST /api/tts/synthesize` accepts an optional `session_id` field on `TTSRequest`. `GET /api/tts/stats` accepts an optional `session_id` query param. Later tasks (frontend) pass `session_id` — never a raw voice string — into these two routes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tts_voice_override.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tts_voice_override.py -v --import-mode=importlib`
Expected: FAIL — `synthesize()`/`get_stats()` don't accept `voice_override` yet (`TypeError: unexpected keyword argument`), and `routes.tts_routes` has no `_resolve_effective_voice` function yet (`AttributeError`).

- [ ] **Step 3: Add `voice_override` to `TTSService`**

In `services/tts/tts_service.py`, `_cache_key` (line 77-79) is unchanged — it already takes `voice` as a plain parameter, so no change needed there; the override just changes what value gets passed in as `voice`.

`synthesize()` currently reads (lines 146-153):

```python
    def synthesize(self, text: str, use_cache: bool = True) -> Optional[bytes]:
        settings = self._load_settings()
        if settings.get("tts_enabled") is False:
            return None
        provider = settings["tts_provider"]
        model = settings["tts_model"]
        voice = settings["tts_voice"]
        speed = _safe_speed(settings.get("tts_speed", "1"))
```

Change to:

```python
    def synthesize(self, text: str, use_cache: bool = True, voice_override: Optional[str] = None) -> Optional[bytes]:
        settings = self._load_settings()
        if settings.get("tts_enabled") is False:
            return None
        provider = settings["tts_provider"]
        model = settings["tts_model"]
        voice = voice_override or settings["tts_voice"]
        speed = _safe_speed(settings.get("tts_speed", "1"))
```

(Everything below this in `synthesize()` — caching, provider dispatch — already reads from the local `voice` variable, so no further change is needed inside the method body.)

`get_stats()` currently reads (lines 200-218):

```python
    def get_stats(self) -> Dict[str, Any]:
        settings = self._load_settings()
        provider = settings["tts_provider"]
        tts_enabled = settings.get("tts_enabled", True)

        cache_files = list(self.cache_dir.glob("*.wav")) + list(self.cache_dir.glob("*.mp3"))
        cache_size = sum(f.stat().st_size for f in cache_files)

        is_available = self.available and tts_enabled
        stats = {
            "available": is_available,
            "ready": is_available,
            "provider": provider,
            "model": settings["tts_model"],
            "voice": settings["tts_voice"],
            "speed": _safe_speed(settings.get("tts_speed", "1")),
            "cache_entries": len(cache_files),
            "cache_size_mb": round(cache_size / (1024 * 1024), 2),
        }
```

Change to:

```python
    def get_stats(self, voice_override: Optional[str] = None) -> Dict[str, Any]:
        settings = self._load_settings()
        provider = settings["tts_provider"]
        tts_enabled = settings.get("tts_enabled", True)

        cache_files = list(self.cache_dir.glob("*.wav")) + list(self.cache_dir.glob("*.mp3"))
        cache_size = sum(f.stat().st_size for f in cache_files)

        is_available = self.available and tts_enabled
        stats = {
            "available": is_available,
            "ready": is_available,
            "provider": provider,
            "model": settings["tts_model"],
            "voice": voice_override or settings["tts_voice"],
            "speed": _safe_speed(settings.get("tts_speed", "1")),
            "cache_entries": len(cache_files),
            "cache_size_mb": round(cache_size / (1024 * 1024), 2),
        }
```

- [ ] **Step 4: Add the resolution helper and wire both routes**

`routes/tts_routes.py` currently has no `Request`/auth imports at all. Change the top of the file from:

```python
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    format: str = "audio"  # "audio" or "base64"

def setup_tts_routes(tts_service):
    """Setup TTS routes with the provided TTS service"""
    router = APIRouter(prefix="/api/tts", tags=["tts"])

    @router.get("/stats")
    async def get_tts_stats():
        """Get TTS service statistics"""
        try:
            return tts_service.get_stats()
        except Exception as e:
            logger.error(f"Failed to get TTS stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/synthesize")
    async def synthesize_speech(request: TTSRequest):
        """Synthesize speech from text"""
        try:
            if not tts_service.available:
                raise HTTPException(
                    status_code=503,
                    detail={"message": "TTS service not available"}
                )

            if request.format == "base64":
                audio_b64 = tts_service.synthesize_to_base64(request.text)
```

to:

```python
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
import logging

from src.auth_helpers import effective_user

logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    format: str = "audio"  # "audio" or "base64"
    session_id: str | None = None


def _resolve_effective_voice(db, session_id: str | None, owner: str | None) -> str | None:
    """Return the tts_voice of the persona bound to session_id, or None if
    there's no session_id/owner, no binding, a dangling binding, or an
    empty override -- fail-open, never raises. Caller owns closing db."""
    if not session_id or not owner:
        return None
    try:
        from src.crew_helpers import resolve_crew_binding
        crew = resolve_crew_binding(db, session_id, owner)
        return crew.tts_voice if crew and crew.tts_voice else None
    except Exception:
        return None


def setup_tts_routes(tts_service):
    """Setup TTS routes with the provided TTS service"""
    router = APIRouter(prefix="/api/tts", tags=["tts"])

    @router.get("/stats")
    async def get_tts_stats(request: Request, session_id: str | None = None):
        """Get TTS service statistics"""
        try:
            voice_override = None
            if session_id:
                from core.database import SessionLocal
                owner = effective_user(request)
                db = SessionLocal()
                try:
                    voice_override = _resolve_effective_voice(db, session_id, owner)
                finally:
                    db.close()
            return tts_service.get_stats(voice_override=voice_override)
        except Exception as e:
            logger.error(f"Failed to get TTS stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/synthesize")
    async def synthesize_speech(request_body: TTSRequest, request: Request):
        """Synthesize speech from text"""
        try:
            if not tts_service.available:
                raise HTTPException(
                    status_code=503,
                    detail={"message": "TTS service not available"}
                )

            voice_override = None
            if request_body.session_id:
                from core.database import SessionLocal
                owner = effective_user(request)
                db = SessionLocal()
                try:
                    voice_override = _resolve_effective_voice(db, request_body.session_id, owner)
                finally:
                    db.close()

            if request_body.format == "base64":
                audio_b64 = tts_service.synthesize_to_base64(request_body.text)
```

Note the parameter rename `request` → `request_body` for the `TTSRequest` body (freeing up `request` for the FastAPI `Request` object, needed for `effective_user`). Every remaining reference to the old `request.text`/`request.format` further down in `synthesize_speech` must be updated to `request_body.text`/`request_body.format` — read the rest of the function (below what's quoted above) and rename them; do not leave stale `request.text` references that would now refer to the wrong object.

Also update the two `tts_service.synthesize(...)` call sites further down in `synthesize_speech` (one in the `base64` branch, one in the `audio` branch) to pass `voice_override=voice_override`:

```python
                audio_b64 = tts_service.synthesize_to_base64(request_body.text)
```

`synthesize_to_base64` doesn't take a voice override in this plan (it's used only by `settings.js`'s admin preview button and `slashCommands.js`, neither of which is session-scoped) — leave it as-is. The `audio` branch's call:

```python
                audio_data = tts_service.synthesize(request_body.text)
```

becomes:

```python
                audio_data = tts_service.synthesize(request_body.text, voice_override=voice_override)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_tts_voice_override.py -v --import-mode=importlib`
Expected: PASS.

Then run the broader existing TTS test files to confirm no regression:

Run: `pytest -k "tts" --import-mode=importlib -v`
Expected: PASS (all pre-existing TTS tests, e.g. `test_tts_speed_malformed.py`, `test_speech_service_toggles.py`, still pass unchanged).

- [ ] **Step 6: Commit**

```bash
git add services/tts/tts_service.py routes/tts_routes.py tests/test_tts_voice_override.py
git commit -m "feat(tts): resolve a session's bound persona voice as an override

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Frontend — `tts-ai.js` becomes session-aware

**Files:**
- Modify: `static/js/tts-ai.js`
- Test: `tests/test_tts_ai_session_voice.py` (new file, source-presence + node-subprocess style matching this codebase's established pattern for `sessions.js`/`crew.js` JS-logic tests)

**Interfaces:**
- Consumes: `getCurrentSessionId()` (exported from `static/js/sessions.js:2224`, already used elsewhere e.g. `modelPicker.js`'s dependency injection). `POST /api/tts/synthesize`'s new optional `session_id` field, `GET /api/tts/stats`'s new optional `session_id` query param (Task 3).
- Produces: `AITTSManager.synthesize(text)` now includes the current session id in its POST body automatically. Browser-provider voice selection re-resolves per-utterance instead of relying on the page-load-cached `browserVoice`. No other method's public signature changes — `play()`, `enqueue()`, `streamingStart/Update/End`, and every `chat.js` call site are untouched.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tts_ai_session_voice.py`:

```python
"""tts-ai.js becomes session-aware without changing any of its callers'
call sites (play/enqueue/streaming*) -- synthesize() and the browser-voice
path both resolve the current session internally via sessions.js's
getCurrentSessionId(), matching how modelPicker.js already consumes it."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_tts_ai_imports_get_current_session_id():
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")
    assert "getCurrentSessionId" in src
    assert re.search(r"import\s*\{[^}]*getCurrentSessionId[^}]*\}\s*from\s*['\"]\./sessions\.js['\"]", src)


def test_synthesize_sends_session_id_in_request_body():
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")
    m = re.search(r"async synthesize\([^)]*\)\s*\{.*?\n    \}", src, re.S)
    assert m is not None, "synthesize method not found"
    assert "session_id" in m.group(0)


def test_resolve_browser_voice_uses_session_scoped_stats():
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")
    assert "_resolveBrowserVoice" in src
    m = re.search(r"_resolveBrowserVoice\([^)]*\)\s*\{.*?\n    \}", src, re.S)
    assert m is not None, "_resolveBrowserVoice method not found"
    assert "session_id" in m.group(0) or "getCurrentSessionId" in m.group(0)


def test_playbrowser_is_async_and_awaits_voice_resolution():
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")
    m = re.search(r"async _playBrowser\([^)]*\)\s*\{.*?\n    \}", src, re.S)
    assert m is not None, "_playBrowser must be declared async"
    assert "_resolveBrowserVoice" in m.group(0)


def test_get_cache_key_includes_session_id():
    src = (ROOT / "static" / "js" / "tts-ai.js").read_text(encoding="utf-8")
    m = re.search(r"getCacheKey\([^)]*\)\s*\{.*?\n    \}", src, re.S)
    assert m is not None, "getCacheKey method not found"
    assert "sessionId" in m.group(0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tts_ai_session_voice.py -v --import-mode=importlib`
Expected: FAIL — none of the referenced identifiers (`getCurrentSessionId` import, `_resolveBrowserVoice`, session-aware `getCacheKey`) exist yet.

- [ ] **Step 3: Import `getCurrentSessionId`**

`static/js/tts-ai.js` currently starts with (lines 1-3):

```javascript
// static/js/tts-ai.js
// AI Text-to-Speech Module — supports server TTS and browser Web Speech API

class AITTSManager {
```

Change to:

```javascript
// static/js/tts-ai.js
// AI Text-to-Speech Module — supports server TTS and browser Web Speech API
import { getCurrentSessionId } from './sessions.js';

class AITTSManager {
```

- [ ] **Step 4: Make `getCacheKey` session-scoped**

Current (lines 95-104):

```javascript
    getCacheKey(text) {
        // Simple hash function for cache key
        let hash = 0;
        for (let i = 0; i < text.length; i++) {
            const char = text.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return hash.toString(36);
    }
```

This is hashed on `text` alone today, which is only safe while there's a single global voice. Once personas can have distinct voices, the same short reply text spoken by two different personas must not collide in the client-side cache. Change to:

```javascript
    getCacheKey(text, sessionId) {
        // Simple hash function for cache key — includes sessionId so two
        // personas with different voices don't collide on the same short
        // reply text in the client-side cache.
        const raw = text + '|' + (sessionId || '');
        let hash = 0;
        for (let i = 0; i < raw.length; i++) {
            const char = raw.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return hash.toString(36);
    }
```

- [ ] **Step 5: Thread session id through `synthesize()`**

Current (lines 106-162, showing the relevant parts):

```javascript
    async synthesize(text, onProgress = null) {
        if (!this.available) {
            throw new Error('AI TTS service not available');
        }

        const plainText = this.extractPlainText(text);

        if (!plainText) {
            throw new Error('No text to synthesize');
        }

        // Browser TTS doesn't use synthesize — handled directly in play()
        if (this.useBrowserTTS) {
            return '__browser_tts__';
        }

        const cacheKey = this.getCacheKey(plainText);

        // Check cache first
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        try {
            if (onProgress) onProgress('synthesizing');

            const response = await fetch('/api/tts/synthesize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: plainText,
                    format: 'audio'
                })
            });
```

Change to:

```javascript
    async synthesize(text, onProgress = null) {
        if (!this.available) {
            throw new Error('AI TTS service not available');
        }

        const plainText = this.extractPlainText(text);

        if (!plainText) {
            throw new Error('No text to synthesize');
        }

        // Browser TTS doesn't use synthesize — handled directly in play()
        if (this.useBrowserTTS) {
            return '__browser_tts__';
        }

        const sessionId = getCurrentSessionId();
        const cacheKey = this.getCacheKey(plainText, sessionId);

        // Check cache first
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        try {
            if (onProgress) onProgress('synthesizing');

            const response = await fetch('/api/tts/synthesize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: plainText,
                    format: 'audio',
                    session_id: sessionId || undefined
                })
            });
```

(The `.set(cacheKey, audioUrl)` call further down in the same method already uses the `cacheKey` local variable — no change needed there, it automatically picks up the new session-scoped key.)

- [ ] **Step 6: Add `_resolveBrowserVoice` and make `_playBrowser` async**

Current `_findBrowserVoice`/`_playBrowser` (lines 164-219):

```javascript
    _findBrowserVoice() {
        if (!this.browserVoice) return null;
        const voices = window.speechSynthesis.getVoices();
        const target = this.browserVoice.toLowerCase();
        // Try exact match first, then partial
        return voices.find(v => v.name.toLowerCase() === target) ||
               voices.find(v => v.name.toLowerCase().includes(target)) ||
               null;
    }

    async play(text) {
        // Stop current audio if playing
        this.stop();

        const plainText = this.extractPlainText(text);
        if (!plainText) return;

        if (this.useBrowserTTS) {
            return this._playBrowser(plainText);
        }

        try {
            const audioUrl = await this.synthesize(text);

            this.currentAudio = new Audio(audioUrl);
            await this.currentAudio.play();
            this.isPlaying = true;
            // Note: onended should be set by the caller (addAITTSButton)
            // to reset button state when audio finishes

        } catch (error) {
            console.error('Failed to play audio:', error);
            throw error;
        }
    }

    _playBrowser(plainText) {
        return new Promise((resolve, reject) => {
            const utterance = new SpeechSynthesisUtterance(plainText);
            const voice = this._findBrowserVoice();
            if (voice) utterance.voice = voice;
            utterance.rate = this.playbackSpeed;

            utterance.onend = () => {
                this.isPlaying = false;
                resolve();
            };
            utterance.onerror = (e) => {
                this.isPlaying = false;
                reject(new Error('Browser TTS error: ' + e.error));
            };

            window.speechSynthesis.speak(utterance);
            this.isPlaying = true;
        });
    }
```

Change to (add `_resolveBrowserVoice` as a new method right before `_findBrowserVoice`, change `_findBrowserVoice` to accept an explicit voice name instead of always reading `this.browserVoice`, and make `_playBrowser` async):

```javascript
    /** Resolve the voice NAME to use for the browser Web Speech API for the
     * CURRENT session — re-fetches per session (not per utterance) since a
     * page can have multiple chats open across a browsing session, each
     * possibly bound to a different persona. Falls back to the page-load
     * global `browserVoice` on any failure or when there's no active
     * session. */
    async _resolveBrowserVoice() {
        const sessionId = getCurrentSessionId();
        if (!sessionId) return this.browserVoice;
        if (this._voiceCacheSessionId === sessionId) return this._voiceCacheValue;
        try {
            const res = await fetch(`/api/tts/stats?session_id=${encodeURIComponent(sessionId)}`, { credentials: 'same-origin' });
            const stats = await res.json();
            const voice = stats.voice || this.browserVoice;
            this._voiceCacheSessionId = sessionId;
            this._voiceCacheValue = voice;
            return voice;
        } catch (e) {
            return this.browserVoice;
        }
    }

    _findBrowserVoice(voiceName) {
        if (!voiceName) return null;
        const voices = window.speechSynthesis.getVoices();
        const target = voiceName.toLowerCase();
        // Try exact match first, then partial
        return voices.find(v => v.name.toLowerCase() === target) ||
               voices.find(v => v.name.toLowerCase().includes(target)) ||
               null;
    }

    async play(text) {
        // Stop current audio if playing
        this.stop();

        const plainText = this.extractPlainText(text);
        if (!plainText) return;

        if (this.useBrowserTTS) {
            return this._playBrowser(plainText);
        }

        try {
            const audioUrl = await this.synthesize(text);

            this.currentAudio = new Audio(audioUrl);
            await this.currentAudio.play();
            this.isPlaying = true;
            // Note: onended should be set by the caller (addAITTSButton)
            // to reset button state when audio finishes

        } catch (error) {
            console.error('Failed to play audio:', error);
            throw error;
        }
    }

    async _playBrowser(plainText) {
        const voiceName = await this._resolveBrowserVoice();
        return new Promise((resolve, reject) => {
            const utterance = new SpeechSynthesisUtterance(plainText);
            const voice = this._findBrowserVoice(voiceName);
            if (voice) utterance.voice = voice;
            utterance.rate = this.playbackSpeed;

            utterance.onend = () => {
                this.isPlaying = false;
                resolve();
            };
            utterance.onerror = (e) => {
                this.isPlaying = false;
                reject(new Error('Browser TTS error: ' + e.error));
            };

            window.speechSynthesis.speak(utterance);
            this.isPlaying = true;
        });
    }
```

Also add the two new cache fields to the constructor. Current constructor (lines 4-29, relevant excerpt):

```javascript
        this.browserVoice = '';
        this.playbackSpeed = 1;
```

Change to:

```javascript
        this.browserVoice = '';
        this._voiceCacheSessionId = null;
        this._voiceCacheValue = null;
        this.playbackSpeed = 1;
```

`play()` and `_playQueueItem()` already do `await this._playBrowser(plainText)` / `await this._playBrowser(plainText)` respectively (both already treat it as a Promise) — no change needed at either call site now that `_playBrowser` is `async` (an `async` function still returns a Promise that `await` handles identically).

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_tts_ai_session_voice.py -v --import-mode=importlib`
Expected: PASS.

Then syntax-check the file (this codebase's established gate for ES module edits):

Run: `node --check static/js/tts-ai.js`
Expected: no output (valid syntax). Note `tts-ai.js` uses `import`/`export` (ES module syntax) — if `node --check` errors on the bare `import` statement outside a module context, copy it to a temp `.mjs` file first and check that instead (matching how `workflows.js`'s DOM-module tests handle this, per the Assist workflow editor's established pattern) rather than treating a module-syntax error as a real bug.

- [ ] **Step 8: Commit**

```bash
git add static/js/tts-ai.js tests/test_tts_ai_session_voice.py
git commit -m "feat(tts): tts-ai.js resolves voice for the active session

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Crew panel voice picker

**Files:**
- Modify: `static/index.html:645-654` (crew form)
- Modify: `static/js/crew.js`
- Test: `tests/test_crew_ui.py`

**Interfaces:**
- Consumes: `crew_to_dict`'s `tts_voice` key (Task 1), `POST/PATCH /api/crew`'s `tts_voice` field (Task 2), `GET /api/tts/stats` (existing, for reading the current global `provider`).
- Produces: the persona edit form in the Crew panel includes a voice field, saved as `tts_voice` on `saveForm`'s payload.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_crew_ui.py`:

```python
def test_crew_modal_has_voice_field():
    src = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="crew-form-voice-select"' in src
    assert 'id="crew-form-voice-input"' in src


def test_crew_js_loads_tts_provider_for_voice_picker():
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    assert "/api/tts/stats" in src


def test_crew_js_save_form_includes_tts_voice():
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    m = re.search(r'async function saveForm\(\)\s*\{.*?\n\}', src, re.S)
    assert m is not None, "saveForm function not found"
    assert "tts_voice" in m.group(0)


def test_crew_js_open_edit_form_populates_voice_field():
    src = (ROOT / "static" / "js" / "crew.js").read_text(encoding="utf-8")
    m = re.search(r'async function openEditForm\([^)]*\)\s*\{.*?\n\}', src, re.S)
    assert m is not None, "openEditForm function not found"
    assert "tts_voice" in m.group(0)
```

(`ROOT` and `re` are already imported at the top of `tests/test_crew_ui.py` — confirm before assuming; read the file's existing imports first.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crew_ui.py -k voice -v --import-mode=importlib`
Expected: FAIL — none of the new ids/strings exist yet.

- [ ] **Step 3: Add the HTML fields**

`static/index.html`'s crew form currently reads (lines 645-654):

```html
      <div id="crew-form-view" style="display:none">
        <label>Name<br><input id="crew-form-name" style="width:100%"></label>
        <label>Avatar URL<br><input id="crew-form-avatar" style="width:100%"></label>
        <label>Personality (system prompt)<br><textarea id="crew-form-personality" rows="6" style="width:100%"></textarea></label>
        <label>Model / Endpoint<br><select id="crew-form-endpoint" style="width:100%"><option value="">(use default — no override)</option></select>
        <label>Greeting<br><input id="crew-form-greeting" style="width:100%"></label>
        <div>Tools<div id="crew-form-tools" style="max-height:160px;overflow:auto"></div></div>
        <div style="margin-top:8px">
          <button id="crew-form-save" class="btn">Save</button>
```

Insert a voice field between the endpoint picker and Greeting, mirroring `#set-ttsVoiceSelect`/`#set-ttsVoiceInput`'s dual-mode shape from the global TTS settings (`static/index.html:2247-2258`):

```html
        <label>Model / Endpoint<br><select id="crew-form-endpoint" style="width:100%"><option value="">(use default — no override)</option></select>
        <label>Voice<br>
          <select id="crew-form-voice-select" style="width:100%;display:none">
            <option value="">(use default — no override)</option>
            <option value="alloy">Alloy</option>
            <option value="ash">Ash</option>
            <option value="coral">Coral</option>
            <option value="echo">Echo</option>
            <option value="fable">Fable</option>
            <option value="nova">Nova</option>
            <option value="onyx">Onyx</option>
            <option value="sage">Sage</option>
            <option value="shimmer">Shimmer</option>
          </select>
          <input id="crew-form-voice-input" type="text" placeholder="af_heart (leave blank to use default)" style="width:100%">
        </label>
        <label>Greeting<br><input id="crew-form-greeting" style="width:100%"></label>
```

- [ ] **Step 4: Wire `crew.js`**

Read the current `openEditForm` and `saveForm` in full before editing (they've been modified by three prior fix waves — do not assume the exact line numbers from any earlier version of this file). Add a small helper near the top of the file (after `loadEndpointOptions`, before `openEditForm`) that fetches the current global TTS provider once and decides which voice input mode to show:

```javascript
let _ttsProviderIsEndpoint = null;
async function loadTtsProviderMode() {
  if (_ttsProviderIsEndpoint !== null) return _ttsProviderIsEndpoint;
  try {
    const stats = await api('/api/tts/stats');
    _ttsProviderIsEndpoint = !!(stats.provider && stats.provider.startsWith('endpoint:'));
  } catch (e) { _ttsProviderIsEndpoint = false; }
  return _ttsProviderIsEndpoint;
}

function _getVoiceFormValue() {
  const sel = $('crew-form-voice-select');
  const inp = $('crew-form-voice-input');
  if (sel && sel.style.display !== 'none') return sel.value;
  return inp ? inp.value : '';
}

function _setVoiceFormValue(value) {
  const sel = $('crew-form-voice-select');
  const inp = $('crew-form-voice-input');
  if (sel) sel.value = value || '';
  if (inp) inp.value = value || '';
}
```

In `openEditForm`, after the existing endpoint-select population block and before the tool-checklist rendering, add:

```javascript
  const isEndpointProvider = await loadTtsProviderMode();
  const voiceSel = $('crew-form-voice-select');
  const voiceInp = $('crew-form-voice-input');
  if (voiceSel && voiceInp) {
    voiceSel.style.display = isEndpointProvider ? '' : 'none';
    voiceInp.style.display = isEndpointProvider ? 'none' : '';
  }
  _setVoiceFormValue(existing ? existing.tts_voice : '');
```

In `saveForm`, the payload construction currently ends with (after Task 5's own earlier changes to this function from the sub-project 1 fix waves — read the live file, this is the LOGICAL insertion point regardless of exact surrounding line numbers):

```javascript
  const payload = {
    name: $('crew-form-name').value,
    avatar: $('crew-form-avatar').value,
    personality: $('crew-form-personality').value,
    greeting: $('crew-form-greeting').value,
  };
```

Add `tts_voice`:

```javascript
  const payload = {
    name: $('crew-form-name').value,
    avatar: $('crew-form-avatar').value,
    personality: $('crew-form-personality').value,
    greeting: $('crew-form-greeting').value,
    tts_voice: _getVoiceFormValue() || null,
  };
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_crew_ui.py -v --import-mode=importlib`
Expected: PASS (full file).

Run: `node --check static/js/crew.js`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add static/index.html static/js/crew.js tests/test_crew_ui.py
git commit -m "feat(crew): voice picker in the persona edit form

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Assistant panel voice picker

**Files:**
- Modify: `routes/assistant_routes.py` (`AssistantSettingsUpdate`, `update_assistant_settings`)
- Modify: `static/js/assistant.js`
- Test: `tests/test_assistant_routes.py` if it exists (check first; otherwise add to whichever existing test file covers `PATCH /api/assistant/settings` — grep for `update_assistant_settings` or `/api/assistant/settings` in `tests/` to find it), `tests/test_crew_ui.py` (for the frontend source-presence assertions, matching Task 5's finding-10-style pattern from sub-project 1's fix waves)

**Interfaces:**
- Consumes: `crew_to_dict`'s `tts_voice` key (Task 1, already returned by `GET /api/assistant/settings`'s existing `"crew": crew_to_dict(crew)` field).
- Produces: `PATCH /api/assistant/settings` accepts an optional `tts_voice` field. The Assistant settings modal includes the same dual-mode voice picker as the Crew panel.

- [ ] **Step 1: Write the failing tests**

No existing test file covers `PATCH /api/assistant/settings` at all (confirmed by
`grep -rl "update_assistant_settings\|/api/assistant/settings" tests/` returning nothing) — this
is new test infrastructure, not an extension of an existing file. Create
`tests/test_assistant_tts_voice.py`, mirroring `tests/test_crew_routes.py`'s isolated-engine
pattern (in-memory SQLite via `StaticPool`, `SessionLocal` monkeypatched onto the route module,
autouse `clear_db` fixture) since `assistant_routes.py` is the same shape — owner-scoped, not
admin-gated, module-level `from core.database import SessionLocal`.

`setup_assistant_routes(task_scheduler)` takes a `task_scheduler` argument, used by
`_get_or_create(owner)` only when no `is_default_assistant=True` row exists yet for that owner
(it calls `await task_scheduler.ensure_assistant_defaults(owner)` and re-queries). Pre-seed the
Assistant row directly via the ORM in each test (matching how `test_crew_routes.py`'s
`test_delete_default_assistant_is_blocked` already pre-seeds an `is_default_assistant=True`
row) so that branch is never reached — the stub `task_scheduler` then never needs a working
`ensure_assistant_defaults`, it just needs to exist as an object.

```python
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
```

(Note: `PATCH /api/assistant/settings`'s response shape is `{"crew": crew_to_dict(...), ...}` —
confirmed from `get_assistant_settings`'s existing return shape; `update_assistant_settings`'s
handler builds its response the same way. Read the live handler before assuming the exact
return statement if it's changed since this plan was written.)

`tests/test_crew_ui.py` additions (these follow the same source-presence pattern as every other test in that file, so the code is exact):

```python
def test_assistant_js_has_voice_field():
    src = (ROOT / "static" / "js" / "assistant.js").read_text(encoding="utf-8")
    assert "assistant-voice-select" in src
    assert "assistant-voice-input" in src


def test_assistant_js_save_includes_tts_voice():
    src = (ROOT / "static" / "js" / "assistant.js").read_text(encoding="utf-8")
    assert "tts_voice" in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crew_ui.py -k "assistant_js" -v --import-mode=importlib`
Expected: FAIL.

Run whichever assistant-settings test file/new test you wrote in Step 1: expected FAIL (`tts_voice` not yet accepted by the pydantic model, so it's silently ignored — assert on the response reflecting it back and it will fail).

- [ ] **Step 3: Wire the backend**

`routes/assistant_routes.py`'s `AssistantSettingsUpdate` currently reads (lines 32-41):

```python
class AssistantSettingsUpdate(BaseModel):
    name: Optional[str] = None
    avatar: Optional[str] = None
    personality: Optional[str] = None
    model: Optional[str] = None
    endpoint_url: Optional[str] = None
    enabled_tools: Optional[list[str]] = None
    allow_autonomous_email: Optional[bool] = None  # convenience toggle
    timezone: Optional[str] = None
    check_ins: Optional[list[CheckInUpdate]] = None
```

Add `tts_voice`:

```python
class AssistantSettingsUpdate(BaseModel):
    name: Optional[str] = None
    avatar: Optional[str] = None
    personality: Optional[str] = None
    model: Optional[str] = None
    endpoint_url: Optional[str] = None
    tts_voice: Optional[str] = None
    enabled_tools: Optional[list[str]] = None
    allow_autonomous_email: Optional[bool] = None  # convenience toggle
    timezone: Optional[str] = None
    check_ins: Optional[list[CheckInUpdate]] = None
```

In `update_assistant_settings` (around line 157-160 today):

```python
            if payload.endpoint_url is not None:
                crew_db.endpoint_url = payload.endpoint_url or None
```

Add right after:

```python
            if payload.endpoint_url is not None:
                crew_db.endpoint_url = payload.endpoint_url or None
            if payload.tts_voice is not None:
                crew_db.tts_voice = payload.tts_voice or None
```

- [ ] **Step 4: Wire the frontend**

Read `static/js/assistant.js`'s `_renderSettingsBody` and its save handler in full before editing. The model/endpoint row currently reads (lines 198-211):

```javascript
      <div class="assistant-field-row">
        <label class="assistant-field" style="flex:1;">
          <span>Model endpoint</span>
          <select id="assistant-endpoint" style="width:100%;">
            <option value="">(loading...)</option>
          </select>
        </label>
        <label class="assistant-field" style="flex:1;">
          <span>Model</span>
          <select id="assistant-model" style="width:100%;">
            <option value="${_esc(crew.model || '')}">${_esc(crew.model || '(default)')}</option>
          </select>
        </label>
      </div>
```

Add a voice row right after it, same dual-mode shape as Task 5's Crew form fields:

```javascript
      <div class="assistant-field-row">
        <label class="assistant-field" style="flex:1;">
          <span>Voice</span>
          <select id="assistant-voice-select" style="width:100%;display:none">
            <option value="">(use default)</option>
            <option value="alloy">Alloy</option>
            <option value="ash">Ash</option>
            <option value="coral">Coral</option>
            <option value="echo">Echo</option>
            <option value="fable">Fable</option>
            <option value="nova">Nova</option>
            <option value="onyx">Onyx</option>
            <option value="sage">Sage</option>
            <option value="shimmer">Shimmer</option>
          </select>
          <input id="assistant-voice-input" type="text" placeholder="af_heart (leave blank to use default)" style="width:100%;" value="${_esc(crew.tts_voice || '')}">
        </label>
      </div>
```

After the `body.innerHTML = ...` assignment and the existing "Populate model/endpoint dropdowns" block (~line 232-265), add a small block resolving which voice input mode to show — reuse the already-fetched TTS provider if this function already has it in scope, otherwise fetch `/api/tts/stats` once here:

```javascript
  // ── Voice picker mode (dropdown vs free-text, mirrors crew.js) ──
  (async () => {
    try {
      const stats = await _fetchJSON('/api/tts/stats');
      const isEndpoint = !!(stats.provider && stats.provider.startsWith('endpoint:'));
      const voiceSel = body.querySelector('#assistant-voice-select');
      const voiceInp = body.querySelector('#assistant-voice-input');
      if (voiceSel && voiceInp) {
        voiceSel.style.display = isEndpoint ? '' : 'none';
        voiceInp.style.display = isEndpoint ? 'none' : '';
        if (isEndpoint) voiceSel.value = crew.tts_voice || '';
      }
    } catch (e) {}
  })();
```

(`_fetchJSON` is already used elsewhere in this file, e.g. `_fetchEndpoints` — confirm its exact signature by reading its definition before assuming `_fetchJSON('/api/tts/stats')` matches the pattern other calls in this file use.)

In the save handler, the payload currently reads (lines 346-360):

```javascript
    const payload = {
      name: body.querySelector('#assistant-name').value.trim(),
      personality: body.querySelector('#assistant-personality').value,
      timezone: body.querySelector('#assistant-timezone').value || null,
      model: body.querySelector('#assistant-model').value || null,
      endpoint_url: body.querySelector('#assistant-endpoint').value || null,
      enabled_tools: selectedTools,
      check_ins: Array.from(body.querySelectorAll('.assistant-checkin-row')).map((row) => ({
```

Add `tts_voice` right after `endpoint_url`:

```javascript
      endpoint_url: body.querySelector('#assistant-endpoint').value || null,
      tts_voice: (function () {
        const sel = body.querySelector('#assistant-voice-select');
        const inp = body.querySelector('#assistant-voice-input');
        const val = (sel && sel.style.display !== 'none') ? sel.value : (inp ? inp.value : '');
        return val || null;
      })(),
      enabled_tools: selectedTools,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_crew_ui.py -v --import-mode=importlib`
Expected: PASS (full file).

Run the assistant-settings backend test file from Step 1: expected PASS.

Run: `node --check static/js/assistant.js`
Expected: no output.

- [ ] **Step 6: Run the full regression sweep**

Run: `pytest --import-mode=importlib -k "crew or tts or assistant" -v`
Expected: PASS across the board — this is the same broad sweep style used to close out every task in sub-project 1, catching any cross-task interaction this task's edits might have caused.

- [ ] **Step 7: Commit**

```bash
git add routes/assistant_routes.py static/js/assistant.js tests/test_crew_ui.py
git commit -m "feat(assistant): voice picker in the Assistant settings panel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

(Include the assistant-settings backend test file in this commit too, whichever file Step 1 determined it belongs in.)

---

## Self-Review Notes

- **Spec coverage:** every section of `docs/superpowers/specs/2026-08-05-crew-voice-per-persona-design.md` maps to a task — data model (Task 1), API changes (Task 3), frontend session-awareness (Task 4), both persona-editing panels (Tasks 5-6), CRUD write path (Task 2).
- **Architecture refinement from spec to plan:** the spec described `TTSService.synthesize()` taking a `session_id` and resolving the binding itself. While researching the live `tts-ai.js` call graph for this plan, I found `sessions.js` already exports `getCurrentSessionId()` (the same getter `modelPicker.js` already consumes via dependency injection) — this means voice resolution can happen once at each of the two true leaf points (`synthesize()`, `_playBrowser()`/`_resolveBrowserVoice()`) with **zero changes to any upstream caller** (`play()`, `enqueue()`, the streaming methods, or any `chat.js` call site), rather than needing `session_id` threaded through every intermediate method as the spec assumed. I also moved persona resolution to the *route* layer (`routes/tts_routes.py`'s new `_resolve_effective_voice` helper) rather than inside `TTSService` itself, keeping that class's DB/auth surface unchanged (it gets a plain `voice_override: str | None`) and matching how `chat_helpers.py`'s `extract_preset` already does persona resolution outside the deep generation logic, not inside it. This is a strict simplification of the spec's stated mechanism — the requirement itself ("everywhere a persona's session speaks uses its voice") is unchanged.
- **Type consistency:** `voice_override` is the exact parameter name in both `TTSService.synthesize()` and `TTSService.get_stats()` (Task 3); `session_id` is the exact field/param name across `TTSRequest`, `GET /stats`'s query param, and `tts-ai.js`'s POST body (Tasks 3-4); `tts_voice` is the exact key name in `crew_to_dict`, both CRUD routes, and both frontend forms' payloads (Tasks 1, 2, 5, 6).
- **Placeholder scan:** no TBDs. Task 6's Step 1 backend test intentionally withholds literal fixture code because the existing test file/pattern for `update_assistant_settings` is unknown until the implementer greps for it — this is a "find and mirror the existing pattern" instruction, not a placeholder for logic that should have been specified; the two frontend tests in the same step are exact and complete.

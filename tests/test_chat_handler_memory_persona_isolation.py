"""Persona isolation for the inline 'remember: ...' chat command
(src/chat_handler.py's ChatHandler.handle_memory_command). This is a second,
separate memory-write path from the auto-extractor and the manage_memory
tool -- it must respect the same persona boundary."""
import sys
import uuid
from types import SimpleNamespace

import pytest

import src.database  # noqa: F401 -- ensures "src.database" is in sys.modules
from core.models import Session as ModelSession
from src.chat_handler import ChatHandler
from src.memory import MemoryManager


def _make_persona_session_pair(owner="alice"):
    from core.database import CrewMember, Session as DbSession, SessionLocal
    db = SessionLocal()
    try:
        crew_a = str(uuid.uuid4())
        db.add(CrewMember(id=crew_a, owner=owner, name="Persona A"))
        sess_a = str(uuid.uuid4())
        db.add(DbSession(id=sess_a, name="s", endpoint_url="http://x", model="m",
                         owner=owner, crew_member_id=crew_a))

        crew_b = str(uuid.uuid4())
        db.add(CrewMember(id=crew_b, owner=owner, name="Persona B"))
        sess_b = str(uuid.uuid4())
        db.add(DbSession(id=sess_b, name="s", endpoint_url="http://x", model="m",
                         owner=owner, crew_member_id=crew_b))
        db.commit()
        return (sess_a, crew_a), (sess_b, crew_b)
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _stub_update_session_last_accessed(monkeypatch):
    # tests/conftest.py installs a bare-bones fake `src.database` module (only
    # SessionLocal/ModelEndpoint MagicMocks) when nothing has imported the real
    # one yet at collection time, so handle_memory_command's lazy
    # `from src.database import update_session_last_accessed` finds the stub,
    # not the real re-export. Add the attribute the same way the real module
    # provides it -- a no-op is fine here, this test isn't exercising that
    # bookkeeping call.
    monkeypatch.setattr(
        sys.modules["src.database"], "update_session_last_accessed",
        lambda session_id: True, raising=False,
    )


def _make_handler(mgr):
    # handle_memory_command only touches self.memory_manager and
    # self.session_manager (save_sessions()); the other constructor
    # dependencies are unused on this path, so simple stand-ins suffice.
    return ChatHandler(
        session_manager=SimpleNamespace(save_sessions=lambda: None),
        memory_manager=mgr,
        chat_processor=None,
        research_handler=None,
        preset_manager=None,
        upload_handler=None,
    )


@pytest.mark.asyncio
async def test_inline_remember_is_scoped_to_the_creating_persona(tmp_path):
    (sess_a_id, persona_a), (sess_b_id, persona_b) = _make_persona_session_pair()
    mgr = MemoryManager(str(tmp_path))
    handler = _make_handler(mgr)

    session_a = ModelSession(id=sess_a_id, name="s", endpoint_url="http://x", model="m", owner="alice")
    result = await handler.handle_memory_command(session_a, "remember: persona A's secret")
    assert result == "Saved to memory: persona A's secret"

    listed_a = mgr.load(owner="alice", persona_id=persona_a)
    assert any("persona A's secret" in e["text"] for e in listed_a)

    listed_b = mgr.load(owner="alice", persona_id=persona_b)
    assert not any("persona A's secret" in e["text"] for e in listed_b)


@pytest.mark.asyncio
async def test_inline_remember_saves_owner_so_it_is_recallable(tmp_path):
    """Pre-existing, persona-unrelated bug fixed as a side effect: the
    original code never passed owner to load()/add_entry(), so in an
    auth-enabled install an inline-remembered fact was saved but invisible
    to a subsequent owner-scoped load() -- saved and never recalled."""
    (sess_a_id, _persona_a), _ = _make_persona_session_pair()
    mgr = MemoryManager(str(tmp_path))
    handler = _make_handler(mgr)

    session_a = ModelSession(id=sess_a_id, name="s", endpoint_url="http://x", model="m", owner="alice")
    await handler.handle_memory_command(session_a, "remember: recallable fact")

    listed = mgr.load(owner="alice")
    assert any("recallable fact" in e["text"] for e in listed)

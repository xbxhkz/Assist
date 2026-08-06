"""Cross-persona isolation for the manage_memory agent tool -- the tool's
whole surface (list/add/edit/delete/search) must respect persona
boundaries once a session is bound to one. Uses a real MemoryManager
(tmp_path-backed) and real CrewMember/Session ORM rows (the app's default
SessionLocal, matching how tests/test_session_crew_binding.py already
exercises resolve_crew_binding against real rows)."""
import uuid

import pytest

import src.ai_interaction as ai
from src.memory import MemoryManager


@pytest.fixture
def memory_manager(tmp_path):
    mgr = MemoryManager(str(tmp_path))
    ai.set_memory_manager(mgr)
    yield mgr
    ai.set_memory_manager(None)


def _make_persona_session(owner="alice", name="Nav"):
    from core.database import SessionLocal, CrewMember, Session as DbSession
    db = SessionLocal()
    try:
        crew_id = str(uuid.uuid4())
        db.add(CrewMember(id=crew_id, owner=owner, name=name))
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m",
                         owner=owner, crew_member_id=crew_id))
        db.commit()
        return sess_id, crew_id
    finally:
        db.close()


def _make_unbound_session(owner="alice"):
    from core.database import SessionLocal, Session as DbSession
    db = SessionLocal()
    try:
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m", owner=owner))
        db.commit()
        return sess_id
    finally:
        db.close()


@pytest.mark.asyncio
async def test_add_then_list_is_scoped_to_the_creating_persona(memory_manager):
    sess_a, _ = _make_persona_session(name="Persona A")
    result = await ai.do_manage_memory("add\nA's private fact", session_id=sess_a, owner="alice")
    assert "memory_id" in result

    listed = await ai.do_manage_memory("list", session_id=sess_a, owner="alice")
    assert "A's private fact" in listed["results"]


@pytest.mark.asyncio
async def test_persona_b_cannot_list_persona_as_memory(memory_manager):
    sess_a, _ = _make_persona_session(owner="alice", name="Persona A")
    sess_b, _ = _make_persona_session(owner="alice", name="Persona B")
    await ai.do_manage_memory("add\nA's private fact", session_id=sess_a, owner="alice")

    listed = await ai.do_manage_memory("list", session_id=sess_b, owner="alice")
    assert "A's private fact" not in listed["results"]


@pytest.mark.asyncio
async def test_persona_b_cannot_search_persona_as_memory(memory_manager):
    sess_a, _ = _make_persona_session(owner="alice", name="Persona A")
    sess_b, _ = _make_persona_session(owner="alice", name="Persona B")
    await ai.do_manage_memory("add\nthe secret launch code is 42", session_id=sess_a, owner="alice")

    found = await ai.do_manage_memory("search\nlaunch code", session_id=sess_b, owner="alice")
    assert "42" not in found["results"]


@pytest.mark.asyncio
async def test_persona_b_cannot_edit_persona_as_memory(memory_manager):
    sess_a, _ = _make_persona_session(owner="alice", name="Persona A")
    sess_b, _ = _make_persona_session(owner="alice", name="Persona B")
    added = await ai.do_manage_memory("add\noriginal text", session_id=sess_a, owner="alice")
    memory_id = added["memory_id"]

    result = await ai.do_manage_memory(f"edit\n{memory_id}\nhijacked text", session_id=sess_b, owner="alice")
    assert "error" in result

    listed = await ai.do_manage_memory("list", session_id=sess_a, owner="alice")
    assert "original text" in listed["results"]
    assert "hijacked text" not in listed["results"]


@pytest.mark.asyncio
async def test_persona_b_cannot_delete_persona_as_memory(memory_manager):
    sess_a, _ = _make_persona_session(owner="alice", name="Persona A")
    sess_b, _ = _make_persona_session(owner="alice", name="Persona B")
    added = await ai.do_manage_memory("add\nsurvives deletion attempt", session_id=sess_a, owner="alice")
    memory_id = added["memory_id"]

    result = await ai.do_manage_memory(f"delete\n{memory_id}", session_id=sess_b, owner="alice")
    assert "error" in result

    listed = await ai.do_manage_memory("list", session_id=sess_a, owner="alice")
    assert "survives deletion attempt" in listed["results"]


@pytest.mark.asyncio
async def test_shared_pool_memory_visible_to_every_persona(memory_manager):
    sess_a, _ = _make_persona_session(owner="alice", name="Persona A")
    unbound = _make_unbound_session(owner="alice")
    await ai.do_manage_memory("add\nshared fact for everyone", session_id=unbound, owner="alice")

    listed = await ai.do_manage_memory("list", session_id=sess_a, owner="alice")
    assert "shared fact for everyone" in listed["results"]


@pytest.mark.asyncio
async def test_unbound_session_keeps_todays_exact_behavior(memory_manager):
    """Fail-open: no persona binding at all -> the full owner pool, exactly
    like before this feature existed."""
    sess_a, _ = _make_persona_session(owner="alice", name="Persona A")
    unbound = _make_unbound_session(owner="alice")
    await ai.do_manage_memory("add\nA's fact", session_id=sess_a, owner="alice")
    await ai.do_manage_memory("add\nunbound fact", session_id=unbound, owner="alice")

    listed = await ai.do_manage_memory("list", session_id=unbound, owner="alice")
    assert "A's fact" in listed["results"]
    assert "unbound fact" in listed["results"]


@pytest.mark.asyncio
async def test_dangling_session_id_falls_back_to_owner_only(memory_manager):
    """Fail-open: a session_id that doesn't resolve to any real session must
    not error -- falls back to owner-only behavior, same as no session_id."""
    await ai.do_manage_memory("add\na fact", session_id=None, owner="alice")
    listed = await ai.do_manage_memory("list", session_id="does-not-exist", owner="alice")
    assert "a fact" in listed["results"]

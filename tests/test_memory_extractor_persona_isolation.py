"""Cross-persona isolation for the AUTO-EXTRACTION memory write path -- this
is the app's dominant memory-write path (runs unattended, default-on every
~4 messages), unlike the explicit manage_memory tool call. A persona-bound
session's auto-extracted facts must be invisible to a different persona,
exactly like the explicit tool path Task 2 already isolated."""
import uuid

import pytest

import src.event_bus
import src.llm_core
from core.models import ChatMessage, Session as ModelSession
from services.memory.memory_extractor import extract_and_store
from src.memory import MemoryManager


def _make_persona_session(owner="alice", name="Nav"):
    from core.database import CrewMember, Session as DbSession, SessionLocal
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


@pytest.mark.asyncio
async def test_auto_extracted_fact_is_scoped_to_the_creating_persona(tmp_path, monkeypatch):
    sess_a_id, persona_a = _make_persona_session(owner="alice", name="Persona A")
    sess_b_id, persona_b = _make_persona_session(owner="alice", name="Persona B")
    mgr = MemoryManager(str(tmp_path))

    async def _fake_llm(url, model, messages, **kwargs):
        return '[{"text": "Persona A secret project is codenamed Phoenix", "category": "fact"}]'

    monkeypatch.setattr(src.llm_core, "llm_call_async", _fake_llm)
    # fire_event touches an async event loop / disk -- neutralize it, mirroring
    # test_memory_extractor_vector_degraded.py.
    monkeypatch.setattr(src.event_bus, "fire_event", lambda *a, **k: None)

    session_a = ModelSession(
        id=sess_a_id, name="s", endpoint_url="http://x", model="m", owner="alice",
        history=[
            ChatMessage("user", "Tell me about my current work."),
            ChatMessage("assistant", "Sure, tell me more."),
        ],
    )

    await extract_and_store(
        session_a, mgr, None,
        endpoint_url="http://x", model="m", headers=None,
    )

    entries_a = mgr.load(owner="alice", persona_id=persona_a)
    assert any("Phoenix" in e["text"] for e in entries_a), (
        "Auto-extracted fact was not visible to the persona whose session "
        "produced it."
    )

    entries_b = mgr.load(owner="alice", persona_id=persona_b)
    assert not any("Phoenix" in e["text"] for e in entries_b), (
        "Auto-extracted fact from persona A's session leaked into persona "
        "B's view -- the dominant memory-write path is not isolated."
    )

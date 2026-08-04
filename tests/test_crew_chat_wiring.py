import uuid
from unittest.mock import MagicMock

from routes.chat_helpers import extract_preset


class _FakeSess:
    def __init__(self, sid):
        self.id = sid


def _make_bound_session(personality="You are a pirate.", name="Cap'n"):
    from core.database import SessionLocal, CrewMember, Session as DbSession
    db = SessionLocal()
    try:
        crew_id = str(uuid.uuid4())
        db.add(CrewMember(id=crew_id, owner="alice", name=name, personality=personality))
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m",
                         owner="alice", crew_member_id=crew_id))
        db.commit()
        return sess_id
    finally:
        db.close()


def _fake_chat_handler(system_prompt="preset prompt", char_name="Preset Name"):
    h = MagicMock()
    h.validate_and_extract_preset.return_value = (0.5, 100, system_prompt, char_name)
    return h


def test_extract_preset_unbound_session_uses_preset_as_before():
    sess_id = str(uuid.uuid4())
    handler = _fake_chat_handler()
    result = extract_preset(handler, "some_preset", sess=_FakeSess(sess_id), owner="alice")
    assert result.system_prompt == "preset prompt"
    assert result.character_name == "Preset Name"


def test_extract_preset_persona_bound_session_overrides_preset():
    sess_id = _make_bound_session(personality="You are a pirate.", name="Cap'n")
    handler = _fake_chat_handler()
    result = extract_preset(handler, "some_preset", sess=_FakeSess(sess_id), owner="alice")
    assert result.system_prompt == "You are a pirate."
    assert result.character_name == "Cap'n"


def test_extract_preset_persona_with_empty_personality_falls_back_to_preset():
    sess_id = _make_bound_session(personality="", name="Cap'n")
    handler = _fake_chat_handler()
    result = extract_preset(handler, "some_preset", sess=_FakeSess(sess_id), owner="alice")
    assert result.system_prompt == "preset prompt"


def test_extract_preset_dangling_crew_reference_falls_back_to_preset():
    from core.database import SessionLocal, Session as DbSession
    db = SessionLocal()
    try:
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m",
                         owner="alice", crew_member_id="deleted-crew-id"))
        db.commit()
    finally:
        db.close()
    handler = _fake_chat_handler()
    result = extract_preset(handler, "some_preset", sess=_FakeSess(sess_id), owner="alice")
    assert result.system_prompt == "preset prompt"


def test_extract_preset_no_sess_arg_behaves_like_before():
    handler = _fake_chat_handler()
    result = extract_preset(handler, "some_preset")
    assert result.system_prompt == "preset prompt"
    assert result.character_name == "Preset Name"

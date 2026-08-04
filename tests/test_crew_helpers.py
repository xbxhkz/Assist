import json
import uuid

from core.database import SessionLocal, CrewMember, Session as DbSession
from src.crew_helpers import crew_to_dict, resolve_crew_binding, crew_disabled_tools


def _make_crew(db, owner="alice", enabled_tools=None, personality="You are helpful."):
    c = CrewMember(
        id=str(uuid.uuid4()), owner=owner, name="Nav", personality=personality,
        enabled_tools=json.dumps(enabled_tools) if enabled_tools is not None else None,
    )
    db.add(c)
    db.commit()
    return c


def _make_session(db, owner="alice", crew_member_id=None):
    s = DbSession(id=str(uuid.uuid4()), name="s", endpoint_url="http://x", model="m",
                 owner=owner, crew_member_id=crew_member_id)
    db.add(s)
    db.commit()
    return s


def test_crew_to_dict_shape():
    db = SessionLocal()
    try:
        c = _make_crew(db, enabled_tools=["web_search"])
        d = crew_to_dict(c)
        assert d["name"] == "Nav" and d["enabled_tools"] == ["web_search"]
        assert d["is_default_assistant"] is False
    finally:
        db.close()


def test_crew_to_dict_malformed_json_tools_becomes_empty_list():
    db = SessionLocal()
    try:
        c = _make_crew(db)
        c.enabled_tools = "{not json"
        d = crew_to_dict(c)
        assert d["enabled_tools"] == []
    finally:
        db.close()


def test_crew_to_dict_endpoint_id_round_trips():
    db = SessionLocal()
    try:
        c = _make_crew(db)
        c.endpoint_id = "ep-123"
        db.commit()
        d = crew_to_dict(c)
        assert d["endpoint_id"] == "ep-123"
    finally:
        db.close()


def test_crew_to_dict_endpoint_id_defaults_to_none():
    db = SessionLocal()
    try:
        c = _make_crew(db)
        d = crew_to_dict(c)
        assert d["endpoint_id"] is None
    finally:
        db.close()


def test_crew_to_dict_all_sentinel_reports_enabled_tools_all():
    db = SessionLocal()
    try:
        c = _make_crew(db)
        c.enabled_tools = "all"
        db.commit()
        d = crew_to_dict(c)
        assert d["enabled_tools_all"] is True
        assert d["enabled_tools"] == []
    finally:
        db.close()


def test_crew_to_dict_enabled_tools_all_false_for_normal_list():
    db = SessionLocal()
    try:
        c = _make_crew(db, enabled_tools=["web_search"])
        d = crew_to_dict(c)
        assert d["enabled_tools_all"] is False
    finally:
        db.close()


def test_crew_to_dict_allow_autonomous_email_true_when_send_email_enabled():
    db = SessionLocal()
    try:
        c = _make_crew(db, enabled_tools=["send_email"])
        d = crew_to_dict(c)
        assert d["allow_autonomous_email"] is True
    finally:
        db.close()


def test_crew_to_dict_allow_autonomous_email_true_when_reply_to_email_enabled():
    db = SessionLocal()
    try:
        c = _make_crew(db, enabled_tools=["reply_to_email"])
        d = crew_to_dict(c)
        assert d["allow_autonomous_email"] is True
    finally:
        db.close()


def test_crew_to_dict_allow_autonomous_email_true_when_tools_all():
    db = SessionLocal()
    try:
        c = _make_crew(db)
        c.enabled_tools = "all"
        db.commit()
        d = crew_to_dict(c)
        assert d["allow_autonomous_email"] is True
    finally:
        db.close()


def test_crew_to_dict_allow_autonomous_email_false_otherwise():
    db = SessionLocal()
    try:
        c = _make_crew(db, enabled_tools=["web_search"])
        d = crew_to_dict(c)
        assert d["allow_autonomous_email"] is False
    finally:
        db.close()


def test_resolve_crew_binding_returns_none_when_session_unbound():
    db = SessionLocal()
    try:
        s = _make_session(db)
        assert resolve_crew_binding(db, s.id, "alice") is None
    finally:
        db.close()


def test_resolve_crew_binding_returns_owner_scoped_crew():
    db = SessionLocal()
    try:
        c = _make_crew(db, owner="alice")
        s = _make_session(db, owner="alice", crew_member_id=c.id)
        found = resolve_crew_binding(db, s.id, "alice")
        assert found is not None and found.id == c.id
    finally:
        db.close()


def test_resolve_crew_binding_never_returns_another_owners_crew():
    db = SessionLocal()
    try:
        c = _make_crew(db, owner="alice")
        s = _make_session(db, owner="alice", crew_member_id=c.id)
        # A different owner querying the same session id must not see it.
        assert resolve_crew_binding(db, s.id, "bob") is None
    finally:
        db.close()


def test_resolve_crew_binding_never_raises_on_missing_session():
    db = SessionLocal()
    try:
        assert resolve_crew_binding(db, "no-such-session", "alice") is None
    finally:
        db.close()


def test_crew_disabled_tools_empty_when_unbound():
    db = SessionLocal()
    try:
        s = _make_session(db)
        assert crew_disabled_tools(db, s.id, "alice") == set()
    finally:
        db.close()


def test_crew_disabled_tools_empty_when_all():
    db = SessionLocal()
    try:
        c = _make_crew(db, owner="alice", enabled_tools=None)
        c.enabled_tools = "all"
        db.commit()
        s = _make_session(db, owner="alice", crew_member_id=c.id)
        assert crew_disabled_tools(db, s.id, "alice") == set()
    finally:
        db.close()


def test_crew_disabled_tools_restricts_to_allowlist():
    from src.tool_policy import known_tool_names
    db = SessionLocal()
    try:
        c = _make_crew(db, owner="alice", enabled_tools=["web_search"])
        s = _make_session(db, owner="alice", crew_member_id=c.id)
        disabled = crew_disabled_tools(db, s.id, "alice")
        assert "web_search" not in disabled
        assert disabled == (known_tool_names() - {"web_search"})
    finally:
        db.close()


def test_crew_disabled_tools_fails_open_on_malformed_json():
    db = SessionLocal()
    try:
        c = _make_crew(db, owner="alice")
        c.enabled_tools = "{not json"
        db.commit()
        s = _make_session(db, owner="alice", crew_member_id=c.id)
        assert crew_disabled_tools(db, s.id, "alice") == set()
    finally:
        db.close()


def test_crew_disabled_tools_never_raises_on_dangling_reference():
    db = SessionLocal()
    try:
        s = _make_session(db, owner="alice", crew_member_id="deleted-crew-id")
        assert crew_disabled_tools(db, s.id, "alice") == set()
    finally:
        db.close()


def test_crew_to_dict_includes_timezone():
    """Regression test for static/js/assistant.js:190-192 reading crew.timezone.
    The Timezone dropdown in Assistant Settings UI must read this key from
    crew_to_dict() response, or it will always show the default instead of the
    user's saved value."""
    db = SessionLocal()
    try:
        c = _make_crew(db, owner="alice")
        c.timezone = "America/New_York"
        db.commit()
        d = crew_to_dict(c)
        assert d["timezone"] == "America/New_York"
    finally:
        db.close()


def test_crew_disabled_tools_includes_shell_tools_in_known_set():
    """Regression test for the agent_tools circular-import workaround: without
    it, known_tool_names()'s first call in a fresh process silently omits
    cmd/powershell (a real circular import between tool_schemas/agent_tools).
    The fix now lives in known_tool_names() itself (src/tool_policy.py), so
    this proves it still holds end-to-end through crew_disabled_tools.
    """
    db = SessionLocal()
    try:
        c = _make_crew(db, owner="alice", enabled_tools=["web_search"])
        s = _make_session(db, owner="alice", crew_member_id=c.id)
        disabled = crew_disabled_tools(db, s.id, "alice")
        assert "cmd" in disabled
        assert "powershell" in disabled
    finally:
        db.close()

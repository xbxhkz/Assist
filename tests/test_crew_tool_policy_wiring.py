import uuid


def _make_restricted_session(owner="alice", enabled_tools=None):
    import json
    from core.database import SessionLocal, CrewMember, Session as DbSession
    db = SessionLocal()
    try:
        crew_id = str(uuid.uuid4())
        db.add(CrewMember(id=crew_id, owner=owner, name="Nav",
                          enabled_tools=json.dumps(enabled_tools or ["web_search"])))
        sess_id = str(uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m",
                         owner=owner, crew_member_id=crew_id))
        db.commit()
        return sess_id
    finally:
        db.close()


def test_crew_disabled_tools_reflects_in_a_composed_policy_for_a_bound_session():
    # This exercises exactly the composition the three chat_routes.py call
    # sites now perform: crew_disabled_tools(...) unioned into disabled_tools
    # before build_effective_tool_policy(disabled_tools=...).
    from src.crew_helpers import crew_disabled_tools
    from src.tool_policy import build_effective_tool_policy
    from core.database import SessionLocal

    sess_id = _make_restricted_session(enabled_tools=["web_search"])
    db = SessionLocal()
    try:
        extra = crew_disabled_tools(db, sess_id, "alice")
    finally:
        db.close()
    policy = build_effective_tool_policy(disabled_tools=extra)
    assert policy.blocks("generate_image") is True
    assert policy.blocks("web_search") is False


def test_unbound_session_adds_no_extra_restriction():
    from core.database import SessionLocal, Session as DbSession
    from src.crew_helpers import crew_disabled_tools
    from src.tool_policy import build_effective_tool_policy
    import uuid as _uuid

    db = SessionLocal()
    try:
        sess_id = str(_uuid.uuid4())
        db.add(DbSession(id=sess_id, name="s", endpoint_url="http://x", model="m", owner="alice"))
        db.commit()
        extra = crew_disabled_tools(db, sess_id, "alice")
    finally:
        db.close()
    policy = build_effective_tool_policy(disabled_tools=extra)
    assert policy.blocks("generate_image") is False
    assert policy.blocks("web_search") is False

"""list_tool_calls flattens persisted tool_events out of chat history -- no
dedicated table, see
docs/superpowers/specs/2026-08-10-mission-control-tool-call-log-design.md.
"""
import json
import uuid
from datetime import datetime, timedelta

import routes.tool_calls_routes as tcr
from core.database import ChatMessage as DBChatMessage, Session as DBSession, SessionLocal


def _unique_owner():
    return "owner-" + uuid.uuid4().hex[:10]


def _make_session(owner):
    db = SessionLocal()
    try:
        sid = str(uuid.uuid4())
        db.add(DBSession(id=sid, name="s", endpoint_url="http://x", model="m", owner=owner))
        db.commit()
        return sid
    finally:
        db.close()


def _add_message(session_id, meta_data, ts, role="assistant"):
    db = SessionLocal()
    try:
        mid = str(uuid.uuid4())
        db.add(DBChatMessage(
            id=mid, session_id=session_id, role=role, content="...",
            meta_data=meta_data, timestamp=ts,
        ))
        db.commit()
        return mid
    finally:
        db.close()


def _add_tool_events(session_id, tool_events, ts):
    return _add_message(session_id, json.dumps({"tool_events": tool_events}), ts)


def test_flattens_tool_events_newest_first():
    owner = _unique_owner()
    sid = _make_session(owner)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    _add_tool_events(sid, [{"round": 1, "tool": "run_command", "command": "ls", "output": "a b c", "exit_code": 0}], t0)
    _add_tool_events(sid, [{"round": 1, "tool": "web_search", "command": "q", "output": "results", "exit_code": None}], t0 + timedelta(minutes=1))

    db = SessionLocal()
    try:
        records, has_more = tcr.list_tool_calls(db, owner=owner)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["web_search", "run_command"]
    assert has_more is False


def test_pagination_spans_multiple_internal_batches(monkeypatch):
    monkeypatch.setattr(tcr, "_BATCH_SIZE", 2)
    owner = _unique_owner()
    sid = _make_session(owner)
    base = datetime(2026, 1, 1, 12, 0, 0)
    for i in range(5):
        _add_tool_events(sid, [{"round": 1, "tool": "tool%d" % i, "command": "c", "output": "o", "exit_code": 0}],
                          base + timedelta(minutes=i))

    db = SessionLocal()
    try:
        records, has_more = tcr.list_tool_calls(db, owner=owner, limit=3)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["tool4", "tool3", "tool2"]
    assert has_more is True


def test_pagination_offset_gets_next_page(monkeypatch):
    monkeypatch.setattr(tcr, "_BATCH_SIZE", 2)
    owner = _unique_owner()
    sid = _make_session(owner)
    base = datetime(2026, 1, 1, 12, 0, 0)
    for i in range(5):
        _add_tool_events(sid, [{"round": 1, "tool": "tool%d" % i, "command": "c", "output": "o", "exit_code": 0}],
                          base + timedelta(minutes=i))

    db = SessionLocal()
    try:
        records, has_more = tcr.list_tool_calls(db, owner=owner, limit=3, offset=3)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["tool1", "tool0"]
    assert has_more is False


def test_session_id_filter():
    owner = _unique_owner()
    sid_a = _make_session(owner)
    sid_b = _make_session(owner)
    _add_tool_events(sid_a, [{"round": 1, "tool": "toolA", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 1))
    _add_tool_events(sid_b, [{"round": 1, "tool": "toolB", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 2))

    db = SessionLocal()
    try:
        records, _ = tcr.list_tool_calls(db, owner=owner, session_id=sid_a)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["toolA"]


def test_session_id_filter_cannot_leak_another_owners_session():
    owner_a = _unique_owner()
    owner_b = _unique_owner()
    sid_b = _make_session(owner_b)
    _add_tool_events(sid_b, [{"round": 1, "tool": "bob_tool", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 1))

    db = SessionLocal()
    try:
        records, _ = tcr.list_tool_calls(db, owner=owner_a, session_id=sid_b)
    finally:
        db.close()

    assert records == []


def test_tool_name_filter():
    owner = _unique_owner()
    sid = _make_session(owner)
    _add_tool_events(sid, [
        {"round": 1, "tool": "run_command", "command": "c1", "output": "o1", "exit_code": 0},
        {"round": 2, "tool": "web_search", "command": "c2", "output": "o2", "exit_code": None},
    ], datetime(2026, 1, 1))

    db = SessionLocal()
    try:
        records, _ = tcr.list_tool_calls(db, owner=owner, tool_name="web_search")
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["web_search"]


def test_corrupt_json_row_is_skipped_not_raised():
    owner = _unique_owner()
    sid = _make_session(owner)
    _add_message(sid, "{not valid json", datetime(2026, 1, 1))
    _add_tool_events(sid, [{"round": 1, "tool": "ok_tool", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 2))

    db = SessionLocal()
    try:
        records, _ = tcr.list_tool_calls(db, owner=owner)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["ok_tool"]


def test_message_without_tool_events_key_is_skipped():
    owner = _unique_owner()
    sid = _make_session(owner)
    _add_message(sid, json.dumps({"model": "some-model"}), datetime(2026, 1, 1))
    _add_tool_events(sid, [{"round": 1, "tool": "ok_tool", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 2))

    db = SessionLocal()
    try:
        records, _ = tcr.list_tool_calls(db, owner=owner)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["ok_tool"]


def test_owner_isolation():
    owner_a = _unique_owner()
    owner_b = _unique_owner()
    sid_a = _make_session(owner_a)
    sid_b = _make_session(owner_b)
    _add_tool_events(sid_a, [{"round": 1, "tool": "alice_tool", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 1))
    _add_tool_events(sid_b, [{"round": 1, "tool": "bob_tool", "command": "c", "output": "o", "exit_code": 0}], datetime(2026, 1, 2))

    db = SessionLocal()
    try:
        records, _ = tcr.list_tool_calls(db, owner=owner_a)
    finally:
        db.close()

    assert [r["tool"] for r in records] == ["alice_tool"]

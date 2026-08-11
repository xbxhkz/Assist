"""Chat-level tool-call log (Mission Control sub-project 2a).

Tool-call data already exists -- every agent turn's tool_events is persisted
inside the assistant ChatMessage.meta_data JSON blob (src/agent_loop.py).
This module queries that existing data at read time; it does not add a new
table. See docs/superpowers/specs/2026-08-10-mission-control-tool-call-log-design.md.
"""
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from core.database import ChatMessage as DBChatMessage, Session as DBSession, SessionLocal

_BATCH_SIZE = 200
# Safety cap on how many _BATCH_SIZE batches list_tool_calls will scan while
# hunting for `limit` matching records. Without this, a tool_name filter that
# matches nothing (or whose matches are all old) walks the owner's ENTIRE
# message history with no upper bound -- meta_data is non-null for virtually
# every message, so there's nothing else to narrow the scan, and there's no
# plain index on timestamp alone (only the composite (session_id, timestamp)).
# When the cap is hit we have no proof there isn't more data past it, so
# has_more must honestly report True rather than claiming the scan was
# exhaustive.
_MAX_BATCHES = 50


def list_tool_calls(
    db,
    owner: Optional[str],
    session_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[Dict], bool]:
    query = (
        db.query(DBChatMessage, DBSession.name)
        .join(DBSession, DBChatMessage.session_id == DBSession.id)
        .filter(
            DBChatMessage.role == "assistant",
            DBChatMessage.meta_data.isnot(None),
        )
    )
    if owner is None:
        query = query.filter(DBSession.owner.is_(None))
    else:
        query = query.filter(DBSession.owner == owner)
    if session_id:
        query = query.filter(DBChatMessage.session_id == session_id)
    if since is not None:
        query = query.filter(DBChatMessage.timestamp >= since)
    if until is not None:
        query = query.filter(DBChatMessage.timestamp <= until)
    query = query.order_by(DBChatMessage.timestamp.desc())

    records: List[Dict] = []
    has_more = False
    seen = 0
    batch_offset = 0
    batches_scanned = 0

    while True:
        batch = query.offset(batch_offset).limit(_BATCH_SIZE).all()
        if not batch:
            break
        batch_offset += len(batch)
        batches_scanned += 1

        for message, session_name in batch:
            try:
                meta = json.loads(message.meta_data)
            except (ValueError, TypeError):
                continue
            if not isinstance(meta, dict):
                continue
            events = meta.get("tool_events")
            if not isinstance(events, list):
                continue
            for event in events:
                if not isinstance(event, dict):
                    continue
                if tool_name and event.get("tool") != tool_name:
                    continue
                if seen < offset:
                    seen += 1
                    continue
                if len(records) >= limit:
                    has_more = True
                    break
                records.append({
                    "session_id": message.session_id,
                    "session_name": session_name,
                    "message_id": message.id,
                    "timestamp": (message.timestamp.isoformat() + "Z") if message.timestamp else None,
                    "round": event.get("round"),
                    "tool": event.get("tool"),
                    "command": event.get("command"),
                    "output": event.get("output"),
                    "exit_code": event.get("exit_code"),
                })
            if has_more:
                break
        if has_more or len(batch) < _BATCH_SIZE:
            break
        if batches_scanned >= _MAX_BATCHES:
            # Gave up before exhausting the owner's history -- we cannot
            # claim there's nothing more past the cap.
            has_more = True
            break

    return records, has_more


from fastapi import APIRouter, HTTPException, Request

from src.auth_helpers import effective_user


def setup_tool_calls_routes() -> APIRouter:
    router = APIRouter(prefix="/api/tool-calls", tags=["tool-calls"])

    @router.get("")
    async def get_tool_calls(
        request: Request,
        session_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        user = effective_user(request)

        since_dt = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid 'since' -- expected ISO 8601")
            if since_dt.tzinfo is not None:
                # ChatMessage.timestamp is stored as naive UTC -- convert an
                # offset-aware caller value instead of silently comparing it
                # as if the offset weren't there.
                since_dt = since_dt.astimezone(timezone.utc).replace(tzinfo=None)
        until_dt = None
        if until:
            try:
                until_dt = datetime.fromisoformat(until)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid 'until' -- expected ISO 8601")
            if until_dt.tzinfo is not None:
                until_dt = until_dt.astimezone(timezone.utc).replace(tzinfo=None)

        safe_limit = max(1, min(limit, 200))
        safe_offset = max(0, offset)

        db = SessionLocal()
        try:
            records, has_more = list_tool_calls(
                db, owner=user, session_id=session_id, tool_name=tool_name,
                since=since_dt, until=until_dt, limit=safe_limit, offset=safe_offset,
            )
        finally:
            db.close()
        return {"tool_calls": records, "has_more": has_more}

    return router

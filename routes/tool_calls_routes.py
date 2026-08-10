"""Chat-level tool-call log (Mission Control sub-project 2a).

Tool-call data already exists -- every agent turn's tool_events is persisted
inside the assistant ChatMessage.meta_data JSON blob (src/agent_loop.py).
This module queries that existing data at read time; it does not add a new
table. See docs/superpowers/specs/2026-08-10-mission-control-tool-call-log-design.md.
"""
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from core.database import ChatMessage as DBChatMessage, Session as DBSession, SessionLocal

_BATCH_SIZE = 200


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

    while True:
        batch = query.offset(batch_offset).limit(_BATCH_SIZE).all()
        if not batch:
            break
        batch_offset += len(batch)

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

    return records, has_more


from fastapi import APIRouter, HTTPException, Request

from src.auth_helpers import get_current_user


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
        user = get_current_user(request)

        since_dt = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid 'since' -- expected ISO 8601")
        until_dt = None
        if until:
            try:
                until_dt = datetime.fromisoformat(until)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid 'until' -- expected ISO 8601")

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

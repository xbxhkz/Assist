"""Active-agents enumeration (Mission Control sub-project 2b).

Reads two already-existing pieces of in-memory state -- src.agent_runs'
per-session run status and SessionManager's in-memory session cache -- to
answer "which of the caller's own chat sessions are running right now."
Adds no new write path. See
docs/superpowers/specs/2026-08-11-mission-control-active-agents-design.md.
"""
from typing import Dict, List, Optional

from fastapi import APIRouter, Request

from src import agent_runs
from src.auth_helpers import effective_user


def list_active_agents(session_manager, owner: Optional[str]) -> List[Dict]:
    active = []
    for session_id in agent_runs.list_active():
        session = session_manager.sessions.get(session_id)
        if session is None:
            continue
        if session.owner != owner:
            continue
        active.append({"session_id": session_id, "session_name": session.name})
    return active


def setup_agent_runs_routes(session_manager) -> APIRouter:
    router = APIRouter(prefix="/api/agent-runs", tags=["agent-runs"])

    @router.get("/active")
    async def get_active_agents(request: Request):
        user = effective_user(request)
        return {"active": list_active_agents(session_manager, user)}

    return router

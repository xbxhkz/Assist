"""Shared data-access helpers for CrewMember ("persona") rows. Used by
routes/assistant_routes.py (the singleton default-assistant feature),
routes/crew_routes.py (the general multi-persona CRUD), and the chat-turn
wiring in routes/chat_helpers.py and routes/chat_routes.py. Never raises --
every function here degrades to an empty/None result on any lookup or
parsing failure, since a persona binding is a self-service customization,
not a security boundary."""
import json

from core.database import CrewMember, Session as DbSession


_EMAIL_TOOLS = {"send_email", "reply_to_email"}


def crew_to_dict(c: CrewMember) -> dict:
    raw = c.enabled_tools
    tools_all = (raw == "all")
    try:
        tools = json.loads(raw) if raw and not tools_all else []
    except Exception:
        tools = []
    return {
        "id": c.id,
        "name": c.name,
        "avatar": c.avatar,
        "personality": c.personality,
        "model": c.model,
        "endpoint_url": c.endpoint_url,
        "endpoint_id": c.endpoint_id,
        "greeting": c.greeting,
        "enabled_tools": tools,
        "enabled_tools_all": tools_all,
        "allow_autonomous_email": any(t in _EMAIL_TOOLS for t in tools) or tools_all,
        "session_id": c.session_id,
        "is_default_assistant": bool(c.is_default_assistant),
        "is_active": bool(c.is_active),
        "sort_order": c.sort_order or 0,
        "timezone": c.timezone,
    }


def resolve_crew_binding(db, session_id: str, owner: str):
    """Return the owner-scoped CrewMember bound to `session_id`, or None if
    the session doesn't exist, has no binding, the binding is dangling, or
    belongs to a different owner. Never raises."""
    try:
        sess = db.query(DbSession).filter(DbSession.id == session_id).first()
        if not sess or not sess.crew_member_id:
            return None
        crew = db.query(CrewMember).filter(
            CrewMember.id == sess.crew_member_id,
            CrewMember.owner == owner,
        ).first()
        return crew
    except Exception:  # noqa: BLE001
        return None


def crew_disabled_tools(db, session_id: str, owner: str) -> set:
    """Return the set of tool names a session's bound persona restricts,
    i.e. every known tool NOT in its enabled_tools allowlist. Empty set
    (no extra restriction) when unbound, "all", empty, missing, or
    unparseable -- fail-open, never raises."""
    try:
        from src.tool_policy import known_tool_names
        crew = resolve_crew_binding(db, session_id, owner)
        if not crew or not crew.enabled_tools or crew.enabled_tools == "all":
            return set()
        try:
            allowed = json.loads(crew.enabled_tools)
        except Exception:
            return set()
        if not isinstance(allowed, list) or not allowed:
            return set()
        return known_tool_names() - set(allowed)
    except Exception:  # noqa: BLE001
        return set()

"""Per-chat-session approval bridge for state-changing shell commands. Mirrors
the AI Operator's wake-Event confirmation: a write command awaits a decision that
the /api/shell/decision route delivers. Fail-closed: timeout → deny."""
import asyncio
import uuid

# session_id -> {"auto_all": bool, "pending": {pid: {command, shell, wake, decision}}}
_SESSIONS: dict = {}


def _sess(session_id):
    return _SESSIONS.setdefault(session_id or "_none",
                                {"auto_all": False, "pending": {}})


def reset_session(session_id):
    _SESSIONS.pop(session_id or "_none", None)


def reset_all():
    _SESSIONS.clear()


def list_pending(session_id):
    s = _sess(session_id)
    return [{"pending_id": pid, "command": p["command"], "shell": p["shell"]}
            for pid, p in s["pending"].items()]


def set_decision(session_id, pending_id, decision) -> bool:
    """Deliver a UI decision. decision ∈ approve | deny | auto_approve_all."""
    s = _sess(session_id)
    p = s["pending"].get(pending_id)
    if p is None:
        return False
    if decision == "auto_approve_all":
        s["auto_all"] = True
        p["decision"] = "approve"
    else:
        p["decision"] = "approve" if decision == "approve" else "deny"
    p["wake"].set()
    return True


async def await_decision(session_id, command, shell, *, timeout=300) -> str:
    s = _sess(session_id)
    if s["auto_all"]:
        return "approve"
    pid = uuid.uuid4().hex[:12]
    s["pending"][pid] = {"command": command, "shell": shell,
                         "wake": asyncio.Event(), "decision": None}
    try:
        await asyncio.wait_for(s["pending"][pid]["wake"].wait(), timeout)
        return s["pending"][pid]["decision"] or "deny"
    except asyncio.TimeoutError:
        return "deny"
    finally:
        s["pending"].pop(pid, None)

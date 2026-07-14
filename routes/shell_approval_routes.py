"""HTTP surface for the shell-command approval card. async so Event.set() runs
on the loop thread (AI Operator lesson)."""
from fastapi import APIRouter, Body, Request

from core.middleware import require_admin
from src.shell_exec.approval import set_decision, list_pending, reset_all


def setup_shell_approval_routes():
    router = APIRouter(prefix="/api/shell")

    @router.get("/pending")
    async def pending(request: Request):
        require_admin(request)
        return {"pending": list_pending(request.query_params.get("session_id"))}

    @router.post("/decision")
    async def decision(request: Request, body: dict = Body(...)):
        require_admin(request)
        ok = set_decision(body.get("session_id"), body.get("pending_id"),
                          body.get("decision"))
        return {"ok": ok}

    @router.post("/reset")
    async def reset(request: Request):
        require_admin(request)
        reset_all()          # consent turned off → drop every session's auto-approve
        return {"ok": True}

    return router

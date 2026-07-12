"""AI Operator session routes. One admin-only session at a time. The session
runs as a background task; the confirm/ask steps set pending state and await an
asyncio.Event that the /decision route fires."""
import asyncio
import logging

from fastapi import APIRouter, Request, HTTPException, Body

from core.middleware import require_admin
from src.settings import get_setting, load_settings, save_settings
from src.operator.session import run_operator, require_consent
from src.operator import primitives

logger = logging.getLogger(__name__)

# Single-session server state.
_SESSION = None  # dict or None


def _reset_for_test():
    global _SESSION
    _SESSION = None


def set_input_control(on: bool):
    s = load_settings()
    s["input_control_enabled"] = bool(on)
    save_settings(s)


def _new_state(goal, owner=None):
    return {"goal": goal, "status": "starting", "round": 0, "transcript": [],
            "pending": None, "_wake": asyncio.Event(), "_decision": None,
            "_answer": None, "_stop": False, "result": None, "owner": owner,
            "_task": None}


async def _run_session(state):
    """Drive run_operator with real primitives + UI-bridged confirm/ask."""
    ctx = {"owner": state.get("owner")}

    async def call_model(messages):
        from src.llm_core import llm_call_async
        from src.ai_interaction import _resolve_model  # (url, model, headers)
        from src.settings import get_user_setting
        owner = state.get("owner")
        spec = get_user_setting("default_model", owner=owner or "", default="") or ""
        url, model, headers = await asyncio.to_thread(_resolve_model, spec, owner)
        return await llm_call_async(url=url, model=model, messages=messages,
                                    headers=headers, temperature=0.2, max_tokens=800, timeout=90)

    async def perceive():
        return await primitives.real_perceive()

    async def decide(goal, history, percept):
        return await primitives.real_decide(goal, history, percept, call_model=call_model)

    async def execute(action):
        state["round"] += 1
        obs = await primitives.real_execute(action, ctx)
        state["transcript"].append({"tool": action.tool, "args": action.args,
                                    "rationale": action.rationale, "obs": obs})
        return obs

    async def confirm(action):
        if state["_stop"]:
            return "stop"
        state["pending"] = {"kind": "confirm", "tool": action.tool,
                            "args": action.args, "rationale": action.rationale}
        state["status"] = "awaiting_confirmation"
        state["_wake"].clear()
        await state["_wake"].wait()
        state["pending"] = None
        if state["_stop"]:
            return "stop"
        state["status"] = "running"
        return state["_decision"]

    async def ask(question):
        if state["_stop"]:
            return ""
        state["pending"] = {"kind": "ask", "question": question}
        state["status"] = "awaiting_answer"
        state["_wake"].clear()
        await state["_wake"].wait()
        state["pending"] = None
        if state["_stop"]:
            return ""
        state["status"] = "running"
        return state["_answer"]

    state["status"] = "running"
    try:
        result = await run_operator(
            state["goal"], perceive=perceive, decide=decide, execute=execute,
            confirm=confirm, ask=ask, should_stop=lambda: state["_stop"],
            max_rounds=int(get_setting("operator_max_rounds", 30)),
            max_seconds=int(get_setting("operator_max_seconds", 600)))
        state["result"] = result
        state["status"] = result["status"]
    except Exception as e:
        state["status"] = "error"
        state["result"] = {"status": "error", "error": str(e)}
        logger.warning("operator session error: %s", e)


def setup_operator_routes():
    router = APIRouter(prefix="/api/operator", tags=["operator"])

    @router.post("/start")
    async def start(request: Request, body: dict = Body(...)):
        require_admin(request)
        global _SESSION
        goal = (body.get("goal") or "").strip()
        if not goal:
            raise HTTPException(400, "goal is required")
        if _SESSION is not None and _SESSION["status"] in (
                "running", "starting", "awaiting_confirmation", "awaiting_answer"):
            raise HTTPException(409, "an operator session is already running")
        try:
            require_consent(get_setting)
        except PermissionError as e:
            raise HTTPException(400, str(e))
        owner = getattr(request.state, "current_user", None)
        _SESSION = _new_state(goal, owner)
        _SESSION["_task"] = asyncio.create_task(_run_session(_SESSION))
        return {"ok": True, "status": "starting"}

    @router.get("/status")
    def status(request: Request):
        require_admin(request)
        if _SESSION is None:
            return {"status": "idle"}
        return {"status": _SESSION["status"], "goal": _SESSION["goal"],
                "round": _SESSION["round"], "pending": _SESSION["pending"],
                "transcript": _SESSION["transcript"], "result": _SESSION["result"]}

    @router.post("/decision")
    async def decision(request: Request, body: dict = Body(...)):
        require_admin(request)
        if _SESSION is None or _SESSION["pending"] is None:
            raise HTTPException(409, "no pending decision")
        d = (body.get("decision") or "").strip().lower()
        if d == "stop":
            _SESSION["_stop"] = True
        elif _SESSION["pending"]["kind"] == "ask":
            _SESSION["_answer"] = body.get("answer", "")
        elif d == "edit":
            _args = body.get("args")
            _SESSION["_decision"] = ("edit", _args if isinstance(_args, dict) else {})
        elif d in ("approve", "deny"):
            _SESSION["_decision"] = d
        else:
            raise HTTPException(400, "decision must be approve|deny|edit|stop")
        _SESSION["_wake"].set()
        return {"ok": True}

    @router.post("/stop")
    async def stop(request: Request):
        require_admin(request)
        if _SESSION is not None:
            _SESSION["_stop"] = True
            _SESSION["status"] = "stopped"
            _SESSION["_wake"].set()
        set_input_control(False)
        return {"ok": True}

    return router

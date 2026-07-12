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


def _probe_endpoint_model(ep_id, owner):
    """(chat_url, model_id, headers) for the model ACTUALLY served on endpoint
    `ep_id` — whatever it currently reports on /v1/models — or (None, None, None).
    This is what makes the operator robust to a stale `default_model` that names
    a model no endpoint is serving (e.g. a vision-only model)."""
    import httpx
    from core.database import ModelEndpoint, SessionLocal
    from src.endpoint_resolver import resolve_endpoint_runtime
    from src.ai_interaction import build_models_url, build_chat_url, build_headers
    db = SessionLocal()
    try:
        ep = db.query(ModelEndpoint).filter(ModelEndpoint.id == ep_id).first()
    finally:
        db.close()
    if not ep:
        return (None, None, None)
    base, api_key = resolve_endpoint_runtime(ep, owner=owner)
    headers = build_headers(api_key, base)
    r = httpx.get(build_models_url(base), headers=headers, timeout=5)
    r.raise_for_status()
    data = r.json()
    items = data if isinstance(data, list) else (data.get("data") or [])
    ids = [m.get("id") for m in items if isinstance(m, dict) and m.get("id")]
    return (build_chat_url(base), (ids[0] if ids else None), headers)


def _resolve_operator_chat(owner, *, probe=None, resolve_model=None):
    """Resolve (url, model, headers) for the operator's decide step. Prefer the
    model actually SERVED on the user's default endpoint; fall back to
    _resolve_model(default_model). Robust to a stale default_model that names a
    model no endpoint currently serves."""
    probe = probe or _probe_endpoint_model
    ep_id = get_setting("default_endpoint_id", "") or ""
    if ep_id:
        try:
            url, model, headers = probe(ep_id, owner)
            if model:
                return (url, model, headers)
        except Exception as e:
            logger.warning("operator: default-endpoint probe failed (%s); "
                           "falling back to default_model", e)
    if resolve_model is None:
        from src.ai_interaction import _resolve_model
        resolve_model = _resolve_model
    from src.settings import get_user_setting
    spec = (get_user_setting("default_model", owner=owner or "", default="")
            or get_setting("default_model", "") or "")
    if not spec:
        raise ValueError("no chat model available — serve a tool-calling model "
                         "and select it as your default")
    return resolve_model(spec, owner)


async def _run_session(state):
    """Drive run_operator with real primitives + UI-bridged confirm/ask."""
    ctx = {"owner": state.get("owner")}

    async def call_model(messages):
        from src.llm_core import llm_call_async
        url, model, headers = await asyncio.to_thread(_resolve_operator_chat, state.get("owner"))
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

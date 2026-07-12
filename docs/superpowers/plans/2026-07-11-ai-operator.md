# AI Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A permissioned, goal-directed AI Operator that runs a bounded *perceive → decide → confirm → act → recapture* loop, executing one user-approved action at a time (reusing the shipped Desktop Control + Input Automation tools) until the goal is done or stopped.

**Architecture:** A pure, fully-injectable async loop (`src/operator/session.py`) tested with fakes (no real GUI/model/mouse); a structured one-action-per-round protocol (`src/operator/actions.py`); real adapters (`src/operator/primitives.py`) over `uia`/`windows`/`TOOL_HANDLERS`/`llm_call_async`; a route-driven session state machine (`routes/operator_routes.py`) that bridges confirmation to the UI via asyncio events; and a plain-IIFE Operator panel.

**Tech Stack:** Python stdlib + asyncio, FastAPI, pytest (`--import-mode=importlib`), vanilla JS (plain IIFE). Reuses `src.desktop.uia`/`windows`/`capture`, `src.agent_tools.TOOL_HANDLERS`, `src.llm_core.llm_call_async`. No new dependency.

## Global Constraints

- All pytest runs use `--import-mode=importlib`.
- **No new third-party dependency.** Pure Python over existing code.
- The operator's action vocabulary is the EXISTING tools: mutating = `{launch_app, control_window, click_element, set_element_text, mouse, keyboard}` (require confirmation); read-only = `{find_files, list_windows, list_ui_elements, capture_screen}` (auto-run).
- **Both consents required to start:** `screen_access_enabled` AND `input_control_enabled` must be True, else refuse.
- **Bounds:** `operator_max_rounds` default 30; `operator_max_seconds` default 600; plus no-progress detection (unchanged percept after a mutating act → stuck).
- **Per-action confirmation** on mutating actions only. **Panic stop** ends the session AND flips `input_control_enabled` off. Admin-gated routes; single active session; every round/action audit-logged.
- The loop core (`run_operator`) is async and takes ALL I/O as injected async callables so unit tests never touch a real screen/model/mouse.
- Follow existing patterns: routes like `routes/mcp_routes.py` (`setup_*_routes` → `APIRouter(prefix=...)`, `require_admin`); frontend a plain IIFE like `static/js/screenAccess.js`/`plugins.js`.

---

### Task 1: `src/operator/actions.py` — Action + vocabulary + parser

**Files:**
- Create: `src/operator/__init__.py` (empty), `src/operator/actions.py`
- Test: `tests/test_operator_actions.py`

**Interfaces:**
- Produces: `MUTATING_TOOLS`, `READONLY_TOOLS`, `OPERATOR_TOOLS` (sets); `Action` dataclass (`kind, tool, args, rationale`); `is_mutating(action)->bool`; `parse_action(reply:str)->Action` (malformed/unknown-tool → `Action(kind="ask")`).

- [ ] **Step 1 — failing test** `tests/test_operator_actions.py`:

```python
import src.operator.actions as a


def test_parse_act_mutating():
    act = a.parse_action('{"kind":"act","tool":"click_element","args":{"name":"Save"},"rationale":"click save"}')
    assert act.kind == "act" and act.tool == "click_element" and act.args == {"name": "Save"}
    assert a.is_mutating(act) is True


def test_parse_act_readonly_not_mutating():
    act = a.parse_action('{"kind":"act","tool":"list_ui_elements","args":{}}')
    assert act.kind == "act" and a.is_mutating(act) is False


def test_parse_done_wait_ask():
    assert a.parse_action('{"kind":"done","rationale":"finished"}').kind == "done"
    assert a.parse_action('{"kind":"wait"}').kind == "wait"
    assert a.parse_action('{"kind":"ask","rationale":"which file?"}').kind == "ask"


def test_parse_tolerates_json_fence_and_prose():
    act = a.parse_action('Sure!\n```json\n{"kind":"act","tool":"keyboard","args":{"action":"type","text":"hi"}}\n```')
    assert act.kind == "act" and act.tool == "keyboard"


def test_malformed_becomes_ask():
    assert a.parse_action("not json at all").kind == "ask"
    assert a.parse_action("").kind == "ask"
    assert a.parse_action("{bad json").kind == "ask"


def test_unknown_tool_becomes_ask():
    assert a.parse_action('{"kind":"act","tool":"rm_rf","args":{}}').kind == "ask"


def test_unknown_kind_becomes_ask():
    assert a.parse_action('{"kind":"teleport"}').kind == "ask"
```

- [ ] **Step 2 — run, expect FAIL:** `python -m pytest tests/test_operator_actions.py --import-mode=importlib -q`
- [ ] **Step 3 — implement.** Create empty `src/operator/__init__.py`, then `src/operator/actions.py`:

```python
"""Operator action protocol: the model returns exactly one structured Action
per round. The vocabulary reuses existing Desktop Control + Input Automation
tools. Anything malformed or disallowed becomes an `ask` — never a wild action."""
from dataclasses import dataclass, field
import json

MUTATING_TOOLS = {"launch_app", "control_window", "click_element",
                  "set_element_text", "mouse", "keyboard"}
READONLY_TOOLS = {"find_files", "list_windows", "list_ui_elements", "capture_screen"}
OPERATOR_TOOLS = MUTATING_TOOLS | READONLY_TOOLS


@dataclass
class Action:
    kind: str                         # "act" | "wait" | "ask" | "done"
    tool: str | None = None           # for kind == "act"
    args: dict = field(default_factory=dict)
    rationale: str = ""


def is_mutating(action):
    return action.kind == "act" and action.tool in MUTATING_TOOLS


def parse_action(reply):
    """Parse the model reply into ONE Action. The model must emit a JSON object
    {kind, tool?, args?, rationale?}. Malformed / unknown-tool / unknown-kind
    all degrade to Action(kind="ask") so the loop never executes a wild action."""
    text = (reply or "").strip()
    try:
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except (ValueError, TypeError):
        return Action(kind="ask", rationale="could not parse the model's action")
    kind = str(data.get("kind", "")).strip().lower()
    rationale = str(data.get("rationale", "") or data.get("question", ""))
    if kind in ("wait", "done", "ask"):
        return Action(kind=kind, rationale=rationale)
    if kind == "act":
        tool = str(data.get("tool", "")).strip()
        if tool not in OPERATOR_TOOLS:
            return Action(kind="ask", rationale=f"unknown or disallowed tool {tool!r}")
        args = data.get("args")
        return Action(kind="act", tool=tool,
                      args=args if isinstance(args, dict) else {}, rationale=rationale)
    return Action(kind="ask", rationale=f"unrecognized action kind {kind!r}")
```

(The `text.index("{")` on a fenced/prose reply grabs the first `{`…last `}` — that's what `test_parse_tolerates_json_fence_and_prose` verifies.)

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** (stage only the 3 files) `feat(operator): action protocol + parser`.

---

### Task 2: `src/operator/session.py` — the bounded loop + consent guard + settings

**Files:**
- Create: `src/operator/session.py`
- Modify: `src/settings.py` (`DEFAULT_SETTINGS`)
- Test: `tests/test_operator_session.py`

**Interfaces:**
- Consumes: `Action`, `is_mutating` (Task 1).
- Produces: `require_consent(get_setting)->None` (raises `PermissionError` unless both consents on); `async run_operator(goal, *, perceive, decide, execute, confirm, ask, max_rounds=30, max_seconds=600, now=time.monotonic) -> dict{status, rounds, history}`. All of `perceive/decide/execute/confirm/ask` are **async** callables. `confirm(action)` returns `"approve" | "deny" | "stop" | ("edit", new_args)`. `status ∈ {done, stopped, round_cap, time_cap, stuck}`.

- [ ] **Step 1 — failing test** `tests/test_operator_session.py`:

```python
import asyncio
import src.settings as settings
import src.operator.session as s
from src.operator.actions import Action


def _run(**kw):
    return asyncio.run(s.run_operator("goal", **kw))


def _seq(actions):
    it = iter(actions)
    async def decide(goal, history, percept):
        return next(it)
    return decide


async def _perceive_const():
    return "SCREEN"


async def _approve(action):
    return "approve"


async def _ask_noop(q):
    return "answer"


def test_default_operator_bounds():
    assert settings.DEFAULT_SETTINGS.get("operator_max_rounds") == 30
    assert settings.DEFAULT_SETTINGS.get("operator_max_seconds") == 600


def test_require_consent_needs_both(monkeypatch):
    import pytest
    store = {"screen_access_enabled": True, "input_control_enabled": False}
    with pytest.raises(PermissionError):
        s.require_consent(lambda k, d=None: store.get(k, d))
    store["input_control_enabled"] = True
    s.require_consent(lambda k, d=None: store.get(k, d))  # no raise


def test_stops_on_done():
    calls = []
    async def execute(a): calls.append(a.tool); return {"output": "ok"}
    r = _run(perceive=_perceive_const, decide=_seq([Action(kind="done", rationale="fin")]),
             execute=execute, confirm=_approve, ask=_ask_noop)
    assert r["status"] == "done" and calls == []


def test_mutating_requires_confirm_readonly_does_not():
    confirmed = []
    async def confirm(a): confirmed.append(a.tool); return "approve"
    async def execute(a): return {"output": "ok"}
    r = _run(perceive=_perceive_const,
             decide=_seq([Action(kind="act", tool="list_ui_elements"),      # read-only
                          Action(kind="act", tool="mouse", args={"action": "click", "x": 1, "y": 2}),  # mutating
                          Action(kind="done")]),
             execute=execute, confirm=confirm, ask=_ask_noop)
    assert confirmed == ["mouse"]  # only the mutating action was confirmed
    assert r["status"] == "done"


def test_deny_never_executes():
    executed = []
    async def execute(a): executed.append(a.tool); return {"output": "ok"}
    async def deny(a): return "deny"
    r = _run(perceive=_perceive_const,
             decide=_seq([Action(kind="act", tool="keyboard", args={"action": "type", "text": "x"}),
                          Action(kind="done")]),
             execute=execute, confirm=deny, ask=_ask_noop)
    assert executed == [] and r["status"] == "done"


def test_edit_replaces_args():
    seen = {}
    async def execute(a): seen.update(a.args); return {"output": "ok"}
    async def edit(a): return ("edit", {"action": "type", "text": "EDITED"})
    r = _run(perceive=_perceive_const,
             decide=_seq([Action(kind="act", tool="keyboard", args={"action": "type", "text": "orig"}),
                          Action(kind="done")]),
             execute=execute, confirm=edit, ask=_ask_noop)
    assert seen == {"action": "type", "text": "EDITED"}


def test_stop_ends_session():
    async def stop(a): return "stop"
    async def execute(a): return {"output": "ok"}
    r = _run(perceive=_perceive_const,
             decide=_seq([Action(kind="act", tool="mouse", args={"action": "click", "x": 1, "y": 1})]),
             execute=execute, confirm=stop, ask=_ask_noop)
    assert r["status"] == "stopped"


def test_round_cap():
    async def execute(a): return {"output": "ok"}
    # always returns a read-only act (auto-runs, never done) -> hits the cap
    async def decide(goal, history, percept): return Action(kind="act", tool="list_windows")
    r = asyncio.run(s.run_operator("g", perceive=_perceive_const, decide=decide,
                                   execute=execute, confirm=_approve, ask=_ask_noop, max_rounds=3))
    assert r["status"] == "round_cap" and r["rounds"] == 3


def test_no_progress_after_mutating_act_is_stuck():
    async def execute(a): return {"output": "ok"}
    async def decide(goal, history, percept):
        return Action(kind="act", tool="mouse", args={"action": "click", "x": 1, "y": 1})
    # percept never changes -> after the first mutating act, next round detects no progress
    r = asyncio.run(s.run_operator("g", perceive=_perceive_const, decide=decide,
                                   execute=execute, confirm=_approve, ask=_ask_noop, max_rounds=10))
    assert r["status"] == "stuck"


def test_time_cap():
    clock = {"t": 0.0}
    def now(): clock["t"] += 100.0; return clock["t"]
    async def decide(goal, history, percept): return Action(kind="act", tool="list_windows")
    async def execute(a): return {"output": "ok"}
    r = asyncio.run(s.run_operator("g", perceive=_perceive_const, decide=decide, execute=execute,
                                   confirm=_approve, ask=_ask_noop, max_rounds=100, max_seconds=150, now=now))
    assert r["status"] == "time_cap"
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement.** In `src/settings.py` `DEFAULT_SETTINGS`, after the `"input_control_enabled": False,` line add:

```python
    "operator_max_rounds": 30,
    "operator_max_seconds": 600,
```

Create `src/operator/session.py`:

```python
"""The bounded operator loop. All I/O is injected as async callables so the
loop unit-tests with fakes (no real screen/model/mouse)."""
import logging
import time

from src.operator.actions import is_mutating

logger = logging.getLogger(__name__)


def require_consent(get_setting):
    """Raise PermissionError unless BOTH screen access and input control are on."""
    if not get_setting("screen_access_enabled", False):
        raise PermissionError("Screen access is off — enable 'Allow screen access' in the sidebar.")
    if not get_setting("input_control_enabled", False):
        raise PermissionError("Input control is off — enable 'Allow input control' in the sidebar.")


async def run_operator(goal, *, perceive, decide, execute, confirm, ask,
                       max_rounds=30, max_seconds=600, now=time.monotonic):
    history = []
    start = now()
    last_act_percept = None  # percept captured just before the previous mutating act
    for rounds in range(1, max_rounds + 1):
        if now() - start > max_seconds:
            return {"status": "time_cap", "rounds": rounds - 1, "history": history}
        percept = await perceive()
        if last_act_percept is not None and percept == last_act_percept:
            return {"status": "stuck", "rounds": rounds, "history": history}
        last_act_percept = None
        action = await decide(goal, history, percept)
        if action.kind == "done":
            history.append(("done", action.rationale))
            logger.info("operator done after %d round(s)", rounds)
            return {"status": "done", "rounds": rounds, "history": history}
        if action.kind == "ask":
            answer = await ask(action.rationale)
            history.append(("ask", action.rationale, answer))
            continue
        if action.kind == "wait":
            history.append(("wait", action.rationale))
            continue
        # kind == "act"
        mutating = is_mutating(action)
        if mutating:
            decision = await confirm(action)
            if decision == "stop":
                logger.info("operator stopped by user at round %d", rounds)
                return {"status": "stopped", "rounds": rounds, "history": history}
            if decision == "deny":
                history.append(("denied", action.tool, action.args))
                continue
            if isinstance(decision, tuple) and decision and decision[0] == "edit":
                action.args = decision[1]
        obs = await execute(action)
        history.append(("act", action.tool, action.args, obs))
        logger.info("operator round %d: %s %s", rounds, action.tool, action.args)
        if mutating:
            last_act_percept = percept
    return {"status": "round_cap", "rounds": max_rounds, "history": history}
```

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(operator): bounded loop + consent guard + settings`.

---

### Task 3: `src/operator/primitives.py` — real perceive / execute / decide adapters

**Files:**
- Create: `src/operator/primitives.py`
- Test: `tests/test_operator_primitives.py`

**Interfaces:**
- Consumes: `src.desktop.uia`/`windows`, `src.agent_tools.TOOL_HANDLERS`, `parse_action`.
- Produces: `async real_perceive(*, get_root=..., list_elements=..., list_windows=...) -> dict{windows, elements}`; `async real_execute(action, ctx, *, handlers=None) -> dict`; `build_decide_prompt(goal, history, percept) -> list[dict]` (chat messages); `async real_decide(goal, history, percept, *, call_model) -> Action`.

- [ ] **Step 1 — failing test** `tests/test_operator_primitives.py`:

```python
import asyncio
import src.operator.primitives as p
from src.operator.actions import Action


def test_real_perceive_merges_windows_and_elements():
    async def run():
        return await p.real_perceive(
            get_root=lambda w: "ROOT",
            list_elements=lambda root: [{"name": "Save", "control_type": "Button"}],
            list_windows=lambda: [{"id": 1, "title": "Notepad"}])
    got = asyncio.run(run())
    assert got["windows"][0]["title"] == "Notepad"
    assert got["elements"][0]["name"] == "Save"


def test_real_perceive_tolerates_uia_failure():
    def boom(root): raise RuntimeError("no uia")
    async def run():
        return await p.real_perceive(get_root=lambda w: "ROOT", list_elements=boom,
                                     list_windows=lambda: [])
    got = asyncio.run(run())
    assert got["elements"] == []  # degrades, doesn't crash


def test_real_execute_dispatches_to_handler():
    seen = {}
    async def fake_handler(content, ctx):
        seen["content"] = content
        return {"output": "clicked", "exit_code": 0}
    act = Action(kind="act", tool="click_element", args={"name": "Save"})
    r = asyncio.run(p.real_execute(act, {"owner": "u"}, handlers={"click_element": fake_handler}))
    assert r["output"] == "clicked"
    import json
    assert json.loads(seen["content"]) == {"name": "Save"}


def test_real_execute_unknown_tool():
    act = Action(kind="act", tool="nope", args={})
    r = asyncio.run(p.real_execute(act, {}, handlers={}))
    assert r["exit_code"] == 1


def test_build_decide_prompt_includes_goal_and_elements():
    msgs = p.build_decide_prompt("do X", [], {"windows": [], "elements": [{"name": "Save"}]})
    blob = " ".join(m["content"] for m in msgs)
    assert "do X" in blob and "Save" in blob and "JSON" in blob


def test_real_decide_parses_model_reply():
    async def call_model(messages):
        return '{"kind":"act","tool":"click_element","args":{"name":"Save"}}'
    act = asyncio.run(p.real_decide("g", [], {"windows": [], "elements": []}, call_model=call_model))
    assert act.kind == "act" and act.tool == "click_element"
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `src/operator/primitives.py`:

```python
"""Real adapters bridging the operator loop to the shipped capabilities:
perception (UIA + windows), execution (existing TOOL_HANDLERS), and the decide
step (the agent model). Deps are injectable so these are unit-testable; the
actual model wire + real GUI are exercised at live-verify."""
import json

from src.operator.actions import OPERATOR_TOOLS, MUTATING_TOOLS, parse_action


async def real_perceive(*, get_root=None, list_elements=None, list_windows=None):
    from src.desktop import uia, windows
    get_root = get_root or uia.get_root
    list_elements = list_elements or uia.list_elements
    list_windows = list_windows or windows.list_windows
    try:
        wins = list_windows()
    except Exception:
        wins = []
    try:
        elements = list_elements(get_root("focused"))
    except Exception:
        elements = []
    return {"windows": wins, "elements": elements}


async def real_execute(action, ctx, *, handlers=None):
    if handlers is None:
        from src.agent_tools import TOOL_HANDLERS
        handlers = TOOL_HANDLERS
    handler = handlers.get(action.tool)
    if handler is None:
        return {"error": f"operator: unknown tool {action.tool}", "exit_code": 1}
    return await handler(json.dumps(action.args), ctx)


def build_decide_prompt(goal, history, percept):
    elems = "\n".join(
        f"- [{e.get('control_type','?')}] {e.get('name','')!r} id={e.get('automation_id','')!r}"
        for e in percept.get("elements", [])[:60])
    wins = ", ".join(w.get("title", "") for w in percept.get("windows", [])[:10])
    hist = "\n".join(str(h) for h in history[-8:])
    system = (
        "You are an on-screen operator. Each turn, return EXACTLY ONE next action "
        "as a single JSON object and nothing else. Schema: "
        '{"kind":"act|wait|ask|done","tool":"<tool>","args":{...},"rationale":"..."}. '
        f"Allowed tools: {sorted(OPERATOR_TOOLS)}. Mutating tools {sorted(MUTATING_TOOLS)} "
        "need user confirmation. Prefer click_element/set_element_text (target controls by "
        "name/automation_id) over raw mouse coordinates. Use capture_screen when you need to "
        "read on-screen text the element list lacks. Use kind=done when the goal is complete, "
        "kind=ask when you need the user, kind=wait to let the UI settle.")
    user = (f"GOAL: {goal}\n\nOPEN WINDOWS: {wins}\n\n"
            f"INTERACTABLE UI ELEMENTS:\n{elems or '(none)'}\n\n"
            f"RECENT HISTORY:\n{hist or '(none)'}\n\nReturn the next action as JSON.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def real_decide(goal, history, percept, *, call_model):
    reply = await call_model(build_decide_prompt(goal, history, percept))
    return parse_action(reply)
```

- [ ] **Step 4 — run, expect PASS.** Also `python -c "import src.operator.primitives"`. **Step 5 — commit** `feat(operator): real perceive/execute/decide adapters`.

---

### Task 4: `routes/operator_routes.py` — session state machine + routes

**Files:**
- Create: `routes/operator_routes.py`
- Modify: `app.py` (include the router)
- Test: `tests/test_operator_routes.py`

**Interfaces:**
- Consumes: `run_operator`, `require_consent` (Task 2); real primitives (Task 3); `get_setting`/`load_settings`/`save_settings`; `require_admin`.
- Produces: `setup_operator_routes() -> APIRouter` with `POST /api/operator/start {goal}`, `GET /api/operator/status`, `POST /api/operator/decision {decision, args?}`, `POST /api/operator/stop`. Module-level `_SESSION` holder + an asyncio-event confirm/ask bridge. Single active session.

- [ ] **Step 1 — failing test** `tests/test_operator_routes.py`:

```python
import asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.operator_routes as orr


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(orr, "require_admin", lambda request: None)
    # both consents on
    monkeypatch.setattr(orr, "get_setting",
                        lambda k, d=None: True if k in ("screen_access_enabled", "input_control_enabled") else d)
    orr._reset_for_test()
    app = FastAPI()
    app.include_router(orr.setup_operator_routes())
    return TestClient(app)


def test_start_refuses_without_consent(client, monkeypatch):
    monkeypatch.setattr(orr, "get_setting", lambda k, d=None: False)
    r = client.post("/api/operator/start", json={"goal": "x"})
    assert r.status_code == 400 and "off" in r.json()["detail"].lower()


def test_status_idle_before_start(client):
    r = client.get("/api/operator/status")
    assert r.status_code == 200 and r.json()["status"] == "idle"


def test_start_requires_goal(client):
    r = client.post("/api/operator/start", json={"goal": ""})
    assert r.status_code == 400


def test_single_session(client, monkeypatch):
    # make the session runner block so the first stays "active"
    async def fake_runner(state):
        state["status"] = "running"
        await asyncio.Event().wait()  # never returns
    monkeypatch.setattr(orr, "_run_session", fake_runner)
    assert client.post("/api/operator/start", json={"goal": "a"}).status_code == 200
    assert client.post("/api/operator/start", json={"goal": "b"}).status_code == 409  # already running


def test_stop_flips_input_control_off(client, monkeypatch):
    flipped = {}
    monkeypatch.setattr(orr, "set_input_control", lambda on: flipped.setdefault("v", on))
    async def fake_runner(state):
        state["status"] = "running"
        await asyncio.Event().wait()
    monkeypatch.setattr(orr, "_run_session", fake_runner)
    client.post("/api/operator/start", json={"goal": "a"})
    r = client.post("/api/operator/stop")
    assert r.status_code == 200 and flipped["v"] is False
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `routes/operator_routes.py`:

```python
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


def _new_state(goal):
    return {"goal": goal, "status": "starting", "round": 0, "transcript": [],
            "pending": None,            # {"kind": "confirm"|"ask", "action"/"question"}
            "_event": asyncio.Event(), "_decision": None, "_answer": None,
            "result": None}


async def _run_session(state):
    """Drive run_operator with real primitives + UI-bridged confirm/ask."""
    ctx = {"owner": None}

    async def call_model(messages):
        from src.llm_core import llm_call_async
        from src.ai_interaction import _resolve_model  # (url, model, headers)
        url, model, headers = await asyncio.to_thread(
            _resolve_model, get_setting("default_model", "") or "", None)
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
        state["pending"] = {"kind": "confirm", "tool": action.tool,
                            "args": action.args, "rationale": action.rationale}
        state["status"] = "awaiting_confirmation"
        state["_event"] = asyncio.Event()
        await state["_event"].wait()
        state["status"] = "running"
        state["pending"] = None
        return state["_decision"]

    async def ask(question):
        state["pending"] = {"kind": "ask", "question": question}
        state["status"] = "awaiting_answer"
        state["_event"] = asyncio.Event()
        await state["_event"].wait()
        state["status"] = "running"
        state["pending"] = None
        return state["_answer"]

    state["status"] = "running"
    try:
        result = await run_operator(
            state["goal"], perceive=perceive, decide=decide, execute=execute,
            confirm=confirm, ask=ask,
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
        _SESSION = _new_state(goal)
        asyncio.create_task(_run_session(_SESSION))
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
    def decision(request: Request, body: dict = Body(...)):
        require_admin(request)
        if _SESSION is None or _SESSION["pending"] is None:
            raise HTTPException(409, "no pending decision")
        d = (body.get("decision") or "").strip().lower()
        if _SESSION["pending"]["kind"] == "ask":
            _SESSION["_answer"] = body.get("answer", "")
        elif d == "edit":
            _SESSION["_decision"] = ("edit", body.get("args") or {})
        elif d in ("approve", "deny", "stop"):
            _SESSION["_decision"] = d
        else:
            raise HTTPException(400, "decision must be approve|deny|edit|stop")
        _SESSION["_event"].set()
        return {"ok": True}

    @router.post("/stop")
    def stop(request: Request):
        require_admin(request)
        global _SESSION
        if _SESSION is not None:
            _SESSION["_decision"] = "stop"
            _SESSION["status"] = "stopped"
            _SESSION["_event"].set()
        set_input_control(False)   # panic: flip input control off
        return {"ok": True}

    return router
```

In `app.py`, near the other `app.include_router(...)` calls (e.g. after the mcp router include), add:

```python
from routes.operator_routes import setup_operator_routes
app.include_router(setup_operator_routes())
```

**Plan-time verification (wire the model call):** `_run_session`'s `call_model` uses the real resolver `src.ai_interaction._resolve_model(spec, owner) -> (url, model, headers)` (the same one `src/agent_loop.py` / `document_processor.py` use), then `llm_call_async(url=, model=, messages=, headers=, ...)`. Confirm the correct model **spec** for the default chat model — passing `get_setting("default_model","")` is the starting point; if an empty/default spec doesn't resolve on this install, use the same spec the chat path uses (grep `_resolve_model(` call sites). This is the one real integration point, exercised at Task 6 live-verify; it does NOT affect the unit tests (they monkeypatch `_run_session`).

- [ ] **Step 4 — run, expect PASS.** Also `python -c "import routes.operator_routes"`. **Step 5 — commit** `feat(operator): session routes + state machine`.

---

### Task 5: `static/js/operator.js` + sidebar panel

**Files:**
- Modify: `static/index.html` (sidebar entry near `tool-plugins-btn`; an `operator-modal`; script include)
- Create: `static/js/operator.js`
- Test: `tests/test_operator_ui.py`

**Interfaces:**
- Consumes: `POST /api/operator/start`, `GET /api/operator/status`, `POST /api/operator/decision`, `POST /api/operator/stop`.
- Produces: sidebar `id="tool-operator-btn"`, `id="operator-modal"`, `id="operator-goal"`, `id="operator-transcript"`, `id="operator-active-indicator"`, Start + Stop buttons; a poll loop rendering pending action cards with Approve/Deny/Edit/Stop.

- [ ] **Step 1 — failing UI guard** `tests/test_operator_ui.py`:

```python
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
def _read(p): return (ROOT / p).read_text(encoding="utf-8")


def test_index_has_operator_entry_and_modal():
    html = _read("static/index.html")
    for el in ('id="tool-operator-btn"', 'id="operator-modal"', 'id="operator-transcript"',
               'id="operator-goal"', 'id="operator-active-indicator"'):
        assert el in html, f"{el} missing"
    assert 'src="/static/js/operator.js"' in html


def test_operator_js_wires_endpoints():
    js = _read("static/js/operator.js")
    for ep in ("/api/operator/start", "/api/operator/status",
               "/api/operator/decision", "/api/operator/stop"):
        assert ep in js, f"{ep} not wired"
```

- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement.** In `static/index.html`, add a sidebar entry after the Plugins item (`id="tool-plugins-btn"`):

```html
        <div class="list-item" id="tool-operator-btn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.5;"><circle cx="12" cy="12" r="3"/><path d="M12 1v6M12 17v6M4.2 4.2l4.3 4.3M15.5 15.5l4.3 4.3M1 12h6M17 12h6"/></svg>
          <span class="grow">AI Operator</span>
          <span id="operator-active-indicator" class="sidebar-notif-dot" style="display:none;background:var(--green,#50fa7b);margin-right:6px;" title="Operator active"></span>
        </div>
```

Add a modal before the Plugins modal:

```html
  <div id="operator-modal" class="modal hidden">
    <div class="modal-content admin-modal-content" role="dialog" aria-label="AI Operator">
      <div class="modal-header"><h4>AI Operator</h4>
        <button class="close-btn" id="close-operator-modal" aria-label="Close">&#x2716;</button></div>
      <div style="display:flex;gap:6px;margin-bottom:8px;">
        <input id="operator-goal" placeholder="Describe the task…" style="flex:1;">
        <button id="operator-start-btn">Start</button>
        <button id="operator-stop-btn" style="display:none;">Stop</button>
      </div>
      <div id="operator-transcript"></div>
      <div id="operator-msg" style="font-size:12px;opacity:0.8;margin-top:6px;"></div>
      <div style="font-size:11px;opacity:0.6;margin-top:6px;">Requires screen access + input control ON. Each action needs your approval.</div>
    </div>
  </div>
```

Add `<script src="/static/js/operator.js"></script>` beside the plugins.js include. Create `static/js/operator.js` (plain IIFE mirroring `plugins.js`): open on `tool-operator-btn` click; `Start` posts `/api/operator/start {goal}`; a `poll()` on an interval GETs `/api/operator/status` and renders the transcript + (when `pending`) an action card with the tool/args/rationale and **Approve/Deny/Edit/Stop** buttons that POST `/api/operator/decision {decision, args?}` (Edit prompts for JSON args); `Stop` posts `/api/operator/stop`; toggles `#operator-active-indicator` and the Stop button by whether status is active; surfaces errors into `#operator-msg`. All fetches use `credentials: 'same-origin'`; listeners via `addEventListener`/`element.onclick` (no inline handlers).

- [ ] **Step 4 — run, expect PASS.** **Step 5 — commit** `feat(operator): sidebar panel + operator.js`.

---

### Task 6: Package + live-verify

- [ ] **Step 1 — full affected suite:** `python -m pytest tests/test_operator_actions.py tests/test_operator_session.py tests/test_operator_primitives.py tests/test_operator_routes.py tests/test_operator_ui.py --import-mode=importlib -q` → all green.
- [ ] **Step 2 — build:** full clean `.\build-installer.ps1` (never `-Fast`). Confirm compile succeeds.
- [ ] **Step 3 — boot-verify** the frozen exe via `Assist.exe --run-py <probe.py> <out>`: import `src.operator.session`/`primitives`/`routes.operator_routes`; assert `require_consent` raises when a setting is off and passes when both on (inject a fake get_setting); assert `parse_action('{"kind":"done"}').kind == "done"`. Print `OPERATOR_BOOT=OK`/`FAIL`. (The full model-driven loop is the user's live test.)
- [ ] **Step 4 — user manual test** (packaged exe): turn ON screen access + input control; open **AI Operator**; goal "open Notepad, type 'hello world', then save as hello.txt"; approve each proposed action; confirm it opens Notepad (`launch_app`), types (`set_element_text`/`keyboard`), drives Save, reports `done`. Test **Deny** (model reconsiders) and **Stop** (session ends + input control flips OFF). Turn one consent off → **Start refuses**. Confirm Desktop Control / Input Automation / Network Scanner / Plugins still work.
- [ ] **Step 5 — commit** the installer: `git add -f installer/Output/Assist-Setup.exe && git commit -m "build: Assist-Setup.exe with AI Operator"`.

## Self-Review

- **Spec coverage:** action protocol + vocabulary + parser (T1); bounded loop with done/round-cap/time-cap/no-progress stops, mutating→confirm/read-only auto-run, deny/edit/stop, ask/wait, consent guard, settings (T2); real perceive/execute/decide adapters reusing uia/windows/TOOL_HANDLERS/llm_call_async (T3); session state machine + routes (start/status/decision/stop), single-session, admin-gated, panic-stop flips input control off (T4); sidebar panel + operator.js with approve/deny/edit/stop + active indicator (T5); package + live-verify incl. both-consent refusal + the Notepad acceptance goal (T6). All spec sections covered.
- **Placeholders:** none — every code step is complete. The one real integration point (`resolve_default_chat` for the model wire) is flagged as a plan-time verification with a concrete grep target, not a code placeholder; it is outside the unit-tested surface (routes tests monkeypatch `_run_session`).
- **Type consistency:** `Action(kind,tool,args,rationale)` identical across T1/T2/T3; `run_operator(...)` signature + injected async `perceive/decide/execute/confirm/ask` consistent T2↔T4; `confirm` returns `"approve"|"deny"|"stop"|("edit",args)` in T2 tests and T4 bridge; `real_perceive/real_execute/real_decide` signatures consistent T3↔T4; endpoint paths (`/api/operator/start|status|decision|stop`) identical T4↔T5; consent keys `screen_access_enabled`/`input_control_enabled` consistent throughout.

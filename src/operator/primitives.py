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

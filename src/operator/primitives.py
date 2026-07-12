"""Real adapters bridging the operator loop to the shipped capabilities:
perception (UIA + windows), execution (existing TOOL_HANDLERS), and the decide
step (the agent model). Deps are injectable so these are unit-testable; the
actual model wire + real GUI are exercised at live-verify."""
import json
import logging

from src.operator.actions import OPERATOR_TOOLS, MUTATING_TOOLS, READONLY_TOOLS, parse_action

logger = logging.getLogger(__name__)


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
        "You drive the on-screen GUI to complete the goal. Each turn, reply with "
        "EXACTLY ONE JSON object and NOTHING else (no prose, no markdown, no code fence). "
        'Schema: {"kind":"act|wait|ask|done","tool":"<tool>","args":{...},"rationale":"..."}. '
        f"You may ONLY use these tools: {sorted(OPERATOR_TOOLS)}. "
        "You have NO file, shell, or document tools — do NOT emit write_file, bash, python, "
        "create_document, or anything not in that list. Accomplish file tasks through the GUI "
        "(open the app with launch_app, type with keyboard/set_element_text, use its Save dialog). "
        f"Mutating tools {sorted(MUTATING_TOOLS)} require user confirmation; read-only tools "
        f"{sorted(READONLY_TOOLS)} run automatically. Prefer click_element/set_element_text "
        "(target controls by name/automation_id from the element list) over raw mouse coordinates. "
        "Use capture_screen only when you need on-screen text the element list lacks. "
        "kind=done when the goal is complete; kind=ask ONLY to ask the user a real question; "
        "kind=wait to let the UI settle. If RECENT HISTORY contains an ('invalid', ...) entry, "
        "your previous reply was rejected — fix it and pick a valid tool from the list.\n"
        "Put every parameter INSIDE the \"args\" object, using these exact keys:\n"
        '  launch_app {"name":"<app/file/url>"}   (key is "name", NOT "app")\n'
        '  click_element {"window_id":<id>|"focused","name":"<label>"}  (or automation_id / control_type / nth)\n'
        '  set_element_text {"window_id":<id>,"automation_id":"<id>","text":"<text>"}\n'
        '  mouse {"action":"move|click|double|right|drag|scroll","x":<int>,"y":<int>,"to_x"?:<int>,"to_y"?:<int>,"amount"?:<int>}\n'
        '  keyboard {"action":"type","text":"<text>"}  OR  {"action":"hotkey","keys":["ctrl","s"]}\n'
        '  control_window {"id":<id>,"action":"focus|minimize|maximize|restore|close"}\n'
        '  find_files {"query":"<name>"}   list_ui_elements {"window_id":<id>|"focused"}   '
        'list_windows {}   capture_screen {}')
    user = (f"GOAL: {goal}\n\nOPEN WINDOWS: {wins}\n\n"
            f"INTERACTABLE UI ELEMENTS:\n{elems or '(none)'}\n\n"
            f"RECENT HISTORY:\n{hist or '(none)'}\n\nReturn the next action as a single JSON object.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def real_decide(goal, history, percept, *, call_model):
    reply = await call_model(build_decide_prompt(goal, history, percept))
    action = parse_action(reply)
    logger.info("operator decide -> kind=%s tool=%s | raw reply: %r",
                action.kind, action.tool, (reply or "")[:300])
    return action

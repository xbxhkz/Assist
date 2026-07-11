"""Input-automation tools: raw mouse/keyboard (SendInput) and UI-Automation
element targeting. Acting tools are gated by input_control_enabled; the reading
tool list_ui_elements is gated by screen_access_enabled. Thin wrappers over
src.desktop.inputraw / src.desktop.uia."""
import json
import logging

from src.settings import get_setting
from src.desktop import inputraw
from src.desktop import uia

logger = logging.getLogger(__name__)

_OFF_MSG = ("Input control is off. Ask the user to enable 'Allow input control' "
            "in the sidebar before moving the mouse or typing.")


def _args(content):
    try:
        return json.loads(content) if content.strip().startswith("{") else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _input_on():
    return bool(get_setting("input_control_enabled", False))


class MouseTool:
    async def execute(self, content, ctx):
        if not _input_on():
            return {"error": _OFF_MSG, "exit_code": 1}
        a = _args(content)
        action = (a.get("action") or "").strip().lower()
        try:
            if action in ("click", "double", "right"):
                inputraw.click(int(a["x"]), int(a["y"]),
                               button="right" if action == "right" else "left",
                               double=(action == "double"))
            elif action == "move":
                inputraw.move(int(a["x"]), int(a["y"]))
            elif action == "drag":
                inputraw.drag(int(a["x"]), int(a["y"]), int(a["to_x"]), int(a["to_y"]))
            elif action == "scroll":
                inputraw.scroll(int(a.get("amount", 0)))
            else:
                return {"error": "mouse: action must be move|click|double|right|drag|scroll",
                        "exit_code": 1}
        except (KeyError, ValueError, TypeError) as e:
            return {"error": f"mouse: bad args ({e})", "exit_code": 1}
        logger.info("mouse: %s %s", action, {k: a.get(k) for k in ("x", "y", "to_x", "to_y", "amount")})
        return {"output": f"mouse {action} ok", "exit_code": 0}


class KeyboardTool:
    async def execute(self, content, ctx):
        if not _input_on():
            return {"error": _OFF_MSG, "exit_code": 1}
        a = _args(content)
        action = (a.get("action") or "").strip().lower()
        try:
            if action == "type":
                inputraw.type_text(str(a.get("text", "")))
            elif action == "hotkey":
                keys = a.get("keys") or []
                if not isinstance(keys, list) or not keys:
                    return {"error": "keyboard: hotkey needs a non-empty keys list", "exit_code": 1}
                inputraw.press_keys([str(k) for k in keys])
            else:
                return {"error": "keyboard: action must be type|hotkey", "exit_code": 1}
        except ValueError as e:
            return {"error": f"keyboard: {e}", "exit_code": 1}
        logger.info("keyboard: %s", action)
        return {"output": f"keyboard {action} ok", "exit_code": 0}

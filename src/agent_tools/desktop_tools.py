"""Native desktop-control tools. Thin wrappers over src/desktop backends;
gates and formatting only. Screen capture is consent-gated."""
import base64
import json
import logging

from src.settings import get_setting
from src.desktop.apps import resolve_app, launch
from src.desktop.filesearch import search
from src.desktop.windows import list_windows, control_window
from src.desktop.capture import capture_png

logger = logging.getLogger(__name__)


def _args(content):
    try:
        return json.loads(content) if content.strip().startswith("{") else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _vision_ready():
    return bool(get_setting("vision_enabled", True) and get_setting("vision_model", ""))


class LaunchAppTool:
    async def execute(self, content, ctx):
        name = (_args(content).get("name") or content).strip()
        if not name or name.startswith("{"):
            return {"error": "launch_app: name required", "exit_code": 1}
        target = resolve_app(name)
        if not target:
            return {"error": f"launch_app: could not find an app named {name!r}", "exit_code": 1}
        try:
            launch(target)
        except Exception as e:
            return {"error": f"launch_app: {e}", "exit_code": 1}
        logger.info("launch_app: %s -> %s", name, target["target"])
        return {"output": f"Launched {target['name']} ({target['target']})", "exit_code": 0}


class FindFilesTool:
    async def execute(self, content, ctx):
        a = _args(content)
        query = (a.get("query") or "").strip()
        if not query:
            return {"error": "find_files: query required", "exit_code": 1}
        from src.constants import DATA_DIR
        import os
        roots = [os.path.expanduser("~")]
        extra = get_setting("tool_path_extra_roots") or []
        roots += [str(r) for r in extra if r]
        if a.get("all_drives"):
            roots += [f"{d}:\\" for d in "CDEFG" if os.path.isdir(f"{d}:\\")]
        hits = search(query, roots=roots, ext=a.get("ext"),
                      all_drives=bool(a.get("all_drives")),
                      max_results=int(a.get("max_results") or 200))
        if not hits:
            return {"output": f"No files matching {query!r}", "exit_code": 0}
        lines = [f"{h['path']}  ({h['size']} bytes)" for h in hits]
        return {"output": f"{len(hits)} match(es):\n" + "\n".join(lines), "exit_code": 0}


class ListWindowsTool:
    async def execute(self, content, ctx):
        wins = list_windows()
        if not wins:
            return {"output": "No visible windows", "exit_code": 0}
        lines = [f"[{w['id']}] {w['title']} — {w['process']} (pid {w['pid']}, {w['state']})"
                 for w in wins]
        return {"output": "\n".join(lines), "exit_code": 0}


class ControlWindowTool:
    async def execute(self, content, ctx):
        a = _args(content)
        action = (a.get("action") or "").strip().lower()
        wid = a.get("id")
        if wid is None or action not in ("focus", "minimize", "maximize", "restore", "close"):
            return {"error": "control_window: id and action (focus|minimize|maximize|restore|close) required", "exit_code": 1}
        try:
            ok = control_window(int(wid), action)
        except Exception as e:
            return {"error": f"control_window: {e}", "exit_code": 1}
        logger.info("control_window: %s %s", wid, action)
        return {"output": f"{action} window {wid}" if ok else f"control_window: {action} failed",
                "exit_code": 0 if ok else 1}


class CaptureScreenTool:
    async def execute(self, content, ctx):
        if not get_setting("screen_access_enabled", False):
            return {"error": "Screen access is off. Ask the user to enable 'Allow screen "
                             "access' in the sidebar before capturing the screen.", "exit_code": 1}
        if not _vision_ready():
            return {"error": "capture_screen: no vision model configured (Settings → vision_model).",
                    "exit_code": 1}
        target = (_args(content).get("target") or "full").strip()
        try:
            png = capture_png(target)
        except Exception as e:
            return {"error": f"capture_screen: {e}", "exit_code": 1}
        logger.info("capture_screen: %s", target)
        uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        return {"output": f"{target} screenshot captured", "image_url": uri, "exit_code": 0}

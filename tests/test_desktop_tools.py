import asyncio
import base64
import json

import src.agent_tools.desktop_tools as dt


def _run(tool, content):
    return asyncio.run(tool.execute(content, {}))


def test_launch_app_resolves_and_launches(monkeypatch):
    launched = {}
    monkeypatch.setattr(dt, "resolve_app", lambda n: {"name": n, "target": "np.exe", "kind": "exe"})
    monkeypatch.setattr(dt, "launch", lambda t: launched.setdefault("t", t))
    res = _run(dt.LaunchAppTool(), json.dumps({"name": "Notepad"}))
    assert res["exit_code"] == 0 and launched["t"]["target"] == "np.exe"


def test_launch_app_unknown(monkeypatch):
    monkeypatch.setattr(dt, "resolve_app", lambda n: None)
    res = _run(dt.LaunchAppTool(), json.dumps({"name": "nope"}))
    assert res["exit_code"] == 1 and "could not find" in res["error"].lower()


def test_find_files_formats_hits(monkeypatch):
    monkeypatch.setattr(dt, "search", lambda q, **k: [{"path": "C:/a.py", "size": 3, "modified": 0}])
    res = _run(dt.FindFilesTool(), json.dumps({"query": "a"}))
    assert "C:/a.py" in res["output"] and res["exit_code"] == 0


def test_list_windows_formats(monkeypatch):
    monkeypatch.setattr(dt, "list_windows", lambda: [{"id": 11, "title": "Notepad", "pid": 1, "process": "notepad.exe", "state": "normal"}])
    res = _run(dt.ListWindowsTool(), "{}")
    assert "Notepad" in res["output"] and "11" in res["output"]


def test_control_window_dispatches(monkeypatch):
    seen = {}
    monkeypatch.setattr(dt, "control_window", lambda wid, action: seen.update(id=wid, a=action) or True)
    res = _run(dt.ControlWindowTool(), json.dumps({"id": 11, "action": "focus"}))
    assert res["exit_code"] == 0 and seen == {"id": 11, "a": "focus"}


def test_capture_refuses_when_toggle_off(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: False)
    res = _run(dt.CaptureScreenTool(), json.dumps({"target": "full"}))
    assert res["exit_code"] == 1 and "screen access is off" in res["error"].lower()


def test_capture_emits_data_uri_when_on(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: True if k == "screen_access_enabled" else d)
    monkeypatch.setattr(dt, "_vision_ready", lambda: True)
    monkeypatch.setattr(dt, "capture_png", lambda target, **k: b"\x89PNG\r\n\x1a\nDATA")
    monkeypatch.setattr(dt, "analyze_image_with_vl_result",
                         lambda path, owner=None: {"text": "A desktop with a code editor open", "model": "vl"})
    res = _run(dt.CaptureScreenTool(), json.dumps({"target": "full"}))
    assert res["exit_code"] == 0
    assert res["image_url"].startswith("data:image/png;base64,")
    assert base64.b64decode(res["image_url"].split(",", 1)[1]).startswith(b"\x89PNG")
    assert "code editor" in res["output"]


def test_capture_needs_vision_model(monkeypatch):
    monkeypatch.setattr(dt, "get_setting", lambda k, d=None: True if k == "screen_access_enabled" else d)
    monkeypatch.setattr(dt, "_vision_ready", lambda: False)
    res = _run(dt.CaptureScreenTool(), json.dumps({"target": "full"}))
    assert res["exit_code"] == 1 and "vision model" in res["error"].lower()

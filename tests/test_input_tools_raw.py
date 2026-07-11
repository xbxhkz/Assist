import asyncio
import json
import src.agent_tools.input_tools as it


def _run(tool, payload, enabled=True, monkeypatch=None):
    monkeypatch.setattr(it, "get_setting",
                        lambda k, d=None: enabled if k == "input_control_enabled" else d)
    return asyncio.run(tool.execute(json.dumps(payload), {"owner": "u"}))


def test_mouse_refuses_when_input_control_off(monkeypatch):
    r = _run(it.MouseTool(), {"action": "click", "x": 1, "y": 2}, enabled=False, monkeypatch=monkeypatch)
    assert r["exit_code"] == 1 and "input control" in r["error"].lower()


def test_mouse_click_calls_backend(monkeypatch):
    calls = {}
    monkeypatch.setattr(it.inputraw, "click", lambda x, y, **k: calls.update(x=x, y=y, kw=k))
    r = _run(it.MouseTool(), {"action": "click", "x": 40, "y": 50}, monkeypatch=monkeypatch)
    assert r["exit_code"] == 0 and calls["x"] == 40 and calls["y"] == 50


def test_mouse_scroll_calls_backend(monkeypatch):
    got = {}
    monkeypatch.setattr(it.inputraw, "scroll", lambda amount, **k: got.update(a=amount))
    r = _run(it.MouseTool(), {"action": "scroll", "amount": -3}, monkeypatch=monkeypatch)
    assert r["exit_code"] == 0 and got["a"] == -3


def test_mouse_bad_action_errors(monkeypatch):
    r = _run(it.MouseTool(), {"action": "teleport"}, monkeypatch=monkeypatch)
    assert r["exit_code"] == 1


def test_keyboard_type_calls_backend(monkeypatch):
    got = {}
    monkeypatch.setattr(it.inputraw, "type_text", lambda s, **k: got.update(s=s))
    r = _run(it.KeyboardTool(), {"action": "type", "text": "hello"}, monkeypatch=monkeypatch)
    assert r["exit_code"] == 0 and got["s"] == "hello"


def test_keyboard_hotkey_calls_backend(monkeypatch):
    got = {}
    monkeypatch.setattr(it.inputraw, "press_keys", lambda keys, **k: got.update(keys=keys))
    r = _run(it.KeyboardTool(), {"action": "hotkey", "keys": ["ctrl", "s"]}, monkeypatch=monkeypatch)
    assert r["exit_code"] == 0 and got["keys"] == ["ctrl", "s"]


def test_keyboard_refuses_when_off(monkeypatch):
    r = _run(it.KeyboardTool(), {"action": "type", "text": "x"}, enabled=False, monkeypatch=monkeypatch)
    assert r["exit_code"] == 1 and "input control" in r["error"].lower()

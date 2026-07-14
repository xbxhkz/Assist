import asyncio
import json
import src.agent_tools.shell_tools as st


def _run(tool, content, ctx=None):
    return asyncio.run(tool.execute(content, ctx or {}))


def test_refuses_when_consent_off(monkeypatch):
    monkeypatch.setattr(st, "get_setting", lambda k, d=None: False)
    res = _run(st.PowerShellTool(), json.dumps({"command": "Get-Process"}))
    assert res["exit_code"] == 1 and "Allow shell commands" in res["error"]


def test_read_runs_without_staging(monkeypatch):
    monkeypatch.setattr(st, "get_setting", lambda k, d=None: True)
    async def fake_run(command, shell, ctx): return {"output": "ok", "exit_code": 0}
    monkeypatch.setattr(st, "run_in_shell", fake_run)
    called = {"await": 0}
    async def fake_await(*a, **k): called["await"] += 1; return "approve"
    monkeypatch.setattr(st, "await_decision", fake_await)
    res = _run(st.PowerShellTool(), json.dumps({"command": "Get-Process"}))
    assert res == {"output": "ok", "exit_code": 0} and called["await"] == 0


def test_write_denied(monkeypatch):
    monkeypatch.setattr(st, "get_setting", lambda k, d=None: True)
    async def fake_await(*a, **k): return "deny"
    monkeypatch.setattr(st, "await_decision", fake_await)
    async def fake_run(*a, **k): raise AssertionError("must not run on deny")
    monkeypatch.setattr(st, "run_in_shell", fake_run)
    res = _run(st.CmdTool(), json.dumps({"command": "del x"}), {"session_id": "s"})
    assert res["exit_code"] == 1 and "denied" in res["output"].lower()


def test_write_approved_runs(monkeypatch):
    monkeypatch.setattr(st, "get_setting", lambda k, d=None: True)
    async def fake_await(*a, **k): return "approve"
    monkeypatch.setattr(st, "await_decision", fake_await)
    async def fake_run(command, shell, ctx): return {"output": "done", "exit_code": 0}
    monkeypatch.setattr(st, "run_in_shell", fake_run)
    res = _run(st.CmdTool(), json.dumps({"command": "del x"}), {"session_id": "s"})
    assert res == {"output": "done", "exit_code": 0}


def test_registered_in_handlers():
    from src.agent_tools import TOOL_HANDLERS
    assert "powershell" in TOOL_HANDLERS and "cmd" in TOOL_HANDLERS

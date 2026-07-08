"""Frozen-app MCP child dispatch.

In the frozen build sys.executable is Assist.exe. Built-in MCP servers were
spawned as `<python> <script.py>` — frozen, that boots the FULL APP, which
registers its own MCP servers, recursively (observed live: 21 Assist.exe
processes / 7.4GB RAM). Fix: children are spawned as
`Assist.exe --run-mcp <script.py>`, the launcher dispatches that to runpy
before booting anything, and a child-env marker hard-stops recursion.
"""
import asyncio
import os
import sys

import src.mcp_child_dispatch as dispatch
import src.builtin_mcp as builtin


def test_maybe_dispatch_ignores_normal_argv():
    assert dispatch.maybe_dispatch(["Assist.exe"]) is False
    assert dispatch.maybe_dispatch(["Assist.exe", "--something"]) is False


def test_maybe_dispatch_runs_script(tmp_path):
    marker = tmp_path / "ran.txt"
    script = tmp_path / "fake_server.py"
    script.write_text(f"open(r'{marker}', 'w').write('yes')\n")
    assert dispatch.maybe_dispatch(["Assist.exe", "--run-mcp", str(script)]) is True
    assert marker.read_text() == "yes"


def test_maybe_dispatch_missing_script_still_true(tmp_path):
    # The flag was given: we must NOT fall through and boot the app.
    assert dispatch.maybe_dispatch(
        ["Assist.exe", "--run-mcp", str(tmp_path / "gone.py")]) is True


def test_python_server_cmd_dev_mode():
    cmd, args = builtin._python_server_cmd("/x/server.py")
    assert cmd == sys.executable
    assert args == ["/x/server.py"]


def test_python_server_cmd_frozen_mode(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    cmd, args = builtin._python_server_cmd("/x/server.py")
    assert cmd == sys.executable
    assert args == ["--run-mcp", "/x/server.py"]


def test_register_builtin_servers_noops_in_mcp_child(monkeypatch):
    """Even if a child ever boots the app again, it must not spawn more."""
    monkeypatch.setenv("ASSIST_MCP_CHILD", "1")

    class FakeManager:
        called = False
        async def connect_server(self, **kw):
            FakeManager.called = True
            return True

    asyncio.run(builtin.register_builtin_servers(FakeManager()))
    assert FakeManager.called is False


def test_children_are_marked(monkeypatch):
    """The spawn env must carry the recursion-guard marker."""
    import inspect
    src_text = inspect.getsource(builtin)
    assert "ASSIST_MCP_CHILD" in src_text

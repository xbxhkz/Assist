"""open_in_vscode agent tool: launch VS Code at a permitted path.

Path resolution goes through _resolve_tool_path, so workspace/allowlist
confinement applies — the model can only open what the user has granted.
The launcher and CLI lookup are injected; no real VS Code in tests.
"""
import asyncio
import json

import src.tool_execution as tex
from src.agent_tools.filesystem_tools import OpenInVscodeTool


def _run(tool, content):
    return asyncio.run(tool.execute(content, {}))


def test_opens_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(tex, "_resolve_tool_path", lambda p: str(tmp_path))
    launched = []
    tool = OpenInVscodeTool(launcher=launched.append,
                            which=lambda n: "C:/vscode/bin/code.cmd")
    res = _run(tool, json.dumps({"path": str(tmp_path)}))
    assert res["exit_code"] == 0
    assert launched == [["C:/vscode/bin/code.cmd", str(tmp_path)]]


def test_opens_file_at_line(monkeypatch, tmp_path):
    f = tmp_path / "main.py"
    f.write_text("x")
    monkeypatch.setattr(tex, "_resolve_tool_path", lambda p: str(f))
    launched = []
    tool = OpenInVscodeTool(launcher=launched.append,
                            which=lambda n: "code")
    res = _run(tool, json.dumps({"path": str(f), "line": 42}))
    assert res["exit_code"] == 0
    assert launched == [["code", "--goto", f"{f}:42"]]


def test_confinement_rejection_is_surfaced(monkeypatch):
    def _deny(p):
        raise ValueError("path 'C:/secret' is outside the allowed roots")
    monkeypatch.setattr(tex, "_resolve_tool_path", _deny)
    tool = OpenInVscodeTool(launcher=lambda argv: None, which=lambda n: "code")
    res = _run(tool, json.dumps({"path": "C:/secret"}))
    assert res["exit_code"] == 1
    assert "outside the allowed roots" in res["error"]


def test_missing_vscode_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(tex, "_resolve_tool_path", lambda p: str(tmp_path))
    tool = OpenInVscodeTool(launcher=lambda argv: None, which=lambda n: None)
    res = _run(tool, json.dumps({"path": str(tmp_path)}))
    assert res["exit_code"] == 1
    assert "VS Code" in res["error"]


def test_path_required():
    tool = OpenInVscodeTool(launcher=lambda argv: None, which=lambda n: "code")
    res = _run(tool, "{}")
    assert res["exit_code"] == 1

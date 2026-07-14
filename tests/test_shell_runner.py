import asyncio
import src.shell_exec.runner as r


def test_powershell_argv_shape():
    argv = r.powershell_argv("Get-Process")
    assert argv[-3:] == ["-NonInteractive", "-Command", "Get-Process"]
    assert argv[1] == "-NoProfile"
    assert "powershell" in argv[0].lower() or "pwsh" in argv[0].lower()


def test_cmd_argv_shape():
    argv = r.cmd_argv("dir")
    assert argv[-2:] == ["/c", "dir"]
    assert argv[0].lower().endswith("cmd") or argv[0].lower().endswith("cmd.exe")


def test_run_in_shell_passthrough(monkeypatch):
    captured = {}
    async def fake_spawn(*argv, **kw):
        captured["argv"] = argv
        return "PROC"
    async def fake_stream(proc, *, timeout, progress_cb):
        captured["proc"] = proc
        return ("hello", "", 0, False)
    monkeypatch.setattr(r, "agent_cwd", lambda: ".", raising=False)
    out = asyncio.run(r.run_in_shell("dir", "cmd", {"subproc_env": None},
                                     spawn=fake_spawn, stream=fake_stream))
    assert out == {"output": "hello", "exit_code": 0}
    assert captured["proc"] == "PROC" and captured["argv"][-1] == "dir"


def test_run_in_shell_timeout(monkeypatch):
    async def fake_spawn(*argv, **kw): return "PROC"
    async def fake_stream(proc, *, timeout, progress_cb): return ("", "", None, True)
    monkeypatch.setattr(r, "agent_cwd", lambda: ".", raising=False)
    out = asyncio.run(r.run_in_shell("ping -t x", "cmd", {},
                                     spawn=fake_spawn, stream=fake_stream))
    assert out["exit_code"] == 124 and "timed out" in out["error"]

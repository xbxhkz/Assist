# AI Shell Execution (PowerShell / cmd) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the AI consent-gated `powershell` and `cmd` tools — read-only commands auto-run; state-changing commands prompt Approve / Deny / Auto-approve-all — leaving the existing `bash` tool untouched.

**Architecture:** A new `src/shell_exec/` package with four focused units (classify, gate, runner, approval bridge), two new agent tools (`PowerShellTool`, `CmdTool`) that compose them, a `/api/shell/decision` route mirroring the AI Operator's asyncio wake-Event bridge, and two small CSP-safe JS files (a chat approval card + a sidebar consent switch).

**Tech Stack:** Python 3 (asyncio), pytest (`--import-mode=importlib`), FastAPI routes, vanilla CSP-safe JS. No new dependencies.

## Global Constraints

- New tools are `powershell` and `cmd` only; the existing `bash` tool is NOT modified.
- `shell_exec_enabled`: a **global** setting, default `False`, added to `src/settings.py` and **reset to `False` at startup** exactly like `screen_access_enabled` / `input_control_enabled` (reset fns at `src/settings.py:353-368`, called at `app.py:985-987`).
- Classifier is conservative: a command is `"read"` **only if** it contains none of the compose/redirect chars `| > < ; & \n` backtick `$(` `&&` `||` **and** its leading token is on the per-shell read-only allowlist; **everything else is `"write"`** (any pipe/redirect/chain or unknown token).
- Approval bridge: blocking `await_decision`, keyed on the chat `session_id` from the tool `ctx` (`src/tool_execution.py:538,560`). approve→run; deny→"denied by the user"; `auto_approve_all`→set a per-session flag and approve; **timeout 300 s → deny** (fail-closed). Session store resets on new session and when consent is turned off.
- The `/api/shell/decision` route is `async def` so `Event.set()` runs on the loop thread (AI Operator lesson).
- UI is CSP-safe like `static/js/operator.js`: `createElement` + `addEventListener` only, all dynamic text via `textContent`.
- Run pytest with `--import-mode=importlib`. Stage only the files each task names — never `git add -A`.

---

### Task 1: Command classifier

**Files:**
- Create: `src/shell_exec/__init__.py` (empty), `src/shell_exec/classify.py`
- Test: `tests/test_shell_classify.py`

**Interfaces:**
- Produces: `classify_command(command: str, shell: str) -> str` returning `"read"` or `"write"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shell_classify.py`:

```python
from src.shell_exec.classify import classify_command as c


def test_powershell_reads():
    assert c("Get-ChildItem", "powershell") == "read"
    assert c("get-process", "powershell") == "read"           # case-insensitive
    assert c("Test-Path C:\\x", "powershell") == "read"
    assert c("Format-Table", "powershell") == "read"


def test_powershell_writes_and_unknown():
    assert c("Remove-Item x", "powershell") == "write"
    assert c("Set-Content x y", "powershell") == "write"
    assert c("frobnicate", "powershell") == "write"           # unknown → write


def test_cmd_reads_and_writes():
    assert c("dir", "cmd") == "read"
    assert c("type foo.txt", "cmd") == "read"
    assert c("del foo.txt", "cmd") == "write"
    assert c("set FOO=bar", "cmd") == "write"                 # not on allowlist


def test_compose_chars_force_write():
    assert c("Get-Content x | Remove-Item", "powershell") == "write"   # pipe
    assert c("echo x > f.txt", "cmd") == "write"                        # redirect
    assert c("dir && del x", "cmd") == "write"                          # chain
    assert c("Get-ChildItem; Remove-Item x", "powershell") == "write"   # semicolon
    assert c("dir & whoami", "cmd") == "write"                          # background/sep


def test_empty_is_write():
    assert c("", "powershell") == "write"
    assert c("   ", "cmd") == "write"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_shell_classify.py --import-mode=importlib -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.shell_exec'`).

- [ ] **Step 3: Implement**

Create `src/shell_exec/__init__.py` (empty file). Create `src/shell_exec/classify.py`:

```python
"""Classify a shell command as read-only ("read") or state-changing ("write").
Conservative and fail-safe: a command is "read" only when it is a single simple
invocation (no pipe/redirect/chain/subshell) of an allowlisted read command;
anything else — any compose character or unrecognized leading token — is "write"."""

# Any of these means the command can compose or redirect, so it may hide a
# mutation (e.g. `Get-Content x | Remove-Item`). Presence → always "write".
_COMPOSE = ("|", ">", "<", ";", "&", "`", "$(", "\n")

# PowerShell reads: the Get-/Format- verbs plus a few explicit safe cmdlets.
_PS_READ = frozenset({"test-path", "select-object", "select-string",
                      "measure-object", "where-object", "resolve-path",
                      "write-output"})
_PS_READ_PREFIX = ("get-", "format-")

# cmd reads. `set` is deliberately excluded: `set X=Y` mutates and we only
# inspect the leading token, so it cannot be distinguished safely.
_CMD_READ = frozenset({"dir", "type", "where", "echo", "ver", "whoami", "hostname"})


def classify_command(command: str, shell: str) -> str:
    cmd = (command or "").strip()
    if not cmd:
        return "write"
    low = cmd.lower()
    if any(tok in low for tok in _COMPOSE):
        return "write"
    lead = low.split()[0]
    if shell == "powershell":
        if lead in _PS_READ or any(lead.startswith(p) for p in _PS_READ_PREFIX):
            return "read"
    elif shell == "cmd":
        if lead in _CMD_READ:
            return "read"
    return "write"
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_shell_classify.py --import-mode=importlib -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/shell_exec/__init__.py src/shell_exec/classify.py tests/test_shell_classify.py
git commit -m "feat(shell): conservative read/write command classifier"
```

---

### Task 2: Consent setting + gate + startup reset

**Files:**
- Modify: `src/settings.py` (add setting near line 42; add `reset_shell_exec` near line 368)
- Modify: `app.py:985-987` (call the reset at startup)
- Create: `src/shell_exec/gate.py`
- Test: `tests/test_shell_gate.py`

**Interfaces:**
- Produces: setting key `"shell_exec_enabled"` (default `False`); `reset_shell_exec()`; `require_shell_consent(get_setting) -> None` (raises `PermissionError` when off).

- [ ] **Step 1: Write the failing test**

Create `tests/test_shell_gate.py`:

```python
import pytest
from src.shell_exec.gate import require_shell_consent


def test_gate_blocks_when_off():
    with pytest.raises(PermissionError):
        require_shell_consent(lambda k, d=None: False)


def test_gate_passes_when_on():
    require_shell_consent(lambda k, d=None: True)  # must not raise


def test_gate_reads_the_right_key():
    seen = {}
    def get(k, d=None):
        seen["key"] = k
        return True
    require_shell_consent(get)
    assert seen["key"] == "shell_exec_enabled"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_shell_gate.py --import-mode=importlib -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.shell_exec.gate'`).

- [ ] **Step 3: Implement the gate**

Create `src/shell_exec/gate.py`:

```python
"""Consent gate for AI shell execution."""


def require_shell_consent(get_setting) -> None:
    """Raise PermissionError unless the shell-execution consent is on."""
    if not get_setting("shell_exec_enabled", False):
        raise PermissionError(
            "Shell execution is off — enable 'Allow shell commands' in the sidebar.")
```

- [ ] **Step 4: Add the setting + startup reset**

In `src/settings.py`, add to the defaults dict right after `"input_control_enabled": False,` (line 42):

```python
    "shell_exec_enabled": False,
```

After `reset_input_control` (near line 368) add:

```python
def reset_shell_exec():
    """Force shell execution off. Called at startup so AI shell access is never
    silently available across restarts (mirrors reset_input_control)."""
    try:
        s = load_settings()
        if s.get("shell_exec_enabled"):
            s["shell_exec_enabled"] = False
            save_settings(s)
    except Exception:
        pass
    try:                                   # also drop any stale approval state
        from src.shell_exec.approval import reset_all
        reset_all()
    except Exception:
        pass
```

In `app.py`, extend the startup reset block (lines 985-987):

```python
        from src.settings import reset_screen_access, reset_input_control, reset_shell_exec
        reset_screen_access()
        reset_input_control()
        reset_shell_exec()
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_shell_gate.py --import-mode=importlib -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/settings.py app.py src/shell_exec/gate.py tests/test_shell_gate.py
git commit -m "feat(shell): shell_exec_enabled consent setting + gate + startup reset"
```

---

### Task 3: Shell runner

**Files:**
- Create: `src/shell_exec/runner.py`
- Test: `tests/test_shell_runner.py`

**Interfaces:**
- Consumes: `_run_subprocess_streaming`, `DEFAULT_BASH_TIMEOUT` from `src/agent_tools/subprocess_tools.py`; `agent_cwd`, `_truncate` from `src/tool_execution.py`.
- Produces: `powershell_argv(command) -> list[str]`, `cmd_argv(command) -> list[str]`, and `async run_in_shell(command, shell, ctx, *, spawn=None, stream=None) -> dict` returning `{output, exit_code}` (or `{error, exit_code, ...}` on timeout).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shell_runner.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_shell_runner.py --import-mode=importlib -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.shell_exec.runner'`).

- [ ] **Step 3: Implement**

Create `src/shell_exec/runner.py`:

```python
"""Run a command in a chosen shell, reusing the agent's subprocess streaming
(timeout, progress, output cap). Only powershell/cmd — bash keeps its own tool."""
import asyncio
import shutil

from src.constants import MAX_OUTPUT_CHARS
from src.agent_tools.subprocess_tools import _run_subprocess_streaming, DEFAULT_BASH_TIMEOUT

# Imported at module level so tests can monkeypatch it on this module.
from src.tool_execution import agent_cwd, _truncate


def powershell_argv(command: str) -> list:
    exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
    return [exe, "-NoProfile", "-NonInteractive", "-Command", command]


def cmd_argv(command: str) -> list:
    exe = shutil.which("cmd") or "cmd"
    return [exe, "/c", command]


_ARGV = {"powershell": powershell_argv, "cmd": cmd_argv}


async def run_in_shell(command, shell, ctx, *, spawn=None, stream=None) -> dict:
    argv = _ARGV[shell](command)
    spawn = spawn or asyncio.create_subprocess_exec
    stream = stream or _run_subprocess_streaming
    proc = await spawn(*argv, stdout=asyncio.subprocess.PIPE,
                       stderr=asyncio.subprocess.PIPE,
                       env=ctx.get("subproc_env"), cwd=agent_cwd())
    stdout, stderr, rc, timed_out = await stream(
        proc, timeout=DEFAULT_BASH_TIMEOUT, progress_cb=ctx.get("progress_cb"))
    if timed_out:
        return {"error": f"{shell}: timed out after {DEFAULT_BASH_TIMEOUT}s — process killed",
                "exit_code": 124, "stdout": _truncate(stdout, MAX_OUTPUT_CHARS),
                "stderr": _truncate(stderr, MAX_OUTPUT_CHARS)}
    output = stdout.rstrip()
    err = stderr.rstrip()
    if err:
        output = (output + "\nSTDERR: " + err).strip() if output else "STDERR: " + err
    return {"output": _truncate(output, MAX_OUTPUT_CHARS) or "(no output)", "exit_code": rc or 0}
```

Note: `agent_cwd`/`_truncate` are imported at module top so the tests can
monkeypatch `r.agent_cwd`; the fake `stream` bypasses the real proc mechanics.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_shell_runner.py --import-mode=importlib -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/shell_exec/runner.py tests/test_shell_runner.py
git commit -m "feat(shell): powershell/cmd runner over the agent subprocess streamer"
```

---

### Task 4: Approval bridge + decision route

**Files:**
- Create: `src/shell_exec/approval.py`, `routes/shell_approval_routes.py`
- Modify: `app.py` (import + `include_router`, next to the operator route include at `app.py:781-782`)
- Test: `tests/test_shell_approval.py`

**Interfaces:**
- Produces: `async await_decision(session_id, command, shell, *, timeout=300) -> str` (`"approve"`|`"deny"`); `set_decision(session_id, pending_id, decision) -> bool`; `list_pending(session_id) -> list[dict]`; `reset_session(session_id)`; `reset_all()`; and `setup_shell_approval_routes()` returning an APIRouter with `GET /api/shell/pending`, `POST /api/shell/decision`, and `POST /api/shell/reset` (clears all sessions — called by the consent switch when turned off, satisfying the spec's "reset when consent is turned off").

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shell_approval.py`:

```python
import asyncio
import src.shell_exec.approval as a


def setup_function():
    a.reset_all()


def test_approve_flow():
    async def run():
        task = asyncio.ensure_future(a.await_decision("s1", "del x", "cmd", timeout=5))
        await asyncio.sleep(0)                       # let it register the pending
        pend = a.list_pending("s1")
        assert len(pend) == 1 and pend[0]["command"] == "del x"
        assert a.set_decision("s1", pend[0]["pending_id"], "approve") is True
        return await task
    assert asyncio.run(run()) == "approve"


def test_deny_flow():
    async def run():
        task = asyncio.ensure_future(a.await_decision("s1", "del x", "cmd", timeout=5))
        await asyncio.sleep(0)
        pid = a.list_pending("s1")[0]["pending_id"]
        a.set_decision("s1", pid, "deny")
        return await task
    assert asyncio.run(run()) == "deny"


def test_auto_approve_all_sets_flag_and_skips_staging():
    async def run():
        t1 = asyncio.ensure_future(a.await_decision("s1", "del x", "cmd", timeout=5))
        await asyncio.sleep(0)
        pid = a.list_pending("s1")[0]["pending_id"]
        a.set_decision("s1", pid, "auto_approve_all")
        first = await t1
        # a subsequent write auto-approves with no pending staged
        second = await a.await_decision("s1", "del y", "cmd", timeout=5)
        assert a.list_pending("s1") == []
        return first, second
    assert asyncio.run(run()) == ("approve", "approve")


def test_timeout_denies():
    assert asyncio.run(a.await_decision("s1", "del x", "cmd", timeout=0.05)) == "deny"


def test_reset_clears_auto_all():
    async def run():
        t1 = asyncio.ensure_future(a.await_decision("s1", "del x", "cmd", timeout=5))
        await asyncio.sleep(0)
        a.set_decision("s1", a.list_pending("s1")[0]["pending_id"], "auto_approve_all")
        await t1
        a.reset_session("s1")                        # e.g. consent turned off
        return await a.await_decision("s1", "del y", "cmd", timeout=0.05)
    assert asyncio.run(run()) == "deny"              # no longer auto-approving
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_shell_approval.py --import-mode=importlib -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.shell_exec.approval'`).

- [ ] **Step 3: Implement the bridge**

Create `src/shell_exec/approval.py`:

```python
"""Per-chat-session approval bridge for state-changing shell commands. Mirrors
the AI Operator's wake-Event confirmation: a write command awaits a decision that
the /api/shell/decision route delivers. Fail-closed: timeout → deny."""
import asyncio
import uuid

# session_id -> {"auto_all": bool, "pending": {pid: {command, shell, wake, decision}}}
_SESSIONS: dict = {}


def _sess(session_id):
    return _SESSIONS.setdefault(session_id or "_none",
                                {"auto_all": False, "pending": {}})


def reset_session(session_id):
    _SESSIONS.pop(session_id or "_none", None)


def reset_all():
    _SESSIONS.clear()


def list_pending(session_id):
    s = _sess(session_id)
    return [{"pending_id": pid, "command": p["command"], "shell": p["shell"]}
            for pid, p in s["pending"].items()]


def set_decision(session_id, pending_id, decision) -> bool:
    """Deliver a UI decision. decision ∈ approve | deny | auto_approve_all."""
    s = _sess(session_id)
    p = s["pending"].get(pending_id)
    if p is None:
        return False
    if decision == "auto_approve_all":
        s["auto_all"] = True
        p["decision"] = "approve"
    else:
        p["decision"] = "approve" if decision == "approve" else "deny"
    p["wake"].set()
    return True


async def await_decision(session_id, command, shell, *, timeout=300) -> str:
    s = _sess(session_id)
    if s["auto_all"]:
        return "approve"
    pid = uuid.uuid4().hex[:12]
    s["pending"][pid] = {"command": command, "shell": shell,
                         "wake": asyncio.Event(), "decision": None}
    try:
        await asyncio.wait_for(s["pending"][pid]["wake"].wait(), timeout)
        return s["pending"][pid]["decision"] or "deny"
    except asyncio.TimeoutError:
        return "deny"
    finally:
        s["pending"].pop(pid, None)
```

- [ ] **Step 4: Implement the route**

Create `routes/shell_approval_routes.py`:

```python
"""HTTP surface for the shell-command approval card. async so Event.set() runs
on the loop thread (AI Operator lesson)."""
from fastapi import APIRouter, Body, Request

from src.shell_exec.approval import set_decision, list_pending


def setup_shell_approval_routes():
    router = APIRouter(prefix="/api/shell")

    @router.get("/pending")
    async def pending(request: Request):
        return {"pending": list_pending(request.query_params.get("session_id"))}

    @router.post("/decision")
    async def decision(request: Request, body: dict = Body(...)):
        ok = set_decision(body.get("session_id"), body.get("pending_id"),
                          body.get("decision"))
        return {"ok": ok}

    @router.post("/reset")
    async def reset(request: Request):
        reset_all()          # consent turned off → drop every session's auto-approve
        return {"ok": True}

    return router
```

Update the import at the top of the route file to include `reset_all`:

```python
from src.shell_exec.approval import set_decision, list_pending, reset_all
```

In `app.py`, next to the operator route include (lines 781-782):

```python
from routes.shell_approval_routes import setup_shell_approval_routes
app.include_router(setup_shell_approval_routes())
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_shell_approval.py --import-mode=importlib -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add src/shell_exec/approval.py routes/shell_approval_routes.py app.py tests/test_shell_approval.py
git commit -m "feat(shell): per-session approval bridge + /api/shell/decision route"
```

---

### Task 5: Tool handlers + registration

**Files:**
- Create: `src/agent_tools/shell_tools.py`
- Modify: `src/agent_tools/__init__.py` (import + `TOOL_HANDLERS` entries near line 43), `src/tool_schemas.py` (two schema entries near the `bash` entry at line 27), `src/tool_index.py` (two descriptions near line 70)
- Test: `tests/test_shell_tools.py`

**Interfaces:**
- Consumes: `require_shell_consent` (Task 2), `classify_command` (Task 1), `await_decision` (Task 4), `run_in_shell` (Task 3).
- Produces: `PowerShellTool`, `CmdTool` (each `async execute(content, ctx) -> dict`); tool names `powershell`, `cmd` in `TOOL_HANDLERS`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shell_tools.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_shell_tools.py --import-mode=importlib -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.agent_tools.shell_tools'`).

- [ ] **Step 3: Implement the tools**

Create `src/agent_tools/shell_tools.py`:

```python
"""Consent-gated PowerShell and cmd agent tools. Read-only commands auto-run;
state-changing commands await the user's approval (Approve/Deny/Auto-approve-all)."""
import json

from src.settings import get_setting
from src.shell_exec.gate import require_shell_consent
from src.shell_exec.classify import classify_command
from src.shell_exec.approval import await_decision
from src.shell_exec.runner import run_in_shell


def _args(content):
    try:
        return json.loads(content) if content.strip().startswith("{") else {}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {}


class _ShellTool:
    shell = ""

    async def execute(self, content, ctx):
        try:
            require_shell_consent(get_setting)
        except PermissionError as e:
            return {"error": str(e), "exit_code": 1}
        command = (_args(content).get("command") or content or "").strip()
        if not command or command.startswith("{"):
            return {"error": f"{self.shell}: command required", "exit_code": 1}
        if classify_command(command, self.shell) == "write":
            decision = await await_decision(ctx.get("session_id"), command, self.shell)
            if decision != "approve":
                return {"output": "Command denied by the user.", "exit_code": 1}
        return await run_in_shell(command, self.shell, ctx)


class PowerShellTool(_ShellTool):
    shell = "powershell"


class CmdTool(_ShellTool):
    shell = "cmd"
```

- [ ] **Step 4: Register the tools**

In `src/agent_tools/__init__.py`, add the import near the other tool imports (by `subprocess_tools`, line 22):

```python
from .shell_tools import PowerShellTool, CmdTool
```

And add to `TOOL_HANDLERS` right after the `"bash"`/`"python"` entries (line 43-44):

```python
    "powershell": PowerShellTool().execute,
    "cmd": CmdTool().execute,
```

In `src/tool_schemas.py`, add two entries mirroring the `bash` schema (line 27), e.g.:

```python
        {"type": "function", "function": {
            "name": "powershell",
            "description": "Run a PowerShell command on this Windows machine. Requires the user's 'Allow shell commands' toggle. Read-only commands (Get-*, Test-Path, ...) run automatically; state-changing commands wait for the user to approve. Prefer dedicated file tools over shell for reading/writing/editing files.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string", "description": "The PowerShell command to execute"}},
                "required": ["command"]}}},
        {"type": "function", "function": {
            "name": "cmd",
            "description": "Run a Windows command-prompt (cmd.exe) command. Requires the user's 'Allow shell commands' toggle. Read-only commands (dir, type, ...) run automatically; state-changing commands wait for approval. Prefer dedicated file tools over shell for file work.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string", "description": "The cmd command to execute"}},
                "required": ["command"]}}},
```

In `src/tool_index.py`, add near the `"bash"` description (line 70):

```python
    "powershell": "Run PowerShell commands on this Windows machine (Get-*, services, processes, installs, scripts). Consent-gated: reads auto-run, state-changing commands need user approval. Prefer dedicated file tools for file read/write/edit.",
    "cmd": "Run Windows cmd.exe commands (dir, copy, del, net, ...). Consent-gated: reads auto-run, state-changing commands need user approval. Prefer dedicated file tools for file read/write/edit.",
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_shell_tools.py --import-mode=importlib -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add src/agent_tools/shell_tools.py src/agent_tools/__init__.py src/tool_schemas.py src/tool_index.py tests/test_shell_tools.py
git commit -m "feat(shell): powershell/cmd agent tools wired to gate+classify+approval+runner"
```

---

### Task 6: UI — approval card + consent switch

**Files:**
- Create: `static/js/shellApproval.js`, `static/js/shellExec.js`
- Modify: `static/index.html` (add both `<script>` tags near `operator.js` at line 2764; add the "Allow shell commands" switch next to the existing consent switches)
- Test: manual/live-verify (Task 7); no unit test framework for these DOM files.

**Interfaces:**
- Consumes: `GET /api/shell/pending?session_id=`, `POST /api/shell/decision` (Task 4); `GET/POST` the settings endpoint used by `screenAccess.js`/`inputControl.js` for the toggle.

- [ ] **Step 1: Consent switch**

Create `static/js/shellExec.js` (mirrors `static/js/inputControl.js`; uses the same
`/api/auth/settings` GET/POST; on turn-OFF it also POSTs `/api/shell/reset` so
re-enabling requires fresh approval — the spec's "reset when consent is turned off"):

```javascript
// Shell-execution sidebar toggle: reflects/updates the global `shell_exec_enabled`
// setting that gates the powershell/cmd tools. Mirrors inputControl.js. Defaults
// off and is reset off server-side on every restart (src/settings.py).
(function () {
  function $(id) { return document.getElementById(id); }

  function reflect(enabled) {
    const toggle = $('shell-exec-toggle');
    const indicator = $('shell-exec-indicator');
    if (toggle) toggle.checked = !!enabled;
    if (indicator) indicator.style.display = enabled ? '' : 'none';
  }

  async function load() {
    try {
      const res = await fetch('/api/auth/settings', { credentials: 'same-origin' });
      if (!res.ok) return;
      reflect(!!(await res.json()).shell_exec_enabled);
    } catch (e) { console.warn('Failed to load shell exec setting', e); }
  }

  async function save(enabled) {
    try {
      await fetch('/api/auth/settings', {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ shell_exec_enabled: enabled })
      });
      if (!enabled) {
        // Turning consent off clears every session's auto-approve elevation.
        try { await fetch('/api/shell/reset', { method: 'POST', credentials: 'same-origin' }); } catch (e) {}
      }
    } catch (e) { console.warn('Failed to save shell exec setting', e); }
  }

  document.addEventListener('DOMContentLoaded', () => {
    load();
    $('shell-exec-toggle')?.addEventListener('change', (e) => {
      const enabled = !!e.target.checked;
      reflect(enabled);
      save(enabled);
    });
  });
})();
```

Add its `<script>` to `static/index.html` next to `inputControl.js`, and add a switch
element beside the existing "Allow input control" switch, copying that switch's exact
markup but with ids `shell-exec-toggle` / `shell-exec-indicator` and the label
"Allow shell commands" (find the input-control switch markup in `index.html` and
duplicate it with these ids).

- [ ] **Step 2: Approval card**

Create `static/js/shellApproval.js` — a plain CSP-safe IIFE (mirror `static/js/operator.js` structure):

```javascript
// Polls /api/shell/pending for the active chat session and renders an approval
// card for each state-changing command: Approve / Deny / Auto-approve all.
(function () {
  const POLL_MS = 1500;
  function sid() { return window.currentSessionId || (window.getSessionId && window.getSessionId()) || ''; }

  async function decide(pending_id, decision) {
    try {
      await fetch('/api/shell/decision', {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid(), pending_id, decision }),
      });
    } catch (e) { /* next poll re-renders */ }
    poll();
  }

  function btn(label, fn) {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = label; b.addEventListener('click', fn);
    return b;
  }

  function render(list) {
    const host = document.getElementById('shell-approval-host');
    if (!host) return;
    const sig = JSON.stringify(list.map((p) => p.pending_id));
    if (sig === host.dataset.sig) return;      // avoid wiping focus each poll
    host.dataset.sig = sig;
    host.innerHTML = '';
    list.forEach((p) => {
      const card = document.createElement('div');
      card.className = 'list-item'; card.style.cssText = 'display:block;padding:8px;border:1px solid var(--yellow,#f1fa8c);border-radius:6px;margin:6px 0;';
      const h = document.createElement('div');
      h.style.cssText = 'font-weight:600;font-size:12px;margin-bottom:4px;';
      h.textContent = 'Run this ' + p.shell + ' command?';
      const pre = document.createElement('pre');
      pre.style.cssText = 'font-size:12px;white-space:pre-wrap;word-break:break-word;margin:4px 0;';
      pre.textContent = p.command;
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;gap:6px;';
      row.appendChild(btn('Approve', () => decide(p.pending_id, 'approve')));
      row.appendChild(btn('Deny', () => decide(p.pending_id, 'deny')));
      row.appendChild(btn('Auto-approve all', () => decide(p.pending_id, 'auto_approve_all')));
      card.appendChild(h); card.appendChild(pre); card.appendChild(row);
      host.appendChild(card);
    });
  }

  async function poll() {
    try {
      const res = await fetch('/api/shell/pending?session_id=' + encodeURIComponent(sid()), { credentials: 'same-origin' });
      if (res.ok) render((await res.json()).pending || []);
    } catch (e) { /* ignore; keep polling */ }
  }

  document.addEventListener('DOMContentLoaded', () => {
    if (!document.getElementById('shell-approval-host')) {
      const host = document.createElement('div');
      host.id = 'shell-approval-host';
      (document.getElementById('chat-messages') || document.body).appendChild(host);
    }
    setInterval(poll, POLL_MS); poll();
  });
})();
```

Add `<script src="/static/js/shellApproval.js"></script>` and `<script src="/static/js/shellExec.js"></script>` to `static/index.html` right after the `operator.js` script tag (line 2764).

- [ ] **Step 3: Commit**

```bash
git add static/js/shellApproval.js static/js/shellExec.js static/index.html
git commit -m "feat(shell): sidebar consent switch + chat approval card (CSP-safe)"
```

Note: the `sid()` helper and the settings-endpoint details must match this repo's
front-end — the implementer should read `static/js/operator.js` (for session id
access + poll/render style) and `static/js/inputControl.js` (for the exact
settings GET/POST) and adapt these files to match, rather than assume the exact
globals used here.

---

### Task 7: Package + live-verify

**Files:**
- Modify: `installer/Output/Assist-Setup.exe` (rebuilt; force-added like prior build commits)

**Interfaces:**
- Consumes: Tasks 1-6.

- [ ] **Step 1: Full affected-suite run**

Run: `python -m pytest tests/test_shell_classify.py tests/test_shell_gate.py tests/test_shell_runner.py tests/test_shell_approval.py tests/test_shell_tools.py --import-mode=importlib -q`
Expected: PASS (all green).

- [ ] **Step 2: Build the installer**

Run: `powershell -ExecutionPolicy Bypass -File .\build-installer.ps1 -Fast`
Expected: ends with `Installer built: ...\installer\Output\Assist-Setup.exe`.

- [ ] **Step 3: Frozen import + registration check**

Run: `./dist/Assist/Assist.exe --run-py -c "from src.agent_tools import TOOL_HANDLERS; from src.shell_exec.classify import classify_command; print('OK', 'powershell' in TOOL_HANDLERS and 'cmd' in TOOL_HANDLERS, classify_command('Get-Process','powershell'), classify_command('Remove-Item x','powershell'))"`
Expected: `OK True read write`.

- [ ] **Step 4: Live-verify in the running app (manual)**

Reinstall; then in the app:
- Toggle **off** → ask the AI to run a PowerShell command → it reports shell execution is off (refused).
- Toggle **on** → ask for `Get-Process` (read) → runs automatically, output returned.
- Ask for a state-changing PowerShell command (e.g. create then delete a temp file) → an approval card appears with the exact command + Approve/Deny/Auto-approve-all → **Approve** → runs.
- Ask for two state-changing commands; click **Auto-approve all** on the first → the second runs with no card.
- Confirm `cmd` works the same (e.g. `dir` auto; `del` prompts).

- [ ] **Step 5: Commit the installer**

```bash
git add -f installer/Output/Assist-Setup.exe
git commit -m "build: Assist-Setup.exe with AI PowerShell/cmd execution"
```

---

## Notes for the executor

- Every pytest run uses `--import-mode=importlib`.
- Tasks 1-5 are pure Python + injected fakes (no real shells spawned). Task 6 is UI (verified live in Task 7).
- The single most important safety property: `classify_command` must never return `"read"` for anything containing a pipe/redirect/chain/subshell — Task 1's `test_compose_chars_force_write` is the guard.
- Do not modify the existing `bash` tool or `subprocess_tools.py` behavior — only reuse `_run_subprocess_streaming`/`DEFAULT_BASH_TIMEOUT`.

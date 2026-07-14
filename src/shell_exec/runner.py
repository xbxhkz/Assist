"""Run a command in a chosen shell, reusing the agent's subprocess streaming
(timeout, progress, output cap). Only powershell/cmd — bash keeps its own tool."""
import asyncio
import shutil

from src.constants import MAX_OUTPUT_CHARS

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
    # Lazy: importing src.agent_tools.subprocess_tools runs the agent_tools package
    # __init__, which imports shell_tools -> this module. Importing here (not at
    # module top) breaks that cycle so `import src.shell_exec.runner` is order-safe.
    from src.agent_tools.subprocess_tools import _run_subprocess_streaming, DEFAULT_BASH_TIMEOUT
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

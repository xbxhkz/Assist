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

"""Classify a shell command as read-only ("read") or state-changing ("write").
Conservative and fail-safe: a command is "read" only when it is a single simple
invocation (no pipe/redirect/chain/subshell) of an allowlisted read command;
anything else — any compose character or unrecognized leading token — is "write"."""

# Any of these means the command can compose or redirect, so it may hide a
# mutation (e.g. `Get-Content x | Remove-Item`). Presence → always "write".
_COMPOSE = ("|", ">", "<", ";", "&", "`", "$(", "\n", "\r")

# PowerShell reads: the Get- verb prefix plus a few explicit safe cmdlets.
# NOTE: "format-" is deliberately NOT a blanket prefix — Format-Volume and
# Format-SecureBootUEFI are destructive despite the Format- verb, so only the
# specific display formatters below are allowlisted as reads.
_PS_READ = frozenset({"test-path", "select-object", "select-string",
                      "measure-object", "where-object", "resolve-path",
                      "write-output",
                      "format-table", "format-list", "format-wide",
                      "format-custom", "format-hex"})
_PS_READ_PREFIX = ("get-",)

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
    # Any newline / vertical / unicode-line-separator / control whitespace is a
    # statement separator that str.split() would silently collapse — force write
    # so a hidden second statement can't ride inside a "read".
    if any(ch.isspace() and ch not in " \t" for ch in cmd):
        return "write"
    lead = low.split()[0]
    if shell == "powershell":
        if lead in _PS_READ or any(lead.startswith(p) for p in _PS_READ_PREFIX):
            return "read"
    elif shell == "cmd":
        if lead in _CMD_READ:
            return "read"
    return "write"

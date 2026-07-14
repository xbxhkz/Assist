"""Consent gate for AI shell execution."""


def require_shell_consent(get_setting) -> None:
    """Raise PermissionError unless the shell-execution consent is on."""
    if not get_setting("shell_exec_enabled", False):
        raise PermissionError(
            "Shell execution is off — enable 'Allow shell commands' in the sidebar.")

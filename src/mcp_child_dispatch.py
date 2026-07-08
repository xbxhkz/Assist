"""Dispatch `Assist.exe --run-mcp <script.py>` to the script, not the app.

The frozen build's sys.executable IS the app. Built-in MCP servers need a
python-like runner for their scripts; without this dispatch every spawned
child boots the full app — which registers its own MCP servers, recursively
(observed live as 21 Assist.exe processes). The launcher calls
maybe_dispatch(sys.argv) before booting anything.
"""
import runpy
import sys


def maybe_dispatch(argv) -> bool:
    """True when argv requested MCP-child mode (caller must exit afterwards).

    Never falls through to the app on a dispatch request — even a missing
    script returns True, so a bad path can't re-boot the full app.
    """
    if len(argv) < 3 or argv[1] != "--run-mcp":
        return False
    script = argv[2]
    try:
        runpy.run_path(script, run_name="__main__")
    except FileNotFoundError:
        print(f"MCP child: script not found: {script}", file=sys.stderr)
    except SystemExit:
        pass
    return True
